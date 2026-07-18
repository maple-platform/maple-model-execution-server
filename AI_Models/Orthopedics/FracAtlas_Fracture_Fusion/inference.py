from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image, ImageFile
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class LandmarkHeatmapNet(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True, landmarks: int = 68):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        channels = self.encoder.feature_info.channels()
        self.lateral3 = ConvBlock(channels[3], 256)
        self.lateral2 = ConvBlock(256 + channels[2], 192)
        self.lateral1 = ConvBlock(192 + channels[1], 128)
        self.lateral0 = ConvBlock(128 + channels[0], 96)
        self.head = nn.Conv2d(96, landmarks, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        f0, f1, f2, f3 = self.encoder(image)
        x = self.lateral3(f3)
        x = self.lateral2(torch.cat([F.interpolate(x, f2.shape[-2:], mode="bilinear", align_corners=False), f2], 1))
        x = self.lateral1(torch.cat([F.interpolate(x, f1.shape[-2:], mode="bilinear", align_corners=False), f1], 1))
        x = self.lateral0(torch.cat([F.interpolate(x, f0.shape[-2:], mode="bilinear", align_corners=False), f0], 1))
        return self.head(x)


def build_model(name: str, pretrained: bool = True) -> torch.nn.Module:
    return timm.create_model(name, pretrained=pretrained, num_classes=1, drop_path_rate=0.15)


def _letterbox(image: Image.Image, size: int, interpolation: int) -> Image.Image:
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), interpolation)
    canvas = Image.new(image.mode, (size, size), color=0)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _restore_mask(mask: np.ndarray, source_size: tuple[int, int], size: int) -> np.ndarray:
    width, height = source_size
    scale = size / max(width, height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    cropped = mask[top : top + resized_height, left : left + resized_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def _encode_png(image: Image.Image) -> str:
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def load_classifier(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def predict(
    input_path: Path,
    convnext_path: Path,
    effnet_path: Path,
    segmentation_path: Path,
    fusion_path: Path,
    output_dir: Path,
    operating_point: str = "high_sensitivity",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    convnext, convnext_checkpoint = load_classifier(convnext_path, device)
    effnet, effnet_checkpoint = load_classifier(effnet_path, device)
    segmentation_checkpoint = torch.load(segmentation_path, map_location="cpu")
    segmentation = LandmarkHeatmapNet(
        segmentation_checkpoint["backbone"], pretrained=False, landmarks=1
    )
    segmentation.load_state_dict(segmentation_checkpoint["model"])
    segmentation.to(device).eval()
    calibration = json.loads(fusion_path.read_text())

    sizes = {
        int(convnext_checkpoint["image_size"]),
        int(effnet_checkpoint["image_size"]),
        int(segmentation_checkpoint["image_size"]),
    }
    if len(sizes) != 1:
        raise ValueError(f"Model input sizes differ: {sorted(sizes)}")
    size = sizes.pop()
    source = Image.open(input_path).convert("RGB")
    resampling = getattr(Image, "Resampling", Image)
    display = _letterbox(source, size, resampling.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(display), IMAGENET_MEAN, IMAGENET_STD)[None].to(device)

    with torch.no_grad():
        convnext_probability = float(convnext(tensor).flatten()[0].sigmoid().item())
        effnet_probability = float(effnet(tensor).flatten()[0].sigmoid().item())
        segmentation_probability = segmentation(tensor).sigmoid()
        segmentation_probability = F.interpolate(
            segmentation_probability, (size, size), mode="bilinear", align_corners=False
        )[0, 0]
        flat = segmentation_probability.flatten()
        features = np.asarray(
            [
                convnext_probability,
                effnet_probability,
                float(flat.max().item()),
                float(flat.topk(k=64).values.mean().item()),
                math.log1p(int((flat >= 0.50).sum().item())),
                math.log1p(int((flat >= 0.70).sum().item())),
                math.log1p(int((flat >= 0.90).sum().item())),
            ],
            dtype=np.float64,
        )
    normalized = (
        features - np.asarray(calibration["standard_scaler_mean"])
    ) / np.asarray(calibration["standard_scaler_scale"])
    logit = float(
        normalized @ np.asarray(calibration["coefficients"]) + calibration["intercept"]
    )
    fused_probability = 1.0 / (1.0 + math.exp(-logit))

    probability = _restore_mask(
        segmentation_probability.cpu().numpy(), source.size, size
    )
    mask_threshold = float(segmentation_checkpoint["threshold"])
    mask = probability >= mask_threshold
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    regions = []
    for component in range(1, component_count):
        x, y, width, height, area = stats[component].tolist()
        if area < 16:
            continue
        values = probability[labels == component]
        regions.append(
            {
                "bbox_xywh": [x, y, width, height],
                "area_pixels": area,
                "max_probability": float(values.max()),
                "mean_probability": float(values.mean()),
            }
        )
    regions.sort(key=lambda item: item["area_pixels"], reverse=True)

    image_array = np.asarray(source).astype(np.float32)
    heat = np.zeros_like(image_array)
    heat[..., 0] = probability * 255.0
    heat[..., 1] = np.sqrt(probability) * 55.0
    alpha = (0.45 * probability)[..., None]
    overlay = Image.fromarray(
        np.clip(image_array * (1.0 - alpha) + heat * alpha, 0, 255).astype(np.uint8)
    )
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"{input_path.stem}_fracture_fusion.png"
    mask_path = output_dir / f"{input_path.stem}_fracture_mask.png"
    overlay.save(overlay_path)
    mask_image.save(mask_path)

    balanced_threshold = float(calibration["balanced_threshold"])
    sensitive_threshold = float(calibration["high_sensitivity_threshold"])
    selected_threshold = (
        sensitive_threshold if operating_point == "high_sensitivity" else balanced_threshold
    )
    result = {
        "prediction": int(fused_probability >= selected_threshold),
        "pred_name": "fracture" if fused_probability >= selected_threshold else "no_fracture",
        "operating_point": operating_point,
        "selected_threshold": selected_threshold,
        "balanced_prediction": int(fused_probability >= balanced_threshold),
        "high_sensitivity_prediction": int(fused_probability >= sensitive_threshold),
        "fused_probability": fused_probability,
        "balanced_threshold": balanced_threshold,
        "high_sensitivity_threshold": sensitive_threshold,
        "component_probabilities": {
            "convnext": convnext_probability,
            "efficientnet_v2": effnet_probability,
            "segmentation_max": float(features[2]),
        },
        "regions": regions,
        "overlay_b64": _encode_png(overlay),
        "mask_b64": _encode_png(mask_image),
        "overlay_file": str(overlay_path),
        "mask_file": str(mask_path),
        "model": "FracAtlas fracture classification-segmentation fusion",
    }
    (output_dir / f"{input_path.stem}_result.json").write_text(
        json.dumps(
            {key: value for key, value in result.items() if not key.endswith("_b64")},
            indent=2,
        )
        + "\n"
    )
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--convnext", type=Path, required=True)
    parser.add_argument("--effnet", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--operating-point",
        choices=("balanced", "high_sensitivity"),
        default="high_sensitivity",
    )
    args = parser.parse_args()
    result = predict(
        args.input_path,
        args.convnext,
        args.effnet,
        args.segmentation,
        args.fusion,
        args.output_dir,
        args.operating_point,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if not key.endswith("_b64")},
            indent=2,
        )
    )


def _maple_split_input(input_data):
    if isinstance(input_data, (str, Path)):
        return str(input_data), {}
    if not isinstance(input_data, dict):
        raise TypeError("input_data must be a path or a dictionary")
    input_path = next(
        (
            input_data[key]
            for key in ("image_path", "input_path", "series_path", "volume_path")
            if input_data.get(key)
        ),
        None,
    )
    if input_path is None:
        raise ValueError(
            "input_data dictionary requires image_path, input_path, "
            "series_path, or volume_path"
        )
    params = {
        key: value for key, value in input_data.items() if not key.endswith("_path")
    }
    return str(input_path), params


def _maple_model_root(model_path):
    path = Path(model_path).expanduser().resolve()
    return path if path.is_dir() else path.parent


def _maple_checkpoint(model_path, filename):
    path = Path(model_path).expanduser().resolve()
    if path.is_file() and path.name == filename:
        return path
    candidate = (path if path.is_dir() else path.parent) / filename
    if not candidate.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {candidate}")
    return candidate


def _maple_public_value(value):
    import numpy as np

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        public = {}
        for key, item in value.items():
            if (
                key in {"ensemble_members", "fold_predictions_months", "runtime_log_tail"}
                or key == "preview"
                or key.endswith(("_preview", "_b64", "_file", "_path"))
            ):
                continue
            public[key] = _maple_public_value(item)
        return public
    if isinstance(value, (list, tuple)):
        return [_maple_public_value(item) for item in value]
    return value


def _maple_finish(result, image_paths, image_roles, summary):
    import numpy as np
    from PIL import Image

    images = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        for path in image_paths
        if path and Path(path).is_file()
    ]
    if not images:
        raise RuntimeError("Inference did not produce a display image")
    prediction = _maple_public_value(result)
    prediction.update(_maple_public_value(summary))
    prediction["image_roles"] = list(image_roles)
    display = images[0] if len(images) == 1 else images
    return display, [prediction]


def _maple_series(input_path):
    source = Path(input_path)
    return str(source.parent if source.is_file() else source)


def main(input_data, model_path: str):
    import tempfile

    input_path, params = _maple_split_input(input_data)
    root = _maple_model_root(model_path)
    with tempfile.TemporaryDirectory(prefix="maple_fracatlas_") as output_dir:
        result = predict(
            Path(input_path),
            root / "convnext.pt",
            root / "efficientnet.pt",
            root / "segmentation.pt",
            root / "fusion.json",
            Path(output_dir),
            str(params.get("operating_point", "high_sensitivity")),
        )
        return _maple_finish(
            result,
            [result["overlay_file"]],
            ["fracture_probability_overlay"],
            {"pred": result["prediction"], "pred_name": result["pred_name"]},
        )


if __name__ == "__main__":
    _cli()

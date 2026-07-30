from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

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


def spatial_softargmax(logits: torch.Tensor) -> torch.Tensor:
    batch, landmarks, height, width = logits.shape
    probabilities = logits.reshape(batch, landmarks, -1).softmax(dim=-1)
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype),
        torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    x = (probabilities * x_grid.flatten()).sum(-1)
    y = (probabilities * y_grid.flatten()).sum(-1)
    return torch.stack([x, y], dim=-1)


def _max_pair(tilts: np.ndarray, start: int, stop: int):
    if stop - start < 2:
        return 0.0, start, start
    block = tilts[start:stop]
    difference = np.abs(block[:, None] - block[None, :])
    difference = np.minimum(difference, 180.0 - difference)
    upper = np.triu(difference, k=1)
    i, j = np.unravel_index(np.argmax(upper), upper.shape)
    return float(upper[i, j]), start + int(i), start + int(j)


def cobb_angles(points: np.ndarray) -> dict:
    vertebrae = np.asarray(points, dtype=np.float64).reshape(17, 4, 2)
    top = np.arctan2(vertebrae[:, 1, 1] - vertebrae[:, 0, 1], vertebrae[:, 1, 0] - vertebrae[:, 0, 0])
    bottom = np.arctan2(vertebrae[:, 3, 1] - vertebrae[:, 2, 1], vertebrae[:, 3, 0] - vertebrae[:, 2, 0])
    tilts = np.rad2deg(np.angle(np.exp(2j * top) + np.exp(2j * bottom)) / 2.0)
    main, upper_index, lower_index = _max_pair(tilts, 0, len(tilts))
    proximal, _, _ = _max_pair(tilts, 0, upper_index + 1)
    thoracolumbar, _, _ = _max_pair(tilts, lower_index, len(tilts))
    return {
        "proximal_thoracic": proximal,
        "main_thoracic": main,
        "thoracolumbar": thoracolumbar,
        "main_upper_vertebra": upper_index,
        "main_lower_vertebra": lower_index,
    }


def _prepare(image: Image.Image, width: int, height: int):
    source_width, source_height = image.size
    scale = min(width / source_width, height / source_height)
    resized_size = (round(source_width * scale), round(source_height * scale))
    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize(resized_size, resampling.BILINEAR)
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas = Image.new("RGB", (width, height), color=0)
    canvas.paste(resized, (left, top))
    tensor = TF.normalize(TF.to_tensor(canvas), IMAGENET_MEAN, IMAGENET_STD)
    return tensor, scale, left, top


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def predict(input_path: str, checkpoint_path: str, output_dir: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    width, height = int(checkpoint["width"]), int(checkpoint["height"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkHeatmapNet(checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    source = Image.open(input_path).convert("RGB")
    tensor, scale, left, top = _prepare(source, width, height)
    with torch.no_grad():
        heatmaps = model(tensor[None].to(device))
        normalized = spatial_softargmax(heatmaps.float())[0].cpu().numpy()
    points = normalized * np.asarray([width, height])
    points[:, 0] = (points[:, 0] - left) / scale
    points[:, 1] = (points[:, 1] - top) / scale
    points[:, 0] = np.clip(points[:, 0], 0, source.width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, source.height - 1)
    measurements = cobb_angles(points)
    overlay = source.copy()
    draw = ImageDraw.Draw(overlay)
    radius = max(2, round(max(source.size) / 600))
    for vertebra in points.reshape(17, 4, 2):
        polygon = [tuple(point) for point in vertebra[[0, 1, 3, 2]]]
        draw.line(polygon + [polygon[0]], fill=(0, 255, 255), width=radius)
        for x, y in vertebra:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 80, 20))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"{Path(input_path).stem}_cobb_landmarks.png"
    overlay.save(output_path)
    result = {
        "landmarks": points.round(2).tolist(),
        "cobb_angles": measurements,
        "image_b64": _encode_png(overlay),
        "output_file": str(output_path),
        "model": "AASCE vertebral landmarks and Cobb angle",
    }
    (output / f"{Path(input_path).stem}_result.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "image_b64"}, indent=2) + "\n"
    )
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = predict(args.input_path, args.checkpoint, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "image_b64"}, indent=2))


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

    input_path, _ = _maple_split_input(input_data)
    with tempfile.TemporaryDirectory(prefix="maple_aasce_") as output_dir:
        result = predict(
            input_path, str(_maple_checkpoint(model_path, "best.pt")), output_dir
        )
        angle = max(
            float(result["cobb_angles"][key])
            for key in ("proximal_thoracic", "main_thoracic", "thoracolumbar")
        )
        return _maple_finish(
            result,
            [result["output_file"]],
            ["cobb_angle_overlay"],
            {"pred": angle, "pred_name": "maximum_cobb_angle_degrees"},
        )


if __name__ == "__main__":
    _cli()

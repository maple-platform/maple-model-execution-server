from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

BINARY_TASKS = (
    "modic",
    "up_endplate",
    "low_endplate",
    "spondylolisthesis",
    "disc_herniation",
    "disc_narrowing",
    "disc_bulging",
)


class DiscGradingModel(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=1, drop_path_rate=0.15
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, len(BINARY_TASKS) + 5),
        )

    def forward(self, image: torch.Tensor):
        output = self.head(self.backbone(image))
        return output[:, : len(BINARY_TASKS)], output[:, len(BINARY_TASKS) :]


def save_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def _gradcam(model: DiscGradingModel, tensor: torch.Tensor, grade_index: int) -> np.ndarray:
    features = model.backbone.forward_features(tensor)
    pooled = model.backbone.forward_head(features, pre_logits=True)
    output = model.head(pooled)
    grade_logit = output[0, len(BINARY_TASKS) + grade_index]
    gradients = torch.autograd.grad(grade_logit, features)[0]
    if features.shape[1] >= features.shape[-1]:
        cam = (gradients.mean((2, 3), keepdim=True) * features).sum(1, keepdim=True)
    else:
        cam = (gradients.mean((1, 2), keepdim=True) * features).sum(3).unsqueeze(1)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam, tensor.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    cam -= cam.min()
    cam /= cam.max().clamp_min(1e-6)
    return cam.detach().float().cpu().numpy()


def _overlay_gradcam(image: Image.Image, cam: np.ndarray) -> Image.Image:
    values = np.asarray(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size), dtype=np.float32
    ) / 255.0
    heat = np.stack(
        [np.clip(values * 2, 0, 1), np.clip(2 - values * 2, 0, 1), np.zeros_like(values)],
        axis=-1,
    )
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    alpha = (0.55 * values)[..., None]
    return Image.fromarray(np.clip((source * (1 - alpha) + heat * alpha) * 255, 0, 255).astype(np.uint8))


def predict(
    input_path: str,
    checkpoint_path: str,
    output_dir: str,
    thresholds_path: str | None = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiscGradingModel(checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    image_size = int(checkpoint["image_size"])
    source_image = Image.open(input_path).convert("L")
    tensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )(source_image)[None].to(device)
    with torch.inference_mode():
        binary_logits, grade_logits = model(tensor)
        probabilities = binary_logits.sigmoid()[0].cpu().tolist()
        grade_probabilities = grade_logits.softmax(dim=1)[0].cpu()
    thresholds = {task: 0.5 for task in BINARY_TASKS}
    if thresholds_path:
        thresholds.update(json.loads(Path(thresholds_path).read_text()))
    findings = {
        task: {
            "prediction": bool(probability >= thresholds[task]),
            "probability": probability,
            "threshold": thresholds[task],
        }
        for task, probability in zip(BINARY_TASKS, probabilities)
    }
    expected_grade = float(
        (grade_probabilities * torch.arange(1, 6, dtype=torch.float32)).sum().item()
    )
    grade_index = int(grade_probabilities.argmax().item())
    cam = _gradcam(model, tensor, grade_index)
    gradcam = _overlay_gradcam(source_image, cam)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gradcam_path = output / f"{Path(input_path).stem}_pfirrmann_gradcam.png"
    gradcam.save(gradcam_path)
    high_confidence_findings = {
        task: value
        for task, value in findings.items()
        if value["prediction"] and value["probability"] >= max(value["threshold"], 0.70)
    }
    result = {
        "model": "SPIDER lumbar disc grading",
        "high_confidence_findings": high_confidence_findings,
        "finding_confidence_floor": 0.70,
        "pfirrmann_grade": int(round(expected_grade)),
        "pfirrmann_expected_grade": expected_grade,
        "pfirrmann_probabilities": {
            str(index + 1): float(value)
            for index, value in enumerate(grade_probabilities.tolist())
        },
        "gradcam_file": str(gradcam_path),
    }
    save_json(output / f"{Path(input_path).stem}_disc_grading.json", result)
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            predict(
                args.input_path, args.checkpoint, args.output_dir, args.thresholds
            ),
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

    input_path, _ = _maple_split_input(input_data)
    root = _maple_model_root(model_path)
    thresholds = root / "thresholds.json"
    with tempfile.TemporaryDirectory(prefix="maple_spider_grade_") as output_dir:
        result = predict(
            input_path,
            str(_maple_checkpoint(model_path, "best.pt")),
            output_dir,
            str(thresholds) if thresholds.exists() else None,
        )
        grade = result["pfirrmann_grade"]
        return _maple_finish(
            result,
            [result["gradcam_file"]],
            ["gradcam_overlay"],
            {"pred": grade, "pred_name": f"Pfirrmann_{grade}"},
        )


if __name__ == "__main__":
    _cli()

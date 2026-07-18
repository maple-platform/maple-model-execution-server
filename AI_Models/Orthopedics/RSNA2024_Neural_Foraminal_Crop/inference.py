"""Inference for an RSNA 2024 lumbar stenosis coordinate crop."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


_CACHE = {}
LABELS = ("normal_mild", "moderate", "severe")


def _draw_prediction(image: Image.Image, label: str) -> None:
    font_size = max(18, round(min(image.size) * 0.045))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    margin = max(10, round(min(image.size) * 0.025))
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    bounds = draw.textbbox((0, 0), label, font=font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x, y = image.width - width - margin, margin
    padding = max(6, round(min(image.size) * 0.012))
    draw.rounded_rectangle((x - padding, y - padding, x + width + padding, y + height + padding), radius=max(4, padding // 2), fill=(32, 32, 32, 178))
    draw.text((x, y), label, fill="white", font=font)
    image.paste(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"))


def _read_plane(path: Path) -> Image.Image:
    if path.suffix.lower() == ".dcm":
        import pydicom

        dataset = pydicom.dcmread(path)
        array = dataset.pixel_array.astype(np.float32)
        low, high = np.percentile(array, (1, 99))
        array = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
        return Image.fromarray((array * 255).astype(np.uint8), mode="L")
    return Image.open(path)


def _load_input(input_path: str) -> tuple[Image.Image, str]:
    path = Path(input_path)
    if path.is_dir():
        files = sorted(
            item for item in path.iterdir()
            if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"}
        )
        if len(files) != 3:
            raise ValueError("A crop directory must contain exactly three ordered slices")
        planes = [np.asarray(_read_plane(item).convert("L")) for item in files]
        if len({plane.shape for plane in planes}) != 1:
            raise ValueError("All three slices must have the same dimensions")
        return Image.fromarray(np.stack(planes, axis=-1)), "three_slice_directory"
    image = _read_plane(path)
    mode = "three_slice_rgb" if image.mode == "RGB" else "single_slice_replicated"
    return image.convert("RGB"), mode


def _model(checkpoint_path: str):
    if checkpoint_path in _CACHE:
        return _CACHE[checkpoint_path]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = timm.create_model(
        checkpoint["backbone"], pretrained=False, num_classes=3, drop_path_rate=0.15
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    value = (model, checkpoint, device)
    _CACHE[checkpoint_path] = value
    return value


def predict(input_path: str, checkpoint_path: str, output_dir: str) -> dict:
    model, checkpoint, device = _model(checkpoint_path)
    image, input_mode = _load_input(input_path)
    transform = transforms.Compose(
        [
            transforms.Resize(
                (checkpoint["image_size"], checkpoint["image_size"]), antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = (
            model(tensor).softmax(1)
            + model(torch.flip(tensor, dims=(3,))).softmax(1)
        )[0] / 2
    values = probabilities.cpu().tolist()
    grade = int(np.argmax(values))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    display = np.asarray(image)
    if display.ndim == 3:
        display = display[..., display.shape[2] // 2]
    preview = Image.fromarray(display.astype(np.uint8)).convert("RGB").resize((512, 512))
    display_label = LABELS[grade].replace("_", " / ").title()
    label = f"Severity: {display_label} ({values[grade]:.1%})"
    _draw_prediction(preview, label)
    preview.save(output / "preview.png")
    result = {
        "task": checkpoint["task"],
        "scope": "coordinate-centered three-slice crop classification",
        "input_mode": input_mode,
        "grade": grade,
        "label": LABELS[grade],
        "expected_grade": float(np.dot(values, np.arange(3))),
        "probabilities": dict(zip(LABELS, values)),
        "preview": str(output / "preview.png"),
        "warning": "Research use only. This model does not localize the lumbar level.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


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
    with tempfile.TemporaryDirectory(prefix="maple_rsna_crop_") as output_dir:
        result = predict(
            input_path, str(_maple_checkpoint(model_path, "best.pt")), output_dir
        )
        return _maple_finish(
            result,
            [result["preview"]],
            ["severity_overlay"],
            {"pred": result["grade"], "pred_name": result["label"]},
        )

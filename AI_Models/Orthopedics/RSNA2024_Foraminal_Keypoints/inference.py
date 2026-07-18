"""Bilateral neural-foraminal keypoints from a sagittal T1 DICOM series."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pydicom
import timm
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


_CACHE = {}


def _normalize(dataset) -> np.ndarray:
    pixels = dataset.pixel_array.astype(np.float32)
    pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0))
    pixels += float(getattr(dataset, "RescaleIntercept", 0.0))
    low, high = np.percentile(pixels, (1, 99))
    pixels = np.clip((pixels - low) / max(high - low, 1e-6), 0, 1)
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1 - pixels
    return (pixels * 255).astype(np.uint8)


def _crop(image, x, y, spacing, size_mm=96.0):
    spacing_y, spacing_x = map(float, spacing)
    half_x, half_y = max(16, round(size_mm / spacing_x / 2)), max(16, round(size_mm / spacing_y / 2))
    padded = np.pad(image, ((half_y, half_y), (half_x, half_x)))
    center_x, center_y = round(x) + half_x, round(y) + half_y
    crop = padded[center_y - half_y:center_y + half_y, center_x - half_x:center_x + half_x]
    return np.asarray(Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR))


def _model(checkpoint_path):
    if checkpoint_path in _CACHE:
        return _CACHE[checkpoint_path]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    points = checkpoint["points"]
    model = timm.create_model(
        checkpoint["backbone"], pretrained=False, num_classes=len(points) * 3, drop_path_rate=0.15
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    _CACHE[checkpoint_path] = model, checkpoint, device
    return _CACHE[checkpoint_path]


def predict(input_path: str, checkpoint_path: str, output_dir: str) -> dict:
    files = list(Path(input_path).glob("*.dcm"))
    if not files:
        raise ValueError("Input must be a directory containing a sagittal T1 DICOM series")
    datasets = [pydicom.dcmread(path) for path in files]
    if all(hasattr(item, "ImagePositionPatient") for item in datasets):
        datasets.sort(key=lambda item: float(item.ImagePositionPatient[0]))
    else:
        datasets.sort(key=lambda item: int(item.InstanceNumber))
    planes = [_normalize(item) for item in datasets]
    indices = np.rint(np.linspace(0, len(planes) - 1, 3)).astype(int)
    image = Image.fromarray(np.stack([planes[index] for index in indices], axis=-1))
    model, checkpoint, device = _model(checkpoint_path)
    transform = transforms.Compose(
        [
            transforms.Resize((checkpoint["image_size"],) * 2, antialias=True),
            transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    with torch.inference_mode():
        coordinates = model(transform(image).unsqueeze(0).to(device)).sigmoid()
    coordinates = coordinates.cpu().numpy().reshape(len(checkpoint["points"]), 3)
    height, width = planes[0].shape
    output = Path(output_dir); crop_dir = output / "crops"; crop_dir.mkdir(parents=True, exist_ok=True)
    overlay = Image.fromarray(planes[indices[1]]).convert("RGB"); draw = ImageDraw.Draw(overlay)
    results = []
    for point, normalized in zip(checkpoint["points"], coordinates):
        x, y = float(normalized[0] * (width - 1)), float(normalized[1] * (height - 1))
        index = int(np.clip(round(normalized[2] * (len(datasets) - 1)), 0, len(datasets) - 1))
        channels = []
        for offset in (-1, 0, 1):
            slice_index = int(np.clip(index + offset, 0, len(datasets) - 1))
            channels.append(_crop(planes[slice_index], x, y, datasets[slice_index].PixelSpacing))
        slug = point.replace(" ", "_").replace("/", "_")
        crop_path = crop_dir / f"{slug}.png"
        Image.fromarray(np.stack(channels, axis=-1), mode="RGB").save(crop_path)
        color = (255, 220, 0) if point.startswith("left") else (0, 220, 255)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        results.append(
            {"point": point, "x_pixel": x, "y_pixel": y, "slice_index": index,
             "instance_number": int(datasets[index].InstanceNumber),
             "normalized_xyz": normalized.tolist(), "crop_path": str(crop_path)}
        )
    preview = output / "preview.png"; overlay.save(preview)
    result = {"task": "bilateral_neural_foraminal_keypoints", "points": results,
              "preview": str(preview), "warning": "Research use only; not a medical device."}
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
    with tempfile.TemporaryDirectory(prefix="maple_rsna_keypoints_") as output_dir:
        result = predict(_maple_series(input_path), str(_maple_checkpoint(model_path, "best.pt")), output_dir)
        return _maple_finish(
            result,
            [result["preview"]],
            ["keypoint_overlay"],
            {"pred": len(result["points"]), "pred_name": "keypoints_localized"},
        )

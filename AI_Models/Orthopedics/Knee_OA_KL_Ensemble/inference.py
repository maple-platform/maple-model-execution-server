from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms


_MODEL_CACHE = {}
_LOCK = threading.Lock()


def _read_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".dcm", ".dicom"}:
        import pydicom

        dataset = pydicom.dcmread(str(path))
        pixels = dataset.pixel_array.astype(np.float32)
        lower, upper = np.percentile(pixels, (0.5, 99.5))
        pixels = np.clip((pixels - lower) / max(upper - lower, 1e-6), 0, 1)
        if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = 1 - pixels
        return Image.fromarray((pixels * 255).astype(np.uint8), mode="L")
    return Image.open(path).convert("L")


def _load_models(checkpoint_dir: Path, device: torch.device):
    key = (str(checkpoint_dir), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    models = []
    for filename in (
        "convnext_tiny_320.pt",
        "convnext_small_320.pt",
        "convnext_tiny_384.pt",
    ):
        checkpoint = torch.load(checkpoint_dir / filename, map_location="cpu")
        model = timm.create_model(
            checkpoint["backbone"], pretrained=False, num_classes=5,
            in_chans=1, drop_path_rate=0.15,
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (checkpoint["image_size"], checkpoint["image_size"]), antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        models.append((filename, model, transform))
    _MODEL_CACHE[key] = models
    return models


def _gradcam(model, tensor: torch.Tensor, grade: int) -> np.ndarray:
    features = model.forward_features(tensor)
    logits = model.forward_head(features)
    gradients = torch.autograd.grad(logits[0, grade], features)[0]
    if features.shape[1] >= features.shape[-1]:
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True)
    else:
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * features).sum(dim=3, keepdim=False).unsqueeze(1)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam, tensor.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    cam -= cam.min()
    cam /= cam.max().clamp_min(1e-6)
    return cam.detach().float().cpu().numpy()


def _overlay_gradcam(image: Image.Image, cam: np.ndarray) -> Image.Image:
    cam_image = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size)
    values = np.asarray(cam_image, dtype=np.float32) / 255.0
    heat = np.stack(
        [np.clip(values * 2.0, 0, 1), np.clip(2.0 - values * 2.0, 0, 1), np.zeros_like(values)],
        axis=-1,
    )
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    alpha = (0.55 * values)[..., None]
    return Image.fromarray(np.clip((source * (1 - alpha) + heat * alpha) * 255, 0, 255).astype(np.uint8))


def predict(input_path: str, checkpoint_dir: str, output_dir: str) -> dict:
    source = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = _read_image(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    member_probabilities = []
    member_names = []
    cams = []
    with _LOCK:
        for name, model, transform in _load_models(Path(checkpoint_dir), device):
            tensor = transform(image)[None].to(device)
            with torch.inference_mode():
                probabilities = (
                    model(tensor).softmax(dim=1)
                    + model(torch.flip(tensor, dims=(3,))).softmax(dim=1)
                ) / 2
            member_probabilities.append(probabilities[0].float().cpu().numpy())
            member_names.append(name)
    probabilities = np.mean(member_probabilities, axis=0)
    grade = int(probabilities.argmax())
    expected = float(probabilities @ np.arange(5))
    with _LOCK:
        for _, model, transform in _load_models(Path(checkpoint_dir), device):
            member_cam = _gradcam(model, transform(image)[None].to(device), grade)
            cams.append(
                np.asarray(
                    Image.fromarray(member_cam.astype(np.float32), mode="F").resize(image.size),
                    dtype=np.float32,
                )
            )
    gradcam = _overlay_gradcam(image, np.mean(cams, axis=0))
    gradcam_path = output / f"{source.stem}_kl_gradcam.png"
    gradcam.save(gradcam_path)
    result = {
        "model": "Knee OA KL ConvNeXt ensemble",
        "input_scope": "single cropped AP/PA knee radiograph",
        "kl_grade": grade,
        "expected_kl_grade": expected,
        "kl_probabilities": {
            f"KL{index}": float(value) for index, value in enumerate(probabilities)
        },
        "oa_kl2_probability": float(probabilities[2:].sum()),
        "ensemble_members": member_names,
        "gradcam_file": str(gradcam_path),
    }
    (output / f"{source.stem}_result.json").write_text(
        json.dumps(result, indent=2)
        + "\n"
    )
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
    with tempfile.TemporaryDirectory(prefix="maple_knee_kl_") as output_dir:
        result = predict(input_path, str(_maple_model_root(model_path)), output_dir)
        grade = result["kl_grade"]
        return _maple_finish(
            result,
            [result["gradcam_file"]],
            ["gradcam_overlay"],
            {"pred": grade, "pred_name": f"KL{grade}"},
        )

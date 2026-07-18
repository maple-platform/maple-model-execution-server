from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.exposure import match_histograms
from transformers import AutoModel


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _age_label(age_months: float) -> str:
    total_months = max(0, int(round(age_months)))
    years, months = divmod(total_months, 12)
    year_unit = "Year" if years == 1 else "Years"
    month_unit = "Month" if months == 1 else "Months"
    return f"Age: {years} {year_unit}, {months} {month_unit}"


def _overlay_age(image: Image.Image, age_months: float) -> None:
    font_size = max(18, round(min(image.size) * 0.045))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    label = _age_label(age_months)
    margin = max(10, round(min(image.size) * 0.025))
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    bounds = draw.textbbox((0, 0), label, font=font)
    text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x, y = image.width - text_width - margin, margin
    pad_x = max(8, round(min(image.size) * 0.012))
    pad_y = max(5, round(min(image.size) * 0.008))
    radius = max(4, round(min(image.size) * 0.006))
    draw.rounded_rectangle(
        (x - pad_x, y - pad_y, x + text_width + pad_x, y + text_height + pad_y),
        radius=radius,
        fill=(32, 32, 32, 178),
    )
    draw.text(
        (x, y),
        label,
        fill=(255, 255, 255),
        font=font,
    )
    image.paste(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"))


def load_grayscale(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".dcm", ".dicom"}:
        from pydicom import dcmread
        from pydicom.pixels import apply_voi_lut

        dicom = dcmread(path)
        image = apply_voi_lut(dicom.pixel_array, dicom).astype(np.float32)
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = image.max() - image
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
    image -= image.min()
    maximum = float(image.max())
    if maximum > 0:
        image /= maximum
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def resize_and_pad(image: np.ndarray, size: int = 512) -> np.ndarray:
    height, width = image.shape
    scale = size / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((size, size), dtype=resized.dtype)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def predict(
    input_path: Path,
    sex: str,
    model_dir: Path,
    crop_model_dir: Path,
    output_dir: Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_model = AutoModel.from_pretrained(
        crop_model_dir, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    image = load_grayscale(input_path)
    height, width = image.shape
    crop_input = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    crop_tensor = torch.from_numpy(crop_input)[None, None].float().to(device)
    image_shape = torch.tensor([[height, width]], device=device)
    with torch.inference_mode():
        x, y, crop_width, crop_height = crop_model(crop_tensor, image_shape)[0].tolist()
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))
    crop_width = int(np.clip(crop_width, 1, width - x))
    crop_height = int(np.clip(crop_height, 1, height - y))
    cropped = image[y : y + crop_height, x : x + crop_width]

    reference = cv2.imread(str(model_dir / "ref_img.png"), cv2.IMREAD_GRAYSCALE)
    matched = match_histograms(cropped, reference).astype(np.float32)
    model_input = resize_and_pad(matched)
    tensor = torch.from_numpy(model_input)[None, None].float().to(device)
    female = torch.tensor([sex == "female"], dtype=torch.long, device=device)
    with torch.inference_mode():
        fold_predictions = [
            float(getattr(model, f"net{index}")(tensor, female).item())
            for index in range(model.num_models)
        ]
    age_months = float(np.mean(fold_predictions))
    fold_std = float(np.std(fold_predictions))

    display = Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8)).convert("RGB")
    _overlay_age(display, age_months)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / f"{input_path.stem}_hand_crop_predicted_age.png"
    display.save(crop_path)
    result = {
        "bone_age_months": age_months,
        "bone_age_years": age_months / 12.0,
        "sex": sex,
        "uncertainty_months": fold_std,
        "crop_bbox_xywh": [x, y, crop_width, crop_height],
        "crop_b64": encode_png(display),
        "crop_file": str(crop_path),
        "model": "Ian Pan RSNA bone-age 3-fold ConvNeXtV2 ensemble",
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
    parser.add_argument("--sex", choices=("female", "male"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--crop-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = predict(
        args.input_path, args.sex, args.model_dir, args.crop_model_dir, args.output_dir
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
    with tempfile.TemporaryDirectory(prefix="maple_bone_age_") as output_dir:
        result = predict(
            Path(input_path),
            str(params.get("sex", "female")).lower(),
            root / "model",
            root / "crop",
            Path(output_dir),
        )
        months = float(result["bone_age_months"])
        return _maple_finish(
            result,
            [result["crop_file"]],
            ["hand_crop_with_predicted_age"],
            {"pred": round(months), "pred_name": f"{months:.1f}_months"},
        )


if __name__ == "__main__":
    _cli()

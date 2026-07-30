"""Platform adapter for LC25000_Colon_ViT.

Loads the paired inference.py, validates its documented return contract, and
serializes images and predictions into a JSON-compatible platform payload.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load_inference(path: str, model_name: str):
    inference_path = Path(path)
    if not inference_path.is_file():
        raise FileNotFoundError(f"inference.py not found: {path}")
    module_name = f"_maple_{model_name}_inference"
    spec = importlib.util.spec_from_file_location(module_name, inference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load inference module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "main", None)):
        raise AttributeError(f"main() not found in {path}")
    return module


def _validate_image(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("image result must be a numpy.ndarray")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("image result must have shape (H, W, 3)")
    if value.dtype != np.uint8:
        raise TypeError("image result must use dtype uint8")
    return value


def _records(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        records = value.to_dict(orient="records")
    elif isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        records = [value]
    else:
        raise TypeError("predictions must be a DataFrame, dict, or list of dicts")
    if not all(isinstance(item, dict) for item in records):
        raise TypeError("each prediction must be a dict")
    return records


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _save_images(images: list[np.ndarray], output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded, files = [], []
    for index, image in enumerate(images):
        suffix = "" if len(images) == 1 else f"_{index + 1:02d}"
        output_path = output_dir / f"{stem}_result{suffix}.png"
        Image.fromarray(image).save(output_path)
        encoded.append(_encode_png(image))
        files.append(str(output_path))
    return encoded, files


def predict(input_path: str, output_dir: str, config: dict) -> dict:
    model_name = config.get("model_name")
    inference_path = config.get("inference_path")
    model_path = config.get("model_path")
    if not model_name or not inference_path or not model_path:
        raise ValueError("config requires model_name, inference_path, and model_path")

    returned = _load_inference(inference_path, model_name).main(input_path, model_path)
    result_type = config.get("result_type")
    output_path = Path(output_dir)
    stem = Path(input_path).stem

    if result_type == "text":
        predictions = _records(returned)
        if not all("pred" in row and "pred_name" in row for row in predictions):
            raise ValueError("text predictions require pred and pred_name fields")
        output_path.mkdir(parents=True, exist_ok=True)
        predictions_file = output_path / f"{stem}_predictions.json"
        safe_predictions = _json_safe(predictions)
        predictions_file.write_text(
            json.dumps(safe_predictions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "images_b64": [],
            "output_files": [str(predictions_file)],
            "predictions": safe_predictions,
        }

    images: list[np.ndarray]
    predictions: list[dict[str, Any]] = []
    if isinstance(returned, tuple):
        if len(returned) != 2:
            raise ValueError("image-plus-predictions result must contain two values")
        image_value, prediction_value = returned
        predictions = _records(prediction_value)
    else:
        image_value = returned
    if isinstance(image_value, list):
        images = [_validate_image(image) for image in image_value]
    else:
        images = [_validate_image(image_value)]
    images_b64, output_files = _save_images(images, output_path, stem)
    safe_predictions = _json_safe(predictions)
    if predictions:
        predictions_file = output_path / f"{stem}_predictions.json"
        predictions_file.write_text(
            json.dumps(safe_predictions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_files.append(str(predictions_file))
    return {
        "images_b64": images_b64,
        "output_files": output_files,
        "predictions": safe_predictions,
    }

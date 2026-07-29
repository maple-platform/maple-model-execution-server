"""Maple external-runtime adapter for RETFound APTOS2019 DR inference.

Contract:
    predict(input_path: str, output_dir: str, config: dict) -> dict
"""
from __future__ import annotations

import base64
import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load_inference(inference_path: str, model_name: str):
    path = Path(inference_path)
    if not path.is_file():
        raise FileNotFoundError(f"inference.py not found: {inference_path}")

    module_name = f"_maple_ophthalmology_{model_name.lower()}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    package_dir = str(path.parent)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inference module: {inference_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_overlay(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Grad-CAM must have shape (H, W, 3), got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"Grad-CAM must have dtype uint8, got {array.dtype}")
    return array


def _validate_predictions(predictions: Any) -> list[dict[str, Any]]:
    if not isinstance(predictions, list) or len(predictions) != 5:
        raise TypeError("RETFound inference must return five class prediction dicts")

    normalized = []
    for expected_index, row in enumerate(predictions):
        if not isinstance(row, dict):
            raise TypeError("Each prediction must be a dict")
        required = {"pred", "pred_name", "prob", "is_top_prediction"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Prediction is missing fields: {sorted(missing)}")
        if int(row["pred"]) != expected_index:
            raise ValueError("Prediction class indices must be ordered 0 through 4")
        normalized.append(
            {
                "pred": int(row["pred"]),
                "pred_name": str(row["pred_name"]),
                "prob": float(row["prob"]),
                "is_top_prediction": bool(row["is_top_prediction"]),
            }
        )

    probability_sum = sum(row["prob"] for row in normalized)
    if abs(probability_sum - 1.0) > 1e-5:
        raise ValueError(f"Class probabilities must sum to 1, got {probability_sum}")
    if sum(row["is_top_prediction"] for row in normalized) != 1:
        raise ValueError("Exactly one class must be marked as the top prediction")
    return normalized


def _png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def predict(input_path: str, output_dir: str, config: dict) -> dict:
    """Run inference and convert Maple's research-model output to API JSON."""
    model_name = str(config.get("model_name") or "RETFound_DR_APTOS2019_GradCAM")
    inference_path = config.get("inference_path")
    model_path = config.get("model_path")
    if not inference_path:
        raise ValueError("config must contain 'inference_path'")
    if not model_path:
        raise ValueError("config must contain 'model_path'")

    inference = _load_inference(str(inference_path), model_name)
    returned = inference.main(input_path, str(model_path))
    if not isinstance(returned, tuple) or len(returned) != 2:
        raise TypeError("RETFound inference main() must return (gradcam_image, predictions)")

    overlay = _validate_overlay(returned[0])
    predictions = _validate_predictions(returned[1])
    top_prediction = next(row for row in predictions if row["is_top_prediction"])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"{Path(input_path).stem}_gradcam.png"
    payload = _png_bytes(overlay)
    output_path.write_bytes(payload)

    return {
        "model": model_name,
        "image_b64": base64.b64encode(payload).decode("ascii"),
        "output_image_role": "gradcam_overlay",
        "predictions": predictions,
        "top_prediction": top_prediction,
        "gradcam_target": top_prediction["pred_name"],
        "output_file": str(output_path),
    }


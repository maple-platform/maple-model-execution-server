"""Maple external-runtime adapter for a MIMIC-CXR binary classifier."""
from __future__ import annotations

import importlib.util
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _load_inference(inference_path: str, model_name: str):
    path = Path(inference_path)
    if not path.is_file():
        raise FileNotFoundError(f"inference.py not found: {path}")
    module_name = f"_maple_pulmonology_{model_name.lower()}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inference module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def predict(input_path: str, output_dir: str, config: dict) -> dict:
    model_name = str(config["model_name"])
    inference = _load_inference(str(config["inference_path"]), model_name)
    returned = inference.main(input_path, str(config["model_path"]))
    if not isinstance(returned, tuple) or len(returned) != 2:
        raise TypeError("Inference main() must return (gradcam_overlay, predictions)")
    overlay, predictions = returned
    if not isinstance(overlay, np.ndarray) or overlay.ndim != 3 or overlay.shape[2] != 3:
        raise TypeError("Grad-CAM overlay must be an RGB numpy array")
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise TypeError("Binary inference must return one prediction")
    required = {"pred", "pred_name", "label", "probability", "threshold"}
    if not required.issubset(predictions[0]):
        raise ValueError(f"Prediction is missing: {sorted(required - set(predictions[0]))}")
    prediction = _json_value(predictions[0])
    buffer = io.BytesIO()
    Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    payload = {
        "model": model_name,
        "input": Path(input_path).name,
        "image_count": 1,
        "labels": ["gradcam_overlay"],
        "predictions": [prediction],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    overlay_path = output / f"{Path(input_path).stem}_gradcam.png"
    result_path = output / f"{Path(input_path).stem}_prediction.json"
    overlay_path.write_bytes(image_bytes)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        **payload,
        "images_b64": [base64.b64encode(image_bytes).decode("ascii")],
        "output_files": [str(overlay_path), str(result_path)],
    }

"""Maple external-runtime adapter for an orthopedic inference package.

Contract:
    predict(input_path: str, output_dir: str, config: dict) -> dict
"""
from __future__ import annotations

import base64
import importlib.util
import io
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load_inference(inference_path: str, model_name: str):
    path = Path(inference_path)
    if not path.is_file():
        raise FileNotFoundError(f"inference.py not found: {inference_path}")
    module_name = f"_maple_orthopedics_{model_name.lower()}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inference module: {inference_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def predict(input_path: str, output_dir: str, config: dict) -> dict:
    model_name = str(config["model_name"])
    inference = _load_inference(config["inference_path"], model_name)
    params = dict(config.get("params") or {})
    input_data = {"input_path": input_path, **params} if params else input_path
    returned = inference.main(input_data, str(config["model_path"]))
    if not isinstance(returned, tuple) or len(returned) != 2:
        raise TypeError("Orthopedic inference main() must return (images, predictions)")

    display, predictions = returned
    images = display if isinstance(display, list) else [display]
    roles = predictions[0].get("image_roles", []) if predictions else []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).name.replace(".nii.gz", "").rsplit(".", 1)[0]

    images_b64 = []
    output_files = []
    labels = []
    for index, image in enumerate(images):
        role = str(roles[index]) if index < len(roles) else f"result_{index:02d}"
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", role).strip("._")
        label = label or f"result_{index:02d}"
        payload = _png_bytes(image)
        output_path = output / f"{stem}_{label}.png"
        output_path.write_bytes(payload)
        labels.append(label)
        output_files.append(str(output_path))
        images_b64.append(base64.b64encode(payload).decode("ascii"))

    return {
        "model": model_name,
        "image_count": len(images_b64),
        "images_b64": images_b64,
        "labels": labels,
        "predictions": _json_value(predictions),
        "output_files": output_files,
    }

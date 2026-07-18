"""SynthStroke_T1 runner -- wraps AI_Models/Neurology/SynthStroke_T1/inference.py.

inference.main() returns a 3-tuple (images, labels, predictions):
    images      : list[np.ndarray (H,W,3) uint8]
    labels      : list[str]  (parallel to images)
    predictions : list[dict] (per-lesion connected-component stats)

Returns JSON-serializable dict:
    images_b64   : list[base64 PNG]
    labels       : list[str]
    output_files : list[str]
    predictions  : list[dict]
"""
import base64
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _load_inference_module(inference_path: str):
    p = Path(inference_path)
    if not p.exists():
        raise FileNotFoundError(f"inference.py not found: {inference_path}")
    module_name = f"_synthstroke_inference_{p.parent.name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ndarray_to_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def predict(input_path: str, output_dir: str, config: dict) -> dict:
    inference_path = config.get("inference_path")
    model_path = config.get("model_path")
    if not inference_path:
        raise ValueError("config must contain 'inference_path'")
    if not model_path:
        raise ValueError("config must contain 'model_path'")

    inference = _load_inference_module(inference_path)
    result = inference.main(input_path, model_path)

    # supports both 3-tuple (images, labels, predictions) and bare image list
    if isinstance(result, tuple) and len(result) == 3:
        images, labels, predictions = result
    else:
        images = result
        labels = [f"image_{i}" for i in range(len(images))]
        predictions = []

    stem = Path(input_path).stem.replace(".nii", "")
    output_files, images_b64 = [], []
    for label, img in zip(labels, images):
        out_path = Path(output_dir) / f"{stem}_{label}.png"
        Image.fromarray(np.asarray(img).astype(np.uint8)).save(out_path)
        output_files.append(str(out_path))
        images_b64.append(_ndarray_to_b64(np.asarray(img)))

    return {
        "images_b64": images_b64,
        "labels": list(labels),
        "output_files": output_files,
        "predictions": predictions,
    }

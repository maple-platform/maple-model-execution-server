"""RSNA Pneumonia runner — wraps AI_Models/Radiology/RSNA_Pneumonia_YOLO26x/inference.py.

predict() returns:
    image_b64       : base64 bounding-box overlay PNG
    predictions     : list of dicts (x1, y1, x2, y2, conf, pred, pred_name)
    detection_count : number of detected pneumonia opacities
    output_file     : saved PNG path
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
    module_name = "_rsna_inference"
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

    result_image, predictions = inference.main(input_path, model_path)

    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"{stem}_detection.png"
    Image.fromarray(result_image.astype(np.uint8)).save(out_path)

    return {
        "image_b64": _ndarray_to_b64(result_image),
        "predictions": predictions,
        "detection_count": len(predictions),
        "output_file": str(out_path),
    }

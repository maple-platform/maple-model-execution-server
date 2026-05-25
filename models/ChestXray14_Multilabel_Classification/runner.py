"""ChestXray14 runner — wraps AI_Models/Radiology/ChestXray14_Multilabel_Classification/inference.py.

predict() returns:
    image_b64         : base64 Grad-CAM overlay PNG
    predictions       : list of per-label dicts (label, prob, threshold, pred, pred_name)
    top_findings      : top-3 findings ranked by probability
    gradcam_target    : label used for Grad-CAM
    model             : model name string
    threshold_method  : description of threshold selection method
    output_file       : saved PNG path
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
    module_name = "_chestxray14_inference"
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
    if not inference_path:
        raise ValueError("config must contain 'inference_path'")

    inference = _load_inference_module(inference_path)

    result_image, predictions, model_output = inference.main(
        input_path,
        model_path=config.get("model_path"),
    )

    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"{stem}_gradcam.png"
    Image.fromarray(result_image.astype(np.uint8)).save(out_path)

    return {
        "image_b64": _ndarray_to_b64(result_image),
        "predictions": predictions,
        "top_findings": model_output["top_findings"],
        "gradcam_target": model_output["gradcam_target"],
        "model": model_output["model"],
        "threshold_method": model_output["threshold_method"],
        "output_file": str(out_path),
    }

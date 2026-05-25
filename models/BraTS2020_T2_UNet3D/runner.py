"""BraTS2020 T2 runner — wraps AI_Models/Neurology/BraTS2020_T2_UNet3D/inference.py.

predict() interface:
    input_path  : absolute path to a NIfTI file (.nii.gz)
    output_dir  : directory to write output PNG files
    config      : config.yaml dict — must contain 'model_path' and 'inference_path'

Returns JSON-serializable dict:
    images_b64   : list of base64-encoded PNG strings
    labels       : list of label strings matching images_b64
    output_files : list of saved PNG file paths
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
    module_name = f"_brats_inference_{p.parent.name}"
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
    results: list[np.ndarray] = inference.main(input_path, model_path)

    _labels_5 = ["3D", "all_regions", "WT", "TC", "ET"]
    _labels_4 = ["all_regions", "WT", "TC", "ET"]
    labels = _labels_5 if len(results) == 5 else _labels_4

    stem = Path(input_path).stem.replace(".nii", "")
    output_files = []
    images_b64 = []

    for label, img in zip(labels, results):
        out_path = Path(output_dir) / f"{stem}_{label}.png"
        Image.fromarray(img.astype(np.uint8)).save(out_path)
        output_files.append(str(out_path))
        images_b64.append(_ndarray_to_b64(img))

    return {
        "images_b64": images_b64,
        "labels": labels,
        "output_files": output_files,
    }

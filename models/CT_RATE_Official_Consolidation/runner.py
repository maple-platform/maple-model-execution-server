"""Maple external-runtime adapter for a CT-RATE classifier."""
from __future__ import annotations
import base64, importlib.util, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

def _load(path, name):
    module_name = f"_maple_ct_rate_{name.lower()}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inference module: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module
    spec.loader.exec_module(module); return module

def _json(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json(v) for v in value]
    return value

def predict(input_path: str, output_dir: str, config: dict) -> dict:
    name = str(config["model_name"])
    returned = _load(str(config["inference_path"]), name).main(input_path, str(config["model_path"]))
    if not isinstance(returned, tuple) or len(returned) != 2:
        raise TypeError("Inference main() must return (gradcam_montage, predictions)")
    image, predictions = returned
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("Grad-CAM montage must be an RGB numpy array")
    if not isinstance(predictions, list) or not predictions:
        raise TypeError("Predictions must be a non-empty list")
    buffer = io.BytesIO(); Image.fromarray(image.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    image_bytes = buffer.getvalue(); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).name.replace(".nii.gz", "").replace(".nii", "")
    image_path = output / f"{stem}_gradcam.png"; result_path = output / f"{stem}_prediction.json"
    payload = {"model": name, "input": Path(input_path).name, "image_count": 1,
               "labels": ["gradcam_axial_montage"], "predictions": _json(predictions)}
    image_path.write_bytes(image_bytes)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**payload, "images_b64": [base64.b64encode(image_bytes).decode("ascii")],
            "output_files": [str(image_path), str(result_path)]}

"""Maple external-runtime adapter for DeepLensNet cataract severity inference.

Contract:
    predict(input_path: str, output_dir: str, config: dict) -> dict

PROVISIONAL multi-image convention (needs administrator confirmation -- see
AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/IMPLEMENTATION_STATUS.md):
this model needs up to three photos per exam (slit-lamp, anterior/retro,
posterior/retro), which does not fit Maple's usual single-file ``input_path``
contract. Until the platform decides how multi-file exams reach ``POST /run``,
this adapter reads ``input_path`` as a small JSON manifest, any subset of:

    {"ns_image": "...", "cortical_image": "...", "psc_image": "..."}

Paths inside the manifest are resolved relative to the manifest's own
directory if not absolute.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

MANIFEST_KEYS = ("ns_image", "cortical_image", "psc_image")


def _json_safe(row: dict) -> dict:
    """Convert numpy/pandas scalar types (e.g. int64, float64) to native Python types."""
    safe = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


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


def _load_manifest(input_path: str) -> dict:
    manifest_path = Path(input_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Expected a JSON manifest at input_path, got missing file: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("The manifest JSON must be an object with any of: " + ", ".join(MANIFEST_KEYS))

    resolved = {}
    for key in MANIFEST_KEYS:
        value = manifest.get(key)
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        resolved[key] = str(candidate)
    if not resolved:
        raise ValueError("Manifest must include at least one of: " + ", ".join(MANIFEST_KEYS))
    return resolved


def _validate_result(result: Any) -> dict:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a hard requirement
        raise RuntimeError("pandas is required to validate the DeepLensNet result") from exc

    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"DeepLensNet inference must return a pd.DataFrame, got {type(result)}")
    if len(result) != 1:
        raise ValueError(f"Expected exactly one result row, got {len(result)}")
    required = {"pred", "pred_name", "ns_score", "pctcol_score", "pctpsc_score"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Result is missing required columns: {sorted(missing)}")

    return result.iloc[0].to_dict()


def predict(input_path: str, output_dir: str, config: dict) -> dict:
    """Run inference and convert Maple's research-model output to API JSON."""
    model_name = str(config.get("model_name") or "DeepLensNet_Cataract_Severity")
    inference_path = config.get("inference_path")
    model_path = config.get("model_path")
    if not inference_path:
        raise ValueError("config must contain 'inference_path'")
    if not model_path:
        raise ValueError("config must contain 'model_path'")

    inference = _load_inference(str(inference_path), model_name)
    photos = _load_manifest(input_path)
    result = inference.main(photos, str(model_path))
    row = _json_safe(_validate_result(result))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"{Path(input_path).stem}_result.json"
    output_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "model": model_name,
        "predictions": [row],
        "top_prediction": row,
        "output_file": str(output_path),
    }

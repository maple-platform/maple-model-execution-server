"""Generic CSV->text runner for clinical-score models.

Reads the input CSV into a DataFrame, calls inference.main(df, model_path),
and normalises the returned DataFrame into a JSON-serializable dict.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_inference_module(inference_path: str):
    p = Path(inference_path)
    if not p.exists():
        raise FileNotFoundError(f"inference.py not found: {inference_path}")
    module_name = f"_score_inference_{p.parent.name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def predict(input_path: str, output_dir: str, config: dict) -> dict:
    inference_path = config.get("inference_path")
    model_path = config.get("model_path")
    if not inference_path:
        raise ValueError("config must contain 'inference_path'")
    if not model_path:
        raise ValueError("config must contain 'model_path'")

    df = pd.read_csv(input_path)
    inference = _load_inference_module(inference_path)
    result = inference.main(df, model_path)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"inference.main() must return a pandas DataFrame, got {type(result)}")

    stem = Path(input_path).stem
    out_csv = Path(output_dir) / f"{stem}_result.csv"
    result.to_csv(out_csv, index=False)

    records = result.to_dict(orient="records")
    first = records[0] if records else {}
    return {
        "result_type": "text",
        "predictions": records,
        "pred": int(first.get("pred")) if "pred" in first else None,
        "pred_name": str(first.get("pred_name")) if "pred_name" in first else None,
        "output_file": str(out_csv),
    }

#!/usr/bin/env python3
"""ICH Score -- 30-day mortality risk after spontaneous intracerebral hemorrhage.

Source: Hemphill JC 3rd, Bonovich DC, Besmertis L, Manley GT, Johnston SC.
The ICH Score: a simple, reliable grading scale for intracerebral hemorrhage
outcome. Stroke. 2001;32(4):891-897.

Input  (pd.DataFrame, one row per patient) columns:
    gcs               : Glasgow Coma Scale total (3-15)
    age               : years
    ich_volume_ml     : hematoma volume (mL, e.g. ABC/2)
    ivh               : intraventricular hemorrhage present (0/1 or bool)
    infratentorial    : infratentorial origin (0/1 or bool)

Returns pd.DataFrame with: pred, pred_name, ich_score, mortality_30day_pct
Points may be forwarded from a segmentation model (e.g. hematoma volume, IVH).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _load_coef(model_path: str) -> dict:
    p = Path(model_path)
    if p.is_dir():
        p = p / "coefficients.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "present"}
    return bool(float(v)) if v is not None and str(v) != "nan" else False


def _gcs_points(gcs: float, table: list[dict]) -> int:
    for band in table:
        if gcs <= band["max"]:
            return band["points"]
    return 0


def _score_row(row: pd.Series, coef: dict) -> dict:
    c = coef["components"]
    score = 0
    score += _gcs_points(float(row["gcs"]), c["gcs"])
    score += c["age_ge_80"]["points_if_true"] if float(row["age"]) >= 80 else 0
    score += c["ich_volume_ge_30ml"]["points_if_true"] if float(row["ich_volume_ml"]) >= 30 else 0
    score += c["ivh_present"]["points_if_true"] if _as_bool(row.get("ivh")) else 0
    score += c["infratentorial"]["points_if_true"] if _as_bool(row.get("infratentorial")) else 0

    mortality = coef["mortality_30day_by_score"].get(str(score))
    name = next((b["name"] for b in coef["risk_bands"] if score <= b["max"]), "very high")
    pred = [b["name"] for b in coef["risk_bands"]].index(name) if name in [b["name"] for b in coef["risk_bands"]] else 0
    return {
        "pred": int(pred),
        "pred_name": name,
        "ich_score": int(score),
        "mortality_30day_pct": mortality,
    }


def main(input_data: pd.DataFrame, model_path: str) -> pd.DataFrame:
    coef = _load_coef(model_path)
    rows = [_score_row(r, coef) for _, r in input_data.iterrows()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else pd.DataFrame([
        {"gcs": 14, "age": 72, "ich_volume_ml": 12, "ivh": 0, "infratentorial": 0},
        {"gcs": 6, "age": 83, "ich_volume_ml": 45, "ivh": 1, "infratentorial": 1},
    ])
    print(main(df, str(Path(__file__).parent / "checkpoint")).to_string(index=False))

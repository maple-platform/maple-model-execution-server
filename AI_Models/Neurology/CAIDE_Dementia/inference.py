#!/usr/bin/env python3
"""CAIDE Dementia Risk Score -- 20-year risk of dementia from midlife factors.

Source: Kivipelto M, Ngandu T, Laatikainen T, Winblad B, Soininen H, Tuomilehto
J. Risk score for the prediction of dementia risk in 20 years among middle aged
people: a longitudinal, population-based study (CAIDE). Lancet Neurol.
2006;5(9):735-741.

This implements the 0-15 point version (without APOE). Input columns:
    age                 : years
    education_years     : years of formal education
    sex                 : 'male'/'female' or 1(male)/0
    sbp                 : systolic blood pressure (mmHg)
    bmi                 : body mass index (kg/m^2)
    cholesterol_mmol    : total cholesterol (mmol/L)  (mg/dL / 38.67 = mmol/L)
    physically_active   : 0/1  (1 = active)

Returns pd.DataFrame with: pred, pred_name, caide_score, dementia_risk_20yr_pct

NOTE: verify point thresholds against the source publication before clinical
use; an APOE e4 term (+2) extends the score to 0-18.
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


def _is_male(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"male", "m", "1"}
    return bool(float(v))


def _active(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "active", "y"}
    return bool(float(v)) if v is not None and str(v) != "nan" else False


def _points_max(value: float, table: list[dict]) -> int:
    for band in table:
        if value <= band["max"]:
            return band["points"]
    return table[-1]["points"]


def _points_min(value: float, table: list[dict]) -> int:
    for band in table:
        if value >= band["min"]:
            return band["points"]
    return table[-1]["points"]


def _score_row(row: pd.Series, coef: dict) -> dict:
    c = coef["components"]
    score = 0
    score += _points_max(float(row["age"]), c["age"])
    score += _points_min(float(row["education_years"]), c["education_years"])
    score += c["sex_male"]["points_if_true"] if _is_male(row.get("sex")) else 0
    score += c["sbp_gt_140"]["points_if_true"] if float(row["sbp"]) > 140 else 0
    score += c["bmi_gt_30"]["points_if_true"] if float(row["bmi"]) > 30 else 0
    score += c["cholesterol_gt_6_5mmol"]["points_if_true"] if float(row["cholesterol_mmol"]) > 6.5 else 0
    score += c["physically_inactive"]["points_if_true"] if not _active(row.get("physically_active")) else 0

    band = next((b for b in coef["risk_20yr_by_band"] if score <= b["max"]),
                coef["risk_20yr_by_band"][-1])
    pred = coef["risk_20yr_by_band"].index(band)
    return {
        "pred": int(pred),
        "pred_name": band["name"],
        "caide_score": int(score),
        "dementia_risk_20yr_pct": band["pct"],
    }


def main(input_data: pd.DataFrame, model_path: str) -> pd.DataFrame:
    coef = _load_coef(model_path)
    rows = [_score_row(r, coef) for _, r in input_data.iterrows()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else pd.DataFrame([
        {"age": 45, "education_years": 14, "sex": "female", "sbp": 128, "bmi": 24,
         "cholesterol_mmol": 5.0, "physically_active": 1},
        {"age": 58, "education_years": 6, "sex": "male", "sbp": 155, "bmi": 32,
         "cholesterol_mmol": 7.2, "physically_active": 0},
    ])
    print(main(df, str(Path(__file__).parent / "checkpoint")).to_string(index=False))

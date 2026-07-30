#!/usr/bin/env python3
"""ASTRAL Score -- predicts unfavorable 90-day functional outcome (mRS>2)
after acute ischemic stroke.

Source: Ntaios G, Faouzi M, Ferrari J, Lang W, Vemmos K, Michel P. An
integer-based score to predict functional outcome in acute ischemic stroke:
the ASTRAL score. Neurology. 2012;78(24):1916-1922.

Components:
    age                         : years          (0.2 pt per year = 1 pt / 5 yr)
    nihss                       : admission NIHSS (1 pt per point)
    onset_to_admission_hours    : >3 h -> 2 pts
    visual_field_defect         : present (0/1) -> 2 pts
    glucose_mmol                : >7.3 or <3.7 mmol/L -> 1 pt   (mg/dL / 18 = mmol/L)
    decreased_consciousness     : NIHSS 1a > 0 (0/1) -> 3 pts

Returns pd.DataFrame with: pred, pred_name, astral_score

NOTE: risk_bands here are indicative categories for triage display. The
validated quantitative anchor is ASTRAL ~= 31 -> ~50% unfavorable outcome.
Verify banding against the source before clinical use.
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


def _score_row(row: pd.Series, coef: dict) -> dict:
    c = coef["components"]
    score = 0.0
    score += float(row["age"]) * c["age_points_per_5yr"]
    score += float(row["nihss"]) * c["nihss_points_per_point"]
    score += c["onset_to_admission_gt_3h"] if float(row["onset_to_admission_hours"]) > 3 else 0
    score += c["visual_field_defect"] if _as_bool(row.get("visual_field_defect")) else 0
    g = float(row["glucose_mmol"])
    score += c["glucose_abnormal"]["points"] if (g > c["glucose_abnormal"]["high_mmol"]
                                                 or g < c["glucose_abnormal"]["low_mmol"]) else 0
    score += c["decreased_consciousness"] if _as_bool(row.get("decreased_consciousness")) else 0
    score = round(score, 1)

    name = next((b["name"] for b in coef["risk_bands"] if score <= b["max"]), "very high")
    pred = [b["name"] for b in coef["risk_bands"]].index(name)
    return {"pred": int(pred), "pred_name": name, "astral_score": score}


def main(input_data: pd.DataFrame, model_path: str) -> pd.DataFrame:
    coef = _load_coef(model_path)
    rows = [_score_row(r, coef) for _, r in input_data.iterrows()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else pd.DataFrame([
        {"age": 68, "nihss": 8, "onset_to_admission_hours": 2, "visual_field_defect": 0,
         "glucose_mmol": 6.0, "decreased_consciousness": 0},
        {"age": 82, "nihss": 18, "onset_to_admission_hours": 5, "visual_field_defect": 1,
         "glucose_mmol": 9.1, "decreased_consciousness": 1},
    ])
    print(main(df, str(Path(__file__).parent / "checkpoint")).to_string(index=False))

#!/usr/bin/env python3
"""GPA -- Graded Prognostic Assessment for patients with brain metastases.

Source: Sperduto PW, Berkey B, Gaspar LE, Mehta M, Curran W. A new prognostic
index and comparison to three other indices for patients with brain metastases:
an analysis of 1,960 patients in the RTOG database. Int J Radiat Oncol Biol
Phys. 2008;70(2):510-514.

Original ("basic") GPA uses four factors, each 0 / 0.5 / 1.0:
    age                : years           (<50 =1.0, 50-59 =0.5, >=60 =0)
    kps                : Karnofsky (0-100) (90-100 =1.0, 70-80 =0.5, <70 =0)
    num_cns_mets       : # of CNS metastases (1 =1.0, 2-3 =0.5, >3 =0)
    extracranial_mets  : extracranial metastases present (0/1) (absent =1.0)
Total 0-4.0 -> median survival group.

Returns pd.DataFrame with: pred, pred_name, gpa_score, median_survival_months
`num_cns_mets` can be forwarded from a brain-metastasis detection model.
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


def _points_max_table(value: float, table: list[dict]) -> float:
    for band in table:
        if value <= band["max"]:
            return band["points"]
    return 0.0


def _points_min_table(value: float, table: list[dict]) -> float:
    for band in table:
        if value >= band["min"]:
            return band["points"]
    return 0.0


def _score_row(row: pd.Series, coef: dict) -> dict:
    c = coef["components"]
    score = 0.0
    score += _points_max_table(float(row["age"]), c["age"])
    score += _points_min_table(float(row["kps"]), c["kps"])
    score += _points_max_table(float(row["num_cns_mets"]), c["num_cns_mets"])
    score += c["extracranial_mets"]["points_if_present"] if _as_bool(row.get("extracranial_mets")) \
        else c["extracranial_mets"]["points_if_absent"]
    score = round(score, 1)

    band = next((b for b in coef["median_survival_months"] if score <= b["max"]),
                coef["median_survival_months"][-1])
    pred = coef["median_survival_months"].index(band)
    return {
        "pred": int(pred),
        "pred_name": band["name"],
        "gpa_score": score,
        "median_survival_months": band["months"],
    }


def main(input_data: pd.DataFrame, model_path: str) -> pd.DataFrame:
    coef = _load_coef(model_path)
    rows = [_score_row(r, coef) for _, r in input_data.iterrows()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else pd.DataFrame([
        {"age": 45, "kps": 90, "num_cns_mets": 1, "extracranial_mets": 0},
        {"age": 66, "kps": 60, "num_cns_mets": 5, "extracranial_mets": 1},
    ])
    print(main(df, str(Path(__file__).parent / "checkpoint")).to_string(index=False))

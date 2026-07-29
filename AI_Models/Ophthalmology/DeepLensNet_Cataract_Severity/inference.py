"""Maple inference entrypoint for DeepLensNet cataract severity quantification.

Research/non-commercial use only. NCBI's repository states: "not intended for
commercial use or purposes beyond research use only." No formal LICENSE file
is published upstream; treat as research-use-only pending legal confirmation.

Original project: https://github.com/ncbi/deeplensnet
Paper: Keenan, Chen, Agron, et al., "DeepLensNet: Deep Learning Automated
Diagnosis and Quantitative Classification of Cataract Type and Severity",
Ophthalmology (2022).
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

# (width, height) fed to InceptionV3, matching the official model_classify.py.
INPUT_SIZE = (334, 501)

# input_data dict key -> (weight file stem, human-readable scale description).
VARIABLES: dict[str, tuple[str, str]] = {
    "ns_image": (
        "NS",
        "Nuclear sclerosis grade from a 45-degree slit-lamp photo "
        "(AREDS-like scale, approx. 0.9-7.1; higher = more severe).",
    ),
    "cortical_image": (
        "PCTCOL",
        "Cortical lens opacity from an anterior/retroillumination photo "
        "(percent of lens area, 0-100).",
    ),
    "psc_image": (
        "PCTPSC",
        "Posterior subcapsular opacity from a posterior/retroillumination photo "
        "(percent of lens area, 0-100).",
    ),
}

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_CACHE_LOCK = Lock()


def _resolve_checkpoint_dir(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"model_path must be the DeepLensNet checkpoint directory containing "
            f"NS.h5 / PCTCOL.h5 / PCTPSC.h5, got: {path}"
        )
    return path


def _load_variable_model(model_dir: Path, variable: str):
    from keras.models import load_model

    cache_key = (str(model_dir), variable)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        weight_path = model_dir / f"{variable}.h5"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {weight_path}")
        model = load_model(str(weight_path))
        _MODEL_CACHE[cache_key] = model
        return model


def _preprocess(image_path: str) -> np.ndarray:
    import keras.applications.inception_v3 as inception_v3

    path = Path(image_path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Each image path must be PNG/JPG/JPEG, got: {path.suffix}")
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    with Image.open(path) as opened:
        resized = opened.convert("RGB").resize(INPUT_SIZE)
    array = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    return inception_v3.preprocess_input(array)


def main(input_data: dict, model_path: str):
    """Score up to three cataract axes from up to three anterior-segment photos.

    Unlike Maple's single-file image contract, DeepLensNet was trained on three
    distinct photographs per eye taken during the same exam, so ``input_data``
    is a dict rather than a single path (mirrors the Manual's pipeline dict
    convention, but this model is standalone -- all three files come from one
    exam, not from a prior model's output).

    Args:
        input_data: dict, at least one key required:
            - "ns_image": str | None -- 45-degree slit-lamp photo (nuclear sclerosis)
            - "cortical_image": str | None -- anterior/retroillumination photo (cortical opacity)
            - "psc_image": str | None -- posterior/retroillumination photo (PSC opacity)
            A missing or falsy key is skipped; that axis is scored as unavailable.
        model_path: directory containing NS.h5, PCTCOL.h5, PCTPSC.h5.

    Returns:
        pd.DataFrame, one row. ``pred`` is the count of axes actually scored and
        ``pred_name`` is a human-readable "VARIABLE=value" summary of only the
        scored axes -- DeepLensNet is a multi-output regressor, not a classifier,
        so there is no single categorical label; the per-axis scores are the
        real result and are carried in ``ns_score`` / ``pctcol_score`` /
        ``pctpsc_score`` (``None`` for axes that were not scored).
    """
    if not isinstance(input_data, dict):
        raise TypeError("input_data must be a dict; see main.__doc__ for the expected keys.")
    if not model_path:
        raise ValueError("model_path must point to the DeepLensNet checkpoint directory.")

    provided = {key: value for key, value in input_data.items() if key in VARIABLES and value}
    if not provided:
        raise ValueError(
            "input_data must include at least one of: " + ", ".join(VARIABLES)
        )

    model_dir = _resolve_checkpoint_dir(model_path)

    scores: dict[str, float | None] = {}
    for key, (variable, _scale) in VARIABLES.items():
        image_path = provided.get(key)
        if not image_path:
            scores[variable] = None
            continue
        tensor = _preprocess(str(image_path))
        model = _load_variable_model(model_dir, variable)
        prediction = model.predict(tensor, verbose=0)
        scores[variable] = float(prediction[0][0])

    scored = {name: value for name, value in scores.items() if value is not None}
    if not scored:
        raise RuntimeError("No cataract axis could be scored from the given input.")

    import pandas as pd

    result_df = pd.DataFrame(
        {
            "pred": [len(scored)],
            "pred_name": ["; ".join(f"{name}={value:.4f}" for name, value in scored.items())],
            "ns_score": [scores["NS"]],
            "pctcol_score": [scores["PCTCOL"]],
            "pctpsc_score": [scores["PCTPSC"]],
        }
    )
    return result_df

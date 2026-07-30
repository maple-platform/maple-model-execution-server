"""Maple inference entrypoint for DeepSeeNet's AREDS Simplified Severity Score.

Public Domain -- "United States Government Work" (NCBI). Research use only
per NCBI's own "not intended for commercial use" notice; see NOTICE.md.

Original project: https://github.com/ncbi-nlp/DeepSeeNet
Paper: Peng Y, Dharssi S, Chen Q, Keenan T, Agron E, Wong W, Chew E, Lu Z.
"DeepSeeNet: A deep learning model for automated classification of
patient-based age-related macular degeneration severity from color fundus
photographs." Ophthalmology. 2019;126(4):565-575.
Targets the official Python 3.6 runtime (matches keras==2.2.4 / tensorflow==
1.15.5 in requirements.txt), so this file intentionally avoids syntax newer
than Python 3.6 (no PEP 585 generics, no ``from __future__ import annotations``).
"""
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)
DRUSEN_LABELS = {0: "small/none", 1: "intermediate", 2: "large"}
BINARY_LABELS = {0: "no", 1: "yes"}

# input_data key -> checkpoint filename, matching the official repo exactly.
RISK_MODELS = {
    "drusen": "drusen_model.h5",
    "pigment": "pigment_model.h5",
    "advanced_amd": "adv_amd_model.h5",
}

_MODEL_CACHE = {}  # type: Dict[Tuple[str, str], Any]
_CACHE_LOCK = Lock()


def _resolve_checkpoint_dir(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            "model_path must be the DeepSeeNet checkpoint directory containing "
            f"drusen_model.h5 / pigment_model.h5 / adv_amd_model.h5, got: {path}"
        )
    return path


def _load_risk_model(model_dir: Path, filename: str):
    from keras.models import load_model

    cache_key = (str(model_dir), filename)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        weight_path = model_dir / filename
        if not weight_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {weight_path}")
        model = load_model(str(weight_path))
        _MODEL_CACHE[cache_key] = model
        return model


def _crop2square(img: Image.Image) -> Image.Image:
    """Center-crop to a square based on the short edge (matches deepseenet.utils)."""
    short_side = min(img.size)
    x0 = (img.size[0] - short_side) / 2
    y0 = (img.size[1] - short_side) / 2
    x1 = img.size[0] - x0
    y1 = img.size[1] - y0
    return img.crop((x0, y0, x1, y1))


def _preprocess(image_path: str) -> np.ndarray:
    from keras.applications import imagenet_utils

    path = Path(image_path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Each eye image must be PNG/JPG/JPEG, got: {path.suffix}")
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    with Image.open(path) as opened:
        square = _crop2square(opened.convert("RGB")).resize(TARGET_SIZE)
    array = np.expand_dims(np.asarray(square, dtype=np.float32), axis=0)
    return imagenet_utils.preprocess_input(array, mode="tf")


def _score_eye(model_dir, image_path):
    # type: (Path, str) -> Dict[str, int]
    tensor = _preprocess(image_path)
    results = {}  # type: Dict[str, int]
    for key, filename in RISK_MODELS.items():
        model = _load_risk_model(model_dir, filename)
        prediction = model.predict(tensor)
        results[key] = int(np.argmax(prediction, axis=1)[0])
    return results


def _simplified_score(left, right):
    # type: (Dict[str, int], Dict[str, int]) -> int
    """AREDS Simplified Severity Score, matching deepseenet_simplified.get_simplified_score."""
    score = 0
    if left["advanced_amd"] == 1:
        score += 5
    if right["advanced_amd"] == 1:
        score += 5
    if left["pigment"] == 1:
        score += 1
    if right["pigment"] == 1:
        score += 1
    if left["drusen"] == 2:
        score += 1
    if right["drusen"] == 2:
        score += 1
    if left["drusen"] == 1 and right["drusen"] == 1:
        score += 1
    return 5 if score >= 5 else score


def main(input_data: dict, model_path: str):
    """Grade one patient's AREDS Simplified Severity Score from two fundus photos.

    Args:
        input_data: dict, both keys required (the score needs both eyes):
            - "left_eye": str -- path to the left eye color fundus photo
            - "right_eye": str -- path to the right eye color fundus photo
        model_path: directory containing drusen_model.h5, pigment_model.h5,
            adv_amd_model.h5 (the official AREDS-trained risk-factor models).

    Returns:
        pd.DataFrame, one row. ``pred`` is the AREDS Simplified Severity Score
        (integer 0-5): +5 if either eye shows advanced AMD, +1 per eye with a
        pigmentary abnormality, +1 per eye with large drusen, +1 more if both
        eyes have intermediate drusen, capped at 5. ``pred_name`` states the
        score; the per-eye risk-factor calls behind it are in the remaining
        columns.
    """
    if not isinstance(input_data, dict):
        raise TypeError("input_data must be a dict with 'left_eye' and 'right_eye'.")
    if not model_path:
        raise ValueError("model_path must point to the DeepSeeNet checkpoint directory.")

    left_path = input_data.get("left_eye")
    right_path = input_data.get("right_eye")
    if not left_path or not right_path:
        raise ValueError("input_data must include both 'left_eye' and 'right_eye'.")

    model_dir = _resolve_checkpoint_dir(model_path)
    left = _score_eye(model_dir, str(left_path))
    right = _score_eye(model_dir, str(right_path))
    score = _simplified_score(left, right)

    import pandas as pd

    result_df = pd.DataFrame(
        {
            "pred": [score],
            "pred_name": [f"AREDS Simplified Severity Score = {score}"],
            "drusen_left": [DRUSEN_LABELS[left["drusen"]]],
            "drusen_right": [DRUSEN_LABELS[right["drusen"]]],
            "pigment_left": [BINARY_LABELS[left["pigment"]]],
            "pigment_right": [BINARY_LABELS[right["pigment"]]],
            "advanced_amd_left": [BINARY_LABELS[left["advanced_amd"]]],
            "advanced_amd_right": [BINARY_LABELS[right["advanced_amd"]]],
        }
    )
    return result_df

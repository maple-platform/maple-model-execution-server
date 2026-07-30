"""Local contract validation for the Maple inference package."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from inference import main


def validate(image: str, checkpoint: str, output: str | None = None) -> None:
    overlay, predictions = main(image, checkpoint)
    assert isinstance(overlay, np.ndarray)
    assert overlay.dtype == np.uint8
    assert overlay.shape == (224, 224, 3)
    assert isinstance(predictions, list) and len(predictions) == 3
    probabilities = [row["prob"] for row in predictions]
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
    assert abs(sum(probabilities) - 1.0) < 1e-5
    assert sum(bool(row["is_top_prediction"]) for row in predictions) == 1
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(output_path)
        output_path.with_suffix(".json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(predictions, ensure_ascii=False, indent=2))
    print("PASS: Maple image/classification contract and probability checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    validate(args.image, args.checkpoint, args.output)

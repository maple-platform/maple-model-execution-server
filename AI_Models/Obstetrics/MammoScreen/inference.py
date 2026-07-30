"""Single- or multi-view MammoScreen inference."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file

from src.configuration import MammoConfig
from src.modeling import MammoEnsemble


def _load_image(path):
    return np.asarray(Image.open(path).convert("L")).copy()


def main(input_data, model_path: str):
    # The platform normally supplies one path. A dict with cc/mlo paths is also
    # accepted for paired-view integration by an administrator.
    if isinstance(input_data, dict):
        views = {key: _load_image(value) for key, value in input_data.items() if key in {"cc", "mlo"}}
    else:
        views = {"view": _load_image(input_data)}
    if not views:
        raise ValueError("At least one mammography image is required")
    model = MammoEnsemble(MammoConfig())
    weights = Path(model_path)
    if weights.is_dir():
        weights = weights / "model.safetensors"
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(views, device="cpu")
    cancer = float(output["cancer"][0])
    density_scores = output["density"][0].cpu().tolist()
    density_labels = ["A", "B", "C", "D"]
    density_index = int(np.argmax(density_scores))
    pred = int(cancer >= 0.5)
    result = {
        "pred": [pred],
        "pred_name": ["cancer positive" if pred else "cancer negative"],
        "prob": [cancer if pred else 1.0 - cancer],
        "cancer_score": [cancer],
        "predicted_density": [density_labels[density_index]],
    }
    result.update({f"density_{label}": [score] for label, score in zip(density_labels, density_scores)})
    return pd.DataFrame(result)

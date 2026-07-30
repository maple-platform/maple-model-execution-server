"""MammoCrop breast/background ROI localization."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from src.configuration import MammoCropConfig
from src.modeling import MammoCropModel


def main(input_data, model_path: str):
    source = Image.open(input_data).convert("L")
    image = np.asarray(source)
    model = MammoCropModel(MammoCropConfig())
    weights = Path(model_path)
    if weights.is_dir():
        weights = weights / "model.safetensors"
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    model.eval()
    resized = model.preprocess(image)
    tensor = torch.from_numpy(resized.copy()).unsqueeze(0).unsqueeze(0).float()
    shape = torch.tensor([image.shape[:2]])
    with torch.inference_mode():
        raw_box = model(tensor, shape)[0].tolist()
    x, y, width, height = [int(value) for value in raw_box]
    x = max(0, min(x, source.width - 1))
    y = max(0, min(y, source.height - 1))
    width = max(1, min(width, source.width - x))
    height = max(1, min(height, source.height - y))
    overlay = source.convert("RGB")
    ImageDraw.Draw(overlay).rectangle((x, y, x + width, y + height), outline=(255, 0, 0), width=max(3, source.width // 300))
    prediction = [{
        "pred": 1,
        "pred_name": "breast ROI",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }]
    return np.asarray(overlay, dtype=np.uint8), prediction

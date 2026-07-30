"""LC25000 colon histopathology inference."""
import json
from pathlib import Path

import pandas as pd
import timm
import torch
from PIL import Image
from safetensors.torch import load_file

LABELS = ["Benign colon tissue", "Colon adenocarcinoma"]


def main(input_data, model_path: str):
    checkpoint = Path(model_path)
    config_path = checkpoint / "config.json" if checkpoint.is_dir() else checkpoint.parent / "config.json"
    weights_path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    config = json.loads(config_path.read_text())
    model = timm.create_model(config["architecture"], pretrained=False, num_classes=config["num_classes"])
    model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model, pretrained_cfg=config["pretrained_cfg"])
    transform = timm.data.create_transform(**data_config, is_training=False)
    with torch.inference_mode():
        scores = torch.softmax(model(transform(Image.open(input_data).convert("RGB")).unsqueeze(0)), dim=1)[0]
    top = int(scores.argmax())
    return pd.DataFrame({
        "pred": [top],
        "pred_name": [LABELS[top]],
        "prob": [float(scores[top])],
        "prob_benign": [float(scores[0])],
        "prob_adenocarcinoma": [float(scores[1])],
    })

"""SIPaKMeD cervical-cell morphology inference."""
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms


def main(input_data, model_path: str):
    checkpoint = Path(model_path)
    labels_path = checkpoint.parent / "labels.json" if checkpoint.is_file() else checkpoint / "labels.json"
    weights_path = checkpoint if checkpoint.is_file() else checkpoint / "best_alexnet.pth"
    label_map = json.loads(labels_path.read_text())
    labels = [label_map[str(index)] for index in range(len(label_map))]
    model = models.alexnet(weights=None)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(labels))
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.inference_mode():
        scores = torch.softmax(model(preprocess(Image.open(input_data).convert("RGB")).unsqueeze(0)), dim=1)[0]
    top = int(scores.argmax())
    result = {"pred": [top], "pred_name": [labels[top]], "prob": [float(scores[top])]}
    result.update({f"prob_{label}": [float(scores[index])] for index, label in enumerate(labels)})
    return pd.DataFrame(result)

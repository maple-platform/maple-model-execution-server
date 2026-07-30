"""Eight-class Kvasir-V2 endoscopy inference."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor

DISPLAY = {
    "dyed-lifted-polyps": "Dyed lifted polyp",
    "dyed-resection-margins": "Dyed resection margin",
    "esophagitis": "Esophagitis",
    "normal-cecum": "Normal cecum",
    "normal-pylorus": "Normal pylorus",
    "normal-z-line": "Normal Z-line",
    "polyps": "Polyp",
    "ulcerative-colitis": "Ulcerative colitis",
}


def main(input_data, model_path: str):
    checkpoint = Path(model_path)
    processor = ViTImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(checkpoint, local_files_only=True).eval()
    inputs = processor(images=Image.open(input_data).convert("RGB"), return_tensors="pt")
    with torch.inference_mode():
        scores = torch.softmax(model(**inputs).logits, dim=1)[0]
    labels = [DISPLAY.get(model.config.id2label[index], model.config.id2label[index]) for index in range(len(scores))]
    top = int(scores.argmax())
    result = {"pred": [top], "pred_name": [labels[top]], "prob": [float(scores[top])]}
    result.update({f"prob_{label}": [float(scores[index])] for index, label in enumerate(labels)})
    return pd.DataFrame(result)

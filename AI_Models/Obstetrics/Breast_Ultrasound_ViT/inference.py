"""Three-class breast-ultrasound inference."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main(input_data, model_path: str):
    checkpoint = Path(model_path)
    processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(checkpoint, local_files_only=True).eval()
    image = Image.open(input_data).convert("RGB")
    with torch.inference_mode():
        scores = torch.softmax(model(**processor(images=image, return_tensors="pt")).logits, dim=1)[0]
    top = int(scores.argmax())
    labels = [model.config.id2label[index] for index in range(len(scores))]
    result = {"pred": [top], "pred_name": [labels[top]], "prob": [float(scores[top])]}
    result.update({f"prob_{label}": [float(scores[index])] for index, label in enumerate(labels)})
    return pd.DataFrame(result)

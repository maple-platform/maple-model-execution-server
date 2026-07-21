#!/usr/bin/env python3
"""Maple entry point for ViT skin-lesion classification with Grad-CAM."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

_CACHE: dict[str, tuple[Any, Any]] = {}
_LABELS = {
    "akiec": "Actinic keratosis", "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis", "df": "Dermatofibroma",
    "mel": "Melanoma", "nv": "Melanocytic nevus", "vasc": "Vascular lesion",
    "Bénin": "Benign", "Mélanome": "Melanoma", "Carcinome": "Carcinoma",
}


def _load(model_path: str):
    key = str(Path(model_path).resolve())
    if key not in _CACHE:
        processor = AutoImageProcessor.from_pretrained(key, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(
            key, local_files_only=True, use_safetensors=True
        ).eval()
        _CACHE[key] = (processor, model)
    return _CACHE[key]


def _token_cam(activation: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
    activation, gradient = activation[0], gradient[0]
    count = activation.shape[0]
    if int((count - 1) ** 0.5) ** 2 == count - 1:
        activation, gradient, count = activation[1:], gradient[1:], count - 1
    side = int(count ** 0.5)
    if side * side != count:
        raise RuntimeError(f"Unexpected visual token count: {count}")
    activation = activation.reshape(side, side, -1)
    gradient = gradient.reshape(side, side, -1)
    weights = gradient.mean(dim=(0, 1), keepdim=True)
    cam = torch.relu((activation * weights).sum(dim=-1)).detach().cpu().numpy()
    cam -= cam.min()
    return cam / (cam.max() + 1e-8)


def _overlay(image: Image.Image, cam: np.ndarray) -> np.ndarray:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    heat = np.asarray(
        Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    color = np.zeros_like(base)
    color[..., 0] = np.clip(heat * 2.0, 0, 1) * 255
    color[..., 1] = np.clip(1.0 - np.abs(heat - 0.6) / 0.6, 0, 1) * 220
    result = 0.55 * base + 0.45 * color
    return np.clip(result, 0, 255).astype(np.uint8)


def main(input_data, model_path: str):
    """Return ``(gradcam_rgb, classification_predictions)``."""
    image = Image.open(str(input_data)).convert("RGB")
    processor, model = _load(model_path)
    holder: dict[str, torch.Tensor] = {}
    layers = getattr(model.vit, "layers", None)
    if layers is None:
        layers = model.vit.encoder.layer
    target_layer = layers[-1].layernorm_before

    def capture(_module, _inputs, output):
        holder["activation"] = output
        output.register_hook(lambda grad: holder.__setitem__("gradient", grad))

    hook = target_layer.register_forward_hook(capture)
    output = model(**processor(images=image, return_tensors="pt"))
    top_index = int(output.logits[0].argmax())
    model.zero_grad(set_to_none=True)
    output.logits[0, top_index].backward()
    hook.remove()

    probabilities = torch.softmax(output.logits.detach(), dim=1)[0].cpu().numpy()
    labels = [_LABELS.get(model.config.id2label[i], model.config.id2label[i]) for i in range(len(probabilities))]
    ranking = np.argsort(probabilities)[::-1]
    predictions = [
        {"pred": int(i), "pred_name": str(labels[i]), "prob": float(probabilities[i])}
        for i in ranking
    ]
    cam = _token_cam(holder["activation"], holder["gradient"])
    return _overlay(image, cam), predictions


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result, predictions = main(root / "sample_data/sample_melanoma.jpg", str(root / "checkpoint"))
    print(result.shape, result.dtype, predictions[0])

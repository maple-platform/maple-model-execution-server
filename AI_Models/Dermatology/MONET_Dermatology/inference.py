#!/usr/bin/env python3
"""Maple entry point for image-text dermatology inference with Grad-CAM."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent
_CACHE: dict[str, tuple[Any, Any]] = {}


def _load(model_path: str):
    key = str(Path(model_path).resolve())
    if key not in _CACHE:
        processor = AutoProcessor.from_pretrained(key, local_files_only=True)
        model = AutoModel.from_pretrained(
            key, local_files_only=True, use_safetensors=True
        ).eval()
        _CACHE[key] = (processor, model)
    return _CACHE[key]


def _cam(activation: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
    activation, gradient = activation[0], gradient[0]
    count = activation.shape[0]
    if int((count - 1) ** .5) ** 2 == count - 1:
        activation, gradient, count = activation[1:], gradient[1:], count - 1
    side = int(count ** .5)
    if side * side != count:
        raise RuntimeError(f"Unexpected visual token count: {count}")
    activation = activation.reshape(side, side, -1)
    gradient = gradient.reshape(side, side, -1)
    weights = gradient.mean(dim=(0, 1), keepdim=True)
    cam = torch.relu((activation * weights).sum(-1)).detach().cpu().numpy()
    cam -= cam.min()
    return cam / (cam.max() + 1e-8)


def _overlay(image: Image.Image, cam: np.ndarray) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32)
    heat = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255
    color = np.zeros_like(base); color[..., 0] = np.clip(heat * 2, 0, 1) * 255
    color[..., 1] = np.clip(1 - np.abs(heat - .6) / .6, 0, 1) * 220
    return np.clip(.55 * base + .45 * color, 0, 255).astype(np.uint8)


def main(input_data, model_path: str):
    spec = json.loads((BASE_DIR / "ref/candidates.json").read_text())
    labels, template = spec["labels"], spec["prompt_template"]
    image = Image.open(str(input_data)).convert("RGB")
    processor, model = _load(model_path)
    vision = model.vision_model
    layers = getattr(vision, "layers", None)
    if layers is None:
        layers = vision.encoder.layers
    holder: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        holder["activation"] = output
        output.register_hook(lambda grad: holder.__setitem__("gradient", grad))

    hook = layers[-1].layer_norm1.register_forward_hook(capture)
    inputs = processor(
        images=image, text=[template.format(label=x) for x in labels],
        padding=True, return_tensors="pt",
    )
    output = model(**inputs)
    top_index = int(output.logits_per_image[0].argmax())
    model.zero_grad(set_to_none=True)
    output.logits_per_image[0, top_index].backward()
    hook.remove()
    probabilities = torch.softmax(output.logits_per_image.detach(), dim=1)[0].cpu().numpy()
    ranking = np.argsort(probabilities)[::-1]
    predictions = [
        {"pred": int(i), "pred_name": str(labels[i]), "prob": float(probabilities[i])}
        for i in ranking
    ]
    return _overlay(image, _cam(holder["activation"], holder["gradient"])), predictions


if __name__ == "__main__":
    sample = next((BASE_DIR / "sample_data").glob("*"))
    result, predictions = main(sample, str(BASE_DIR / "checkpoint"))
    print(result.shape, result.dtype, predictions[0])

"""Maple binary-classification entry point for one MIMIC-CXR checkpoint."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models


_MODEL = None
_DEVICE = None
_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _make_model(architecture: str) -> torch.nn.Module:
    builders = {
        "densenet121": models.densenet121,
        "efficientnet_b0": models.efficientnet_b0,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    try:
        return builders[architecture](weights=None, num_classes=1)
    except KeyError as exc:
        raise ValueError(f"Unsupported architecture: {architecture}") from exc


def _preprocess(input_data):
    if not isinstance(input_data, (str, Path)):
        raise TypeError("input_data must be a PNG/JPG image path")
    path = Path(input_data)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported input type: {path.suffix}")
    original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"Could not decode image: {path}")
    image = cv2.resize(original, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
    tensor = tensor.div(255.0).sub(0.5).div(0.25)
    batch = tensor.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
    return batch, original


def _load(model_path: str):
    global _MODEL, _DEVICE
    root = Path(model_path)
    config_path = root / "model_config.json"
    checkpoint_path = root / "best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint directory: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if _MODEL is not None:
        return _MODEL, _DEVICE, config
    with _LOCK:
        if _MODEL is None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("model_id") != config["model_id"]:
                raise ValueError("Checkpoint model_id does not match model_config.json")
            if checkpoint.get("architecture") != config["architecture"]:
                raise ValueError("Checkpoint architecture does not match model_config.json")
            model = _make_model(config["architecture"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _MODEL = model.to(_DEVICE).eval()
    return _MODEL, _DEVICE, config


def _gradcam_layer(model: torch.nn.Module, architecture: str):
    if architecture == "densenet121":
        return model.features.denseblock4, "features.denseblock4"
    if architecture in {"resnet34", "resnet50"}:
        return model.layer4[-1], "layer4[-1]"
    if architecture == "efficientnet_b0":
        return model.features[-1], "features[-1]"
    raise ValueError(f"Unsupported Grad-CAM architecture: {architecture}")


def _gradcam(model, batch, original, target_layer, threshold):
    activations = []
    gradients = []

    def capture(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = target_layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(batch).flatten()
        logit = logits[0]
        probability = float(torch.sigmoid(logit.detach()).item())
        positive = probability >= threshold
        score = logit if positive else -logit
        score.backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture tensors")
        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))[0]
        minimum = float(cam.min().item())
        maximum = float(cam.max().item())
        available = maximum > minimum
        if available:
            cam = (cam - minimum) / (maximum - minimum)
            heat = cv2.resize(cam.cpu().numpy(), (original.shape[1], original.shape[0]))
            heat = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            base = np.stack([original, original, original], axis=-1)
            overlay = cv2.addWeighted(base, 0.65, heat, 0.35, 0)
        else:
            overlay = np.stack([original, original, original], axis=-1)
        return probability, overlay.astype(np.uint8), available, positive
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def main(input_data, model_path: str):
    """Return a Grad-CAM RGB overlay and one binary classification result."""
    batch, original = _preprocess(input_data)
    model, device, config = _load(model_path)
    target_layer, target_layer_name = _gradcam_layer(model, config["architecture"])
    threshold = float(config["threshold"])
    with _INFERENCE_LOCK, torch.enable_grad():
        probability, overlay, gradcam_available, positive = _gradcam(
            model, batch.to(device), original, target_layer, threshold
        )
    predicted_name = config["positive_name"] if positive else config["negative_name"]
    prediction = {
        "pred": int(positive),
        "pred_name": predicted_name,
        "label": config["target"],
        "probability": probability,
        "threshold": threshold,
        "model_id": config["model_id"],
        "architecture": config["architecture"],
        "test_auroc": float(config["test_auroc"]),
        "test_auprc": float(config["test_auprc"]),
        "test_f1": float(config["test_f1"]),
        "gradcam_target": predicted_name,
        "gradcam_layer": target_layer_name,
        "gradcam_score": "positive_logit" if positive else "negative_logit",
        "gradcam_available": bool(gradcam_available),
        "gradcam_interpretation": "regions supporting the thresholded predicted class",
        "image_roles": ["gradcam_overlay"],
    }
    return overlay, [prediction]

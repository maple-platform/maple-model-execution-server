#!/usr/bin/env python3
"""Maple single-image inference entry point for NIH-TorchXRayVision.

This module intentionally contains only pure model inference code. It does not
start a FastAPI server and does not depend on a Docker runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - deployment dependency path
    cv2 = None

import torch
import torch.nn as nn

_TORCHVISION = None
_XRV = None


MODEL_NAME = "TorchXRayVision resnet50-res512-all"
MODEL_WEIGHTS = "resnet50-res512-all"
NIH_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

# Preliminary thresholds selected on NIH ChestX-ray14 by maximizing Youden index.
# These are operating thresholds for this project analysis only and require
# validation-set recalibration before clinical use.
THRESHOLDS = {
    "Atelectasis": 0.11,
    "Cardiomegaly": 0.01,
    "Effusion": 0.14,
    "Infiltration": 0.23,
    "Mass": 0.05,
    "Nodule": 0.06,
    "Pneumonia": 0.06,
    "Pneumothorax": 0.08,
    "Consolidation": 0.07,
    "Edema": 0.05,
    "Emphysema": 0.03,
    "Fibrosis": 0.01,
    "Pleural_Thickening": 0.04,
    "Hernia": 0.01,
}
THRESHOLD_METHOD = "label-wise Youden index from NIH ChestX-ray14 preliminary analysis"

_MODEL_CACHE: dict[str, Any] = {
    "model": None,
    "device": None,
    "label_indices": None,
    "target_layer": None,
}


def _get_torchvision():
    global _TORCHVISION
    if _TORCHVISION is None:
        try:
            import torchvision
        except Exception as exc:  # pragma: no cover - deployment dependency path
            raise ImportError(f"Failed to import torchvision: {exc}") from exc
        _TORCHVISION = torchvision
    return _TORCHVISION


def _get_xrv():
    global _XRV
    if _XRV is None:
        try:
            import torchxrayvision as xrv
        except Exception as exc:  # pragma: no cover - deployment dependency path
            raise ImportError(f"Failed to import torchxrayvision: {exc}") from exc
        _XRV = xrv
    return _XRV


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv


def load_model(model_path: str | None = None) -> tuple[nn.Module, torch.device, list[int], nn.Module]:
    """Load the TorchXRayVision pretrained model once and reuse it globally.

    ``model_path`` is accepted for the Maple interface. This model uses
    TorchXRayVision pretrained weights, so ``None`` or an empty string is valid.
    """
    del model_path  # The pretrained TorchXRayVision weight name is fixed.

    if _MODEL_CACHE["model"] is not None:
        return (
            _MODEL_CACHE["model"],
            _MODEL_CACHE["device"],
            _MODEL_CACHE["label_indices"],
            _MODEL_CACHE["target_layer"],
        )

    xrv = _get_xrv()
    device = _select_device()
    model = xrv.models.ResNet(weights=MODEL_WEIGHTS)
    model = model.to(device)
    model.eval()

    pathologies = list(getattr(model, "pathologies", []))
    missing_labels = [label for label in NIH_LABELS if label not in pathologies]
    if missing_labels:
        raise RuntimeError(f"Model pathologies do not include NIH labels: {missing_labels}")
    label_indices = [pathologies.index(label) for label in NIH_LABELS]
    target_layer = _find_last_conv_layer(model)

    _MODEL_CACHE.update(
        {
            "model": model,
            "device": device,
            "label_indices": label_indices,
            "target_layer": target_layer,
        }
    )
    return model, device, label_indices, target_layer


def _load_grayscale_array(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            image = image.convert("L")
            return np.asarray(image, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to load image {image_path}: {exc}") from exc


def _to_rgb_uint8(image_array: np.ndarray) -> np.ndarray:
    image = image_array.astype(np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.stack([image_uint8, image_uint8, image_uint8], axis=-1)


def preprocess_image(input_data: str | Path) -> tuple[torch.Tensor, np.ndarray]:
    """Convert a PNG/JPG CXR into the TorchXRayVision 512x512 tensor format."""
    image_path = Path(input_data)
    original_gray = _load_grayscale_array(image_path)

    xrv = _get_xrv()
    torchvision = _get_torchvision()
    img = xrv.datasets.normalize(original_gray, 255)
    img = img[None, :, :]
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)]
    )
    img = transform(img).astype(np.float32)
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    return tensor, original_gray


def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    label_indices: list[int],
) -> dict[str, float]:
    """Run model inference and return NIH label probabilities."""
    image_tensor = image_tensor.to(device)
    with torch.inference_mode():
        outputs = model(image_tensor)
        if isinstance(outputs, dict):
            outputs = outputs.get("out", next(iter(outputs.values())))
        probs = outputs[:, label_indices].detach().cpu().numpy()[0]
    return {label: float(probs[idx]) for idx, label in enumerate(NIH_LABELS)}


def make_predictions(probs: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-label predictions and probability-ranked top-3 findings."""
    predictions: list[dict[str, Any]] = []
    for label in NIH_LABELS:
        prob = float(probs[label])
        threshold = float(THRESHOLDS[label])
        pred = int(prob >= threshold)
        predictions.append(
            {
                "label": label,
                "prob": prob,
                "threshold": threshold,
                "pred": pred,
                "pred_name": "positive" if pred else "negative",
            }
        )

    ranked = sorted(predictions, key=lambda item: item["prob"], reverse=True)
    top_findings = [
        {
            "rank": rank,
            "label": item["label"],
            "prob": item["prob"],
            "threshold": item["threshold"],
            "pred": item["pred"],
        }
        for rank, item in enumerate(ranked[:3], start=1)
    ]
    return predictions, top_findings


def overlay_heatmap(original_gray: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the original CXR and return RGB uint8."""
    if cv2 is None:
        raise ImportError("opencv-python-headless is required for Grad-CAM overlay.")

    base_rgb = _to_rgb_uint8(original_gray)
    heatmap = np.clip(cam, 0, 1)
    heatmap = cv2.resize(heatmap, (base_rgb.shape[1], base_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap_color, alpha, 0)
    return blended.astype(np.uint8)


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_gray: np.ndarray,
    target_label: str,
    device: torch.device,
    label_indices: list[int],
    target_layer: nn.Module,
) -> np.ndarray:
    """Generate a hook-based Grad-CAM overlay for the target label."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def backward_hook(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        x = image_tensor.to(device)
        outputs = model(x)
        if isinstance(outputs, dict):
            outputs = outputs.get("out", next(iter(outputs.values())))
        target_index = NIH_LABELS.index(target_label)
        model_output_index = label_indices[target_index]
        score = outputs[0, model_output_index]
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation, dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max <= cam_min:
            raise RuntimeError("Grad-CAM map is empty after normalization.")
        cam = (cam - cam_min) / (cam_max - cam_min)
        return overlay_heatmap(original_gray, cam.detach().cpu().numpy())
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


def main(input_data: str, model_path: str | None = None) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Maple inference entry point.

    Returns ``(result_image, predictions, model_output)``. If a Maple backend only
    accepts two return values, this module can be adapted to return the fallback
    ``(result_image, predictions)`` while keeping ``model_output`` as metadata.
    """
    model, device, label_indices, target_layer = load_model(model_path)
    image_tensor, original_gray = preprocess_image(input_data)
    probs = predict(model, image_tensor, device, label_indices)
    predictions, top_findings = make_predictions(probs)
    gradcam_target = top_findings[0]["label"]

    try:
        result_image = generate_gradcam(
            model=model,
            image_tensor=image_tensor,
            original_gray=original_gray,
            target_label=gradcam_target,
            device=device,
            label_indices=label_indices,
            target_layer=target_layer,
        )
    except Exception as exc:  # pragma: no cover - deployment/runtime dependent
        warnings.warn(
            f"Grad-CAM generation failed for target {gradcam_target}: {exc}. "
            "Returning original RGB image instead.",
            RuntimeWarning,
        )
        result_image = _to_rgb_uint8(original_gray)

    model_output = {
        "top_findings": top_findings,
        "gradcam_target": gradcam_target,
        "model": MODEL_NAME,
        "threshold_method": THRESHOLD_METHOD,
    }
    return result_image, predictions, model_output


def _save_rgb_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(output_path)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run Maple single-image inference.")
    parser.add_argument("--image", required=True, type=Path, help="Input PNG/JPG chest X-ray image path.")
    parser.add_argument("--output", required=True, type=Path, help="Output Grad-CAM overlay PNG path.")
    args = parser.parse_args()

    try:
        result_image, predictions, model_output = main(str(args.image), model_path=None)
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}", file=sys.stderr)
        return 1

    gradcam_target = model_output["gradcam_target"]
    out_path = args.output.with_stem(f"output_{args.image.stem}_{gradcam_target}_gc")
    _save_rgb_image(result_image, out_path)
    print("[INFO] Predictions:")
    print(json.dumps(predictions, indent=2))
    print("[INFO] Top findings:")
    print(json.dumps(model_output["top_findings"], indent=2))
    print(f"[INFO] Grad-CAM target: {gradcam_target}")
    print(f"[DONE] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

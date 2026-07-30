"""Maple inference entrypoint for RETFound APTOS2019 DR classification.

Research/non-commercial use only. RETFound is licensed CC BY-NC 4.0.
The checkpoint must be obtained from the official RETFound benchmark release.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from model_def import RETFound_mae


LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}
IMAGE_SIZE = 224
_MODEL_CACHE: dict[tuple[str, str, int], torch.nn.Module] = {}
_CACHE_LOCK = Lock()


def _resolve_checkpoint(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    preferred = [path / "checkpoint-best.pth", path / "checkpoint.pth"]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = sorted([*path.glob("*.pth"), *path.glob("*.pt")])
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .pth or .pt checkpoint found under: {path}")
    raise ValueError(f"Multiple checkpoints found; pass a file path explicitly: {candidates}")


def _safe_torch_load(path: Path) -> Any:
    """Load tensors while allowing only RETFound's argparse metadata class."""
    try:
        with torch.serialization.safe_globals([argparse.Namespace]):
            # Always map the training checkpoint to CPU. It also contains optimizer
            # state, which must never consume GPU memory during inference loading.
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint could not be loaded in PyTorch weights-only safe mode. "
            "Use the official RETFound APTOS checkpoint and PyTorch 2.5 or newer."
        ) from exc


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state_dict = payload["model"]
    elif isinstance(payload, dict) and all(isinstance(k, str) for k in payload):
        state_dict = payload
    else:
        raise ValueError("Unsupported checkpoint structure; expected a state dict or {'model': state_dict}.")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = _resolve_checkpoint(model_path)
    cache_key = (str(checkpoint), str(device), checkpoint.stat().st_mtime_ns)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        model = RETFound_mae(img_size=IMAGE_SIZE, num_classes=len(LABELS), global_pool=True)
        state_dict = _extract_state_dict(_safe_torch_load(checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"norm.weight", "norm.bias"}
        real_missing = [key for key in missing if key not in allowed_missing]
        if real_missing or unexpected:
            raise RuntimeError(
                "Checkpoint is not the expected RETFound-MAE APTOS 5-class model. "
                f"missing={real_missing}, unexpected={list(unexpected)}"
            )
        if model.head.out_features != len(LABELS):
            raise RuntimeError(f"Expected 5 output classes, got {model.head.out_features}.")

        model.to(device).eval()
        _MODEL_CACHE.clear()
        _MODEL_CACHE[cache_key] = model
        return model


def _preprocess(image_path: str) -> tuple[torch.Tensor, np.ndarray]:
    path = Path(image_path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("input_data must be a PNG/JPG/JPEG image path.")
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
    resize_size = int(IMAGE_SIZE / (224 / 256))
    view_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    )
    model_view = view_transform(rgb)
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )(model_view)
    return tensor.unsqueeze(0), np.asarray(model_view, dtype=np.uint8)


def _gradcam(model: torch.nn.Module, tensor: torch.Tensor) -> tuple[np.ndarray, torch.Tensor, int]:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = model.blocks[-1].norm1.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor)
            if logits.ndim == 3 and logits.shape[1] == 1:
                logits = logits[:, 0, :]
            if logits.shape != (1, len(LABELS)):
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            target = int(logits.argmax(dim=1).item())
            logits[0, target].backward()
    finally:
        handle.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture ViT activations/gradients.")
    activation = activations[0][:, 1:, :]
    gradient = gradients[0][:, 1:, :]
    patch_count = activation.shape[1]
    side = int(patch_count**0.5)
    if side * side != patch_count:
        raise RuntimeError(f"Patch token count is not square: {patch_count}")

    weights = gradient.mean(dim=1, keepdim=True)
    cam = torch.relu((activation * weights).sum(dim=2))[0]
    cam = cam.reshape(side, side).detach().float().cpu().numpy()
    cam -= cam.min()
    maximum = float(cam.max())
    if maximum <= 1e-12:
        raise RuntimeError("Grad-CAM is empty for this input; no misleading fallback image was produced.")
    cam /= maximum
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    return cam, logits.detach(), target


def _overlay(model_view: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap_bgr = cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(model_view, 0.55, heatmap_rgb, 0.45, 0).astype(np.uint8)


def main(input_data, model_path: str):
    """Return ``(gradcam_rgb_uint8, predictions)`` for one fundus image."""
    if not isinstance(input_data, (str, Path)):
        raise TypeError("input_data must be a PNG/JPG/JPEG image path string.")
    if not model_path:
        raise ValueError("model_path must point to the official fine-tuned APTOS checkpoint.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor, model_view = _preprocess(str(input_data))
    model = _load_model(model_path, device)
    cam, logits, target = _gradcam(model, tensor.to(device))
    probabilities = torch.softmax(logits[0], dim=0).cpu().numpy()
    predictions = [
        {
            "pred": int(index),
            "pred_name": LABELS[index],
            "prob": float(probability),
            "is_top_prediction": index == target,
        }
        for index, probability in enumerate(probabilities)
    ]
    return _overlay(model_view, cam), predictions

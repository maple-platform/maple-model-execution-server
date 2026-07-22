"""Maple inference for one CT-RATE 3D classification checkpoint."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from monai.networks.layers.factories import Act, Conv, Norm, split_args
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.transforms import Compose, CropForeground, EnsureType, LoadImage, Orientation, Resize, ScaleIntensityRange
from monai.utils import has_option
from safetensors.torch import load_file
from torch import nn
import torch.nn.functional as F


_MODEL = None
_CONFIG = None
_DEVICE = None
_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


class SegResBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, norm, kernel_size=3, act="relu"):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size) if isinstance(kernel_size, (tuple, list)) else kernel_size // 2
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act1 = get_act_layer(act)
        self.conv1 = Conv[Conv.CONV, spatial_dims](in_channels, in_channels, kernel_size, 1, padding, bias=False)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act2 = get_act_layer(act)
        self.conv2 = Conv[Conv.CONV, spatial_dims](in_channels, in_channels, kernel_size, 1, padding, bias=False)

    def forward(self, x):
        return x + self.conv2(self.act2(self.norm2(self.conv1(self.act1(self.norm1(x))))))


class SegResEncoder(nn.Module):
    def __init__(self, blocks_down=(1, 2, 2, 4, 4), init_filters=32, in_channels=1):
        super().__init__()
        spatial_dims = 3
        norm = split_args("batch")
        if has_option(Norm[norm[0], spatial_dims], "affine"):
            norm[1].setdefault("affine", True)
        act = split_args("relu")
        if has_option(Act[act[0]], "inplace"):
            act[1].setdefault("inplace", True)
        filters = init_filters
        self.conv_init = Conv[Conv.CONV, spatial_dims](in_channels, filters, 3, 1, 1, bias=False)
        self.layers = nn.ModuleList()
        for index, count in enumerate(blocks_down):
            level = nn.ModuleDict()
            level["blocks"] = nn.Sequential(*[
                SegResBlock(spatial_dims, filters, norm, 3, act) for _ in range(count)
            ])
            level["downsample"] = (
                Conv[Conv.CONV, spatial_dims](filters, 2 * filters, 3, 2, 1, bias=False)
                if index < len(blocks_down) - 1 else nn.Identity()
            )
            self.layers.append(level)
            filters *= 2

    def forward(self, x):
        outputs = []
        x = self.conv_init(x)
        for level in self.layers:
            x = level["blocks"](x)
            outputs.append(x)
            x = level["downsample"](x)
        return outputs


class Specialist(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Dropout(0.25),
            nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.15), nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class MultiTask(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Dropout(0.25),
            nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, outputs),
        )

    def forward(self, x):
        return self.net(x)


class CTModel(nn.Module):
    def __init__(self, encoder, head, mean=None, std=None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.register_buffer("feature_mean", mean if mean is not None else torch.zeros(512))
        self.register_buffer("feature_std", std if std is not None else torch.ones(512))

    def forward_features(self, x):
        z = self.encoder(x)[-1]
        pooled = F.adaptive_avg_pool3d(z, 1).flatten(1)
        return z, (pooled - self.feature_mean) / self.feature_std

    def forward(self, x):
        _, features = self.forward_features(x)
        return self.head(features)


TRANSFORM = Compose([
    LoadImage(ensure_channel_first=True),
    EnsureType(),
    Orientation(axcodes="SPL"),
    ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
    CropForeground(allow_smaller=True),
    Resize((128, 256, 256), mode="trilinear", align_corners=False),
])


def _torch_load(path, weights_only=False):
    # Checkpoints were serialized with NumPy 2; retain compatibility with the
    # NumPy 1.x runtime used by the platform image.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def _load(model_path):
    global _MODEL, _CONFIG, _DEVICE
    root = Path(model_path)
    config = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
    if _MODEL is not None:
        return _MODEL, _DEVICE, _CONFIG
    with _LOCK:
        if _MODEL is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            family = config["family"]
            checkpoint = _torch_load(root / "model.pt", weights_only=(family == "end_to_end_3d"))
            encoder = SegResEncoder()
            if family == "end_to_end_3d":
                labels = checkpoint["labels"]
                head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.2), nn.Linear(512, len(labels)))
                model = CTModel(encoder, head)
                state = {}
                for key, value in checkpoint["model"].items():
                    if key.startswith("encoder."):
                        state[key] = value
                    elif key.startswith("head."):
                        state[key] = value
                missing, unexpected = model.load_state_dict(state, strict=False)
                if set(missing) != {"feature_mean", "feature_std"} or unexpected:
                    raise RuntimeError(f"Unexpected end-to-end checkpoint keys: missing={missing}, unexpected={unexpected}")
            else:
                encoder.load_state_dict(load_file(str(root / "encoder.safetensors")), strict=True)
                mean = torch.from_numpy(np.asarray(checkpoint["mean"], dtype="float32"))
                std = torch.from_numpy(np.asarray(checkpoint["std"], dtype="float32"))
                if family == "official_18":
                    labels = [config["target"]]
                    head = Specialist()
                    head.load_state_dict(checkpoint["model_state"], strict=True)
                elif family == "multitask":
                    labels = checkpoint["metrics"]["outputs"]
                    head = MultiTask(len(labels))
                    head.load_state_dict(checkpoint["state"], strict=True)
                else:
                    raise ValueError(f"Unsupported CT-RATE family: {family}")
                model = CTModel(encoder, head, mean, std)
            config["labels"] = labels
            _MODEL = model.to(device).eval()
            _CONFIG = config
            _DEVICE = device
    return _MODEL, _DEVICE, _CONFIG


def _preprocess(input_data):
    if not isinstance(input_data, (str, Path)):
        raise TypeError("input_data must be a NIfTI path")
    path = Path(input_data)
    if not path.is_file():
        raise FileNotFoundError(f"Input CT not found: {path}")
    if not (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz")):
        raise ValueError("CT-RATE models accept .nii or .nii.gz volumes")
    return TRANSFORM(str(path)).unsqueeze(0)


def _montage(volume, cam, slices):
    originals, overlays = [], []
    for index in slices:
        base = np.uint8(np.clip(volume[index], 0, 1) * 255)
        rgb = np.repeat(base[..., None], 3, axis=2)
        heat = cv2.applyColorMap(np.uint8(np.clip(cam[index], 0, 1) * 255), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        originals.append(rgb)
        overlays.append(cv2.addWeighted(rgb, 0.65, heat, 0.35, 0))
    return np.concatenate([np.concatenate(originals, axis=1), np.concatenate(overlays, axis=1)], axis=0)


def main(input_data, model_path):
    model, device, config = _load(model_path)
    batch = _preprocess(input_data).to(device)
    with _INFERENCE_LOCK:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features, standardized = model.forward_features(batch)
            logits = model.head(standardized)[0]
        probabilities = torch.sigmoid(logits.detach().float()).cpu().numpy()
        threshold = float(config.get("threshold", 0.5))
        positive_indices = np.flatnonzero(probabilities >= threshold)
        target_index = int(positive_indices[np.argmax(probabilities[positive_indices])]) if len(positive_indices) else int(np.argmax(probabilities))
        positive = bool(probabilities[target_index] >= threshold)
        score = logits[target_index] if positive else -logits[target_index]
        features.retain_grad()
        score.backward()
        gradient = features.grad
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * features).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=batch.shape[2:], mode="trilinear", align_corners=False)[0, 0]
        cam = cam.detach().float().cpu().numpy()
    volume = batch[0, 0].detach().float().cpu().numpy()
    cam_min, cam_max = float(cam.min()), float(cam.max())
    available = cam_max > cam_min + 1e-8
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8) if available else np.zeros_like(cam)
    energy = cam.mean(axis=(1, 2))
    # Restrict displayed slices to the thoracic air-bearing range. Ranking all
    # slices by CAM energy alone frequently selects liver/abdomen even though
    # the package targets thoracic findings.
    central = volume[:, 32:-32, 32:-32]
    # CT-RATE NIfTI files commonly store HU+1024 (air ~= 0 before the
    # training transform, ~= 0.333 after it). Also support true-HU NIfTI.
    if float(volume.min()) >= 0.30:
        air_fraction = ((central > 0.325) & (central < 0.48)).mean(axis=(1, 2))
    else:
        air_fraction = ((central > 0.005) & (central < 0.30)).mean(axis=(1, 2))
    air_cutoff = max(0.02, float(np.quantile(air_fraction, 0.60)))
    thoracic = np.flatnonzero(air_fraction >= air_cutoff)
    candidates = thoracic[np.argsort(energy[thoracic])[::-1]] if len(thoracic) >= 6 else np.argsort(energy)[::-1]
    selected = []
    for index in candidates:
        if all(abs(int(index) - old) >= 6 for old in selected):
            selected.append(int(index))
        if len(selected) == 6:
            break
    if len(selected) < 6:
        selected = np.linspace(16, volume.shape[0] - 17, 6).round().astype(int).tolist()
    selected.sort()
    overlay = _montage(volume, cam, selected).astype(np.uint8)
    predictions = []
    metrics = config.get("metrics", {})
    for index, (label, probability) in enumerate(zip(config["labels"], probabilities.tolist())):
        pred = int(probability >= threshold)
        predictions.append({
            "label": label,
            "probability": float(probability),
            "threshold": threshold,
            "pred": pred,
            "pred_name": label if pred else f"No {label}",
            "model_id": config["model_id"],
            "family": config["family"],
            "validation_metrics": metrics,
            "gradcam_available": bool(available and index == target_index),
            "gradcam_target": config["labels"][target_index],
            "gradcam_score": "positive_logit" if positive else "negative_logit",
            "gradcam_layer": "encoder.layers.4.blocks",
            "gradcam_slices": selected,
            "image_roles": ["gradcam_axial_montage"],
        })
    return overlay, predictions

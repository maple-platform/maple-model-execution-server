#!/usr/bin/env python3
"""Maple inference entry point for SynthStroke stroke-lesion segmentation.

Upstream: liamchalcroft/SynthStroke (MIT) -- "Synthetic Data for Robust Stroke
Segmentation", MELBA 2025. This file is a *self-contained* re-implementation of
the upstream inference path: it rebuilds the MONAI UNet directly from the
checkpoint's config.json and loads weights from model.safetensors, so it does
NOT depend on `huggingface_hub` and runs fully offline.

Input:  T1-weighted NIfTI path (.nii.gz)  +  checkpoint folder/file path
Output: (images, labels, predictions)
  images      : list[np.ndarray (H, W, 3) uint8 RGB]  (order matches `labels`)
  labels      : list[str]  -- semantic name of each image
  predictions : list[dict] -- per-lesion connected-component stats
                {lesion_id, volume_ml, voxels, centroid_vox, bbox_vox}

The 3-tuple return extends the Manual's image-result contract. The paired
`predictions` are the "detection grounding" the platform can forward to
downstream scoring models (e.g. ICH Score) without a separate detector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MODEL_CACHE: dict[str, Any] = {}

PATCH_SIZE = 128
SW_OVERLAP = 0.5
GAUSSIAN_SIGMA_SCALE = 0.125
SW_BATCH_SIZE = 1
USE_TTA = True
VOXEL_SPACING_MM = 1.0
ML_PER_VOXEL = (VOXEL_SPACING_MM ** 3) / 1000.0   # 1 mm^3 = 0.001 mL

LESION_COLOR = np.array([230, 40, 40], dtype=np.uint8)
MIN_LESION_VOXELS = 5


# --- model construction (config.json -> MONAI UNet) + safetensors weights ---

def _resolve_checkpoint(model_path: str) -> tuple[Path, Path]:
    p = Path(model_path)
    if p.is_dir():
        weights = p / "model.safetensors"
        config = p / "config.json"
    else:
        weights = p
        config = p.parent / "config.json"
    if not weights.exists():
        raise FileNotFoundError(f"model.safetensors not found near: {model_path}")
    if not config.exists():
        raise FileNotFoundError(f"config.json not found near: {model_path}")
    return weights, config


def _build_model(config_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    from monai.networks.nets import UNet

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out_channels = int(cfg.get("out_channels", 6))
    net = UNet(
        spatial_dims=int(cfg.get("spatial_dims", 3)),
        in_channels=int(cfg.get("in_channels", 1)),
        out_channels=out_channels,
        channels=tuple(cfg.get("channels", [32, 64, 128, 256, 320, 320])),
        strides=tuple(cfg.get("strides", [2, 2, 2, 2, 2])),
        kernel_size=int(cfg.get("kernel_size", 3)),
        up_kernel_size=int(cfg.get("up_kernel_size", 3)),
        num_res_units=int(cfg.get("num_res_units", 1)),
        act=cfg.get("act", "PRELU"),
        norm=cfg.get("norm", "INSTANCE"),
        dropout=float(cfg.get("dropout", 0.0)),
        bias=bool(cfg.get("bias", True)),
        adn_ordering=cfg.get("adn_ordering", "NDA"),
    ).to(device)
    return net, out_channels


def _load_model(model_path: str, device: torch.device):
    from safetensors.torch import load_file

    weights_path, config_path = _resolve_checkpoint(model_path)
    net, out_channels = _build_model(config_path, device)

    state_dict = load_file(str(weights_path))
    stripped = {}
    for k, v in state_dict.items():
        stripped[k[len("unet."):] if k.startswith("unet.") else k] = v
    net.load_state_dict(stripped, strict=True)
    net.eval()

    stroke_class = 5 if out_channels == 6 else 1
    return net, stroke_class


def _get_model(model_path: str, device: torch.device):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = _load_model(model_path, device)
    return _MODEL_CACHE[model_path]


# --- preprocessing (mirror upstream SynthStrokeModel.preprocess_image) ---

def _preprocess(nifti_path: str, device: torch.device):
    import monai as mn

    batch = {"img": nifti_path}
    preproc = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys="img", image_only=True),
        mn.transforms.EnsureChannelFirstD(keys="img"),
        mn.transforms.OrientationD(keys="img", axcodes="RAS"),
        mn.transforms.SpacingD(keys="img", pixdim=VOXEL_SPACING_MM),
        mn.transforms.HistogramNormalizeD(keys="img"),
        mn.transforms.NormalizeIntensityD(keys="img", nonzero=False, channel_wise=True),
        mn.transforms.ToTensorD(keys="img", dtype=torch.float32),
    ])
    return preproc(batch)["img"]  # (1, H, W, D)


# --- inference (sliding-window + optional TTA) ---

_TTA_FLIPS = [
    (True, False, False), (False, True, False), (False, False, True),
    (True, True, False), (True, False, True), (False, True, True),
    (True, True, True),
]


def _sliding_window(model: torch.nn.Module, x: torch.Tensor):
    import monai as mn
    inferer = mn.inferers.SlidingWindowInferer(
        roi_size=[PATCH_SIZE, PATCH_SIZE, PATCH_SIZE],
        sw_batch_size=SW_BATCH_SIZE,
        overlap=SW_OVERLAP,
        mode="gaussian",
        sigma_scale=GAUSSIAN_SIGMA_SCALE,
        cval=0.0,
        progress=False,
    )
    return inferer(x, model)


def _infer(model: torch.nn.Module, img: torch.Tensor) -> np.ndarray:
    x = img.unsqueeze(0)  # (1, 1, H, W, D)
    with torch.inference_mode():
        if USE_TTA:
            running = torch.softmax(_sliding_window(model, x), dim=1)
            count = 1
            for fx, fy, fz in _TTA_FLIPS:
                dims = [d for d, f in zip((2, 3, 4), (fx, fy, fz)) if f]
                flipped = torch.flip(x, dims=dims)
                pred = torch.softmax(_sliding_window(model, flipped), dim=1)
                running = running + torch.flip(pred, dims=dims)
                count += 1
            probs = running / count
        else:
            probs = torch.softmax(_sliding_window(model, x), dim=1)
    return probs.squeeze(0).cpu().numpy()


# --- connected-component lesion stats ---

def _lesion_predictions(mask: np.ndarray) -> list[dict]:
    from scipy import ndimage as ndi

    labeled, n = ndi.label(mask)
    preds: list[dict] = []
    if n == 0:
        return preds
    for lid in range(1, n + 1):
        comp = labeled == lid
        vox = int(comp.sum())
        if vox < MIN_LESION_VOXELS:
            continue
        coords = np.argwhere(comp)
        centroid = coords.mean(axis=0)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        preds.append({
            "lesion_id": 0,
            "volume_ml": round(vox * ML_PER_VOXEL, 3),
            "voxels": vox,
            "centroid_vox": [int(round(c)) for c in centroid],
            "bbox_vox": [int(mins[0]), int(mins[1]), int(mins[2]),
                          int(maxs[0]), int(maxs[1]), int(maxs[2])],
        })
    preds.sort(key=lambda d: d["volume_ml"], reverse=True)
    for i, d in enumerate(preds, 1):
        d["lesion_id"] = i
    return preds


# --- visualization ---

def _normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
    s = slice_2d.astype(np.float32)
    nz = s[s != 0]
    if len(nz) == 0:
        return np.zeros_like(s)
    lo, hi = np.percentile(nz, [1, 99])
    if hi <= lo:
        lo, hi = float(nz.min()), float(nz.max())
    if hi <= lo:
        return np.zeros_like(s)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def _best_axial_slice(mask_3d: np.ndarray) -> int:
    scores = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    if scores.max() == 0:
        return mask_3d.shape[0] // 2
    return int(scores.argmax())


def _overlay(volume: np.ndarray, mask: np.ndarray, idx: int, alpha: float = 0.45) -> np.ndarray:
    base = _normalize_for_display(volume[idx])
    rgb = (np.stack([base] * 3, axis=-1) * 255).clip(0, 255).astype(np.uint8)
    m = mask[idx]
    if m.any():
        rgb[m] = ((1 - alpha) * rgb[m].astype(np.float32)
                  + alpha * LESION_COLOR.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return np.rot90(rgb).copy()


def _render_3d(volume: np.ndarray, mask: np.ndarray):
    try:
        import os
        import tempfile

        import vedo
        from PIL import Image as PILImage
        from skimage import measure
        from skimage.measure import block_reduce

        meshes = []
        nz = volume[volume != 0]
        if len(nz):
            vol_ds = block_reduce(volume, (3, 3, 3), np.mean)
            iso = float(np.percentile(vol_ds[vol_ds != 0], 15))
            padded = np.pad(vol_ds, 1, constant_values=volume.min() - 1)
            v, f, _, _ = measure.marching_cubes(padded, level=iso)
            v = (v - 1) * 3
            brain = vedo.Mesh([v, f]).smooth(niter=5, pass_band=0.2)
            brain.color((160, 160, 170)).alpha(0.13).lighting("ambient")
            meshes.append(brain)

        if mask.any():
            padded = np.pad(mask.astype(np.float32), 1, constant_values=0)
            v, f, _, _ = measure.marching_cubes(padded, level=0.5)
            v = v - 1
            lesion = vedo.Mesh([v, f]).smooth(niter=20, pass_band=0.1)
            lesion.color((230, 40, 40)).alpha(0.9).lighting("glossy")
            meshes.append(lesion)

        if len(meshes) < 2:
            return None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp = tf.name
        plt = vedo.Plotter(offscreen=True, size=(1000, 1000), bg=(12, 12, 20), bg2=(20, 25, 40))
        plt.show(*meshes, axes=9, interactive=False, resetcam=True)
        plt.camera.Azimuth(45)
        plt.camera.Elevation(20)
        plt.render()
        plt.screenshot(tmp)
        plt.close()
        img = np.array(PILImage.open(tmp).convert("RGB"))
        os.unlink(tmp)
        return img
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] 3D rendering skipped: {exc}")
        return None


# --- public API ---

def main(input_data: str, model_path: str):
    nifti_path = Path(input_data)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI not found: {nifti_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stroke_class = _get_model(model_path, device)

    img = _preprocess(str(nifti_path), device).to(device)     # (1, H, W, D)
    probs = _infer(model, img)                                 # (C, H, W, D)
    pred = probs.argmax(axis=0)                                # (H, W, D)
    mask = pred == stroke_class

    volume = np.asarray(img[0].detach().cpu().numpy(), dtype=np.float32)

    volume = np.transpose(volume, (2, 0, 1))   # (D, H, W)
    mask = np.transpose(mask, (2, 0, 1))

    predictions = _lesion_predictions(mask)

    images: list[np.ndarray] = []
    labels: list[str] = []

    render = _render_3d(volume, mask)
    if render is not None:
        images.append(render)
        labels.append("3d_overlay")

    idx = _best_axial_slice(mask)
    images.append(_overlay(volume, mask, idx))
    labels.append("lesion_overlay")

    return images, labels, predictions


if __name__ == "__main__":
    import argparse

    from PIL import Image

    ap = argparse.ArgumentParser(description="SynthStroke stroke-lesion segmentation")
    ap.add_argument("--nifti", required=True, type=Path, help="Input T1 NIfTI (.nii.gz)")
    ap.add_argument("--weights", required=True, type=Path,
                    help="checkpoint folder or model.safetensors (config.json alongside)")
    ap.add_argument("--out_dir", default="results", type=Path, help="Output directory")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images, labels, predictions = main(str(args.nifti), str(args.weights))

    stem = args.nifti.stem.replace(".nii", "")
    for label, im in zip(labels, images):
        out = args.out_dir / f"{stem}_{label}.png"
        Image.fromarray(im).save(out)
        print(f"Saved: {out}")
    print(f"Lesions detected: {len(predictions)}")
    for p in predictions:
        print(f"  #{p['lesion_id']}: {p['volume_ml']} mL  centroid={p['centroid_vox']}")

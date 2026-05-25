#!/usr/bin/env python3
"""Maple inference entry point for BraTS Brain Tumor Segmentation.

Supports all four modalities (FLAIR, T1, T1ce, T2) — model architecture is
identical across modalities; only the checkpoint differs.

Input:  NIfTI file path (.nii.gz)  +  model checkpoint path (.pt)
Output: list[np.ndarray (H, W, 3) uint8 RGB]
  [0] WT (Whole Tumor)     — red overlay
  [1] TC (Tumor Core)      — green overlay
  [2] ET (Enhancing Tumor) — blue overlay
  Each image shows the axial slice with the most segmentation content.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# src/ 경로를 sys.path에 추가 (workspace 내 공용 모듈 사용)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.unet3d import build_unet3d

_MODEL_CACHE: dict[str, Any] = {}

THRESHOLD = 0.5
REGION_COLORS = {
    "WT": np.array([220, 50, 50],   dtype=np.uint8),
    "TC": np.array([50, 200, 50],   dtype=np.uint8),
    "ET": np.array([50, 120, 255],  dtype=np.uint8),
}
REGION_NAMES = ["WT", "TC", "ET"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _zscore_normalize(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    mask = volume != 0
    if not mask.any():
        return volume
    vals = volume[mask]
    mean, std = float(vals.mean()), float(vals.std())
    if std < 1e-6:
        std = 1.0
    out = volume.copy()
    out[mask] = (vals - mean) / std
    return out


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int, int]]:
    d, h, w = tensor.shape[-3:]
    pad_d = (-d % multiple)
    pad_h = (-h % multiple)
    pad_w = (-w % multiple)
    orig_shape = (d, h, w)
    if pad_d == pad_h == pad_w == 0:
        return tensor, orig_shape
    return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d)), orig_shape


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_unet3d(in_channels=1, out_channels=3).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _get_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    key = checkpoint_path
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _load_model(checkpoint_path, device)
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
    """Normalize using non-zero (brain tissue) pixels only to avoid dark background bias."""
    s = slice_2d.astype(np.float32)
    nonzero = s[s != 0]
    if len(nonzero) == 0:
        return np.zeros_like(s)
    lo, hi = np.percentile(nonzero, [1, 99])
    if hi <= lo:
        lo, hi = float(nonzero.min()), float(nonzero.max())
    if hi <= lo:
        return np.zeros_like(s)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _best_axial_slice(mask_3d: np.ndarray, brain_mask: np.ndarray) -> int:
    """Pick the axial slice where mask overlaps most with actual brain tissue."""
    valid = mask_3d & brain_mask
    scores = valid.reshape(valid.shape[0], -1).sum(axis=1)
    if scores.max() == 0:
        # no overlap with brain — fall back to slice with most brain tissue
        brain_scores = brain_mask.reshape(brain_mask.shape[0], -1).sum(axis=1)
        return int(brain_scores.argmax()) if brain_scores.max() > 0 else mask_3d.shape[0] // 2
    return int(scores.argmax())


def _make_all_region_overlay(
    volume: np.ndarray,   # (D, H, W)
    masks: np.ndarray,    # (3, D, H, W) bool  — WT, TC, ET
    alpha: float = 0.45,
) -> np.ndarray:
    """Single axial slice with all 3 regions blended together."""
    brain_mask = volume != 0
    combined = masks.any(axis=0)
    idx = _best_axial_slice(combined, brain_mask)

    base = _normalize_for_display(volume[idx])
    rgb = np.stack([base, base, base], axis=-1)
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    for i, (name, color) in enumerate(REGION_COLORS.items()):
        m = masks[i, idx]
        if m.any():
            rgb[m] = (
                (1.0 - alpha) * rgb[m].astype(np.float32)
                + alpha * color.astype(np.float32)
            ).clip(0, 255).astype(np.uint8)

    return np.rot90(rgb).copy()


def _make_region_overlay(
    volume: np.ndarray,   # (D, H, W)
    masks: np.ndarray,    # (3, D, H, W) bool
    region_idx: int,
    alpha: float = 0.45,
) -> np.ndarray:
    """Single-region axial overlay at the most representative slice for that region."""
    brain_mask = volume != 0
    idx = _best_axial_slice(masks[region_idx], brain_mask)

    base = _normalize_for_display(volume[idx])
    rgb = np.stack([base, base, base], axis=-1)
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    m = masks[region_idx, idx]
    if m.any():
        color = REGION_COLORS[REGION_NAMES[region_idx]]
        rgb[m] = (
            (1.0 - alpha) * rgb[m].astype(np.float32)
            + alpha * color.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

    return np.rot90(rgb).copy()


def _keep_largest_component(mask: np.ndarray, n: int = 1) -> np.ndarray:
    """Keep only the N largest connected components — removes scattered noise."""
    from scipy import ndimage as ndi
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask
    sizes = ndi.sum(mask, labeled, range(1, num + 1))
    top_labels = np.argsort(sizes)[::-1][:n] + 1
    return np.isin(labeled, top_labels)


def _build_brain_mesh(volume: np.ndarray):
    """Build a semi-transparent brain surface mesh from MRI intensity isosurface."""
    from skimage import measure
    from skimage.measure import block_reduce
    from scipy import ndimage as ndi
    import vedo

    nonzero = volume[volume != 0]
    if len(nonzero) == 0:
        return None

    # 3x downsample — preserves gyral detail while keeping mesh size manageable
    vol_ds = block_reduce(volume, (3, 3, 3), np.mean)
    iso_ds = float(np.percentile(vol_ds[vol_ds != 0], 15))

    padded = np.pad(vol_ds, 1, constant_values=volume.min() - 1)
    verts, faces, _, _ = measure.marching_cubes(padded, level=iso_ds)
    verts = (verts - 1) * 3  # padding offset + coord restore

    m = vedo.Mesh([verts, faces])
    m.smooth(niter=5, pass_band=0.2)
    m.color((160, 160, 170)).alpha(0.15).lighting("ambient")
    return m


def _render_3d(masks: np.ndarray, volume: np.ndarray | None = None) -> np.ndarray | None:
    """Render combined WT/TC/ET isosurface + brain surface."""
    try:
        import tempfile, os
        from skimage import measure
        import vedo
        from PIL import Image as PILImage

        KEEP_N = {"WT": 3, "TC": 2, "ET": 2}
        COLORS = {"WT": (220, 80, 80), "TC": (60, 200, 80), "ET": (60, 130, 255)}
        ALPHAS = {"WT": 0.25, "TC": 0.70, "ET": 0.92}

        meshes = []

        if volume is not None:
            brain_mesh = _build_brain_mesh(volume)
            if brain_mesh is not None:
                meshes.append(brain_mesh)

        brain_region = (volume != 0) if volume is not None else None

        for i, region in enumerate(REGION_NAMES):
            raw = masks[i].astype(bool)
            if brain_region is not None:
                raw = raw & brain_region
            mask = _keep_largest_component(raw, n=KEEP_N[region])
            if not mask.any():
                continue
            padded = np.pad(mask.astype(np.float32), 1, constant_values=0)
            verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
            verts = verts - 1
            m = vedo.Mesh([verts, faces])
            m.smooth(niter=30, pass_band=0.05)
            m.color(COLORS[region]).alpha(ALPHAS[region]).lighting("glossy")
            meshes.append(m)

        if len(meshes) == 0 or (volume is not None and len(meshes) == 1):
            return None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        plt = vedo.Plotter(offscreen=True, size=(1000, 1000), bg=(12, 12, 20), bg2=(20, 25, 40))
        plt.show(*meshes, axes=9, interactive=False, resetcam=True)
        plt.camera.Azimuth(45)
        plt.camera.Elevation(20)
        plt.render()
        plt.screenshot(tmp_path)
        plt.close()

        img = np.array(PILImage.open(tmp_path).convert("RGB"))
        os.unlink(tmp_path)
        return img

    except Exception as exc:
        print(f"[WARNING] 3D rendering failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def main(input_data: str, model_path: str) -> list[np.ndarray]:
    """Maple inference entry point.

    Args:
        input_data: absolute path to a NIfTI file (.nii.gz)
        model_path: absolute path to best.pt checkpoint

    Returns:
        [0] 3D combined rendering (WT/TC/ET isosurface) — RGB uint8, None 대신 빈 배열 반환 시 생략
        [1] Axial overlay — all 3 regions on the most tumor-dense slice
        [2] WT axial overlay (red)
        [3] TC axial overlay (green)
        [4] ET axial overlay (blue)
    """
    import nibabel as nib

    nifti_path = Path(input_data)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI not found: {nifti_path}")

    # Load & preprocess
    volume = nib.load(str(nifti_path)).get_fdata(dtype=np.float32)   # (H, W, D)
    volume = np.transpose(volume, (2, 0, 1))                          # → (D, H, W)
    volume = _zscore_normalize(volume)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_model(model_path, device)

    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
    padded, orig_shape = _pad_to_multiple(tensor)

    with torch.no_grad():
        logits = model(padded)

    d, h, w = orig_shape
    logits = logits[..., :d, :h, :w]
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()             # (3, D, H, W)
    masks = probs >= THRESHOLD                                          # (3, D, H, W) bool

    results: list[np.ndarray] = []

    # [0] 3D isosurface rendering (뇌 표면 컨텍스트 포함)
    rendering_3d = _render_3d(masks, volume=volume)
    if rendering_3d is not None:
        results.append(rendering_3d)

    # [1] all-region axial overlay
    results.append(_make_all_region_overlay(volume, masks))

    # [2~4] per-region axial overlays
    for i in range(3):
        results.append(_make_region_overlay(volume, masks, i))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description="BraTS brain tumor segmentation inference")
    parser.add_argument("--nifti",    required=True, type=Path, help="Input NIfTI file (.nii.gz)")
    parser.add_argument("--weights",  required=True, type=Path, help="Checkpoint path (best.pt)")
    parser.add_argument("--out_dir",  default="results", type=Path, help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = main(str(args.nifti), str(args.weights))

    stem = args.nifti.stem.replace(".nii", "")
    labels = ["3D", "all_regions", "WT", "TC", "ET"]
    for label, img in zip(labels, results):
        out_path = args.out_dir / f"{stem}_{label}.png"
        Image.fromarray(img).save(out_path)
        print(f"Saved: {out_path}")

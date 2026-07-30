"""LungMask R231 bilateral lung segmentation for NIfTI CT."""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from lungmask import LMInferer


_MODEL = None
_LOCK = threading.Lock()
_RUN = threading.Lock()


def _load(model_path):
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                _MODEL = LMInferer(
                    modelname="R231",
                    modelpath=str(Path(model_path) / "unet_r231-d5d2fc3d.pth"),
                    force_cpu=not torch.cuda.is_available(),
                    batch_size=32,
                    tqdm_disable=True,
                )
    return _MODEL


def _overlay(image, mask):
    base = np.uint8(np.clip((image + 1000) / 1400, 0, 1) * 255)
    rgb = np.repeat(base[..., None], 3, axis=2)
    colors = {1: np.array([255, 80, 50]), 2: np.array([50, 180, 255])}
    for label, color in colors.items():
        selected = mask == label
        rgb[selected] = (rgb[selected] * 0.5 + color * 0.5).astype("uint8")
    return cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_NEAREST)


def _representative_slices(mask, count=8):
    """Select stable apex-to-base previews using cumulative mask volume."""
    area = (mask > 0).sum(axis=(1, 2))
    positive = np.flatnonzero(area > 0)
    if not len(positive):
        return []
    eligible = np.flatnonzero(area >= area.max() * 0.05)
    if not len(eligible):
        eligible = positive
    if len(eligible) <= count:
        return eligible.astype(int).tolist()

    cumulative = np.cumsum(area[eligible], dtype=np.float64)
    targets = np.array([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]) * cumulative[-1]
    selected = {int(eligible[min(np.searchsorted(cumulative, target), len(eligible) - 1)])
                for target in targets}
    selected.add(int(np.argmax(area)))

    # Quantiles can collide for unusually short masks. Fill deterministically.
    for index in np.linspace(0, len(eligible) - 1, count).round().astype(int):
        selected.add(int(eligible[index]))
        if len(selected) >= count:
            break
    if len(selected) < count:
        for index in eligible[np.argsort(area[eligible])[::-1]]:
            selected.add(int(index))
            if len(selected) >= count:
                break
    return sorted(selected)[:count]


def _remove_unilateral_out_of_lung_slices(mask):
    """Suppress inferior/superior false positives outside bilateral lung support."""
    areas = [(mask == label).sum(axis=(1, 2)) for label in (1, 2)]
    supported = [(area >= area.max() * 0.01) if area.max() else np.zeros_like(area, bool)
                 for area in areas]
    bilateral = np.flatnonzero(supported[0] & supported[1])
    if not len(bilateral):
        return mask
    cleaned = mask.copy()
    cleaned[:bilateral[0]] = 0
    cleaned[bilateral[-1] + 1:] = 0
    return cleaned


def main(input_data, model_path):
    """Return eight preview overlays, the full 3D mask, and measurements."""
    path = Path(input_data)
    if not path.is_file() or not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        raise ValueError("Expected .nii or .nii.gz CT")

    image = sitk.ReadImage(str(path))
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    offset_applied = stats.GetMinimum() >= 0 and stats.GetMaximum() > 3000
    hu = (
        sitk.Cast(image, sitk.sitkFloat32) - 1024.0
        if offset_applied
        else sitk.Cast(image, sitk.sitkFloat32)
    )
    with _RUN:
        mask = _load(model_path).apply(hu)
    mask = _remove_unilateral_out_of_lung_slices(mask)

    volume = sitk.GetArrayFromImage(hu)
    mask_slices = np.flatnonzero((mask > 0).any(axis=(1, 2))).astype(int).tolist()
    selected = _representative_slices(mask, count=8)
    overlays = [_overlay(volume[z], mask[z]) for z in selected]
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_image.CopyInformation(image)

    voxel_ml = float(np.prod(image.GetSpacing()) / 1000.0)
    labels = {1: "lung_label_1", 2: "lung_label_2"}
    total_header_volume = float((mask > 0).sum()) * voxel_ml
    volume_plausible = 500.0 <= total_header_volume <= 12000.0
    measurements = []
    for label, label_name in labels.items():
        voxels = int((mask == label).sum())
        header_volume = voxels * voxel_ml
        measurements.append({
            "label": label_name,
            "present": bool(voxels),
            "voxel_count": voxels,
            "volume_ml_from_header": header_volume if volume_plausible else None,
            "volume_quality": "header_spacing_plausible" if volume_plausible else "rejected_implausible_header_spacing",
            "header_spacing_mm": list(image.GetSpacing()),
            "quantification_warning": "Header-derived volume is suppressed when total segmented volume is implausible; verify physical NIfTI spacing before quantitative use.",
            "model_id": "lungmask_r231",
            "task": "segmentation",
            "postprocessing": "bilateral_slice_support_crop",
            "input_intensity_offset_applied": bool(offset_applied),
            "segmentation_slices": selected,
            "preview_slices": selected,
            "returned_slice_count": len(selected),
            "mask_positive_slice_count": len(mask_slices),
            "mask_slice_range": [mask_slices[0], mask_slices[-1]] if mask_slices else None,
            "image_roles": ["segmentation_overlay_slice"],
        })
    return overlays, mask_image, measurements

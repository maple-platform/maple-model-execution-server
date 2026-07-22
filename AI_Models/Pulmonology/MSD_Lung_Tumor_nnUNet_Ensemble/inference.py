"""Maple NIfTI inference for the performance-gated MSD lung-tumor nnU-Net."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


_PREDICTOR = None
_LOCK = threading.Lock()
_RUN = threading.Lock()


def _load(model_path):
    global _PREDICTOR
    if _PREDICTOR is None:
        with _LOCK:
            if _PREDICTOR is None:
                root = Path(model_path)
                config = json.loads((root / "model_config.json").read_text())
                predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=True,
                    perform_everything_on_device=torch.cuda.is_available(),
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=False,
                )
                predictor.initialize_from_trained_model_folder(
                    str(root),
                    use_folds=tuple(config["folds"]),
                    checkpoint_name="checkpoint_best.pth",
                )
                _PREDICTOR = predictor
    return _PREDICTOR


def _overlay(image, mask):
    lo, hi = -1000.0, 400.0
    base = np.uint8(np.clip((image - lo) / (hi - lo), 0, 1) * 255)
    rgb = np.repeat(base[..., None], 3, axis=2)
    selected = mask > 0
    rgb[selected] = (rgb[selected] * 0.45 + np.array([255, 50, 30]) * 0.55).astype("uint8")
    return cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_NEAREST)


def _representative_slices(mask, count=8):
    """Preserve tumor extent and select volume-representative previews."""
    area = (mask > 0).sum(axis=(1, 2))
    positive = np.flatnonzero(area > 0)
    if len(positive) <= count:
        return positive.astype(int).tolist()

    cumulative = np.cumsum(area[positive], dtype=np.float64)
    targets = np.array([0.10, 0.25, 0.50, 0.75, 0.90]) * cumulative[-1]
    selected = {int(positive[0]), int(positive[-1]), int(np.argmax(area))}
    selected.update(
        int(positive[min(np.searchsorted(cumulative, target), len(positive) - 1)])
        for target in targets
    )
    for index in np.linspace(0, len(positive) - 1, count).round().astype(int):
        selected.add(int(positive[index]))
        if len(selected) >= count:
            break
    if len(selected) < count:
        for index in positive[np.argsort(area[positive])[::-1]]:
            selected.add(int(index))
            if len(selected) >= count:
                break
    return sorted(selected)[:count]


def main(input_data, model_path):
    """Return representative overlays, the full 3D tumor mask, and measurements."""
    path = Path(input_data)
    if not path.is_file() or not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        raise ValueError("Expected .nii or .nii.gz thoracic CT")

    input_image = sitk.ReadImage(str(path))
    stats = sitk.StatisticsImageFilter()
    stats.Execute(input_image)
    offset_applied = stats.GetMinimum() >= 0 and stats.GetMaximum() > 3000
    hu_image = (
        sitk.Cast(input_image, sitk.sitkFloat32) - 1024.0
        if offset_applied
        else sitk.Cast(input_image, sitk.sitkFloat32)
    )

    with tempfile.TemporaryDirectory(prefix="maple_msd_lung_") as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / "case_0000.nii.gz"
        if offset_applied:
            sitk.WriteImage(hu_image, str(source))
        else:
            try:
                os.link(path, source)
            except OSError:
                shutil.copy2(path, source)
        output = tmp / "seg"
        with _RUN:
            _load(model_path).predict_from_files(
                [[str(source)]],
                [str(output)],
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
        segmentation = sitk.ReadImage(str(output) + ".nii.gz")
        mask = sitk.GetArrayFromImage(segmentation)

    volume = sitk.GetArrayFromImage(hu_image)
    mask_slices = np.flatnonzero((mask > 0).any(axis=(1, 2))).astype(int).tolist()
    selected = _representative_slices(mask, count=8)
    overlays = [_overlay(volume[z], mask[z]) for z in selected]
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_image.CopyInformation(input_image)

    voxels = int((mask > 0).sum())
    header_volume = voxels * float(np.prod(input_image.GetSpacing()) / 1000.0)
    plausible = 0.0 <= header_volume <= 5000.0
    result = [{
        "label": "lung_tumor",
        "present": bool(voxels),
        "voxel_count": voxels,
        "volume_ml_from_header": header_volume if plausible else None,
        "volume_quality": "header_spacing_plausible" if plausible else "rejected_implausible_header_spacing",
        "header_spacing_mm": list(input_image.GetSpacing()),
        "input_intensity_offset_applied": bool(offset_applied),
        "model_id": "msd_lung_nnunet_5fold",
        "task": "segmentation",
        "segmentation_slices": selected,
        "preview_slices": selected,
        "returned_slice_count": len(selected),
        "mask_positive_slice_count": len(mask_slices),
        "mask_slice_range": [mask_slices[0], mask_slices[-1]] if mask_slices else None,
        "interpretation_warning": "Research segmentation; requires external clinical validation and verified physical spacing.",
        "image_roles": ["segmentation_overlay_slice"],
    }]
    return overlays, mask_image, result

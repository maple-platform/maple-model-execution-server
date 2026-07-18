from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage


VERTEBRAE = [
    *[f"vertebrae_C{i}" for i in range(1, 8)],
    *[f"vertebrae_T{i}" for i in range(1, 13)],
    *[f"vertebrae_L{i}" for i in range(1, 6)],
]


def _panel(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    gray = np.clip((image.astype(np.float32) + 300.0) / 1500.0, 0, 1)
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    for label in np.unique(mask):
        if label == 0:
            continue
        selected = mask == label
        color = np.asarray(
            [
                (int(label) * 47) % 205 + 40,
                (int(label) * 83) % 205 + 40,
                (int(label) * 131) % 205 + 40,
            ]
        )
        rgb[selected] = (0.45 * rgb[selected] + 0.55 * color).astype(np.uint8)
    return Image.fromarray(np.rot90(rgb))


def _vertebra_previews(image, mask, labels, label_map, output: Path):
    paths, roles = [], []
    for label in labels:
        counts = np.count_nonzero(mask == label, axis=(0, 1))
        index = int(np.argmax(counts))
        name = label_map[label].replace("vertebrae_", "")
        preview = _panel(image[:, :, index], np.where(mask[:, :, index] == label, label, 0))
        path = output / f"{name}_segmentation.png"
        preview.save(path)
        paths.append(str(path))
        roles.append(f"{name}_segmentation")
    return paths, roles


def predict(input_path: str, weights_dir: str, output_dir: str, fast: bool = False) -> dict:
    os.environ["TOTALSEG_HOME_DIR"] = str(Path(weights_dir).resolve())
    os.environ.setdefault("TOTALSEG_DISABLE_USAGE_STATS", "1")
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    source = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = source.name.replace(".nii.gz", "").replace(".nii", "")
    mask_path = output / f"{stem}_vertebrae.nii.gz"
    totalsegmentator(
        source,
        mask_path,
        ml=True,
        task="total",
        roi_subset=VERTEBRAE,
        statistics=False,
        fast=fast,
        device="gpu",
        quiet=True,
        nr_thr_saving=1,
        nr_thr_resamp=1,
    )

    source_image = nib.load(str(source))
    mask_image = nib.load(str(mask_path))
    image = np.asarray(source_image.dataobj)
    mask = np.asarray(mask_image.dataobj).astype(np.uint8)
    label_map = class_map["total"]
    labels = sorted(int(label) for label in np.unique(mask) if label > 0)
    voxel_volume_mm3 = float(np.prod(mask_image.header.get_zooms()))
    vertebrae = []
    for label in labels:
        selected = mask == label
        centroid_voxel = np.asarray(ndimage.center_of_mass(selected))
        centroid_world = nib.affines.apply_affine(mask_image.affine, centroid_voxel)
        vertebrae.append(
            {
                "label": label_map[label].replace("vertebrae_", ""),
                "mask_value": label,
                "centroid_world_mm": centroid_world.tolist(),
                "centroid_voxel": centroid_voxel.tolist(),
                "volume_mm3": float(selected.sum() * voxel_volume_mm3),
                "mean_hu": float(image[selected].mean()),
                "voxels": int(selected.sum()),
            }
        )

    preview_files, preview_roles = _vertebra_previews(
        image, mask, labels, label_map, output
    )
    result = {
        "model": "VerSe vertebrae CT / TotalSegmentator 2.15",
        "task": "vertebra_localization_identification_and_segmentation",
        "segmentation_file": str(mask_path),
        "preview_files": preview_files,
        "preview_roles": preview_roles,
        "vertebrae": vertebrae,
        "structures_present": [row["label"] for row in vertebrae],
        "fast_mode": bool(fast),
        "unsupported_variants": ["L6", "T13"],
        "warning": "Research use only; not a medical device.",
    }
    (output / f"{stem}_result.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "preview_b64"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return result


def _maple_split_input(input_data):
    if isinstance(input_data, (str, Path)):
        return str(input_data), {}
    if not isinstance(input_data, dict):
        raise TypeError("input_data must be a path or a dictionary")
    input_path = next(
        (
            input_data[key]
            for key in ("image_path", "input_path", "series_path", "volume_path")
            if input_data.get(key)
        ),
        None,
    )
    if input_path is None:
        raise ValueError(
            "input_data dictionary requires image_path, input_path, "
            "series_path, or volume_path"
        )
    params = {
        key: value for key, value in input_data.items() if not key.endswith("_path")
    }
    return str(input_path), params


def _maple_model_root(model_path):
    path = Path(model_path).expanduser().resolve()
    return path if path.is_dir() else path.parent


def _maple_checkpoint(model_path, filename):
    path = Path(model_path).expanduser().resolve()
    if path.is_file() and path.name == filename:
        return path
    candidate = (path if path.is_dir() else path.parent) / filename
    if not candidate.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {candidate}")
    return candidate


def _maple_public_value(value):
    import numpy as np

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        public = {}
        for key, item in value.items():
            if (
                key in {"ensemble_members", "fold_predictions_months", "runtime_log_tail"}
                or key == "preview"
                or key.endswith(("_preview", "_b64", "_file", "_path"))
            ):
                continue
            public[key] = _maple_public_value(item)
        return public
    if isinstance(value, (list, tuple)):
        return [_maple_public_value(item) for item in value]
    return value


def _maple_finish(result, image_paths, image_roles, summary):
    import numpy as np
    from PIL import Image

    images = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        for path in image_paths
        if path and Path(path).is_file()
    ]
    if not images:
        raise RuntimeError("Inference did not produce a display image")
    prediction = _maple_public_value(result)
    prediction.update(_maple_public_value(summary))
    prediction["image_roles"] = list(image_roles)
    display = images[0] if len(images) == 1 else images
    return display, [prediction]


def _maple_series(input_path):
    source = Path(input_path)
    return str(source.parent if source.is_file() else source)


def main(input_data, model_path: str):
    import tempfile

    input_path, params = _maple_split_input(input_data)
    with tempfile.TemporaryDirectory(prefix="maple_verse_") as output_dir:
        result = predict(
            input_path,
            str(_maple_model_root(model_path)),
            output_dir,
            bool(params.get("fast", False)),
        )
        return _maple_finish(
            result,
            result["preview_files"],
            result["preview_roles"],
            {"pred": len(result["vertebrae"]), "pred_name": "vertebrae_identified"},
        )

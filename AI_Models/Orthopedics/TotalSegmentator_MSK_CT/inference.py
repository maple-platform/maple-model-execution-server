from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MSK_ROIS = [
    "sacrum", "vertebrae_S1",
    *[f"vertebrae_L{i}" for i in range(5, 0, -1)],
    *[f"vertebrae_T{i}" for i in range(12, 0, -1)],
    *[f"vertebrae_C{i}" for i in range(7, 0, -1)],
    "humerus_left", "humerus_right", "scapula_left", "scapula_right",
    "clavicula_left", "clavicula_right", "femur_left", "femur_right",
    "hip_left", "hip_right", "spinal_cord", "skull", "sternum",
    "costal_cartilages",
    *[f"rib_{side}_{i}" for side in ("left", "right") for i in range(1, 13)],
    "gluteus_maximus_left", "gluteus_maximus_right",
    "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right",
    "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right",
]


def _label_color(label: int) -> tuple[int, int, int]:
    return (
        (label * 47) % 205 + 40,
        (label * 83) % 205 + 40,
        (label * 131) % 205 + 40,
    )


def _panel(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    gray = np.clip((image.astype(np.float32) + 300.0) / 1500.0, 0, 1)
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    labels = np.unique(mask)
    for label in labels[labels > 0]:
        selected = mask == label
        color = np.asarray(_label_color(int(label)))
        rgb[selected] = (0.45 * rgb[selected] + 0.55 * color).astype(np.uint8)
    return Image.fromarray(np.rot90(rgb))


def _group_labels_by_slice(mask, labels, maximum_group_size: int = 8):
    areas = {
        label: np.count_nonzero(mask == label, axis=(0, 1)).astype(np.float32)
        for label in labels
    }
    uncovered = set(labels)
    groups = []
    while uncovered:
        best_slice, best_candidates, best_score = 0, [], (-1, -1.0)
        for index in range(mask.shape[2]):
            candidates = []
            for label in uncovered:
                maximum = float(areas[label].max())
                if maximum > 0 and areas[label][index] >= max(1.0, maximum * 0.20):
                    candidates.append((label, float(areas[label][index] / maximum)))
            candidates.sort(key=lambda item: item[1], reverse=True)
            candidates = candidates[:maximum_group_size]
            score = (len(candidates), sum(value for _, value in candidates))
            if score > best_score:
                best_slice, best_candidates, best_score = index, candidates, score
        if not best_candidates:
            label = next(iter(uncovered))
            best_slice = int(np.argmax(areas[label]))
            best_candidates = [(label, 1.0)]
        group_labels = [label for label, _ in best_candidates]
        groups.append((best_slice, group_labels))
        uncovered.difference_update(group_labels)
    return groups


def _group_preview(image, mask, index, labels, label_map) -> Image.Image:
    selected = np.isin(mask[:, :, index], labels)
    grouped_mask = np.where(selected, mask[:, :, index], 0)
    panel = _panel(image[:, :, index], grouped_mask)
    height = 512
    panel = panel.resize((max(1, round(panel.width * height / panel.height)), height))
    layer = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        title_font = font = ImageFont.load_default()
    names = [label_map[label] for label in labels]
    text_width = max(
        [draw.textbbox((0, 0), f"Axial slice {index}", font=title_font)[2]]
        + [draw.textbbox((0, 0), name, font=font)[2] for name in names]
    )
    padding, swatch, row_height = 8, 12, 20
    legend_width = text_width + swatch + padding * 3
    legend_height = 27 + len(labels) * row_height + padding
    left = panel.width - legend_width - 10
    top = 10
    draw.rounded_rectangle(
        (left, top, left + legend_width, top + legend_height),
        radius=6,
        fill=(28, 28, 28, 178),
    )
    draw.text((left + padding, top + 6), f"Axial slice {index}", fill="white", font=title_font)
    for row, label in enumerate(labels):
        row_top = top + 29 + row * row_height
        draw.rectangle(
            (left + padding, row_top + 3, left + padding + swatch, row_top + 3 + swatch),
            fill=_label_color(label),
        )
        draw.text(
            (left + padding * 2 + swatch, row_top),
            label_map[label],
            fill="white",
            font=font,
        )
    return Image.alpha_composite(panel.convert("RGBA"), layer).convert("RGB")


def _structure_previews(image, mask, labels, label_map, output: Path):
    paths, roles, groups = [], [], []
    for number, (index, group_labels) in enumerate(_group_labels_by_slice(mask, labels), 1):
        names = [label_map[label] for label in group_labels]
        short_name = "__".join(names[:2])
        if len(names) > 2:
            short_name += f"__plus_{len(names) - 2}"
        role = f"axial_group_{number:02d}_{short_name}"
        path = output / f"{role}.png"
        _group_preview(image, mask, index, group_labels, label_map).save(path)
        paths.append(str(path))
        roles.append(role)
        groups.append({"image_role": role, "slice_index": index, "structures": names})
    return paths, roles, groups


def predict(
    input_path: str,
    weights_dir: str,
    output_dir: str,
    fast: bool = False,
) -> dict:
    os.environ["TOTALSEG_HOME_DIR"] = str(Path(weights_dir).resolve())
    os.environ.setdefault("TOTALSEG_DISABLE_USAGE_STATS", "1")
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    source = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mask_path = output / f"{source.name.replace('.nii.gz', '').replace('.nii', '')}_msk_seg.nii.gz"
    statistics_path = output / f"{source.name.replace('.nii.gz', '').replace('.nii', '')}_statistics.json"
    totalsegmentator(
        source, mask_path, ml=True, task="total", roi_subset=MSK_ROIS,
        statistics=False, fast=fast, device="gpu", quiet=True,
        nr_thr_saving=1, nr_thr_resamp=1,
    )

    source_image = nib.load(str(source))
    mask_image = nib.load(str(mask_path))
    image = np.asarray(source_image.dataobj)
    mask = np.asarray(mask_image.dataobj).astype(np.uint8)
    label_map = class_map["total"]
    labels = sorted(int(value) for value in np.unique(mask) if value > 0)
    preview_files, preview_roles, overlay_groups = _structure_previews(
        image, mask, labels, label_map, output
    )
    voxel_volume_mm3 = float(np.prod(mask_image.header.get_zooms()))
    statistics = {}
    for label in labels:
        selected = mask == label
        statistics[label_map[label]] = {
            "volume_mm3": float(selected.sum() * voxel_volume_mm3),
            "mean_hu": float(image[selected].mean()),
            "voxels": int(selected.sum()),
        }
    statistics_path.write_text(json.dumps(statistics, indent=2) + "\n")
    result = {
        "model": "TotalSegmentator 2.15 MSK CT",
        "segmentation_file": str(mask_path),
        "preview_files": preview_files,
        "preview_roles": preview_roles,
        "overlay_groups": overlay_groups,
        "structures_present": [label_map[label] for label in labels],
        "label_map": {str(label): label_map[label] for label in labels},
        "statistics": statistics,
        "fast_mode": bool(fast),
    }
    (output / f"{source.stem}_result.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "preview_b64"}, indent=2) + "\n"
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
    with tempfile.TemporaryDirectory(prefix="maple_totalseg_") as output_dir:
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
            {"pred": len(result["structures_present"]), "pred_name": "MSK_structures_present"},
        )

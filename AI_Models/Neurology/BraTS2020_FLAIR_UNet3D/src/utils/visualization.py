from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure


REGION_COLORS = {
    "wt": np.array([1.0, 0.2, 0.2], dtype=np.float32),
    "tc": np.array([0.2, 1.0, 0.2], dtype=np.float32),
    "et": np.array([0.2, 0.5, 1.0], dtype=np.float32),
}
REGION_MESH_COLORS = {
    "WT": "tomato",
    "TC": "limegreen",
    "ET": "dodgerblue",
}
REGION_INDEX = {"wt": 0, "tc": 1, "et": 2}
REGION_INDEX_UPPER = {"WT": 0, "TC": 1, "ET": 2}


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def select_representative_slice(mask_3d: np.ndarray) -> int:
    mask_3d = _to_numpy(mask_3d)
    if mask_3d.ndim != 3:
        raise ValueError(f"mask_3d must be 3D, got shape={mask_3d.shape}")
    slice_scores = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    if float(slice_scores.max()) <= 0:
        return int(mask_3d.shape[0] // 2)
    return int(slice_scores.argmax())


def normalize_slice_for_display(image_slice: np.ndarray) -> np.ndarray:
    image_slice = np.asarray(image_slice, dtype=np.float32)
    finite = np.isfinite(image_slice)
    if not np.any(finite):
        return np.zeros_like(image_slice, dtype=np.float32)
    values = image_slice[finite]
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(image_slice, dtype=np.float32)
    normalized = (image_slice - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def create_overlay(
    image_slice: np.ndarray,
    gt_mask_slice: np.ndarray | None = None,
    pred_mask_slice: np.ndarray | None = None,
    title: str | None = None,
    color: np.ndarray | None = None,
) -> plt.Figure:
    color = REGION_COLORS["wt"] if color is None else color
    base = normalize_slice_for_display(image_slice)
    rgb = np.repeat(base[..., None], 3, axis=-1)
    if pred_mask_slice is not None:
        pred = _to_numpy(pred_mask_slice).astype(bool)
        rgb[pred] = 0.55 * rgb[pred] + 0.45 * color

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(np.rot90(rgb), interpolation="nearest")
    if gt_mask_slice is not None and np.any(gt_mask_slice):
        ax.contour(np.rot90(_to_numpy(gt_mask_slice).astype(np.float32)), levels=[0.5], colors="yellow", linewidths=0.8)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def save_region_overlays(
    image: np.ndarray,
    pred: np.ndarray,
    label: np.ndarray | None,
    case_id: str,
    output_dir: str | Path,
) -> None:
    image = _to_numpy(image)
    pred = _to_numpy(pred)
    label_np = None if label is None else _to_numpy(label)
    if image.shape[0] != 1:
        raise ValueError(f"image must have shape (1, D, H, W), got {image.shape}")
    if pred.shape[0] != 3:
        raise ValueError(f"pred must have shape (3, D, H, W), got {pred.shape}")
    if label_np is not None and label_np.shape[0] != 3:
        raise ValueError(f"label must have shape (3, D, H, W), got {label_np.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_3d = image[0]

    for region, channel in REGION_INDEX.items():
        gt_mask = None if label_np is None else label_np[channel] > 0
        combined_mask = pred[channel] > 0 if gt_mask is None else np.logical_or(pred[channel] > 0, gt_mask)
        slice_index = select_representative_slice(combined_mask)
        fig = create_overlay(
            image_3d[slice_index],
            gt_mask_slice=None if gt_mask is None else gt_mask[slice_index],
            pred_mask_slice=pred[channel, slice_index] > 0,
            title=f"{case_id} {region.upper()} slice={slice_index} | pred fill, GT yellow contour",
            color=REGION_COLORS[region],
        )
        fig.savefig(output_dir / f"{case_id}_{region}_overlay.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def create_blank_rendering(output_path: str | Path, message: str = "Empty mask") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 900), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 44)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), message, font=font)
    x = (image.width - (bbox[2] - bbox[0])) // 2
    y = (image.height - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), message, fill=(80, 80, 80), font=font)
    image.save(output_path)
    return output_path


def mask_to_mesh(mask_3d: np.ndarray, spacing: tuple[float, float, float] | None = None, smooth: bool = True):
    mask = _to_numpy(mask_3d).astype(bool)
    if mask.ndim != 3:
        raise ValueError(f"mask_3d must have shape (D, H, W), got {mask.shape}")
    if not np.any(mask):
        return None

    try:
        from vedo import Mesh
    except Exception as exc:
        raise RuntimeError("vedo is required for 3D rendering. Install vedo and vtk-osmesa.") from exc

    padded = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=spacing)
    if spacing is None:
        verts = verts - 1.0
    else:
        spacing_array = np.asarray(spacing, dtype=np.float32)
        verts = verts - spacing_array
    mesh = Mesh([verts, faces])
    if smooth:
        try:
            mesh.smooth(niter=20, pass_band=0.08)
        except TypeError:
            mesh.smooth()
    return mesh


def render_vedo_meshes(
    meshes: list,
    output_path: str | Path,
    title: str | None = None,
    camera: dict | None = None,
    size: tuple[int, int] = (1200, 900),
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshes = [mesh for mesh in meshes if mesh is not None]
    if not meshes:
        return create_blank_rendering(output_path, "Empty mask")

    plotter = None
    try:
        from vedo import Plotter, Text2D

        plotter = Plotter(offscreen=True, size=size, bg="white")
        actors = list(meshes)
        if title:
            actors.append(Text2D(title, pos="top-center", c="black", s=1.0))
        plotter.show(*actors, axes=0, interactive=False, resetcam=True)
        if camera:
            plotter.camera.SetPosition(*camera.get("position", plotter.camera.GetPosition()))
            plotter.camera.SetFocalPoint(*camera.get("focal_point", plotter.camera.GetFocalPoint()))
            plotter.camera.SetViewUp(*camera.get("view_up", plotter.camera.GetViewUp()))
        plotter.screenshot(str(output_path))
        return output_path
    except Exception as exc:
        print(f"WARNING: 3D rendering failed for {output_path}: {exc}")
        return create_blank_rendering(output_path, f"3D rendering failed\n{type(exc).__name__}")
    finally:
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass


def render_region_3d(
    mask_3d: np.ndarray,
    output_path: str | Path,
    region_name: str,
    spacing: tuple[float, float, float] | None = None,
) -> Path:
    region = region_name.upper()
    try:
        mesh = mask_to_mesh(mask_3d, spacing=spacing, smooth=True)
        if mesh is None:
            return create_blank_rendering(output_path, f"Empty {region} mask")
        mesh.c(REGION_MESH_COLORS.get(region, "tomato")).alpha(0.82).lighting("plastic")
        return render_vedo_meshes([mesh], output_path, title=f"{region} prediction")
    except Exception as exc:
        print(f"WARNING: Could not render {region} 3D mask: {exc}")
        return create_blank_rendering(output_path, f"{region} rendering unavailable")


def render_all_regions_3d(pred: np.ndarray, output_path: str | Path, spacing: tuple[float, float, float] | None = None) -> Path:
    pred_np = _to_numpy(pred)
    if pred_np.shape[0] != 3:
        raise ValueError(f"pred must have shape (3, D, H, W), got {pred_np.shape}")
    meshes = []
    try:
        for region, channel in REGION_INDEX_UPPER.items():
            mesh = mask_to_mesh(pred_np[channel] > 0, spacing=spacing, smooth=True)
            if mesh is None:
                continue
            alpha = 0.28 if region == "WT" else 0.62
            mesh.c(REGION_MESH_COLORS[region]).alpha(alpha).lighting("plastic")
            meshes.append(mesh)
        if not meshes:
            return create_blank_rendering(output_path, "Empty WT/TC/ET masks")
        return render_vedo_meshes(meshes, output_path, title="WT / TC / ET prediction")
    except Exception as exc:
        print(f"WARNING: Could not render combined 3D regions: {exc}")
        return create_blank_rendering(output_path, "3D rendering unavailable")


def save_3d_renderings_vedo(
    pred: np.ndarray,
    case_id: str,
    output_dir: str | Path,
    modality: str,
    spacing: tuple[float, float, float] | None = None,
) -> dict[str, str]:
    pred_np = _to_numpy(pred)
    if pred_np.shape[0] != 3:
        raise ValueError(f"pred must have shape (3, D, H, W), got {pred_np.shape}")
    pred_np = pred_np > 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "regions_3d": output_dir / f"{case_id}_{modality}_3d_regions.png",
        "WT_3d": output_dir / f"{case_id}_{modality}_3d_WT.png",
        "TC_3d": output_dir / f"{case_id}_{modality}_3d_TC.png",
        "ET_3d": output_dir / f"{case_id}_{modality}_3d_ET.png",
    }
    render_all_regions_3d(pred_np, paths["regions_3d"], spacing=spacing)
    for region, channel in REGION_INDEX_UPPER.items():
        render_region_3d(pred_np[channel], paths[f"{region}_3d"], region, spacing=spacing)
    return {key: str(path) for key, path in paths.items()}


def _blend_region_overlay(base: np.ndarray, masks: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    rgb = np.repeat(normalize_slice_for_display(base)[..., None], 3, axis=-1)
    for region, channel in REGION_INDEX.items():
        mask = masks[channel].astype(bool)
        if np.any(mask):
            rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * REGION_COLORS[region]
    return np.clip(rgb, 0.0, 1.0)


def _view_slices(image_3d: np.ndarray, masks_4d: np.ndarray, label_4d: np.ndarray | None = None):
    combined = np.any(masks_4d > 0, axis=0)
    if label_4d is not None:
        combined = np.logical_or(combined, np.any(label_4d > 0, axis=0))
    if not np.any(combined):
        d, h, w = (s // 2 for s in image_3d.shape)
    else:
        d = int(combined.reshape(combined.shape[0], -1).sum(axis=1).argmax())
        h = int(combined.sum(axis=(0, 2)).argmax())
        w = int(combined.sum(axis=(0, 1)).argmax())

    views = [
        ("Axial", image_3d[d], masks_4d[:, d], None if label_4d is None else label_4d[:, d], d),
        ("Coronal", image_3d[:, h, :], masks_4d[:, :, h, :], None if label_4d is None else label_4d[:, :, h, :], h),
        ("Sagittal", image_3d[:, :, w], masks_4d[:, :, :, w], None if label_4d is None else label_4d[:, :, :, w], w),
    ]
    return views


def save_prediction_context_panel(
    image: np.ndarray,
    pred: np.ndarray,
    case_id: str,
    output_path: str | Path,
    modality: str,
    label: np.ndarray | None = None,
) -> Path:
    image_np = _to_numpy(image)
    pred_np = _to_numpy(pred).astype(bool)
    label_np = None if label is None else _to_numpy(label).astype(bool)
    if image_np.shape[0] != 1:
        raise ValueError(f"image must have shape (1, D, H, W), got {image_np.shape}")
    if pred_np.shape[0] != 3:
        raise ValueError(f"pred must have shape (3, D, H, W), got {pred_np.shape}")
    if label_np is not None and label_np.shape[0] != 3:
        raise ValueError(f"label must have shape (3, D, H, W), got {label_np.shape}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_3d = image_np[0]
    views = _view_slices(image_3d, pred_np, label_np)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=150, facecolor="#111111")
    for ax, (view_name, image_slice, pred_slice, label_slice, index) in zip(axes, views):
        overlay = _blend_region_overlay(image_slice, pred_slice)
        ax.imshow(np.rot90(overlay), interpolation="nearest")
        if label_slice is not None:
            for region, channel in REGION_INDEX.items():
                contour = np.rot90(label_slice[channel].astype(np.float32))
                if np.any(contour):
                    ax.contour(contour, levels=[0.5], colors="yellow", linewidths=0.7, alpha=0.9)
        ax.set_title(f"{view_name} {index}", color="white", fontsize=11)
        ax.axis("off")

    legend_items = "WT red  |  TC green  |  ET blue"
    if label_np is not None:
        legend_items += "  |  GT yellow contour"
    fig.suptitle(f"{case_id} - {modality.upper()} prediction on MRI", color="white", fontsize=14, y=0.98)
    fig.text(0.5, 0.035, legend_items, ha="center", color="white", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94], pad=0.4)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path

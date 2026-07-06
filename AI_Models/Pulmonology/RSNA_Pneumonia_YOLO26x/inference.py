#!/usr/bin/env python3
"""Maple inference entry point for RSNA Pneumonia Opacity Detection.

Input:  DICOM chest X-ray file path (str)
Output: (result_image, predictions)
  - result_image: np.ndarray (H, W, 3) uint8 RGB — bbox overlay
  - predictions:  list[dict]  — per-box confidence scores

CLI:
    python inference.py --dicom sample_data/sample_02.dcm --weights checkpoint/best.pt --output results/output.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_MODEL_CACHE: dict[str, Any] = {"model": None}

CONF_THRESHOLD = 0.15
IMGSZ = 1024
DEDUP_OVERLAP = 0.5  # suppress lower-conf box when overlap fraction exceeds this


# ---------------------------------------------------------------------------
# DICOM preprocessing — replicates training-time pipeline (equalize)
# ---------------------------------------------------------------------------

def _normalize_to_uint8(pixel_array: np.ndarray) -> np.ndarray:
    image = pixel_array.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(image[finite], (1.0, 99.0))
    if high <= low:
        low, high = float(image[finite].min()), float(image[finite].max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return (image * 255.0).round().astype(np.uint8)


def _dicom_to_rgb(dicom_path: Path) -> np.ndarray:
    import pydicom
    from PIL import Image, ImageOps

    ds = pydicom.dcmread(dicom_path)
    pixel = ds.pixel_array
    if pixel.ndim == 3:
        pixel = pixel[..., 0]

    gray = _normalize_to_uint8(pixel)

    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        gray = 255 - gray

    gray = np.asarray(ImageOps.equalize(Image.fromarray(gray)), dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load_model(model_path: str):
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model


def _get_model(model_path: str):
    if _MODEL_CACHE["model"] is None:
        _MODEL_CACHE["model"] = _load_model(model_path)
    return _MODEL_CACHE["model"]


# ---------------------------------------------------------------------------
# Dedup — suppress overlapping lower-confidence boxes
# ---------------------------------------------------------------------------

def _overlap_fraction(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    fracs = [inter / a for a in (area_a, area_b) if a > 0]
    return max(fracs, default=0.0)


def _dedup(
    boxes: list[tuple[int, int, int, int]],
    confs: list[float],
    threshold: float = DEDUP_OVERLAP,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    pairs = sorted(zip(boxes, confs), key=lambda x: x[1], reverse=True)
    kept_boxes: list[tuple[int, int, int, int]] = []
    kept_confs: list[float] = []
    for box, conf in pairs:
        if all(_overlap_fraction(box, kb) < threshold for kb in kept_boxes):
            kept_boxes.append(box)
            kept_confs.append(conf)
    return kept_boxes, kept_confs


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def _draw_boxes(
    image_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    confs: list[float],
) -> np.ndarray:
    import cv2

    overlay = image_rgb.copy()
    color = (56, 120, 255)   # YOLO blue (RGB)
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, image_rgb.shape[0] / 1400)
    font_thickness = max(1, int(font_scale * 1.8))

    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        label = f"Pneumonia  {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        tag_y = y1 - th - baseline - 6
        # 박스 위가 잘리면 아래쪽에 표시
        if tag_y < 0:
            tag_y = y2 + baseline + 4
            text_y = tag_y + th
        else:
            text_y = y1 - baseline - 3

        cv2.rectangle(overlay, (x1, tag_y), (x1 + tw + 6, tag_y + th + baseline + 4), color, -1)
        cv2.putText(overlay, label, (x1 + 3, text_y),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return overlay


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def main(input_data: str, model_path: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Maple inference entry point.

    Args:
        input_data: absolute path to a DICOM chest X-ray file
        model_path: absolute path to best.pt (YOLO weights)

    Returns:
        result_image: RGB uint8 ndarray with bbox overlay (no detections → original image)
        predictions:  list of dicts with keys: x1, y1, x2, y2, conf, pred, pred_name
    """
    dicom_path = Path(input_data)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM not found: {dicom_path}")

    image_rgb = _dicom_to_rgb(dicom_path)
    model = _get_model(model_path)

    results = model.predict(
        source=image_rgb,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    result = results[0]
    h_orig, w_orig = image_rgb.shape[:2]
    h_pred, w_pred = result.orig_shape

    scale_x = w_orig / w_pred
    scale_y = h_orig / h_pred

    boxes: list[tuple[int, int, int, int]] = []
    confs: list[float] = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            boxes.append((
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            ))
            confs.append(conf)

    boxes, confs = _dedup(boxes, confs)
    result_image = _draw_boxes(image_rgb, boxes, confs) if boxes else image_rgb.copy()

    predictions = [
        {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf,
            "pred": 1,
            "pred_name": "pneumonia_opacity",
        }
        for (x1, y1, x2, y2), conf in zip(boxes, confs)
    ]

    return result_image, predictions


# ---------------------------------------------------------------------------
# CLI for local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description="RSNA pneumonia inference (single DICOM)")
    parser.add_argument("--dicom", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result_img, preds = main(str(args.dicom), str(args.weights))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result_img).save(args.output)
    print(f"Detections: {len(preds)}")
    for p in preds:
        print(f"  [{p['x1']},{p['y1']},{p['x2']},{p['y2']}] conf={p['conf']:.3f}")
    print(f"Saved: {args.output}")



"""Kvasir-SEG YOLOv8 polyp detection inference."""
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics-maple")

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


def main(input_data, model_path: str):
    weights = Path(model_path)
    if weights.is_dir():
        weights = weights / "kvasir-yolov8-best.pt"
    image = Image.open(input_data).convert("RGB")
    prediction = YOLO(str(weights)).predict(input_data, conf=0.20, device="cpu", verbose=False)[0]
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    detections = []
    line_width = max(3, image.width // 200)
    for box, confidence in zip(prediction.boxes.xyxy.tolist(), prediction.boxes.conf.tolist()):
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=line_width)
        detections.append({
            "pred": 0, "pred_name": "polyp", "confidence": float(confidence),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return np.asarray(overlay, dtype=np.uint8), detections

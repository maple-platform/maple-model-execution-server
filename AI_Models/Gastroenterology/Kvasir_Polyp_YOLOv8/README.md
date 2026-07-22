# Kvasir Polyp YOLOv8

Single-class polyp detection using `pourmand1376/Kvasir-SEG-yolov8`.

- Checkpoint: `checkpoint/kvasir-yolov8-best.pt`
- Input: colonoscopy PNG/JPG/JPEG image
- Output `[0]`: RGB bounding-box overlay
- Output `[1]`: list of polyp coordinates and confidence scores
- Confidence threshold: 0.20

Research use only; detections require clinical review.

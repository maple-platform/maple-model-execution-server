# Kvasir_Polyp_YOLOv8

## Overview
- **Department**: Gastroenterology
- **Model type**: standalone
- **task_type**: bbox detection
- **result_type**: `['bbox_overlay', 'detection_predictions']`
- **output_image_role**: `bbox_overlay`

## Target condition
- Colorectal polyp in colonoscopy images

## Input data
| Item | Format | Description |
|---|---|---|
| Colonoscopy image | PNG/JPG/JPEG | RGB endoscopy frame |

## Output
- **Return**: `(overlay, detections)`
- `overlay`: `np.ndarray`, shape `(H, W, 3)`, dtype `uint8`, RGB
- `detections`: list of dictionaries containing `pred`, `pred_name`, `confidence`, and `x1/y1/x2/y2`

## Model files
| File | Description |
|---|---|
| `checkpoint/kvasir-yolov8-best.pt` | YOLOv8 polyp detector weights |

## Model details
| Item | Value |
|---|---|
| Architecture | YOLOv8 |
| Training data | Kvasir-SEG |
| Detection threshold | 0.20 |

## Preprocessing
1. Decode the image through Ultralytics.
2. Apply the model's native letterbox and normalization pipeline.
3. Keep detections with confidence at least 0.20.

## Execution example
```python
from inference import main

overlay, detections = main(
    input_data="sample_data/sample_01.jpg",
    model_path="checkpoint/kvasir-yolov8-best.pt",
)
print(overlay.shape, detections)
```

## Notes
- Research use only; detections require clinical review.

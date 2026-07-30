# MammoCrop_ROI

## Overview
- **Department**: Obstetrics
- **Model type**: standalone
- **task_type**: bbox detection
- **result_type**: `['bbox_overlay', 'detection_predictions']`
- **output_image_role**: `bbox_overlay`

## Target region
- Breast/background crop region in mammography

## Input data
| Item | Format | Description |
|---|---|---|
| Mammogram | PNG/JPG/JPEG | Grayscale or RGB mammography image |

## Output
- Return: `(overlay, predictions)`.
- `overlay`: RGB `np.ndarray`, `(H, W, 3)`, `uint8`.
- `predictions`: list containing `pred`, `pred_name`, and `x/y/width/height`.

## Model files
| File | Description |
|---|---|
| `checkpoint/model.safetensors` | MammoCrop localization weights |

## Preprocessing
1. Convert the image to grayscale.
2. Resize to 256 x 256 and normalize to `[-1, 1]`.
3. Rescale predicted `xywh` coordinates to the original image.

## Execution example
```python
from inference import main
overlay, predictions = main("sample_data/sample_01.png", "checkpoint/model.safetensors")
print(overlay.shape, predictions)
```

## Notes
- Source: `ianpan/mammo-crop`.
- The box is a breast/background crop, not cancer localization.

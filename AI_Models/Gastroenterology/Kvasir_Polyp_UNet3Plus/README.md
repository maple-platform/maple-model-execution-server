# Kvasir_Polyp_UNet3Plus

## Overview
- **Department**: Gastroenterology
- **Model type**: standalone
- **task_type**: segmentation
- **result_type**: `['segmentation_overlay']`
- **output_image_role**: `segmentation_overlay`

## Target condition
- Colorectal polyp in colonoscopy images

## Input data
| Item | Format | Description |
|---|---|---|
| Colonoscopy image | PNG/JPG/JPEG | RGB endoscopy frame |

## Output
- **Return type**: `np.ndarray`, shape `(H, W, 3)`, dtype `uint8`, RGB
- **Content**: original image with the predicted binary polyp mask overlaid in red

## Model files
| File | Description |
|---|---|
| `checkpoint/model.safetensors` | UNet3+ segmentation weights |

## Model details
| Item | Value |
|---|---|
| Architecture | UNet3+ with EfficientNet-B0 encoder |
| Training data | Kvasir-SEG augmented training set |
| Published metrics | Dice 0.9234, IoU 0.8577 |

## Preprocessing
1. Convert the input image to RGB.
2. Resize to 256 x 256.
3. Convert pixels to a float tensor in `[0, 1]`.
4. Threshold the sigmoid mask at 0.5 and resize it to the source resolution.

## Execution example
```python
from inference import main

result = main(
    input_data="sample_data/sample_01.jpg",
    model_path="checkpoint/model.safetensors",
)
print(result.shape, result.dtype)
```

## Notes
- Research use only; segmentation requires clinical review.

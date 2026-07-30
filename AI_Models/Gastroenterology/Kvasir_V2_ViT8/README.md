# Kvasir_V2_ViT8

## Overview
- **Department**: Gastroenterology
- **Model type**: standalone
- **task_type**: classification
- **result_type**: `text`
- **output_image_role**: `null`

## Target conditions
- Polyps, esophagitis, ulcerative colitis, dyed lesions, and normal GI landmarks

## Input data
| Item | Format | Description |
|---|---|---|
| Endoscopy image | PNG/JPG/JPEG | Upper or lower GI endoscopy frame |

## Output
- **Return type**: `pd.DataFrame`
- Required columns: `pred`, `pred_name`
- Additional columns: top probability and all eight class probabilities

## Model files
| File | Description |
|---|---|
| `checkpoint/pytorch_model.bin` | ViT classification weights |
| `checkpoint/config.json` | Architecture and class mapping |
| `checkpoint/preprocessor_config.json` | Image preprocessing configuration |

## Model details
| Item | Value |
|---|---|
| Architecture | ViT-Base, patch size 16, 224 x 224 input |
| Training data | Kvasir-V2 |
| Number of classes | 8 |

## Preprocessing
1. Convert the input image to RGB.
2. Apply `ViTImageProcessor` from the local checkpoint directory.
3. Apply softmax to the eight output logits.

## Execution example
```python
from inference import main

result = main(
    input_data="sample_data/sample_01.jpg",
    model_path="checkpoint",
)
print(result)
```

## Notes
- Research use only; scores are not calibrated clinical probabilities.

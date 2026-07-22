# Breast_Ultrasound_ViT

## Overview
- **Department**: Obstetrics
- **Model type**: standalone
- **task_type**: classification
- **result_type**: `text`
- **output_image_role**: `null`

## Target conditions
- Benign breast lesion, malignant breast lesion, and normal breast ultrasound

## Input data
| Item | Format | Description |
|---|---|---|
| Breast ultrasound | PNG/JPG/JPEG | RGB ultrasound image |

## Output
- `pd.DataFrame` with `pred`, `pred_name`, `prob`, and three class probabilities.

## Model files
| File | Description |
|---|---|
| `checkpoint/model.safetensors` | ViT weights |
| `checkpoint/config.json` | Architecture and label mapping |
| `checkpoint/preprocessor_config.json` | Image preprocessing configuration |

## Preprocessing
1. Convert the image to RGB.
2. Apply the local Hugging Face image processor.
3. Apply softmax to the three logits.

## Execution example
```python
from inference import main
result = main("sample_data/sample_01.png", "checkpoint")
print(result)
```

## Notes
- Source: `hugging-science/breast-cancer-detector-2`.
- Research use only; not a standalone clinical diagnosis.

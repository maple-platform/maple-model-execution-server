# LC25000_Colon_ViT

## Overview
- **Department**: Gastroenterology
- **Model type**: standalone
- **task_type**: classification
- **result_type**: `text`
- **output_image_role**: `null`

## Target condition
- Colon adenocarcinoma in H&E histopathology tiles

## Input data
| Item | Format | Description |
|---|---|---|
| Histopathology tile | PNG/JPG/JPEG | RGB H&E colon tissue tile |

## Output
- **Return type**: `pd.DataFrame`
- Required columns: `pred`, `pred_name`
- Additional columns: `prob`, `prob_benign`, `prob_adenocarcinoma`

## Model files
| File | Description |
|---|---|
| `checkpoint/model.safetensors` | ViT classification weights |
| `checkpoint/config.json` | Architecture and preprocessing configuration |

## Model details
| Item | Value |
|---|---|
| Architecture | ViT-Base patch16 224 |
| Fine-tuning data | LC25000 colon subset |
| Classes | Benign colon tissue, colon adenocarcinoma |

## Preprocessing
1. Convert the tile to RGB.
2. Apply the normalization and resize configuration stored in `config.json`.
3. Apply softmax to the two output logits.

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
- The upstream Owkin license restricts use to non-commercial research and education.
- Research use only; this output is not a clinical diagnosis.

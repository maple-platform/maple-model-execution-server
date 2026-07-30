# Cervical_Cytology_SIPaKMeD

## Overview
- **Department**: Obstetrics
- **Model type**: standalone
- **task_type**: classification
- **result_type**: `text`
- **output_image_role**: `null`

## Target conditions
- Five cervical cell morphology categories used in Pap-smear assessment

## Input data
| Item | Format | Description |
|---|---|---|
| Cervical cell crop | PNG/JPG/JPEG | Cropped RGB Pap-smear cell image |

## Output
- `pd.DataFrame` with `pred`, `pred_name`, `prob`, and five class probabilities.

## Model files
| File | Description |
|---|---|
| `checkpoint/best_alexnet.pth` | Fine-tuned AlexNet weights |
| `checkpoint/labels.json` | Class-index mapping |

## Preprocessing
1. Convert to RGB and resize to 224 x 224.
2. Apply ImageNet mean/std normalization.
3. Apply softmax to the five logits.

## Execution example
```python
from inference import main
result = main("sample_data/sample_01.png", "checkpoint")
print(result)
```

## Notes
- Source: `hp1318/alexnet-finetuned-sipakmed`.
- Single-cell morphology output is not a patient-level cancer diagnosis.

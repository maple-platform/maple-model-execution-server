# Breast_IDC_Keras

## Overview
- **Department**: Obstetrics
- **Model type**: standalone
- **task_type**: classification
- **result_type**: `text`
- **output_image_role**: `null`

## Target condition
- Patch-level invasive ductal carcinoma (IDC)

## Input data
| Item | Format | Description |
|---|---|---|
| Histopathology patch | PNG/JPG/JPEG | RGB breast tissue patch |

## Output
- `pd.DataFrame` with `pred`, `pred_name`, `prob`, and `idc_positive_prob`.

## Model files
| File | Description |
|---|---|
| `checkpoint/CanDetect.h5` | Keras CNN weights |

## Preprocessing
1. Convert to RGB and resize to 50 x 50.
2. The model rescales pixel values by `1/255`.

## Execution example
```python
from inference import main
result = main("sample_data/sample_01.png", "checkpoint/CanDetect.h5")
print(result)
```

## Notes
- Source: `MUmairAB/Breast_Cancer_Detector`.
- Research use only; patch-level output is not a patient-level diagnosis.

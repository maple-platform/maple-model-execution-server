# MammoScreen

## Overview
- **Department**: Obstetrics
- **Model type**: standalone
- **task_type**: classification
- **result_type**: `text`
- **output_image_role**: `null`

## Target conditions
- Mammographic breast cancer risk and breast density A-D

## Input data
| Item | Format | Description |
|---|---|---|
| Cropped mammogram | PNG/JPG/JPEG | One cropped grayscale mammography view |

## Output
- `pd.DataFrame` with `pred`, `pred_name`, `prob`, `cancer_score`, predicted density, and density A-D probabilities.

## Model files
| File | Description |
|---|---|
| `checkpoint/model.safetensors` | MammoScreen ensemble weights |

## Preprocessing
1. Convert the image to grayscale.
2. Resize/pad it for each of three ensemble members.
3. Normalize to `[-1, 1]` and average member outputs.

## Execution example
```python
from inference import main
result = main("sample_data/sample_01.png", "checkpoint/model.safetensors")
print(result)
```

## Notes
- Source: `ianpan/mammoscreen`.
- The score is not guaranteed to be a calibrated clinical probability.

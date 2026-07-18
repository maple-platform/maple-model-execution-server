# RSNA 2024 Subarticular Keypoints

Predicts bilateral L1/L2 through L5/S1 subarticular `(x, y, z)` locations from a complete axial T2 DICOM series. Two R3D-18 coordinate regressors are averaged and a third 32-bin z classifier refines the slice location. Compatible 96 mm three-slice crops are exported for downstream grading.

Held-out RSNA 2024 test: XY mean error 2.76 mm, z mean error 4.07 mm, and 3D mean error 5.43 mm over 1,917 visible points in 235 series. Research use only.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: DICOM series directory path (`str`)
- **Input extensions**: `dcm`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `bbox_overlay`
- **Prediction fields**: `pred`, `pred_name`, `points`

### Image order

| Index | Content |
|---|---|
| `[0]` | Lumbar MRI with 10 predicted keypoints |

### Direct execution

Run this example from the model directory.

```python
from inference import main

images, predictions = main(
    input_data="sample_data/series",
    model_path="checkpoint",
)
print(predictions)
```

`result/result.json` lists the role-named PNG files from a completed sample inference.
<!-- MAPLE_MANUAL_END -->

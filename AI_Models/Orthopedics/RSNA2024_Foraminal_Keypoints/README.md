# RSNA 2024 Foraminal Keypoints

Predicts left and right L1/L2 through L5/S1 `(x, y, z)` locations and exports compatible three-slice crops from a sagittal T1 DICOM series. Test XY mean error is 5.41 mm and 3D mean error is 6.67 mm across 1,970 points in 199 series. Research use only; not a medical device.

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

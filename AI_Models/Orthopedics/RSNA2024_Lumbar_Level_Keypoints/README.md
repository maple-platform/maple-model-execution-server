# RSNA 2024 Lumbar Level Keypoints

Predicts L1/L2 through L5/S1 `(x, y, z)` locations from one sagittal T2/STIR DICOM series. It also exports a 96 mm three-adjacent-slice RGB crop for every level, compatible with `RSNA2024_Spinal_Canal_Crop`.

Held-out study-level test result: XY mean error 5.92 mm, median 3.92 mm, and 3D mean error 6.76 mm across 980 visible points in 198 series. RSNA competition data remains subject to the competition terms. Research use only; not a medical device.

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
| `[0]` | Lumbar MRI with 5 predicted keypoints |

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

# RSNA 2024 Neural Foraminal Pipeline

End-to-end sagittal T1 DICOM inference for ten bilateral foraminal keypoints, crop generation, and three-grade narrowing classification. Predicted-coordinate test QWK is 0.529 and severe AUROC is 0.925 across 1,910 levels. Research use only; not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: DICOM series directory path (`str`)
- **Input extensions**: `dcm`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `classification_probabilities`
- **Prediction fields**: `pred`, `pred_name`, `maximum_grade`, `maximum_label`, `levels or points`

### Image order

| Index | Content |
|---|---|
| `[0]` | Keypoint overview |
| `[1]~[N]` | One annotated bilateral neural foraminal narrowing crop per point |

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

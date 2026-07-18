# RSNA 2024 Spinal Canal Pipeline

End-to-end sagittal T2/STIR DICOM-series inference: five lumbar level keypoints, 96 mm three-slice crop generation, and Normal/Mild, Moderate, or Severe spinal-canal stenosis grading per level.

On the held-out study split with predicted keypoints (950 levels), accuracy was 0.906, balanced accuracy 0.790, QWK 0.819, and severe AUROC 0.985. RSNA competition data remains subject to the competition terms. Research use only; not a medical device.

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
| `[1]~[N]` | One annotated spinal canal stenosis crop per point |

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

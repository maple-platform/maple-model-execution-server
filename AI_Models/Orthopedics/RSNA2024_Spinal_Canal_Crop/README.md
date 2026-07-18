# RSNA 2024 Spinal Canal Crop

Three-class lumbar spinal-canal stenosis grading from a coordinate-centered, 96 mm, three-adjacent-slice sagittal T2/STIR crop. Input is either the prepared RGB PNG (channels are previous/current/next slices) or a directory containing exactly three ordered image/DICOM slices.

This is an oracle-coordinate crop classifier and does not find the lumbar level or stenosis location. RSNA competition data remains subject to the competition terms. Research use only; not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `png, jpg, jpeg, dcm`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `classification_probabilities`
- **Prediction fields**: `pred`, `pred_name`, `grade`, `label`, `expected_grade`, `probabilities`, `input_mode`

### Image order

| Index | Content |
|---|---|
| `[0]` | Input crop annotated with spinal canal stenosis grade |

### Direct execution

Run this example from the model directory.

```python
from inference import main

images, predictions = main(
    input_data="sample_data/sample.png",
    model_path="checkpoint",
)
print(predictions)
```

`result/result.json` lists the role-named PNG files from a completed sample inference.
<!-- MAPLE_MANUAL_END -->

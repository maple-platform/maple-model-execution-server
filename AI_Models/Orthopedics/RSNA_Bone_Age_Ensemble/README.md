# RSNA Bone Age Ensemble

Pediatric hand X-ray bone-age inference with automatic hand cropping. Path-only input defaults to `female`. The local adapter also accepts `{"image_path": "...", "sex": "male"}`; exposing that optional dictionary form requires agreement with the platform runner. The upstream Apache-2.0 model reports 4.16 months MAE on the 200-image RSNA challenge test set.

Weights are stored under `checkpoint/` locally and excluded from Git. RSNA data terms apply to source images. This model is for research use and is not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `dcm, png, jpg, jpeg`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `classification_probabilities`
- **Prediction fields**: `pred`, `pred_name`, `bone_age_months`, `bone_age_years`, `sex`, `uncertainty_months`, `crop_bbox_xywh`

### Image order

| Index | Content |
|---|---|
| `[0]` | Normalized hand crop with predicted age overlaid at top right |

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

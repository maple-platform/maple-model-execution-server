# Knee OA KL and JSN

Explainable bilateral PA knee radiograph pipeline combining knee detection, femur/tibia segmentation, medial and lateral joint-space-narrowing assessment, and final Kellgren-Lawrence grade 0-4. The published model reports 75.86% five-class accuracy on MOST and 64.48% on external OAI, with 98.78% bone segmentation Dice.

Input must be a bilateral PA knee radiograph. The adapted source and weights are CC BY-NC-SA 4.0, so this package is non-commercial and share-alike. This model is for research use and is not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `dcm, png, jpg, jpeg`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `segmentation_overlay`
- **Prediction fields**: `pred`, `pred_name`, `knees`

### Image order

| Index | Content |
|---|---|
| `[0]` | Original bilateral knee radiograph |
| `[1]` | Right knee segmentation overlay |
| `[2]` | Left knee segmentation overlay |

### Direct execution

Run this example from the model directory.

```python
from inference import main

images, predictions = main(
    input_data="sample_data/smoke_bilateral_views.jpg",
    model_path="checkpoint",
)
print(predictions)
```

`result/result.json` lists the role-named PNG files from a completed sample inference.
<!-- MAPLE_MANUAL_END -->

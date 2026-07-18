# SPIDER Disc Grading

Multitask lumbar disc-patch inference for Pfirrmann grade and seven degenerative findings. Patient-level held-out performance is 0.8390 mean binary AUROC, 0.591 grade MAE, and 0.7068 quadratic weighted kappa.

Input is a cropped sagittal disc patch, not a whole MRI volume. The checkpoint under `checkpoint/` is excluded from Git. SPIDER is CC BY 4.0. This model is for research use and is not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `png, jpg, jpeg`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `gradcam_overlay`
- **Prediction fields**: `pred`, `pred_name`, `pfirrmann_grade`, `pfirrmann_expected_grade`, `pfirrmann_probabilities`, `high_confidence_findings`

### Image order

| Index | Content |
|---|---|
| `[0]` | Disc patch with Pfirrmann prediction GradCAM overlay |

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

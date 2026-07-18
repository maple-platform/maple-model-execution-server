# FracAtlas Fracture Fusion

Fracture screening and heatmap localization using two classifiers plus a segmentation model. Held-out performance is 0.9700 AUROC and 0.8612 average precision; the high-sensitivity operating point reaches 0.9683 recall.

Path-only input uses the `high_sensitivity` operating point. The local adapter also accepts `{"image_path": "...", "operating_point": "balanced"}`; exposing that optional dictionary form requires agreement with the platform runner. Checkpoints under `checkpoint/` are excluded from Git. FracAtlas is CC BY 4.0. This model is for research use and is not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `png, jpg, jpeg`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `segmentation_overlay`
- **Prediction fields**: `pred`, `pred_name`, `fused_probability`, `operating_point`, `regions`

### Image order

| Index | Content |
|---|---|
| `[0]` | Original radiograph with fracture-probability overlay |

### Direct execution

Run this example from the model directory.

```python
from inference import main

images, predictions = main(
    input_data="sample_data/sample.jpg",
    model_path="checkpoint",
)
print(predictions)
```

`result/result.json` lists the role-named PNG files from a completed sample inference.
<!-- MAPLE_MANUAL_END -->

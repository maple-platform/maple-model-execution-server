# AASCE Scoliosis Cobb

Standing AP spine X-ray inference for 68 vertebral landmarks and three Cobb-angle estimates. Local held-out evaluation: 7.696 degree Cobb MAE and 0.02183 normalized landmark error.

The checkpoint is stored under `checkpoint/best.pt` locally and is excluded from Git. Confirm the AASCE challenge data terms before redistribution or clinical use. This model is for research use and is not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `png, jpg, jpeg`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `bbox_overlay`
- **Prediction fields**: `pred`, `pred_name`, `landmarks`, `cobb_angles`

### Image order

| Index | Content |
|---|---|
| `[0]` | Spine radiograph with 68 landmarks and vertebral polygons |

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

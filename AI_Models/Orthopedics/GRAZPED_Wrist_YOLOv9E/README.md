# GRAZPED Wrist YOLOv9-E

Nine-class object detection for pediatric wrist radiographs. Held-out patient-level test performance is 0.860 mAP50 overall and 0.984 fracture mAP50, with 0.934 fracture recall.

The checkpoint is stored under `checkpoint/yolov9-e-640.pt` locally and excluded from Git. The dataset is CC BY 4.0; bundled YOLOv9 model code is MIT licensed. This model is for research use and is not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `png, jpg, jpeg`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `bbox_overlay`
- **Prediction fields**: `pred`, `pred_name`, `detections`, `fracture_detected`, `fracture_max_confidence`

### Image order

| Index | Content |
|---|---|
| `[0]` | Original wrist radiograph with fracture-only bounding boxes and no text |

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

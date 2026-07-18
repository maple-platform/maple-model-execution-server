# RSNA 2024 Subarticular Pipeline

End-to-end axial T2 DICOM series inference: bilateral lumbar keypoint localization, physical 96 mm three-slice crop extraction, and three-grade subarticular stenosis classification for ten side/level points.

Held-out RSNA 2024 predicted-coordinate test: accuracy 0.791, QWK 0.681, and severe AUROC 0.865 over 1,857 side/level crops. The pipeline is for research use only and is not a medical device.

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
| `[1]~[N]` | One annotated bilateral subarticular stenosis crop per point |

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

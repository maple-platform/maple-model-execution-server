# Knee OA KL Ensemble

Ordinal KL 0-4 classification for a single cropped AP/PA knee radiograph. The package averages flip-TTA probabilities from ConvNeXt-Tiny 320, ConvNeXt-Small 320, and ConvNeXt-Tiny 384 models.

KneeXrayData test performance (1,656 images): accuracy 0.7373, balanced accuracy 0.7363, quadratic weighted kappa 0.8848, grade MAE 0.2838, and KL>=2 AUROC 0.9630. The separate auto-crop test split (1,526 images) produced accuracy 0.7392 and QWK 0.8824.

The input must already contain one knee joint. Use `Knee_OA_KL_JSN` for bilateral joint detection, segmentation, JSN, and per-knee grading. KneeXrayData is CC BY 4.0; the ImageNet-pretrained backbone provenance must also be considered for downstream distribution. Research use only; not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `dcm, png, jpg, jpeg`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `gradcam_overlay`
- **Prediction fields**: `pred`, `pred_name`, `kl_grade`, `expected_kl_grade`, `kl_probabilities`, `oa_kl2_probability`

### Image order

| Index | Content |
|---|---|
| `[0]` | Knee radiograph with KL prediction GradCAM overlay |

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

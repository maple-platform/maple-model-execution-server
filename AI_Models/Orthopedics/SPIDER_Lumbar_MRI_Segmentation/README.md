# SPIDER Lumbar MRI Segmentation

3D lumbar MRI segmentation and anatomical labeling of vertebrae, intervertebral discs, and spinal canal. On all 87 official validation volumes, final Dice scores are 0.9264 foreground, 0.9288 vertebrae, 0.8558 discs, and 0.9321 spinal canal, with zero QC failures.

The public `main()` return contains five separate sagittal overlays plus label and foreground-volume QC data. The inference engine also creates a temporary MHA label map, but that temporary file is not part of the public return. Runtime is roughly 30-60 seconds per volume on an NVIDIA A6000. The engine is Apache-2.0 and SPIDER data is CC BY 4.0. This model is for research use and is not a medical device.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `nii.gz, nii`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `segmentation_overlay`
- **Prediction fields**: `pred`, `pred_name`, `foreground_voxels`, `labels_present`, `qc_passed`, `label_schema`

### Image order

| Index | Content |
|---|---|
| `[0]~[4]` | Five separate sagittal lumbar MRI segmentation overlays |

### Direct execution

Run this example from the model directory.

```python
from inference import main

images, predictions = main(
    input_data="sample_data/sample.nii.gz",
    model_path="checkpoint",
)
print(predictions)
```

`result/result.json` lists the role-named PNG files from a completed sample inference.
<!-- MAPLE_MANUAL_END -->

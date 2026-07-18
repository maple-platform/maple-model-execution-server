# TotalSegmentator MSK CT

CT segmentation of 74 musculoskeletal structures: vertebrae C1-S1, sacrum, humeri, scapulae, clavicles, femora, hip bones, ribs, sternum, costal cartilage, skull, spinal cord, and major pelvic/paraspinal muscles. The public `main()` return contains per-structure statistics and grouped axial overlays with compact legends. The engine's temporary multilabel NIfTI is not part of the public return.

Path-only input uses the full-resolution public `total` task. The local adapter accepts `{"volume_path": "...", "fast": true}` for lower-resolution inference; exposing that optional dictionary form requires agreement with the platform runner. TotalSegmentator and these task weights are Apache-2.0. This model is for research use and is not a medical device.

The Maple package was reconstructed on four public v2 cases with a macro Dice of 0.9657 across 182 present structure-case pairs. Current release checkpoint directories are named `1559subj` and may include these public cases, so this is an integration/alignment check, not an independent estimate of clinical generalization.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `nii.gz, nii`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `segmentation_overlay`
- **Prediction fields**: `pred`, `pred_name`, `structures_present`, `label_map`, `statistics`, `fast_mode`

### Image order

| Index | Content |
|---|---|
| `[0]~[N]` | Grouped axial overlays for structures visible on the same slice, with a compact color legend overlaid on each image |

### Direct execution

Run this example from the model directory.

```python
from inference import main

images, predictions = main(
    input_data="sample_data/sample_ct.nii.gz",
    model_path="checkpoint",
)
print(predictions)
```

`result/result.json` lists the role-named PNG files from a completed sample inference.
<!-- MAPLE_MANUAL_END -->

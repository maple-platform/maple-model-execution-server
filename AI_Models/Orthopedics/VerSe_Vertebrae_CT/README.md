# VerSe Vertebrae CT

Segments and identifies C1-C7, T1-T12, and L1-L5 from a CT NIfTI volume using the public TotalSegmentator 2.15 model. The public `main()` return contains one overlay per detected vertebra plus level-specific world/voxel centroids, volumes, voxel counts, and mean HU values. The engine's temporary multi-label NIfTI is not part of the public return.

L6 and T13 anatomical variants are not supported by the upstream label set. Full VerSe held-out evaluation is tracked separately; research use only.

<!-- MAPLE_MANUAL_START -->
## Maple Manual Interface

- **Entrypoint**: `main(input_data, model_path)`
- **Input contract**: image or volume file path (`str`)
- **Input extensions**: `nii.gz, nii`
- **Return**: `(np.ndarray | list[np.ndarray], list[dict])`
- **Primary image role**: `segmentation_overlay`
- **Prediction fields**: `pred`, `pred_name`, `vertebrae`, `structures_present`, `fast_mode`, `unsupported_variants`

### Image order

| Index | Content |
|---|---|
| `[0]~[N]` | One CT segmentation overlay per detected vertebra; result filenames identify the vertebral level |

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

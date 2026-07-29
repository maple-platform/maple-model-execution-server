# RETFound DR APTOS2019 + Grad-CAM (Maple submission package)

This package adapts the RETFound-MAE APTOS2019 checkpoint to Maple's researcher
submission contract. It accepts one color fundus photograph and returns:

1. a `224 × 224 × 3`, RGB, `uint8` Grad-CAM overlay; and
2. probabilities for DR grades 0–4 (`No DR` through `Proliferative DR`).

## Scope

Included: single-image inference, APTOS five-grade classification, model caching,
and ViT Grad-CAM. Excluded by project decision: training/fine-tuning, CSV or tabular
input, multimodal fusion, OCT, segmentation, pipelines, runner/server/Docker work,
and platform registration.

## Required external artifacts

The repository does **not** contain a checkpoint or patient image. Obtain the
official fine-tuned APTOS checkpoint from the RETFound benchmark link in
`checkpoint/README.md` and place it at `checkpoint/checkpoint-best.pth`. Add one
legally usable test image under `sample_data/`.

A CC BY 4.0 Wikimedia Commons fundus image is included as a smoke-test input;
see `sample_data/README.md` for attribution. It is not an APTOS validation case.

Only use a checkpoint downloaded from the official source. The loader uses
PyTorch's `weights_only=True` mode and allowlists only the checkpoint's standard
`argparse.Namespace` metadata class.

The five-class mapping follows APTOS2019's diagnosis values:

| Index | Meaning |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Maple entrypoint

```python
from inference import main

overlay, probabilities = main(
    "sample_data/example.png",
    "checkpoint/checkpoint-best.pth",
)
```

The overlay is drawn on the exact center-cropped image seen by the model. It is
an explanatory saliency visualization, not a lesion segmentation or causal proof.
An empty Grad-CAM raises an error instead of returning a normal-looking fallback.

## Validation before submission

```bash
python validate_package.py \
  --image sample_data/fundus_diabetic_retinopathy_ccby4.png \
  --checkpoint checkpoint/checkpoint-best.pth \
  --output validation/gradcam_overlay.png
```

Expected checks: RGB `uint8` output, shape `(224, 224, 3)`, five finite
probabilities summing to approximately 1, and one top prediction.

## License and clinical limitation

RETFound is released under **CC BY-NC 4.0**. This package is therefore restricted
to attributed research/non-commercial use unless separate permission is obtained.
It is not a medical device and must not be used for autonomous diagnosis or
patient-care decisions. External validation, bias analysis, calibration, and
clinical governance remain required.

Sources:

- RETFound: https://github.com/rmaphoh/RETFound
- RETFound paper: https://www.nature.com/articles/s41586-023-06555-x
- APTOS2019 data: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

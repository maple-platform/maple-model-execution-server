# RETFound Glaucoma PAPILA + Grad-CAM (Maple submission package)

This package adapts the RETFound-MAE PAPILA checkpoint to Maple's researcher
submission contract. It accepts one color fundus photograph and returns:

1. a `224 × 224 × 3`, RGB, `uint8` Grad-CAM overlay; and
2. probabilities for 3 glaucoma classes (`Normal`, `Glaucoma suspect`, `Glaucoma`).

Structurally identical to `RETFound_DR_APTOS2019_GradCAM` (same backbone,
same preprocessing, same Grad-CAM method) -- only the fine-tuned checkpoint
and the classification head/labels differ.

## Scope

Included: single-image inference, PAPILA 3-class glaucoma classification,
model caching, and ViT Grad-CAM. Excluded by project decision:
training/fine-tuning, CSV or tabular input, multimodal fusion (PAPILA also
has clinical/demographic data not used here), OCT, segmentation, pipelines,
runner/server/Docker work, and platform registration.

## Required external artifacts

The repository does **not** contain a checkpoint. Obtain the official
fine-tuned PAPILA checkpoint from the RETFound benchmark link in
`checkpoint/README.md` and place it at `checkpoint/checkpoint-best.pth`. Two
real PAPILA test-set images (CC BY 4.0, redistributable with attribution) are
included under `sample_data/` -- unlike the DR package, these are not a
substitute smoke-test image.

Only use a checkpoint downloaded from the official source. The loader uses
PyTorch's `weights_only=True` mode and allowlists only the checkpoint's
standard `argparse.Namespace` metadata class.

The three-class mapping was verified directly against the PAPILA data-split
archive's own folder names (not assumed from the paper):

| Index | Meaning | Source folder name |
|---:|---|---|
| 0 | Normal (non-glaucoma) | `anormal` |
| 1 | Glaucoma suspect | `bsuspectglaucoma` |
| 2 | Glaucoma | `cglaucoma` |

## Maple entrypoint

```python
from inference import main

overlay, probabilities = main(
    "sample_data/papila_test_cglaucoma_RET102OD.jpg",
    "checkpoint/checkpoint-best.pth",
)
```

The overlay is drawn on the exact center-cropped image seen by the model. It
is an explanatory saliency visualization, not a lesion segmentation or causal
proof. An empty Grad-CAM raises an error instead of returning a
normal-looking fallback.

## ⚠️ Known limitation: read the probabilities, not just the top label

Batch validation against the full official PAPILA test split (98 images; see
`IMPLEMENTATION_STATUS.md` for the complete table) found that the "Glaucoma"
class (index 2) **never won the 3-way argmax**, including for all 14
confirmed-glaucoma test images (12 were top-classified as "Normal", 2 as
"Glaucoma suspect"). The class order itself is confirmed correct -- index 2's
*mean* probability is still highest for true glaucoma cases (0.227) versus
the other two classes (0.133, 0.123) -- but that signal is too weak in
absolute terms to ever dominate an argmax, likely reflecting training-set
class imbalance rather than a broken checkpoint or wrong label mapping.

**Practical consequence:** `pred` / `pred_name` / `top_prediction` will
almost never say "Glaucoma" in practice, even for glaucoma patients. Do not
present the top label alone as a glaucoma screening result. The clinically
meaningful signal is the **relative size of the `Glaucoma` class probability
across patients**, not whether it happens to be the single largest of three
numbers for any given patient. Any downstream display of this model's output
should show all three probabilities, and should be interpreted by a clinician
with this calibration behavior in mind.

## Validation before submission

```bash
python validate_package.py \
  --image sample_data/papila_test_cglaucoma_RET102OD.jpg \
  --checkpoint checkpoint/checkpoint-best.pth \
  --output validation/gradcam_overlay.png
```

Expected checks: RGB `uint8` output, shape `(224, 224, 3)`, three finite
probabilities summing to approximately 1, and one top prediction.

## License and clinical limitation

RETFound is released under **CC BY-NC 4.0**. This package is therefore
restricted to attributed research/non-commercial use unless separate
permission is obtained. It is not a medical device and must not be used for
autonomous diagnosis or patient-care decisions. External validation, bias
analysis, calibration, and clinical governance remain required.

Sources:

- RETFound: https://github.com/rmaphoh/RETFound_MAE
- RETFound paper: https://www.nature.com/articles/s41586-023-06555-x
- PAPILA data: https://doi.org/10.6084/m9.figshare.14798004 (CC BY 4.0)
- PAPILA paper: https://doi.org/10.1038/s41597-022-01388-1

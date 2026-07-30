# RETFound_Glaucoma_PAPILA runtime adapter

This folder connects Maple's generic runtime server to:

`AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA/inference.py`

## Files

- `config.yaml`: selects `runtime-medical` and supplies inference/checkpoint paths.
- `runner.py`: calls `main()`, saves the Grad-CAM PNG, and returns JSON-safe
  classification results. Identical structure to
  `RETFound_DR_APTOS2019_GradCAM`'s runner (single-image input), adjusted for
  the 3-class glaucoma head.

## Runner output

| Key | Meaning |
|---|---|
| `model` | Maple model name |
| `image_b64` | Base64-encoded Grad-CAM PNG |
| `output_image_role` | `gradcam_overlay` |
| `predictions` | Ordered classes 0-2 (normal, suspect, glaucoma) with probabilities |
| `top_prediction` | Highest-probability class |
| `gradcam_target` | Class used as the Grad-CAM backward target |
| `output_file` | Saved PNG path inside the runtime |

The ~3.4 GB checkpoint is delivered separately and must appear under the
configured checkpoint directory as `checkpoint-best.pth`.

**Known limitation:** batch validation against the full official PAPILA test
split found the "Glaucoma" class never wins the top-1 argmax, even for
confirmed glaucoma cases (class order is confirmed correct; the probability
signal is real but too small to dominate a 3-way argmax -- likely a
class-imbalance effect). Any UI surfacing this model's output should show
all three class probabilities, not just `top_prediction`. See
`AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA/IMPLEMENTATION_STATUS.md`
for the full validation data.

RETFound is CC BY-NC 4.0 and is limited to research/non-commercial use unless
separate permission is obtained.

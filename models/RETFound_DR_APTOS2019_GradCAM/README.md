# RETFound_DR_APTOS2019_GradCAM runtime adapter

This folder connects Maple's generic runtime server to:

`AI_Models/Ophthalmology/RETFound_DR_APTOS2019_GradCAM/inference.py`

## Files

- `config.yaml`: selects `runtime-medical` and supplies inference/checkpoint paths.
- `runner.py`: calls `main()`, saves the Grad-CAM PNG, and returns JSON-safe
  classification results.

## Runner output

| Key | Meaning |
|---|---|
| `model` | Maple model name |
| `image_b64` | Base64-encoded Grad-CAM PNG |
| `output_image_role` | `gradcam_overlay` |
| `predictions` | Ordered DR grades 0–4 with probabilities |
| `top_prediction` | Highest-probability grade |
| `gradcam_target` | Grade used as the Grad-CAM backward target |
| `output_file` | Saved PNG path inside the runtime |

The 3.64 GB checkpoint is delivered separately and must appear under the
configured checkpoint directory as `checkpoint-best.pth`.

RETFound is CC BY-NC 4.0 and is limited to research/non-commercial use unless
separate permission is obtained.


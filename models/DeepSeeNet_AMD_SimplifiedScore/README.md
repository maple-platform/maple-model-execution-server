# DeepSeeNet_AMD_SimplifiedScore runtime adapter

This folder connects Maple's generic runtime server to:

`AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/inference.py`

## Files

- `config.yaml`: selects `runtime-medical` and supplies inference/checkpoint paths.
- `runner.py`: reads a two-image JSON manifest, calls `main()`, validates the
  result, and returns JSON-safe classification results.

## PROVISIONAL two-image input convention

Unlike the single-file DR/segmentation models already registered, this model
needs two co-registered photos (left eye, right eye) per exam. Until the
platform decides how multi-file exams should reach `POST /run`, `runner.py`
expects `input_path` to point to a small JSON manifest:

```json
{"left_eye": "/app/inputs/left_eye.jpg", "right_eye": "/app/inputs/right_eye.jpg"}
```

Relative paths inside the manifest resolve against the manifest file's own
directory. **This convention is a placeholder for local testing, not a
platform-wide decision** -- see
`AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/IMPLEMENTATION_STATUS.md`.

## Runner output

| Key | Meaning |
|---|---|
| `model` | Maple model name |
| `predictions` | One-element list with the score row (`pred`, `pred_name`, per-eye risk factors) |
| `top_prediction` | Same row (kept for parity with image-model runners) |
| `output_file` | Saved JSON path inside the runtime |

The three ~257 MiB checkpoints are delivered separately and must appear
under the configured checkpoint directory as `drusen_model.h5` /
`pigment_model.h5` / `adv_amd_model.h5`.

DeepSeeNet is public domain (US Government Work); NCBI's own README still
restricts use to research/non-commercial purposes. Not FDA-evaluated; not a
medical device.

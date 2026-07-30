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

## Container deployment status

`config.yaml` currently selects `runtime-medical` for registry consistency,
but the shared image does **not** contain a compatible framework stack.
DeepSeeNet's verified environment requires Python 3.6, TensorFlow 1.15.5,
Keras 2.2.4, NumPy 1.18.5, and Pillow 8.4.0. These legacy dependencies are
incompatible with the current shared PyTorch/medical runtime.

Production enablement requires a dedicated legacy TensorFlow runtime image
with the standard `/app/models`, `/app/AI_Models`, `/app/inputs`, and
`/app/outputs` mounts plus a Gateway runtime mapping. Until then, direct
Python validation is supported but the compose API route is pending.

## Preparing files for the future container path

Place the three checkpoints on the host:

```text
AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/checkpoint/drusen_model.h5
AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/checkpoint/pigment_model.h5
AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/checkpoint/adv_amd_model.h5
```

Prepare the mounted input directory:

```text
inputs/deepseenet_exam/
├── manifest.json
├── left_eye.jpg
└── right_eye.jpg
```

`manifest.json`:

```json
{
  "left_eye": "left_eye.jpg",
  "right_eye": "right_eye.jpg"
}
```

Once the dedicated runtime is available, call:

```bash
curl -sS http://localhost:8110/infer/v2 \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "DeepSeeNet_AMD_SimplifiedScore",
    "input_path": "/app/inputs/deepseenet_exam/manifest.json",
    "output_dir": "/app/outputs/DeepSeeNet_AMD_SimplifiedScore",
    "params": {"timeout": 600}
  }'
```

Both eye paths are mandatory. The current metadata lists image extensions as
the clinical input formats, while the API transport uses a JSON manifest.
This provisional distinction must be replaced by a platform-wide multi-file
exam contract before production exposure.

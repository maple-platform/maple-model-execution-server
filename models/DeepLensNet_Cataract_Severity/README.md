# DeepLensNet_Cataract_Severity runtime adapter

This folder connects Maple's generic runtime server to:

`AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/inference.py`

## Files

- `config.yaml`: selects `runtime-medical` and supplies inference/checkpoint paths.
- `runner.py`: reads a multi-image JSON manifest, calls `main()`, validates the
  result, and returns JSON-safe regression results.

## PROVISIONAL multi-image input convention

Unlike the single-file DR/segmentation models already registered, this model
needs up to three co-registered photos (slit-lamp, anterior/retro,
posterior/retro) per exam. Until the platform decides how multi-file exams
should reach `POST /run`, `runner.py` expects `input_path` to point to a
small JSON manifest, any subset of:

```json
{"ns_image": "/app/inputs/ns.jpg", "cortical_image": "/app/inputs/ant.jpg", "psc_image": "/app/inputs/pos.jpg"}
```

Relative paths inside the manifest resolve against the manifest file's own
directory. **This convention is a placeholder for local testing, not a
platform-wide decision** -- see
`AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/IMPLEMENTATION_STATUS.md`.

## Runner output

| Key | Meaning |
|---|---|
| `model` | Maple model name |
| `predictions` | One-element list with the score row (`pred`, `pred_name`, `ns_score`, `pctcol_score`, `pctpsc_score`) |
| `top_prediction` | Same row (kept for parity with image-model runners) |
| `output_file` | Saved JSON path inside the runtime |

The three ~276 MiB checkpoints are delivered separately and must appear
under the configured checkpoint directory as `NS.h5` / `PCTCOL.h5` /
`PCTPSC.h5`.

No upstream LICENSE file exists; NCBI's README restricts use to
research/non-commercial purposes. Not FDA-evaluated; not a medical device.

## Container deployment status

`config.yaml` currently selects `runtime-medical` so the Gateway can validate
and route the model consistently. That does **not** mean the shared image
contains a compatible framework stack. The model requires Python 3.8,
TensorFlow 2.3.1, Keras 2.4.3, NumPy 1.18.5, and Pillow 7.2.0. The shared
runtime is built for newer platform models, so installing these pins into it
would create dependency conflicts.

Production enablement therefore requires a dedicated legacy TensorFlow
runtime image with the same `/app/models`, `/app/AI_Models`, `/app/inputs`,
and `/app/outputs` mounts. Until that image and a corresponding Gateway
runtime mapping exist, direct Python validation is supported but the
compose-based API path should be considered pending.

## Preparing files for the future container path

Place the three checkpoints on the host:

```text
AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/checkpoint/NS.h5
AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/checkpoint/PCTCOL.h5
AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/checkpoint/PCTPSC.h5
```

Create a directory under the mounted `inputs/` folder:

```text
inputs/deeplensnet_exam/
├── manifest.json
├── ns.jpg
├── cortical.jpg
└── psc.jpg
```

Use relative paths so the same manifest works on the host and in a container:

```json
{
  "ns_image": "ns.jpg",
  "cortical_image": "cortical.jpg",
  "psc_image": "psc.jpg"
}
```

Once the dedicated runtime is available, the Gateway request will be:

```bash
curl -sS http://localhost:8110/infer/v2 \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "DeepLensNet_Cataract_Severity",
    "input_path": "/app/inputs/deeplensnet_exam/manifest.json",
    "output_dir": "/app/outputs/DeepLensNet_Cataract_Severity",
    "params": {"timeout": 600}
  }'
```

The current upload metadata lists JPG/PNG because those are the clinical
source formats. The API transport is nevertheless a JSON manifest. The
platform must add an explicit multi-file exam contract before this
provisional convention is exposed as a stable public API.

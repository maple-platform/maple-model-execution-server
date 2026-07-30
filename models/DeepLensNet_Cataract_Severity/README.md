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

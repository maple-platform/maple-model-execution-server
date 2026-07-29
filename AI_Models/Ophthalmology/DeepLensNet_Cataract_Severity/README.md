# DeepLensNet Cataract Severity (Maple submission package)

This package adapts DeepLensNet's official InceptionV3-based CSV-batch script
to Maple's researcher submission contract. It accepts up to three
anterior-segment eye photographs from one exam and returns continuous
severity scores for the three axes DeepLensNet was trained to quantify.

## Scope

Included: single-exam inference (three-photo input), the three official
InceptionV3 regressors (NS / PCTCOL / PCTPSC), and model caching. Excluded by
project decision: training/fine-tuning, batch/CSV input, attention-map
visualization (the upstream repo ships a separate `attention_map_samples.zip`
that was not adapted here), runner/server/Docker work, and platform
registration.

## Target quantities

| `input_data` key | Source photo | Output field | Meaning / scale |
|---|---|---|---|
| `ns_image` | 45-degree slit-lamp photo | `ns_score` | Nuclear sclerosis, AREDS-like grade (~0.9-7.1; higher = more severe) |
| `cortical_image` | Anterior/retroillumination photo | `pctcol_score` | Cortical lens opacity, % of lens area (0-100) |
| `psc_image` | Posterior/retroillumination photo | `pctpsc_score` | Posterior subcapsular opacity, % of lens area (0-100) |

Any subset of the three keys may be provided; a missing key is scored as
`null` and simply skipped (matches the upstream script's handling of missing
files as `'N/A'`).

## Required external artifacts

The repository does **not** contain the checkpoint. Obtain the official
weights from the NCBI FTP link in `checkpoint/README.md` and place
`NS.h5` / `PCTCOL.h5` / `PCTPSC.h5` in `checkpoint/`. Three sample photos
(one subject, all three views) are included under `sample_data/`, taken
directly from the upstream repository's own test set.

## Maple entrypoint

```python
from inference import main

result = main(
    {
        "ns_image": "sample_data/sample_01_ns_slitlamp.jpg",
        "cortical_image": "sample_data/sample_01_cortical_retro.jpg",
        "psc_image": "sample_data/sample_01_psc_retro.jpg",
    },
    "checkpoint/",
)
print(result)
```

Returns a one-row `pd.DataFrame`:

| Column | Meaning |
|---|---|
| `pred` | Count of axes actually scored (1-3) |
| `pred_name` | Human-readable `"VARIABLE=value; ..."` summary of the scored axes |
| `ns_score` | Nuclear sclerosis grade, or `null` if `ns_image` was not given |
| `pctcol_score` | Cortical opacity %, or `null` if `cortical_image` was not given |
| `pctpsc_score` | PSC opacity %, or `null` if `psc_image` was not given |

DeepLensNet is a three-axis regressor, not a classifier -- there is no single
categorical diagnosis, so `pred`/`pred_name` are a schema-compliant summary
rather than a clinical label. The three `*_score` columns are the real
result.

**Multi-image input note for the administrator:** unlike Maple's usual
single-file contract, this model needs three co-registered photos from the
same exam. See `IMPLEMENTATION_STATUS.md` for the open question on how the
platform's upload flow should deliver three files to one `POST /run` call.

## Validation before submission

Validated locally by calling `main()` directly (see `IMPLEMENTATION_STATUS.md`
for the exact environment, command, and result compared against the upstream
repo's own `output_file.csv`). All three scores matched within ~0.3%.

## License and clinical limitation

No LICENSE file is published upstream. NCBI's README states: *"The
performance characteristics of this product have not been evaluated by the
Food and Drug Administration and is not intended for commercial use or
purposes beyond research use only."* Treat as research/non-commercial use
only; confirm redistribution terms with NCBI before commercial use. See
`NOTICE.md` for the full attribution and clinical-limitation notice.

Sources:

- DeepLensNet: https://github.com/ncbi/deeplensnet
- Paper: https://www.sciencedirect.com/science/article/pii/S0161642021009672

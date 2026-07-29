# DeepSeeNet AMD Simplified Score (Maple submission package)

This package adapts DeepSeeNet's official `DeepSeeNetSimplifiedScore` class to
Maple's researcher submission contract. It accepts one left-eye and one
right-eye color fundus photograph from the same patient and returns the
AREDS Simplified Severity Score for age-related macular degeneration (AMD).

## Scope

Included: single-exam inference (two-photo input) and the three official
risk-factor classifiers behind the AREDS Simplified Severity Score (drusen
size, pigmentary abnormality, advanced AMD) plus the official scoring
formula. Excluded by project decision: training, the separate geographic
atrophy (GA) and central-GA models (upstream tag `0.2`), runner/server/Docker
work, and platform registration.

## Input

| `input_data` key | Meaning |
|---|---|
| `left_eye` | Path to the left eye's color fundus photograph |
| `right_eye` | Path to the right eye's color fundus photograph |

Both are required -- the AREDS Simplified Severity Score is only defined
for a patient with both eyes graded.

## Required external artifacts

The repository does **not** contain the checkpoints. Obtain them from the
official GitHub Release links in `checkpoint/README.md` and place
`drusen_model.h5` / `pigment_model.h5` / `adv_amd_model.h5` in `checkpoint/`.
Two sample photos (the upstream repo's own demo images) are included under
`sample_data/`.

## Maple entrypoint

```python
from inference import main

result = main(
    {"left_eye": "sample_data/left_eye.jpg", "right_eye": "sample_data/right_eye.jpg"},
    "checkpoint/",
)
print(result)
```

Returns a one-row `pd.DataFrame`:

| Column | Meaning |
|---|---|
| `pred` | AREDS Simplified Severity Score (integer 0-5) |
| `pred_name` | `"AREDS Simplified Severity Score = N"` |
| `drusen_left` / `drusen_right` | `small/none` \| `intermediate` \| `large` |
| `pigment_left` / `pigment_right` | `no` \| `yes` (pigmentary abnormality) |
| `advanced_amd_left` / `advanced_amd_right` | `no` \| `yes` |

Score formula (matches upstream `get_simplified_score` exactly): +5 if either
eye has advanced AMD; +1 per eye with a pigmentary abnormality; +1 per eye
with large drusen; +1 more if both eyes have intermediate drusen; capped at 5.

**Multi-image input note for the administrator:** this model needs two
co-registered photos from the same exam, not Maple's usual single-file
contract. See `IMPLEMENTATION_STATUS.md` for the same open upload-flow
question already flagged for `DeepLensNet_Cataract_Severity`.

## Validation before submission

Validated locally by calling `main()` directly against the upstream repo's
own demo images and comparing to its documented example run. Two of three
risk factors (drusen, advanced AMD) matched exactly; the third (pigment,
left eye) landed on the opposite side of an almost perfectly tied decision
boundary (48.9% vs 51.1%) -- see `IMPLEMENTATION_STATUS.md` for the full
analysis, including the raw softmax outputs for both eyes.

## License and clinical limitation

Public domain ("United States Government Work", NCBI). NCBI's README
separately states: *"The performance characteristics of this product have
not been evaluated by the Food and Drug Administration and is not intended
for commercial use or purposes beyond research use only."* See `NOTICE.md`
for the full attribution and clinical-limitation notice.

Sources:

- DeepSeeNet: https://github.com/ncbi-nlp/DeepSeeNet
- Paper: Peng Y, Dharssi S, Chen Q, Keenan T, Agron E, Wong W, Chew E, Lu Z.
  DeepSeeNet: A deep learning model for automated classification of
  patient-based age-related macular degeneration severity from color fundus
  photographs. Ophthalmology. 2019;126(4):565-575.

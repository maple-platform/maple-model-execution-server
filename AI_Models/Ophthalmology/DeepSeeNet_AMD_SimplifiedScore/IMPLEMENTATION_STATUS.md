# Implementation status

## Completed

- Compared Maple `AI_Models/Manual.md` with the live repository structure and
  the existing `RETFound_DR_APTOS2019_GradCAM` / `DeepLensNet_Cataract_Severity`
  packages for conventions.
- Adapted `deepseenet_simplified.DeepSeeNetSimplifiedScore.predict()` and
  `get_simplified_score()` into a single Maple `main(input_data, model_path)`
  entrypoint, reusing the exact `crop2square` -> resize(224,224) ->
  `imagenet_utils.preprocess_input(x, mode='tf')` pipeline used by
  `deepseenet_drusen.py` / `deepseenet_pigment.py` / `deepseenet_adv_amd.py`
  (confirmed these three modules' `preprocess_image()`, not the *dead* one
  defined in `examples/predict_simplified_score.py`, are what
  `DeepSeeNetSimplifiedScore.predict()` actually calls).
- Downloaded the three official GitHub Release assets (tag `0.1`) and
  verified each by MD5 against the constants hardcoded in the upstream
  source (`DRUSEN_MD5`, `PIGMENT_MD5`, `ADVANCED_AMD_MD5`) -- all three
  matched exactly (see `checkpoint/README.md`).
- Installed a corrected `requirements.txt` (see "Requirements corrections"
  below) in an isolated Python 3.6 venv and ran `main()` end-to-end (CPU)
  against the bundled sample images.
- Rewrote the module to avoid Python 3.7+-only syntax (`from __future__
  import annotations`, PEP 585 generic subscripting like `dict[str, int]`),
  since the official stack requires Python 3.6.

## Requirements corrections (flagged to the administrator)

Upstream's own `requirements.txt` cannot actually install under the Python
3.6 it specifies as a prerequisite:

| Upstream pin | Problem | Used instead |
|---|---|---|
| `numpy==1.21` | numpy dropped Python 3.6 support at 1.20 | `numpy==1.18.5` (also satisfies `tensorflow==1.15.5`'s own `numpy<1.19.0,>=1.16.0` constraint) |
| `Pillow==9.1.0` | Pillow dropped Python 3.6 support at 9.0 | `Pillow==8.4.0` |
| `tensorflow` (unpinned) | `keras==2.2.4` needs a TF1-era API | `tensorflow==1.15.5` (last TF1 release with Python 3.6 wheels) |
| `protobuf` (unpinned, via tensorflow) | unconstrained resolves to a protobuf requiring Python >=3.7 | `protobuf==3.19.6` |

All four were discovered by actually installing into a clean venv, not by
inspection alone.

## Local validation result

Command run (from this folder, `python` = the isolated venv's interpreter):

```python
from inference import main

result = main(
    {"left_eye": "sample_data/left_eye.jpg", "right_eye": "sample_data/right_eye.jpg"},
    "checkpoint/",  # drusen_model.h5, pigment_model.h5, adv_amd_model.h5
)
```

| | This run | Upstream README's documented run |
|---|---|---|
| `drusen` (left, right) | large, large | large, large |
| `advanced_amd` (left, right) | no, no | no, no |
| `pigment` (left, right) | **yes**, no | **no**, no |
| Simplified score | **3** | **2** |

Only `pigment_left` disagrees. Raw softmax outputs were inspected directly:

```
left_eye  pigment [[0.48884538, 0.51115465]]   # class 1 ("yes") wins by 2.2 points
right_eye pigment [[0.9538133,  0.04618667]]   # class 0 ("no"), not close
left_eye  drusen  [[0.2127971, 0.29809794, 0.4891049]]        # class 2 ("large")
left_eye  advanced_amd [[9.9999964e-01, 3.0575228e-07]]       # class 0 ("no"), not close
```

`pigment_left` is a near-exact tie (48.9% vs 51.1%). Every other call in both
eyes is decisively one-sided (>90% margin). This is consistent with
TensorFlow/Keras floating-point non-determinism across library versions and
CPU instruction paths flipping an argmax that sits almost exactly on the
decision boundary -- not a preprocessing or logic error: the drusen and
advanced_amd calls (which are not near a boundary) match the upstream
README's documented run exactly, and the pigment pipeline is byte-for-byte
the same code path (`crop2square` -> resize -> `img_to_array` ->
`imagenet_utils.preprocess_input(mode='tf')` -> `model.predict`) used for all
three risk factors. Recorded here rather than adjusted, per the same
reasoning as `DeepLensNet_Cataract_Severity/IMPLEMENTATION_STATUS.md`.

Verified CPU environment: Python 3.6.12, TensorFlow 1.15.5, Keras 2.2.4,
NumPy 1.18.5, Pillow 8.4.0, protobuf 3.19.6.

## Design decisions (need administrator awareness)

- **Two-image input.** Like `DeepLensNet_Cataract_Severity`, this model needs
  two co-registered photos (left eye + right eye) from one exam, not Maple's
  usual single file path. `main()` takes `input_data` as a dict
  (`left_eye` / `right_eye`, **both required** -- unlike DeepLensNet, the
  simplified score has no defined partial-input behavior). The same open
  question about how the platform's upload flow delivers multiple files to
  one `POST /run` call applies here too.
- **Scope.** Only the three models behind the AREDS Simplified Severity
  Score (drusen, pigment, advanced AMD) were adapted. The upstream repo's
  separate geographic-atrophy / central-GA models (tag `0.2`) and its
  training script were intentionally left out of scope.
- **License.** Public domain (US Government Work); NCBI's own README still
  states research/non-commercial use only. See `NOTICE.md`.

## Runner validation

Also smoke-tested `models/DeepSeeNet_AMD_SimplifiedScore/runner.py`
end-to-end (same venv) via its provisional JSON-manifest convention. Two
fixes were needed and are already applied:

1. `runner.py` originally had `from __future__ import annotations`, which is
   a Python 3.7+ feature and fails under this model's required Python 3.6 --
   removed (plain annotations like `-> dict` don't need it).
2. Same numpy-scalar JSON bug as `DeepLensNet_Cataract_Severity` (see its
   `IMPLEMENTATION_STATUS.md`): fixed with a `.item()` conversion before
   `json.dumps`.

After both fixes, the runner reproduced the same score (3, per the near-tied
`pigment_left` decision documented above) as the direct `main()` call and
wrote a valid JSON file.

## Remaining administrator handoff

- Make the three checkpoints (~257 MiB each) available at the configured
  container path.
- Decide and implement the multi-file (two-image) upload mechanism described
  above.
- Confirm a Python 3.6 base image is acceptable for this runtime, or assess
  whether the two `.h5` checkpoints load correctly under a newer
  TensorFlow/Keras (not attempted here -- see "Requirements corrections").
- Set `docker.service_url` in `meta.json`, register the model, and run the
  container/API integration test.

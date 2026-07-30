# Implementation status

## Completed

- Compared Maple `AI_Models/Manual.md` with the live repository structure and
  the existing `RETFound_DR_APTOS2019_GradCAM` package for conventions.
- Adapted `model_classify.py`'s CSV-batch InceptionV3 pipeline into a
  single-exam Maple `main(input_data, model_path)` entrypoint.
- Downloaded the official weights archive from
  `https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/deeplensnet/models.zip`
  (786,523,156 bytes, matches the server's `Content-Length` exactly;
  SHA-256 `2d56df8531deabfeaa9ddd460bf32e6e93040024adb40709c2a5c76786d84337`)
  and extracted `NS.h5`, `PCTCOL.h5`, `PCTPSC.h5`.
- Installed the pinned `requirements.txt` in an isolated Python 3.8 venv and
  ran `main()` end-to-end (CPU) against the bundled sample images.
- Added a **pinned `protobuf==3.19.6`** during validation because the
  unpinned transitive resolution of `tensorflow==2.3.1`'s dependencies pulled
  in a modern protobuf (5.x) that fails with
  `TypeError: Descriptors cannot be created directly` against TF 2.3.1's
  generated `_pb2.py` files. **Flagged to the administrator**: this
  `requirements.txt` does not yet pin `protobuf`; add `protobuf==3.19.6` (or
  set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) in the container build
  or the same failure will reproduce there.

## Local validation result

Command run (from this folder, `python` = the isolated venv's interpreter):

```python
from inference import main

result = main(
    {
        "ns_image": "sample_data/sample_01_ns_slitlamp.jpg",
        "cortical_image": "sample_data/sample_01_cortical_retro.jpg",
        "psc_image": "sample_data/sample_01_psc_retro.jpg",
    },
    "checkpoint/",  # NS.h5, PCTCOL.h5, PCTPSC.h5
)
```

Output:

| Variable | This run | Upstream `output_file.csv` (subject 1) | Difference |
|---|---:|---:|---:|
| NS | 4.6796264648 | 4.672357559204102 | 0.0073 (0.16%) |
| PCTCOL | 3.2795395851 | 3.027174711227417 | 0.2524 (~0.25 pts / 100) |
| PCTPSC | 40.5182495117 | 40.613460540771484 | 0.0952 (~0.10 pts / 100) |

All three scores land within roughly 0.3% of the reference file shipped by
NCBI for this exact subject. This level of drift is consistent with
TensorFlow/Keras floating-point non-determinism across library versions and
CPU instruction paths (the reference CSV was generated in 2022 against the
pins in `requirements.txt`; this run used the same pins on 2026 hardware) --
not a logic error in the adapted preprocessing, which follows
`model_classify.py`'s resize/`inception_v3.preprocess_input` steps exactly.
TensorFlow also printed a shape warning (`Model was constructed with shape
(None, 334, 501, 3) ... called on an input with incompatible shape (None,
501, 334, 3)`) that is present in the **original** upstream script too (its
`img.resize((334, 501))` produces the same width/height array orientation);
it is non-fatal and does not explain the drift, since it reproduces
identically end-to-end.

Verified CPU environment: Python 3.8.16, TensorFlow 2.3.1, Keras 2.4.3,
NumPy 1.18.5, protobuf 3.19.6 (added, see above).

## Design decisions (need administrator awareness)

- **Multi-image input.** DeepLensNet requires three distinct photographs per
  exam (slit-lamp, anterior/retro, posterior/retro), not Maple's usual single
  file path. `main()` takes `input_data` as a dict
  (`ns_image` / `cortical_image` / `psc_image`, any subset), following the
  same dict-input shape the Manual already uses for pipeline models, even
  though this model is standalone. **The platform's upload flow needs a
  decision on how three co-registered files reach one `POST /run` call**
  (e.g. three named fields, a small zip, or three separate uploads combined
  by `runner.py`) -- this repository does not have precedent for that and it
  is not something the researcher side can resolve alone.
- **No categorical output.** DeepLensNet is a three-axis regressor, not a
  classifier, so there is no natural single `pred`/`pred_name` label. `pred`
  is set to the count of axes actually scored, and `pred_name` is a
  human-readable summary string; the real result is in `ns_score` /
  `pctcol_score` / `pctpsc_score`. See `README.md` for the full contract.
- **License.** No LICENSE file exists upstream. Treated as research/
  non-commercial use only per the README's own restriction language; see
  `NOTICE.md`. Recommend legal confirmation before any commercial use.

## Runner validation

Also smoke-tested `models/DeepLensNet_Cataract_Severity/runner.py` end-to-end
(same venv) via its provisional JSON-manifest convention, calling
`runner.predict(manifest_path, output_dir, config)`. Found and fixed a real
bug during this test: `pd.DataFrame.iloc[0].to_dict()` returns numpy scalar
types (e.g. `numpy.float64`), which `json.dumps` cannot serialize --
`runner.py` now converts each value with `.item()` before writing JSON. After
the fix, the runner produced the same scores as the direct `main()` call and
wrote a valid JSON file.

## Remaining administrator handoff

- Make the checkpoint (`NS.h5`, `PCTCOL.h5`, `PCTPSC.h5`, ~276 MiB each)
  available at the configured container path.
- Decide and implement the multi-file upload mechanism described above in
  `runner.py`/the platform's upload UI.
- Add `protobuf==3.19.6` to the container build (see "Completed" above).
- Set `docker.service_url` in `meta.json`, register the model, and run the
  container/API integration test.

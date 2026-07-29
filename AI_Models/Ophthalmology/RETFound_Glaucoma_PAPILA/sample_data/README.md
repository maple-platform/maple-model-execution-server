# Sample input

Two images taken directly from the PAPILA dataset's own test split (not a
substitute image, unlike the DR package -- PAPILA is published under CC BY 4.0
which permits redistribution with attribution):

| File | PAPILA class | Original path in the data split |
|---|---|---|
| `papila_test_cglaucoma_RET102OD.jpg` | Glaucoma | `PAPILA/test/cglaucoma/RET102OD.jpg` |
| `papila_test_anormal_RET199OD.jpg` | Normal | `PAPILA/test/anormal/RET199OD.jpg` |

Source and attribution:

- Kovalyk, O., Morales-Sánchez, J., Verdú-Monedero, R. et al. *PAPILA:
  Dataset with fundus images and clinical data of both eyes of the same
  patient for glaucoma assessment.* Scientific Data 9, 291 (2022).
  https://doi.org/10.1038/s41597-022-01388-1
- Data: https://doi.org/10.6084/m9.figshare.14798004 (CC BY 4.0)
- Data split used: the same PAPILA.zip referenced in RETFound_MAE's
  `BENCHMARK.md` (https://github.com/rmaphoh/RETFound_MAE)

These are real PAPILA test-set images with known ground truth (per their
folder placement), unlike the DR package's smoke-test image. Still, treat any
local inference result as a package sanity check, not a validated benchmark
run -- the official benchmark numbers come from evaluating the full test
split, not one image per class.

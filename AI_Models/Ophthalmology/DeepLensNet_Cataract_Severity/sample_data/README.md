# Sample input

Three photographs of the same subject (subject 1 in the upstream repo's
`image_set/`), one per DeepLensNet axis:

| File | Role | `input_data` key |
|---|---|---|
| `sample_01_ns_slitlamp.jpg` | 45-degree slit-lamp photo | `ns_image` |
| `sample_01_cortical_retro.jpg` | Anterior/retroillumination photo | `cortical_image` |
| `sample_01_psc_retro.jpg` | Posterior/retroillumination photo | `psc_image` |

Source: https://github.com/ncbi/deeplensnet (`image_set/1_NS.jpg`,
`image_set/1_ANT.jpg`, `image_set/1_POS.jpg`), shipped by NCBI in the upstream
repository specifically as its own test/validation samples. No separate image
license is stated beyond the repository's overall "For Research Use Only"
notice (see `../README.md`).

Expected scores for this subject, from the upstream repo's own
`output_file.csv` (row `ID=1`), usable to confirm a correct local setup:

| Variable | Expected value |
|---|---|
| `NS` | 4.672357559204102 |
| `PCTCOL` | 3.027174711227417 |
| `PCTPSC` | 40.613460540771484 |

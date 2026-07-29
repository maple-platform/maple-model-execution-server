# Checkpoint placement

Place the official DeepSeeNet risk-factor models here:

- `drusen_model.h5`
- `pigment_model.h5`
- `adv_amd_model.h5`

Official source (GitHub Release assets, tag `0.1`):

- `https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/drusen_model.h5`
- `https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/pigment_model.h5`
- `https://github.com/ncbi-nlp/DeepSeeNet/releases/download/0.1/adv_amd_model.h5`
- Repository: https://github.com/ncbi-nlp/DeepSeeNet

Verified during packaging (sizes and MD5 match the constants hardcoded in
the upstream source, e.g. `deepseenet/deepseenet_drusen.py`'s `DRUSEN_MD5`):

| File | Size (bytes) | MD5 |
|---|---:|---|
| `drusen_model.h5` | 269,139,776 | `997a8229f972482e127e8a32d1967549` |
| `pigment_model.h5` | 269,138,496 | `e38f60fa9c0fc6cd7a5022b07b722927` |
| `adv_amd_model.h5` | 269,138,496 | `0adbf448491ead63ac384da671c4f7ee` |

Not bundled here because of size. Only use copies obtained from the official
GitHub Release above (or via the upstream package's own `keras.utils.get_file`
auto-download, which uses the same URLs and hashes).

Two additional models exist upstream (tag `0.2`: `ga_model.h5` for geographic
atrophy, `cga_model.h5` for central geographic atrophy) but are **out of
scope** for this package -- they are not part of the AREDS Simplified
Severity Score computed here. See `README.md` for scope notes.

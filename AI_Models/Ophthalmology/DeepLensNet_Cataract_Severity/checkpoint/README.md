# Checkpoint placement

Place the official DeepLensNet checkpoints here as three separate files:

- `NS.h5`
- `PCTCOL.h5`
- `PCTPSC.h5`

Official source:

- Download: https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/deeplensnet/models.zip
- Repository: https://github.com/ncbi/deeplensnet

Verified during packaging:

- Archive: `models.zip`, 786,523,156 bytes, `Last-Modified: 2022-01-19` (via HTTP
  `Content-Length`/`Last-Modified` headers from the URL above)
- SHA-256 and local load/validation results: see `IMPLEMENTATION_STATUS.md` in
  the parent folder once local validation has run.

The checkpoint is not bundled in this repository because its redistribution
terms are not formally stated upstream (no LICENSE file; see `../README.md`
for the license caveat) and because of its size. Only use a copy obtained from
the official NCBI FTP link above.

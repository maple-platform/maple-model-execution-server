# Checkpoint placement

Place the official fine-tuned RETFound-MAE PAPILA checkpoint here as
`checkpoint-best.pth`. The loader also accepts a direct `.pth`/`.pt` path.

Official benchmark links:

- Checkpoint: https://drive.google.com/drive/folders/1cHOX6C4NQVi9B6n-7Bxxg7b4-wdI4c73
- Benchmark instructions: https://github.com/rmaphoh/RETFound_MAE/blob/main/BENCHMARK.md
- Data split used for fine-tuning (to confirm class order yourself if desired):
  https://drive.google.com/file/d/1JltYs7WRWEU0yyki1CQw5-10HEbqCMBE/view

Do not substitute the foundation-only pretrained weight: this package expects
a fine-tuned 3-class head. Only load a checkpoint obtained from a trusted
source. The official training checkpoint contains `argparse.Namespace`
metadata. The loader keeps `weights_only=True` and allowlists only that
standard metadata class.

Verified official file:

- Google Drive file ID: `1CraCqBclTSCSNzn0jogyIqjBNYcep9rx`
- Size: `3,639,964,135` bytes (~3.39 GiB), modified 2024-01-27
- SHA-256: `2a7040436c3b5b9f25bdb45cb047e34397f9b6e4f207d946660e25b81865f97b`
- Model head: `(3, 1024)`, input size `224`, global pool enabled (architecture
  load/shape check pending -- see IMPLEMENTATION_STATUS.md)

Class order (verified directly against the PAPILA data-split archive's
folder names, not assumed): the split folders are named `anormal`,
`bsuspectglaucoma`, `cglaucoma` -- lettered `a`/`b`/`c` specifically so that
`torchvision.datasets.ImageFolder`'s alphabetical class-to-index assignment
produces 0=normal, 1=suspect, 2=glaucoma.

The checkpoint is not bundled because its redistribution terms and size must
be confirmed by the submitting researcher.

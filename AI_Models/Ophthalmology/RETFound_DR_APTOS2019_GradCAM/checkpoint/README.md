# Checkpoint placement

Place the official fine-tuned RETFound-MAE APTOS2019 checkpoint here as
`checkpoint-best.pth`. The loader also accepts a direct `.pth`/`.pt` path.

Official benchmark links:

- Checkpoints: https://drive.google.com/drive/folders/16kL5V-1U7ACc-68PSHjAq6vyXRJvUoq3
- Benchmark instructions: https://github.com/rmaphoh/RETFound/blob/main/BENCHMARK.md

Do not substitute the foundation-only pretrained weight: this package expects a
fine-tuned 5-class head. Only load a checkpoint obtained from a trusted source.
The official training checkpoint contains `argparse.Namespace` metadata. The
loader keeps `weights_only=True` and allowlists only that standard metadata class.

Verified official file:

- Google Drive file ID: `1Ujzb6Xd1naWC0NngHah-DbHSgqxOiyJX`
- Size: `3,639,988,647` bytes
- SHA-256: `A96B9DBCB78EFF373912FB0A48D316DC35FFB0715BB2A1B9128B096D780EDBBF`
- Model head: `(5, 1024)`, input size `224`, global pool enabled, epoch `27`

The checkpoint is not bundled because its redistribution terms and size must be
confirmed by the submitting researcher.

# Implementation status

## Completed

- Compared Maple `AI_Models/Manual.md` with the live repository structure and
  the existing `RETFound_DR_APTOS2019_GradCAM` package for conventions.
- Verified the PAPILA class order directly against the actual RETFound
  benchmark data-split archive (`PAPILA.zip` from `BENCHMARK.md`), not
  assumed: folder names `anormal` / `bsuspectglaucoma` / `cglaucoma` are
  deliberately lettered so `torchvision.datasets.ImageFolder`'s alphabetical
  class-to-index assignment produces 0=normal, 1=suspect, 2=glaucoma.
- Downloaded the official PAPILA checkpoint from the Google Drive link in
  `BENCHMARK.md` and verified its size (3,639,964,135 bytes) and computed its
  SHA-256 (`2a7040436c3b5b9f25bdb45cb047e34397f9b6e4f207d946660e25b81865f97`,
  recorded in `checkpoint/README.md`).
- Downloaded two real PAPILA test-set images (CC BY 4.0) directly from the
  same data-split archive for `sample_data/`, with known ground truth from
  their folder placement (`cglaucoma`/`anormal`).
- Wrote `inference.py` / `model_def.py` as a direct adaptation of
  `RETFound_DR_APTOS2019_GradCAM`'s package (identical backbone/preprocessing/
  Grad-CAM method), changing only the label count/names and checkpoint.

## Local validation (completed after moving work to a drive with headroom)

The development machine's C: drive dropped to ~2 GB free after downloading
the 3.39 GB checkpoint. Work was moved to a D: drive location
(`D:\maple_work\`) with 100+ GB free, where `torch==2.5.1` /
`torchvision==0.20.1` / `timm==0.9.2` (Python 3.10) were installed and the
package was fully exercised.

**Structural check passed:** `checkpoint-best.pth` loads into `RETFound_mae`
with a 3-class head and no missing/unexpected keys (the same strict check
`RETFound_DR_APTOS2019_GradCAM/inference.py` performs), confirming this is
genuinely a fine-tuned PAPILA checkpoint, not the foundation-only backbone.

**Class-order check: batch-verified across the full official test split**
(98 images: 67 `anormal`, 17 `bsuspectglaucoma`, 14 `cglaucoma`), not just the
two bundled samples. Mean softmax probability per true folder:

| True folder (n) | mean P(index 0) | mean P(index 1) | mean P(index 2) |
|---|---:|---:|---:|
| `anormal` (67) | **0.751** | 0.116 | 0.133 |
| `bsuspectglaucoma` (17) | 0.275 | **0.602** | 0.123 |
| `cglaucoma` (14) | 0.579 | 0.194 | **0.227** |

Index 1 ("suspect") is unambiguous: it is the clear maximum only for its own
true class and stays low elsewhere. Index 0 ("normal") and index 2
("glaucoma") are directionally correct -- each is highest, on average, for
its own true class relative to the other two -- which corroborates the
`anormal`/`bsuspectglaucoma`/`cglaucoma` folder-name-order verification
already done against the data-split archive. **The class-order mapping in
`inference.py`'s `LABELS` (0=normal, 1=suspect, 2=glaucoma) is treated as
confirmed** on this basis.

**Important limitation found -- argmax essentially never selects "Glaucoma":**
Across all 98 test images, index 2 won the argmax (was the single highest
probability) **zero times**, including for all 14 true `cglaucoma` images (12
were argmax-classified as "Normal", 2 as "Glaucoma suspect"). Full per-folder
argmax tally:

| True folder (n) | argmax=index0 | argmax=index1 | argmax=index2 |
|---|---:|---:|---:|
| `anormal` (67) | 62 | 5 | 0 |
| `bsuspectglaucoma` (17) | 5 | 12 | 0 |
| `cglaucoma` (14) | 12 | 2 | 0 |

So while index 2's *mean probability* is directionally higher for true
glaucoma cases (supporting the label-order conclusion above), it is never
large enough in absolute terms to win a 3-way argmax -- meaning `pred`/
`top_prediction` will almost never say "Glaucoma" in practice, even for
confirmed glaucoma patients. RETFound's reported ~0.85 AUC for PAPILA is a
ranking metric computed from the continuous probability, which this data is
still consistent with (the P(index2) ranking is real, just not large enough
in absolute value to dominate an argmax against a probably-imbalanced
training distribution). **Decision (user-confirmed): ship as-is, framed
around the continuous probabilities rather than the top label** -- see the
"Known limitation" section in `README.md` and `NOTICE.md`. Do not present
`top_prediction` alone as a glaucoma screening result; the per-class
probabilities (especially the `Glaucoma` probability trend) are the
meaningful output.

**Grad-CAM/output contract:** verified via `validate_package.py` on both
bundled sample images -- RGB `uint8`, shape `(224, 224, 3)`, three finite
probabilities summing to ~1, exactly one `is_top_prediction=True`. Passed.

Verified environment: Python 3.10.13, PyTorch 2.5.1+cpu, torchvision
0.20.1+cpu, timm 0.9.2, NumPy 1.26.4, OpenCV 4.9.0, CPU-only.

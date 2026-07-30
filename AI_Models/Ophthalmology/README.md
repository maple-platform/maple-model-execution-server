# Ophthalmology models — sample data & quick test guide

Four models are currently submitted under this department. Each folder is a
self-contained Maple package (see `../Manual.md`); this file is just a quick
index of what sample images exist and the exact `main()` call to try each
model once its checkpoint is in place.

**Checkpoints are not in this repo** (see each model's `checkpoint/README.md`
for the official download link, verified size, and hash). Everything below
assumes you've already placed the checkpoint under `checkpoint/` in that
model's folder.

| Model | Disease / body part | Input | Sample images |
|---|---|---|---|
| `RETFound_DR_APTOS2019_GradCAM` | Diabetic retinopathy (posterior) | 1 fundus photo | `sample_data/fundus_diabetic_retinopathy_ccby4.png` |
| `DeepSeeNet_AMD_SimplifiedScore` | AMD (posterior) | 2 fundus photos (left + right eye) | `sample_data/left_eye.jpg`, `right_eye.jpg` |
| `RETFound_Glaucoma_PAPILA` | Glaucoma (posterior) | 1 fundus photo | `sample_data/papila_test_cglaucoma_RET102OD.jpg`, `papila_test_anormal_RET199OD.jpg` |
| `DeepLensNet_Cataract_Severity` | Cataract (anterior) | up to 3 anterior-segment photos | `sample_data/sample_01_ns_slitlamp.jpg`, `sample_01_cortical_retro.jpg`, `sample_01_psc_retro.jpg` |

## 1. RETFound_DR_APTOS2019_GradCAM

Single image path in, `(gradcam_overlay, predictions[5])` out.

```python
import sys
sys.path.insert(0, "AI_Models/Ophthalmology/RETFound_DR_APTOS2019_GradCAM")
from inference import main

overlay, predictions = main(
    "AI_Models/Ophthalmology/RETFound_DR_APTOS2019_GradCAM/sample_data/fundus_diabetic_retinopathy_ccby4.png",
    "AI_Models/Ophthalmology/RETFound_DR_APTOS2019_GradCAM/checkpoint/checkpoint-best.pth",
)
```

## 2. DeepSeeNet_AMD_SimplifiedScore

Both eyes required (dict input) -- see the model's own `README.md` for the
"Glaucoma"-style limitation note that does **not** apply here; this one is
well-behaved.

```python
import sys
sys.path.insert(0, "AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore")
from inference import main

result_df = main(
    {
        "left_eye": "AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/sample_data/left_eye.jpg",
        "right_eye": "AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/sample_data/right_eye.jpg",
    },
    "AI_Models/Ophthalmology/DeepSeeNet_AMD_SimplifiedScore/checkpoint",
)
```

## 3. RETFound_Glaucoma_PAPILA

Single image path in, `(gradcam_overlay, predictions[3])` out. ⚠️ Read
`RETFound_Glaucoma_PAPILA/README.md`'s "Known limitation" section before
interpreting the output -- the top-1 label almost never says "Glaucoma"; the
per-class probabilities are the meaningful signal.

```python
import sys
sys.path.insert(0, "AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA")
from inference import main

overlay, predictions = main(
    "AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA/sample_data/papila_test_cglaucoma_RET102OD.jpg",
    "AI_Models/Ophthalmology/RETFound_Glaucoma_PAPILA/checkpoint/checkpoint-best.pth",
)
```

## 4. DeepLensNet_Cataract_Severity

Any subset of the three photo keys (dict input); all three are provided in
`sample_data/`.

```python
import sys
sys.path.insert(0, "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity")
from inference import main

result_df = main(
    {
        "ns_image": "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/sample_data/sample_01_ns_slitlamp.jpg",
        "cortical_image": "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/sample_data/sample_01_cortical_retro.jpg",
        "psc_image": "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/sample_data/sample_01_psc_retro.jpg",
    },
    "AI_Models/Ophthalmology/DeepLensNet_Cataract_Severity/checkpoint",
)
```

## Sample image provenance at a glance

| File(s) | Source | License |
|---|---|---|
| `fundus_diabetic_retinopathy_ccby4.png` | Wikimedia Commons (smoke-test only, not an APTOS case) | CC BY 4.0 |
| `left_eye.jpg`, `right_eye.jpg` | ncbi-nlp/DeepSeeNet's own demo images | Public domain (US Government Work) |
| `papila_test_*.jpg` | Real PAPILA test-set images, own ground truth | CC BY 4.0 |
| `sample_01_*.jpg` | ncbi/deeplensnet's own test images (subject 1) | No formal license; NCBI research-use-only notice |

Each model's own `sample_data/README.md` has the full attribution and, where
applicable, the expected/reference output for that exact sample.

## Requirements note

Each model pins a **different** Python/framework stack (see each
`requirements.txt`) -- they are not designed to run in the same virtual
environment:

| Model | Python | Framework |
|---|---|---|
| `RETFound_DR_APTOS2019_GradCAM` | 3.11+ | PyTorch 2.5.1 / timm |
| `DeepSeeNet_AMD_SimplifiedScore` | 3.6 | TensorFlow 1.15.5 / Keras 2.2.4 |
| `RETFound_Glaucoma_PAPILA` | 3.11+ | PyTorch 2.5.1 / timm |
| `DeepLensNet_Cataract_Severity` | 3.8 | TensorFlow 2.3.1 / Keras 2.4.3 |

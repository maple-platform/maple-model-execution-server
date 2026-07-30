# Attribution notice

This package contains an adapted, inference-only model definition derived from:

- RETFound (RETFound_MAE), rmaphoh et al.: https://github.com/rmaphoh/RETFound_MAE
- Paper: Zhou et al., *A foundation model for generalizable disease detection
  from retinal images*, Nature 622, 156–163 (2023).
- Fine-tuning benchmark and checkpoint: `BENCHMARK.md` in the above repository.

The upstream code and model are licensed under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0):
https://creativecommons.org/licenses/by-nc/4.0/

The fine-tuning dataset, PAPILA, is separately published under CC BY 4.0:

- Kovalyk, O., Morales-Sánchez, J., Verdú-Monedero, R. et al. *PAPILA:
  Dataset with fundus images and clinical data of both eyes of the same
  patient for glaucoma assessment.* Scientific Data 9, 291 (2022).
  https://doi.org/10.1038/s41597-022-01388-1
- Data: https://doi.org/10.6084/m9.figshare.14798004

Local changes isolate the RETFound-MAE ViT inference definition (identical to
`RETFound_DR_APTOS2019_GradCAM/model_def.py`), add the Maple entrypoint for a
3-class PAPILA glaucoma head, and implement ViT Grad-CAM output. The class
order (0=normal, 1=glaucoma suspect, 2=glaucoma) was verified directly against
the actual PAPILA data-split archive's folder structure (`anormal` /
`bsuspectglaucoma` / `cglaucoma`, deliberately lettered to fix
`torchvision.datasets.ImageFolder`'s alphabetical class-to-index assignment)
rather than assumed. No endorsement by the original authors is implied.

## Clinical limitation

This is not a medical device and must not be used for autonomous diagnosis or
patient-care decisions. External validation, bias analysis, calibration, and
clinical governance remain required.

**Specific finding from local validation (see `IMPLEMENTATION_STATUS.md`):**
across the full official PAPILA test split (98 images), this checkpoint's
"Glaucoma" class probability never won the 3-way argmax, including for all 14
confirmed-glaucoma test images. The class order is confirmed correct (the
"Glaucoma" probability is still highest, on average, for true glaucoma cases
relative to the other two classes), but this signal is not large enough in
absolute terms to dominate a hard top-1 decision -- likely a class-imbalance
calibration effect, not a labeling or loading error. Treat the per-class
probabilities as a relative risk signal, not the top-1 label as a diagnosis.

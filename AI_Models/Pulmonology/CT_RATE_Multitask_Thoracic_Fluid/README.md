# CT_RATE_Multitask_Thoracic_Fluid

CT-RATE Pleural + Pericardial effusion classifier (multitask__thoracic_fluid).

- Source checkpoint: `044_multitask__thoracic_fluid`
- Family: `multitask`
- Target: `Pleural + Pericardial effusion`
- Validation: macro_auroc=0.8638, macro_auprc=0.4998
- Decision threshold: `0.5`

The Maple entry point is `main(input_data, model_path)`. It accepts a `.nii` or
`.nii.gz` CT volume and returns an RGB axial Grad-CAM montage plus a list of
classification probabilities. Preprocessing is identical to the CT-RATE
end-to-end training pipeline: SPL orientation, HU clipping/scaling, foreground
crop, and resize to `128 x 256 x 256`.

Grad-CAM is a coarse model-attention explanation, not lesion segmentation. The
reported validation results are internal CT-RATE results and are not external or
prospective clinical validation. This package is for research use only.

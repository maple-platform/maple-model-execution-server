# CT_RATE_Official_Interlobular_Septal_Thickening

CT-RATE Interlobular septal thickening classifier (official__interlobular_septal_thickening).

- Source checkpoint: `017_official__interlobular_septal_thickening`
- Family: `official_18`
- Target: `Interlobular septal thickening`
- Validation: auroc=0.7930, auprc=0.2594
- Decision threshold: `0.5`

The Maple entry point is `main(input_data, model_path)`. It accepts a `.nii` or
`.nii.gz` CT volume and returns an RGB axial Grad-CAM montage plus a list of
classification probabilities. Preprocessing is identical to the CT-RATE
end-to-end training pipeline: SPL orientation, HU clipping/scaling, foreground
crop, and resize to `128 x 256 x 256`.

Grad-CAM is a coarse model-attention explanation, not lesion segmentation. The
reported validation results are internal CT-RATE results and are not external or
prospective clinical validation. This package is for research use only.

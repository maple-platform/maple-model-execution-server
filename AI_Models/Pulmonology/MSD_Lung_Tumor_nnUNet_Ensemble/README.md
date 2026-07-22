# MSD_Lung_Tumor_nnUNet_Ensemble

Five-fold nnU-Net ensemble trained on 63 labeled MSD Task06 Lung CT scans. Cross-validation mean foreground Dice: 0.6070. Input must be a thoracic CT NIfTI with valid spatial metadata. The model returns the complete spatially referenced 3D tumor mask as one compressed NIfTI file and up to eight independent representative axial overlays spanning the predicted tumor extent. Research use only.

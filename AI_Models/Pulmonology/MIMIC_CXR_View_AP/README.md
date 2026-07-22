# MIMIC_CXR_View_AP

Classifies whether a MIMIC-CXR chest radiograph uses AP rather than PA projection.

- Target: `View AP`
- Architecture: `efficientnet_b0`
- Tier: 1
- Threshold: 0.07
- Internal test AUROC/AUPRC/F1: 0.998 / 0.999 / 0.993

The Maple entry point is `main(input_data, model_path)`. It accepts one PNG/JPG
chest radiograph and returns `(gradcam_overlay, predictions)`. The overlay is an
RGB uint8 image and `predictions` contains one classification result with
probability, threshold, model provenance, and internal metrics.

Preprocessing exactly matches training evaluation: grayscale decode, direct
224 x 224 `INTER_AREA` resize, `(x / 255 - 0.5) / 0.25` normalization, and
three-channel repetition. DICOM is not accepted directly.

Grad-CAM uses an architecture-specific final feature stage: `denseblock4` for
DenseNet121, `layer4[-1]` for ResNet34/50, and `features[-1]` for EfficientNet-B0.
It backpropagates the positive logit for a positive decision and the negative
logit (`-logit`) for a negative decision, so the overlay explains the thresholded
predicted class. Grad-CAM is not validated lesion localization or segmentation.

Metrics and thresholds are from the internal MIMIC-CXR official split. This
research model has not undergone external or prospective clinical validation and
must not be used for diagnosis or as the sole basis for clinical decisions.

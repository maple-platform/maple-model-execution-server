# ISIC2019_ViT3

A supervised ViT classifier with three ISIC-derived diagnostic groups.

## Input and output

Pass an absolute PNG/JPEG path to `main(input_data, model_path)`; pass the extracted checkpoint directory as `model_path`. The returned tuple is `[0]` an RGB `uint8` Grad-CAM overlay and `[1]` three class probabilities in descending order. Zero-shot scores, where applicable, are relative to configured candidates and are not calibrated disease probabilities.

## Local smoke test

```bash
python inference.py
```

The bundled sample is `sample_data/sample_melanoma.jpg`. Model weights belong in `checkpoint/`; large weights and sample data are delivered separately from Git according to the platform manual.

## Limitations

This research model is not a medical device and must not be used as a stand-alone diagnosis. Validate preprocessing, labels, calibration, population shift, and localization behavior on the target clinical cohort.

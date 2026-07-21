# DermFoundation_SCIN

Google Derm Foundation embeddings with the locally trained SCIN common-13 logistic classifier.

## Input and output

Pass an absolute PNG/JPEG path to `main(input_data, model_path)`; pass the extracted checkpoint directory as `model_path`. The returned tuple is `[0]` an RGB `uint8` 3×3 occlusion-sensitivity overlay and `[1]` thirteen class probabilities in descending order. `occlusion_overlay` is a new platform role and requires administrator registration.

## Local smoke test

```bash
python inference.py
```

The bundled sample is `sample_data/sample_scin.png`. Model weights belong in `checkpoint/`; large weights and sample data are delivered separately from Git according to the platform manual.

## Limitations

This research model is not a medical device and must not be used as a stand-alone diagnosis. Validate preprocessing, labels, calibration, population shift, and localization behavior on the target clinical cohort.

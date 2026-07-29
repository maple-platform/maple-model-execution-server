# DermFoundation_SCIN

## Overview
- **Department**: Dermatology
- **Model type**: standalone
- **task_type**: classification
- **result_type**: `['occlusion_overlay', 'classification_probabilities']`
- **output_image_role**: `occlusion_overlay` (administrator registration required)

Google Derm Foundation embeddings with the locally trained SCIN common-13 logistic classifier.

## Input and output

Pass an absolute PNG/JPEG path to `main(input_data, model_path)`; pass the extracted checkpoint directory as `model_path`. The returned tuple is `[0]` an RGB `uint8` 3×3 occlusion-sensitivity overlay and `[1]` thirteen class probabilities in descending order. `occlusion_overlay` is a new platform role and requires administrator registration.

## Execution example

```python
from inference import main
overlay, predictions = main("sample_data/sample_scin.png", "checkpoint")
print(overlay.shape, predictions)
```

The bundled sample is `sample_data/sample_scin.png`. Model weights belong in `checkpoint/`; large weights and sample data are delivered separately from Git according to the platform manual.

## Limitations

This research model is not a medical device and must not be used as a stand-alone diagnosis. Validate preprocessing, labels, calibration, population shift, and localization behavior on the target clinical cohort.

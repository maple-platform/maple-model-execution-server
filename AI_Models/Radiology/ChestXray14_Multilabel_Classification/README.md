# ChestXray14_Multilabel_Classification

## Overview
- **Department**: Radiology
- **Model type**: standalone
- **task_type**: classification
- **result_type**: image
- **output_image_role**: gradcam_overlay

## Target Findings
This model predicts 14 NIH ChestX-ray14 thoracic findings: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, and Hernia.

## Input
| Item | Type | Description |
|------|------|-------------|
| input_data | image path string | PNG/JPG chest X-ray image path |

## Output
`main(input_data, model_path)` returns:

1. `result_image`: Grad-CAM overlay, `np.ndarray`, shape `(H, W, 3)`, dtype `uint8`, RGB.
2. `predictions`: list of 14 dictionaries containing `label`, `prob`, `threshold`, `pred`, and `pred_name`.
3. `model_output`: top-3 findings, Grad-CAM target label, model name, and threshold method.

If a Maple backend only accepts two return values, `inference.py` can be adapted to return `(result_image, predictions)`.

## Model
| Item | Value |
|------|-------|
| Architecture | TorchXRayVision ResNet |
| Weights | `resnet50-res512-all` pretrained weights |
| New training | None |
| Evaluation dataset | NIH ChestX-ray14 test split |
| Macro AUROC | 0.81721461 |
| Optimal-threshold macro accuracy | 0.74501596 |
| Optimal-threshold macro sensitivity | 0.74781398 |
| Optimal-threshold macro specificity | 0.74279418 |

## Preprocessing
1. Load PNG/JPG image.
2. Convert RGB image to grayscale.
3. Apply `xrv.datasets.normalize(img, 255)`.
4. Add channel dimension.
5. Apply `xrv.datasets.XRayCenterCrop()`.
6. Apply `xrv.datasets.XRayResizer(512)`.
7. Convert to `torch.float32` tensor and add batch dimension.

No CLAHE, histogram matching, or Gaussian blur is applied.

## Thresholds
Label-wise thresholds are preliminary NIH ChestX-ray14 Youden-index thresholds and require validation-set recalibration before clinical use.

## Run Example
```bash
python inference.py --image sample_data/sample_01.png --output sample_gradcam_overlay.png
```

```python
from inference import main

result_image, predictions, model_output = main(
    input_data="sample_data/sample_01.png",
    model_path=""
)
```

## Notes
This model is intended for research and platform integration testing, not standalone clinical decision-making.

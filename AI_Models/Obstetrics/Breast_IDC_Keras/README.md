# Breast IDC Keras

Patch-level binary classification of breast histopathology images using the public `MUmairAB/Breast_Cancer_Detector` weights. The output is a DataFrame containing `pred`, `pred_name`, `prob`, and `idc_positive_prob`.

- Checkpoint: `checkpoint/CanDetect.h5`
- Input: PNG/JPG/JPEG image; resized to 50 x 50 RGB
- Classes: IDC negative, IDC positive

Research use only. A patch-level score is not a patient-level breast cancer diagnosis.

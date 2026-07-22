# MammoCrop ROI

Mammography breast/background crop localization using `ianpan/mammo-crop`.

- Checkpoint: `checkpoint/model.safetensors`
- Input: one grayscale or RGB PNG/JPG/JPEG mammogram
- Output `[0]`: RGB input image with a red breast ROI rectangle
- Output `[1]`: prediction list with `x`, `y`, `width`, and `height`

The predicted ROI removes background before downstream screening. It does **not** localize cancer or a breast lesion. Research use only.

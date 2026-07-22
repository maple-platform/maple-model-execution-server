# MammoScreen

Breast cancer screening and density classification using `ianpan/mammoscreen`.

- Checkpoint: `checkpoint/model.safetensors`
- Input: cropped PNG/JPG/JPEG mammogram
- Output: cancer score, binary threshold result, and density A-D probabilities
- Optional integration input: `{"cc": path, "mlo": path}` for paired views

The public model's cancer score is not guaranteed to be a calibrated clinical probability. Research use only.

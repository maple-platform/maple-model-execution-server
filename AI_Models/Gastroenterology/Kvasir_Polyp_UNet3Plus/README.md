# Kvasir Polyp UNet3+

Polyp segmentation using `andreribeiro87/unet3plus-efficientnet-kvasir-seg`.

- Checkpoint: `checkpoint/model.safetensors`
- Input: colonoscopy PNG/JPG/JPEG image
- Output: RGB image with the predicted polyp mask overlaid in red
- Published model-card metrics: Dice 0.9234, IoU 0.8577

Research use only; segmentation requires clinical review.

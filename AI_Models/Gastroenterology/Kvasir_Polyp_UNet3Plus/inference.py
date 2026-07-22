"""Kvasir-SEG polyp segmentation inference."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import resize, to_tensor

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))
from modeling_unet3plus import UNet3PlusConfig, UNet3PlusForSegmentation  # noqa: E402


def main(input_data, model_path: str):
    image = Image.open(input_data).convert("RGB")
    config = UNet3PlusConfig(
        backbone="efficientnet", in_channels=3, out_channels=1,
        inter_ch=64, img_size=256,
    )
    model = UNet3PlusForSegmentation(config)
    weights = Path(model_path)
    if weights.is_dir():
        weights = weights / "model.safetensors"
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    model.eval()
    tensor = to_tensor(resize(image, [config.img_size, config.img_size])).unsqueeze(0)
    with torch.inference_mode():
        probability = model(pixel_values=tensor)["logits"].sigmoid()[0, 0]
    mask = Image.fromarray(((probability >= 0.5).cpu().numpy() * 255).astype("uint8"))
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    tint = Image.new("RGB", image.size, (255, 49, 88))
    overlay = image.copy()
    overlay.paste(Image.blend(image, tint, 0.55), mask=mask)
    return np.asarray(overlay, dtype=np.uint8)

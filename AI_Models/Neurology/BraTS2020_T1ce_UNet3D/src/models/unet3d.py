from __future__ import annotations

import torch
from monai.networks.nets import UNet


def build_unet3d(in_channels: int = 1, out_channels: int = 3) -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

"""Minimal RETFound-MAE ViT definition adapted from rmaphoh/RETFound.

Original project: https://github.com/rmaphoh/RETFound
License: CC BY-NC 4.0. Changes: isolated the MAE ViT used for inference.
"""
from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.use_global_pool = global_pool
        if self.use_global_pool:
            self.fc_norm = kwargs["norm_layer"](kwargs["embed_dim"])
            del self.norm

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        if self.use_global_pool:
            # Keep RETFound's singleton token dimension. timm 0.9.2's inherited
            # forward_head() applies token pooling and expects [B, tokens, C].
            return self.fc_norm(x[:, 1:, :].mean(dim=1, keepdim=True))
        x = self.norm(x)
        return x[:, 0]


def RETFound_mae(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

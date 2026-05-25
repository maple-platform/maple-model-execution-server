from __future__ import annotations

import torch
from monai.losses import DiceLoss


class DiceBCELoss(torch.nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice_loss = DiceLoss(sigmoid=True, include_background=True)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce

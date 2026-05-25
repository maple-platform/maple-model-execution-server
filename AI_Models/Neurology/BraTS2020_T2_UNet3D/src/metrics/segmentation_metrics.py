from __future__ import annotations

import torch


REGION_NAMES = ("wt", "tc", "et")


@torch.no_grad()
def dice_scores_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, float]:
    if logits.shape != labels.shape:
        raise ValueError(f"logits and labels must have the same shape, got {logits.shape} and {labels.shape}")
    if logits.ndim != 5 or logits.shape[1] != 3:
        raise ValueError(f"Expected shape (B, 3, D, H, W), got {tuple(logits.shape)}")

    preds = (torch.sigmoid(logits) >= threshold).to(torch.float32)
    labels = labels.to(torch.float32)
    reduce_dims = (1, 2, 3)
    per_channel_scores = []

    for channel in range(3):
        pred_c = preds[:, channel]
        label_c = labels[:, channel]
        intersection = (pred_c * label_c).sum(dim=reduce_dims)
        pred_sum = pred_c.sum(dim=reduce_dims)
        label_sum = label_c.sum(dim=reduce_dims)
        denominator = pred_sum + label_sum
        dice = torch.where(
            denominator > 0,
            (2.0 * intersection + eps) / (denominator + eps),
            torch.ones_like(denominator),
        )
        per_channel_scores.append(float(dice.mean().item()))

    metrics = {f"dice_{name}": score for name, score in zip(REGION_NAMES, per_channel_scores)}
    metrics["dice_mean"] = float(sum(per_channel_scores) / len(per_channel_scores))
    return metrics

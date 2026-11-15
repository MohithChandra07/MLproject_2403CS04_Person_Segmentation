"""Loss functions for binary person segmentation.

Includes:
- BCELoss: Binary cross-entropy with logits using nn.BCEWithLogitsLoss.
- DiceLoss: Soft Dice loss using sigmoid, intersection, union, and smoothing.
- BCEDiceLoss: Combined BCE + Dice loss with configurable weights.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """Binary Cross Entropy loss using nn.BCEWithLogitsLoss."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss from raw logits and ground truth targets.

        Args:
            logits: Predicted raw logits of shape (B, 1, H, W) or matching shape.
            targets: Binary ground truth masks with values in {0, 1}.

        Returns:
            Scalar BCE loss tensor.
        """
        if targets.shape != logits.shape:
            targets = targets.view_as(logits)
        targets = targets.float()
        return self.criterion(logits, targets)


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    Computes:
        probs = sigmoid(logits)
        intersection = sum(probs * targets)
        union = sum(probs) + sum(targets)
        dice = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - dice
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss from raw logits and ground truth targets.

        Args:
            logits: Raw logits tensor before sigmoid.
            targets: Binary ground truth masks with values in {0, 1}.

        Returns:
            Scalar Dice loss tensor.
        """
        probs = torch.sigmoid(logits)

        if targets.shape != probs.shape:
            targets = targets.view_as(probs)
        targets = targets.float()

        if probs.dim() > 1:
            # Flatten spatial dimensions per sample in the batch: (B, -1)
            b = probs.size(0)
            p_flat = probs.view(b, -1)
            t_flat = targets.view(b, -1)

            intersection = (p_flat * t_flat).sum(dim=1)
            union = p_flat.sum(dim=1) + t_flat.sum(dim=1)

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss = 1.0 - dice
            return loss.mean()
        else:
            intersection = (probs * targets).sum()
            union = probs.sum() + targets.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss with equal or configurable weights."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute weighted combined loss.

        Args:
            logits: Raw logits tensor from U-Net.
            targets: Binary ground truth masks.
            return_components: If True, returns (total_loss, bce_loss, dice_loss).

        Returns:
            total_loss tensor (or tuple with components if return_components=True).
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        if return_components:
            return total_loss, bce_loss, dice_loss
        return total_loss


def build_loss(cfg: dict) -> BCEDiceLoss:
    """Build combined loss function from configuration dictionary."""
    loss_cfg = cfg.get("loss", {})
    return BCEDiceLoss(
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        smooth=loss_cfg.get("smooth", 1.0),
    )


import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask):

        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Make dimensions consistent
        targets = targets.unsqueeze(1)
        valid_mask = valid_mask.unsqueeze(1)

        # Ignore invalid pixels
        probabilities = probabilities * valid_mask
        targets = targets * valid_mask

        # Calculate Dice per image
        batch_size = probabilities.shape[0]

        probabilities = probabilities.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            (2 * intersection + self.smooth)
            /
            (denominator + self.smooth)
        )

        # Average Dice loss across the batch
        dice_loss = 1 - dice

        return dice_loss.mean()


class CombinedLoss(nn.Module):

    def __init__(
        self,
        dice_weight=0.5,
        flood_weight=2.0
    ):
        super().__init__()

        self.dice_weight = dice_weight

        # Give flood pixels more importance
        self.bce = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor(flood_weight)
        )

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        targets,
        valid_mask
    ):

        targets = targets.unsqueeze(1)
        valid_mask = valid_mask.unsqueeze(1)

        # BCE
        bce = self.bce(
            logits,
            targets
        )

        # Ignore invalid pixels
        bce = bce * valid_mask

        valid_pixels = valid_mask.sum().clamp(min=1)

        bce_loss = (
            bce.sum() / valid_pixels
        )

        # Dice
        dice_loss = self.dice(
            logits,
            targets.squeeze(1),
            valid_mask.squeeze(1)
        )

        # Combined loss
        total_loss = (
            (1 - self.dice_weight) * bce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss
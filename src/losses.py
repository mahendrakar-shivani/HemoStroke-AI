# losses.py — Loss functions for Attention-UNet segmentation training

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Differentiable Dice loss = 1 - Dice coefficient.
    Directly optimizes overlap, which matters most when lesions are small
    relative to the whole image (most pixels are background).
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)

        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class ComboLoss(nn.Module):
    """
    Weighted BCE + Dice combined.

    Even in a lesion-positive image, the lesion itself is usually a small
    fraction of the 256x256 pixels. Plain BCE lets the model minimize loss
    by mostly predicting background everywhere -- it gets "free" credit from
    the vast majority of correctly-predicted negative pixels while barely
    learning the lesion. `pos_weight` counteracts this by making every
    missed lesion pixel cost `pos_weight` times more than a missed
    background pixel, forcing the model to actually engage with the
    (rare) positive class instead of ignoring it.
    """
    def __init__(self, bce_weight=0.3, pos_weight=10.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
        self.dice = DiceLoss()

    def weighted_bce(self, y_pred, y_true):
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        loss = -(self.pos_weight * y_true * torch.log(y_pred)
                  + (1 - y_true) * torch.log(1 - y_pred))
        return loss.mean()

    def forward(self, y_pred, y_true):
        bce_loss = self.weighted_bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

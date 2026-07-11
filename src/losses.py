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
    BCE + Dice combined.
    BCE gives stable pixel-wise gradients from the start of training.
    Dice directly rewards better lesion overlap, which BCE alone under-weights
    when lesions are a small fraction of the image (common in stroke masks).
    """
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, y_pred, y_true):
        bce_loss = self.bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

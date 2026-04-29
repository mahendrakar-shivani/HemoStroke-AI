# cnn_lightweight.py — Lightweight CNN (fast, small, real-time)

import torch
import torch.nn as nn

class CNNLightweight(nn.Module):
    """
    Lightweight CNN — 2 conv blocks + depthwise separable conv.
    Smallest model, fastest inference.
    Best for: real-time deployment, low-resource hospitals
    """
    def __init__(self):
        super(CNNLightweight, self).__init__()

        # Standard conv block
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)         # 256 → 128
        )

        # Depthwise separable convolution (efficient)
        self.block2 = nn.Sequential(
            # Depthwise
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Pointwise
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)         # 128 → 64
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)         # 64 → 32
        )

        # Global Average Pooling instead of Flatten
        # reduces parameters dramatically
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
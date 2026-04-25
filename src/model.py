# model.py — CNN Architecture for Stroke Detection

import torch
import torch.nn as nn

class StrokeCNN(nn.Module):
    """
    CNN for binary classification:
        0 = Normal brain
        1 = Stroke (ischemia or bleeding)

    Input shape:  (batch, 1, 256, 256)
    Output shape: (batch, 1)  — probability of stroke
    """
    def __init__(self):
        super(StrokeCNN, self).__init__()

        # Block 1: detects basic edges — 256×256 → 128×128
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 2: detects shapes and textures — 128×128 → 64×64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 3: detects stroke-specific patterns — 64×64 → 32×32
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 4: high-level feature extraction — 32×32 → 16×16
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Classifier: 256 channels × 16×16 = 65536 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),          # prevents overfitting
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()              # output: 0.0 to 1.0
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
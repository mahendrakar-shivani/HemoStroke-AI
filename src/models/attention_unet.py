# attention_unet.py — Attention U-Net for ischemic stroke lesion segmentation
#
# Standard U-Net encoder/decoder with attention gates on the skip connections.
# Each attention gate learns to suppress irrelevant background regions in the
# encoder features before they're merged into the decoder, so the network
# focuses on lesion-relevant areas instead of averaging in noise.

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers — the basic U-Net building block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention gate (Oktay et al., 2018).
    gate    : decoder feature map (coarser, more semantic)
    skip    : encoder feature map (finer, from the skip connection)
    Learns per-pixel weights (0-1) applied to `skip` before concatenation,
    so the decoder only pulls in encoder detail from relevant regions.
    """
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attention = self.relu(g + s)
        attention = self.psi(attention)          # (B, 1, H, W) weights in [0, 1]
        return skip * attention


class UpBlock(nn.Module):
    """Upsample, attention-gate the skip connection, concat, then conv."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            gate_channels=in_channels // 2,
            skip_channels=skip_channels,
            inter_channels=skip_channels // 2,
        )
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attention(gate=x, skip=skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary lesion segmentation.
    Input : (B, 1, 256, 256) preprocessed CT slice
    Output: (B, 1, 256, 256) sigmoid lesion probability map
    """
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, c)          # 256
        self.enc2 = ConvBlock(c, c * 2)                 # 128
        self.enc3 = ConvBlock(c * 2, c * 4)              # 64
        self.enc4 = ConvBlock(c * 4, c * 8)              # 32
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(c * 8, c * 16)        # 16

        # Decoder (each UpBlock: upsample + attention-gated skip + conv)
        self.up4 = UpBlock(c * 16, c * 8, c * 8)
        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)

        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.sigmoid(self.out_conv(d1))

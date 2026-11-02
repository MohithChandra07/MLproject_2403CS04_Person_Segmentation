"""U-Net implemented from scratch using PyTorch."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolution blocks with BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder: MaxPool followed by DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Decoder: Transposed convolution + skip connection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2
        )

        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):

        x = self.up(x)

        assert x.shape[-2:] == skip.shape[-2:], (
            f"Skip mismatch: {x.shape} vs {skip.shape}"
        )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """U-Net for binary person segmentation."""

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=32
    ):
        super().__init__()

        b = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, b)

        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)

        # Bottleneck
        self.down4 = Down(b * 8, b * 16)

        # Decoder
        self.up1 = Up(b * 16, b * 8)
        self.up2 = Up(b * 8, b * 4)
        self.up3 = Up(b * 4, b * 2)
        self.up4 = Up(b * 2, b)

        # Final output layer
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Raw logits, no sigmoid
        return self.outc(x)


def build_model(cfg: dict) -> UNet:

    m = cfg["model"]

    return UNet(
        m["in_channels"],
        m["out_channels"],
        m["base_channels"]
    )


def count_parameters(model: nn.Module) -> int:

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

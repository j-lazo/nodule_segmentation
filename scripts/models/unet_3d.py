import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 3D U-Net
# ============================================================

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),)

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, trilinear: bool = True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

        self.trilinear = trilinear

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        x1 = F.pad(
            x1,
            [
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2,
                diffZ // 2, diffZ - diffZ // 2,
            ]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, depth=5, trilinear=True):
        super().__init__()

        self.depth = depth

        # Encoder
        self.inc = DoubleConv3D(in_channels, base_ch)

        self.downs = nn.ModuleList()
        ch = base_ch
        for _ in range(depth - 1):
            self.downs.append(Down3D(ch, ch * 2))
            ch *= 2

        # Bottleneck channels
        self.bottleneck_channels = ch

        # Decoder
        self.ups = nn.ModuleList()
        for _ in range(depth - 1):
            self.ups.append(Up3D(ch + ch // 2, ch // 2, trilinear))
            ch //= 2

        self.outc = OutConv3D(base_ch, out_channels)

    def forward(self, x):
        x_enc = []

        x = self.inc(x)
        x_enc.append(x)

        # Down path
        for down in self.downs:
            x = down(x)
            x_enc.append(x)

        # Up path
        for i, up in enumerate(self.ups):
            skip = x_enc[-(i + 2)]
            x = up(x, skip)

        logits = self.outc(x)
        return logits


#class UNet3D(nn.Module):
#    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_ch: int = 32, trilinear: bool = True):
#        super().__init__()
#        self.inc = DoubleConv3D(in_channels, base_ch)
#        self.down1 = Down3D(base_ch, base_ch * 2)
#        self.down2 = Down3D(base_ch * 2, base_ch * 4)
#        self.down3 = Down3D(base_ch * 4, base_ch * 8)
#        self.down4 = Down3D(base_ch * 8, base_ch * 16)
#
#        self.up1 = Up3D(base_ch * 16 + base_ch * 8, base_ch * 8, trilinear)
#        self.up2 = Up3D(base_ch * 8 + base_ch * 4, base_ch * 4, trilinear)
#        self.up3 = Up3D(base_ch * 4 + base_ch * 2, base_ch * 2, trilinear)
#        self.up4 = Up3D(base_ch * 2 + base_ch, base_ch, trilinear)
#        self.outc = OutConv3D(base_ch, out_channels)
#
#    def forward(self, x):
#        x1 = self.inc(x)     # base
#        x2 = self.down1(x1)  # 2base
#        x3 = self.down2(x2)  # 4base
#        x4 = self.down3(x3)  # 8base
#        x5 = self.down4(x4)  # 16base
#
#        x = self.up1(x5, x4)
#        x = self.up2(x, x3)
#        x = self.up3(x, x2)
#        x = self.up4(x, x1)
#        logits = self.outc(x)
#        return logits
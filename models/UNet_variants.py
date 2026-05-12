import torch
import torch.nn as nn

from models.UNet import ConvBlock, UNet


class UNetNoSkip(nn.Module):
    """
    Ablation Study: Remove skip connections.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 5) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(512, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))
        x = self.bottleneck(x)
        x = self.dec4(self.upconv4(x))
        x = self.dec3(self.upconv3(x))
        x = self.dec2(self.upconv2(x))
        x = self.dec1(self.upconv1(x))
        return self.final_conv(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class UNetShallow(nn.Module):
    """
    Ablation Study: Reduced depth.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 5) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.upconv3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.final_conv(d1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class UNetWide(nn.Module):
    """
    Ablation Study: Double the network width.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 5) -> None:
        super().__init__()
        f = [128, 256, 512, 1024, 2048]
        self.enc1 = ConvBlock(in_channels, f[0])
        self.enc2 = ConvBlock(f[0], f[1])
        self.enc3 = ConvBlock(f[1], f[2])
        self.enc4 = ConvBlock(f[2], f[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f[3], f[4])

        self.upconv4 = nn.ConvTranspose2d(f[4], f[3], 2, stride=2)
        self.dec4 = ConvBlock(f[4], f[3])
        self.upconv3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = ConvBlock(f[3], f[2])
        self.upconv2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = ConvBlock(f[2], f[1])
        self.upconv1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = ConvBlock(f[1], f[0])
        self.final_conv = nn.Conv2d(f[0], out_channels, 1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.final_conv(d1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ── 4. Narrow U-Net (0.5x Filters) ────────────────────────────────────────────


class UNetNarrow(nn.Module):
    """
    Ablation Study: Half the network width.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 5) -> None:
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.enc1 = ConvBlock(in_channels, f[0])
        self.enc2 = ConvBlock(f[0], f[1])
        self.enc3 = ConvBlock(f[1], f[2])
        self.enc4 = ConvBlock(f[2], f[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f[3], f[4])

        self.upconv4 = nn.ConvTranspose2d(f[4], f[3], 2, stride=2)
        self.dec4 = ConvBlock(f[4], f[3])
        self.upconv3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = ConvBlock(f[3], f[2])
        self.upconv2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = ConvBlock(f[2], f[1])
        self.upconv1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = ConvBlock(f[1], f[0])
        self.final_conv = nn.Conv2d(f[0], out_channels, 1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.final_conv(d1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class UNetNoBN(nn.Module):
    """
    Ablation Study: Remove Batch Normalization.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 5) -> None:
        super().__init__()

        def simple_conv(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = simple_conv(in_channels, 64)
        self.enc2 = simple_conv(64, 128)
        self.enc3 = simple_conv(128, 256)
        self.enc4 = simple_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = simple_conv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = simple_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = simple_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = simple_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = simple_conv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.final_conv(d1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class UNetResidual(UNet):
    """
    Ablation Study: Add residual connections instead of Concatenation.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 5) -> None:
        super().__init__(in_channels, out_channels)
        self.dec4 = ConvBlock(512, 512)
        self.dec3 = ConvBlock(256, 256)
        self.dec2 = ConvBlock(128, 128)
        self.dec1 = ConvBlock(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self.upconv4(b) + e4)
        d3 = self.dec3(self.upconv3(d4) + e3)
        d2 = self.dec2(self.upconv2(d3) + e2)
        d1 = self.dec1(self.upconv1(d2) + e1)
        return self.final_conv(d1)


class UNetDeepBottleneck(nn.Module):
    """
    Ablation Study: 5 levels of depth.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 5) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(1024, 2048)

        self.upconv5 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec5 = ConvBlock(2048, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        b = self.bottleneck(self.pool(e5))

        d5 = self.dec5(torch.cat([self.upconv5(b), e5], dim=1))
        d4 = self.dec4(torch.cat([self.upconv4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.final_conv(d1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

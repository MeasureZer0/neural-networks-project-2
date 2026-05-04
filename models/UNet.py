import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 5) -> None:
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(self.pool(enc1_out))
        enc3_out = self.enc3(self.pool(enc2_out))
        enc4_out = self.enc4(self.pool(enc3_out))

        bottleneck_out = self.bottleneck(self.pool(enc4_out))

        dec4_out = self.upconv4(bottleneck_out)
        dec4_out = torch.cat((dec4_out, enc4_out), dim=1)  # type: ignore[assignment]
        dec4_out = self.dec4(dec4_out)

        dec3_out = self.upconv3(dec4_out)
        dec3_out = torch.cat((dec3_out, enc3_out), dim=1)  # type: ignore[assignment]
        dec3_out = self.dec3(dec3_out)

        dec2_out = self.upconv2(dec3_out)
        dec2_out = torch.cat((dec2_out, enc2_out), dim=1)  # type: ignore[assignment]
        dec2_out = self.dec2(dec2_out)

        dec1_out = self.upconv1(dec2_out)
        dec1_out = torch.cat((dec1_out, enc1_out), dim=1)  # type: ignore[assignment]
        dec1_out = self.dec1(dec1_out)

        return self.final_conv(dec1_out)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

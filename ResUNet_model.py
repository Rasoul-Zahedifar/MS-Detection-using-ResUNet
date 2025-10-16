import torch
import torch.nn as nn

# ------------------------------
# Residual Block
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection if in_channels != out_channels
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# ------------------------------
# ResUNet
# ------------------------------
class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[64, 128, 256, 512]):
        super(ResUNet, self).__init__()

        # Encoder
        self.encoder1 = ResidualBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ResidualBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ResidualBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ResidualBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(filters[3], filters[3]*2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(filters[3]*2, filters[3], kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(filters[3]*2, filters[3])
        self.upconv3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(filters[2]*2, filters[2])
        self.upconv2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(filters[1]*2, filters[1])
        self.upconv1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(filters[0]*2, filters[0])

        # Output
        self.conv_last = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))
        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        out = self.conv_last(d1)
        return out


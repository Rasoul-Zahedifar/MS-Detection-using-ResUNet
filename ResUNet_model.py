"""
ResUNet (Residual U-Net) model for medical image segmentation
Combines residual connections with U-Net architecture for better gradient flow
"""
import torch
import torch.nn as nn
import config


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for the first convolution
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection: adjust dimensions if needed
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        
        return out


class ResUNet(nn.Module):
    """
    ResUNet model for binary segmentation
    
    Architecture:
        - Encoder: 4 residual blocks with max pooling
        - Bottleneck: 1 residual block
        - Decoder: 4 transposed convolutions with skip connections
        
    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        out_channels (int): Number of output channels (1 for binary segmentation)
        filters (list): List of filter sizes for each encoder level
    """
    
    def __init__(self, in_channels=1, out_channels=1, filters=[64, 128, 256, 512]):
        super(ResUNet, self).__init__()
        
        # ===========================
        # Encoder (Downsampling Path)
        # ===========================
        self.encoder1 = ResidualBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.encoder2 = ResidualBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.encoder3 = ResidualBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.encoder4 = ResidualBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2, stride=2)
        
        # ===========================
        # Bottleneck
        # ===========================
        self.bottleneck = ResidualBlock(filters[3], filters[3] * 2)
        
        # ===========================
        # Decoder (Upsampling Path)
        # ===========================
        self.upconv4 = nn.ConvTranspose2d(filters[3] * 2, filters[3], 
                                         kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(filters[3] * 2, filters[3])
        
        self.upconv3 = nn.ConvTranspose2d(filters[3], filters[2], 
                                         kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(filters[2] * 2, filters[2])
        
        self.upconv2 = nn.ConvTranspose2d(filters[2], filters[1], 
                                         kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(filters[1] * 2, filters[1])
        
        self.upconv1 = nn.ConvTranspose2d(filters[1], filters[0], 
                                         kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(filters[0] * 2, filters[0])
        
        # ===========================
        # Output Layer
        # ===========================
        self.conv_last = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output segmentation map of shape (B, 1, H, W)
        """
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder path with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.decoder1(d1)
        
        # Output
        out = self.conv_last(d1)
        out = self.sigmoid(out)
        
        return out


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================
# Usage Example
# ===========================
if __name__ == "__main__":
    # Create model
    model = ResUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        filters=config.FILTERS
    )
    
    # Move to device
    model = model.to(config.DEVICE)
    
    # Print model info
    print("=" * 60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Total parameters: {count_parameters(model):,}")
    print("=" * 60)
    
    # Test forward pass
    dummy_input = torch.randn(2, config.IN_CHANNELS, 256, 256).to(config.DEVICE)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

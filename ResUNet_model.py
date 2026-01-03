"""
ResUNet-Transformer (Residual U-Net with Transformer) model for medical image segmentation
Combines residual connections, U-Net architecture, and Transformer blocks for enhanced feature learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module for vision transformers
    
    Args:
        dim (int): Dimension of input features
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, C) where N is sequence length, C is feature dimension
        Returns:
            out: (B, N, C)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network
    
    Args:
        dim (int): Dimension of input features
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to input dim
        dropout (float): Dropout rate
    """
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, C) where N is sequence length, C is feature dimension
        Returns:
            out: (B, N, C)
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchEmbedding(nn.Module):
    """
    Convert 2D image patches to sequence of embeddings
    
    Args:
        img_size (tuple): Image size (H, W)
        patch_size (int): Patch size
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    
    def __init__(self, img_size=(16, 16), patch_size=1, in_channels=1024, embed_dim=512):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, 
                             stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, N, embed_dim) where N is number of patches
        """
        B, C, H, W = x.shape
        
        # Project to embedding space
        x = self.proj(x)  # (B, embed_dim, H', W')
        _, _, H_p, W_p = x.shape
        
        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        
        return x


class PatchDeEmbedding(nn.Module):
    """
    Convert sequence of embeddings back to 2D image
    
    Args:
        embed_dim (int): Embedding dimension
        out_channels (int): Number of output channels
        img_size (tuple): Target image size (H, W)
        patch_size (int): Patch size
    """
    
    def __init__(self, embed_dim=512, out_channels=1024, img_size=(16, 16), patch_size=1):
        super(PatchDeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            out: (B, out_channels, H, W)
        """
        B, N, _ = x.shape
        
        # Project back to image space
        x = self.proj(x)  # (B, N, out_channels * patch_size^2)
        
        # Reshape to 2D
        H_p = W_p = int(math.sqrt(N))
        
        if self.patch_size == 1:
            # Simple case: each token is one pixel
            x = x.reshape(B, N, self.out_channels)  # (B, N, out_channels)
            x = x.permute(0, 2, 1)  # (B, out_channels, N)
            x = x.reshape(B, self.out_channels, H_p, W_p)
        else:
            # General case: each token represents a patch
            x = x.reshape(B, N, self.out_channels, self.patch_size, self.patch_size)
            x = x.permute(0, 2, 1, 3, 4)  # (B, out_channels, N, patch_size, patch_size)
            x = x.contiguous().view(B, self.out_channels, H_p * self.patch_size, 
                                   W_p * self.patch_size)
        
        return x


class ResUNet(nn.Module):
    """
    ResUNet-Transformer hybrid model for binary segmentation
    
    Architecture:
        - Encoder: 4 residual blocks with max pooling
        - Bottleneck: Residual block + Transformer blocks for global context
        - Decoder: 4 transposed convolutions with skip connections
        
    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        out_channels (int): Number of output channels (1 for binary segmentation)
        filters (list): List of filter sizes for each encoder level
        use_transformer (bool): Whether to use transformer blocks in bottleneck
        transformer_layers (int): Number of transformer layers in bottleneck
        transformer_heads (int): Number of attention heads in transformer
        transformer_dim (int): Embedding dimension for transformer (default: bottleneck channels)
    """
    
    def __init__(self, in_channels=1, out_channels=1, filters=[64, 128, 256, 512],
                 use_transformer=True, transformer_layers=2, transformer_heads=8, 
                 transformer_dim=None):
        super(ResUNet, self).__init__()
        
        self.use_transformer = use_transformer
        bottleneck_channels = filters[3] * 2
        
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
        # Bottleneck with Transformer
        # ===========================
        # Initial residual block
        self.bottleneck_conv = ResidualBlock(filters[3], bottleneck_channels)
        
        # Transformer blocks for global context
        if use_transformer:
            # Calculate bottleneck spatial dimensions (after 4 pooling operations)
            # Input size is divided by 2^4 = 16
            try:
                from config import IMAGE_SIZE
                bottleneck_h = IMAGE_SIZE[0] // 16
                bottleneck_w = IMAGE_SIZE[1] // 16
            except:
                # Default to 16x16 for 256x256 input
                bottleneck_h = 16
                bottleneck_w = 16
            
            # Set transformer dimension
            if transformer_dim is None:
                transformer_dim = bottleneck_channels
            
            # Patch embedding: convert 2D feature map to sequence
            self.patch_embed = PatchEmbedding(
                img_size=(bottleneck_h, bottleneck_w),
                patch_size=1,
                in_channels=bottleneck_channels,
                embed_dim=transformer_dim
            )
            
            # Transformer blocks
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(
                    dim=transformer_dim,
                    num_heads=transformer_heads,
                    mlp_ratio=4.0,
                    dropout=0.1
                ) for _ in range(transformer_layers)
            ])
            
            # Patch de-embedding: convert sequence back to 2D
            self.patch_deembed = PatchDeEmbedding(
                embed_dim=transformer_dim,
                out_channels=bottleneck_channels,
                img_size=(bottleneck_h, bottleneck_w),
                patch_size=1
            )
            
            # Projection layer to ensure channel consistency
            self.bottleneck_proj = nn.Conv2d(bottleneck_channels, bottleneck_channels, 
                                            kernel_size=1)
        else:
            self.bottleneck_proj = nn.Identity()
        
        # ===========================
        # Decoder (Upsampling Path)
        # ===========================
        self.upconv4 = nn.ConvTranspose2d(bottleneck_channels, filters[3], 
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
        b = self.bottleneck_conv(self.pool4(e4))
        
        # Apply transformer if enabled
        if self.use_transformer:
            # Convert to sequence
            b_seq = self.patch_embed(b)  # (B, N, embed_dim)
            
            # Apply transformer blocks
            for transformer_block in self.transformer_blocks:
                b_seq = transformer_block(b_seq)
            
            # Convert back to 2D
            b = self.patch_deembed(b_seq)  # (B, C, H, W)
            
            # Projection for consistency
            b = self.bottleneck_proj(b)
        
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
        filters=config.FILTERS,
        use_transformer=getattr(config, 'USE_TRANSFORMER', True),
        transformer_layers=getattr(config, 'TRANSFORMER_LAYERS', 2),
        transformer_heads=getattr(config, 'TRANSFORMER_HEADS', 8),
        transformer_dim=getattr(config, 'TRANSFORMER_DIM', None)
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

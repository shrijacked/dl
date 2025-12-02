"""
ConvTransGFusion: Hybrid CNN-Transformer with Attention-Guided Feature Fusion

Architecture Overview:
1. ConvNeXt Branch - Extracts local convolutional features through 4 stages
2. Swin Transformer Branch - Captures global self-attention patterns through 4 stages
3. Feature Alignment - Aligns spatial dimensions via bilinear interpolation
4. AGFF (Attention-Guided Feature Fusion):
   - Feature Calibration: Normalizes and projects features to common space
   - Channel Attention: Recalibrates channel-wise responses
   - Spatial Attention: Refines spatial focus
   - Fusion: Combines spatial and channel attended features
5. Classification Head - LayerNorm → Pool → Dropout → Linear

Key innovations:
- Dual-branch architecture combining CNN locality with Transformer global context
- Attention-guided fusion that learns to weight contributions from both branches
- Multi-scale spatial and channel attention for refined feature selection
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .train_utils import TrainingConfig, add_common_cli, run_training


# =============================================================================
# ConvNeXt Block Components
# =============================================================================

class ConvNeXtBlock(nn.Module):
    """
    One ConvNeXt Block as described in the diagram:
    - Depthwise Conv (7×7, groups=dim)
    - Permute (NCHW->NHWC)
    - LayerNorm
    - PWConv (dim -> 4×dim)
    - GELU Activation
    - PWConv (4×dim -> dim)
    - Residual + Drop Path
    """
    
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        # Depthwise conv with 7x7 kernel
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        # Pointwise convolutions implemented as linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer scale parameter
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        # Depthwise conv
        x = self.dwconv(x)
        # Permute to NHWC for LayerNorm
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        # Pointwise convs
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Apply layer scale
        if self.gamma is not None:
            x = self.gamma * x
        # Permute back to NCHW
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        # Residual with drop path
        x = shortcut + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) for regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtStage(nn.Module):
    """A stage of ConvNeXt blocks with optional downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        downsample: bool = True,
        drop_path_rates: Optional[List[float]] = None,
    ):
        super().__init__()
        
        # Downsampling layer (LayerNorm + Conv 2x2 stride 2)
        if downsample and in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.GroupNorm(1, in_channels),  # LayerNorm equivalent for NCHW
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            )
        else:
            self.downsample = nn.Identity()
        
        # ConvNeXt blocks
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks
        
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(out_channels, drop_path=drop_path_rates[i])
            for i in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXtBranch(nn.Module):
    """
    ConvNeXt Branch:
    - Stem: 3->96, stride=4 => (B×96×H/4×W/4)
    - Stage 1: (96 -> 96), 3 ConvNeXtBlocks
    - Stage 2: (96 -> 192), 3 ConvNeXtBlocks
    - Stage 3: (192 -> 384), 9 ConvNeXtBlocks
    - Stage 4: (384 -> 768), 3 ConvNeXtBlocks
    - Output: (B×768×Hc×Wc)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        dims: Tuple[int, ...] = (96, 96, 192, 384, 768),
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        # Stem: patchify with 4x4 conv, stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),  # LayerNorm for NCHW
        )
        
        # Compute drop path rates for each block
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = ConvNeXtStage(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                num_blocks=depths[i],
                downsample=(i > 0),  # No downsample for stage 1
                drop_path_rates=dpr[cur:cur + depths[i]],
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.norm = nn.GroupNorm(1, dims[-1])  # Final LayerNorm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        return x


# =============================================================================
# Swin Transformer Branch (using timm for pretrained weights)
# =============================================================================

class SwinBranch(nn.Module):
    """
    Swin Transformer Branch using timm's pretrained model:
    - Patch Embedding: 3->96, patch=4×4 => (B×96×H/4×W/4)
    - Stage 1: 2 SwinBlocks, 3 Heads (B×96×H1×W1)
    - Stage 2: 2 SwinBlocks, 6 Heads (B×192×H2×W2)
    - Stage 3: 6 SwinBlocks, 12 Heads (B×384×H3×W3)
    - Stage 4: 2 SwinBlocks, 24 Heads (B×768×Hs×Ws)
    - Output: (B×768×Hs×Ws)
    """
    
    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        
        # Use timm's Swin-Tiny as backbone
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,  # Remove classification head
        )
        
        # Get the output dimension (768 for Swin-Tiny)
        self.out_dim = 768
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features before global pooling
        # timm's Swin returns (B, H*W, C) or (B, C) depending on version
        features = self.backbone.forward_features(x)
        
        # Handle different output formats from timm
        if features.dim() == 2:
            # Shape is (B, C) - need to add spatial dimensions
            # This happens when the model includes global pooling
            B, C = features.shape
            features = features.view(B, C, 1, 1)
        elif features.dim() == 3:
            # Shape is (B, N, C) where N = H*W
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B, C, H, W)
        elif features.dim() == 4:
            # Could be (B, H, W, C) from some versions
            if features.shape[-1] == self.out_dim:
                # (B, H, W, C) -> (B, C, H, W)
                features = features.permute(0, 3, 1, 2)
            # else already (B, C, H, W)
        
        return features


# =============================================================================
# Attention-Guided Feature Fusion (AGFF)
# =============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention module:
    - GlobalAvgPool
    - Dense -> ReLU -> Dense
    - Sigmoid
    - Elementwise Multiply => Channel Attended
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        # Global average pooling
        y = self.avgpool(x).view(B, C)
        # Channel attention weights
        y = self.fc(y).view(B, C, 1, 1)
        # Apply attention
        return x * y


class SpatialAttention(nn.Module):
    """
    Spatial Attention module:
    - Conv2D(1×1)
    - Sigmoid
    - Elementwise Multiply => Spatially Attended
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial attention map
        attn = self.sigmoid(self.conv(x))
        return x * attn


class FeatureCalibration(nn.Module):
    """
    Feature Calibration module:
    - Norm(Conv), Norm(Swin)
    - Linear->half dims
    - Weighted scale
    - => Combined
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm_conv = nn.LayerNorm(in_channels)
        self.norm_swin = nn.LayerNorm(in_channels)
        
        # Project to half dimensions
        half_dim = out_channels // 2
        self.proj_conv = nn.Linear(in_channels, half_dim)
        self.proj_swin = nn.Linear(in_channels, half_dim)
        
        # Learnable combination weights
        self.weight_conv = nn.Parameter(torch.ones(1) * 0.5)
        self.weight_swin = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(
        self, conv_feat: torch.Tensor, swin_feat: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = conv_feat.shape
        
        # Reshape to (B, H*W, C) for LayerNorm
        conv_feat = conv_feat.permute(0, 2, 3, 1)  # (B, H, W, C)
        swin_feat = swin_feat.permute(0, 2, 3, 1)
        
        # Normalize
        conv_feat = self.norm_conv(conv_feat)
        swin_feat = self.norm_swin(swin_feat)
        
        # Project to half dimensions
        conv_proj = self.proj_conv(conv_feat)
        swin_proj = self.proj_swin(swin_feat)
        
        # Weighted combination and concatenation
        # Apply weights (normalized via softmax-like approach)
        weights = F.softmax(torch.stack([self.weight_conv, self.weight_swin]), dim=0)
        combined = torch.cat([
            weights[0] * conv_proj,
            weights[1] * swin_proj
        ], dim=-1)
        
        # Reshape back to (B, C, H, W)
        combined = combined.permute(0, 3, 1, 2)
        
        return combined


class AGFF(nn.Module):
    """
    Attention-Guided Feature Fusion (AGFF):
    - Feature Calibration: Norm and project features
    - Channel Attention: Recalibrate channel responses
    - Spatial Attention: Refine spatial focus
    - Fuse: Combine spatial + channel attended features => (B×768×Hc×Wc)
    """
    
    def __init__(self, in_channels: int = 768, out_channels: int = 768):
        super().__init__()
        
        # Feature calibration
        self.calibration = FeatureCalibration(in_channels, out_channels)
        
        # Channel attention
        self.channel_attn = ChannelAttention(out_channels, reduction=16)
        
        # Spatial attention
        self.spatial_attn = SpatialAttention(out_channels)
        
        # Final projection to combine attentions
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(
        self, conv_feat: torch.Tensor, swin_feat: torch.Tensor
    ) -> torch.Tensor:
        # Feature calibration and combination
        calibrated = self.calibration(conv_feat, swin_feat)
        
        # Channel attention branch
        channel_attended = self.channel_attn(calibrated)
        
        # Spatial attention branch
        spatial_attended = self.spatial_attn(calibrated)
        
        # Fuse spatial + channel attended features
        fused = torch.cat([channel_attended, spatial_attended], dim=1)
        fused = self.fuse_proj(fused)
        
        return fused


# =============================================================================
# Classification Head
# =============================================================================

class ClassificationHead(nn.Module):
    """
    Classification Head:
    - LayerNorm(768)
    - AdaptiveAvgPool2d(1)
    - Flatten => (B×768)
    - Dropout(p=0.1)
    - Linear (768->num_classes)
    - Softmax => Output Probs (applied during inference)
    """
    
    def __init__(self, in_features: int = 768, num_classes: int = 11, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.pool(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)
        x = self.norm(x)  # LayerNorm
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        return x


# =============================================================================
# ConvTransGFusion Model
# =============================================================================

class ConvTransGFusion(nn.Module):
    """
    ConvTransGFusion: Hybrid CNN-Transformer with Attention-Guided Feature Fusion
    
    Architecture:
    1. ConvNeXt Branch - Local convolutional features
    2. Swin Transformer Branch - Global attention features
    3. Feature Alignment - Bilinear interpolation to match spatial dims
    4. AGFF - Attention-guided fusion of both branches
    5. Classification Head - Final classification
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        drop_path_rate: Drop path rate for ConvNeXt branch
        dropout: Dropout rate for classification head
        pretrained_swin: Use pretrained Swin weights
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 11,
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
        pretrained_swin: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # ConvNeXt Branch
        self.convnext_branch = ConvNeXtBranch(
            in_channels=in_channels,
            dims=(96, 96, 192, 384, 768),
            depths=(3, 3, 9, 3),
            drop_path_rate=drop_path_rate,
        )
        
        # Swin Transformer Branch
        self.swin_branch = SwinBranch(
            in_channels=in_channels,
            pretrained=pretrained_swin,
        )
        
        # Attention-Guided Feature Fusion
        self.agff = AGFF(in_channels=768, out_channels=768)
        
        # Classification Head
        self.head = ClassificationHead(
            in_features=768,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        # Initialize weights for new modules
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvNeXt branch
        conv_feat = self.convnext_branch(x)  # (B, 768, Hc, Wc)
        
        # Swin Transformer branch
        swin_feat = self.swin_branch(x)  # (B, 768, Hs, Ws)
        
        # Feature Alignment: Align Swin to ConvNeXt spatial dimensions
        # Extract target dimensions from ConvNeXt output
        _, _, Hc, Wc = conv_feat.shape
        
        # Bilinear interpolation to match ConvNeXt spatial dims
        if swin_feat.shape[2:] != (Hc, Wc):
            swin_feat = F.interpolate(
                swin_feat,
                size=(Hc, Wc),
                mode='bilinear',
                align_corners=False,
            )
        
        # Attention-Guided Feature Fusion
        fused = self.agff(conv_feat, swin_feat)  # (B, 768, Hc, Wc)
        
        # Classification
        logits = self.head(fused)  # (B, num_classes)
        
        return logits


# =============================================================================
# Model Builder Function
# =============================================================================

def build_convtransgfusion(
    num_classes: int = 11,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Build ConvTransGFusion model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained Swin Transformer weights
        drop_path_rate: Drop path rate for ConvNeXt branch
        dropout: Dropout rate for classification head
    
    Returns:
        ConvTransGFusion model
    """
    model = ConvTransGFusion(
        in_channels=1,  # Grayscale input
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        dropout=dropout,
        pretrained_swin=pretrained,
    )
    return model


# =============================================================================
# Training Entry Point
# =============================================================================

def main() -> None:
    """Training entry point for ConvTransGFusion."""
    defaults = TrainingConfig(
        model_name="convtransgfusion",
        input_channels=1,
        input_size=224,
        epochs=50,
        batch_size=32,  # Reduced due to dual-branch memory usage
        lr=5e-4,
        momentum=0.9,
        weight_decay=5e-2,
        step_size=15,
        gamma=0.1,
        num_workers=4,
        seed=42,
    )
    
    parser = argparse.ArgumentParser(
        description="Train ConvTransGFusion on OrganAMNIST"
    )
    add_common_cli(parser, defaults)
    parser.add_argument(
        "--drop-path-rate",
        type=float,
        default=0.1,
        help="Drop path rate for ConvNeXt branch (default: 0.1)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for classification head (default: 0.1)",
    )
    args = parser.parse_args()
    
    drop_path_rate = args.drop_path_rate
    dropout = args.dropout
    
    def build_fn(num_classes: int) -> nn.Module:
        return build_convtransgfusion(
            num_classes=num_classes,
            pretrained=True,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
        )
    
    run_training(build_fn, defaults)


if __name__ == "__main__":
    main()


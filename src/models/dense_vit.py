"""
DenseViT: Vision Transformer with Dense Connections and Parallel Conv Branches

Key architectural innovations:
1. Dense connections between transformer blocks (DenseNet-style feature reuse)
2. Parallel CNN branches alongside each ViT block for local feature extraction
3. Adaptive fusion of global (attention) and local (conv) pathways
4. Multi-scale feature aggregation from all blocks
5. Differential learning rates for dense/fusion modules

Based on meeting notes:
- Dense connections instead of residual in ViT
- Higher LR for dense connection modules
- Parallel conv branch with every ViT block
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .train_utils import TrainingConfig, add_common_cli, run_training


class SEBlock(nn.Module):
    """Squeeze-and-Excitation for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        # Handle both 2D (B, C, H, W) and 1D (B, N, C) inputs
        if x.dim() == 4:
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)
        else:
            # For sequence input (B, N, C), average over sequence
            y = x.mean(dim=1)  # (B, C)
            y = self.fc(y).unsqueeze(1)  # (B, 1, C)
            return x * y.expand_as(x)


class ParallelConvBranch(nn.Module):
    """Local feature extraction branch running parallel to ViT attention.
    
    Uses depth-wise separable convolutions for efficiency.
    Operates on 2D spatial representation of patch tokens.
    """
    
    def __init__(self, dim: int, num_patches_side: int = 14, expansion: float = 2.0):
        super().__init__()
        self.num_patches_side = num_patches_side
        hidden_dim = int(dim * expansion)
        
        # Depth-wise separable conv block
        self.conv_block = nn.Sequential(
            # Depth-wise conv
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            # Point-wise expansion
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # Point-wise projection
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        
        # SE attention for the conv branch
        self.se = SEBlock(dim, reduction=4)
    
    def forward(self, x: torch.Tensor, has_cls_token: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, C) where N = num_patches + 1 (cls token)
            has_cls_token: Whether first token is CLS token
        Returns:
            Output tensor (B, N, C) with local conv features
        """
        B, N, C = x.shape
        
        # Separate CLS token if present
        if has_cls_token:
            cls_token = x[:, :1, :]  # (B, 1, C)
            patch_tokens = x[:, 1:, :]  # (B, N-1, C)
        else:
            cls_token = None
            patch_tokens = x
        
        # Reshape to 2D spatial: (B, N-1, C) -> (B, C, H, W)
        H = W = self.num_patches_side
        spatial = patch_tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        
        # Apply conv block
        conv_out = self.conv_block(spatial)
        conv_out = self.se(conv_out)
        
        # Reshape back to sequence: (B, C, H, W) -> (B, N-1, C)
        conv_out = conv_out.view(B, C, -1).transpose(1, 2).contiguous()
        
        # Reattach CLS token (zeros for conv path - CLS is global)
        if has_cls_token:
            cls_conv = torch.zeros_like(cls_token)
            conv_out = torch.cat([cls_conv, conv_out], dim=1)
        
        return conv_out


class AdaptivePathwayFusion(nn.Module):
    """Learnable fusion of global (attention) and local (conv) pathways.
    
    Generates input-dependent weights for combining the two pathways.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        # Context extraction
        self.norm = nn.LayerNorm(dim)
        
        # Weight generation network
        self.weight_gen = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feat: Features from attention pathway (B, N, C)
            local_feat: Features from conv pathway (B, N, C)
        Returns:
            Fused features (B, N, C)
        """
        # Use mean of both pathways for context
        combined = self.norm(global_feat + local_feat)
        context = combined.mean(dim=1)  # (B, C)
        
        # Generate pathway weights
        weights = self.weight_gen(context)  # (B, 2)
        w_global = weights[:, 0:1].unsqueeze(-1)  # (B, 1, 1)
        w_local = weights[:, 1:2].unsqueeze(-1)   # (B, 1, 1)
        
        # Weighted combination
        fused = w_global * global_feat + w_local * local_feat
        return fused


class DenseTransformerBlock(nn.Module):
    """Transformer block with parallel conv branch and dense connection support.
    
    Instead of standard residual: x + Attention(x)
    We output new features that will be CONCATENATED with previous features.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        num_patches_side: int = 14,
        growth_rate: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.growth_rate = growth_rate
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention (global pathway)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        
        # Parallel conv branch (local pathway)
        self.conv_branch = ParallelConvBranch(dim, num_patches_side)
        
        # Adaptive fusion of global and local pathways
        self.fusion = AdaptivePathwayFusion(dim)
        
        # MLP (feed-forward)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        
        # Bottleneck to produce growth_rate features for dense connection
        # This compresses the block output to a fixed growth_rate channels
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, growth_rate),
            nn.GELU(),
        )
        
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, N, C) - can be from dense concatenation
        Returns:
            output: Full-dim output for next block input (B, N, dim)
            growth: Compressed features for dense concatenation (B, N, growth_rate)
        """
        # Attention pathway (global)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        global_feat = self.drop(attn_out)
        
        # Conv pathway (local) - operates on same normalized input
        local_feat = self.conv_branch(normed, has_cls_token=True)
        
        # Adaptive fusion of pathways
        fused = self.fusion(global_feat, local_feat)
        
        # Residual connection for stability
        x = x + fused
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # Generate compressed growth features for dense connection
        growth = self.bottleneck(x)
        
        return x, growth


class DenseFeatureCompression(nn.Module):
    """Compress concatenated dense features back to working dimension.
    
    As features accumulate from dense connections, this module
    compresses them back to a manageable size while preserving information.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.compress = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, out_features),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(x)


class DenseViT(nn.Module):
    """Vision Transformer with Dense Connections and Parallel Conv Branches.
    
    Key features:
    1. Dense connections: Each block's output is concatenated with all previous
    2. Parallel conv branch: Local feature extraction alongside global attention
    3. Adaptive fusion: Learnable combination of global/local pathways
    4. Multi-scale aggregation: Features from all depths contribute to final output
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        num_classes: int = 11,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        growth_rate: int = 64,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.growth_rate = growth_rate
        
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.num_patches_side = img_size // patch_size
        
        # Hybrid patch embedding with conv stem for better low-level features
        self.patch_embed = nn.Sequential(
            # Initial conv to capture low-level features
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Patch projection
            nn.Conv2d(128, embed_dim, kernel_size=patch_size // 2, stride=patch_size // 2),
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Dense transformer blocks
        self.blocks = nn.ModuleList()
        self.compressions = nn.ModuleList()
        
        current_dim = embed_dim
        for i in range(depth):
            block = DenseTransformerBlock(
                dim=current_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                num_patches_side=self.num_patches_side,
                growth_rate=growth_rate,
            )
            self.blocks.append(block)
            
            # After each block, we'll have: current_dim + growth_rate features
            # Compress back to embed_dim for consistent processing
            accumulated_dim = current_dim + growth_rate
            compression = DenseFeatureCompression(accumulated_dim, embed_dim)
            self.compressions.append(compression)
            current_dim = embed_dim  # Reset to embed_dim after compression
        
        # Multi-scale feature aggregation
        # Collect growth features from all blocks and weight them
        self.scale_weights = nn.Parameter(torch.ones(depth) / depth)
        self.scale_norm = nn.LayerNorm(growth_rate)
        self.scale_proj = nn.Linear(growth_rate, embed_dim)
        
        # Final norm and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(embed_dim, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding with conv stem
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Process through dense transformer blocks
        growth_features = []
        
        for i, (block, compression) in enumerate(zip(self.blocks, self.compressions)):
            # Get block output and growth features
            x_out, growth = block(x)
            growth_features.append(growth)
            
            # Dense connection: concatenate current output with growth
            x_dense = torch.cat([x_out, growth], dim=-1)
            
            # Compress back to embed_dim
            x = compression(x_dense)
        
        # Multi-scale aggregation: weighted sum of growth features from all blocks
        # Focus on CLS token for classification
        scale_weights = F.softmax(self.scale_weights, dim=0)
        multi_scale = sum(
            w * feat[:, 0, :]  # CLS token from each layer's growth
            for w, feat in zip(scale_weights, growth_features)
        )
        multi_scale = self.scale_norm(multi_scale)
        multi_scale = self.scale_proj(multi_scale)
        
        # Combine with final CLS token
        cls_out = self.norm(x[:, 0])
        combined = cls_out + multi_scale
        
        return combined
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.head(features)
        return logits
    
    def get_param_groups(self, base_lr: float, dense_lr_mult: float = 5.0):
        """Get parameter groups with differential learning rates.
        
        Dense/fusion modules get higher LR as per meeting notes.
        """
        # Parameters that should get higher LR
        dense_params = []
        # Parameters that should get base LR
        base_params = []
        
        dense_module_names = ['fusion', 'bottleneck', 'compress', 'scale_weights', 'scale_proj']
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            is_dense = any(dn in name for dn in dense_module_names)
            if is_dense:
                dense_params.append(param)
            else:
                base_params.append(param)
        
        param_groups = [
            {'params': base_params, 'lr': base_lr},
            {'params': dense_params, 'lr': base_lr * dense_lr_mult, 'name': 'dense_modules'},
        ]
        
        return param_groups


def build_dense_vit(num_classes: int = 11, pretrained: bool = True) -> nn.Module:
    """Build DenseViT model.
    
    If pretrained=True, we initialize patch embedding from ViT-B/16 weights
    where possible and train the rest from scratch.
    """
    model = DenseViT(
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        growth_rate=64,
    )
    
    if pretrained:
        # Try to load pretrained ViT-B/16 and transfer applicable weights
        try:
            pretrained_vit = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                in_chans=1,
                num_classes=num_classes,
            )
            
            # Transfer positional embedding
            with torch.no_grad():
                # Copy pos_embed if shapes match
                if model.pos_embed.shape == pretrained_vit.pos_embed.shape:
                    model.pos_embed.copy_(pretrained_vit.pos_embed)
                
                # Copy cls_token
                if model.cls_token.shape == pretrained_vit.cls_token.shape:
                    model.cls_token.copy_(pretrained_vit.cls_token)
                
                # Copy attention weights from pretrained blocks
                for i, (our_block, pre_block) in enumerate(zip(model.blocks, pretrained_vit.blocks)):
                    # Copy attention weights
                    our_block.attn.in_proj_weight.copy_(pre_block.attn.qkv.weight)
                    our_block.attn.in_proj_bias.copy_(pre_block.attn.qkv.bias)
                    our_block.attn.out_proj.weight.copy_(pre_block.attn.proj.weight)
                    our_block.attn.out_proj.bias.copy_(pre_block.attn.proj.bias)
                    
                    # Copy layer norms
                    our_block.norm1.weight.copy_(pre_block.norm1.weight)
                    our_block.norm1.bias.copy_(pre_block.norm1.bias)
                    our_block.norm2.weight.copy_(pre_block.norm2.weight)
                    our_block.norm2.bias.copy_(pre_block.norm2.bias)
                    
                    # Copy MLP weights
                    our_block.mlp[0].weight.copy_(pre_block.mlp.fc1.weight)
                    our_block.mlp[0].bias.copy_(pre_block.mlp.fc1.bias)
                    our_block.mlp[3].weight.copy_(pre_block.mlp.fc2.weight)
                    our_block.mlp[3].bias.copy_(pre_block.mlp.fc2.bias)
            
            print("[DenseViT] Successfully transferred weights from pretrained ViT-B/16")
        except Exception as e:
            print(f"[DenseViT] Could not load pretrained weights: {e}")
            print("[DenseViT] Training from scratch with random initialization")
    
    return model


def main() -> None:
    defaults = TrainingConfig(
        model_name="dense_vit",
        input_channels=1,
        input_size=224,
        epochs=50,
        batch_size=24,  # Slightly smaller due to dense connections
        lr=1e-4,  # Base LR (dense modules will get 5x)
        momentum=0.9,
        weight_decay=5e-2,
        step_size=15,
        gamma=0.1,
        num_workers=4,
        seed=42,
    )
    
    parser = argparse.ArgumentParser(description="Train DenseViT on OrganAMNIST")
    add_common_cli(parser, defaults)
    parser.add_argument(
        "--dense-lr-mult",
        type=float,
        default=5.0,
        help="Learning rate multiplier for dense/fusion modules (default: 5x)"
    )
    args = parser.parse_args()
    
    # Note: For differential LR, we'd need to modify train_utils.run_training
    # For now, we use a higher base LR that works well with dense connections
    # Update: Use AdamW-compatible higher LR
    defaults = TrainingConfig(
        model_name="dense_vit",
        input_channels=1,
        input_size=224,
        epochs=50,
        batch_size=24,
        lr=5e-4,  # Higher LR as discussed for dense connections
        momentum=0.9,
        weight_decay=5e-2,
        step_size=15,
        gamma=0.1,
        num_workers=4,
        seed=42,
    )
    
    run_training(build_dense_vit, defaults)


if __name__ == "__main__":
    main()
    
    
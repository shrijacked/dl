 All models are adapted for **grayscale medical image classification (11 classes)** on the **OrganAMNIST dataset** (224×224 grayscale images of abdominal CT organ scans).

---

## 1. ResNet-50 (Residual Network)


Detailed neural network architecture diagram for **ResNet-50** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Initial Conv Layer**: 
  - Conv2d(1→64, kernel=7×7, stride=2, padding=3)
  - BatchNorm2d(64)
  - ReLU
  - MaxPool2d(kernel=3×3, stride=2, padding=1)
  
- **Residual Blocks** (Bottleneck architecture with 1×1 → 3×3 → 1×1 convolutions):
  - **Stage 1 (conv2_x)**: 3 bottleneck blocks, 64→256 channels
  - **Stage 2 (conv3_x)**: 4 bottleneck blocks, 128→512 channels (stride=2 on first block)
  - **Stage 3 (conv4_x)**: 6 bottleneck blocks, 256→1024 channels (stride=2 on first block)
  - **Stage 4 (conv5_x)**: 3 bottleneck blocks, 512→2048 channels (stride=2 on first block)

- **Each Bottleneck Block**:
  ```
  Input → [1×1 Conv → BN → ReLU] → [3×3 Conv → BN → ReLU] → [1×1 Conv → BN] → (+Input) → ReLU
  ```
  - Skip connection with 1×1 conv when dimensions change

- **Classification Head**:
  - AdaptiveAvgPool2d(1×1)
  - Flatten
  - Linear(2048 → 11)

**Visual Style**: Use rectangular blocks for layers, arrows for data flow, clearly show skip connections with curved/dashed lines, indicate channel dimensions at each stage.

---

## 2. ResNet-101 (Deeper Residual Network)


Detailed neural network architecture diagram for **ResNet-101** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Initial Conv Layer**: Same as ResNet-50 (Conv 7×7, stride=2, 64 channels)
  
- **Residual Blocks** (Bottleneck architecture):
  - **Stage 1 (conv2_x)**: 3 bottleneck blocks, 256 channels
  - **Stage 2 (conv3_x)**: 4 bottleneck blocks, 512 channels
  - **Stage 3 (conv4_x)**: **23 bottleneck blocks**, 1024 channels (this is the key difference from ResNet-50)
  - **Stage 4 (conv5_x)**: 3 bottleneck blocks, 2048 channels

- **Classification Head**:
  - AdaptiveAvgPool2d(1×1)
  - Flatten
  - Linear(2048 → 11)

**Visual Style**: Emphasize the much deeper Stage 3, use a compact notation (like "×23") to show repeated blocks, highlight total depth of 101 layers.

---

## 3. ResNeXt-50 (32×4d)


Detailed neural network architecture diagram for **ResNeXt-50 (32×4d)** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Key Innovation**: Uses "cardinality" - parallel grouped convolutions instead of single wide convolutions
- **32×4d meaning**: 32 parallel pathways, each with 4-channel width (total 128 channels in bottleneck)

- **Initial Conv Layer**: Conv 7×7, stride=2, 64 channels → BN → ReLU → MaxPool

- **ResNeXt Bottleneck Block Structure**:
  ```
  Input → [1×1 Conv, 128ch] → [3×3 Conv, 128ch, groups=32] → [1×1 Conv, 256ch] → (+Skip) → ReLU
  ```
  - **groups=32**: The 3×3 conv is split into 32 parallel groups of 4 channels each

- **Stages** (same count as ResNet-50):
  - Stage 1: 3 blocks, 256 channels output
  - Stage 2: 4 blocks, 512 channels output  
  - Stage 3: 6 blocks, 1024 channels output
  - Stage 4: 3 blocks, 2048 channels output

- **Classification Head**: AdaptiveAvgPool → Linear(2048 → 11)

**Visual Style**: Show the grouped convolution as 32 parallel small pathways merging, use a split-and-merge visual representation for the cardinality concept.

---

## 4. ResNeXt-101 (32×8d)


Detailed neural network architecture diagram for **ResNeXt-101 (32×8d)** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **32×8d meaning**: 32 parallel pathways, each with 8-channel width (total 256 channels in bottleneck)

- **Initial Conv Layer**: Conv 7×7, stride=2, 64 channels → BN → ReLU → MaxPool

- **Stages**:
  - Stage 1: 3 blocks, 256 channels
  - Stage 2: 4 blocks, 512 channels
  - Stage 3: **23 blocks**, 1024 channels (deeper like ResNet-101)
  - Stage 4: 3 blocks, 2048 channels

- **Classification Head**: AdaptiveAvgPool → Linear(2048 → 11)

**Visual Style**: Combine the grouped convolution visualization from ResNeXt-50 with the deeper Stage 3 from ResNet-101.

---

## 5. EfficientNet-B3


Detailed neural network architecture diagram for **EfficientNet-B3** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×300×300 grayscale image (larger input size)
- **Key Innovation**: Compound scaling of depth, width, and resolution; MBConv blocks with Squeeze-and-Excitation

- **Stem**: Conv 3×3, stride=2, 40 channels → BN → Swish

- **MBConv Block Structure (Mobile Inverted Bottleneck)**:
  ```
  Input → [1×1 Conv expand] → [Depthwise 3×3 or 5×5] → [SE Block] → [1×1 Conv project] → (+Skip)
  ```

- **Squeeze-and-Excitation (SE) Block**:
  ```
  Features → GlobalAvgPool → FC(reduce) → Swish → FC(expand) → Sigmoid → Scale Features
  ```

- **7 Stages of MBConv Blocks**:
  | Stage | Operator | Channels | Layers |
  |-------|----------|----------|--------|
  | 1 | MBConv1, k3×3 | 24 | 2 |
  | 2 | MBConv6, k3×3 | 32 | 3 |
  | 3 | MBConv6, k5×5 | 48 | 3 |
  | 4 | MBConv6, k3×3 | 96 | 5 |
  | 5 | MBConv6, k5×5 | 136 | 5 |
  | 6 | MBConv6, k5×5 | 232 | 6 |
  | 7 | MBConv6, k3×3 | 384 | 2 |

- **Head**: Conv 1×1 (1536 ch) → GlobalAvgPool → Dropout(0.3) → Linear(1536 → 11)

**Visual Style**: Show the inverted bottleneck expansion/compression, highlight SE attention mechanism, use Swish activation indicators.

---

## 6. DenseNet-121 (Dense Connections)


Detailed neural network architecture diagram for **DenseNet-121** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Key Innovation**: Dense connections - each layer receives feature maps from ALL preceding layers in the block

- **Initial Layers**:
  - Conv 7×7, stride=2, 64 channels
  - BatchNorm → ReLU
  - MaxPool 3×3, stride=2

- **Dense Block Structure** (growth_rate = 32):
  - Each Dense Layer: BN → ReLU → Conv1×1(128) → BN → ReLU → Conv3×3(32)
  - Output concatenated with all previous outputs
  
- **4 Dense Blocks with Transition Layers**:
  | Block | # Dense Layers | Output Channels |
  |-------|----------------|-----------------|
  | Dense Block 1 | 6 | 64 + 6×32 = 256 |
  | Transition 1 | - | 128 (halved) |
  | Dense Block 2 | 12 | 128 + 12×32 = 512 |
  | Transition 2 | - | 256 (halved) |
  | Dense Block 3 | 24 | 256 + 24×32 = 1024 |
  | Transition 3 | - | 512 (halved) |
  | Dense Block 4 | 16 | 512 + 16×32 = 1024 |

- **Transition Layer**: BN → ReLU → Conv1×1(half channels) → AvgPool 2×2

- **Classification Head**: BN → ReLU → GlobalAvgPool → Linear(1024 → 11)

**Visual Style**: Show dense connections as multiple arrows from each layer to all subsequent layers in the block, use different colors for each layer's output, highlight feature concatenation.

---

## 7. DenseNet-121 Adaptive (Custom - With Attention)


Detailed neural network architecture diagram for **Adaptive DenseNet-121** - a custom architecture with attention mechanisms for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Key Innovations**: 
  1. SE (Squeeze-and-Excitation) attention in every dense layer
  2. Adaptive per-layer gating based on input content

- **Enhanced Dense Layer**:
  ```
  Input → BN → ReLU → Conv1×1(128) → BN → ReLU → Conv3×3(32) → SE Block → Output
  ```

- **Squeeze-and-Excitation Block** (reduction=4):
  ```
  Features(C) → GlobalAvgPool → FC(C/4) → ReLU → FC(C) → Sigmoid → Scale Features
  ```

- **Enhanced Dense Block with Adaptive Gating**:
  ```
  [All Layer Outputs Concatenated] → GlobalAvgPool → MLP → Sigmoid → Per-Layer Gates
  Each layer output is scaled by its corresponding gate (0-1)
  ```
  - Gate MLP: Linear(total_ch → hidden) → ReLU → Linear(hidden → num_layers) → Sigmoid
  - hidden = max(8, total_channels // 8)

- **Transition Layer with SE Attention**:
  ```
  BN → ReLU → Conv1×1(half) → SE Block(reduction=8) → AvgPool 2×2
  ```

- **4 Dense Blocks**: Same structure as DenseNet-121 (6, 12, 24, 16 layers)

- **Classification Head**: BN → ReLU → GlobalAvgPool → Linear(1024 → 11)

**Visual Style**: Show SE blocks as attention modules (squeeze-excite pattern), visualize the gating mechanism with gate values controlling information flow, use attention/gate indicators.

---

## 8. ConvNeXt-Tiny


Detailed neural network architecture diagram for **ConvNeXt-Tiny** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Key Innovation**: Modernized ConvNet using ideas from Vision Transformers (larger kernels, LayerNorm, GELU, fewer activations)

- **Stem** (Patchify): Conv 4×4, stride=4, 96 channels → LayerNorm

- **ConvNeXt Block Structure**:
  ```
  Input → DepthwiseConv 7×7 → LayerNorm → Linear(4×expand) → GELU → Linear(project) → (+Input scaled by layer_scale)
  ```
  - Uses large 7×7 depthwise convolutions
  - Inverted bottleneck (expand then project)
  - Layer Scale: learnable per-channel scaling factor (~1e-6 init)

- **4 Stages**:
  | Stage | Blocks | Channels | Spatial Size |
  |-------|--------|----------|--------------|
  | 1 | 3 | 96 | 56×56 |
  | 2 | 3 | 192 | 28×28 |
  | 3 | 9 | 384 | 14×14 |
  | 4 | 3 | 768 | 7×7 |

- **Downsampling**: LayerNorm → Conv 2×2, stride=2 (between stages)

- **Classification Head**: GlobalAvgPool → LayerNorm → Linear(768 → 11)

**Visual Style**: Emphasize the large 7×7 depthwise conv, show the inverted bottleneck pattern, indicate LayerNorm positions (unlike BatchNorm in ResNets).

---

## 9. Vision Transformer Small (ViT-S/16)


Detailed neural network architecture diagram for **Vision Transformer Small (ViT-S/16)** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Patch Size**: 16×16 → 14×14 = 196 patches

- **Patch Embedding**:
  ```
  Image → Split into 16×16 patches → Linear projection(1×16×16 → 384) → Add positional embeddings
  ```
  - Prepend learnable [CLS] token: Total sequence = 197 tokens

- **Transformer Encoder** (12 blocks):
  ```
  Each Block:
  Input → LayerNorm → Multi-Head Self-Attention(6 heads) → (+Input) → LayerNorm → MLP → (+Input)
  ```
  
- **Multi-Head Self-Attention**:
  - 6 attention heads
  - Head dimension: 384/6 = 64
  - Q, K, V projections → Scaled dot-product attention → Concat → Linear project

- **MLP Block**:
  ```
  Linear(384 → 1536) → GELU → Dropout → Linear(1536 → 384) → Dropout
  ```
  - Expansion ratio: 4×

- **Classification Head**:
  ```
  Extract [CLS] token → LayerNorm → Linear(384 → 11)
  ```

**Dimensions**:
- Embedding dim: 384
- Heads: 6
- Depth: 12 blocks
- MLP ratio: 4

**Visual Style**: Show the image being split into patches, visualize self-attention as connecting all patches to each other, highlight the [CLS] token flow through the network.

---

## 10. Vision Transformer Base (ViT-B/16)


Detailed neural network architecture diagram for **Vision Transformer Base (ViT-B/16)** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Patch Size**: 16×16 → 196 patches + 1 [CLS] token = 197 tokens

- **Key Differences from ViT-S/16**:
  - Embedding dim: **768** (vs 384)
  - Attention heads: **12** (vs 6)
  - Depth: **12 blocks** (same)
  - MLP hidden: **3072** (768 × 4)

- **Transformer Encoder Block**:
  ```
  Input → LN → MHSA(12 heads, dim=64 per head) → (+Input) → LN → MLP(768→3072→768) → (+Input)
  ```

- **Classification Head**: [CLS] token → LN → Linear(768 → 11)

**Visual Style**: Similar to ViT-S but emphasize the larger model capacity, show the attention patterns conceptually.

---

## 11. Swin Transformer Tiny


Detailed neural network architecture diagram for **Swin Transformer Tiny** adapted for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Key Innovation**: Hierarchical vision transformer with shifted windows for efficient local-global attention

- **Patch Embedding**: Conv 4×4, stride=4 → 96 channels, 56×56 spatial

- **4 Stages with Patch Merging**:
  | Stage | Layers | Channels | Window Size | Spatial |
  |-------|--------|----------|-------------|---------|
  | 1 | 2 | 96 | 7×7 | 56×56 |
  | 2 | 2 | 192 | 7×7 | 28×28 |
  | 3 | 6 | 384 | 7×7 | 14×14 |
  | 4 | 2 | 768 | 7×7 | 7×7 |

- **Swin Transformer Block** (alternating):
  ```
  Block 1: LN → W-MSA (regular windows) → (+Input) → LN → MLP → (+Input)
  Block 2: LN → SW-MSA (shifted windows) → (+Input) → LN → MLP → (+Input)
  ```

- **Window-based Multi-head Self-Attention (W-MSA)**:
  - Divide feature map into non-overlapping 7×7 windows
  - Apply attention within each window independently
  - Much more efficient than global attention

- **Shifted Window MSA (SW-MSA)**:
  - Shift windows by (3,3) to enable cross-window connections
  - Uses cyclic shift + masking for efficient implementation

- **Patch Merging** (downsampling between stages):
  ```
  Concatenate 2×2 neighboring patches → Linear(4C → 2C)
  ```

- **Classification Head**: AdaptiveAvgPool → Linear(768 → 11)

**Visual Style**: Show the hierarchical pyramid structure, visualize window partitioning, illustrate the shift operation between adjacent transformer blocks.

---

## 12. Swin-MultiScale (Custom - Multi-Scale Feature Fusion)


Detailed neural network architecture diagram for **Swin-MultiScale** - a custom architecture with multi-scale feature fusion for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Key Innovations**:
  1. Extract features from ALL 4 Swin stages (not just final)
  2. Attention-weighted multi-scale fusion
  3. Deep supervision with auxiliary classification heads

- **Backbone**: Swin-Tiny (features_only mode)
  - Stage 1: 56×56, 96 channels
  - Stage 2: 28×28, 192 channels
  - Stage 3: 14×14, 384 channels
  - Stage 4: 7×7, 768 channels

- **Multi-Scale Feature Extraction**:
  ```
  Each Stage Output → Global Average Pool → Flatten → (B, C_i) features
  ```

- **Scale Projection** (project each scale to common dimension):
  ```
  Each (B, C_i) → Linear(C_i → 512) → LayerNorm → GELU → (B, 512)
  ```

- **Attention-Weighted Fusion**:
  ```
  Concatenate all projections → (B, 4×512)
  Weight Network: Linear(2048 → 512) → ReLU → Linear(512 → 4) → Softmax
  Weighted Sum: Σ(weight_i × projected_i) → (B, 512)
  Apply SE attention → LayerNorm → Fused features
  ```

- **Auxiliary Heads** (for deep supervision during training):
  ```
  Each stage → Linear(C_i → C_i/2) → GELU → Dropout → Linear(C_i/2 → 11)
  ```

- **Main Classification Head**:
  ```
  Fused features(512) → Dropout(0.1) → Linear(512 → 11)
  ```

- **Training Loss**: main_loss + 0.4 × average(aux_losses)

**Visual Style**: Show the pyramid backbone with feature extraction at each level, visualize the fusion module as merging 4 streams with learned attention weights, indicate auxiliary heads branching off.

---

## 13. DenseViT (Custom - Dense Connections + Parallel Conv)


Detailed neural network architecture diagram for **DenseViT** - a custom hybrid architecture combining Vision Transformer with DenseNet-style connections for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 1×224×224 grayscale image
- **Key Innovations**:
  1. Dense connections between transformer blocks (like DenseNet)
  2. Parallel CNN branch alongside each transformer block
  3. Adaptive fusion of global (attention) and local (conv) pathways
  4. Multi-scale feature aggregation from all blocks

- **Hybrid Patch Embedding (Conv Stem)**:
  ```
  Conv 7×7, stride=2, 64ch → BN → GELU →
  Conv 3×3, stride=1, 128ch → BN → GELU →
  Conv 8×8, stride=8, 768ch (patch projection)
  ```
  → 14×14 = 196 patches

- **Tokens**: [CLS] token + 196 patch tokens + positional embeddings

- **Dense Transformer Block** (12 blocks):
  ```
  ┌─ Input ─┬──────────────────────────────────────────────┐
  │         ↓                                              │
  │     LayerNorm                                          │
  │    ┌────┴────┐                                         │
  │    ↓         ↓                                         │
  │  MHSA   ParallelConv                                   │
  │  (12h)   (DepthSep)                                    │
  │    └────┬────┘                                         │
  │         ↓                                              │
  │  AdaptiveFusion (learned weights)                      │
  │         ↓                                              │
  │      (+Input)                                          │
  │         ↓                                              │
  │     LayerNorm → MLP → (+Input)                         │
  │         ↓                                              │
  │    Bottleneck (768 → 64) → Growth Features             │
  └─────────┴──────────────────────────────────────────────┘
  ```

- **Parallel Conv Branch**:
  ```
  Reshape patches to 14×14 spatial →
  DepthwiseConv 3×3 → BN → GELU →
  PointwiseConv (expand 2×) → BN → GELU →
  PointwiseConv (project) → BN →
  SE Attention → Reshape back to sequence
  ```

- **Adaptive Pathway Fusion**:
  ```
  (Global + Local) → LayerNorm → Mean → 
  FC(768→192) → GELU → FC(192→2) → Softmax →
  [w_global, w_local] weights →
  Output = w_global × Global + w_local × Local
  ```

- **Dense Connections**:
  - Each block outputs: (full_output, growth_features)
  - growth_features: 64 channels (compressed)
  - Concatenate: [block_output || growth] → Compress(768+64 → 768)

- **Multi-Scale Aggregation**:
  ```
  Collect [CLS] from all 12 growth features →
  Learnable scale weights → Softmax →
  Weighted sum → LayerNorm → Project(64 → 768) →
  Add to final [CLS] token
  ```

- **Classification Head**: LayerNorm → Dropout → Linear(768 → 11)

**Parameters**:
- Embed dim: 768
- Heads: 12
- Depth: 12 blocks
- Growth rate: 64
- MLP ratio: 4

**Visual Style**: Show the dual-pathway architecture (transformer + conv running in parallel), visualize dense connections accumulating features, show the fusion mechanism as a learned gate.

---

## 14. ResNet-18 Baseline


Simple neural network architecture diagram for **ResNet-18** baseline for grayscale medical image classification.

**Architecture Specifications:**
- **Input**: 3×128×128 (grayscale expanded to 3 channels, smaller size for baseline)
- **Simpler than ResNet-50**: Uses Basic Blocks instead of Bottleneck Blocks

- **Initial Conv**: Conv 7×7, stride=2, 64 channels → BN → ReLU → MaxPool

- **Basic Block Structure**:
  ```
  Input → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+Input) → ReLU
  ```

- **4 Stages**:
  | Stage | Blocks | Channels |
  |-------|--------|----------|
  | 1 | 2 | 64 |
  | 2 | 2 | 128 |
  | 3 | 2 | 256 |
  | 4 | 2 | 512 |

- **Classification Head**: AdaptiveAvgPool → Linear(512 → 11)

**Visual Style**: Simple and clean diagram, show the basic residual connections, suitable as a baseline comparison.

---

## Summary Comparison Table


Create a comparison table/infographic showing all 14 model architectures with the following attributes:

| Model | Type | Input Size | Parameters (approx) | Key Innovation |
|-------|------|------------|---------------------|----------------|
| ResNet-50 | CNN | 224×224 | 25M | Residual connections |
| ResNet-101 | CNN | 224×224 | 44M | Deeper residual |
| ResNeXt-50 | CNN | 224×224 | 25M | Grouped convolutions (cardinality) |
| ResNeXt-101 | CNN | 224×224 | 88M | Deeper + grouped |
| EfficientNet-B3 | CNN | 300×300 | 12M | Compound scaling + SE + MBConv |
| DenseNet-121 | CNN | 224×224 | 8M | Dense connections |
| DenseNet-121 Adaptive | CNN | 224×224 | 9M | Dense + SE + Adaptive gating |
| ConvNeXt-Tiny | CNN | 224×224 | 28M | Modernized ConvNet (ViT ideas) |
| ViT-S/16 | Transformer | 224×224 | 22M | Pure attention, patch-based |
| ViT-B/16 | Transformer | 224×224 | 86M | Larger pure transformer |
| Swin-Tiny | Transformer | 224×224 | 28M | Hierarchical + shifted windows |
| Swin-MultiScale | Transformer | 224×224 | 30M | Multi-scale fusion + deep supervision |
| DenseViT | Hybrid | 224×224 | 95M | Dense + parallel conv + transformer |
| ResNet-18 | CNN | 128×128 | 11M | Simple baseline |

**Visual Style**: Create a visual spectrum from pure CNNs → Hybrid → Pure Transformers, with model complexity indicated by size/color.

---

## Architecture Family Tree


Create an architecture family tree showing the evolution and relationships:

```
Classical CNNs
├── ResNet Family
│   ├── ResNet-18 (baseline)
│   ├── ResNet-50
│   ├── ResNet-101
│   └── ResNeXt (+ grouped convolutions)
│       ├── ResNeXt-50
│       └── ResNeXt-101
│
├── Dense Connections
│   ├── DenseNet-121
│   └── DenseNet-121 Adaptive (+ SE attention + gating)
│
├── Efficient Architectures
│   ├── EfficientNet-B3 (compound scaling)
│   └── ConvNeXt-Tiny (modernized ConvNet)
│
Vision Transformers
├── ViT Family
│   ├── ViT-S/16
│   ├── ViT-B/16
│   └── DenseViT (+ dense connections + parallel conv)
│
└── Hierarchical Transformers
    ├── Swin-Tiny
    └── Swin-MultiScale (+ multi-scale fusion)
```

**Visual Style**: Tree diagram with branches showing inheritance of ideas, color-code by architecture family.


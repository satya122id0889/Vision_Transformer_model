# Vision Transformer (ViT) Implementation

This project is a step-by-step implementation of a Vision Transformer (ViT) model, inspired by the architecture of `google/siglip-base-patch16-224`. The code reconstructs the key components of the transformer pipeline from scratch using PyTorch, validating their behavior against the HuggingFace implementation.

## 🔍 Overview

- Implemented **patch embedding** using `Conv2D`
- Constructed **positional embeddings** using `nn.Embedding`
- Visualized patch embeddings pre- and post-HuggingFace model
- Reproduced **self-attention** using multi-head attention modules
- Built **encoder layers** with residual connections, LayerNorm, and MLP
- Assembled a complete **Vision Transformer encoder** from scratch
- Compared model outputs with HuggingFace SigLIP ViT to ensure accuracy (max diff < 1e-6)

## 📦 Features

- Full PyTorch-based reimplementation of:
  - Patch embedding
  - Positional embedding
  - Multi-head self-attention
  - MLP with GELU activation
  - Encoder layers
  - Vision Transformer block
- HuggingFace-compatible state dict loading and validation
- Modular code structure using `@dataclass` and clean class definitions

## 🧪 Dependencies

- `torch`
- `torchvision`
- `transformers` (for HuggingFace models)
- `PIL` and `matplotlib` (for image processing and visualization)

Install all requirements:
```bash
pip install torch torchvision transformers pillow matplotlib

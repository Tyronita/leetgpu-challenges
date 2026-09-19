import torch
import triton
import triton.language as tl


# Q, K, V, output are tensors on the GPU — shape (N, d_model) float32
# positions is an int32 tensor of shape (N,) with token position indices
# Apply RoPE to Q and K, then compute multi-head scaled dot-product attention.
#
# Hint: a fused Triton kernel can apply RoPE and attention in a single pass
# (à la Flash-Attention), avoiding materialising the full (h, N, N) score matrix.
#
# Performance target: N=142, d_model=1280, h=20 (ESM-2 650M / SNCA sequence)
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    pass

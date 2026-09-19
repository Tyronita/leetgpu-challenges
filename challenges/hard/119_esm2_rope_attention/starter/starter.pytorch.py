import torch


# Q, K, V, output are tensors on the GPU — shape (N, d_model) float32
# positions is an int32 tensor of shape (N,) with token position indices
# Apply RoPE to Q and K, then compute multi-head scaled dot-product attention.
#
# RoPE rotation for head dim d_k:
#   freq[i] = 1 / 10000^(2i / d_k)
#   rotated[2i]   = x[2i]   * cos(pos * freq[i]) - x[2i+1] * sin(pos * freq[i])
#   rotated[2i+1] = x[2i]   * sin(pos * freq[i]) + x[2i+1] * cos(pos * freq[i])
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

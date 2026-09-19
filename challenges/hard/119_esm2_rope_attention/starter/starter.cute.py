import torch
import torch.utils.cpp_extension as cpp_ext


# CuTe-based solution — write a CUDA extension using CuTe tiled layouts.
# Q, K, V, output are tensors on the GPU — shape (N, d_model) float32
# positions is an int32 tensor of shape (N,)
#
# Hint: CuTe's TiledMMA and TiledCopy abstractions map naturally to the
# per-head blocked matrix multiply in RoPE attention.
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

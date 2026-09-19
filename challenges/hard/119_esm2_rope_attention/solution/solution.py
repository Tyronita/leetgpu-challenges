import math

import torch


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
    d_k = d_model // h

    # Reshape (N, d_model) → (N, h, d_k)
    Qh = Q.view(N, h, d_k)
    Kh = K.view(N, h, d_k)
    Vh = V.view(N, h, d_k)

    # ── RoPE ─────────────────────────────────────────────────────────────────
    half = d_k // 2
    # inv_freq[i] = 1 / 10000^(2i / d_k),  shape (half,)
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=Q.device) / d_k)
    )
    # angles: (N, half)  — broadcast positions (N,1) × inv_freq (1, half)
    angles = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    cos_ = angles.cos().unsqueeze(1)  # (N, 1, half) → broadcasts over h
    sin_ = angles.sin().unsqueeze(1)

    def apply_rope(x):
        x_even = x[..., 0::2]  # (N, h, half)
        x_odd = x[..., 1::2]
        rot = torch.empty_like(x)
        rot[..., 0::2] = x_even * cos_ - x_odd * sin_
        rot[..., 1::2] = x_even * sin_ + x_odd * cos_
        return rot

    Qr = apply_rope(Qh)  # (N, h, d_k)
    Kr = apply_rope(Kh)

    # ── Scaled dot-product attention ──────────────────────────────────────────
    # (h, N, d_k) for batched matmul
    Qr = Qr.permute(1, 0, 2)
    Kr = Kr.permute(1, 0, 2)
    Vt = Vh.permute(1, 0, 2)

    scale = 1.0 / math.sqrt(d_k)
    scores = torch.bmm(Qr, Kr.transpose(1, 2)) * scale  # (h, N, N)
    attn = torch.softmax(scores, dim=-1)
    head_out = torch.bmm(attn, Vt)  # (h, N, d_k)

    # ── Concat heads → (N, d_model) ───────────────────────────────────────────
    result = head_out.permute(1, 0, 2).reshape(N, d_model)
    output.copy_(result)

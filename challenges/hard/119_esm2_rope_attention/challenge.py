import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandTensor


class Challenge(ChallengeBase):
    name = "ESM-2 RoPE Attention"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    # ── Reference implementation ──────────────────────────────────────────────

    @staticmethod
    def _apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embeddings to x (N, h, d_k)."""
        N, h, d_k = x.shape
        assert d_k % 2 == 0
        half = d_k // 2
        # freq[i] = 1 / 10000^(2i / d_k)  for i in [0, half)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=x.device) / d_k))
        # angles: (N, half)
        angles = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)
        cos = angles.cos().unsqueeze(1).expand(N, h, half)  # (N, h, half)
        sin = angles.sin().unsqueeze(1).expand(N, h, half)

        x_even = x[..., 0::2]  # (N, h, half)
        x_odd = x[..., 1::2]   # (N, h, half)
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos
        return rotated

    def reference_impl(
        self,
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
        # Reshape: (N, d_model) -> (N, h, d_k)
        Qh = Q.view(N, h, d_k)
        Kh = K.view(N, h, d_k)
        Vh = V.view(N, h, d_k)

        Qr = self._apply_rope(Qh, positions)
        Kr = self._apply_rope(Kh, positions)

        # Attention: (h, N, d_k) for batched matmul
        Qr = Qr.permute(1, 0, 2)  # (h, N, d_k)
        Kr = Kr.permute(1, 0, 2)
        Vt = Vh.permute(1, 0, 2)

        scores = torch.bmm(Qr, Kr.transpose(1, 2)) / math.sqrt(d_k)  # (h, N, N)
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.bmm(attn, Vt)  # (h, N, d_k)

        # Concat heads: (h, N, d_k) -> (N, h, d_k) -> (N, d_model)
        result = head_out.permute(1, 0, 2).reshape(N, d_model)
        output.copy_(result)

    def reference_impl_jax(self, Q, K, V, positions, N, d_model, h):
        import jax
        import jax.numpy as jnp

        d_k = d_model // h

        def apply_rope_jax(x, positions):
            half = d_k // 2
            inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, d_k, 2, dtype=jnp.float32) / d_k))
            angles = positions[:, None].astype(jnp.float32) * inv_freq[None, :]
            cos_ = jnp.cos(angles)[:, None, :]
            sin_ = jnp.sin(angles)[:, None, :]
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            rot = jnp.empty_like(x)
            return jnp.concatenate(
                [x_even * cos_ - x_odd * sin_, x_even * sin_ + x_odd * cos_], axis=-1
            )[..., jnp.argsort(jnp.arange(d_k) % 2 * half + jnp.arange(d_k) // 2)]

        Qh = jnp.reshape(Q, (N, h, d_k))
        Kh = jnp.reshape(K, (N, h, d_k))
        Vh = jnp.reshape(V, (N, h, d_k))
        Qr = apply_rope_jax(Qh, positions)
        Kr = apply_rope_jax(Kh, positions)
        Qr = jnp.transpose(Qr, (1, 0, 2))
        Kr = jnp.transpose(Kr, (1, 0, 2))
        Vt = jnp.transpose(Vh, (1, 0, 2))
        scores = jnp.matmul(Qr, jnp.transpose(Kr, (0, 2, 1))) / math.sqrt(d_k)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attn, Vt)
        return jnp.reshape(jnp.transpose(out, (1, 0, 2)), (N, d_model))

    # ── Signature ─────────────────────────────────────────────────────────────

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "positions": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "h": (ctypes.c_int, "in"),
        }

    # ── Test helpers ──────────────────────────────────────────────────────────

    def _make_case(self, N, d_model, h, positions=None):
        dtype = torch.float32
        dev = self.device
        Q = torch.randn(N, d_model, device=dev, dtype=dtype)
        K = torch.randn(N, d_model, device=dev, dtype=dtype)
        V = torch.randn(N, d_model, device=dev, dtype=dtype)
        if positions is None:
            positions = torch.arange(N, device=dev, dtype=torch.int32)
        else:
            positions = torch.tensor(positions, device=dev, dtype=torch.int32)
        output = torch.empty(N, d_model, device=dev, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "positions": positions, "output": output, "N": N, "d_model": d_model, "h": h}

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        dev = self.device
        N, d_model, h = 2, 4, 2
        Q = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]], device=dev, dtype=dtype)
        K = Q.clone()
        V = Q.clone()
        positions = torch.tensor([0, 1], device=dev, dtype=torch.int32)
        output = torch.empty(N, d_model, device=dev, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "positions": positions, "output": output, "N": N, "d_model": d_model, "h": h}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        cases = []
        # position 0 = no rotation (cos=1, sin=0) — should match vanilla MHA
        cases.append(self._make_case(4, 8, 2, positions=[0, 0, 0, 0]))
        # contiguous positions, single head
        cases.append(self._make_case(8, 64, 1))
        # ESM-2 micro: 4-token sequence, 4 heads, d=256
        cases.append(self._make_case(4, 256, 4))
        # non-contiguous positions (masked token, typical ESM-2 input)
        cases.append(self._make_case(6, 128, 2, positions=[0, 1, 2, 3, 4, 5]))
        # larger N
        cases.append(self._make_case(32, 256, 4))
        return cases

    def generate_performance_test(self) -> Dict[str, Any]:
        # ESM-2 650M exact config: SNCA (L=140) + BOS + EOS = 142 tokens
        return {
            "Q": RandTensor((142, 1280), -1.0, 1.0),
            "K": RandTensor((142, 1280), -1.0, 1.0),
            "V": RandTensor((142, 1280), -1.0, 1.0),
            "positions": torch.arange(142, dtype=torch.int32),
            "output": OutTensor((142, 1280)),
            "N": 142,
            "d_model": 1280,
            "h": 20,
        }

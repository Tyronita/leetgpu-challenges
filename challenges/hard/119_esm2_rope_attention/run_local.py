"""
Local test runner — mirrors what the leetgpu service does.
Usage: python run_local.py [--device cpu|mps|cuda]
"""
import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

# ── Bootstrap: make 'core' importable ─────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]  # repo root
CHALLENGES_DIR = ROOT / "challenges"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CHALLENGES_DIR))
from core.challenge_base import OutTensor, RandTensor

# ── Load challenge ─────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("challenge", Path(__file__).parent / "challenge.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ── Load solution ──────────────────────────────────────────────────────────
sol_spec = importlib.util.spec_from_file_location(
    "solution", Path(__file__).parent / "solution" / "solution.py"
)
sol_mod = importlib.util.module_from_spec(sol_spec)
sol_spec.loader.exec_module(sol_mod)
solve = sol_mod.solve


def materialise(v, device):
    if isinstance(v, RandTensor):
        t = torch.empty(v.shape, dtype=torch.float32, device=device).uniform_(v.low, v.high)
        return t
    if isinstance(v, OutTensor):
        return torch.empty(v.shape, dtype=torch.float32, device=device)
    return v.to(device) if isinstance(v, torch.Tensor) else v


def run_case(challenge, case, device, label):
    ref_case = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()}
    sol_case = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()}

    # Reference
    ref_out = ref_case.get("output")
    challenge.reference_impl(**ref_case)

    # Solution
    solve(**sol_case)
    sol_out = sol_case.get("output")

    ok = torch.allclose(ref_out, sol_out, atol=challenge.atol, rtol=challenge.rtol)
    max_err = (ref_out - sol_out).abs().max().item()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}  max_err={max_err:.2e}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    device = args.device

    challenge = mod.Challenge(device=device)
    print(f"\n{'='*60}")
    print(f"  {challenge.name}  |  device={device}")
    print(f"  atol={challenge.atol}  rtol={challenge.rtol}")
    print(f"{'='*60}")

    all_pass = True

    # ── Example test ──────────────────────────────────────────────────────
    print("\n── Example test ──")
    case = challenge.generate_example_test()
    case = {k: materialise(v, device) for k, v in case.items()}
    ok = run_case(challenge, case, device, "example")
    all_pass &= ok

    # ── Functional tests ──────────────────────────────────────────────────
    print("\n── Functional tests ──")
    for i, raw_case in enumerate(challenge.generate_functional_test()):
        case = {k: materialise(v, device) for k, v in raw_case.items()}
        ok = run_case(challenge, case, device, f"functional[{i}]  N={case['N']} d={case['d_model']} h={case['h']}")
        all_pass &= ok

    # ── Performance test ─────────────────────────────────────────────────
    print("\n── Performance test (timing) ──")
    raw = challenge.generate_performance_test()
    case = {k: materialise(v, device) for k, v in raw.items()}
    ref_case = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()}
    sol_case = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()}

    # Warmup
    for _ in range(3):
        challenge.reference_impl(**{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()})
        solve(**{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()})

    REPS = 20
    t0 = time.perf_counter()
    for _ in range(REPS):
        challenge.reference_impl(**{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()})
    ref_ms = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS):
        solve(**{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()})
    sol_ms = (time.perf_counter() - t0) / REPS * 1000

    # Correctness on perf input
    challenge.reference_impl(**ref_case)
    solve(**sol_case)
    ok = torch.allclose(ref_case["output"], sol_case["output"], atol=challenge.atol, rtol=challenge.rtol)
    all_pass &= ok
    max_err = (ref_case["output"] - sol_case["output"]).abs().max().item()

    N, d, h = case["N"], case["d_model"], case["h"]
    print(f"  Config: N={N}, d_model={d}, h={h}")
    print(f"  Reference: {ref_ms:.2f} ms/call")
    print(f"  Solution:  {sol_ms:.2f} ms/call  ({ref_ms/sol_ms:.2f}x speedup)")
    print(f"  Correctness: {'PASS' if ok else 'FAIL'}  max_err={max_err:.2e}")

    print(f"\n{'='*60}")
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")
    print(f"{'='*60}\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

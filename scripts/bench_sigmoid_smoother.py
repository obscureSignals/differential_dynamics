#!/usr/bin/env python3
"""
Micro-benchmark for training-step cost comparing:
- Hard baseline (torchcomp.compexp_gain) on x_rms (full op)
- Sigmoid (our backend compexp_gain_mode with C++ smoother) on x_rms

This approximates end-to-end training iteration time for the two students under
otherwise identical conditions (loss = sum of outputs), using CPU.

Usage:
  python scripts/bench_sigmoid_smoother.py --fs 44100 --sec 1.0 --B 1 --reps 10 --k 2.0
"""
from __future__ import annotations

import argparse
import time
import torch

from differential_dynamics.backends.torch.gain import compexp_gain_mode
from third_party.torchcomp_core.torchcomp import ms2coef, compexp_gain


def timed(fn, reps: int = 5):
    # Warmup + time
    for _ in range(2):
        fn()
    t0 = time.time()
    for _ in range(reps):
        fn()
    t1 = time.time()
    return (t1 - t0) / max(1, reps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", type=int, default=44100)
    p.add_argument("--sec", type=float, default=1.0)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--k", type=float, default=2.0)
    args = p.parse_args()

    torch.manual_seed(0)
    B = args.B
    T = int(round(args.fs * args.sec))
    device = torch.device("cpu")
    dtype = torch.float32

    # Fake x_rms input: envelope in (0,1]
    x_rms = (0.1 + 0.9 * torch.rand(B, T, dtype=dtype, device=device)).contiguous()

    # Fixed params
    CT = -24.0
    CR = 4.0
    ET = -1000.0
    ER = 1.0
    at_ms = 10.0
    rt_ms = 100.0

    # Coeffs for hard
    at = ms2coef(torch.tensor(at_ms, dtype=dtype), args.fs).expand(B)
    rt = ms2coef(torch.tensor(rt_ms, dtype=dtype), args.fs).expand(B)

    # Hard forward+backward step
    def step_hard():
        x = x_rms.clone().detach().requires_grad_(True)
        y = compexp_gain(x, CT, CR, ET, ER, at, rt)
        loss = y.sum()
        loss.backward()

    # Sigmoid forward+backward step via backend (C++ smoother internally)
    def step_sig():
        x = x_rms.clone().detach().requires_grad_(True)
        # Use coefficient-based API (pass alphas directly)
        y = compexp_gain_mode(
            x_rms=x,
            comp_thresh=CT,
            comp_ratio=CR,
            exp_thresh=ET,
            exp_ratio=ER,
            alpha_a=at,
            alpha_r=rt,
            ar_mode="sigmoid",
            k=args.k,
            smoother_backend="torchscript",
        )
        loss = y.sum()
        loss.backward()

    t_hard = timed(step_hard, args.reps)
    t_sig = timed(step_sig, args.reps)

    print(f"CPU {B}x{T} — hard: {t_hard*1e3:.1f} ms/iter, sigmoid(C++): {t_sig*1e3:.1f} ms/iter, ratio: {t_sig/t_hard:.2f}x")


if __name__ == "__main__":
    main()

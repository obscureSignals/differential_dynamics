#!/usr/bin/env python3
"""
Parity test for the sigmoid smoother against the reference TorchScript implementation.

This script:
- Creates small random inputs (B=1..2, T~256)
- Computes forward outputs with:
  (A) C++ extension-backed smoother via backend entrypoint (sigmoid_smoother)
  (B) TorchScript reference _var_alpha_smooth_sigmoid
- Compares forward outputs (abs max diff)
- Compares gradients w.r.t. g, alpha_a, alpha_r by backpropagating a simple loss

Note: We do not compare grad wrt k here because the TorchScript path doesn't expose k
as a differentiable parameter in the same way. We validate g/alpha grads only.
"""
from __future__ import annotations

import math
import torch

from differential_dynamics.backends.torch.sigmoid_smoother_ext import sigmoid_smoother
import differential_dynamics.backends.torch.gain as gain_mod


def run_once(B: int = 2, T: int = 256, k: float = 1.3, seed: int = 123) -> None:
    torch.manual_seed(seed)
    dtype = torch.float32
    device = torch.device("cpu")

    g = torch.rand(B, T, dtype=dtype, device=device) * 0.9 + 0.1  # (0.1,1.0]
    alpha_a = torch.rand(B, dtype=dtype, device=device) * 0.9 + 0.05
    alpha_r = torch.rand(B, dtype=dtype, device=device) * 0.9 + 0.05

    # Forward: extension-backed via wrapper
    g_ext = g.clone().detach().requires_grad_(True)
    aa_ext = alpha_a.clone().detach().requires_grad_(True)
    ar_ext = alpha_r.clone().detach().requires_grad_(True)
    y_ext = sigmoid_smoother(g_ext, aa_ext, ar_ext, k)

    # Forward: TorchScript reference
    g_ref = g.clone().detach().requires_grad_(True)
    aa_ref = alpha_a.clone().detach().requires_grad_(True)
    ar_ref = alpha_r.clone().detach().requires_grad_(True)
    y_ref = gain_mod._var_alpha_smooth_sigmoid(g_ref, aa_ref, ar_ref, float(k))

    # Compare forward
    max_abs_diff = (y_ext - y_ref).abs().max().item()
    print(f"Forward max|diff|: {max_abs_diff:.3e}")

    # Backward: simple scalar loss (sum of outputs)
    y_ext.sum().backward()
    y_ref.sum().backward()

    def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).abs().max().item()

    gd = maxdiff(g_ext.grad, g_ref.grad)
    aad = maxdiff(aa_ext.grad, aa_ref.grad)
    ard = maxdiff(ar_ext.grad, ar_ref.grad)

    print(f"Grad max|diff| g: {gd:.3e}  alpha_a: {aad:.3e}  alpha_r: {ard:.3e}")

    # Loose tolerances to account for potential tiny numeric deviations
    assert max_abs_diff < 1e-5, "Forward mismatch too large"
    assert gd < 1e-4, "grad g mismatch too large"
    assert aad < 1e-4, "grad alpha_a mismatch too large"
    assert ard < 1e-4, "grad alpha_r mismatch too large"


if __name__ == "__main__":
    run_once(B=2, T=256, k=1.3, seed=123)
    print("Parity test passed.")

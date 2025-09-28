# CPU Sigmoid Smoother (C++ custom autograd) — Plan

Goal

- Achieve training speed on par (≤ 5× slower) than torchcomp (hard A/R with custom backward) on this Mac, for typical non-blind parameter-recovery experiments.
- Make sigmoid-gated smoother practical on CPU for short-to-medium windows (e.g., 0.25–2.0 s at 44.1 kHz), with stable gradients and minimal overhead.

Tenets

- Borrow as much as practical from torchcomp’s design:
  - Single reverse-scan custom backward (O(B·T)), no dynamic autograd graph over time.
  - Minimal saved state; recompute cheap intermediates in backward.
  - Tight memory access and contiguous tensors; per-batch parallelism only.
- Keep the rest of the pipeline in PyTorch:
  - Detector envelope (x_rms) remains precomputed in Python code.
  - Static curve (dB domain + clamping + db2amp) remains in Python code.
  - Only the smoother is implemented as a fused C++ op with custom autograd.
- Coefficient-based API: pass α_a, α_r directly (no ms conversions in smoother API).

Scope

- Implement a CPU C++/ATen extension exposing a fused forward/backward for the sigmoid-gated one-pole smoother only.
- Integrate behind the existing coefficient-based backend function (compexp_gain_mode), with a fallback to the current TorchScript smoother.
- Optional CUDA can follow later; not required to hit ≤ 5× target.

Interface (Python)

- def sigmoid_smoother(g: Tensor, alpha_a: Tensor, alpha_r: Tensor, k: float) -> Tensor
  - Inputs: g_raw_linear (B, T), alpha_a (B,), alpha_r (B,), k (float or (B,)) — all float32 contiguous CPU tensors
  - Output: y (B, T) float32 contiguous CPU tensor
- Implemented via a torch.autograd.Function that calls into the C++ extension for forward and backward.
- Called from compexp_gain_mode(...) when ar_mode="sigmoid".

Mathematics

Forward (per batch b, time t)

- y[-1] = 1.0
- Δ_db(t) = db(g_t) − db(y_{t−1}), db(x) = 20·log10(max(x, eps))
- s_t = σ(k · Δ_db(t))
- α_t = s_t · α_r + (1 − s_t) · α_a
- y_t = (1 − α_t) · y_{t−1} + α_t · g_t = y_{t−1} + α_t · (g_t − y_{t−1})

Backward (reverse scan)

Let:
- upstream = dL/dy_t (per-time incoming gradient)
- C = 20 / ln(10)
- ddb(x) = C / max(x, eps)
- Δy = g_t − y_{t−1}
- Δα = α_r − α_a
- s′ = s_t (1 − s_t)

Local derivatives:
- dL/dα_t (local) = upstream · ∂y_t/∂α_t = upstream · Δy
- ∂α_t/∂α_a = 1 − s_t,  ∂α_t/∂α_r = s_t
  - grad_αa += sum(dL/dα_t · (1 − s_t))
  - grad_αr += sum(dL/dα_t · s_t)
- Gate wrt k, g_t, y_{t−1}
  - ∂α_t/∂k = Δα · s′ · Δ_db(t)
  - ∂α_t/∂g_t = Δα · s′ · k · ddb(g_t)
  - ∂α_t/∂y_{t−1} = −Δα · s′ · k · ddb(y_{t−1})
- dL/dg_t:
  - ∂y_t/∂g_t = α_t + Δy · ∂α_t/∂g_t
  - grad_g[:, t] += upstream · (α_t + Δy · ∂α_t/∂g_t)
- Accumulator to y_{t−1} for reverse step:
  - ∂y_t/∂y_{t−1} = 1 − α_t + Δy · ∂α_t/∂y_{t−1}
  - grad_y_prev = upstream · (1 − α_t + Δy · ∂α_t/∂y_{t−1})
- Accumulate dL/dk:
  - grad_k += sum(dL/dα_t · ∂α_t/∂k)

Notes
- eps: use the same eps as forward db() (e.g., 1e-7) for both db(·) and ddb(·).
- α_a, α_r are per-batch scalars (can be expanded to (B,) in code). If later generalized, keep current API.

Data to save for backward

- Save y (B, T), g (B, T), alpha_a (B,), alpha_r (B,), k (float) in ctx for backward.
- Recompute s_t, α_t, Δ_db at each t during the reverse loop — cheaper than storing all per-time intermediates and reduces memory.
- If memory is extremely tight and T large, consider saving only y and recomputing g from static curve if accessible; but g is likely cheap to save from forward.

Implementation details (C++/ATen)

- Enforce contiguous float32 tensors on CPU; use .contiguous() on Python side if needed.
- Use raw pointer access or TensorAccessor for efficiency.
- Parallelization: parallelize over batch dimension (B) with at::parallel_for; keep the time loop serial inside each batch element.
- Avoid allocations inside loops; pre-allocate grad buffers and temporaries.
- Provide a small wrapper function that handles broadcast of alpha_a/alpha_r/scalar k to per-batch values.

Integration in Python backend

- differential_dynamics/backends/torch/gain.py, sigmoid path:
  - After computing gain_raw_linear (g), call sigmoid_smoother(g, alpha_a, alpha_r, k) to get y.
  - Keep TorchScript smoother as a fallback path behind a flag:
    - Prefer extension if available; else fall back (with a warning) so experiments still run.

Build & packaging

- Directory structure:
  - csrc/sigmoid_smoother.cpp (forward/backward kernels)
  - differential_dynamics/backends/torch/sigmoid_smoother_ext.py (autograd Function + import fallback)
  - setup for extension: use setuptools + torch.utils.cpp_extension.CppExtension
- Provide a build script or lazy-build on import (load via load(name, sources=...)).
- Document toolchain requirements (Clang/LLVM on macOS; ensure consistent C++17 standard).

Validation plan

- Unit tests (small shapes): B=1, T=256; random g ∈ (1e-3, 1], alpha_a/r ∈ (0,1), k ∈ [0.3, 3.0]
  - Forward parity: compare y from extension vs TorchScript smoother (abs diff < 1e-6 to 1e-5)
  - Backward parity: compare gradients w.r.t. inputs vs PyTorch autograd on the TorchScript path for the same inputs (using gradcheck-like numerical checks or direct autograd on small tensors)
- Numerical stability cases:
  - Very small g or y_{t−1} near eps; ensure no NaNs; verify ddb uses eps
  - Large k (e.g., 4–6): ensure gradients are finite (may be small due to s′)

Benchmarks

- Measure per-iteration forward+backward time for typical training shapes on this Mac (CPU):
  - B=1, T≈22k (0.5 s), 44.1k (1.0 s), 88.2k (2.0 s)
  - Compare:
    - torchcomp hard (reference)
    - TorchScript + autograd (current sigmoid) — expected very slow
    - C++ extension (target ≤ 5× torchcomp)
- Report memory usage qualitatively (peak resident set) if convenient.

Rollout plan / Checklist

- [ ] Create C++ extension skeleton (build + Python wrapper)
- [ ] Implement forward kernel (per-batch serial over T)
- [ ] Implement backward kernel (reverse scan) with minimal saved tensors
- [ ] Python autograd.Function wrapper + fallback path
- [ ] Unit tests for forward/backward parity vs TorchScript on small shapes
- [ ] Benchmarks on this Mac, with B/T grid; verify ≤ 5× torchcomp for 0.5–2.0 s windows
- [ ] Integrate into compexp_gain_mode (sigmoid branch)
- [ ] Add a flag/env to toggle extension vs fallback (debugging convenience)
- [ ] Document build instructions (macOS toolchain)

Risk & mitigation

- Numerical mismatch due to db eps or dtype: enforce identical eps and float32; add explicit clamps.
- Performance below target due to memory access patterns: ensure contiguous, avoid tiny kernel calls per time step; use raw pointers and tight loops.
- Complexity creep: keep scope limited to coefficients-based smoother; no ms conversions, no detector/static curve in C++.

Future work (optional)

- CUDA kernel for further speed on GPU; same API and math.
- Time-varying α_t input path (if needed for research variants).
- Alternative gating (e.g., hysteresis) once base op is solid.

Appendix: Pseudocode (per batch)

Forward
- y_prev = 1.0
- for t in [0..T-1]:
  - g = G[t]
  - delta_db = db(g) - db(y_prev)
  - s = sigmoid(k * delta_db)
  - alpha_t = s * alpha_r + (1 - s) * alpha_a
  - y = y_prev + alpha_t * (g - y_prev)
  - Y[t] = y; y_prev = y

Backward (given grad_out Y)
- grad_alpha_a = 0; grad_alpha_r = 0; grad_k = 0; grad_g[:] = 0
- grad_y_prev = 0
- for t in [T-1..0]:
  - upstream = grad_out[t] + grad_y_prev
  - g = G[t]; y_prev = (t==0 ? 1.0 : Y[t-1]); y = Y[t]
  - delta_db = db(g) - db(y_prev); s = sigmoid(k * delta_db); s_prime = s * (1 - s)
  - alpha_t = s * alpha_r + (1 - s) * alpha_a
  - delta_y = g - y_prev
  - dL_dalpha = upstream * delta_y
  - grad_alpha_a += dL_dalpha * (1 - s)
  - grad_alpha_r += dL_dalpha * s
  - grad_k       += dL_dalpha * (alpha_r - alpha_a) * s_prime * delta_db
  - dalpha_dg    = (alpha_r - alpha_a) * s_prime * k * ddb(g)
  - dalpha_dyp   = -(alpha_r - alpha_a) * s_prime * k * ddb(y_prev)
  - dy_dg        = alpha_t + delta_y * dalpha_dg
  - dy_dyp       = 1 - alpha_t + delta_y * dalpha_dyp
  - grad_g[t]   += upstream * dy_dg
  - grad_y_prev  = upstream * dy_dyp

End of plan.


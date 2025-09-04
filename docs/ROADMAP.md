# Differential Dynamics: Project Roadmap

This document captures the plan for building, benchmarking, and publishing a differentiable compressor suite with switchable attack/release (A/R) gating variants.

## Goals
- Forward-path benchmarking of multiple A/R gating variants with identical shared machinery.
- Training/learnability benchmarking (parameter recovery, convergence speed/stability).
- Usable, reproducible repo and a publishable paper.

## Shared machinery (kept identical across variants)
- Detector: amplitude EMA with time-constant mapping via `ms2coef` (10–90% convention).
  - Forward benches: use a fast EMA (torchaudio `lfilter`) for speed.
  - Training: may use the same EMA or repo’s `avg`; ensure consistency during comparisons.
- Static curve (dB domain) and clamping pattern exactly as in torchcomp:
  - `g_db = min(comp_slope·(CT − L), exp_slope·(ET − L)).neg().relu().neg()`.
  - `g_target = db2amp(g_db)`.
- Time constants via `ms2coef(ms, fs)` consistently throughout.

## API plan
Add a single entrypoint that mirrors torchcomp’s `compexp_gain` and adds a gating mode:

```
compexp_gain_mode(
  x_rms, comp_thresh, comp_ratio, exp_thresh, exp_ratio,
  at_ms, rt_ms, fs,
  ar_mode="hard",              # "hard" | "sigmoid" | "hysteresis" | "smoothmax" | "leveldep_tau"
  gate_cfg: Optional[dict]=None # e.g., {"k_db": 2.0, "beta": 0.15, ...}
) -> gain (B, T)
```

- `ar_mode="hard"`: call `torchcomp.compexp_gain` (baseline; forward+backward unchanged).
- Non-hard modes: compute the same `g_target` with torchcomp helpers, then smooth with a gate-specific smoother.

## Gating variants
- Hard (baseline): torchcomp’s if/else A/R switch.
- Sigmoid: gate on gain trajectory (prefer dB diff); `s = sigmoid(k·diff)`; `α_t = s·α_a + (1−s)·α_r`.
  - Calibrate `k` by desired dB transition width (e.g., `k_db ≈ 4.4 / W_dB`).
  - Optional gate smoothing `s̄_t = (1−β)·s̄_{t−1} + β·s_t` to reduce jitter.
- Hysteresis (smooth Schmitt): leaky gate state with up/down offsets.
- Smoothmax / level-dependent τ: interpolate α_t via smoothmax or map τ_rel with a sigmoid.

## Smoother kernels
- Forward benches (fast, differentiable): TorchScript variable-α recurrence:
  - `y_t = (1 − α_t)·y_{t−1} + α_t·g_target[:, t]`, α_t from the chosen gate.
- Optional forward-only faster path: Numba or tiny C++ kernel if needed.
- Long-term (for training efficiency): a custom "variable-α smoother" op that takes `α_t` and uses torchcomp’s efficient IIR backward to produce `dL/dx` and `dL/dα_t` (so gradients flow into gate params).

## Benchmarking plan (forward-path)
- Signals: step, ramp, bursts, AM tones; real audio snippets.
- Metrics (primary): gain RMSE in dB with silence masking.
- Secondary: MR-STFT distance on output audio; transient timing (63% rise/decay), overshoot/settling.
- Plots: gain traces, Pareto (forward error vs gate hyperparameters).
- Speed: detector via `lfilter` EMA; hard via torchcomp; non-hard via TorchScript smoother; wrap in `torch.no_grad()`.

## Training/learnability plan
- Targets: synthesize (x, y_ref) with the classical hard baseline (torchcomp) across randomized parameters {τ_a, τ_r, T, R, K}.
- For each `ar_mode`:
  - Hard: train torchcomp parameters (reference behavior).
  - Non-hard (Phase 1): TorchScript smoother + autograd through time; chunk sequences (1–4 s) to bound memory.
  - Non-hard (Phase 2): variable-α smoother custom op reusing torchcomp’s backward; compute `α_t` via gate; pass into op; backprop into gate parameters.
- Metrics: parameter RMSE, time-to-convergence, success rate across seeds, gradient stats near boundary.

## Performance guidelines
- Forward-only: prefer `lfilter` EMA and TorchScript smoother; avoid NumPy copies.
- Keep float32 and CPU (avoid CPU↔MPS transfers) unless explicitly targeting GPU.
- For long sequences in training: chunked processing or truncated BPTT.

## Repo structure
- `differential_dynamics/`
  - `backends/torch/`
    - `envelope.py` (existing)
    - `gain.py` (new) — `compexp_gain_mode` and smoother implementations
  - `baselines/`
    - `classical_compressor.py` (fast baseline; forward-only TorchScript smoother)
  - `benchmarks/`
    - `signals.py`, `metrics.py`
    - `bench_forward.py` (scripts/)
    - `train_param_recovery.py` (scripts/)
  - `docs/`
    - `ROADMAP.md` (this file)
- `third_party/torchcomp_core` (subtree)

## Milestones
1) Implement `compexp_gain_mode` with `ar_mode={hard, sigmoid}` (TorchScript smoother) and integrate with bench.
2) Forward-path benchmarks: figures for steps/ramps/bursts and real audio; gain RMSE tables.
3) Add hysteresis gate and a small k/β sweep; jitter analysis.
4) Parameter recovery training with pure Torch smoother; report convergence and stability.
5) Implement variable-α smoother custom op to reuse torchcomp backward; rerun training.
6) Paper and repo polish: docs, configs, CI to regenerate figures.

## Open questions / future work
- Exact gate on gain vs detector: we’ll standardize on gating the gain trajectory (more stable equivalence across modes).
- Multiband case (Dolby A–motivated): after single-band results are solid, reuse the same gating API per band.
- Cross-backend (JAX/TF) wrappers later if there’s demand; PyTorch-first for now.

---
Maintainer notes:
- Keep “hard” mode as an exact pass-through to `torchcomp.compexp_gain` so the baseline remains identical and torchcomp’s backward is available as-is.
- Use fast EMA for forward benches to avoid skewing speed comparisons by detector cost; document detector invariants clearly.


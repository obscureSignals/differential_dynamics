# Differential Dynamics: Project Roadmap

Status update (2025-10-08)
- New goal: Differentiable SSL-style compressor (auto-release dual RC ladder), peak detector, and sidechain feedback operating entirely in the dB domain.
- Current state: Forward path implemented; CPU backward implemented with analytic adjoints for most parameters and finite-difference gradients for the time constants. Verified by tests in tests/test_ssl_smoother_backward.py. Parameter-recovery training script is not yet implemented.
- Next steps (near term):
  - Add a parameter-recovery experiment (script + minimal nn.Module wrapper around SSL_comp_gain/ssl2_smoother) with dB-domain loss.
  - Expose toggles for time-constant gradient method (FD vs analytic) and epsilon.
  - Optional: surrogate gradients around the hard A/R gate if needed (forward remains hard).
  - Longer term: analytic gradients for time constants in the fused backward.

Note on archived content
- The remainder of this file includes the earlier roadmap targeting a generic compexp_gain_mode with hard vs sigmoid gates. Those sections are retained for context and marked as archived where applicable. Prefer the SSL-specific docs in docs/ssl_auto_release.md for ground truth.

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

## Scope (Phase 1: Compression-only)
- For this phase, we focus on compression only. Disable expansion by default: set exp_thresh = -1000 dB and exp_ratio = 1.0 in benchmarks and training.

## API plan
Add a single entrypoint that mirrors torchcomp’s `compexp_gain` and adds a gating mode:

```
compexp_gain_mode(
  x_rms, comp_thresh, comp_ratio, exp_thresh, exp_ratio,
  at_ms, rt_ms, fs,
  ar_mode="hard",              # "hard" | "sigmoid"
  k: float = 1.5
) -> gain (B, T)
```

- `ar_mode="hard"`: call `torchcomp.compexp_gain` (baseline; forward+backward unchanged).
- `ar_mode="sigmoid"`: compute the same `g_target` with torchcomp helpers, then apply a sigmoid-gated variable-α smoother.
- `k_db` calibration: choose via desired transition width `W_dB` using `k_db ≈ 4.4 / W_dB` (default width in the 3–6 dB range).
- Gating operates in dB (ratio-symmetric) by design.
- Compression-only defaults for Phase 1: `exp_thresh = -1000 dB`, `exp_ratio = 1.0`.

## Gating variants
- Hard (baseline): torchcomp’s if/else A/R switch.
- Sigmoid: gate on gain trajectory in dB; `s = sigmoid(k_db · (gain_db(g_tgt) − gain_db(y_{t−1})))`; `α_t = s·α_r + (1−s)·α_a`.
  - Calibrate `k_db` by desired dB transition width (e.g., `k_db ≈ 4.4 / W_dB`).
  - No gate EMA (`beta`) in the model for now.

Future ideas (deferred; revisit only if sigmoid underperforms):
- Hysteresis (smooth Schmitt) with up/down offsets.
- Smoothmax / level-dependent τ mappings.

## Smoother kernels
- Forward benches and training (Phase 1): TorchScript variable-α recurrence (no `beta`):
  - `y_t = (1 − α_t)·y_{t−1} + α_t·g_target[:, t]`, with `α_t` from the chosen gate.
- Optional forward-only faster path: Numba kernel (CPU) is available.
- Long-term (for training efficiency): a custom "variable-α smoother" op that takes `α_t` and uses torchcomp’s efficient IIR backward to produce `dL/dx` and `dL/dα_t` (so gradients flow into gate params).

## Benchmarking plan (forward-path)
- Signals: step, ramp, bursts, AM tones; real audio snippets.
- Primary metrics:
  - Parameter recovery: RMSE/abs error on {CT, CR, τ_a, τ_r} across randomized configs.
  - Training efficiency: wall-clock and steps-to-threshold (e.g., gain RMSE_dB < ε on held-out), plus success rate across seeds.
  - Forward-path faithfulness: gain RMSE in dB with silence masking.
- Expansion disabled in this phase (exp_thresh = -1000 dB, exp_ratio = 1.0).
- Secondary: waveform L1 / MR-STFT on output audio; transient timing (63% rise/decay), overshoot/settling.
- Plots: gain traces, Pareto (forward error vs. k_db), and training curves.
- Speed: detector via `lfilter` EMA; hard via torchcomp; non-hard via TorchScript smoother; wrap forward-only benches in `torch.no_grad()`.

## Dataset plan (teacher = hard A/R)
- Teacher everywhere: use the hard attack/release baseline to generate targets; the sigmoid variant is a student only.
- Compression-only: disable expansion by setting `exp_thresh = -1000 dB`, `exp_ratio = 1.0`.
- Detector: fixed for Phase 1 (e.g., 20 ms via `ms2coef` + `lfilter`). Store `detector_ms` in metadata; optionally cache `x_rms`.
- Sample rate: single (e.g., 44100). Fail loudly on mismatch.
- Per-example artifacts:
  - `x`: clean input (mono, float32)
  - `g_ref_hard`: per-sample gain from hard baseline
  - `y_ref = g_ref_hard * x`
  - `theta_ref`: `{comp_thresh_db, comp_ratio, attack_ms, release_ms}` used by the teacher
  - metadata: `{fs, detector_ms, seed, processing_version}`
- Signal sources: use existing generators (step, burst, ramp, tone) plus a few musical segments that hover near threshold or exhibit beating/AM. Sample teacher params per clip, e.g., `CT ∈ [-36, -18] dB`, `CR ∈ {2,4,8}`, `τ_a ∈ [2,50] ms` (log-uniform), `τ_r ∈ [20,400] ms` (log-uniform).

## Training/learnability plan
- Targets: `(x, y_ref)` with `g_ref_hard = y_ref / (x + eps)` or `(x, g_ref_hard)` directly; expansion is disabled in this phase.
- Students to train/compare (same static curve and fixed detector):
  - Hard student: simulate with `torchcomp.compexp_gain` and learn `θ = {CT, CR, τ_a, τ_r}` via autograd.
  - Sigmoid student (Phase 1): simulate with `compexp_gain_mode(..., ar_mode="sigmoid")` and learn the same `θ`; use `k_db` as a hyperparameter with annealing (start low, even < 1, and increase during training).
  - (Phase 2, optional): variable-α smoother custom op reusing torchcomp’s backward; pass `α_t` from the gate and backprop into gate parameters if Phase 1 becomes a bottleneck.
- Losses/metrics:
  - Primary loss: gain-trace RMSE in dB, `RMSE_dB(g_pred, g_ref_hard, mask=active_mask(x_rms))`.
  - Benchmark metrics: parameter recovery error (|ΔCT|, |Δlog CR|, |Δτ_a|, |Δτ_r|) and training efficiency (time/steps-to-threshold on `RMSE_dB`).
  - Optional auxiliary: waveform loss between `y_pred` and `y_ref` (e.g., ESR/L1/MR-STFT) for diagnostics; not required initially.

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

## Blind parameter estimation (Phase B)
- Goal: Estimate compressor parameters θ from compressed audio only (blind), and optionally reconstruct a cleaner signal.
- Rationale: Non-blind setups are mainly for benchmarking and simulator validation; blind inference is practical in the wild. Although ill-posed, neural priors learned from diverse content can make θ estimation useful.

Data and splits
- Use the existing dataset builder (hard-gate teacher) with permutations (fixed θ per perm) and disjoint train/val/test splits.
- For blind inputs, the estimator consumes only y_ref (compressed audio) and optional side features; θ_ref is used as label. x and g_ref are available for auxiliary training losses only.

Features (inputs to the estimator)
- Core: y waveform windows (1–4 s).
- Lightweight side-features derived from y (compute offline or on-the-fly):
  - y_rms_20ms: EMA envelope via the same ms2coef + lfilter pipeline used for detectors.
  - Multi-τ envelopes (optional): e.g., 5, 20, 50, 100 ms to help disentangle τ_a/τ_r.
  - Loudness proxy (optional): LUFS or RMS dB per window as a scalar side-channel.
  - Optional band-limited envelopes (future): a small number of subband envelopes (e.g., 4–8 bands) to capture spectral dependence of dynamics without heavy STFT features.

Estimator model
- Input branches: waveform y → small TCN/Conv1D stack; envelope features → shallow CNN/MLP; concatenate (+ LUFS scalar), then MLP → θ̂.
- Parameterization for stability: CR via exp(logit)+1; τ via sigmoid(logit)→coef→ms; CT free in dB.

Losses
- Primary: parameter losses in normalized spaces
  - L_CT = MSE(CT̂ − CT), L_logCR = MSE(log(CR̂−1) − log(CR−1)), L_logτa/τr similarly (log-ms domain).
- Auxiliary (training only): simulator-consistency using x
  - ŷ = sim(x; θ̂) with either hard or sigmoid smoother; L_wave = ESR/L1/MRSTFT(ŷ, y_ref) (small weight to regularize).
  - Optional: predict a global level offset δ to stabilize CT across level variations; use CT_eff = CT̂ + δ in the auxiliary sim.

Evaluation (blind)
- Report parameter RMSE on val/test per permutation.
- Stratify by near-threshold occupancy (fraction of frames within ±W dB of CT) to surface cases where the sigmoid smoother aids estimation.
- (Optional) If a gain-head is added later, report gain RMSE dB on y-only reconstructions (x̂ ≈ y/ĝ) for qualitative decompression.

Comparison of priors (hard vs sigmoid)
- Train two blind estimators that differ only in the auxiliary simulator used during training: hard vs sigmoid. Hypothesis: sigmoid prior improves recoverability on near-threshold microdynamics (beating/AM) where hard switching produces ambiguous signatures in y.

## Milestones
1) Implement and validate `compexp_gain_mode` with `ar_mode={hard, sigmoid}` (TorchScript smoother) and integrate with bench.
2) Forward-path benchmarks: figures for steps/ramps/bursts and real audio; establish default `k_db` via transition-width sweep.
3) Parameter recovery training (Phase 1) with TorchScript smoother; add `k_db` annealing schedule; report convergence, stability, efficiency.
4) (Optional, later) Implement variable-α smoother custom op to reuse torchcomp backward; rerun training if Phase 1 is bottlenecked.
5) Paper and repo polish: docs, configs, CI to regenerate figures.

## Open questions / future work
- Program-dependent release (PDR): make the release time constant a smooth function of level, e.g., τ_r(L) = τ_min + σ(k·(L − L0))·(τ_max − τ_min). Differentiable via α_r(L) = ms2coef(τ_r(L), fs). Needed for Dolby A–style behavior.
- Soft knee: implement a smooth transition around CT with knee width K (dB), e.g., via softplus/smooth-min or polynomial/tanh knee; produces a differentiable static curve.
- Exact gate on gain vs detector: we’ll standardize on gating the gain trajectory (more stable equivalence across modes).
- Multiband case (Dolby A–motivated): after single-band results are solid, reuse the same gating API per band.
- Cross-backend (JAX/TF) wrappers later if there’s demand; PyTorch-first for now.

---
Maintainer notes:
- Keep “hard” mode as an exact pass-through to `torchcomp.compexp_gain` so the baseline remains identical and torchcomp’s backward is available as-is.
- Use fast EMA for forward benches to avoid skewing speed comparisons by detector cost; document detector invariants clearly.


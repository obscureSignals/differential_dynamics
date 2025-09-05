"""
Switchable compressor/expander gain utilities (PyTorch backend).

This module keeps the detector semantics and static curve identical to torchcomp
and swaps only the attack/release (A/R) gating and smoothing policy via `ar_mode`.

Modes supported:
- "hard": exact baseline – delegates to torchcomp.compexp_gain (forward+backward unchanged)
- "sigmoid": fully-differentiable gate blending α_attack/α_release based on gain trajectory

Design principles:
- Detector and dB static curve are computed exactly as torchcomp does, so all variants
  differ only in the smoother (A/R policy), ensuring a fair comparison.
- Time constants use the 10–90% convention via ms2coef(ms, fs) across the board.
- Forward benches favor TorchScript loops (fast, pure Torch). Training can later
  replace the smoother with a custom op that reuses torchcomp's efficient backward.
"""

import torch
from typing import Optional, Dict, Union
import numpy as np
from numba import njit, prange

from differential_dynamics.benchmarks.bench_utilities import gain_db
from third_party.torchcomp_core.torchcomp import amp2db, db2amp, ms2coef, compexp_gain


@torch.jit.script
def _var_alpha_smooth_sigmoid(
    gain_raw_linear: torch.Tensor,  # (B, T) target gain (linear)
    alpha_a: torch.Tensor,  # (B,) attack coeff in (0,1)
    alpha_r: torch.Tensor,  # (B,) release coeff in (0,1)
    k: float,  # gate sharpness (linear or dB domain)
    gate_db: bool,  # True to gate on dB gain difference
    beta: float = 0.0,  # optional EMA on gate (0 disables)
) -> torch.Tensor:
    """
    Variable-α one-pole smoother with a sigmoid gate on the gain trajectory.

    Recurrence (per batch b, time t):
        s_t   = sigmoid(k * diff_t)
        α_t   = s_t * α_attack + (1 - s_t) * α_release
        y_t   = (1 - α_t) * y_{t-1} + α_t * g_tgt_t

    where diff_t is the difference between target gain and current smoothed gain.
    If gate_db is True, diff is computed in dB (more stable near small values);
    otherwise in linear domain.

    Args:
      gain_raw_linear: target gain g_tgt (B, T), linear scale in (0, 1].
      alpha_a, alpha_r: per-batch coefficients in (0,1) from ms2coef(at_ms/rt_ms).
      k: gate sharpness. Larger k sharpens the transition; calibrate via dB width.
      gate_db: gate on dB difference when True; otherwise linear difference.
      beta: optional EMA smoothing on the gate (0 disables), e.g., 0.1..0.3.

    Returns:
      y: smoothed gain (B, T), linear scale.

    Notes:
      - This function is TorchScript-compiled to reduce Python overhead in loops.
    """
    B = gain_raw_linear.size(0)
    T = gain_raw_linear.size(1)
    y = torch.empty_like(gain_raw_linear)
    prev_smoothed_gain = torch.ones(
        B, dtype=gain_raw_linear.dtype, device=gain_raw_linear.device
    )
    s_prev = torch.zeros(B, dtype=gain_raw_linear.dtype, device=gain_raw_linear.device)
    for t in range(T):
        gain_raw_linear_now = gain_raw_linear[:, t]
        if gate_db:
            # Gate on gain trajectory in dB
            diff = gain_db(gain_raw_linear_now) - gain_db(prev_smoothed_gain)
        else:
            # Gate on linear difference
            diff = gain_raw_linear_now - prev_smoothed_gain
        s = torch.sigmoid(k * diff)
        if beta > 0.0:
            s = (1.0 - beta) * s_prev + beta * s
        alpha_t = s * alpha_a + (1.0 - s) * alpha_r
        prev_smoothed_gain = (
            1.0 - alpha_t
        ) * prev_smoothed_gain + alpha_t * gain_raw_linear_now
        y[:, t] = prev_smoothed_gain
        s_prev = s
    return y


# ------------------------- Numba smoother (forward-only) -------------------------
@njit(parallel=True, fastmath=True)
def _var_alpha_smooth_sigmoid_numba_core(
    gain_raw_linear: np.ndarray,  # (B, T) float32
    alpha_a: np.ndarray,          # (B,)   float32
    alpha_r: np.ndarray,          # (B,)   float32
    k: float,                     # scalar
    gate_db: bool,                # scalar
    beta: float                   # scalar
) -> np.ndarray:
    B, T = gain_raw_linear.shape
    out = np.empty_like(gain_raw_linear)
    eps = 1e-7
    LN10_INV_20 = 8.685889638065036  # 20 / ln(10)
    for b in prange(B):
        prev = 1.0
        s_prev = 0.0
        a = alpha_a[b]
        r = alpha_r[b]
        for t in range(T):
            gt = gain_raw_linear[b, t]
            if gate_db:
                gtn = gt if gt > eps else eps
                prv = prev if prev > eps else eps
                diff = LN10_INV_20 * (np.log(gtn) - np.log(prv))
            else:
                diff = gt - prev
            s = 1.0 / (1.0 + np.exp(-k * diff))
            if beta > 0.0:
                s = (1.0 - beta) * s_prev + beta * s
            alpha_t = s * a + (1.0 - s) * r
            prev = (1.0 - alpha_t) * prev + alpha_t * gt
            out[b, t] = prev
            s_prev = s
    return out


def _var_alpha_smooth_sigmoid_numba(
    gain_raw_linear: torch.Tensor,
    alpha_a: torch.Tensor,
    alpha_r: torch.Tensor,
    k: float,
    gate_db: bool,
    beta: float = 0.0,
) -> torch.Tensor:
    """
    Numba-backed forward-only implementation of the sigmoid-gated variable-α smoother.
    Use this for fast forward-path benchmarking on CPU. Not suitable for autograd.
    """
    # Ensure contiguous float32 CPU arrays for Numba
    g_np = gain_raw_linear.detach().to(torch.float32).cpu().contiguous().numpy()
    a_np = alpha_a.detach().to(torch.float32).cpu().contiguous().numpy()
    r_np = alpha_r.detach().to(torch.float32).cpu().contiguous().numpy()

    y_np = _var_alpha_smooth_sigmoid_numba_core(g_np, a_np, r_np, float(k), bool(gate_db), float(beta))
    y = torch.from_numpy(y_np).to(gain_raw_linear.device, dtype=gain_raw_linear.dtype)
    return y


def compexp_gain_mode(
    x_rms: torch.Tensor,
    comp_thresh: Union[torch.Tensor, float],
    comp_ratio: Union[torch.Tensor, float],
    exp_thresh: Union[torch.Tensor, float],
    exp_ratio: Union[torch.Tensor, float],
    at_ms: Union[torch.Tensor, float],
    rt_ms: Union[torch.Tensor, float],
    fs: int,
    ar_mode: str = "hard",
    gate_cfg: Optional[Dict[str, Union[float, bool]]] = None,
    smoother_backend: str = "torchscript",  # "torchscript" | "numba"
) -> torch.Tensor:
    """
    Switchable compressor/expander gain function.

    This keeps the detector semantics and the dB static curve identical to
    torchcomp and swaps only the A/R gating/smoothing policy via `ar_mode`.

    Modes:
      - "hard": delegates entirely to torchcomp.compexp_gain (baseline; preserves backward)
      - "sigmoid": smooth gate between attack/release based on gain trajectory

    Args (mirror torchcomp.compexp_gain):
      x_rms: (B, T) non-negative detector signal (e.g., amplitude EMA of |x|).
      comp_thresh / exp_thresh: thresholds in dB.
      comp_ratio: > 1 (downward compression above comp_thresh).
      exp_ratio: in (0,1) (downward expansion below exp_thresh).
      at_ms / rt_ms: attack/release times in milliseconds (10–90% convention).
      fs: sample rate.
      ar_mode: "hard" | "sigmoid" (more modes can be added).
      gate_cfg: optional dict for gate parameters:
        - k_db: float, gate sharpness in dB (≈ 4.4 / transition_width_dB)
        - beta: float, gate EMA smoothing (0 disables)
        - gate_db: bool, gate on dB difference when True
      smoother_backend: "torchscript" (default) for differentiable Torch smoother,
                        or "numba" for faster forward-only CPU recurrence.

    Returns:
      (B, T) linear gain to apply to the signal.

    Notes:
      - The hard mode calls torchcomp directly, ensuring identical baseline
        behavior and backward.
      - Non-hard modes compute the same static curve and apply a TorchScript
        smoother for forward benchmarking (fully differentiable).
    """
    if ar_mode == "hard":
        # Delegate entirely to torchcomp baseline (identical static curve + smoother,
        # and preserves its custom backward). This is the exact reference.
        at = ms2coef(torch.as_tensor(at_ms), fs)
        rt = ms2coef(torch.as_tensor(rt_ms), fs)
        return compexp_gain(
            x_rms=x_rms,
            comp_thresh=comp_thresh,
            comp_ratio=comp_ratio,
            exp_thresh=exp_thresh,
            exp_ratio=exp_ratio,
            at=at,
            rt=rt,
        )

    # Compute the static curve exactly like torchcomp (dB domain + clamping):
    #   g_db = min(comp_slope*(CT - L), exp_slope*(ET - L)).neg().relu().neg()
    # This ensures we only reduce or keep unity gain (<= 0 dB).
    B, T = x_rms.shape
    dev, dt = x_rms.device, x_rms.dtype
    comp_thresh_t = torch.as_tensor(comp_thresh, device=dev, dtype=dt).expand(B)
    exp_thresh_t = torch.as_tensor(exp_thresh, device=dev, dtype=dt).expand(B)
    comp_ratio_t = torch.as_tensor(comp_ratio, device=dev, dtype=dt).expand(B)
    exp_ratio_t = torch.as_tensor(exp_ratio, device=dev, dtype=dt).expand(B)

    comp_slope = 1.0 - 1.0 / comp_ratio_t
    exp_slope = 1.0 - 1.0 / exp_ratio_t

    L = amp2db(torch.clamp(x_rms, min=1e-7))
    gain_raw_db = (
        torch.minimum(
            comp_slope[:, None] * (comp_thresh_t[:, None] - L),
            exp_slope[:, None] * (exp_thresh_t[:, None] - L),
        )
        .neg()
        .relu()
        .neg()
    )
    gain_raw_linear = db2amp(gain_raw_db)

    # Time constants: convert ms -> one-pole coefficients (10–90% convention)
    alpha_a = ms2coef(torch.as_tensor(at_ms, device=dev, dtype=dt), fs).expand(B)
    alpha_r = ms2coef(torch.as_tensor(rt_ms, device=dev, dtype=dt), fs).expand(B)

    # Gate configuration with sensible defaults; can be overridden via gate_cfg
    k_db = 2.0
    beta = 0.0
    gate_db = True
    if gate_cfg is not None:
        if "k_db" in gate_cfg and gate_cfg["k_db"] is not None:
            k_db = float(gate_cfg["k_db"])  # approximate 4.4 / width_dB
        if "beta" in gate_cfg and gate_cfg["beta"] is not None:
            beta = float(gate_cfg["beta"])  # small gate smoothing (0.0..0.3)
        if "gate_db" in gate_cfg and gate_cfg["gate_db"] is not None:
            gate_db = bool(gate_cfg["gate_db"])  # True to gate in dB domain

    if ar_mode == "sigmoid":
        if smoother_backend == "torchscript":
            return _var_alpha_smooth_sigmoid(
                gain_raw_linear=gain_raw_linear,
                alpha_a=alpha_a,
                alpha_r=alpha_r,
                k=k_db,
                gate_db=gate_db,
                beta=beta,
            )
        elif smoother_backend == "numba":
            return _var_alpha_smooth_sigmoid_numba(
                gain_raw_linear=gain_raw_linear,
                alpha_a=alpha_a,
                alpha_r=alpha_r,
                k=k_db,
                gate_db=gate_db,
                beta=beta,
            )
        else:
            raise ValueError(f"Unknown smoother_backend: {smoother_backend}
                             (expected 'torchscript' or 'numba')")

    raise ValueError(f"Unknown ar_mode: {ar_mode}")

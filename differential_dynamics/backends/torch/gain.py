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

from typing import Union

import numpy as np
import torch
from numba import njit, prange

from differential_dynamics.benchmarks.bench_utilities import gain_db, np_gain_db_scalar
from differential_dynamics.backends.torch.ssl_smoother_ext import ssl2_smoother
from third_party.torchcomp_core.torchcomp import amp2db, db2amp, compexp_gain

# Prefer compiled C++ smoother for sigmoid gating; falls back to Python if build fails
from .sigmoid_smoother_ext import sigmoid_smoother


@torch.jit.script
def _var_alpha_smooth_sigmoid(
    gain_raw_linear: torch.Tensor,  # (B, T) target gain (linear)
    alpha_a: torch.Tensor,  # (B,) attack coeff in (0,1)
    alpha_r: torch.Tensor,  # (B,) release coeff in (0,1)
    k: float,  # gate sharpness
) -> torch.Tensor:
    """
    Variable-α one-pole smoother with a sigmoid gate on the gain trajectory.

    Recurrence (per batch b, time t):
        s_t   = sigmoid(k * diff_t)
        α_t   = s_t * α_attack + (1 - s_t) * α_release
        y_t   = (1 - α_t) * y_{t-1} + α_t * g_tgt_t

    where diff_t is the difference between target gain and current smoothed gain.

    Args:
      gain_raw_linear: target gain g_tgt (B, T), linear scale in (0, 1].
      alpha_a, alpha_r: per-batch coefficients in (0,1) from ms2coef(at_ms/rt_ms).
      k: gate sharpness. Larger k sharpens the transition; calibrate via dB width.

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
        # Gate on gain trajectory in dB
        diff = gain_db(gain_raw_linear_now) - gain_db(prev_smoothed_gain)
        s = torch.sigmoid(k * diff)
        alpha_t = s * alpha_r + (1.0 - s) * alpha_a
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
    alpha_a: np.ndarray,  # (B,)   float32
    alpha_r: np.ndarray,  # (B,)   float32
    k: float,  # scalar
) -> np.ndarray:
    B, T = gain_raw_linear.shape
    y = np.empty_like(gain_raw_linear)
    for b in prange(B):
        prev_smoothed_gain = 1.0
        s_prev = 0.0
        a = alpha_a[b]
        r = alpha_r[b]
        for t in range(T):
            gain_raw_linear_now = gain_raw_linear[b, t]
            diff = np_gain_db_scalar(gain_raw_linear_now) - np_gain_db_scalar(
                prev_smoothed_gain
            )
            s = 1.0 / (1.0 + np.exp(-k * diff))
            alpha_t = s * r + (1.0 - s) * a
            prev_smoothed_gain = (
                1.0 - alpha_t
            ) * prev_smoothed_gain + alpha_t * gain_raw_linear_now
            y[b, t] = prev_smoothed_gain
            s_prev = s
    return y


def _var_alpha_smooth_sigmoid_numba(
    gain_raw_linear: torch.Tensor,
    alpha_a: torch.Tensor,
    alpha_r: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """
    Numba-backed forward-only implementation of the sigmoid-gated variable-α smoother.
    Use this for fast forward-path benchmarking on CPU. Not suitable for autograd.
    """
    # Ensure contiguous float32 CPU arrays for Numba
    g_np = gain_raw_linear.detach().to(torch.float32).cpu().contiguous().numpy()
    a_np = alpha_a.detach().to(torch.float32).cpu().contiguous().numpy()
    r_np = alpha_r.detach().to(torch.float32).cpu().contiguous().numpy()

    y_np = _var_alpha_smooth_sigmoid_numba_core(g_np, a_np, r_np, float(k))
    y = torch.from_numpy(y_np).to(gain_raw_linear.device, dtype=gain_raw_linear.dtype)
    return y


# ------------------------- SSL Numba smoother (forward-only) -------------------------
@njit(parallel=True, fastmath=True)
def _SSL_smooth_numba_core(
    gain_raw_dB: np.ndarray,  # (B, T) float32
    af_np: np.ndarray,  # (B,)   float32
    as_np: np.ndarray,  # (B,)   float32
    sf_np: np.ndarray,  # (B,)   float32
    ss_np: np.ndarray,  # (B,)   float32
    k: float,  # scalar
    fs: float,  # scalar
) -> np.ndarray:

    B, T = gain_raw_dB.shape
    y = np.empty_like(gain_raw_dB)
    Ts = 1.0 / fs
    for b in prange(B):
        prev_smoothed_gain_dB = 0
        af_ = 1.0 / af_np[b]
        as_ = 1.0 / as_np[b]
        shunt_f_ = 1.0 / sf_np[b]
        shunt_s_ = 1.0 / ss_np[b]
        state = np.zeros(2)
        for t in range(T):
            gain_raw_dB_now = gain_raw_dB[b, t]
            diff = gain_raw_dB_now - prev_smoothed_gain_dB
            s = 1.0 / (1.0 + np.exp(-k * diff))
            series_f_ = (1 - s) * af_
            series_s_ = (1 - s) * as_
            Ad, Bd = zoh_discretize_step(series_f_, series_s_, shunt_f_, shunt_s_, Ts)

            # state update
            state = Ad @ state + Bd * gain_raw_dB_now
            prev_smoothed_gain_dB = np.sum(state)  # [1 1] * state
            y[b, t] = prev_smoothed_gain_dB
    return y


@njit
def zoh_discretize_step(series_f_, series_s_, shunt_f_, shunt_s_, Ts):
    # Vin --[ Ra ]-- Vout --[ Rf || Cf ]-- Vm --[ Rs || Cs ]-- GND
    # States(two-terminal capacitor drops): x1 = Vout - Vm(fast), x2 = Vm(slow)
    # Output: y = Vout = x1 + x2
    # Series current: I = (Vin - y) / Ra
    # Cf * dx1 / dt = I - x1 / Rf
    # Cs * dx2 / dt = I - x2 / Rs

    # = > dx1 / dt = (1 / (Cf * Ra)) * Vin - (1 / (Cf * Ra)) * x1 - (1 / (Cf * Ra)) * x2 - (1 / (Cf * Rf)) * x1
    #     dx2 / dt = (1 / (Cs * Ra)) * Vin - (1 / (Cs * Ra)) * x1 - (1 / (Cs * Ra)) * x2 - (1 / (Cs * Rs)) * x2

    # Continuous-time A and B
    A = np.array(
        [[-(series_f_ + shunt_f_), -series_f_], [-(series_s_), -(series_s_ + shunt_s_)]]
    )
    B = np.array([[series_f_], [series_s_]])

    # Augmented matrix
    M = np.zeros((3, 3))
    M[0:2, 0:2] = A
    M[0:2, 2] = B[:, 0]

    # Matrix exponential
    F = expm(M * Ts)

    Ad = F[0:2, 0:2]
    Bd = F[0:2, 2]

    return Ad, Bd


def _SSL_smooth_numba(
    gain_raw_dB: torch.Tensor,
    T_attack_fast_t: torch.Tensor,
    T_attack_slow_t: torch.Tensor,
    T_shunt_fast_t: torch.Tensor,
    T_shunt_slow_t: torch.Tensor,
    k: float,
    fs: float,
) -> torch.Tensor:
    """
    Numba-backed forward-only implementation of the sigmoid-gated variable-α smoother.
    Use this for fast forward-path benchmarking on CPU. Not suitable for autograd.
    """
    # Ensure contiguous float32 CPU arrays for Numba
    g_np = gain_raw_dB.detach().to(torch.float32).cpu().contiguous().numpy()
    af_np = T_attack_fast_t.detach().to(torch.float32).cpu().contiguous().numpy()
    as_np = T_attack_slow_t.detach().to(torch.float32).cpu().contiguous().numpy()
    sf_np = T_shunt_fast_t.detach().to(torch.float32).cpu().contiguous().numpy()
    ss_np = T_shunt_slow_t.to(torch.float32).cpu().contiguous().numpy()

    y_np = _SSL_smooth_numba_core(g_np, af_np, as_np, sf_np, ss_np, float(k), fs)
    y = torch.from_numpy(y_np).to(gain_raw_dB.device, dtype=gain_raw_dB.dtype)
    return y


def compexp_gain_mode(
    x_rms: torch.Tensor,
    comp_thresh: Union[torch.Tensor, float],
    comp_ratio: Union[torch.Tensor, float],
    exp_thresh: Union[torch.Tensor, float],
    exp_ratio: Union[torch.Tensor, float],
    alpha_a: Union[torch.Tensor, float],
    alpha_r: Union[torch.Tensor, float],
    ar_mode: str = "hard",
    k: float = 1.0,
    smoother_backend: str = "torchscript",  # "torchscript" | "numba"
) -> torch.Tensor:
    """
    Switchable compressor/expander gain function using attack/release coefficients.

    This keeps the detector semantics and the dB static curve identical to
    torchcomp and swaps only the A/R gating/smoothing policy via `ar_mode`.

    Modes:
      - "hard": delegates to torchcomp.compexp_gain (baseline; preserves backward)
      - "sigmoid": smooth gate between attack/release based on gain trajectory

    Args:
      x_rms: (B, T) non-negative detector signal (e.g., amplitude EMA of |x|).
      comp_thresh / exp_thresh: thresholds in dB.
      comp_ratio: > 1 (downward compression above comp_thresh).
      exp_ratio: in (0,1) (downward expansion below exp_thresh).
      alpha_a / alpha_r: attack/release one-pole coefficients in (0,1).
      ar_mode: "hard" | "sigmoid".
      k: float, gate sharpness for sigmoid mode.
      smoother_backend: "torchscript" (default) or "numba" (forward-only CPU).

    Returns:
      (B, T) linear gain to apply to the signal.
    """
    B, T = x_rms.shape
    dev, dt = x_rms.device, x_rms.dtype

    if ar_mode == "hard":
        return compexp_gain(
            x_rms=x_rms,
            comp_thresh=comp_thresh,
            comp_ratio=comp_ratio,
            exp_thresh=exp_thresh,
            exp_ratio=exp_ratio,
            at=torch.as_tensor(alpha_a, device=dev, dtype=dt).expand(B),
            rt=torch.as_tensor(alpha_r, device=dev, dtype=dt).expand(B),
        )

    # Compute static curve in dB with clamping (identical to torchcomp)
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

    alpha_a_t = torch.as_tensor(alpha_a, device=dev, dtype=dt).expand(B)
    alpha_r_t = torch.as_tensor(alpha_r, device=dev, dtype=dt).expand(B)

    if ar_mode == "sigmoid":
        if smoother_backend == "torchscript":
            # Prefer C++ extension if available; falls back to Python loop if not.
            # This replaces the TorchScript smoother to avoid building a large autograd graph.
            return sigmoid_smoother(gain_raw_linear, alpha_a_t, alpha_r_t, float(k))
        elif smoother_backend == "numba":
            return _var_alpha_smooth_sigmoid_numba(
                gain_raw_linear=gain_raw_linear,
                alpha_a=alpha_a_t,
                alpha_r=alpha_r_t,
                k=k,
            )
        else:
            raise ValueError(
                f"Unknown smoother_backend: {smoother_backend} (expected 'torchscript' or 'numba')"
            )
    else:
        raise ValueError(f"Unknown ar_mode: {ar_mode}")


def SSL_comp_gain(
    x_peak_dB: torch.Tensor,
    comp_thresh: Union[torch.Tensor, float],
    comp_ratio: Union[torch.Tensor, float],
    T_attack_fast: Union[torch.Tensor, float],
    T_attack_slow: Union[torch.Tensor, float],
    T_shunt_fast: Union[torch.Tensor, float],
    T_shunt_slow: Union[torch.Tensor, float],
    fs: float,
    k: float = 1.0,
    smoother_backend: str = "torchscript",  # "torchscript" | "numba"
) -> torch.Tensor:
    """
    SSL-style compressor gain function

    Args:
      x_peak_dB: (B, T) this should be 20log10(|x|).
      comp_thresh: threshold in dB.
      comp_ratio: > 1 (downward compression above comp_thresh).
      T_attack_fast: fast attack time constant
      T_attack_slow: slow attack time constant
      T_shunt_fast: fast release time constant
      T_shunt_slow: slow release time constant
      k: float, gate sharpness for sigmoid mode.

    Returns:
      (B, T) dB gain to apply to the signal.
    """
    B, T = x_peak_dB.shape
    dev, dt = x_peak_dB.device, x_peak_dB.dtype

    # Compute static curve in dB with clamping (identical to torchcomp)
    comp_thresh_t = torch.as_tensor(comp_thresh, device=dev, dtype=dt).expand(B)
    comp_ratio_t = torch.as_tensor(comp_ratio, device=dev, dtype=dt).expand(B)

    comp_slope = 1.0 - 1.0 / comp_ratio_t

    gain_raw_dB = (
        comp_slope[:, None] * (comp_thresh_t[:, None] - x_peak_dB).neg().relu().neg()
    )

    T_attack_fast_t = torch.as_tensor(T_attack_fast, device=dev, dtype=dt).expand(B)
    T_attack_slow_t = torch.as_tensor(T_attack_slow, device=dev, dtype=dt).expand(B)
    T_shunt_fast_t = torch.as_tensor(T_shunt_fast, device=dev, dtype=dt).expand(B)
    T_shunt_slow_t = torch.as_tensor(T_shunt_slow, device=dev, dtype=dt).expand(B)

    y_db = ssl2_smoother(
        gain_raw_dB,
        T_attack_fast_t,
        T_attack_slow_t,
        T_shunt_fast_t,
        T_shunt_slow_t,
        k=k,
        fs=fs,
    )
    return y_db

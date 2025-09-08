from typing import Union

import torch
import torch.nn.functional as F
from torchlpc import sample_wise_lpc

from .core import compressor_core

__all__ = [
    "compexp_gain",
    "limiter_gain",
    "ms2coef",
    "avg",
    "amp2db",
    "db2amp",
    "coef2ms",
]


def amp2db(x: torch.Tensor) -> torch.Tensor:
    """Convert amplitude to decibels using 20*log10(x).

    Note: x must be strictly positive to avoid -inf. Upstream callers enforce
    this (e.g., assert x_rms > 0). Add an epsilon externally if needed.
    """
    return 20 * torch.log10(x)


def db2amp(x: torch.Tensor) -> torch.Tensor:
    """Convert decibels to linear amplitude via 10 ** (x / 20)."""
    return 10 ** (x / 20)


def ms2coef(ms: torch.Tensor, sr: int) -> torch.Tensor:
    """Map milliseconds to a one-pole smoothing coefficient in (0, 1).

    Uses a 10–90% envelope-time convention via the factor 2200/sr. This differs
    from the more common exponential Euler mapping alpha = exp(-1/(tau*fs)) but
    is consistent across the API when paired with coef2ms.
    """
    return 1 - torch.exp(-2200 / ms / sr)


def coef2ms(coef: torch.Tensor, sr: int) -> torch.Tensor:
    """Inverse of ms2coef: map smoothing coefficient back to milliseconds."""
    return -2200 / (sr * torch.log(1 - coef))


def avg(rms: torch.Tensor, avg_coef: Union[torch.Tensor, float]):
    """Compute a running average (one-pole IIR) of a non-negative sequence.

    The recursion is: y[n] = (1 - a) * y[n-1] + a * x[n], with a = avg_coef.
    Implemented via sample_wise_lpc for numerical efficiency over batches.
    """

    avg_coef = torch.as_tensor(
        avg_coef, dtype=rms.dtype, device=rms.device
    ).broadcast_to(rms.shape[0])
    assert torch.all(avg_coef > 0) and torch.all(avg_coef <= 1)

    return sample_wise_lpc(
        rms * avg_coef.unsqueeze(1),
        avg_coef[:, None, None].broadcast_to(rms.shape + (1,)) - 1,
    )


def compexp_gain(
    x_rms: torch.Tensor,
    comp_thresh: Union[torch.Tensor, float],
    comp_ratio: Union[torch.Tensor, float],
    exp_thresh: Union[torch.Tensor, float],
    exp_ratio: Union[torch.Tensor, float],
    at: Union[torch.Tensor, float],
    rt: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Compressor/Expander gain computer (dB-domain static curve + smoothing).

    Pipeline:
      1) Convert x_rms to dB.
      2) Compute a piecewise-linear dB gain g_db using comp/exp slopes.
      3) Clamp g_db <= 0 via (-x).relu().neg() pattern.
      4) Convert to linear factor and smooth with hard A/R one-pole.

    Returns linear gain (B, T) to multiply with the signal.
    """
    device, dtype = x_rms.device, x_rms.dtype
    factory_func = lambda x: torch.as_tensor(
        x, device=device, dtype=dtype
    ).broadcast_to(x_rms.shape[0])
    comp_ratio = factory_func(comp_ratio)
    exp_ratio = factory_func(exp_ratio)
    comp_thresh = factory_func(comp_thresh)
    exp_thresh = factory_func(exp_thresh)
    at = factory_func(at)
    rt = factory_func(rt)

    # Domain/sanity checks. x_rms must be > 0 to avoid log10 issues.
    assert torch.all(x_rms > 0)
    assert torch.all(comp_ratio > 1)
    # exp_ratio < 1 corresponds to downward expansion below exp_thresh.
    assert torch.all(exp_ratio <= 1) and torch.all(exp_ratio > 0)
    assert torch.all(at > 0) and torch.all(at < 1)
    assert torch.all(rt > 0) and torch.all(rt < 1)

    comp_slope = 1 - 1 / comp_ratio  # slope in dB for the compressor branch
    exp_slope = 1 - 1 / exp_ratio  # slope in dB for the expander branch

    log_x_rms = amp2db(x_rms)
    # Minimum of the two linear segments, then clamp positive parts to 0 dB.
    g = (
        torch.minimum(
            comp_slope[:, None] * (comp_thresh[:, None] - log_x_rms),
            exp_slope[:, None] * (exp_thresh[:, None] - log_x_rms),
        )
        .neg()
        .relu()
        .neg()
    )
    f = db2amp(g)
    # Smoothing initial state: 1.0 gain.
    zi = x_rms.new_ones(f.shape[0])
    return compressor_core(f, zi, at, rt)


def limiter_gain(
    x: torch.Tensor,
    threshold: torch.Tensor,
    at: torch.Tensor,
    rt: torch.Tensor,
) -> torch.Tensor:
    """Peak limiter gain computer.

    This uses the same one-pole with hard A/R for both peak detection and gain
    smoothing. The peak detector runs on |x| with (rt, at) swapped (fast attack,
    slow release), then we compute the instantaneous limiting factor f = min(1,
    T_lin / x_peak). Finally we smooth f with (at, rt).
    """
    assert torch.all(threshold <= 0)
    assert torch.all(at > 0) and torch.all(at < 1)
    assert torch.all(rt > 0) and torch.all(rt < 1)

    factory_func = lambda h: torch.as_tensor(
        h, device=x.device, dtype=x.dtype
    ).broadcast_to(x.shape[0])
    threshold = factory_func(threshold)
    at = factory_func(at)
    rt = factory_func(rt)

    zi = x.new_zeros(x.shape[0])
    lt = db2amp(threshold)
    # Peak detection: note the swapped coefficients (rt, at)
    x_peak = compressor_core(x.abs(), zi, rt, at)
    # f = clamp(T_lin / x_peak, 0, 1) written branchlessly
    f = F.relu(1 - lt[:, None] / x_peak).neg() + 1
    return compressor_core(f, torch.ones_like(zi), at, rt)

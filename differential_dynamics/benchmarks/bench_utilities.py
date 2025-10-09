import torch
from torchaudio.functional import lfilter
import numpy as np

try:
    from numba import njit
except Exception:  # numba is optional at runtime

    def njit(*args, **kwargs):
        def wrap(f):
            return f

        return wrap


def gain_db(g: torch.Tensor, eps_amp: float = 1e-7) -> torch.Tensor:
    """Convert linear gain to dB with a small floor for stability.
    g: (B,T)
    """
    return 20.0 * torch.log10(torch.clamp(g, min=eps_amp))


def db_gain(db: torch.Tensor) -> torch.Tensor:
    """Convert dB to linear gain.
    g: (B,T)
    """
    return torch.pow(10.0, db / 20.0)


@njit(cache=True)
def np_gain_db_scalar(x: float, eps_amp: float = 1e-7) -> float:
    """Numba-friendly scalar version of gain in dB: 20*log10(max(x, eps)).
    Intended for use inside Numba kernels to match torch-side gain_db.
    """
    v = x if x > eps_amp else eps_amp
    return 20.0 * np.log10(v)


def rmse_db(
    g_ref: torch.Tensor,
    g_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps_amp: float = 1e-7,
) -> torch.Tensor:
    """RMSE between gain traces in dB. If mask is provided, compute over mask only."""
    gd_ref = gain_db(g_ref, eps_amp)
    gd_hat = gain_db(g_hat, eps_amp)
    diff = gd_ref - gd_hat
    if mask is not None:
        diff = diff[mask]
    return torch.sqrt(torch.mean(diff**2))


def active_mask_from_env(env: torch.Tensor, thresh_db: float = -100.0) -> torch.Tensor:
    """Mask where envelope is above a floor (avoid counting deep silence)."""
    env_db = 20.0 * torch.log10(torch.clamp(env, min=10 ** (thresh_db / 20.0)))
    return env_db > thresh_db


def rise_time_63(g: torch.Tensor, t0_idx: int) -> int | None:
    """Return samples to reach 63% of step response, or None if not reached. g: (T,)"""
    if g.ndim != 1 or t0_idx >= g.shape[0] - 1:
        return None
    y0 = g[t0_idx]
    tail = g[t0_idx:]
    if tail.numel() <= 1:
        return None
    y1 = torch.max(tail)
    target = y0 + 0.632 * (y1 - y0)
    after = tail
    idx = torch.where(after >= target)[0]
    return int(idx[0].item()) if idx.numel() else None


def ema_1pole_lfilter(x: torch.Tensor, alpha) -> torch.Tensor:
    """One-pole IIR smoothing (exponential moving average) via lfilter.
    Use this instead of avg() for efficiency in benchmarks."""
    # y[n] = (1 - alpha) y[n-1] + alpha x[n]
    alpha = float(alpha) if not torch.is_tensor(alpha) else alpha.item()
    a = torch.tensor([1.0, -(1.0 - alpha)], device=x.device, dtype=x.dtype)  # a0, a1
    b = torch.tensor([alpha, 0], device=x.device, dtype=x.dtype)  # b0, b1
    y = lfilter(x.abs().unsqueeze(1), a, b).squeeze(1)
    return y

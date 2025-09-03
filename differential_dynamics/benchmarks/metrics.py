import torch
import torch.nn.functional as F

from third_party.torchcomp_core.torchcomp import ms2coef, avg


def gain_db(g: torch.Tensor, eps_amp: float = 1e-7) -> torch.Tensor:
    """Convert linear gain to dB with a small floor for stability.
    g: (B,T)
    """
    return 20.0 * torch.log10(torch.clamp(g, min=eps_amp))


def rmse_db(g_ref: torch.Tensor, g_hat: torch.Tensor, mask: torch.Tensor | None = None, eps_amp: float = 1e-7) -> torch.Tensor:
    """RMSE between gain traces in dB. If mask is provided, compute over mask only."""
    gd_ref = gain_db(g_ref, eps_amp)
    gd_hat = gain_db(g_hat, eps_amp)
    diff = gd_ref - gd_hat
    if mask is not None:
        diff = diff[mask]
    return torch.sqrt(torch.mean(diff**2))


def envelope_amplitude(x: torch.Tensor, alpha_det: torch.Tensor | float) -> torch.Tensor:
    """Amplitude EMA detector (consistent with repo). x: (B,T)."""
    alpha = torch.as_tensor(alpha_det, device=x.device, dtype=x.dtype)
    return avg(x.abs(), alpha)


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


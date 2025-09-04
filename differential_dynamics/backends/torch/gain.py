import torch
from typing import Optional, Dict, Union

from third_party.torchcomp_core.torchcomp import amp2db, db2amp, ms2coef, compexp_gain


@torch.jit.script
def _var_alpha_smooth_sigmoid(
    g_tgt: torch.Tensor,       # (B, T) target gain (linear)
    alpha_a: torch.Tensor,     # (B,) attack coeff in (0,1)
    alpha_r: torch.Tensor,     # (B,) release coeff in (0,1)
    k: float,                  # gate sharpness (linear or dB domain)
    gate_db: bool,             # True to gate on dB gain difference
    beta: float = 0.0          # optional EMA on gate (0 disables)
) -> torch.Tensor:
    B = g_tgt.size(0)
    T = g_tgt.size(1)
    y = torch.empty_like(g_tgt)
    prev = torch.ones(B, dtype=g_tgt.dtype, device=g_tgt.device)
    s_prev = torch.zeros(B, dtype=g_tgt.dtype, device=g_tgt.device)
    eps = 1e-7
    for t in range(T):
        gt = g_tgt[:, t]
        if gate_db:
            # Gate on gain trajectory in dB
            diff = 20.0 * torch.log10(torch.clamp(gt, min=eps)) - 20.0 * torch.log10(torch.clamp(prev, min=eps))
        else:
            # Gate on linear difference
            diff = gt - prev
        s = torch.sigmoid(k * diff)
        if beta > 0.0:
            s = (1.0 - beta) * s_prev + beta * s
        alpha_t = s * alpha_a + (1.0 - s) * alpha_r
        prev = (1.0 - alpha_t) * prev + alpha_t * gt
        y[:, t] = prev
        s_prev = s
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
) -> torch.Tensor:
    """
    Switchable compressor/expander gain function that keeps detector and static curve
    identical to torchcomp, and swaps only the A/R gating/smoothing.

    Modes:
      - "hard": exact baseline via torchcomp.compexp_gain
      - "sigmoid": smooth gate between attack/release based on gain trajectory

    Args mirror torchcomp.compexp_gain, with additional ar_mode and gate_cfg.
    Returns linear gain (B, T).
    """
    if ar_mode == "hard":
        # Delegate entirely to torchcomp baseline (identical forward/backward)
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

    # Compute static curve exactly like torchcomp
    B, T = x_rms.shape
    dev, dt = x_rms.device, x_rms.dtype
    comp_thresh_t = torch.as_tensor(comp_thresh, device=dev, dtype=dt).expand(B)
    exp_thresh_t  = torch.as_tensor(exp_thresh,  device=dev, dtype=dt).expand(B)
    comp_ratio_t  = torch.as_tensor(comp_ratio,  device=dev, dtype=dt).expand(B)
    exp_ratio_t   = torch.as_tensor(exp_ratio,   device=dev, dtype=dt).expand(B)

    comp_slope = 1.0 - 1.0 / comp_ratio_t
    exp_slope  = 1.0 - 1.0 / exp_ratio_t

    L = amp2db(torch.clamp(x_rms, min=1e-7))
    g_db = torch.minimum(
        comp_slope[:, None] * (comp_thresh_t[:, None] - L),
        exp_slope[:, None]  * (exp_thresh_t[:, None]  - L),
    ).neg().relu().neg()
    g_tgt = db2amp(g_db)

    # Time constants
    alpha_a = ms2coef(torch.as_tensor(at_ms, device=dev, dtype=dt), fs).expand(B)
    alpha_r = ms2coef(torch.as_tensor(rt_ms, device=dev, dtype=dt), fs).expand(B)

    # Gate config
    k_db = 2.0
    beta = 0.0
    gate_db = True
    if gate_cfg is not None:
        if "k_db" in gate_cfg and gate_cfg["k_db"] is not None:
            k_db = float(gate_cfg["k_db"])  # approximate 4.4 / width_dB
        if "beta" in gate_cfg and gate_cfg["beta"] is not None:
            beta = float(gate_cfg["beta"])   # small gate smoothing (0.0..0.3)
        if "gate_db" in gate_cfg and gate_cfg["gate_db"] is not None:
            gate_db = bool(gate_cfg["gate_db"])  # True to gate in dB domain

    if ar_mode == "sigmoid":
        return _var_alpha_smooth_sigmoid(g_tgt=g_tgt, alpha_a=alpha_a, alpha_r=alpha_r, k=k_db, gate_db=gate_db, beta=beta)

    raise ValueError(f"Unknown ar_mode: {ar_mode}")


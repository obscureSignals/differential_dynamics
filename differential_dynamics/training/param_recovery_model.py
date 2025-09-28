# param_recovery_model.py
"""
Model and utilities for parameter recovery experiments (non-blind).

Exposes:
- GlobalParamModel: global/shared θ across a small dataset with stable parameterization
- linear_anneal: simple linear schedule utility
"""
from __future__ import annotations

from typing import Dict

import math
import torch
from torch import nn
from third_party.torchcomp_core.torchcomp import ms2coef, coef2ms, compexp_gain
from differential_dynamics.backends.torch.gain import compexp_gain_mode


class GlobalParamModel(nn.Module):
    """Global, shared learnable parameters θ for a small dataset.

    Parameterization for stability:
      - comp_ratio = exp(ratio_logit) + 1  (enforces > 1)
      - at/rt coefficients in (0,1) via sigmoid(logit)
      - comp_thresh is a free parameter in dB
    """

    def __init__(self, fs: int, init: Dict[str, float] | None = None):
        super().__init__()
        if fs is None or fs <= 0:
            raise ValueError(f"GlobalParamModel: fs must be > 0, got {fs}")
        init = init or {"CT": -24.0, "CR": 4.0, "AT_MS": 10.0, "RT_MS": 100.0}
        # Threshold in dB
        self.comp_thresh = nn.Parameter(torch.tensor(float(init["CT"])) )
        # Ratio as exp(logit)+1 to enforce >1
        self.ratio_logit = nn.Parameter(torch.log(torch.tensor(float(init["CR"])) - 1.0))
        # Attack/Release time constants in (0,1) via sigmoid(logit)
        def ms2coef_scalar(ms: float) -> float:
            coef = ms2coef(torch.tensor(ms, dtype=torch.float32), fs).item()
            # Clamp away from exact 0/1 to avoid inf/NaN in logit and boundary issues downstream
            eps = 1e-6
            return float(min(max(coef, eps), 1.0 - eps))
        self.at_logit = nn.Parameter(torch.logit(torch.tensor(ms2coef_scalar(float(init["AT_MS"]))), eps=1e-6))
        self.rt_logit = nn.Parameter(torch.logit(torch.tensor(ms2coef_scalar(float(init["RT_MS"]))), eps=1e-6))
        self.fs = fs

    def params_readable(self) -> Dict[str, float]:
        """Return current parameters in human units for logging/reporting."""
        ratio_t = self.ratio_logit.exp() + 1.0
        ratio_t = torch.nan_to_num(ratio_t, nan=torch.tensor(2.0, device=ratio_t.device, dtype=ratio_t.dtype))
        ratio = float(torch.clamp_min(ratio_t, 1.0 + 1e-4).item())
        # Clamp coefficients away from 0/1 and handle NaNs before converting back to ms
        eps = 1e-6
        at_coef = torch.sigmoid(self.at_logit)
        rt_coef = torch.sigmoid(self.rt_logit)
        at_coef = torch.nan_to_num(at_coef, nan=torch.tensor(0.5, device=at_coef.device, dtype=at_coef.dtype))
        rt_coef = torch.nan_to_num(rt_coef, nan=torch.tensor(0.5, device=rt_coef.device, dtype=rt_coef.dtype))
        at_coef = torch.clamp(at_coef, eps, 1.0 - eps)
        rt_coef = torch.clamp(rt_coef, eps, 1.0 - eps)
        at_ms = coef2ms(at_coef, self.fs).item()
        rt_ms = coef2ms(rt_coef, self.fs).item()
        return {
            "comp_thresh_db": float(self.comp_thresh.item()),
            "comp_ratio": float(ratio),
            "attack_ms": float(at_ms),
            "release_ms": float(rt_ms),
        }

    def forward(self, x_rms: torch.Tensor, ar_mode: str, k: float | None = None) -> torch.Tensor:
        """Predict gain using either the hard or sigmoid smoother.

        - Hard mode consumes A/R coefficients directly (as torchcomp expects).
        - Sigmoid mode consumes A/R times in ms (as compexp_gain_mode expects) and takes k.
        """
        ct = self.comp_thresh
        cr = self.ratio_logit.exp() + 1.0
        at_coef = torch.sigmoid(self.at_logit)
        rt_coef = torch.sigmoid(self.rt_logit)
        # Guard against NaNs and boundary values before calling simulators
        eps = 1e-6
        cr_safe = torch.nan_to_num(cr, nan=torch.tensor(2.0, device=cr.device, dtype=cr.dtype))
        cr_safe = torch.clamp_min(cr_safe, 1.0 + 1e-4)
        at_coef = torch.nan_to_num(at_coef, nan=torch.tensor(0.5, device=at_coef.device, dtype=at_coef.dtype))
        rt_coef = torch.nan_to_num(rt_coef, nan=torch.tensor(0.5, device=rt_coef.device, dtype=rt_coef.dtype))
        at_coef = torch.clamp(at_coef, eps, 1.0 - eps)
        rt_coef = torch.clamp(rt_coef, eps, 1.0 - eps)

        if ar_mode == "hard":
            # Use torchcomp baseline directly for the smoother (preserves custom backward)
            return compexp_gain(
                x_rms=x_rms,
                comp_thresh=ct,
                comp_ratio=cr_safe,
                exp_thresh=-1000.0,
                exp_ratio=1.0,
                at=at_coef,
                rt=rt_coef,
            )

        if ar_mode == "sigmoid":
            if k is None:
                raise ValueError("k must be provided for sigmoid mode")
            return compexp_gain_mode(
                x_rms=x_rms,
                comp_thresh=ct,
                comp_ratio=cr_safe,
                exp_thresh=-1000.0,
                exp_ratio=1.0,
                alpha_a=at_coef,
                alpha_r=rt_coef,
                ar_mode="sigmoid",
                k=float(k),
                smoother_backend="torchscript",
            )

        raise ValueError(f"Unknown ar_mode: {ar_mode}")


def linear_anneal(start: float, end: float, step: int, total_steps: int) -> float:
    """Linear schedule from start to end over total_steps.

    Returns end if total_steps <= 0. Clamps step to [0, total_steps].
    """
    if total_steps <= 0:
        return end
    t = min(max(step, 0), total_steps)
    return float(start + (end - start) * (t / total_steps))


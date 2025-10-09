# param_recovery_model_sll.py
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
from differential_dynamics.backends.torch.gain import SSL_comp_gain


class ParamRecoveryModelSLL(nn.Module):
    """Global, shared learnable parameters θ for SSL-style parameter recovery.

    Differences vs old compressor model:
      - Time constants are parameterized directly in seconds (bounded via sigmoid).
      - Model consumes x_peak_dB (20*log10(|x|)) and outputs y_dB (gain in dB).
      - Forward uses SSL_comp_gain (hard gate only for backward support).
    """

    def __init__(self, fs: int, init: Dict[str, float] | None = None):
        super().__init__()
        if fs is None or fs <= 0:
            raise ValueError(f"ParamRecoveryModelSLL: fs must be > 0, got {fs}")
        self.fs = float(fs)

        # Defaults (seconds) and other params
        init = init or {
            "CT": -24.0,  # comp_thresh in dB
            "CR": 4.0,  # comp_ratio (>1)
            "T_AF": 0.010,  # attack fast (s)
            "T_AS": 0.050,  # attack slow (s)
            "T_SF": 0.030,  # shunt fast (s)
            "T_SS": 0.200,  # shunt slow (s)
            "FB": 0.5,  # feedback coeff in [0,1]
        }

        # Reasonable bounds (seconds)
        # +/- 20% on ssl values
        self.T_af_min = 820 * 0.47e-6 * 0.8
        self.T_af_max = 270e3 * 0.47e-6 * 1.2
        self.T_as_min = 820 * 6.8e-6 * 0.8

        # allow slow attack TC to go very large to simulate single time constant settings
        # 100x slowest ssl value
        self.T_as_max = 270e3 * 6.8e-6 * 100

        # +/- 20% on ssl values
        self.T_sf_min = 91e3 * 0.47e-6 * 0.8
        self.T_sf_max = 1.2e6 * 0.47e-6 * 1.2
        self.T_ss_min = 750e3 * 6.8e-6 * 0.8

        # allow slow shunt TC to go very large to simulate single time constant settings
        # 100x ssl value
        self.T_ss_max = 750e3 * 6.8e-6 * 100

        # Learnable parameters
        self.comp_thresh = nn.Parameter(
            torch.tensor(float(init["CT"]), dtype=torch.float32)
        )
        # Ratio as exp(logit)+1 to enforce >1
        self.ratio_logit = nn.Parameter(
            torch.log(torch.tensor(float(init["CR"])) - 1.0)
        )
        # Feedback coeff via sigmoid(u) to [0,1]
        fb_init = float(init.get("FB", 0.5))
        fb_init = min(max(fb_init, 1e-6), 1 - 1e-6)
        self.fb_logit = nn.Parameter(
            torch.logit(torch.tensor(fb_init, dtype=torch.float32))
        )

        # Time constants (seconds) via bounded sigmoid transform
        def _init_u_from_seconds(
            val_s: float, tmin: float, tmax: float
        ) -> torch.Tensor:
            s = (float(val_s) - tmin) / (tmax - tmin)
            s = min(max(s, 1e-6), 1 - 1e-6)
            return torch.logit(torch.tensor(s, dtype=torch.float32))

        self.u_T_af = nn.Parameter(
            _init_u_from_seconds(float(init["T_AF"]), self.T_af_min, self.T_af_max)
        )
        self.u_T_as = nn.Parameter(
            _init_u_from_seconds(float(init["T_AS"]), self.T_as_min, self.T_as_max)
        )
        self.u_T_sf = nn.Parameter(
            _init_u_from_seconds(float(init["T_SF"]), self.T_sf_min, self.T_sf_max)
        )
        self.u_T_ss = nn.Parameter(
            _init_u_from_seconds(float(init["T_SS"]), self.T_ss_min, self.T_ss_max)
        )

    def _to_seconds(self, u: torch.Tensor, tmin: float, tmax: float) -> torch.Tensor:
        return tmin + (tmax - tmin) * torch.sigmoid(u)

    def params_readable(self) -> Dict[str, float]:
        """Return current parameters in human units for logging/reporting."""
        # Ratio
        ratio_t = self.ratio_logit.exp() + 1.0
        ratio_t = torch.nan_to_num(ratio_t, nan=torch.tensor(2.0, dtype=ratio_t.dtype))
        ratio = float(torch.clamp_min(ratio_t, 1.0 + 1e-4).item())
        # Feedback
        fb = torch.sigmoid(self.fb_logit)
        fb = torch.nan_to_num(fb, nan=torch.tensor(0.5, dtype=fb.dtype))
        fb = float(torch.clamp(fb, 0.0, 1.0).item())
        # Time constants (s → ms for display)
        T_af_s = self._to_seconds(self.u_T_af, self.T_af_min, self.T_af_max)
        T_as_s = self._to_seconds(self.u_T_as, self.T_as_min, self.T_as_max)
        T_sf_s = self._to_seconds(self.u_T_sf, self.T_sf_min, self.T_sf_max)
        T_ss_s = self._to_seconds(self.u_T_ss, self.T_ss_min, self.T_ss_max)
        return {
            "comp_thresh_db": float(self.comp_thresh.item()),
            "comp_ratio": ratio,
            "T_attack_fast_ms": float(T_af_s.item() * 1000.0),
            "T_attack_slow_ms": float(T_as_s.item() * 1000.0),
            "T_shunt_fast_ms": float(T_sf_s.item() * 1000.0),
            "T_shunt_slow_ms": float(T_ss_s.item() * 1000.0),
            "feedback_coeff": fb,
        }

    def forward(self, x_peak_dB: torch.Tensor) -> torch.Tensor:
        """Predict gain in dB using the SSL smoother (hard gate, CPU-only).

        Args:
          x_peak_dB: (B, T) 20*log10(|x|) envelope in dB (any device/dtype will be cast).
        Returns:
          y_dB: (B, T) gain in dB (CPU float32).
        """
        # Move to CPU float32 as the extension is CPU-only
        if x_peak_dB.device.type != "cpu":
            x_peak_dB = x_peak_dB.cpu()
        x_peak_dB = x_peak_dB.contiguous().float()

        # Safe parameters
        cr = self.ratio_logit.exp() + 1.0
        cr = torch.nan_to_num(cr, nan=torch.tensor(2.0, dtype=cr.dtype))
        cr = torch.clamp_min(cr, 1.0 + 1e-4)
        fb = torch.sigmoid(self.fb_logit)
        fb = torch.nan_to_num(fb, nan=torch.tensor(0.5, dtype=fb.dtype))
        fb = torch.clamp(fb, 0.0, 1.0)

        # Seconds
        T_af_s = self._to_seconds(self.u_T_af, self.T_af_min, self.T_af_max)
        T_as_s = self._to_seconds(self.u_T_as, self.T_as_min, self.T_as_max)
        T_sf_s = self._to_seconds(self.u_T_sf, self.T_sf_min, self.T_sf_max)
        T_ss_s = self._to_seconds(self.u_T_ss, self.T_ss_min, self.T_ss_max)

        y_dB = SSL_comp_gain(
            x_peak_dB=x_peak_dB,
            comp_thresh=self.comp_thresh,
            comp_ratio=cr,
            T_attack_fast=T_af_s,
            T_attack_slow=T_as_s,
            T_shunt_fast=T_sf_s,
            T_shunt_slow=T_ss_s,
            k=0.0,  # ignored in hard mode
            feedback_coeff=fb,
            fs=self.fs,
            soft_gate=False,
        )
        return y_dB


# Backwards-compat: allow old import name to resolve
GlobalParamModel = ParamRecoveryModelSLL


def linear_anneal(start: float, end: float, step: int, total_steps: int) -> float:
    """Linear schedule from start to end over total_steps.

    Returns end if total_steps <= 0. Clamps step to [0, total_steps].
    """
    if total_steps <= 0:
        return end
    t = min(max(step, 0), total_steps)
    return float(start + (end - start) * (t / total_steps))

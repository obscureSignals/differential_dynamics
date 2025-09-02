import torch
import torch.nn as nn
import torch.nn.functional as F

# Hard (baseline) uses the third_party torchcomp core directly.
# Differentiable modes are implemented in pure Torch for clarity.

from third_party.torchcomp_core.torchcomp.core import compressor_core as _hard_core


def _alpha_from_tau(tau_sec: torch.Tensor, fs: float) -> torch.Tensor:
    tau_sec = torch.as_tensor(tau_sec, device=tau_sec.device, dtype=tau_sec.dtype)
    return torch.exp(-1.0 / (tau_sec * fs))


def _soft_gate(x_env_t, e_prev, k: float = 50.0):
    return torch.sigmoid(k * (x_env_t - e_prev))


class EnvelopeAR(nn.Module):
    """Attack/Release envelope with switchable gating.

    mode: one of {"hard", "sigmoid"}
      - hard: uses upstream hard if/else via torchcomp core
      - sigmoid: fully differentiable blend between attack and release
    """

    def __init__(self, fs: float, mode: str = "hard", k: float = 50.0):
        super().__init__()
        self.fs = float(fs)
        self.mode = mode
        self.k = float(k)

    def forward(self, x_env: torch.Tensor, tau_a: torch.Tensor, tau_r: torch.Tensor) -> torch.Tensor:
        """Compute smoothed envelope given a non-negative detector input x_env (e.g., RMS or |x|).
        Shapes:
          x_env: (B, T), tau_a: (B,) or scalar tensor, tau_r: (B,) or scalar tensor
        Returns:
          e: (B, T)
        """
        assert x_env.ndim == 2
        B, T = x_env.shape
        tau_a = tau_a.expand(B)
        tau_r = tau_r.expand(B)

        if self.mode == "hard":
            # Map tau -> coefficient like the upstream core expects (0<coef<1)
            alpha_a = _alpha_from_tau(tau_a, self.fs)
            alpha_r = _alpha_from_tau(tau_r, self.fs)
            # The upstream core signature is compressor_core(x, zi, at, rt)
            # where it applies y_t = (1-coef)*y_{t-1} + coef*x_t with hard if/else
            zi = torch.zeros(B, device=x_env.device, dtype=x_env.dtype)
            # We pass the target coefficients as at, rt
            return _hard_core(x_env, zi, alpha_a, alpha_r)

        elif self.mode == "sigmoid":
            alpha_a = _alpha_from_tau(tau_a, self.fs).unsqueeze(1)
            alpha_r = _alpha_from_tau(tau_r, self.fs).unsqueeze(1)

            e_prev = torch.zeros(B, device=x_env.device, dtype=x_env.dtype)
            out = []
            k = self.k
            for t in range(T):
                xt = x_env[:, t]
                s = _soft_gate(xt, e_prev, k)
                alpha_t = s * alpha_a.squeeze(1) + (1 - s) * alpha_r.squeeze(1)
                e_t = (1 - alpha_t) * xt + alpha_t * e_prev
                out.append(e_t)
                e_prev = e_t
            return torch.stack(out, dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


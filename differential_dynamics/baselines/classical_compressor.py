from typing import Union
import torch
import numpy as np
from numba import njit, prange

from third_party.torchcomp_core.torchcomp import amp2db, ms2coef, db2amp, avg


class ClassicalCompressor:
    """
    A classical compressor/expander implementation to benchmark against differentiable compressors.
    """

    def __init__(
        self,
        comp_thresh: Union[torch.Tensor, float],
        comp_ratio: Union[torch.Tensor, float],
        exp_thresh: Union[torch.Tensor, float],
        exp_ratio: Union[torch.Tensor, float],
        attack_time_ms: Union[torch.Tensor, float],
        release_time_ms: Union[torch.Tensor, float],
        fs: int = 44100.0,
        detector_time_ms: torch.Tensor = 20.0,
    ):

        self.comp_thresh = comp_thresh
        self.comp_ratio = comp_ratio
        self.exp_thresh = exp_thresh
        self.exp_ratio = exp_ratio
        self.attack_time_ms = attack_time_ms
        self.release_time_ms = release_time_ms
        self.fs = fs
        self.detector_time_ms = detector_time_ms

        self.comp_slope = 1 - 1 / comp_ratio  # slope in dB for the compressor branch
        self.exp_slope = 1 - 1 / exp_ratio  # slope in dB for the expander branch
        self.detector_coeff = ms2coef(torch.as_tensor(detector_time_ms), fs)
        self.at = ms2coef(torch.as_tensor(attack_time_ms), fs)
        self.rt = ms2coef(torch.as_tensor(release_time_ms), fs)

        assert comp_ratio >= 1
        # exp_ratio < 1 corresponds to downward expansion below exp_thresh.
        assert 1 >= exp_ratio > 0
        assert 0 < self.at < 1
        assert 0 < self.rt < 1

    def classical_compexp_gain(self, x_rms):
        """
        Compresses the input data by zeroing out values below the threshold.

        Args:
            x_rms: The input data to be compressed, with moving average already applied (shape (B, T)).

        Returns:
            tensor: linear gains for each sample of the input data (shape (B, T))
        """

        # Domain/sanity checks. x_rms must be > 0 to avoid log10 issues.
        assert torch.all(x_rms > 0)

        # Coerce scalar params to (B,) tensors on the same device/dtype as x_rms
        B, T = x_rms.shape
        self.comp_thresh = torch.as_tensor(
            self.comp_thresh, device=x_rms.device, dtype=x_rms.dtype
        ).broadcast_to(B)
        self.comp_ratio = torch.as_tensor(
            self.comp_ratio, device=x_rms.device, dtype=x_rms.dtype
        ).broadcast_to(B)
        self.exp_thresh = torch.as_tensor(
            self.exp_thresh, device=x_rms.device, dtype=x_rms.dtype
        ).broadcast_to(B)
        self.exp_ratio = torch.as_tensor(
            self.exp_ratio, device=x_rms.device, dtype=x_rms.dtype
        ).broadcast_to(B)
        self.comp_slope = torch.as_tensor(
            self.comp_slope, device=x_rms.device, dtype=x_rms.dtype
        ).broadcast_to(B)
        self.exp_slope = torch.as_tensor(
            self.exp_slope, device=x_rms.device, dtype=x_rms.dtype
        ).broadcast_to(B)
        self.at = torch.as_tensor(
            self.at, device=x_rms.device, dtype=x_rms.dtype
        ).broadcast_to(B)
        self.rt = torch.as_tensor(
            self.rt, device=x_rms.device, dtype=x_rms.dtype
        ).broadcast_to(B)

        # Convert x_rms to dB.
        log_x_rms = amp2db(x_rms)
        # Use the static curves to compute the raw gain in dB.
        # We want to use the lowest value (most gain reduction) from either the compressor or expander branch.
        # With rational thresholds (expander threshold below compressor threshold),
        # only one branch will be non-zero.
        # The static curves are not valid above zero (we are always reducing gain or
        # leaving it unchanged), so we will clamp positive values to zero.
        gain_raw_db = (
            torch.minimum(
                self.comp_slope[:, None] * (self.comp_thresh[:, None] - log_x_rms),
                self.exp_slope[:, None] * (self.exp_thresh[:, None] - log_x_rms),
            )
            .neg()  # invert
            .relu()  # clamp to >= 0
            .neg()  # invert back (now <= 0 dB)
        )

        # Convert to linear gain
        gain_raw_linear = db2amp(gain_raw_db)
        return ar_smooth_numba(gain_raw_linear, self.at, self.rt)

    def compress(self, x):
        """
        Compress the input signal x using the classical compressor.
        Args:
            x: The input signal to be compressed. Shape (B, T)
        Returns:
            tensor: The compressed signal. Shape (B, T)
        """

        # get 'rms' values
        x_rms = avg(x.abs(), self.detector_coeff)
        # get gains
        gains = self.classical_compexp_gain(x_rms)
        # apply gains to input signal and return
        return gains * x


@njit(parallel=True, fastmath=True)
def ar_smooth_numba_core(gain_raw_linear, at, rt):
    B, T = gain_raw_linear.shape
    out = np.empty_like(gain_raw_linear)
    for b in prange(B):
        prev_smoothed_gain = 1.0
        a = at[b]
        r = rt[b]
        for t in range(T):
            gain_raw_linear_now = gain_raw_linear[b, t]
            coeff = a if gain_raw_linear_now < prev_smoothed_gain else r
            prev_smoothed_gain = (
                1.0 - coeff
            ) * prev_smoothed_gain + coeff * gain_raw_linear_now
            out[b, t] = prev_smoothed_gain
    return out


def ar_smooth_numba(
    gain_raw_linear: torch.Tensor, at: torch.Tensor, rt: torch.Tensor
) -> torch.Tensor:
    y = ar_smooth_numba_core(
        gain_raw_linear.cpu().numpy(), at.cpu().numpy(), rt.cpu().numpy()
    )
    return torch.from_numpy(y).to(gain_raw_linear.device, gain_raw_linear.dtype)

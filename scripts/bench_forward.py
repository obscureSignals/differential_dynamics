#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt

from third_party.torchcomp_core.torchcomp import ms2coef, compexp_gain, amp2db, db2amp
from differential_dynamics.Baselines.classical_compressor import ClassicalCompressor
from differential_dynamics.backends.torch.envelope import EnvelopeAR
from differential_dynamics.benchmarks.signals import step, tone, burst, ramp
from differential_dynamics.benchmarks.metrics import envelope_amplitude, active_mask_from_env, rmse_db


def run_bench():
    fs = 48000
    T = fs // 2  # 0.5 s
    B = 1

    # 1) Make a test input (try step; change to ramp/burst/tone for other probes)
    x = step(fs, T, B=B, at=0.25, amp_before=0.1, amp_after=0.8)

    # 2) Shared detector (alpha_det) for fairness
    detector_ms = torch.tensor(20.0)
    alpha_det = ms2coef(detector_ms, fs)
    x_rms = envelope_amplitude(x, alpha_det)

    # 3) Classical baseline (uses its own detector on raw x for y, but we also compute gain from y/x)
    comp = ClassicalCompressor(
        comp_thresh=-24.0,
        comp_ratio=4.0,
        exp_thresh=-60.0,
        exp_ratio=0.5,
        attack_time_ms=10.0,
        release_time_ms=100.0,
        fs=fs,
        detector_time_ms=detector_ms,
    )
    y_classical = comp.compress(x)
    g_classical = (y_classical / (x + 1e-12)).clamp(min=0.0)

    # 4) Variant A: torchcomp gain (hard A/R), same detector
    at = ms2coef(torch.tensor(10.0), fs)
    rt = ms2coef(torch.tensor(100.0), fs)
    g_hard = compexp_gain(
        x_rms=x_rms.clamp_min(1e-7),
        comp_thresh=-24.0, comp_ratio=4.0,
        exp_thresh=-60.0, exp_ratio=0.5,
        at=at, rt=rt
    )

    # 5) Variant B: sigmoid gating envelope + same static curve (forward-only)
    env = EnvelopeAR(fs=fs, mode="sigmoid", k=40.0)
    _ = env(x_rms, torch.tensor(0.010), torch.tensor(0.100))  # computes a smooth envelope (not used directly)
    L = amp2db(x_rms.clamp_min(1e-7))
    comp_slope = (1.0 - 1.0/4.0)
    exp_slope  = (1.0 - 1.0/0.5)
    g_db = torch.minimum(comp_slope*( -24.0 - L), exp_slope*( -60.0 - L)).neg().relu().neg()
    g_sigmoid = db2amp(g_db)

    # 6) Metrics (primary: gain RMSE dB on active regions)
    mask = active_mask_from_env(x_rms, thresh_db=-100.0)
    m_hard = rmse_db(g_classical, g_hard, mask=mask)
    m_sig  = rmse_db(g_classical, g_sigmoid, mask=mask)
    print(f"Gain RMSE dB — hard: {m_hard.item():.3f} dB, sigmoid: {m_sig.item():.3f} dB")

    # 7) Plot gain traces
    t = torch.arange(T)/fs
    gd_classical = 20*torch.log10(g_classical.clamp_min(1e-7))[0].cpu()
    gd_hard      = 20*torch.log10(g_hard.clamp_min(1e-7))[0].cpu()
    gd_sig       = 20*torch.log10(g_sigmoid.clamp_min(1e-7))[0].cpu()

    plt.figure(figsize=(10,5))
    plt.plot(t, gd_classical, label="classical")
    plt.plot(t, gd_hard, '--', label="hard")
    plt.plot(t, gd_sig, ':', label="sigmoid")
    plt.xlabel("time (s)"); plt.ylabel("gain (dB)"); plt.legend(); plt.grid(True)
    plt.title("Gain traces")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_bench()


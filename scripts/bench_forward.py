#!/usr/bin/env python3
import argparse
import logging
import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import torch
import torchaudio

from differential_dynamics.backends.torch.gain import compexp_gain_mode
from differential_dynamics.baselines.classical_compressor import ClassicalCompressor
from differential_dynamics.benchmarks.bench_utilities import (
    active_mask_from_env,
    rmse_db,
    ema_1pole_lfilter,
    gain_db,
)
from differential_dynamics.benchmarks.signals import step, tone, burst, ramp
from third_party.torchcomp_core.torchcomp import (
    ms2coef,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

with torch.no_grad():

    def run_bench(
        test_signal_type: str,
        test_file_path: str = None,
        comp_thresh: Union[float, str] = -24.0,
        comp_ratio: float = 4.0,
        exp_thresh: float = -1000.0,
        exp_ratio: float = 1.0,
        attack_time_ms: float = 10.0,
        release_time_ms: float = 100.0,
        k: float = 1.0,
    ):

        if test_signal_type == "file":
            test_signal, fs = load_audio(test_file_path)
            test_signal = convert_to_mono(test_signal)
            T = fs * 5  # length in samples
            start_trim = (
                fs * 10
            )  # trim first start_trim seconds to avoid initial silence
            test_signal = test_signal[start_trim : start_trim + T]
            test_signal = test_signal.unsqueeze(0)

        else:
            fs = 44100
            T = fs // 2  # 0.5 s
            B = 1  # batch size
            freq = 1000
            if test_signal_type == "tone":
                test_signal = tone(freq=freq, fs=fs, T=T, B=B, amp=0.5)
            elif test_signal_type == "step":
                test_signal = step(
                    fs=fs, T=T, B=B, at=0.25, amp_before=0.01, amp_after=1.0
                )
            elif test_signal_type == "burst":
                test_signal = burst(
                    fs=fs, T=T, B=B, start=0.2, dur=0.1, amp=0.8, freq=freq
                )
            elif test_signal_type == "ramp":
                test_signal = ramp(fs=fs, T=T, B=B, start=0.2, dur=0.4, a0=0.1, a1=0.8)
            else:
                raise ValueError(f"Unknown test signal type: {test_signal_type}")

        # Use the same detector for all compressors
        detector_time_ms = torch.tensor(20.0)  # time constant for detector in ms
        alpha_det = ms2coef(detector_time_ms, fs)  # covert to coefficient
        test_signal_rms = ema_1pole_lfilter(test_signal.abs(), alpha_det)
        test_signal_rms = test_signal_rms.clamp_min(1e-7)  # avoid log(0) issues

        # Set compressor threshold to mean input level if 'auto'
        if isinstance(comp_thresh, str) and comp_thresh.lower() == "auto":
            comp_thresh = torch.mean(gain_db(test_signal_rms))

        # Classical baseline
        comp = ClassicalCompressor(
            comp_thresh=comp_thresh,
            comp_ratio=comp_ratio,
            exp_thresh=exp_thresh,
            exp_ratio=exp_ratio,
            attack_time_ms=attack_time_ms,
            release_time_ms=release_time_ms,
            fs=fs,
            detector_time_ms=detector_time_ms,
        )

        g_classical = comp.classical_compexp_gain(test_signal_rms)

        # # Variant A: torchcomp gain (hard A/R), same detector
        # g_hard = compexp_gain(
        #     x_rms=test_signal_rms.clamp_min(1e-7),
        #     comp_thresh=comp_thresh,
        #     comp_ratio=comp_ratio,
        #     exp_thresh=exp_thresh,
        #     exp_ratio=exp_ratio,
        #     at=ms2coef(torch.tensor(attack_time_ms), fs),
        #     rt=ms2coef(torch.tensor(release_time_ms), fs),
        # )

        # Variant B: sigmoid gating envelope + same static curve (forward-only)
        g_sigmoid = compexp_gain_mode(
            x_rms=test_signal_rms.clamp_min(1e-7),
            comp_thresh=comp_thresh,
            comp_ratio=comp_ratio,
            exp_thresh=exp_thresh,
            exp_ratio=exp_ratio,
            at_ms=attack_time_ms,
            rt_ms=release_time_ms,
            fs=fs,
            ar_mode="sigmoid",
            k=k,
            smoother_backend="numba",
        )

        # Metrics (primary: gain RMSE dB on active regions)
        mask = active_mask_from_env(test_signal_rms, thresh_db=-100.0)
        # m_hard = rmse_db(g_classical, g_hard, mask=mask)
        m_sig = rmse_db(g_classical, g_sigmoid, mask=mask)
        # print(
        #     f"Gain RMSE dB — hard: {m_hard.item():.3f} dB, sigmoid: {m_sig.item():.3f} dB"
        # )
        print(f"Gain RMSE dB — sigmoid: {m_sig.item():.3f} dB")

        # Plot gain traces
        t = torch.arange(T) / fs
        gd_classical = 20 * torch.log10(g_classical.clamp_min(1e-7))[0].cpu()
        # gd_hard = 20 * torch.log10(g_hard.clamp_min(1e-7))[0].cpu()
        gd_sig = 20 * torch.log10(g_sigmoid.clamp_min(1e-7))[0].cpu()
        input_env = 20 * torch.log10(test_signal_rms.clamp_min(1e-7))[0].cpu()

        plt.figure(figsize=(10, 5))
        plt.plot(t, gd_classical, label="Classical")
        # plt.plot(t, gd_hard, "--", label="hard")
        plt.plot(t, gd_sig, label="Sigmoid")
        plt.plot(t, input_env, label="Test Signal Envelope", alpha=0.3)
        # plt.plot(t, test_signal.squeeze(), label="input signal", alpha=0.3)

        plt.axhline(y=comp_thresh, linestyle="--", label="Threshold")
        # plt.text(0, comp_thresh - 2, "Threshold")
        plt.text(
            T / (2 * fs),
            -36,
            f"Gain RMSE: {m_sig.item():.3f} dB",
            horizontalalignment="center",
        )
        plt.xlabel("time (s)")
        plt.ylabel("gain (dB)")
        plt.legend()
        plt.grid(True)
        plt.title("Gain traces")
        plt.ylim(-40.0, 6.0)
        plt.tight_layout()
        plt.show()


def load_audio(file_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and return the audio data and sample rate.

    Args:
        file_path: Path to the audio file

    Returns:
        Tuple containing audio data as PyTorch tensor and sample rate
    """
    try:
        # torchaudio.load returns a tensor of shape [channels, samples]
        waveform, sr = torchaudio.load(file_path, normalize=False)
        logger.info(
            f"Loaded waveform shape: {tuple(waveform.shape)}, dtype: {waveform.dtype}, sr: {sr}"
        )
        return waveform, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def convert_to_mono(audio: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to mono by summing channels if stereo.

    Args:
        audio: Audio data as a PyTorch tensor, shape (channels, samples) or (samples,)

    Returns:
        Mono audio as a PyTorch tensor, shape (samples,)
    """
    if audio.dim() > 1:
        logger.info(f"Converting audio from {audio.shape[0]} channels to mono")
        # Convert to float if needed before applying mean operation
        if audio.dtype != torch.float32:
            if audio.dtype in [torch.int16, torch.short]:
                # Normalize int16 values to [-1, 1] float range
                audio = audio.float() / 32768.0
            elif audio.dtype == torch.int32:
                audio = audio.float() / 2147483648.0
            else:
                audio = audio.float()
        mono = torch.mean(audio, dim=0)
        logger.info(f"Mono waveform length (samples): {mono.shape[-1]}")
        return mono
    logger.info(f"Audio is already mono. Length (samples): {audio.shape[-1]}")
    return audio


def float_or_auto(value):
    if value == "auto":
        return value
    return float(value)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Benchmark differentiable vs classical compressor"
    )
    p.add_argument(
        "--test-signal-type",
        type=str,
        default="step",
        choices=["tone", "step", "burst", "ramp", "file"],
        help="Type of test signal to use",
    )
    p.add_argument("--test-file-path", type=str, default=None, help="Path to test file")
    p.add_argument(
        "--comp-thresh",
        type=float_or_auto,
        default=-24,
        help="Compressor threshold in dB (float or 'auto')",
    )
    p.add_argument("--comp-ratio", type=float, default=4.0, help="Compressor ratio")
    p.add_argument(
        "--exp-thresh", type=float, default=-1000.0, help="Expander threshold in dB"
    )
    p.add_argument("--exp-ratio", type=float, default=0.5, help="Expander ratio")
    p.add_argument(
        "--attack-time-ms", type=float, default=10.0, help="Attack time in ms"
    )
    p.add_argument(
        "--release-time-ms", type=float, default=100.0, help="Release time in ms"
    )
    p.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="Multiplier for diff in sigmoid gating (higher = harder)",
    )

    args = p.parse_args()

    # Validate arguments
    if args.test_signal_type == "file":
        if not args.test_file_path:
            p.error("--test_file_path is required when test-signal is 'file'")
        args.test_file_path = os.path.abspath(args.test_file_path)
        if not os.path.isfile(args.test_file_path):
            raise FileNotFoundError(f"The file '{args.test_file_path}' does not exist")

    run_bench(
        test_signal_type=args.test_signal_type,
        test_file_path=args.test_file_path,
        comp_thresh=args.comp_thresh,
        comp_ratio=args.comp_ratio,
        exp_thresh=args.exp_thresh,
        exp_ratio=args.exp_ratio,
        attack_time_ms=args.attack_time_ms,
        release_time_ms=args.release_time_ms,
        k=args.k,
    )

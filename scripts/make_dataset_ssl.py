#!/usr/bin/env python3
"""
Dataset builder: offline generation of (x, x_peak_dB, g_ref, y_ref, meta) examples.

- Outputs are written one example per directory under <output-dir>/<split>/perm_XXX/clip_XXXX/.

Per-example artifacts:
- x.wav        : clean mono audio (float32)
- x_peak_dB.pt : detector envelope in dB = gain_db(abs(x)) (float32) of shape (1, T)
- g_ref_dB.pt  : teacher gain in dB (float32) of shape (1, T)
- y_ref.wav    : db_gain(g_ref_dB) * x (float32)
- meta.yaml    : {fs, theta_ref, seed, processing_version, num_samples, source, perm_idx, detector}

Synthetic signals come from differential_dynamics.benchmarks.signals (tone/step/burst/ramp).
Optional musical clips are cropped from --music-dir.
"""

import argparse
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import math
import torch
import torchaudio
import yaml

from differential_dynamics.backends.torch.gain import SSL_comp_gain
from differential_dynamics.benchmarks.bench_utilities import gain_db, db_gain
from differential_dynamics.benchmarks.signals import (
    tone,
    step as step_sig,
    burst,
    ramp,
    composite_program,
    composite_program_jitter,
    probe_p0_quasi_static,
    probe_p1_smoothed_pairs,
    probe_p1b_preconditioned_pairs,
    probe_p2_echo_and_silence,
    probe_p2b_release_drop_near_threshold,
    probe_p3_db_dither,
    probe_p5_transient_churn,
    probe_p6_slope_sweep,
    probe_p7_release_staircase,
    probe_p8_attack_doublet,
    probe_p9_am_sweep,
    probe_p10_prbs_amp,
)

import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def frac01(v: str) -> float:
    """Argparse type validator for fractions in [0, 1]."""
    x = float(v)
    if not (0.0 <= x <= 1.0):
        raise argparse.ArgumentTypeError("--music-frac must be between 0 and 1")
    return x


@dataclass
class Theta:
    comp_thresh: float
    comp_ratio: float
    attack_time_fast_ms: float
    attack_time_slow_ms: float
    release_time_fast_ms: float
    release_time_slow_ms: float
    feedback_coeff: float
    k: float = 1  # gate sharpness (unused in hard mode)
    soft_gate: bool = False  # hard A/R

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def ensure_mono(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.size(0) == 1:
        return x
    return torch.mean(x, dim=0, keepdim=True)


def resample_if_needed(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return waveform
    tfm = torchaudio.transforms.Resample(sr, target_sr)
    return tfm(waveform)


def sample_theta(
    rng: random.Random,
    attack_time_fast_ms_min: float,
    attack_time_slow_ms_min: float,
    release_time_fast_ms_min: float,
    release_time_slow_ms_min: float,
    attack_time_fast_ms_max: float,
    attack_time_slow_ms_max: float,
    release_time_fast_ms_max: float,
    release_time_slow_ms_max: float,
    feedback_coeff_min: float,
    feedback_coeff_max: float,
) -> Theta:
    """Sample a single teacher parameter set θ.

    The sampled θ is intended to be applied to ALL clips within a permutation,
    so call this exactly once per perm (not per-clip) to avoid leakage of
    content dependence into θ.

    """
    # Threshold in [-35, -5] dB (absolute, no param conditioning in probes)
    thresh = rng.uniform(-35.0, -5.0)
    # Ratio from {2, 4, 10}
    ratio = rng.choice([2.0, 4.0, 10.0])

    attack_time_fast_ms: float
    attack_time_slow_ms: float
    release_time_fast_ms: float
    release_time_slow_ms: float
    feedback_coeff: float

    # Attack ms log-uniform within bounds
    attack_time_fast_ms = math.exp(
        rng.uniform(
            math.log(attack_time_fast_ms_min), math.log(attack_time_fast_ms_max)
        )
    )
    attack_time_slow_ms = math.exp(
        rng.uniform(
            math.log(attack_time_slow_ms_min), math.log(attack_time_slow_ms_max)
        )
    )

    # Release ms log-uniform within bounds
    release_time_fast_ms = math.exp(
        rng.uniform(
            math.log(release_time_fast_ms_min), math.log(release_time_fast_ms_max)
        )
    )
    release_time_slow_ms = math.exp(
        rng.uniform(
            math.log(release_time_slow_ms_min), math.log(release_time_slow_ms_max)
        )
    )

    # Feedback coeff uniform within bounds
    feedback_coeff = rng.uniform(feedback_coeff_min, feedback_coeff_max)

    return Theta(
        comp_thresh=thresh,
        comp_ratio=ratio,
        attack_time_fast_ms=attack_time_fast_ms,
        attack_time_slow_ms=attack_time_slow_ms,
        release_time_fast_ms=release_time_fast_ms,
        release_time_slow_ms=release_time_slow_ms,
        feedback_coeff=feedback_coeff,
    )


def process_example(
    x: torch.Tensor,
    fs: int,
    theta: Theta,
    normalize: str = "peak",  # "none" | "peak"
    target_peak: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Compute x_rms, teacher gain (hard A/R), and the compressed target.

    Args:
      x: (1, T) or (T,) clean audio (float32 preferred). Will be coerced to shape (1, T).
      fs: sample rate (Hz).
      theta: teacher parameters.

    Returns:
      x: (1, T) float32 clean audio
      x_peak_dB: (1, T) float32 detector envelope
      g_ref_dB: (1, T) float32 teacher gain in dB
      y_ref: (1, T) float32 compressed target (g_ref * x)
    """
    x = x.to(torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(0)

    # Optional per-clip normalization (apply before computing envelopes/teacher)
    norm_info: dict = {"method": normalize}
    with torch.no_grad():
        if normalize == "peak":
            peak_before = float(torch.max(torch.abs(x)).item())
            if peak_before > 1e-9 and math.isfinite(peak_before):
                scale = float(target_peak / peak_before)
            else:
                scale = 1.0
            x = x * scale
            norm_info.update(
                {
                    "target_peak": float(target_peak),
                    "peak_before": float(peak_before),
                    "scale": float(scale),
                    "peak_after": float(torch.max(torch.abs(x)).item()),
                }
            )
        else:
            norm_info.update(
                {
                    "target_peak": None,
                    "peak_before": None,
                    "scale": 1.0,
                    "peak_after": None,
                }
            )

    test_signal_peak_dB = gain_db(x.abs())

    g_ref_dB = SSL_comp_gain(
        x_peak_dB=test_signal_peak_dB,
        comp_thresh=theta.comp_thresh,
        comp_ratio=theta.comp_ratio,
        T_attack_fast=theta.attack_time_fast_ms / 1000,
        T_attack_slow=theta.attack_time_slow_ms / 1000,
        T_shunt_fast=theta.release_time_fast_ms / 1000,
        T_shunt_slow=theta.release_time_slow_ms / 1000,
        feedback_coeff=theta.feedback_coeff,
        k=theta.k,
        fs=fs,
        soft_gate=theta.soft_gate,
    )

    # Fail loudly on non-finite artifacts
    if not torch.isfinite(test_signal_peak_dB).all():
        raise RuntimeError(
            "Non-finite x_peak_dB detected; check inputs and normalization"
        )
    if not torch.isfinite(g_ref_dB).all():
        raise RuntimeError(
            "Non-finite teacher gain (g_ref_dB) detected; check time-constant floors and inputs"
        )

    y_ref = db_gain(g_ref_dB) * x
    if not torch.isfinite(y_ref).all():
        raise RuntimeError(
            "Non-finite y_ref detected after applying teacher gain; check g_ref_dB and x"
        )
    return x, test_signal_peak_dB, g_ref_dB, y_ref, norm_info


def save_example(
    out_dir: Path,
    idx: int,
    split: str,
    x: torch.Tensor,
    x_peak_dB: torch.Tensor,
    g_ref_dB: torch.Tensor,
    y_ref: torch.Tensor,
    theta: Theta,
    fs: int,
    seed: int,
    src: str,
    perm_idx: int,
    processing_version: str = "v0",
    norm_info: dict | None = None,
    extra_meta: dict | None = None,
) -> None:
    """Persist artifacts for a single example under clip_XXXX/.

    Files written:
      - x.wav, y_ref.wav
      - x_peak_dB.pt, g_ref_dB.pt
      - meta.yaml (includes fs, theta_ref, seed, processing_version, num_samples, source, perm_idx, detector)
    """
    ex_dir = out_dir / split / f"clip_{idx:04d}"
    ex_dir.mkdir(parents=True, exist_ok=True)

    # Waveforms
    torchaudio.save(str(ex_dir / "x.wav"), x.cpu(), fs)
    torchaudio.save(str(ex_dir / "y_ref.wav"), y_ref.cpu(), fs)

    # Tensors
    torch.save(x_peak_dB.cpu(), ex_dir / "x_peak_dB.pt")
    torch.save(g_ref_dB.cpu(), ex_dir / "g_ref_dB.pt")

    meta = {
        "fs": fs,
        "theta_ref": theta.to_dict(),
        "seed": seed,
        "processing_version": processing_version,
        "num_samples": int(x.shape[-1]),
        "source": src,
        "perm_idx": int(perm_idx),
        "detector": "abs",
    }
    if norm_info is not None:
        meta["normalization"] = norm_info
    if extra_meta is not None:
        # Merge extra metadata at top-level for provenance
        try:
            meta.update(extra_meta)
        except Exception:
            meta["extra_meta"] = extra_meta
    with open(ex_dir / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=True)


def list_music_files(music_dir: Path) -> list[Path]:
    """List audio files under music_dir."""
    paths: list[Path] = []
    for ext in (".wav", ".flac", ".mp3", ".m4a"):
        paths.extend([p for p in music_dir.rglob(f"*{ext}")])
    return paths


def allocate_music_files(
    files: list[Path], split_pcts: tuple[float, float, float], seed: int
) -> dict[str, list[Path]]:
    """Shuffle files and allocate disjoint sets to train/val/test by percentages."""
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)
    total = len(files)
    p_train, p_val, p_test = split_pcts
    denom = max(p_train + p_val + p_test, 1e-9)
    n_train = int(round(total * (p_train / denom)))
    n_val = int(round(total * (p_val / denom)))
    n_test = max(0, total - n_train - n_val)
    return {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val : n_train + n_val + n_test],
    }


def random_music_crops(
    files: list[Path], target_sr: int, clip_dur_s: float, n_clips: int, seed: int
) -> list[Tuple[torch.Tensor, int, str]]:
    """From a pool of files, produce up to n_clips random crops without mixing files across splits."""
    rng = random.Random(seed)
    T = int(round(clip_dur_s * target_sr))
    out: list[Tuple[torch.Tensor, int, str]] = []
    file_indices = list(range(len(files)))
    rng.shuffle(file_indices)
    # Iterate over files, taking random crops until we hit n_clips
    while len(out) < n_clips and file_indices:
        idx = file_indices[0]
        p = files[idx]
        try:
            y, sr = torchaudio.load(str(p))
        except Exception:
            file_indices.pop(0)
            continue
        y = ensure_mono(y)
        y = resample_if_needed(y, sr, target_sr)
        if y.shape[-1] < T:
            # Too short; drop this file
            file_indices.pop(0)
            continue
        # Take as many non-overlapping crops as needed from this file
        max_start = y.shape[-1] - T
        starts = list(range(0, max_start + 1, T))
        rng.shuffle(starts)
        for s in starts:
            if len(out) >= n_clips:
                break
            clip = y[:, s : s + T]
            out.append((clip, target_sr, str(p)))
        # Move to next file
        file_indices.pop(0)
    return out


def parse_split_pcts(raw: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 3 or sum(p <= 0 for p in parts) > 0:
        raise ValueError("--split-pcts must be three positive numbers, e.g., 80,10,10")
    return (parts[0], parts[1], parts[2])


def parse_probe_mix(raw: str) -> list[tuple[str, float]]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    mix: list[tuple[str, float]] = []
    total = 0.0
    allowed = {
        "P0",
        "P1",
        "P1B",
        "P2",
        "P2B",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
        "P10",
    }
    for part in parts:
        k, v = part.split(":")
        k = k.strip().upper()
        if k not in allowed:
            raise ValueError(
                f"Unknown probe key '{k}' in --probe-mix (allowed {sorted(list(allowed))})"
            )
        w = float(v)
        if w < 0:
            raise ValueError("Probe weight must be non-negative")
        mix.append((k, w))
        total += w
    if total <= 0:
        raise ValueError("Sum of probe weights must be > 0")
    # Normalize to sum 1
    mix = [(k, w / total) for (k, w) in mix]
    return mix


def choice_with_weights(rng: random.Random, mix: list[tuple[str, float]]) -> str:
    r = rng.random()
    acc = 0.0
    for k, w in mix:
        acc += w
        if r <= acc:
            return k
    return mix[-1][0]


def main():
    """Entry point: generate train/val/test in one call, with permutations per split."""
    p = argparse.ArgumentParser(
        description="Build train/val/test datasets with hard-gate teacher targets (compression-only)"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory to write {split}/perm_XXX/clip_XXXX",
    )
    p.add_argument("--fs", type=int, default=44100, help="Sample rate (Hz)")

    p.add_argument(
        "--clip-dur-s", type=float, default=12.0, help="Clip duration in seconds"
    )

    # Optional per-clip normalization
    p.add_argument(
        "--normalize",
        type=str,
        default="peak",
        choices=["none", "peak"],
        help="Per-clip normalization to apply before computing envelopes/teacher",
    )

    p.add_argument(
        "--target-peak",
        type=float,
        default=0.9,
        help="Target absolute peak amplitude when --normalize=peak",
    )

    # Dataset processing version tag (written to meta.yaml)
    p.add_argument(
        "--processing-version",
        type=str,
        default="v0",
        help="Version tag written into meta.yaml for reproducibility",
    )

    # Seeds: separate θ from content to avoid leakage while keeping θ fixed per perm across splits
    p.add_argument(
        "--theta-seed",
        type=int,
        default=1337,
        help="Seed to sample theta per permutation deterministically",
    )

    p.add_argument(
        "--seed-train",
        type=int,
        default=1001,
        help="RNG seed for train signal generation/crops",
    )

    p.add_argument(
        "--seed-val",
        type=int,
        default=1002,
        help="RNG seed for val signal generation/crops",
    )

    p.add_argument(
        "--seed-test",
        type=int,
        default=1003,
        help="RNG seed for test signal generation/crops",
    )

    p.add_argument(
        "--music-split-seed",
        type=int,
        default=2024,
        help="Seed to allocate music files into disjoint splits",
    )

    # Music
    p.add_argument(
        "--music-dir",
        type=str,
        default=None,
        help="Directory containing musical audio files (files will be split by file into train/val/test)",
    )

    # Optional: fix feedback coeff for the teacher (processor-faithful if hardware has fixed fb)
    p.add_argument(
        "--fixed-fb",
        type=float,
        default=1.0,
        help="If set, override sampled feedback_coeff with this constant (0..1) across all clips/perms",
    )

    p.add_argument(
        "--music-frac",
        type=frac01,
        default=0.0,
        help="Fraction of clips per split to source from music (rest from synthetic), must be in [0,1]",
    )

    # Time constants - min/max
    p.add_argument(
        "--attack-time-fast-ms-min",
        type=float,
        default=820 * 0.47e-6 * 1000 * 0.8,
        help="Minimum fast attack time in ms",
    )

    p.add_argument(
        "--attack-time-fast-ms-max",
        type=float,
        default=270000 * 0.47e-6 * 1000 * 1.2,
        help="Maximum fast attack time in ms",
    )

    p.add_argument(
        "--attack-time-slow-ms-min",
        type=float,
        default=820 * 6.8e-6 * 1000 * 0.8,
        help="Minimum slow attack time in ms",
    )

    p.add_argument(
        "--attack-time-slow-ms-max",
        type=float,
        default=270e3 * 6.8e-6 * 1000 * 1.2,
        help="Maximum slow attack time in ms",
    )

    p.add_argument(
        "--release-time-fast-ms-min",
        type=float,
        default=91e3 * 0.47e-6 * 1000 * 0.8,
        help="Minimum fast release time in ms",
    )

    p.add_argument(
        "--release-time-fast-ms-max",
        type=float,
        default=91e3 * 0.47e-6 * 1000 * 1.2,
        help="Maximum fast release time in ms",
    )

    p.add_argument(
        "--release-time-slow-ms-min",
        type=float,
        default=750e3 * 6.8e-6 * 1000 * 0.8,
        help="Minimum Slow release time in ms",
    )

    p.add_argument(
        "--release-time-slow-ms-max",
        type=float,
        default=750e3 * 6.8e-6 * 1000 * 1.2,
        help="Maximum Slow release time in ms",
    )

    p.add_argument(
        "--feedback-coeff-min",
        type=float,
        default=0.0,
        help="Minimum feedback coefficient",
    )

    p.add_argument(
        "--feedback-coeff-max",
        type=float,
        default=1.0,
        help="Maximum feedback coefficient",
    )

    # Totals and split percentages
    p.add_argument(
        "--num-total",
        type=int,
        default=120,
        help="Total clips per split (train/val/test)",
    )

    # Probe mixture for synthetic clips (single-stage, processor-faithful)
    p.add_argument(
        "--probe-mix",
        type=str,
        default="P0:0.05,P1:0.18,P1B:0.18,P2:0.15,P2B:0.15,P3:0.03,P5:0.07,P6:0.08,P7:0.06,P8:0.03,P9:0.01,P10:0.01",
        help="Weighted mix of probe types (P0..P10); CT-agnostic probes to maximize identifiability",
    )
    p.add_argument(
        "--nonsteady-only",
        action="store_true",
        help="If set, ignore steady-heavy probes (P0,P3,P4) and use a non-steady mix (P1,P1B,P2,P2B,P5)",
    )

    # Ramps-only dataset (overrides probe mix and music): envelope ramp from -40 dB to 0 dB
    p.add_argument(
        "--ramps-only",
        action="store_true",
        help="Generate only envelope ramps (−40 dB to 0 dB) for all clips; overrides music/probe mix",
    )
    p.add_argument(
        "--ramps-direction",
        type=str,
        choices=["up", "down", "alternate"],
        default="alternate",
        help="Direction for ramps when --ramps-only is set: up (−40→0 dB), down (0→−40 dB), or alternate per clip",
    )

    # Statics-only dataset: constant-level plateaus (full-clip), level uniform in [-45, 0] dB
    p.add_argument(
        "--statics-only",
        action="store_true",
        help="Generate only constant plateaus (full-clip) with levels uniform in [-45,0] dB; overrides music/probe mix",
    )

    p.add_argument(
        "--split-pcts",
        type=str,
        default="80,10,10",
        help="Percent split for train,val,test (e.g., 80,10,10)",
    )

    # Permutations: number of global-theta datasets to create
    p.add_argument(
        "--num-perms",
        type=int,
        default=1,
        help="How many parameter permutations to generate; each permutation shares the same clips within a split but uses a single random theta for all clips",
    )

    # Switch-matrix mode: enumerate fixed single-pole attack/release permutations (6x4=24)
    p.add_argument(
        "--use-switch-matrix",
        action="store_true",
        help="If set, generate 24 single-pole permutations from hardware-like switch positions; overrides --num-perms",
    )

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # System-ID dataset: build a single identification set under 'train'
    rng = random.Random(args.seed_train)
    n_total = int(args.num_total)

    # Special modes: ramps-only or statics-only
    if args.ramps_only or args.statics_only:
        T = int(round(args.clip_dur_s * args.fs))
        items: list[Tuple[torch.Tensor, int, str]] = []
        if args.ramps_only:
            db_lo = -40.0
            db_hi = 0.0
            for i in range(n_total):
                is_up = args.ramps_direction == "up" or (
                    args.ramps_direction == "alternate" and (i % 2 == 0)
                )
                if is_up:
                    start_db, end_db = db_lo, db_hi
                    tag = "RAMP_UP"
                else:
                    start_db, end_db = db_hi, db_lo
                    tag = "RAMP_DOWN"
                slope_db = torch.linspace(start_db, end_db, T, dtype=torch.float32)
                slope_amp = (10.0 ** (slope_db / 20.0)).to(torch.float32)
                x = slope_amp[None, :]
                items.append((x, args.fs, f"synth:{tag}"))
        else:
            # Statics-only: one plateau per clip, level uniform in [-45,0] dB
            rng_f = random.Random(args.seed_train)
            for i in range(n_total):
                L_db = rng_f.uniform(-45.0, 0.0)
                a = float(10 ** (L_db / 20.0))
                x = torch.full((1, T), a, dtype=torch.float32)
                items.append((x, args.fs, f"synth:PLATEAU:{L_db:.3f}dB"))
        items_by_split = {"train": items}
    else:
        n_music = int(round(n_total * args.music_frac))
        n_synth = max(0, n_total - n_music)
        items: list[Tuple[torch.Tensor, int, str]] = []

        # Optional music crops
        if args.music_dir is not None:
            all_files = list_music_files(Path(args.music_dir))
            if all_files:
                crops = random_music_crops(
                    all_files, args.fs, args.clip_dur_s, n_music, args.seed_train
                )
                items.extend(
                    [(clip, args.fs, f"music:{src}") for (clip, _, src) in crops]
                )
        # If we didn’t reach n_music, convert deficit to synth
        if len(items) < n_music:
            n_synth += n_music - len(items)

    if not (args.ramps_only or args.statics_only):
        # Synthetic content via mixed probes (unique per clip)
        T = int(round(args.clip_dur_s * args.fs))
        # Non-steady-state preset if requested
        mix = (
            parse_probe_mix("P1:0.30,P1B:0.30,P2:0.20,P2B:0.15,P5:0.05")
            if args.nonsteady_only
            else parse_probe_mix(args.probe_mix)
        )

        def gen_probe_clip(i: int) -> tuple[torch.Tensor, int, str]:
            per_rng = random.Random(args.seed_train + i)
            probe = choice_with_weights(per_rng, mix)
            if probe == "P0":
                x = probe_p0_quasi_static(fs=args.fs, T=T, B=1)
            elif probe == "P1":
                x = probe_p1_smoothed_pairs(
                    fs=args.fs,
                    T=T,
                    B=1,
                    rng=per_rng,
                    edge_ms_grid=[5, 10, 20, 50, 100, 200, 400],
                )
            elif probe == "P1B":
                x = probe_p1b_preconditioned_pairs(fs=args.fs, T=T, B=1, rng=per_rng)
            elif probe == "P2":
                x = probe_p2_echo_and_silence(fs=args.fs, T=T, B=1, rng=per_rng)
            elif probe == "P2B":
                x = probe_p2b_release_drop_near_threshold(
                    fs=args.fs, T=T, B=1, rng=per_rng
                )
            elif probe == "P3":
                x = probe_p3_db_dither(fs=args.fs, T=T, B=1, rng=per_rng)
            elif probe == "P4":
                x = composite_program_jitter(fs=args.fs, T=T, B=1, rng=per_rng)
            elif probe == "P5":
                x = probe_p5_transient_churn(fs=args.fs, T=T, B=1, rng=per_rng)
            elif probe == "P6":
                x = probe_p6_slope_sweep(fs=args.fs, T=T, B=1, rng=per_rng)
            elif probe == "P7":
                x = probe_p7_release_staircase(fs=args.fs, T=T, B=1, rng=per_rng)
            elif probe == "P8":
                x = probe_p8_attack_doublet(fs=args.fs, T=T, B=1, rng=per_rng)
            elif probe == "P9":
                x = probe_p9_am_sweep(fs=args.fs, T=T, B=1, rng=per_rng)
            elif probe == "P10":
                x = probe_p10_prbs_amp(fs=args.fs, T=T, B=1, rng=per_rng)
            else:
                raise RuntimeError(f"Unhandled probe type {probe}")
            # Time-scale jitter 0.5x/1x/2x (deterministic per clip)
            r = per_rng.random()
            ts = 0.5 if r < 0.25 else (2.0 if r > 0.75 else 1.0)
            if abs(ts - 1.0) > 1e-6:
                # linear resample by index mapping
                Bv, Tv = x.shape
                t_src = torch.linspace(0, Tv - 1, steps=Tv, dtype=torch.float32)
                t_warp = torch.clamp(t_src / ts, 0, Tv - 1)
                t0 = t_warp.floor().to(torch.long)
                t1 = torch.clamp(t0 + 1, max=Tv - 1)
                w = (t_warp - t0.to(torch.float32)).unsqueeze(0)
                x = (1 - w) * x[:, t0] + w * x[:, t1]
            # Terminal silence tail on ~30% of clips
            if per_rng.random() < 0.30:
                tail_s = per_rng.uniform(0.5, 2.0)
                tail = min(T, int(round(tail_s * args.fs)))
                if tail > 0:
                    x[:, max(0, T - tail) :] = 0.0
            return x.clamp(0.0, 1.0), args.fs, f"synth:{probe}"

        for i in range(n_synth):
            items.append(gen_probe_clip(i))

        # Adjust length exactly to n_total
        if len(items) > n_total:
            items = items[:n_total]
        elif len(items) < n_total:
            pad_needed = n_total - len(items)
            base = len(items)
            for j in range(pad_needed):
                items.append(gen_probe_clip(base + j))

        items_by_split = {"train": items}

    # Generate permutations
    if args.use_switch_matrix:
        # Build switch-matrix permutations (24): 6 attack fast positions x 4 release fast positions
        C_fast = 0.47e-6
        C_slow = 6.8e-6
        R_attacks = [820.0, 2700.0, 8200.0, 27000.0, 82000.0, 270000.0]
        R_shunts = [180000.0, 270000.0, 560000.0, 1200000.0]
        Tc_attack_slow_large_ms = 1e6
        # Floor slow shunt TC to a safe value relative to sample period to avoid numerical overflow
        ms_per_sample = 1000.0 / float(args.fs)
        Tc_shunt_slow_small_ms = max(0.01, ms_per_sample)

        # Sample a single base theta for comp/feedback to keep consistent across permutations
        base_rng = random.Random(args.theta_seed)
        base_theta = sample_theta(
            rng=base_rng,
            attack_time_fast_ms_min=args.attack_time_fast_ms_min,
            attack_time_slow_ms_min=args.attack_time_slow_ms_min,
            release_time_fast_ms_min=args.release_time_fast_ms_min,
            release_time_slow_ms_min=args.release_time_slow_ms_min,
            attack_time_fast_ms_max=args.attack_time_fast_ms_max,
            attack_time_slow_ms_max=args.attack_time_slow_ms_max,
            release_time_fast_ms_max=args.release_time_fast_ms_max,
            release_time_slow_ms_max=args.release_time_slow_ms_max,
            feedback_coeff_min=args.feedback_coeff_min,
            feedback_coeff_max=args.feedback_coeff_max,
        )
        if args.fixed_fb is not None:
            base_theta.feedback_coeff = float(min(max(args.fixed_fb, 0.0), 1.0))

        thetas = []
        switch_meta = []
        # 24 single-pole permutations
        for R_attack in R_attacks:
            Tc_attack_fast_ms = R_attack * C_fast * 1000.0
            for R_shunt in R_shunts:
                Tc_shunt_fast_ms = R_shunt * C_fast * 1000.0
                th = Theta(
                    comp_thresh=base_theta.comp_thresh,
                    comp_ratio=base_theta.comp_ratio,
                    attack_time_fast_ms=Tc_attack_fast_ms,
                    attack_time_slow_ms=Tc_attack_slow_large_ms,
                    release_time_fast_ms=Tc_shunt_fast_ms,
                    release_time_slow_ms=Tc_shunt_slow_small_ms,
                    feedback_coeff=base_theta.feedback_coeff,
                    k=0.0,
                    soft_gate=False,
                )
                thetas.append(th)
                switch_meta.append(
                    {
                        "switch_matrix": True,
                        "attack_mode": "single",
                        "release_mode": "single",
                        "switch_Ra_ohm": R_attack,
                        "switch_Rf_ohm": R_shunt,
                        "T_as_large_ms": Tc_attack_slow_large_ms,
                        "T_ss_large_ms": Tc_shunt_slow_small_ms,
                    }
                )
        # 6 dual-pole (auto release) permutations
        Tc_shunt_fast_auto_ms = 91000.0 * C_fast * 1000.0
        Tc_shunt_slow_auto_ms = 750000.0 * C_slow * 1000.0
        for R_attack in R_attacks:
            Tc_attack_fast_ms = R_attack * C_fast * 1000.0
            Tc_attack_slow_ms = R_attack * C_slow * 1000.0
            th = Theta(
                comp_thresh=base_theta.comp_thresh,
                comp_ratio=base_theta.comp_ratio,
                attack_time_fast_ms=Tc_attack_fast_ms,
                attack_time_slow_ms=Tc_attack_slow_ms,
                release_time_fast_ms=Tc_shunt_fast_auto_ms,
                release_time_slow_ms=Tc_shunt_slow_auto_ms,
                feedback_coeff=base_theta.feedback_coeff,
                k=0.0,
                soft_gate=False,
            )
            thetas.append(th)
            switch_meta.append(
                {
                    "switch_matrix": True,
                    "attack_mode": "dual",
                    "release_mode": "auto",
                    "switch_Ra_ohm": R_attack,
                    "auto_Rf_ohm": 91000.0,
                    "auto_Rs_ohm": 750000.0,
                }
            )

        for perm_idx, (theta, smeta) in enumerate(zip(thetas, switch_meta), start=1):
            for split_name, items in items_by_split.items():
                split_sub = f"{split_name}/perm_{perm_idx:03d}"
                for idx, (x, fs, src) in enumerate(items, start=1):
                    # For statics-only plateaus, synthesize a burn-in pre-roll at the same level,
                    # run the teacher on (pre + main), then crop to the last T samples to remove transients.
                    if args.statics_only and ("PLATEAU" in src):
                        T_target = int(round(args.clip_dur_s * args.fs))
                        # Parse plateau level in dB from src (format: synth:PLATEAU:<L_db>dB)
                        try:
                            lvl_str = src.split(":")[2]
                            L_db = float(lvl_str.rstrip("dB"))
                        except Exception:
                            L_db = None
                        if L_db is not None:
                            a = float(10 ** (L_db / 20.0))
                            # Burn-in: fixed, theta-independent pre-roll based on global fast TC bounds
                            pre_s = 20.0 * max(
                                float(args.attack_time_fast_ms_max) / 1000.0,
                                float(args.release_time_fast_ms_max) / 1000.0,
                            )
                            pre_N = int(round(pre_s * fs))
                            pre_N = min(10 * T_target, pre_N)
                            x_full = torch.full(
                                (1, T_target + pre_N), a, dtype=torch.float32
                            )
                            x2, x_peak_dB2, g_ref_dB2, y_ref2, norm_info = (
                                process_example(
                                    x=x_full,
                                    fs=fs,
                                    theta=theta,
                                    normalize="none",
                                    target_peak=args.target_peak,
                                )
                            )
                            # Crop to last T_target samples
                            x = x2[:, -T_target:]
                            x_peak_dB = x_peak_dB2[:, -T_target:]
                            g_ref_dB = g_ref_dB2[:, -T_target:]
                            y_ref = y_ref2[:, -T_target:]
                            norm_info = {
                                "method": "none",
                                "burn_in_samples": int(pre_N),
                            }
                        else:
                            x, x_peak_dB, g_ref_dB, y_ref, norm_info = process_example(
                                x=x,
                                fs=fs,
                                theta=theta,
                                normalize=(
                                    "none"
                                    if (args.ramps_only or args.statics_only)
                                    else args.normalize
                                ),
                                target_peak=args.target_peak,
                            )
                    else:
                        # Per-clip normalization override for synth (~30%) to diversify absolute levels
                        _rng_norm = random.Random((perm_idx * 10_000) + idx)
                        _disable_norm = (
                            src.startswith("synth:")
                            and args.normalize == "peak"
                            and (_rng_norm.random() < 0.30)
                        )
                        normalize_this = (
                            "none"
                            if (args.ramps_only or args.statics_only or _disable_norm)
                            else args.normalize
                        )
                        x, x_peak_dB, g_ref_dB, y_ref, norm_info = process_example(
                            x=x,
                            fs=fs,
                            theta=theta,
                            normalize=normalize_this,
                            target_peak=args.target_peak,
                        )
                    if args.ramps_only or args.statics_only:
                        src_tail = src.split(":")[-1]
                        if src_tail == "RAMP_UP":
                            extra_meta = {
                                "probe_type": src_tail,
                                "ramp_start_db": -40.0,
                                "ramp_end_db": 0.0,
                            }
                        elif src_tail == "RAMP_DOWN":
                            extra_meta = {
                                "probe_type": src_tail,
                                "ramp_start_db": 0.0,
                                "ramp_end_db": -40.0,
                            }
                        elif src_tail.startswith("PLATEAU"):
                            # Format: PLATEAU:<L_db>dB
                            try:
                                lvl = float(src_tail.split(":")[1].rstrip("dB"))
                            except Exception:
                                lvl = float("nan")
                            extra_meta = {"probe_type": "PLATEAU", "plateau_db": lvl}
                        else:
                            extra_meta = {"probe_type": src_tail}
                    else:
                        extra_meta = {}
                    if args.fixed_fb is not None:
                        extra_meta["fixed_feedback_coeff"] = float(
                            min(max(args.fixed_fb, 0.0), 1.0)
                        )
                    save_example(
                        out_dir=out_dir,
                        idx=idx,
                        split=split_sub,
                        x=x,
                        x_peak_dB=x_peak_dB,
                        g_ref_dB=g_ref_dB,
                        y_ref=y_ref,
                        theta=theta,
                        fs=fs,
                        seed={
                            "train": args.seed_train,
                            "val": args.seed_val,
                            "test": args.seed_test,
                        }[split_name],
                        src=src,
                        perm_idx=perm_idx,
                        processing_version=args.processing_version,
                        norm_info=norm_info,
                        extra_meta=extra_meta if extra_meta else None,
                    )
                # Friendly summary per permutation (handle single vs auto release metadata)
                if smeta.get("release_mode") == "single":
                    rf_part = f"Rf={smeta.get('switch_Rf_ohm', float('nan')):.0f}Ω"
                else:
                    rf_part = (
                        f"Rf(auto)={smeta.get('auto_Rf_ohm', float('nan')):.0f}Ω "
                        f"Rs(auto)={smeta.get('auto_Rs_ohm', float('nan')):.0f}Ω"
                    )
                print(
                    f"Wrote {split_sub} with SWITCH-MATRIX Ra={smeta.get('switch_Ra_ohm', float('nan')):.0f}Ω {rf_part} "
                    f"T_af={theta.attack_time_fast_ms:.3f}ms T_sf={theta.release_time_fast_ms:.3f}ms (clips: {len(items)})"
                )
    else:
        for perm_idx in range(1, args.num_perms + 1):
            # θ is sampled ONCE per permutation and reused across all splits
            theta_rng = random.Random(args.theta_seed + perm_idx)
            theta = sample_theta(
                rng=theta_rng,
                attack_time_fast_ms_min=args.attack_time_fast_ms_min,
                attack_time_slow_ms_min=args.attack_time_slow_ms_min,
                release_time_fast_ms_min=args.release_time_fast_ms_min,
                release_time_slow_ms_min=args.release_time_slow_ms_min,
                attack_time_fast_ms_max=args.attack_time_fast_ms_max,
                attack_time_slow_ms_max=args.attack_time_slow_ms_max,
                release_time_fast_ms_max=args.release_time_fast_ms_max,
                release_time_slow_ms_max=args.release_time_slow_ms_max,
                feedback_coeff_min=args.feedback_coeff_min,
                feedback_coeff_max=args.feedback_coeff_max,
            )
            # Override feedback if fixed-fb is provided
            if args.fixed_fb is not None:
                theta.feedback_coeff = float(min(max(args.fixed_fb, 0.0), 1.0))
            for split_name, items in items_by_split.items():
                split_sub = f"{split_name}/perm_{perm_idx:03d}"
                for idx, (x, fs, src) in enumerate(items, start=1):
                    # Teacher = SSL hard-gate teacher; detector = abs
                    if args.statics_only and ("PLATEAU" in src):
                        T_target = int(round(args.clip_dur_s * args.fs))
                        try:
                            lvl_str = src.split(":")[2]
                            L_db = float(lvl_str.rstrip("dB"))
                        except Exception:
                            L_db = None
                        if L_db is not None:
                            a = float(10 ** (L_db / 20.0))
                            pre_s = 20.0 * max(
                                float(args.attack_time_fast_ms_max) / 1000.0,
                                float(args.release_time_fast_ms_max) / 1000.0,
                            )
                            pre_N = int(round(pre_s * fs))
                            pre_N = min(10 * T_target, pre_N)
                            x_full = torch.full(
                                (1, T_target + pre_N), a, dtype=torch.float32
                            )
                            x2, x_peak_dB2, g_ref_dB2, y_ref2, norm_info = (
                                process_example(
                                    x=x_full,
                                    fs=fs,
                                    theta=theta,
                                    normalize="none",
                                    target_peak=args.target_peak,
                                )
                            )
                            x = x2[:, -T_target:]
                            x_peak_dB = x_peak_dB2[:, -T_target:]
                            g_ref_dB = g_ref_dB2[:, -T_target:]
                            y_ref = y_ref2[:, -T_target:]
                            norm_info = {
                                "method": "none",
                                "burn_in_samples": int(pre_N),
                            }
                        else:
                            x, x_peak_dB, g_ref_dB, y_ref, norm_info = process_example(
                                x=x,
                                fs=fs,
                                theta=theta,
                                normalize=(
                                    "none"
                                    if (args.ramps_only or args.statics_only)
                                    else args.normalize
                                ),
                                target_peak=args.target_peak,
                            )
                    else:
                        # Per-clip normalization override for synth (~30%) to diversify absolute levels
                        _rng_norm = random.Random((perm_idx * 10_000) + idx)
                        _disable_norm = (
                            src.startswith("synth:")
                            and args.normalize == "peak"
                            and (_rng_norm.random() < 0.30)
                        )
                        normalize_this = (
                            "none"
                            if (args.ramps_only or args.statics_only or _disable_norm)
                            else args.normalize
                        )
                        x, x_peak_dB, g_ref_dB, y_ref, norm_info = process_example(
                            x=x,
                            fs=fs,
                            theta=theta,
                            normalize=normalize_this,
                            target_peak=args.target_peak,
                        )
                    # Persist artifacts; keep the split-specific seed in metadata for provenance
                    # Compose extra metadata
                    if args.ramps_only or args.statics_only:
                        src_tail = src.split(":")[-1]
                        if src_tail == "RAMP_UP":
                            extra_meta = {
                                "probe_type": src_tail,
                                "ramp_start_db": -40.0,
                                "ramp_end_db": 0.0,
                            }
                        elif src_tail == "RAMP_DOWN":
                            extra_meta = {
                                "probe_type": src_tail,
                                "ramp_start_db": 0.0,
                                "ramp_end_db": -40.0,
                            }
                        elif src_tail.startswith("PLATEAU"):
                            try:
                                lvl = float(src_tail.split(":")[1].rstrip("dB"))
                            except Exception:
                                lvl = float("nan")
                            extra_meta = {"probe_type": "PLATEAU", "plateau_db": lvl}
                        else:
                            extra_meta = {"probe_type": src_tail}
                    else:
                        extra_meta = {"probe_type": src.split(":")[-1]}
                    if args.fixed_fb is not None:
                        extra_meta["fixed_feedback_coeff"] = float(
                            min(max(args.fixed_fb, 0.0), 1.0)
                        )
                    save_example(
                        out_dir=out_dir,
                        idx=idx,
                        split=split_sub,
                        x=x,
                        x_peak_dB=x_peak_dB,
                        g_ref_dB=g_ref_dB,
                        y_ref=y_ref,
                        theta=theta,
                        fs=fs,
                        seed={
                            "train": args.seed_train,
                            "val": args.seed_val,
                            "test": args.seed_test,
                        }[split_name],
                        src=src,
                        perm_idx=perm_idx,
                        processing_version=args.processing_version,
                        norm_info=norm_info,
                        extra_meta=extra_meta,
                    )
                print(
                    f"Wrote {split_sub} with thresh={theta.comp_thresh:.1f}dB ratio={theta.comp_ratio:.1f} attack_fast={theta.attack_time_fast_ms:.1f}ms attack_slow={theta.attack_time_slow_ms:.1f}ms release_fast={theta.release_time_fast_ms:.1f}ms release_slow={theta.release_time_slow_ms:.1f}ms (clips: {len(items)})"
                )


if __name__ == "__main__":
    main()

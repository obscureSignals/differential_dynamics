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
)

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
    # Threshold in [-36, -18] dB
    thresh = rng.uniform(-36.0, -18.0)
    # Ratio from {2, 4, 8}
    ratio = rng.choice([2.0, 4.0, 8.0])

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


def synth_clip(kind: str, fs: int, T: int) -> torch.Tensor:
    """Legacy primitive generators (kept for variety). Returns (1, T).

    Note: most synthetic content should come from composite_program() to avoid
    template leakage across splits. These primitives can still be sprinkled in
    as additional variety during early experiments.
    """
    if kind == "tone":
        return tone(freq=1000.0, fs=fs, T=T, B=1, amp=0.5)
    if kind == "step":
        return step_sig(fs=fs, T=T, B=1, at=0.25, amp_before=0.05, amp_after=0.8)
    if kind == "burst":
        return burst(fs=fs, T=T, B=1, start=0.2, dur=0.1, amp=0.8, freq=1000.0)
    if kind == "ramp":
        return ramp(fs=fs, T=T, B=1, start=0.2, dur=0.5, a0=0.1, a1=0.8)
    raise ValueError(f"Unknown synth kind: {kind}")


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

    y_ref = db_gain(g_ref_dB) * x
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
        "--clip-dur-s", type=float, default=2, help="Clip duration in seconds"
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

    p.add_argument(
        "--music-frac",
        type=frac01,
        default=0.5,
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

    split_pcts = parse_split_pcts(args.split_pcts)
    # Per split counts
    denom = sum(split_pcts)
    counts = {
        name: int(round(args.num_total * (pct / denom)))
        for name, pct in zip(["train", "val", "test"], split_pcts)
    }

    # Allocate music files to splits by file (no cross-split leakage by file)
    files_by_split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    if args.music_dir is not None:
        all_files = list_music_files(Path(args.music_dir))
        files_by_split = allocate_music_files(
            all_files, split_pcts, args.music_split_seed
        )

    # Build content per split (disjoint via per-split seeds)
    def build_split_items(split: str, seed: int) -> list[Tuple[torch.Tensor, int, str]]:
        """Produce a list of (x, fs, src) triples for a single split.

        - Uses split-specific seeds so signals differ across train/val/test.
        - Music: allocated by file to avoid cross-split leakage. We draw non-overlapping
          crops per file until we reach the requested count.
        - Synthetic: generated via composite_program() to mimic program-like content.
        - Ensures the final count matches n_total by trimming/padding synthetics.
        """
        rng = random.Random(seed)
        n_total = counts[split]
        n_music = int(round(n_total * args.music_frac))
        n_synth = max(0, n_total - n_music)
        items: list[Tuple[torch.Tensor, int, str]] = []
        # Music crops
        if args.music_dir is not None and files_by_split[split]:
            crops = random_music_crops(
                files_by_split[split], args.fs, args.clip_dur_s, n_music, seed
            )
            items.extend([(clip, args.fs, f"music:{src}") for (clip, _, src) in crops])
        # If not enough music, fill deficit with synthetic
        if len(items) < n_music:
            deficit = n_music - len(items)
            n_synth += deficit
        # Synthetic via selected mode
        T = int(round(args.clip_dur_s * args.fs))
        for _ in range(n_synth):
            x = composite_program(fs=args.fs, T=T, B=1, rng=rng)
            items.append((x, args.fs, "synth:composite"))
        if len(items) != n_total:
            # Adjust by trimming or padding synthetics
            if len(items) > n_total:
                items = items[:n_total]
            else:
                pad_needed = n_total - len(items)
                for _ in range(pad_needed):
                    x = composite_program(fs=args.fs, T=T, B=1, rng=rng)
                    items.append((x, args.fs, "synth:composite"))
        return items

    items_by_split = {
        "train": build_split_items("train", args.seed_train),
        "val": build_split_items("val", args.seed_val),
        "test": build_split_items("test", args.seed_test),
    }

    # Generate permutations
    if args.use_switch_matrix:
        # Build switch-matrix permutations (24): 6 attack fast positions x 4 release fast positions
        Cf = 0.47e-6
        Cs = 6.8e-6
        Ras = [820.0, 2700.0, 8200.0, 27000.0, 82000.0, 270000.0]
        Rfs = [180000.0, 270000.0, 560000.0, 1200000.0]
        T_as_large_ms = 1e9 * Cs * 1000.0
        T_ss_large_ms = 1e9 * Cs * 1000.0

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

        thetas = []
        switch_meta = []
        # 24 single-pole permutations
        for Ra in Ras:
            T_af_ms = Ra * Cf * 1000.0
            for Rf in Rfs:
                T_sf_ms = Rf * Cf * 1000.0
                th = Theta(
                    comp_thresh=base_theta.comp_thresh,
                    comp_ratio=base_theta.comp_ratio,
                    attack_time_fast_ms=T_af_ms,
                    attack_time_slow_ms=T_as_large_ms,
                    release_time_fast_ms=T_sf_ms,
                    release_time_slow_ms=T_ss_large_ms,
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
                        "switch_Ra_ohm": Ra,
                        "switch_Rf_ohm": Rf,
                        "T_as_large_ms": T_as_large_ms,
                        "T_ss_large_ms": T_ss_large_ms,
                    }
                )
        # 6 dual-pole (auto release) permutations
        T_sf_auto_ms = 91000.0 * Cf * 1000.0
        T_ss_auto_ms = 750000.0 * Cs * 1000.0
        for Ra in Ras:
            T_af_ms = Ra * Cf * 1000.0
            T_as_ms = Ra * Cs * 1000.0
            th = Theta(
                comp_thresh=base_theta.comp_thresh,
                comp_ratio=base_theta.comp_ratio,
                attack_time_fast_ms=T_af_ms,
                attack_time_slow_ms=T_as_ms,
                release_time_fast_ms=T_sf_auto_ms,
                release_time_slow_ms=T_ss_auto_ms,
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
                    "switch_Ra_ohm": Ra,
                    "auto_Rf_ohm": 91000.0,
                    "auto_Rs_ohm": 750000.0,
                }
            )

        for perm_idx, (theta, smeta) in enumerate(zip(thetas, switch_meta), start=1):
            for split_name, items in items_by_split.items():
                split_sub = f"{split_name}/perm_{perm_idx:03d}"
                for idx, (x, fs, src) in enumerate(items, start=1):
                    x, x_peak_dB, g_ref_dB, y_ref, norm_info = process_example(
                        x=x,
                        fs=fs,
                        theta=theta,
                        normalize=args.normalize,
                        target_peak=args.target_peak,
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
                        extra_meta=smeta,
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
            for split_name, items in items_by_split.items():
                split_sub = f"{split_name}/perm_{perm_idx:03d}"
                for idx, (x, fs, src) in enumerate(items, start=1):
                    # Teacher = SSL hard-gate teacher; detector = abs
                    x, x_peak_dB, g_ref_dB, y_ref, norm_info = process_example(
                        x=x,
                        fs=fs,
                        theta=theta,
                        normalize=args.normalize,
                        target_peak=args.target_peak,
                    )
                    # Persist artifacts; keep the split-specific seed in metadata for provenance
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
                    )
                print(
                    f"Wrote {split_sub} with thresh={theta.comp_thresh:.1f}dB ratio={theta.comp_ratio:.1f} attack_fast={theta.attack_time_fast_ms:.1f}ms attack_slow={theta.attack_time_slow_ms:.1f}ms release_fast={theta.release_time_fast_ms:.1f}ms release_slow={theta.release_time_slow_ms:.1f}ms (clips: {len(items)})"
                )


if __name__ == "__main__":
    main()

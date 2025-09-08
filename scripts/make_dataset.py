#!/usr/bin/env python3
"""
Dataset builder: offline generation of (x, x_rms, g_ref_hard, y_ref, meta) examples.

Design choices/invariants (Phase 1):
- Teacher is the hard A/R baseline (compression-only); student will be compared against this.
- Detector is fixed (e.g., 20 ms) and implemented using lfilter-based EMA for speed.
- Sample rate is fixed across the dataset (default 44.1 kHz). Fail loudly if not satisfied.
- Outputs are written one example per directory under <output-dir>/<split>/clip_XXXX/.

Per-example artifacts:
- x.wav      : clean mono audio (float32)
- x_rms.pt   : detector envelope (float32) of shape (1, T)
- g_ref.pt   : teacher gain (hard A/R) (float32) of shape (1, T)
- y_ref.wav  : g_ref * x (float32)
- meta.yaml  : {fs, detector_ms, theta_ref, seed, processing_version, num_samples}

Synthetic signals come from differential_dynamics.benchmarks.signals (tone/step/burst/ramp).
Optional musical clips are cropped from --music-dir.
"""

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import math
import torch
import torchaudio
import yaml

from differential_dynamics.baselines.classical_compressor import ClassicalCompressor
from differential_dynamics.benchmarks.bench_utilities import ema_1pole_lfilter
from differential_dynamics.benchmarks.signals import (
    tone,
    step as step_sig,
    burst,
    ramp,
    composite_program,
)
from third_party.torchcomp_core.torchcomp import ms2coef


@dataclass
class Theta:
    comp_thresh_db: float
    comp_ratio: float
    attack_ms: float
    release_ms: float

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


def sample_theta(rng: random.Random) -> Theta:
    """Sample a single teacher parameter set θ.

    The sampled θ is intended to be applied to ALL clips within a permutation,
    so call this exactly once per perm (not per-clip) to avoid leakage of
    content dependence into θ.

    Ranges (reasonable studio-like defaults):
      - comp_thresh_db CT: uniform in [-36, -18] dB
      - comp_ratio    CR: choice from {2, 4, 8}
      - attack_ms   τ_a: log-uniform in [2, 50] ms
      - release_ms  τ_r: log-uniform in [20, 400] ms
    """
    # Threshold in [-36, -18] dB
    ct = rng.uniform(-36.0, -18.0)
    # Ratio from {2, 4, 8}
    cr = rng.choice([2.0, 4.0, 8.0])
    # Attack ms log-uniform [2, 50]
    at = math.exp(rng.uniform(math.log(2.0), math.log(50.0)))
    # Release ms log-uniform [20, 400]
    rt = math.exp(rng.uniform(math.log(20.0), math.log(400.0)))
    return Theta(ct, cr, at, rt)


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
    detector_ms: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute x_rms, teacher gain (hard A/R), and the compressed target.

    Args:
      x: (1, T) or (T,) clean audio (float32 preferred). Will be coerced to shape (1, T).
      fs: sample rate (Hz).
      theta: teacher parameters (CT/CR/attack/release in ms).
      detector_ms: detector time constant used to compute x_rms.

    Returns:
      x: (1, T) float32 clean audio
      x_rms: (1, T) float32 detector envelope
      g_ref: (1, T) float32 hard A/R teacher gain
      y_ref: (1, T) float32 compressed target (g_ref * x)
    """
    x = x.to(torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    # Detector envelope (RMS-like EMA of |x|)
    alpha_det = ms2coef(torch.tensor(detector_ms, dtype=torch.float32), fs)
    x_rms = ema_1pole_lfilter(x.abs(), alpha_det).clamp_min(1e-7)

    # Hard teacher gain (compression-only) via fast classical baseline
    comp = ClassicalCompressor(
        comp_thresh=theta.comp_thresh_db,
        comp_ratio=theta.comp_ratio,
        exp_thresh=-1000.0,
        exp_ratio=1.0,
        attack_time_ms=theta.attack_ms,
        release_time_ms=theta.release_ms,
        fs=fs,
        detector_time_ms=detector_ms,
    )
    g_ref = comp.classical_compexp_gain(x_rms)
    y_ref = g_ref * x
    return x, x_rms, g_ref, y_ref


def save_example(
    out_dir: Path,
    idx: int,
    split: str,
    x: torch.Tensor,
    x_rms: torch.Tensor,
    g_ref: torch.Tensor,
    y_ref: torch.Tensor,
    theta: Theta,
    fs: int,
    detector_ms: float,
    seed: int,
    processing_version: str = "v0",
) -> None:
    """Persist artifacts for a single example under clip_XXXX/.

    Files written:
      - x.wav, y_ref.wav
      - x_rms.pt, g_ref.pt
      - meta.yaml
    """
    ex_dir = out_dir / split / f"clip_{idx:04d}"
    ex_dir.mkdir(parents=True, exist_ok=True)

    # Waveforms
    torchaudio.save(str(ex_dir / "x.wav"), x.cpu(), fs)
    torchaudio.save(str(ex_dir / "y_ref.wav"), y_ref.cpu(), fs)

    # Tensors
    torch.save(x_rms.cpu(), ex_dir / "x_rms.pt")
    torch.save(g_ref.cpu(), ex_dir / "g_ref.pt")

    meta = {
        "fs": fs,
        "detector_ms": float(detector_ms),
        "theta_ref": theta.to_dict(),
        "seed": seed,
        "processing_version": processing_version,
        "num_samples": int(x.shape[-1]),
    }
    with open(ex_dir / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=True)


def list_music_files(music_dir: Path) -> list[Path]:
    """List audio files under music_dir."""
    paths: list[Path] = []
    for ext in (".wav", ".flac", ".mp3", ".m4a"):
        paths.extend([p for p in music_dir.rglob(f"*{ext}")])
    return paths


def allocate_music_files(files: list[Path], split_pcts: tuple[float, float, float], seed: int) -> dict[str, list[Path]]:
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
    p.add_argument("--output-dir", type=str, required=True, help="Root directory to write processed/{split}/perm_XXX/clip_XXXX")
    p.add_argument("--fs", type=int, default=44100, help="Sample rate (Hz)")
    p.add_argument("--detector-ms", type=float, default=20.0, help="Detector EMA time constant (ms)")
    p.add_argument("--clip-dur-s", type=float, default=2.0, help="Clip duration in seconds")

    # Seeds: separate θ from content to avoid leakage while keeping θ fixed per perm across splits
    p.add_argument("--theta-seed", type=int, default=1337, help="Seed to sample theta per permutation deterministically")
    p.add_argument("--seed-train", type=int, default=1001, help="RNG seed for train signal generation/crops")
    p.add_argument("--seed-val", type=int, default=1002, help="RNG seed for val signal generation/crops")
    p.add_argument("--seed-test", type=int, default=1003, help="RNG seed for test signal generation/crops")
    p.add_argument("--music-split-seed", type=int, default=2024, help="Seed to allocate music files into disjoint splits")

    # Music
    p.add_argument("--music-dir", type=str, default=None, help="Directory containing musical audio files (files will be split by file into train/val/test)")
    p.add_argument("--music-frac", type=float, default=0.5, help="Fraction of clips per split to source from music (rest from synthetic)")

    # Totals and split percentages
    p.add_argument("--num-total", type=int, default=120, help="Total clips per split (train/val/test)")
    p.add_argument("--split-pcts", type=str, default="80,10,10", help="Percent split for train,val,test (e.g., 80,10,10)")

    # Permutations: number of global-theta datasets to create
    p.add_argument("--num-perms", type=int, default=1, help="How many parameter permutations to generate; each permutation shares the same clips within a split but uses a single random theta for all clips")

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
        files_by_split = allocate_music_files(all_files, split_pcts, args.music_split_seed)

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
            crops = random_music_crops(files_by_split[split], args.fs, args.clip_dur_s, n_music, seed)
            items.extend([(clip, args.fs, f"music:{src}") for (clip, _, src) in crops])
        # If not enough music, fill deficit with synthetic
        if len(items) < n_music:
            deficit = n_music - len(items)
            n_synth += deficit
        # Synthetic via composite programs
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

    # Generate permutations with fixed theta per permutation across splits
    for perm_idx in range(1, args.num_perms + 1):
        # θ is sampled ONCE per permutation and reused across all splits
        theta_rng = random.Random(args.theta_seed + perm_idx)
        theta = sample_theta(theta_rng)
        for split_name, items in items_by_split.items():
            split_sub = f"{split_name}/perm_{perm_idx:03d}"
            for idx, (x, fs, src) in enumerate(items, start=1):
                # Teacher = hard A/R classical baseline; detector_ms fixed
                x, x_rms, g_ref, y_ref = process_example(
                    x, fs, theta, detector_ms=args.detector_ms
                )
                # Persist artifacts; keep the split-specific seed in metadata for provenance
                save_example(
                    out_dir,
                    idx,
                    split_sub,
                    x,
                    x_rms,
                    g_ref,
                    y_ref,
                    theta,
                    fs,
                    args.detector_ms,
                    seed={"train": args.seed_train, "val": args.seed_val, "test": args.seed_test}[split_name],
                )
            print(
                f"Wrote {split_sub} with CT={theta.comp_thresh_db:.1f}dB CR={theta.comp_ratio:.1f} AT={theta.attack_ms:.1f}ms RT={theta.release_ms:.1f}ms (clips: {len(items)})"
            )


if __name__ == "__main__":
    main()

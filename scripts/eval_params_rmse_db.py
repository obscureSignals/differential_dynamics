#!/usr/bin/env python3
"""
Evaluate parameter sets against a test split and report RMSE in dB.

Usage examples:
  # Compare test params against dataset reference g_ref_dB (preferred)
  python scripts/eval_params_rmse_db.py \
    --data-dir /path/to/processed_dataset \
    --test-params-yaml /path/to/test_params.yaml

  # Compare test params against forward from ref params (if no g_ref_dB in dataset)
  python scripts/eval_params_rmse_db.py \
    --data-dir /path/to/processed_dataset \
    --ref-params-yaml /path/to/ref_params.yaml \
    --test-params-yaml /path/to/test_params.yaml

Params YAML format (keys):
  comp_thresh_db: float
  comp_ratio: float (>1)
  T_attack_fast_ms: float
  T_attack_slow_ms: float
  T_shunt_fast_ms: float
  T_shunt_slow_ms: float
  feedback_coeff: float

Outputs:
  - Prints dataset size, fs, and average RMSE in dB over the test split.
  - If g_ref_dB.pt exists under each clip, use it as target; otherwise, forward(ref) is used as target.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
import yaml
import torch

from differential_dynamics.backends.torch.gain import SSL_comp_gain


def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_clip_dirs(root: Path) -> List[Path]:
    """Find clip directories under root by locating x_peak_dB.pt.

    This does not assume a particular split layout; it simply searches recursively
    for directories containing x_peak_dB.pt and returns the parent directories.
    If root itself is a clip directory (contains x_peak_dB.pt), it is returned.
    """
    root = Path(root)
    clips: List[Path] = []
    # Case 1: root itself is a clip dir
    if (root / "x_peak_dB.pt").is_file():
        return [root]
    # Case 2: search recursively
    for p in root.rglob("x_peak_dB.pt"):
        clips.append(p.parent)
    return sorted(clips)


def rmse_db_unmasked(y_ref_db: torch.Tensor, y_hat_db: torch.Tensor) -> torch.Tensor:
    diff = y_ref_db - y_hat_db
    return torch.sqrt(torch.mean(diff**2))


def forward_ssl(
    x_peak_dB: torch.Tensor,
    fs: float,
    params: Dict[str, float],
) -> torch.Tensor:
    """Run SSL hard-gate forward and return y_dB.

    params keys (ms are converted to seconds inside):
      comp_thresh_db, comp_ratio, T_attack_fast_ms, T_attack_slow_ms,
      T_shunt_fast_ms, T_shunt_slow_ms, feedback_coeff
    """
    # Convert ms -> seconds
    T_af = float(params["T_attack_fast_ms"]) / 1000.0
    T_as = float(params["T_attack_slow_ms"]) / 1000.0
    T_sf = float(params["T_shunt_fast_ms"]) / 1000.0
    T_ss = float(params["T_shunt_slow_ms"]) / 1000.0

    y_db = SSL_comp_gain(
        x_peak_dB=x_peak_dB,
        comp_thresh=float(params["comp_thresh_db"]),
        comp_ratio=float(params["comp_ratio"]),
        T_attack_fast=T_af,
        T_attack_slow=T_as,
        T_shunt_fast=T_sf,
        T_shunt_slow=T_ss,
        k=0.0,  # ignored for hard gate
        feedback_coeff=float(params["feedback_coeff"]),
        fs=float(fs),
        soft_gate=False,
    )
    return y_db


def summarize_dataset(root: Path, fs_cli: Optional[int]) -> Tuple[int, int]:
    clips = find_clip_dirs(root)
    if not clips:
        raise RuntimeError(f"No clips found under {root} (no x_peak_dB.pt)")
    # Try to read fs from meta.yaml in the first clip; fall back to CLI or 48000
    fs = 0
    meta_path = clips[0] / "meta.yaml"
    if meta_path.is_file():
        meta = load_yaml(meta_path)
        fs = int(meta.get("fs", 0))
    if fs <= 0:
        fs = int(fs_cli) if fs_cli else 48000
    return len(clips), fs


def main():
    p = argparse.ArgumentParser(
        description="Evaluate RMSE dB over test split for given parameter sets"
    )
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Processed dataset root (contains train/ val/ test/)",
    )
    p.add_argument(
        "--ref-params-yaml",
        type=str,
        default=None,
        help="YAML with reference params (used as target if g_ref_dB.pt absent)",
    )
    p.add_argument(
        "--test-params-yaml",
        type=str,
        required=True,
        help="YAML with test params to evaluate",
    )
    p.add_argument(
        "--device", type=str, default="cpu", help="cpu only (SSL smoother is CPU-only)"
    )
    p.add_argument(
        "--limit", type=int, default=0, help="Optional: limit number of clips (0 = all)"
    )
    p.add_argument(
        "--fs",
        type=int,
        default=0,
        help="Optional: sample rate if meta.yaml does not contain fs (default 48000)",
    )
    args = p.parse_args()

    root = Path(args.data_dir)
    device = torch.device(args.device)

    # Load params
    test_params = load_yaml(Path(args.test_params_yaml))
    ref_params: Optional[Dict[str, float]] = None
    if args.ref_params_yaml:
        ref_params = load_yaml(Path(args.ref_params_yaml))

    # Enumerate clip directories (flexible layout)
    clip_dirs = find_clip_dirs(root)
    if not clip_dirs:
        raise RuntimeError(f"No clips found under {root} (no x_peak_dB.pt)")
    if args.limit and args.limit > 0:
        clip_dirs = clip_dirs[: args.limit]

    # Infer fs using meta.yaml if available, else --fs, else 48000
    _, fs = summarize_dataset(root, fs_cli=args.fs if args.fs > 0 else None)

    losses: List[float] = []
    with torch.no_grad():
        for ex_dir in clip_dirs:
            x_peak_dB = torch.load(ex_dir / "x_peak_dB.pt").to(
                device=device, dtype=torch.float32
            )
            # Determine target: dataset g_ref_dB if present, else forward(ref)
            target_path = ex_dir / "g_ref_dB.pt"
            if target_path.is_file():
                y_ref_dB = torch.load(target_path).to(
                    device=device, dtype=torch.float32
                )
            else:
                if ref_params is None:
                    raise RuntimeError(
                        f"g_ref_dB.pt not found under {ex_dir}, and --ref-params-yaml was not provided"
                    )
                y_ref_dB = forward_ssl(x_peak_dB, fs=fs, params=ref_params).to(
                    torch.float32
                )

            y_test_dB = forward_ssl(x_peak_dB, fs=fs, params=test_params).to(
                torch.float32
            )
            loss = rmse_db_unmasked(y_ref_dB, y_test_dB)
            losses.append(float(loss.item()))

    avg_rmse = float(sum(losses) / max(1, len(losses)))
    print("==== Eval summary ====")
    print(f"clips: {len(clip_dirs)}  fs: {fs}")
    print(f"avg_rmse_db: {avg_rmse}")


if __name__ == "__main__":
    main()

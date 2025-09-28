#!/usr/bin/env python3
"""
Analyze dataset difficulty metrics strictly with respect to the hard teacher.

Metrics (per clip):
- near_threshold_occupancy: mean(|L - CT| < W_dB) where L = 20*log10(x_rms), CT from meta
- ar_flip_rate_hz: sign-change rate per second of the hard A/R decision
  decision_t = (g_target_linear[t] < y_prev) with y_prev = smoothed gain (g_ref)
- avg_gain_reduction_db: mean(-g_db) where g_db <= 0 (magnitude of reduction)

Outputs:
- Prints aggregate stats per split (mean/median over clips).
- Writes a summary YAML per split under <data-dir>/<split>_difficulty.yaml with per-clip entries.

Usage:
  python scripts/analyze_difficulty.py --data-dir /path/to/processed --W-db 3.0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import math
import statistics as stats

import torch
import yaml

from third_party.torchcomp_core.torchcomp import amp2db, db2amp


def list_meta(root: Path, split: str) -> List[Path]:
    p = root / split
    if not p.is_dir():
        return []
    return sorted(p.rglob("clip_*/meta.yaml"))


def load_tensors(ex_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    meta = yaml.safe_load(open(ex_dir / "meta.yaml", "r"))
    x_rms = torch.load(ex_dir / "x_rms.pt")
    g_ref = torch.load(ex_dir / "g_ref.pt")
    return x_rms.to(torch.float32), g_ref.to(torch.float32), meta


def static_gain_raw_linear(x_rms: torch.Tensor, ct: float, cr: float, et: float = -1000.0, er: float = 1.0) -> torch.Tensor:
    """Compute hard teacher static dB curve and convert to linear.
    Mirrors torchcomp's compexp static curve shape (compression-only defaults ok).
    x_rms: (B, T)
    Returns (B, T) linear g_target
    """
    B, T = x_rms.shape
    L = amp2db(torch.clamp(x_rms, min=1e-7))
    comp_ratio = torch.as_tensor(cr, device=x_rms.device, dtype=x_rms.dtype).expand(B)
    exp_ratio = torch.as_tensor(er, device=x_rms.device, dtype=x_rms.dtype).expand(B)
    comp_thresh = torch.as_tensor(ct, device=x_rms.device, dtype=x_rms.dtype).expand(B)
    exp_thresh = torch.as_tensor(et, device=x_rms.device, dtype=x_rms.dtype).expand(B)
    comp_slope = 1.0 - 1.0 / comp_ratio
    exp_slope = 1.0 - 1.0 / exp_ratio
    gain_raw_db = (
        torch.minimum(
            comp_slope[:, None] * (comp_thresh[:, None] - L),
            exp_slope[:, None] * (exp_thresh[:, None] - L),
        )
        .neg()
        .relu()
        .neg()
    )
    return db2amp(gain_raw_db)


def difficulty_for_clip(x_rms: torch.Tensor, g_ref: torch.Tensor, meta: Dict, W_db: float) -> Dict[str, float]:
    fs = int(meta["fs"])
    theta = meta.get("theta_ref", {})
    ct = float(theta.get("comp_thresh_db"))
    cr = float(theta.get("comp_ratio"))

    # near-threshold occupancy using the detector
    L = 20.0 * torch.log10(torch.clamp(x_rms, min=1e-7))
    occ = torch.mean((torch.abs(L - ct) < W_db).to(torch.float32)).item()

    # Recompute static target (linear) and count A/R decision flips
    g_tgt = static_gain_raw_linear(x_rms, ct=ct, cr=cr)
    B, T = g_tgt.shape
    assert B == 1, "Expected (1, T) tensors in dataset"
    y = g_ref[0]
    tgt = g_tgt[0]
    # decision: True if attack (tgt < y_prev), False if release
    y_prev = torch.tensor(1.0, dtype=y.dtype, device=y.device)
    flips = 0
    last_dec = None
    for t in range(T):
        dec = bool(tgt[t] < y_prev)
        if last_dec is not None and dec != last_dec:
            flips += 1
        # update y_prev for next step
        y_prev = y[t]
        last_dec = dec
    flip_rate_hz = flips / max(T / fs, 1e-12)

    # avg gain reduction magnitude in dB
    g_db = 20.0 * torch.log10(torch.clamp(g_ref, min=1e-7))
    avg_red = float((-g_db.clamp_max(0.0)).mean().item())

    return {
        "near_threshold_occupancy": float(occ),
        "ar_flip_rate_hz": float(flip_rate_hz),
        "avg_gain_reduction_db": float(avg_red),
    }


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(stats.fmean(values)),
        "median": float(stats.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def analyze_split(root: Path, split: str, W_db: float) -> Dict:
    metas = list_meta(root, split)
    per_clip = []
    for mp in metas:
        ex_dir = mp.parent
        try:
            x_rms, g_ref, meta = load_tensors(ex_dir)
        except Exception as e:
            continue
        d = difficulty_for_clip(x_rms, g_ref, meta, W_db)
        d_entry = {"clip_dir": str(ex_dir), **d}
        per_clip.append(d_entry)

    # Aggregate
    occs = [d["near_threshold_occupancy"] for d in per_clip]
    flips = [d["ar_flip_rate_hz"] for d in per_clip]
    reds = [d["avg_gain_reduction_db"] for d in per_clip]
    summary = {
        "num_clips": len(per_clip),
        "near_threshold_occupancy": summarize(occs),
        "ar_flip_rate_hz": summarize(flips),
        "avg_gain_reduction_db": summarize(reds),
        "clips": per_clip,
    }
    return summary


def main():
    p = argparse.ArgumentParser(description="Analyze hard-teacher difficulty metrics per split")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--W-db", type=float, default=3.0, help="Near-threshold band width in dB")
    args = p.parse_args()

    root = Path(args.data_dir)
    out = {}
    for split in ["train", "val", "test"]:
        s = analyze_split(root, split, args.W_db)
        out[split] = s
        print(f"[{split}]  clips={s['num_clips']}  occ.mean={s['near_threshold_occupancy']['mean']:.3f}  flips.mean={s['ar_flip_rate_hz']['mean']:.2f} Hz  red_db.mean={s['avg_gain_reduction_db']['mean']:.2f}")
        # Write per-split summary yaml
        with open(root / f"{split}_difficulty.yaml", "w") as f:
            yaml.safe_dump(s, f, sort_keys=False)


if __name__ == "__main__":
    main()

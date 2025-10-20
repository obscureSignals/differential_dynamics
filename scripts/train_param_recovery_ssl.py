#!/usr/bin/env python3
"""
Train/evaluate global SSL parameter recovery (dB-domain) against the hard-gate teacher

Key points:
- Loads dB-domain tensors written by the SSL dataset builder:
  * x_peak_dB.pt (20*log10(|x|) envelope)
  * g_ref_dB.pt  (teacher gain in dB)
- The model consumes x_peak_dB and outputs y_dB; loss is RMSE in dB with a dB mask.
- Early stopping on validation loss with patience.
- Imports ParamRecoveryModelSLL.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import os
import math
import torch
import yaml
from torch.utils.data import Dataset, DataLoader


def compute_loss(
    x_peak_dB: torch.Tensor,
    g_ref_dB: torch.Tensor,
    y_dB: torch.Tensor,
    loss_mode: str,
    comp_thresh_val: float | None,
    post_ct_only: bool,
) -> torch.Tensor:
    """Compute RMSE in dB with optional signal weighting and post-CT masking.

    - loss_mode: 'signal' uses derivative/region weighting; 'none' is plain RMSE.
    - post_ct_only: if True, restrict to frames with x_peak_dB > comp_thresh_val.
    """
    diff = g_ref_dB - y_dB
    if post_ct_only and comp_thresh_val is not None:
        mask = x_peak_dB > float(comp_thresh_val)
    else:
        mask = torch.ones_like(diff, dtype=torch.bool)

    if loss_mode == "signal":
        # Base weights as in rmse_db_weighted_signal
        B, T = diff.shape
        active = x_peak_dB > -80.0
        dx = x_peak_dB[:, 1:] - x_peak_dB[:, :-1]
        dy = g_ref_dB[:, 1:] - g_ref_dB[:, :-1]
        steady = (
            (dx.abs() < 0.01)
            & (dy.abs() < 0.01)
            & (g_ref_dB[:, 1:] < -1.0)
            & (x_peak_dB[:, 1:] > -60.0)
        )
        trans = (dx.abs() > 0.05) | (dy.abs() > 0.05)
        w = torch.ones_like(diff)
        w = w * active.float()
        w_tail = w[:, 1:]
        w_tail = w_tail + 0.5 * steady.float() + 0.5 * trans.float()
        w[:, 1:] = w_tail
        # Apply post-CT mask if requested
        w = w * mask.float()
        # Weighted RMSE across the batch
        num = (w * diff * diff).sum()
        den = torch.clamp(w.sum(), min=1e-8)
        rmse = torch.sqrt(num / den)
        return rmse
    else:
        # Plain RMSE with optional mask
        if mask.any():
            d = diff[mask]
        else:
            d = diff.view(-1)
        num = (d * d).sum()
        den = torch.clamp(
            torch.tensor(d.numel(), dtype=d.dtype, device=d.device), min=1e-8
        )
        return torch.sqrt(num / den)


from differential_dynamics.training.param_recovery_model_sll import (
    ParamRecoveryModelSLL,
)


def rmse_db_unmasked(y_ref_db: torch.Tensor, y_hat_db: torch.Tensor) -> torch.Tensor:
    """RMSE directly in dB over all samples."""
    diff = y_ref_db - y_hat_db
    return torch.sqrt(torch.mean(diff**2))


def rmse_db_weighted_signal(
    x_peak_db: torch.Tensor, y_ref_db: torch.Tensor, y_hat_db: torch.Tensor
) -> torch.Tensor:
    """Signal-derived weighted RMSE in dB.

    Uses only x_peak_db and y_ref_db to build per-frame weights:
    - Exclude deep silence (x < -80 dB)
    - Emphasize steady compressed regions (statics)
    - Emphasize transients/edges (dynamics)
    """
    # Base diff
    diff = y_ref_db - y_hat_db
    B, T = diff.shape
    # Active mask (exclude deep silence)
    active = x_peak_db > -80.0
    # Derivatives (align to T by padding first sample)
    dx = x_peak_db[:, 1:] - x_peak_db[:, :-1]
    dy = y_ref_db[:, 1:] - y_ref_db[:, :-1]
    # Steady where both signals barely change and compression is active
    steady = (
        (dx.abs() < 0.01)
        & (dy.abs() < 0.01)
        & (y_ref_db[:, 1:] < -1.0)
        & (x_peak_db[:, 1:] > -60.0)
    )
    # Transients where either changes notably
    trans = (dx.abs() > 0.05) | (dy.abs() > 0.05)
    # Build weights
    w = torch.ones_like(diff)
    w = w * active.float()
    w_tail = w[:, 1:]
    w_tail = w_tail + 0.5 * steady.float() + 0.5 * trans.float()
    w[:, 1:] = w_tail
    # Normalize and compute RMSE
    w_sum = torch.clamp(w.mean(), min=1e-8)
    rmse = torch.sqrt(((w * diff * diff).mean() / w_sum))
    return rmse


def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_meta_paths(root: Path, split: str, perm_id: int | None = None) -> List[Path]:
    """Find meta.yaml files for a given split.

    If perm_id is provided, restrict to that permutation directory only:
      - <root>/<split>/perm_{perm_id:03d}/clip_YYYY/meta.yaml
    Otherwise include all clips recursively under <root>/<split>.
    """
    split_dir = root / split
    if not split_dir.is_dir():
        return []
    if perm_id is not None:
        perm_dir = split_dir / f"perm_{perm_id:03d}"
        if not perm_dir.is_dir():
            return []
        return sorted(perm_dir.rglob("clip_*/meta.yaml"))
    # Prefer recursive glob to handle both forms
    return sorted(split_dir.rglob("clip_*/meta.yaml"))


class FullSplitDataset(Dataset):
    """Dataset that loads clips for a split, optionally restricted to a permutation.

    Returns dict with tensors and meta for compatibility with the trainer.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        perm_id: int | None = None,
        allow_probe_types: List[str] | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.perm_id = perm_id
        all_items = find_meta_paths(self.root, split, perm_id=perm_id)
        if not all_items:
            where = f"{self.root}/{split}"
            if perm_id is not None:
                where = f"{self.root}/{split}/perm_{perm_id:03d}"
            raise RuntimeError(f"No items found under {where}")
        if allow_probe_types is None:
            self.items = all_items
        else:
            # Filter by probe_type in meta
            filt: List[Path] = []
            for mp in all_items:
                try:
                    with open(mp, "r") as f:
                        m = yaml.safe_load(f)
                    pt = m.get("probe_type") or (m.get("source") or "")
                    # Accept if tagged probe_type in allow list, or source contains synth:P#
                    tag = str(pt).split(":")[-1]
                    if tag in allow_probe_types:
                        filt.append(mp)
                except Exception:
                    continue
            if not filt:
                where = f"{self.root}/{split}"
                raise RuntimeError(
                    f"No items found for allowed probes {allow_probe_types} under {where}"
                )
            self.items = filt

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        meta_path = self.items[idx]
        ex_dir = meta_path.parent
        meta = load_yaml(meta_path)
        x_peak_dB = torch.load(ex_dir / "x_peak_dB.pt")
        g_ref_dB = torch.load(ex_dir / "g_ref_dB.pt")
        theta_ref = meta.get("theta_ref", {})
        return {
            "x_peak_dB": x_peak_dB.to(torch.float32),
            "g_ref_dB": g_ref_dB.to(torch.float32),
            "meta": meta,
            "theta_ref": theta_ref,
        }


def collate(batch: List[Dict]) -> Dict:
    keys = ["x_peak_dB", "g_ref_dB"]
    out = {k: torch.cat([b[k] for b in batch], dim=0) for k in keys}
    out["meta"] = [b["meta"] for b in batch]
    out["theta_ref"] = [b["theta_ref"] for b in batch]
    return out


def evaluate(
    model: ParamRecoveryModelSLL,
    loader: DataLoader,
    device: torch.device,
    loss_mode: str = "signal",
    post_ct_only: bool = False,
) -> float:
    model.eval()
    total_num = 0.0
    total_den = 0.0
    with torch.no_grad():
        for batch in loader:
            x_peak_dB = batch["x_peak_dB"].to(device)
            g_ref_dB = batch["g_ref_dB"].to(device)
            y_dB = model(x_peak_dB)
            ct_val = (
                float(model.comp_thresh.item())
                if hasattr(model, "comp_thresh")
                else None
            )
            # Recompute numerator/denominator to aggregate properly across batches
            diff = g_ref_dB - y_dB
            if post_ct_only and ct_val is not None:
                mask = x_peak_dB > float(ct_val)
            else:
                mask = torch.ones_like(diff, dtype=torch.bool)
            if loss_mode == "signal":
                active = x_peak_dB > -80.0
                dx = x_peak_dB[:, 1:] - x_peak_dB[:, :-1]
                dy = g_ref_dB[:, 1:] - g_ref_dB[:, :-1]
                steady = (
                    (dx.abs() < 0.01)
                    & (dy.abs() < 0.01)
                    & (g_ref_dB[:, 1:] < -1.0)
                    & (x_peak_dB[:, 1:] > -60.0)
                )
                trans = (dx.abs() > 0.05) | (dy.abs() > 0.05)
                w = torch.ones_like(diff)
                w = w * active.float()
                w_tail = w[:, 1:]
                w_tail = w_tail + 0.5 * steady.float() + 0.5 * trans.float()
                w[:, 1:] = w_tail
                w = w * mask.float()
                num = float((w * diff * diff).sum().item())
                den = float(torch.clamp(w.sum(), min=1e-8).item())
            else:
                if mask.any():
                    d = diff[mask]
                else:
                    d = diff.view(-1)
                num = float((d * d).sum().item())
                den = float(max(1e-8, d.numel()))
            total_num += num
            total_den += den
    rmse = math.sqrt(total_num / max(1e-8, total_den))
    return float(rmse)


def summarize_target_params(loader: DataLoader) -> Dict[str, float | int | list]:
    """Extract a representative target θ from metadata (first few items).
    Adapts to SSL theta fields.
    """
    thetas: List[Dict] = []
    for i, batch in enumerate(loader):
        thetas.extend(batch["theta_ref"])  # type: ignore[index]
        if len(thetas) >= 32:
            break

    def norm(d: Dict) -> Dict[str, float]:
        return {
            "comp_thresh_db": float(d.get("comp_thresh", d.get("CT", 0.0))),
            "comp_ratio": float(d.get("comp_ratio", d.get("CR", 0.0))),
            "T_attack_fast_ms": float(
                d.get("attack_time_fast_ms", d.get("T_AF_MS", 0.0))
            ),
            "T_attack_slow_ms": float(
                d.get("attack_time_slow_ms", d.get("T_AS_MS", 0.0))
            ),
            "T_shunt_fast_ms": float(
                d.get("release_time_fast_ms", d.get("T_SF_MS", 0.0))
            ),
            "T_shunt_slow_ms": float(
                d.get("release_time_slow_ms", d.get("T_SS_MS", 0.0))
            ),
            "feedback_coeff": float(d.get("feedback_coeff", d.get("FB", 0.0))),
        }

    thetas_n = [norm(t) for t in thetas]
    keys = [tuple(t.items()) for t in thetas_n]
    unique = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(dict(k))
    summary: Dict[str, float | int | list] = {}
    summary["num_unique_target_param_sets"] = len(unique)
    summary["target_params"] = unique[0] if unique else {}
    return summary


def main():
    p = argparse.ArgumentParser(description="Parameter recovery with early stopping")
    # Solver selection: analytic (operator-Jacobian contraction) or FD (scalar-loss)
    # We set environment variables for the C++ kernel accordingly.
    p.add_argument(
        "--solver",
        type=str,
        default="analytic",
        choices=["analytic", "fd"],
        help="Gradient solver: 'analytic' uses operator-Jacobian contraction; 'fd' uses scalar-loss FD",
    )

    p.add_argument(
        "--use-saltation",
        type=bool,
        default=False,
        help="Use saltation for discontinuity handling in analytic solver",
    )

    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Processed dataset root (expects a 'train' set)",
    )
    # System identification defaults (always on)
    p.add_argument(
        "--stop-delta",
        type=float,
        default=1e-3,
        help="Convergence threshold on normalized parameter changes for early stop",
    )
    p.add_argument(
        "--stop-patience-epochs",
        type=int,
        default=10,
        help="Number of consecutive epochs below stop-delta required to stop",
    )
    p.add_argument(
        "--device", type=str, default="cpu", help="cpu only (SSL smoother is CPU-only)"
    )
    p.add_argument(
        "--perm-id",
        type=int,
        default=1,
        help="If set, restrict to this permutation (perm_XXX).",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=500)

    p.add_argument(
        "--loss-weights",
        type=str,
        default="none",
        choices=["none", "signal"],
        help="Loss weighting strategy: 'signal' uses signal-derived per-frame weights; 'none' is plain RMSE",
    )
    p.add_argument(
        "--loss-post-ct-only",
        action="store_true",
        help="Restrict loss to frames with x_peak_dB > comp_thresh (compressed region)",
    )
    p.add_argument(
        "--fix-fb",
        type=float,
        default=None,
        help="If set, fix feedback coefficient to this constant (0..1) and do not learn it",
    )
    p.add_argument(
        "--fb-lr-scale",
        type=float,
        default=0.3,
        help="Learning rate scale for feedback parameter group (ignored if --fix-fb is set)",
    )
    p.add_argument(
        "--fix-fb-from-meta",
        action="store_true",
        help="If set, freeze feedback to the dataset metadata value (per permutation)",
    )
    p.add_argument(
        "--fix-ct-from-meta",
        action="store_true",
        help="If set, freeze compressor threshold (CT) to the dataset metadata value",
    )

    # Preset to freeze time constants for ramp-based ratio/CT identification
    p.add_argument(
        "--freeze-tconsts-fast-slow",
        action="store_true",
        help="Freeze T_af and T_sf to their minimum (fast) and T_as and T_ss to their maximum (slow) to enforce quasi-static ramp behavior",
    )

    p.add_argument("--lr", type=float, default=0.05)
    # LR schedule (start/end) — preserves param-group ratios via multiplicative schedule
    p.add_argument(
        "--lr-start",
        type=float,
        default=None,
        help="Initial learning rate (defaults to --lr)",
    )
    p.add_argument(
        "--lr-end", type=float, default=1e-2, help="Final learning rate at last epoch"
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=100,
        help="Number of epochs with no improvement before stopping",
    )
    # No extra stabilization flags by default; keep script simple

    p.add_argument(
        "--dbg-fd-ratio",
        action="store_true",
        help="Print finite-difference vs autograd gradient for ratio_logit on first batch each epoch",
    )
    p.add_argument(
        "--dbg-fd-eps", type=float, default=1e-3, help="FD step in ratio_logit space"
    )
    p.add_argument(
        "--dbg-fd-every", type=int, default=1, help="Run FD check every N epochs"
    )

    args = p.parse_args()

    root = Path(args.data_dir)
    device = torch.device(args.device)

    # Configure kernel gradient mode (minimal interface)
    if args.solver == "fd":
        # Scalar-loss FD for time-constant gradients (event-sensitive by default)
        os.environ["SSL_USE_FD_TCONST_GRADS"] = "1"
        # Unset analytic toggles to avoid confusion
        os.environ.pop("SSL_USE_ANALYTIC_JAC", None)
        os.environ.pop("SSL_USE_ANALYTIC_JAC_AD", None)
        os.environ.pop("SSL_USE_ANALYTIC_JAC_BD", None)
        os.environ.pop("SSL_ANALYTIC_BD_METHOD", None)
    else:
        # Analytic operator-Jacobian contraction (fixed mask). Use phi (linear-solve) for Bd.
        os.environ["SSL_USE_FD_TCONST_GRADS"] = "0"
        os.environ["SSL_USE_ANALYTIC_JAC"] = "1"
        os.environ["SSL_ANALYTIC_BD_METHOD"] = "phi"
        if args.use_saltation:
            os.environ["SSL_USE_SALTATION"] = "1"
            os.environ["SSL_SALTATION_MAX_FLIPS"] = "2048"
            os.environ.setdefault("SSL_SALTATION_EPS_REL", "1e-2")
            os.environ.setdefault("SSL_SALTATION_MAX_BACKOFF", "6")

    print(f"[train_param_recovery_ssl] solver={args.solver}")

    # Build dataset (System-ID: use only 'train')
    ds_train = FullSplitDataset(root, split="train", perm_id=args.perm_id)

    # Assume consistent sample rate across dataset (enforced by builder)
    fs = int(ds_train[0]["meta"]["fs"])  # type: ignore[index]

    # Model and optim
    model = ParamRecoveryModelSLL(fs=fs)
    model.to(device)

    # Optional: freeze time constants to fast/slow preset for ramp identification
    if args.freeze_tconsts_fast_slow:
        with torch.no_grad():
            # Helper to map target seconds to internal u (logit of normalized [tmin,tmax])
            def _u_from_sec(val_s: float, tmin: float, tmax: float) -> torch.Tensor:
                s = (float(val_s) - tmin) / (tmax - tmin)
                s = max(min(s, 1.0 - 1e-6), 1e-6)
                return torch.logit(torch.tensor(s, dtype=torch.float32))

            # Set fast branches to minimal (fast) and slow branches to maximal (slow)
            model.u_T_af.copy_(
                _u_from_sec(model.T_af_min, model.T_af_min, model.T_af_max)
            )
            model.u_T_sf.copy_(
                _u_from_sec(model.T_sf_min, model.T_sf_min, model.T_sf_max)
            )
            model.u_T_as.copy_(
                _u_from_sec(model.T_as_max, model.T_as_min, model.T_as_max)
            )
            model.u_T_ss.copy_(
                _u_from_sec(model.T_ss_max, model.T_ss_min, model.T_ss_max)
            )
        # Freeze gradients
        model.u_T_af.requires_grad_(False)
        model.u_T_as.requires_grad_(False)
        model.u_T_sf.requires_grad_(False)
        model.u_T_ss.requires_grad_(False)
        print(
            "[train_param_recovery_ssl] Frozen TCs preset: T_af/T_sf=fast, T_as/T_ss=slow"
        )

    # Resolve LR schedule endpoints
    lr_start = float(args.lr_start) if args.lr_start is not None else float(args.lr)
    lr_end = float(args.lr_end)

    # Cosine LR schedule (epoch-wise) that scales all groups by the same factor
    def _make_cosine_lambda(lr_start_v: float, lr_end_v: float, epochs_v: int):
        def _f(ep: int) -> float:
            T = max(1, int(epochs_v))
            e = min(max(ep, 0), T)
            gamma = float(lr_end_v) / max(1e-12, float(lr_start_v))
            return float(
                gamma + (1.0 - gamma) * 0.5 * (1.0 + math.cos(math.pi * e / T))
            )

        return _f

    # Optionally fix feedback to a constant (no learning)
    if args.fix_fb is not None:
        fb_const = float(min(max(args.fix_fb, 0.0), 1.0 - 1e-6))
        with torch.no_grad():
            model.fb_logit.copy_(
                torch.logit(torch.tensor(fb_const, dtype=torch.float32))
            )
        model.fb_logit.requires_grad_(False)
        print(
            f"[train_param_recovery_ssl] Fixing feedback_coeff={fb_const:.6f} (frozen)"
        )

    # Optimizer with smaller LR for ratio, and optionally reduced LR for feedback
    ratio_lr = lr_start * 0.5
    param_groups = []
    param_groups.append({"params": [model.ratio_logit], "lr": ratio_lr})

    other_params = [model.comp_thresh]
    # Add time-constant params only if trainable
    if model.u_T_af.requires_grad:
        other_params.append(model.u_T_af)
    if model.u_T_as.requires_grad:
        other_params.append(model.u_T_as)
    if model.u_T_sf.requires_grad:
        other_params.append(model.u_T_sf)
    if model.u_T_ss.requires_grad:
        other_params.append(model.u_T_ss)
    # Feedback param group only if learnable
    if model.fb_logit.requires_grad:
        fb_lr = lr_start * float(args.fb_lr_scale)
        param_groups.append({"params": [model.fb_logit], "lr": fb_lr})
    # Remaining group at base LR
    param_groups.append({"params": other_params, "lr": lr_start})

    opt = torch.optim.Adam(param_groups)

    # DataLoaders (system identification: deterministic order, no shuffle)
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    # Ground-truth target summary (fail loudly if mixed) and print for reference
    target_summary = summarize_target_params(train_loader)
    if target_summary.get("num_unique_target_param_sets", 0) != 1:
        raise RuntimeError(
            f"Expected a single target parameter set (use --perm-id). Found: {target_summary.get('num_unique_target_param_sets')}"
        )
    print("Ground truth (from dataset):", target_summary.get("target_params"))

    # If requested, fix compressor threshold (CT) from metadata and rebuild optimizer accordingly
    if args.fix_ct_from_meta:
        tp = target_summary.get("target_params", {})
        ct_meta = tp.get("comp_thresh_db", None)
        if ct_meta is None:
            raise RuntimeError(
                "--fix-ct-from-meta set but comp_thresh_db not found in metadata"
            )
        with torch.no_grad():
            model.comp_thresh.copy_(torch.tensor(float(ct_meta), dtype=torch.float32))
        model.comp_thresh.requires_grad_(False)
        print(
            f"[train_param_recovery_ssl] Fixing comp_thresh from metadata: {float(ct_meta):.6f} dB"
        )
        # Rebuild optimizer param groups to reflect frozen CT (and possibly frozen fb/T*)
        ratio_lr = float(ratio_lr)  # keep prior ratio group lr
        param_groups = []
        param_groups.append({"params": [model.ratio_logit], "lr": ratio_lr})
        other_params = []
        if model.fb_logit.requires_grad:
            fb_lr = lr_start * float(args.fb_lr_scale)
            param_groups.append({"params": [model.fb_logit], "lr": fb_lr})
        # Time constants only if trainable
        if model.u_T_af.requires_grad:
            other_params.append(model.u_T_af)
        if model.u_T_as.requires_grad:
            other_params.append(model.u_T_as)
        if model.u_T_sf.requires_grad:
            other_params.append(model.u_T_sf)
        if model.u_T_ss.requires_grad:
            other_params.append(model.u_T_ss)
        if other_params:
            param_groups.append({"params": other_params, "lr": lr_start})
        opt = torch.optim.Adam(param_groups)
        # Recreate scheduler bound to the new optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=_make_cosine_lambda(lr_start, lr_end, args.epochs)
        )

    # If requested, fix feedback from metadata and rebuild optimizer accordingly
    if args.fix_fb_from_meta and args.fix_fb is None:
        tp = target_summary.get("target_params", {})
        fb_meta = tp.get("feedback_coeff", None)
        if fb_meta is None:
            raise RuntimeError(
                "--fix-fb-from-meta was set but feedback_coeff not found in metadata"
            )
        fb_const = float(min(max(float(fb_meta), 0.0), 1.0 - 1e-6))
        with torch.no_grad():
            model.fb_logit.copy_(
                torch.logit(torch.tensor(fb_const, dtype=torch.float32))
            )
        model.fb_logit.requires_grad_(False)
        print(
            f"[train_param_recovery_ssl] Fixing feedback_coeff from metadata: {fb_const:.6f}"
        )
        # Rebuild optimizer param groups to reflect frozen fb
        ratio_lr = lr_start * 0.5
        param_groups = []
        param_groups.append({"params": [model.ratio_logit], "lr": ratio_lr})
        other_params = [model.comp_thresh]
        if model.u_T_af.requires_grad:
            other_params.append(model.u_T_af)
        if model.u_T_as.requires_grad:
            other_params.append(model.u_T_as)
        if model.u_T_sf.requires_grad:
            other_params.append(model.u_T_sf)
        if model.u_T_ss.requires_grad:
            other_params.append(model.u_T_ss)
        if model.fb_logit.requires_grad:
            fb_lr = lr_start * float(args.fb_lr_scale)
            param_groups.append({"params": [model.fb_logit], "lr": fb_lr})
        param_groups.append({"params": other_params, "lr": lr_start})
        opt = torch.optim.Adam(param_groups)

    best_state = None
    patience_left = args.early_stop_patience
    global_step = 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=_make_cosine_lambda(lr_start, lr_end, args.epochs)
    )

    # System identification convergence tracking
    prev_params = None
    stable_epochs = 0

    # Single-stage training loop (track best checkpoint by train RMSE)
    best_train_loss = float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        # Accumulate numerators/denominators to compute true epoch RMSE (not avg of batch RMSEs)
        epoch_num = 0.0
        epoch_den = 0.0
        for batch in train_loader:
            x_peak_dB = batch["x_peak_dB"].to(device)
            g_ref_dB = batch["g_ref_dB"].to(device)

            y_dB = model(x_peak_dB)

            loss = compute_loss(
                x_peak_dB,
                g_ref_dB,
                y_dB,
                loss_mode=args.loss_weights,
                comp_thresh_val=float(model.comp_thresh.item()),
                post_ct_only=bool(args.loss_post_ct_only),
            )
            if not torch.isfinite(loss):
                print("[warn] non-finite loss encountered; skipping step")
                opt.zero_grad()
                continue
            opt.zero_grad()
            loss.backward()

            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # FD vs autograd check for ratio (first batch each epoch)
            if args.dbg_fd_ratio and (epoch % max(1, args.dbg_fd_every) == 0):
                if "_did_dbg_epoch" not in locals() or not _did_dbg_epoch:
                    with torch.no_grad():
                        u0 = model.ratio_logit.detach().clone()
                        eps = float(args.dbg_fd_eps)

                        def _run_loss_for_u(uval: torch.Tensor) -> float:
                            model.ratio_logit.copy_(uval)
                            y_tmp = model(x_peak_dB)
                            Ltmp = compute_loss(
                                x_peak_dB,
                                g_ref_dB,
                                y_tmp,
                                loss_mode=args.loss_weights,
                                comp_thresh_val=float(model.comp_thresh.item()),
                                post_ct_only=bool(args.loss_post_ct_only),
                            )
                            return float(Ltmp.item())

                        Lp = _run_loss_for_u(u0 + eps)
                        Lm = _run_loss_for_u(u0 - eps)
                        g_fd = (Lp - Lm) / (2.0 * eps)
                        g_aut = (
                            float(model.ratio_logit.grad.item())
                            if model.ratio_logit.grad is not None
                            else float("nan")
                        )
                        ratio_now = (
                            1.0 + (model.ratio_rmax - 1.0) * torch.sigmoid(u0)
                        ).item()
                        print(
                            f"[dbg_fd_ratio] epoch={epoch} fd={g_fd:.6g} aut={g_aut:.6g} ratio={ratio_now:.4f} CT={float(model.comp_thresh.item()):.3f}"
                        )
                        model.ratio_logit.copy_(u0)
                    _did_dbg_epoch = True

            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                print("[warn] non-finite gradients encountered; skipping step")
                opt.zero_grad()
                continue
            opt.step()
            # Accumulate epoch numerators/denominators for correct RMSE averaging
            with torch.no_grad():
                diff = g_ref_dB - y_dB
                if bool(args.loss_post_ct_only):
                    mask = x_peak_dB > float(model.comp_thresh.item())
                else:
                    mask = torch.ones_like(diff, dtype=torch.bool)
                if args.loss_weights == "signal":
                    active = x_peak_dB > -80.0
                    dx = x_peak_dB[:, 1:] - x_peak_dB[:, :-1]
                    dy = g_ref_dB[:, 1:] - g_ref_dB[:, :-1]
                    steady = (
                        (dx.abs() < 0.01)
                        & (dy.abs() < 0.01)
                        & (g_ref_dB[:, 1:] < -1.0)
                        & (x_peak_dB[:, 1:] > -60.0)
                    )
                    trans = (dx.abs() > 0.05) | (dy.abs() > 0.05)
                    w = torch.ones_like(diff)
                    w = w * active.float()
                    w_tail = w[:, 1:]
                    w_tail = w_tail + 0.5 * steady.float() + 0.5 * trans.float()
                    w[:, 1:] = w_tail
                    w = w * mask.float()
                    num = float((w * diff * diff).sum().item())
                    den = float(torch.clamp(w.sum(), min=1e-8).item())
                else:
                    if mask.any():
                        d = diff[mask]
                    else:
                        d = diff.view(-1)
                    num = float((d * d).sum().item())
                    den = float(max(1e-8, d.numel()))
                epoch_num += num
                epoch_den += den
            with torch.no_grad():
                model.ratio_logit.clamp_(-10.0, 10.0)
                model.comp_thresh.clamp_(-60.0, 0.0)
                if model.fb_logit.requires_grad:
                    model.fb_logit.clamp_(-8.0, 8.0)
                model.u_T_af.clamp_(-12.0, 12.0)
                # Remove u-space clamps; bounds are enforced by sigmoid->seconds mapping
                # If needed, clamp seconds via parameterization, not u.
            global_step += 1
        avg_train = math.sqrt(epoch_num / max(1e-8, epoch_den))
        # Show current LR (first group) for reference
        cur_lr_ratio = scheduler.get_last_lr()[0]
        cur_lr = scheduler.get_last_lr()[1]
        print(
            f"epoch={epoch} train_rmse_db={avg_train:.4f} lr={cur_lr_ratio:.4g},{cur_lr:.4g} params={model.params_readable()}"
        )
        # Step LR scheduler per epoch
        scheduler.step()
        # Track best state
        if avg_train <= best_train_loss:
            best_train_loss = avg_train
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch

    # 1) Compute final (last-epoch) metrics BEFORE touching state
    final_train_loss = evaluate(
        model,
        train_loader,
        device,
        loss_mode=args.loss_weights,
        post_ct_only=bool(args.loss_post_ct_only),
    )
    final_params = model.params_readable()
    print(f"[final] epoch={args.epochs-1} final_train_rmse_db={final_train_loss:.4f}")

    # 2) Compute best metrics by restoring best_state (if any)
    if best_state is not None:
        model.load_state_dict(best_state)
        best_train_loss = evaluate(
            model,
            train_loader,
            device,
            loss_mode=args.loss_weights,
            post_ct_only=bool(args.loss_post_ct_only),
        )
        print(f"[best] epoch={best_epoch} best_train_rmse_db={best_train_loss:.4f}")
    best_params = model.params_readable()

    # Summary with both final and best
    target_summary = (
        target_summary
        if "target_summary" in locals()
        else summarize_target_params(train_loader)
    )
    if target_summary.get("num_unique_target_param_sets", 0) != 1:
        raise RuntimeError(
            f"Expected a single target parameter set (use --perm-id). Found: {target_summary.get('num_unique_target_param_sets')}"
        )
    summary = {
        "final_train_rmse_db": float(final_train_loss),
        "best_train_rmse_db": float(best_train_loss),
        "best_epoch": int(best_epoch),
        "target": target_summary,
        "final_params": final_params,
        "best_params": best_params,
    }

    # Print summary
    print("==== Training summary ====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save params and summary (both final and best)
    out_final = root / "final_params_ssl.yaml"
    with open(out_final, "w") as f:
        yaml.safe_dump(final_params, f)
    out_best = root / "best_params_ssl.yaml"
    with open(out_best, "w") as f:
        yaml.safe_dump(best_params, f)
    out_summary = root / "summary_ssl.yaml"
    with open(out_summary, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=True)
    print(f"Saved final params to {out_final}")
    print(f"Saved best params to {out_best}")
    print(f"Saved summary to {out_summary}")


if __name__ == "__main__":
    main()

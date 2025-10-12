#!/usr/bin/env python3
"""
Train/evaluate global SSL parameter recovery (dB-domain) against the hard-gate teacher
using the full train/val/test dataset structure under --data-dir.

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
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from differential_dynamics.training.param_recovery_model_sll import (
    ParamRecoveryModelSLL,
)


def rmse_db_unmasked(y_ref_db: torch.Tensor, y_hat_db: torch.Tensor) -> torch.Tensor:
    """RMSE directly in dB over all samples."""
    diff = y_ref_db - y_hat_db
    return torch.sqrt(torch.mean(diff**2))


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

    def __init__(self, root: Path, split: str, perm_id: int | None = None):
        self.root = Path(root)
        self.split = split
        self.perm_id = perm_id
        self.items = find_meta_paths(self.root, split, perm_id=perm_id)
        if not self.items:
            where = f"{self.root}/{split}"
            if perm_id is not None:
                where = f"{self.root}/{split}/perm_{perm_id:03d}"
            raise RuntimeError(f"No items found under {where}")

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
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            x_peak_dB = batch["x_peak_dB"].to(device)
            g_ref_dB = batch["g_ref_dB"].to(device)
            y_dB = model(x_peak_dB)
            loss = rmse_db_unmasked(g_ref_dB, y_dB)
            losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


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
    p = argparse.ArgumentParser(
        description="Parameter recovery using full train/val/test splits with early stopping"
    )
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
        "--data-dir",
        type=str,
        required=True,
        help="Processed dataset root (contains train/ val/ test/)",
    )
    p.add_argument(
        "--device", type=str, default="cpu", help="cpu only (SSL smoother is CPU-only)"
    )
    p.add_argument(
        "--perm-id",
        type=int,
        default=None,
        help="If set, restrict train/val/test to this permutation (perm_XXX).",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=100,
        help="Number of epochs with no val improvement before stopping",
    )
    # No extra stabilization flags by default; keep script simple

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

    print(f"[train_param_recovery_ssl] solver={args.solver}")

    # Build datasets
    ds_train = FullSplitDataset(root, split="train", perm_id=args.perm_id)
    ds_val = FullSplitDataset(root, split="val", perm_id=args.perm_id)
    ds_test = FullSplitDataset(root, split="test", perm_id=args.perm_id)

    # Assume consistent sample rate across dataset (enforced by builder)
    fs = int(ds_train[0]["meta"]["fs"])  # type: ignore[index]

    # Model and optim
    model = ParamRecoveryModelSLL(fs=fs)
    model.to(device)

    # Simple optimizer without extra tuning
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # DataLoaders
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    # Ground-truth target summary (fail loudly if mixed) and print for reference
    target_summary = summarize_target_params(train_loader)
    if target_summary.get("num_unique_target_param_sets", 0) != 1:
        raise RuntimeError(
            f"Expected a single target parameter set (use --perm-id). Found: {target_summary.get('num_unique_target_param_sets')}"
        )
    print("Ground truth (from dataset):", target_summary.get("target_params"))

    best_val = float("inf")
    best_state = None
    patience_left = args.early_stop_patience
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses: List[float] = []
        for batch in train_loader:
            x_peak_dB = batch["x_peak_dB"].to(device)
            g_ref_dB = batch["g_ref_dB"].to(device)

            y_dB = model(x_peak_dB)

            loss = rmse_db_unmasked(g_ref_dB, y_dB)
            if not torch.isfinite(loss):
                print("[warn] non-finite loss encountered; skipping step")
                opt.zero_grad()
                continue
            opt.zero_grad()
            loss.backward()
            # Check grads for NaN/Inf and skip step if found
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        bad_grad = True
                        break
            if bad_grad:
                print("[warn] non-finite gradients encountered; skipping step")
                opt.zero_grad()
                continue
            # Gradient clipping to prevent blow-ups from FD gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            # Post-step clamps to keep unconstrained logits in a safe numeric range
            with torch.no_grad():
                model.ratio_logit.clamp_(-6.0, 6.0)
                model.fb_logit.clamp_(-8.0, 8.0)
                model.u_T_af.clamp_(-6.0, 6.0)
                model.u_T_as.clamp_(-6.0, 6.0)
                model.u_T_sf.clamp_(-6.0, 6.0)
                model.u_T_ss.clamp_(-6.0, 6.0)

            epoch_losses.append(loss.item())
            global_step += 1

        # Compute validation loss
        val_loss = evaluate(model, val_loader, device)

        avg_train = sum(epoch_losses) / max(1, len(epoch_losses))
        print(
            f"epoch={epoch} train_rmse_db={avg_train:.4f} val_rmse_db={val_loss:.4f} params={model.params_readable()}"
        )

        # Early stopping
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"Early stopping at epoch {epoch} (best val_rmse_db={best_val:.4f})"
                )
                break

    # Restore best state if available; otherwise keep current params
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test using best params
    test_loss = evaluate(model, test_loader, device)
    print(f"test_rmse_db={test_loss:.4f}")

    # Prepare summary
    # Reuse earlier ground-truth summary for consistency
    # (Recompute in case loader was modified, but typically unchanged)
    target_summary = (
        target_summary
        if "target_summary" in locals()
        else summarize_target_params(train_loader)
    )
    # Fail loudly if we unexpectedly mixed multiple target parameter sets
    if target_summary.get("num_unique_target_param_sets", 0) != 1:
        raise RuntimeError(
            f"Expected a single target parameter set (use --perm-id). Found: {target_summary.get('num_unique_target_param_sets')}"
        )
    best_params = model.params_readable()
    summary = {
        "best_val_rmse_db": float(best_val),
        "test_rmse_db": float(test_loss),
        "target": target_summary,
        "best_params": best_params,
    }

    # Print summary
    print("==== Training summary ====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save params and summary
    out_params = root / "final_params_ssl.yaml"
    with open(out_params, "w") as f:
        yaml.safe_dump(best_params, f)
    out_summary = root / "summary_ssl.yaml"
    with open(out_summary, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=True)
    print(f"Saved final params to {out_params}")
    print(f"Saved summary to {out_summary}")


if __name__ == "__main__":
    main()

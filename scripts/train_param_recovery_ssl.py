#!/usr/bin/env python3
"""
Train/evaluate global SSL parameter recovery (dB-domain) against the hard-gate teacher
using the full train/val/test dataset structure under --data-dir.

Key points:
- Loads dB-domain tensors written by the SSL dataset builder:
  * x_peak_dB.pt (20*log10(|x|) envelope)
  * g_ref_dB.pt  (teacher gain in dB)
- The model consumes x_peak_dB and outputs y_db; loss is RMSE in dB with a dB mask.
- Early stopping on validation loss with patience.
- Imports ParamRecoveryModelSLL.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from differential_dynamics.training.param_recovery_model_sll import (
    ParamRecoveryModelSLL,
)


def _rmse_db_masked(y_ref_db: torch.Tensor, y_hat_db: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """RMSE directly in dB. If mask is empty or None, compute unmasked.
    Expects y_ref_db and y_hat_db to be in dB already.
    """
    diff = y_ref_db - y_hat_db
    if mask is not None:
        if mask.numel() == 0 or torch.count_nonzero(mask) == 0:
            pass
        else:
            diff = diff[mask]
    return torch.sqrt(torch.mean(diff**2))


def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_meta_paths(root: Path, split: str) -> List[Path]:
    """Recursively find meta.yaml files for a given split.

    Supports layouts with or without permutations, e.g.:
      - <root>/<split>/perm_XXX/clip_YYYY/meta.yaml
      - <root>/<split>/clip_YYYY/meta.yaml
    """
    split_dir = root / split
    if not split_dir.is_dir():
        return []
    # Prefer recursive glob to handle both forms
    return sorted(split_dir.rglob("clip_*/meta.yaml"))


class FullSplitDataset(Dataset):
    """Dataset that loads all clips under <root>/<split>/**/clip_*/meta.yaml.

    Returns dict with tensors and meta for compatibility with the prior trainer.
    """

    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        self.split = split
        self.items = find_meta_paths(self.root, split)
        if not self.items:
            raise RuntimeError(f"No items found under {self.root}/{split}")

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
    mask_thresh_db: float = -100.0,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            x_peak_dB = batch["x_peak_dB"].to(device)
            g_ref_dB = batch["g_ref_dB"].to(device)
            mask = x_peak_dB > mask_thresh_db
            y_db = model(x_peak_dB)
            loss = _rmse_db_masked(g_ref_dB, y_db, mask)
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
            "T_attack_fast_ms": float(d.get("attack_time_fast_ms", d.get("T_AF_MS", 0.0))),
            "T_attack_slow_ms": float(d.get("attack_time_slow_ms", d.get("T_AS_MS", 0.0))),
            "T_shunt_fast_ms": float(d.get("release_time_fast_ms", d.get("T_SF_MS", 0.0))),
            "T_shunt_slow_ms": float(d.get("release_time_slow_ms", d.get("T_SS_MS", 0.0))),
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
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Processed dataset root (contains train/ val/ test/)",
    )
    p.add_argument("--device", type=str, default="cpu", help="cpu only (SSL smoother is CPU-only)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=100,
        help="Number of epochs with no val improvement before stopping",
    )
    args = p.parse_args()

    root = Path(args.data_dir)
    device = torch.device(args.device)

    # Build datasets
    ds_train = FullSplitDataset(root, split="train")
    ds_val = FullSplitDataset(root, split="val")
    ds_test = FullSplitDataset(root, split="test")

    # Assume consistent sample rate across dataset (enforced by builder)
    fs = int(ds_train[0]["meta"]["fs"])  # type: ignore[index]

    # Model and optim
    model = ParamRecoveryModelSLL(fs=fs)
    model.to(device)
    # Use a slightly smaller LR for time-constant u-parameters if desired
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            mask = x_peak_dB > -100.0

            y_db = model(x_peak_dB)

            loss = _rmse_db_masked(g_ref_dB, y_db, mask=mask)
            opt.zero_grad()
            loss.backward()
            opt.step()

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
    target_summary = summarize_target_params(train_loader)
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

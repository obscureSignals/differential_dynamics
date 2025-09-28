#!/usr/bin/env python3
"""
Train/evaluate global parameter recovery against a hard-gate teacher using the full
train/val/test dataset structure under --data-dir.

Differences vs the original script:
- No --split argument. Instead, this script automatically loads:
  * train:  <data_dir>/train/**/clip_*/meta.yaml
  * val:    <data_dir>/val/**/clip_*/meta.yaml
  * test:   <data_dir>/test/**/clip_*/meta.yaml
- Early stopping on validation loss with patience.
- Imports the model from differential_dynamics.training.param_recovery_model.

Usage examples:
  python scripts/train_param_recovery.py --data-dir data/processed \
    --ar-mode sigmoid --epochs 100 --k-start 0.5 --k-end 2.0 --k-anneal-steps 2000 \
    --early-stop-patience 10 --batch-size 4 --lr 1e-2

  python scripts/train_param_recovery.py --data-dir data/processed \
    --ar-mode hard --epochs 100 --early-stop-patience 10
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
import yaml
from torch.utils.data import Dataset, DataLoader

from differential_dynamics.benchmarks.bench_utilities import (
    rmse_db,
    active_mask_from_env,
)
from differential_dynamics.training.param_recovery_model import (
    GlobalParamModel,
    linear_anneal,
)


def _safe_rmse_db(
    g_ref: torch.Tensor, g_pred: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """RMSE dB that falls back to unmasked when mask is empty to avoid NaNs."""
    if mask is not None:
        # If mask reduces all elements away, mean() would be NaN; guard it.
        if mask.numel() == 0 or torch.count_nonzero(mask) == 0:
            return rmse_db(g_ref, g_pred, mask=None)
    return rmse_db(g_ref, g_pred, mask=mask)


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
        x, sr = torchaudio.load(str(ex_dir / "x.wav"))
        x_rms = torch.load(ex_dir / "x_rms.pt")
        g_ref = torch.load(ex_dir / "g_ref.pt")
        theta_ref = meta["theta_ref"]
        return {
            "x": x.to(torch.float32),
            "x_rms": x_rms.to(torch.float32),
            "g_ref": g_ref.to(torch.float32),
            "meta": meta,
            "theta_ref": theta_ref,
        }


def collate(batch: List[Dict]) -> Dict:
    keys = ["x", "x_rms", "g_ref"]
    out = {k: torch.cat([b[k] for b in batch], dim=0) for k in keys}
    out["meta"] = [b["meta"] for b in batch]
    out["theta_ref"] = [b["theta_ref"] for b in batch]
    return out


def evaluate(
    model: GlobalParamModel,
    loader: DataLoader,
    device: torch.device,
    ar_mode: str,
    k_val: float | None,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            x_rms = batch["x_rms"].to(device)
            g_ref = batch["g_ref"].to(device)
            mask = active_mask_from_env(x_rms, thresh_db=-100.0)
            if ar_mode == "sigmoid":
                g_pred = model(x_rms, ar_mode="sigmoid", k=k_val)
            else:
                g_pred = model(x_rms, ar_mode="hard")
            loss = _safe_rmse_db(g_ref, g_pred, mask=mask)
            losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def summarize_target_params(loader: DataLoader) -> Dict[str, float | int | list]:
    """Extract target theta from dataset metadata.
    If multiple unique thetas are found, report count and the first as exemplar.
    """
    # Collect a few samples to determine uniqueness
    thetas: List[Dict] = []
    for i, batch in enumerate(loader):
        # theta_ref is a list of dicts (one per item in batch)
        thetas.extend(batch["theta_ref"])  # type: ignore[index]
        if len(thetas) >= 32:  # sample up to 32 items
            break

    # Normalize and dedup by tuple
    def norm(d: Dict) -> Dict[str, float]:
        return {
            "comp_thresh_db": float(
                d.get("comp_thresh_db", d.get("CT", d.get("comp_thresh", 0.0)))
            ),
            "comp_ratio": float(d.get("comp_ratio", d.get("CR", d.get("ratio", 0.0)))),
            "attack_ms": float(
                d.get("attack_ms", d.get("AT_MS", d.get("attack_time_ms", 0.0)))
            ),
            "release_ms": float(
                d.get("release_ms", d.get("RT_MS", d.get("release_time_ms", 0.0)))
            ),
        }

    thetas_n = [norm(t) for t in thetas]
    # Deduplicate
    keys = [tuple(t.items()) for t in thetas_n]
    unique = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(dict(k))
    summary: Dict[str, float | int | list] = {}
    summary["num_unique_target_param_sets"] = len(unique)
    if unique:
        summary["target_params"] = unique[0]
    else:
        summary["target_params"] = {
            "comp_thresh_db": None,
            "comp_ratio": None,
            "attack_ms": None,
            "release_ms": None,
        }
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
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=100,
        help="Number of epochs with no val improvement before stopping",
    )
    p.add_argument(
        "--ar-mode",
        type=str,
        default="sigmoid",
        choices=["hard", "sigmoid"],
        help="Student smoother type",
    )
    p.add_argument(
        "--k-start",
        type=float,
        default=0.1,
        help="Initial k (sigmoid sharpness), only used in sigmoid mode",
    )
    p.add_argument(
        "--k-end",
        type=float,
        default=50.0,
        help="Final k for linear annealing, only used in sigmoid mode",
    )
    p.add_argument(
        "--k-anneal-steps",
        type=int,
        default=50.0,
        help="Steps over which to anneal k from start to end",
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
    model = GlobalParamModel(fs=fs)
    model.to(device)
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
    best_val_k = None  # k used during best val (sigmoid only)
    patience_left = args.early_stop_patience
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses: List[float] = []
        for batch in train_loader:
            x_rms = batch["x_rms"].to(device)
            g_ref = batch["g_ref"].to(device)
            mask = active_mask_from_env(x_rms, thresh_db=-100.0)

            if args.ar_mode == "sigmoid":
                k_val = linear_anneal(
                    args.k_start, args.k_end, global_step, args.k_anneal_steps
                )
                g_pred = model(x_rms, ar_mode="sigmoid", k=k_val)
            else:
                g_pred = model(x_rms, ar_mode="hard")

            loss = _safe_rmse_db(g_ref, g_pred, mask=mask)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())
            global_step += 1

        # Compute validation loss (use mid-anneal k for sigmoid to stabilize eval)
        if args.ar_mode == "sigmoid":
            mid_step = (args.k_anneal_steps // 2) if args.k_anneal_steps > 0 else 0
            k_eval = linear_anneal(
                args.k_start,
                args.k_end,
                mid_step,
                args.k_anneal_steps,
            )
        else:
            k_eval = None
        val_loss = evaluate(model, val_loader, device, args.ar_mode, k_eval)

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
            best_val_k = float(k_eval) if args.ar_mode == "sigmoid" else None
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
    # Ensure k_at_best_val is defined for sigmoid when val never improved
    if args.ar_mode == "sigmoid" and best_val_k is None:
        mid_step = (args.k_anneal_steps // 2) if args.k_anneal_steps > 0 else 0
        best_val_k = linear_anneal(
            args.k_start, args.k_end, mid_step, args.k_anneal_steps
        )

    # Evaluate on test using best params
    if args.ar_mode == "sigmoid":
        # Test with k at best val (mid-anneal), final k, and hard gate
        test_loss_best = evaluate(model, test_loader, device, "sigmoid", best_val_k)
        test_loss_final_k = evaluate(model, test_loader, device, "sigmoid", args.k_end)
        test_loss_hard = evaluate(model, test_loader, device, "hard", None)
        k_best_display = float(best_val_k) if best_val_k is not None else None
        # For compatibility with previous print, set test_loss to final_k
        test_loss = test_loss_final_k
    else:
        test_loss_best = evaluate(model, test_loader, device, "hard", None)
        test_loss_final_k = None
        test_loss_hard = test_loss_best
        k_best_display = None
        test_loss = test_loss_best

    print(f"test_rmse_db={test_loss:.4f}")

    # Prepare summary
    target_summary = summarize_target_params(train_loader)
    best_params = model.params_readable()
    summary = {
        "ar_mode": args.ar_mode,
        "best_val_rmse_db": float(best_val),
        "k_at_best_val": k_best_display,
        "test_rmse_db_best_params": float(test_loss_best),
        "test_rmse_db_best_params_final_k": (
            float(test_loss_final_k) if test_loss_final_k is not None else None
        ),
        "test_rmse_db_best_params_hard": float(test_loss_hard),
        "test_rmse_db": float(test_loss),
        "target": target_summary,
        "best_params": best_params,
    }

    # Print summary
    print("==== Training summary ====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save params and summary
    out_params = root / f"final_params_{args.ar_mode}.yaml"
    with open(out_params, "w") as f:
        yaml.safe_dump(best_params, f)
    out_summary = root / f"summary_{args.ar_mode}.yaml"
    with open(out_summary, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=True)
    print(f"Saved final params to {out_params}")
    print(f"Saved summary to {out_summary}")


if __name__ == "__main__":
    main()

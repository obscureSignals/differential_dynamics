#!/usr/bin/env python3
"""
Training scaffold: parameter recovery against a hard-gate teacher dataset.

- Loads offline-processed examples (x, x_rms, g_ref_hard, meta) from scripts/make_dataset.py
- Optimizes global compressor parameters θ = {CT, CR, τ_a, τ_r} shared across a small dataset
- Supports two student modes for the smoother:
  - hard   : torchcomp.compexp_gain (baseline, custom autograd; coefficients directly)
  - sigmoid: compexp_gain_mode(..., ar_mode="sigmoid", k=...) (differentiable gate in dB; at_ms/rt_ms path)
- Primary objective: gain-trace RMSE in dB vs g_ref_hard on active regions
- Optionally anneals k (sigmoid sharpness) from a low value to a higher value during training

Notes:
- Expansion is disabled (compression-only) consistently.
- Detector is fixed (x_rms provided by the dataset) for Phase 1 to avoid confounding.
- This is a minimal scaffold, not a production trainer: no checkpointing, early stop, etc.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import yaml

from differential_dynamics.benchmarks.bench_utilities import (
    rmse_db,
    active_mask_from_env,
)
from differential_dynamics.backends.torch.gain import compexp_gain_mode
from third_party.torchcomp_core.torchcomp import ms2coef, compexp_gain, coef2ms


def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class HardTeacherDataset(Dataset):
    """Dataset that reads artifacts written by scripts/make_dataset.py.

    Expects directory structure: <root>/<split>/clip_XXXX/{x.wav, x_rms.pt, g_ref.pt, meta.yaml}
    Where <split> can include a permutation, e.g., "train/perm_001".
    """

    def __init__(self, root: Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        # meta.yaml fingerprints each clip and includes theta_ref and detector_ms
        self.items = sorted((self.root / split).glob("clip_*/meta.yaml"))
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
    """Simple collate that concatenates along the batch dim (shape (B, T)).

    Assumes all clips in a batch share the same T. This holds if clips are
    created at fixed duration by the dataset builder.
    """
    keys = ["x", "x_rms", "g_ref"]
    # Concatenate tensors along batch dimension
    out = {k: torch.cat([b[k] for b in batch], dim=0) for k in keys}
    # Keep meta and theta_ref as lists for optional analysis/reporting
    out["meta"] = [b["meta"] for b in batch]
    out["theta_ref"] = [b["theta_ref"] for b in batch]
    return out


class GlobalParamModel(nn.Module):
    """Global, shared learnable parameters θ for a small dataset.

    Parameterization for stability:
      - comp_ratio = exp(ratio_logit) + 1  (enforces > 1)
      - at/rt coefficients in (0,1) via sigmoid(logit)
      - comp_thresh is a free parameter in dB
    """

    def __init__(self, fs: int, init: Dict[str, float] | None = None):
        super().__init__()
        init = init or {"CT": -24.0, "CR": 4.0, "AT_MS": 10.0, "RT_MS": 100.0}
        # Threshold in dB
        self.comp_thresh = nn.Parameter(torch.tensor(float(init["CT"])))
        # Ratio as exp(logit)+1 to enforce >1
        self.ratio_logit = nn.Parameter(
            torch.log(torch.tensor(float(init["CR"])) - 1.0)
        )

        # Attack/Release time constants in (0,1) via sigmoid(logit)
        def ms2coef_scalar(ms: float) -> float:
            return ms2coef(torch.tensor(ms, dtype=torch.float32), fs).item()

        self.at_logit = nn.Parameter(
            torch.logit(torch.tensor(ms2coef_scalar(float(init["AT_MS"]))))
        )
        self.rt_logit = nn.Parameter(
            torch.logit(torch.tensor(ms2coef_scalar(float(init["RT_MS"]))))
        )
        self.fs = fs

    def params_readable(self) -> Dict[str, float]:
        """Return current parameters in human units for logging/reporting."""
        ratio = (self.ratio_logit.exp() + 1.0).item()
        at_ms = coef2ms(torch.sigmoid(self.at_logit), self.fs).item()
        rt_ms = coef2ms(torch.sigmoid(self.rt_logit), self.fs).item()
        return {
            "comp_thresh_db": self.comp_thresh.item(),
            "comp_ratio": ratio,
            "attack_ms": at_ms,
            "release_ms": rt_ms,
        }

    def forward(
        self, x_rms: torch.Tensor, ar_mode: str, k: float | None = None
    ) -> torch.Tensor:
        """Predict gain using either the hard or sigmoid smoother.

        - Hard mode consumes A/R coefficients directly (as torchcomp expects).
        - Sigmoid mode consumes A/R times in ms (as compexp_gain_mode expects) and takes k.
        """
        ct = self.comp_thresh
        cr = self.ratio_logit.exp() + 1.0
        at_coef = torch.sigmoid(self.at_logit)
        rt_coef = torch.sigmoid(self.rt_logit)

        if ar_mode == "hard":
            # Use torchcomp baseline directly for the smoother (preserves custom backward)
            return compexp_gain(
                x_rms=x_rms,
                comp_thresh=ct,
                comp_ratio=cr,
                exp_thresh=-1000.0,
                exp_ratio=1.0,
                at=at_coef,
                rt=rt_coef,
            )

        if ar_mode == "sigmoid":
            if k is None:
                raise ValueError("k must be provided for sigmoid mode")
            # Convert coefficients to ms using the vendored torchcomp helper
            at_ms_val = coef2ms(at_coef, self.fs).item()
            rt_ms_val = coef2ms(rt_coef, self.fs).item()
            return compexp_gain_mode(
                x_rms=x_rms,
                comp_thresh=ct,
                comp_ratio=cr,
                exp_thresh=-1000.0,
                exp_ratio=1.0,
                at_ms=at_ms_val,
                rt_ms=rt_ms_val,
                fs=self.fs,
                ar_mode="sigmoid",
                k=float(k),
                smoother_backend="torchscript",
            )

        raise ValueError(f"Unknown ar_mode: {ar_mode}")


def linear_anneal(start: float, end: float, step: int, total_steps: int) -> float:
    """Linear schedule from start to end over total_steps.

    Returns end if total_steps <= 0. Clamps step to [0, total_steps].
    """
    if total_steps <= 0:
        return end
    t = min(max(step, 0), total_steps)
    return float(start + (end - start) * (t / total_steps))


def main():
    """Entry point: train global θ against a hard teacher for hard/sigmoid students."""
    p = argparse.ArgumentParser(description="Parameter recovery training (hard vs sigmoid)")
    p.add_argument("--data-dir", type=str, required=True, help="Processed dataset root (from make_dataset.py)")
    p.add_argument("--split", type=str, default="train", help="Split to train on (train/val/test)")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--ar-mode", type=str, default="sigmoid", choices=["hard", "sigmoid"], help="Student smoother type") 
    p.add_argument("--k-start", type=float, default=0.5, help="Initial k (sigmoid sharpness), only used in sigmoid mode")
    p.add_argument("--k-end", type=float, default=2.0, help="Final k for linear annealing, only used in sigmoid mode")
    p.add_argument("--k-anneal-steps", type=int, default=1000, help="Steps over which to anneal k from start to end")
    args = p.parse_args()

    ds = HardTeacherDataset(Path(args.data_dir), split=args.split)
    # Assume consistent sample rate across dataset (enforced during dataset build)
    fs = int(ds[0]["meta"]["fs"])  # type: ignore[index]
    model = GlobalParamModel(fs=fs)
    device = torch.device(args.device)
    model.to(device)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_losses: List[float] = []
        for batch in loader:
            # Inputs: x_rms (detector envelope) and teacher gain g_ref
            x_rms = batch["x_rms"].to(device)
            g_ref = batch["g_ref"].to(device)
            # Mask out deep silence to focus on active regions when computing RMSE_dB
            mask = active_mask_from_env(x_rms, thresh_db=-100.0)

            # Simulate with selected smoother
            if args.ar_mode == "sigmoid":
                # Anneal k to avoid early gradient sparsity; increase sharpness over time
                k_val = linear_anneal(args.k_start, args.k_end, global_step, args.k_anneal_steps)
                g_pred = model(x_rms, ar_mode="sigmoid", k=k_val)
            else:
                g_pred = model(x_rms, ar_mode="hard")

            # Primary objective: gain-trace RMSE in dB vs hard teacher
            loss = rmse_db(g_ref, g_pred, mask=mask)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())
            global_step += 1

        # Log simple summary (parameters in human units)
        readable = model.params_readable()
        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(
            f"epoch={epoch} avg_loss={avg_loss:.4f} CT={readable['comp_thresh_db']:.2f}dB CR={readable['comp_ratio']:.2f} "
            f"AT={readable['attack_ms']:.1f}ms RT={readable['release_ms']:.1f}ms"
        )

    # Save final params for inspection
    out_path = Path(args.data_dir) / f"final_params_{args.ar_mode}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(model.params_readable(), f)
    print(f"Saved final params to {out_path}")


if __name__ == "__main__":
    main()

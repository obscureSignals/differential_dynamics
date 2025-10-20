#!/usr/bin/env python3
"""
Calibrate analytic saltation gradients vs FD for SSL smoother time constants.

Runs three CT-agnostic probe scenarios and compares per-parameter gradients
{T_af, T_as, T_sf, T_ss} between:
  - analytic + saltation (operator-Jacobian contraction with per-flip jumps)
  - FD time-constant gradients (scalar-loss)

Usage:
  # Orchestrated comparison (spawns subprocesses to ensure env takes effect)
  python scripts/calibrate_saltation.py

  # Single mode run (debug)
  SSL_USE_SALTATION=1 python scripts/calibrate_saltation.py --single-run --mode analytic
  SSL_USE_FD_TCONST_GRADS=1 python scripts/calibrate_saltation.py --single-run --mode fd
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

# Local imports delayed until after env configuration in single-run mode


@dataclass
class Theta:
    comp_thresh_db: float = -24.0
    comp_ratio: float = 4.0
    T_af_s: float = 0.005    # 5 ms
    T_as_s: float = 0.300    # 300 ms (slow attack, rarely used in SSL attacks)
    T_sf_s: float = 0.200    # 200 ms
    T_ss_s: float = 1.000    # 1000 ms
    feedback_coeff: float = 0.999
    k: float = 0.0           # hard gate path


def _set_env_for_mode(mode: str) -> None:
    # Always CPU float32
    os.environ.setdefault("SSL_DEVICE", "cpu")
    # Avoid ambiguity: clear toggles
    for k in [
        "SSL_USE_FD_TCONST_GRADS",
        "SSL_USE_ANALYTIC_JAC",
        "SSL_USE_ANALYTIC_JAC_AD",
        "SSL_USE_ANALYTIC_JAC_BD",
        "SSL_ANALYTIC_BD_METHOD",
        "SSL_USE_SALTATION",
        "SSL_SALTATION_MAX_FLIPS",
        "SSL_SALTATION_EPS_REL",
        "SSL_SALTATION_MAX_BACKOFF",
        "SSL_DEBUG_SALTATION",
    ]:
        os.environ.pop(k, None)

    if mode == "fd":
        os.environ["SSL_USE_FD_TCONST_GRADS"] = "1"
    elif mode == "analytic":
        os.environ["SSL_USE_FD_TCONST_GRADS"] = "0"
        os.environ["SSL_USE_ANALYTIC_JAC"] = "1"
        os.environ["SSL_ANALYTIC_BD_METHOD"] = "phi"
        os.environ["SSL_USE_SALTATION"] = "1"
        os.environ.setdefault("SSL_SALTATION_MAX_FLIPS", "4096")
        os.environ.setdefault("SSL_SALTATION_EPS_REL", "1e-3")
        os.environ.setdefault("SSL_SALTATION_MAX_BACKOFF", "8")
        # Optional debug line throttling inside the kernel
        os.environ.setdefault("SSL_DEBUG_SALTATION", "0")
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _build_probes(fs: int, T: int) -> Dict[str, torch.Tensor]:
    # Import here to avoid importing gain/signals before env is set in single-run
    from differential_dynamics.benchmarks.signals import (
        step as step_sig,
        probe_p8_attack_doublet,
        probe_p2b_release_drop_near_threshold,
    )

    probes: Dict[str, torch.Tensor] = {}
    # Single up-step at mid-clip
    probes["single_step_up"] = step_sig(fs=fs, T=T, B=1, at=0.5, amp_before=0.08, amp_after=0.80)
    # Attack doublet
    probes["attack_doublet"] = probe_p8_attack_doublet(fs=fs, T=T, B=1)
    # Drop with gap (history vs no-history)
    probes["drop_with_gap"] = probe_p2b_release_drop_near_threshold(fs=fs, T=T, B=1)
    return probes


def _compute_grads_single_mode(mode: str, fs: int, clip_sec: float, theta: Theta) -> Dict[str, Dict[str, float]]:
    _set_env_for_mode(mode)
    # Import after env is fixed so the extension picks it up
    from differential_dynamics.backends.torch.gain import SSL_comp_gain
    from differential_dynamics.benchmarks.bench_utilities import gain_db

    T = int(round(clip_sec * fs))
    probes = _build_probes(fs, T)

    results: Dict[str, Dict[str, float]] = {}
    for name, x_amp in probes.items():
        x_peak_db = gain_db(x_amp.abs().to(torch.float32))
        # Parameters (seconds) with gradients
        T_af = torch.tensor(theta.T_af_s, dtype=torch.float32, requires_grad=True)
        T_as = torch.tensor(theta.T_as_s, dtype=torch.float32, requires_grad=True)
        T_sf = torch.tensor(theta.T_sf_s, dtype=torch.float32, requires_grad=True)
        T_ss = torch.tensor(theta.T_ss_s, dtype=torch.float32, requires_grad=True)

        y_db = SSL_comp_gain(
            x_peak_dB=x_peak_db,
            comp_thresh=theta.comp_thresh_db,
            comp_ratio=theta.comp_ratio,
            T_attack_fast=T_af,
            T_attack_slow=T_as,
            T_shunt_fast=T_sf,
            T_shunt_slow=T_ss,
            feedback_coeff=theta.feedback_coeff,
            k=theta.k,
            fs=fs,
            soft_gate=False,
        )
        # Scalar loss: 0.5 * mean(y^2) to provide stable gradients
        L = 0.5 * torch.mean(y_db * y_db)
        L.backward()
        results[name] = {
            "dL_dT_af": float(T_af.grad.item()),
            "dL_dT_as": float(T_as.grad.item()),
            "dL_dT_sf": float(T_sf.grad.item()),
            "dL_dT_ss": float(T_ss.grad.item()),
            "L": float(L.item()),
        }
    return results


def _orchestrate(fs: int, clip_sec: float, theta: Theta, tol_rel: float, tol_abs: float) -> int:
    # Run in subprocesses to ensure env-at-import semantics
    def _parse_last_json_line(stdout: str) -> Dict[str, Dict[str, float]]:
        lines = stdout.strip().splitlines()
        for line in reversed(lines):
            s = line.strip()
            if not s:
                continue
            try:
                return json.loads(s)
            except Exception:
                continue
        raise ValueError("No JSON payload found in child stdout")

    def run_mode_sub(mode: str) -> Dict[str, Dict[str, float]]:
        env = os.environ.copy()
        cmd = [sys.executable, __file__, "--single-run", "--mode", mode, "--fs", str(fs), "--clip-sec", str(clip_sec)]
        env["SSL_CALIB_THETA_JSON"] = json.dumps(theta.__dict__)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            tail_out = "\n".join(p.stdout.splitlines()[-10:])
            tail_err = "\n".join(p.stderr.splitlines()[-10:])
            raise RuntimeError(f"Child {mode} failed rc={p.returncode}\nstdout tail:\n{tail_out}\nstderr tail:\n{tail_err}")
        try:
            return _parse_last_json_line(p.stdout)
        except Exception as e:
            tail_out = "\n".join(p.stdout.splitlines()[-10:])
            tail_err = "\n".join(p.stderr.splitlines()[-10:])
            raise RuntimeError(f"Child {mode} produced non-JSON output. {e}\nstdout tail:\n{tail_out}\nstderr tail:\n{tail_err}")

    res_analytic = run_mode_sub("analytic")
    res_fd = run_mode_sub("fd")

    # Compare
    all_ok = True
    print("=== Saltation calibration (analytic vs FD) ===")
    for probe in sorted(res_analytic.keys()):
        ra = res_analytic[probe]; rf = res_fd[probe]
        print(f"Probe: {probe}")
        for k in ["dL_dT_af", "dL_dT_as", "dL_dT_sf", "dL_dT_ss"]:
            a = float(ra[k]); f = float(rf[k]);
            abs_err = abs(a - f)
            rel_err = abs_err / max(1e-12, abs(f))
            ok = (abs_err <= tol_abs) or (rel_err <= tol_rel)
            status = "OK" if ok else "FAIL"
            print(f"  {k}: analytic={a:.6e} fd={f:.6e} abs={abs_err:.3e} rel={rel_err*100:.2f}% {status}")
            all_ok = all_ok and ok
    return 0 if all_ok else 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--single-run", action="store_true", help="Run a single mode and emit JSON of grads")
    p.add_argument("--mode", type=str, choices=["analytic", "fd"], default="analytic")
    p.add_argument("--fs", type=int, default=44100)
    p.add_argument("--clip-sec", type=float, default=2.0)
    p.add_argument("--tol-rel", type=float, default=0.02, help="Relative tolerance for grad match")
    p.add_argument("--tol-abs", type=float, default=1e-6, help="Absolute tolerance for tiny grads")
    args = p.parse_args()

    # Theta from env or defaults
    th_env = os.environ.get("SSL_CALIB_THETA_JSON")
    if th_env:
        d = json.loads(th_env)
        theta = Theta(**d)
    else:
        theta = Theta()

    if args.single_run:
        # Single mode JSON output
        out = _compute_grads_single_mode(args.mode, args.fs, args.clip_sec, theta)
        print(json.dumps(out, sort_keys=True))
        return

    # Orchestrate both modes and compare
    rc = _orchestrate(args.fs, args.clip_sec, theta, args.tol_rel, args.tol_abs)
    sys.exit(rc)


if __name__ == "__main__":
    main()

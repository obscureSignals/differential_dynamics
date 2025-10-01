# SSL 2-state smoother C++ extension wrapper
#
# Exposes a PyTorch autograd.Function with:
# - forward: calls into the C++ CPU kernel (y_db in dB)
# - backward: implemented on CPU for hard gate (soft_gate=False). Soft gate will raise.
#
# Build modes (controlled by env):
# - SSL_SMOOTHER_DEBUG=1 -> build with -g -O0 -fno-omit-frame-pointer, verbose=True, separate cache dir
# - SSL_SMOOTHER_FORCE_REBUILD=1 -> remove cached build and force a full rebuild
# - default (unset)      -> build with -O3 and native arch on macOS
#
# The C++ extension is built lazily via torch.utils.cpp_extension.load from
# csrc/ssl_smoother.cpp. CPU-only, float32-only by design.

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Tuple

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load


def _truthy_env(v: str | None) -> bool:
    if v is None:
        return False
    v = v.strip().lower()
    return v in {"1", "true", "yes", "on", "y", "t"}


def _load_ext() -> any:
    here = Path(__file__).resolve().parent.parent.parent.parent
    src = here / "csrc" / "ssl_smoother.cpp"
    debug = _truthy_env(os.environ.get("SSL_SMOOTHER_DEBUG"))
    force_rebuild = _truthy_env(os.environ.get("SSL_SMOOTHER_FORCE_REBUILD"))
    name = "ssl_smoother_cpu_dbg" if debug else "ssl_smoother_cpu"
    # Build in a cache dir under the project to avoid conflicts
    build_dir_path = here / "build" / name
    build_dir = str(build_dir_path)

    if force_rebuild and build_dir_path.exists():
        print(f"[ssl_smoother_ext] Forcing rebuild: removing {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)

    # Ensure build directory exists to avoid FileNotFoundError on lock file
    os.makedirs(build_dir, exist_ok=True)

    if debug:
        # Debug-friendly flags for stepping in lldb
        extra_cflags = [
            "-g",
            "-O0",
            "-fno-omit-frame-pointer",
            "-std=c++17",
            "-DSSL_SMOOTHER_DEBUG",
        ]
        verbose = True
    else:
        # Optimized build for normal usage
        extra_cflags = ["-O3", "-std=c++17"]
        if os.uname().sysname == "Darwin":
            extra_cflags += ["-march=native"]
        verbose = False

    print(f"[ssl_smoother_ext] debug={debug} name={name} build_dir={build_dir}")

    ext = load(
        name=name,
        sources=[str(src)],
        build_directory=build_dir,
        extra_cflags=extra_cflags,
        verbose=verbose,
    )
    print(f"[ssl_smoother_ext] loaded: {ext.__file__}")
    return ext


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = _load_ext()
    return _ext


class SSL2StateSmootherFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x_peak_dB: torch.Tensor,  # (B,T) dB
        T_af: torch.Tensor,  # (B,)
        T_as: torch.Tensor,  # (B,)
        T_sf: torch.Tensor,  # (B,)
        T_ss: torch.Tensor,  # (B,)
        comp_slope: torch.Tensor,  # (B,),
        comp_thresh: torch.Tensor,  # (B,),
        feedback_coeff: torch.Tensor,  # (B,),
        k: torch.Tensor,  # (B,),
        fs: float,
        soft_gate: bool,
    ) -> torch.Tensor:
        if x_peak_dB.device.type != "cpu":
            raise RuntimeError("SSL2StateSmootherFunction: CPU tensors required")
        for t in (T_af, T_as, T_sf, T_ss, comp_slope, comp_thresh, feedback_coeff):
            if t.device.type != "cpu":
                raise RuntimeError(
                    "SSL2StateSmootherFunction: CPU tensors required for all inputs except fs"
                )
        if x_peak_dB.dtype != torch.float32:
            x_peak_dB = x_peak_dB.float()

        T_af = T_af.contiguous().float()
        T_as = T_as.contiguous().float()
        T_sf = T_sf.contiguous().float()
        T_ss = T_ss.contiguous().float()
        comp_slope = comp_slope.contiguous().float()
        comp_thresh = comp_thresh.contiguous().float()
        feedback_coeff = feedback_coeff.contiguous().float()
        k = k.contiguous().float()

        ext = _get_ext()
        y_db = ext.forward(
            x_peak_dB.contiguous(),
            T_af,
            T_as,
            T_sf,
            T_ss,
            comp_slope,
            comp_thresh,
            feedback_coeff,
            k,
            float(fs),
            soft_gate,
        )
        # Save tensors for backward implementation later
        ctx.save_for_backward(
            x_peak_dB,
            T_af,
            T_as,
            T_sf,
            T_ss,
            comp_slope,
            comp_thresh,
            feedback_coeff,
            k,
        )
        ctx.fs = float(fs)
        ctx.soft_gate = bool(soft_gate)
        return y_db

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x_peak_dB, T_af, T_as, T_sf, T_ss, comp_slope, comp_thresh, feedback_coeff, k = ctx.saved_tensors
        fs = ctx.fs
        soft_gate = ctx.soft_gate
        if grad_out.device.type != "cpu":
            grad_out = grad_out.cpu()
        ext = _get_ext()
        grads = ext.backward(
            grad_out.contiguous().float(),
            x_peak_dB.contiguous().float(),
            T_af.contiguous().float(),
            T_as.contiguous().float(),
            T_sf.contiguous().float(),
            T_ss.contiguous().float(),
            comp_slope.contiguous().float(),
            comp_thresh.contiguous().float(),
            feedback_coeff.contiguous().float(),
            k.contiguous().float(),
            float(fs),
            bool(soft_gate),
        )
        # grads order: gx, gT_af, gT_as, gT_sf, gT_ss, g_slope, g_thresh, g_fb, g_k
        gx, gT_af, gT_as, gT_sf, gT_ss, g_slope, g_thresh, g_fb, g_k = grads
        # Return grads for each tensor input in forward, and None for fs, soft_gate
        return (
            gx,      # x_peak_dB
            gT_af,   # T_af
            gT_as,   # T_as
            gT_sf,   # T_sf
            gT_ss,   # T_ss
            g_slope, # comp_slope
            g_thresh,# comp_thresh
            g_fb,    # feedback_coeff
            g_k,     # k
            None,    # fs (float)
            None,    # soft_gate (bool)
        )


def ssl2_smoother(
    x_peak_dB: torch.Tensor,
    T_af: torch.Tensor,
    T_as: torch.Tensor,
    T_sf: torch.Tensor,
    T_ss: torch.Tensor,
    comp_slope: torch.Tensor,
    comp_thresh: torch.Tensor,
    feedback_coeff: torch.Tensor,
    k: torch.Tensor,
    fs: float,
    soft_gate: bool,
) -> torch.Tensor:
    """Convenience wrapper for SSL 2-state smoother on CPU.

    Returns y_db (in dB). Backward/gradients are implemented for hard gate
    (soft_gate=False). Passing soft_gate=True will raise at backward.
    """
    return SSL2StateSmootherFunction.apply(
        x_peak_dB,
        T_af,
        T_as,
        T_sf,
        T_ss,
        comp_slope,
        comp_thresh,
        feedback_coeff,
        k,
        fs,
        soft_gate,
    )

# SSL 2-state smoother C++ extension wrapper
#
# Exposes a PyTorch autograd.Function with:
# - forward: calls into the C++ CPU kernel (y_db in dB)
# - backward: fails loudly for now (to be implemented next)
#
# The C++ extension is built lazily via torch.utils.cpp_extension.load from
# csrc/ssl_smoother.cpp. CPU-only, float32-only by design.

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load


def _load_ext() -> any:
    here = Path(__file__).resolve().parent.parent.parent.parent
    src = here / "csrc" / "ssl_smoother.cpp"
    name = "ssl_smoother_cpu"
    # Build in a cache dir under the project to avoid conflicts
    build_dir = str(here / "build" / name)
    # Ensure build directory exists to avoid FileNotFoundError on lock file
    os.makedirs(build_dir, exist_ok=True)
    # Extra cxx flags: O3 and native arch for speed (adjust as needed)
    extra_cflags = ["-O3", "-std=c++17"]
    if os.uname().sysname == "Darwin":
        extra_cflags += ["-march=native"]
    ext = load(
        name=name,
        sources=[str(src)],
        build_directory=build_dir,
        extra_cflags=extra_cflags,
        verbose=False,
    )
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
        g_raw_db: torch.Tensor,  # (B,T) dB
        T_af: torch.Tensor,  # (B,)
        T_as: torch.Tensor,  # (B,)
        T_sf: torch.Tensor,  # (B,)
        T_ss: torch.Tensor,  # (B,)
        k: float,
        fs: float,
    ) -> torch.Tensor:
        if g_raw_db.device.type != "cpu":
            raise RuntimeError("SSL2StateSmootherFunction: CPU tensors required")
        for t in (T_af, T_as, T_sf, T_ss):
            if t.device.type != "cpu":
                raise RuntimeError(
                    "SSL2StateSmootherFunction: CPU tensors required for time constants"
                )
        if g_raw_db.dtype != torch.float32:
            g_raw_db = g_raw_db.float()
        T_af = T_af.contiguous().float()
        T_as = T_as.contiguous().float()
        T_sf = T_sf.contiguous().float()
        T_ss = T_ss.contiguous().float()

        ext = _get_ext()
        y_db = ext.forward(
            g_raw_db.contiguous(), T_af, T_as, T_sf, T_ss, float(k), float(fs)
        )
        # Save tensors for backward implementation later
        ctx.save_for_backward(g_raw_db, y_db, T_af, T_as, T_sf, T_ss)
        ctx.k = float(k)
        ctx.fs = float(fs)
        return y_db

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Explicitly fail loudly to match project policy.
        raise RuntimeError("SSL2StateSmootherFunction.backward not implemented yet")


def ssl2_smoother(
    g_raw_db: torch.Tensor,
    T_af: torch.Tensor,
    T_as: torch.Tensor,
    T_sf: torch.Tensor,
    T_ss: torch.Tensor,
    k: float,
    fs: float,
) -> torch.Tensor:
    """Convenience wrapper: forward-only SSL 2-state smoother on CPU.

    Returns y_db (in dB). Gradients will error until backward is implemented.
    """
    return SSL2StateSmootherFunction.apply(g_raw_db, T_af, T_as, T_sf, T_ss, k, fs)

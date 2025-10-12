import math
import os
import torch
import pytest

from differential_dynamics.backends.torch.ssl_smoother_ext import ssl2_smoother


def fd_grad(fn, x, eps=1e-4):
    with torch.no_grad():
        x_p = x.clone()
        x_m = x.clone()
        x_p += eps
        x_m -= eps
        Lp = fn(x_p)
        Lm = fn(x_m)
        return (Lp - Lm) / (2.0 * eps)


@pytest.mark.parametrize("B,T", [(1, 64)])
def test_hard_gate_backward_small(B, T):
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    fs = 44100.0
    soft_gate = False

    # Signals: x_peak_dB in [-48, 0] dB to avoid deep clamps only
    x_peak_dB = (
        -48.0 + 48.0 * torch.rand(B, T, device=device, dtype=dtype)
    ).contiguous()

    # Time constants (seconds), reasonably separated
    T_af = torch.full(
        (B,), 0.010, device=device, dtype=dtype, requires_grad=True
    )  # 10 ms
    T_as = torch.full(
        (B,), 0.050, device=device, dtype=dtype, requires_grad=True
    )  # 50 ms
    T_sf = torch.full(
        (B,), 0.030, device=device, dtype=dtype, requires_grad=True
    )  # 30 ms
    T_ss = torch.full(
        (B,), 0.200, device=device, dtype=dtype, requires_grad=True
    )  # 200 ms

    # Static curve
    comp_thresh = torch.full(
        (B,), -12.0, device=device, dtype=dtype, requires_grad=True
    )
    comp_ratio = torch.full((B,), 4.0, device=device, dtype=dtype)
    comp_slope = (1.0 - 1.0 / comp_ratio).detach().clone().requires_grad_(True)

    # Feedback coeff
    fb = torch.full((B,), 0.5, device=device, dtype=dtype, requires_grad=True)

    # Gate sharpness (unused in hard mode)
    k = torch.zeros(B, device=device, dtype=dtype)

    # Autograd run
    y_db = ssl2_smoother(
        x_peak_dB=x_peak_dB,
        T_af=T_af,
        T_as=T_as,
        T_sf=T_sf,
        T_ss=T_ss,
        comp_slope=comp_slope,
        comp_thresh=comp_thresh,
        feedback_coeff=fb,
        k=k,
        fs=fs,
        soft_gate=soft_gate,
    )
    L = y_db.sum()
    L.backward()

    # Finite differences helpers
    def run_loss(
        _T_af=None,
        _T_as=None,
        _T_sf=None,
        _T_ss=None,
        _comp_slope=None,
        _comp_thresh=None,
        _fb=None,
    ):
        y = ssl2_smoother(
            x_peak_dB=x_peak_dB,
            T_af=T_af if _T_af is None else _T_af,
            T_as=T_as if _T_as is None else _T_as,
            T_sf=T_sf if _T_sf is None else _T_sf,
            T_ss=T_ss if _T_ss is None else _T_ss,
            comp_slope=comp_slope if _comp_slope is None else _comp_slope,
            comp_thresh=comp_thresh if _comp_thresh is None else _comp_thresh,
            feedback_coeff=fb if _fb is None else _fb,
            k=k,
            fs=fs,
            soft_gate=soft_gate,
        )
        return float(y.sum().item())

    # Tolerances (float32, O(T) sums, hard clamp subgradient): be modest
    atol = 5e-3
    rtol = 5e-2
    # Time-constant gradients are noisier (fixed-mask FD vs analytic): allow looser rtol
    rtol_tconst = 2e-1

    # Check comp_thresh grad via FD on b=0
    eps = 1e-3
    comp_thresh_p = comp_thresh.detach().clone()
    comp_thresh_p[0] += eps
    comp_thresh_m = comp_thresh.detach().clone()
    comp_thresh_m[0] -= eps
    Lp = run_loss(_comp_thresh=comp_thresh_p)
    Lm = run_loss(_comp_thresh=comp_thresh_m)
    gnum = (Lp - Lm) / (2 * eps)
    gaut = float(comp_thresh.grad[0].item())
    assert math.isfinite(gaut)
    assert abs(gnum - gaut) <= atol + rtol * max(
        1.0, abs(gnum)
    ), f"comp_thresh grad mismatch: num={gnum} aut={gaut}"

    # Check comp_slope grad
    comp_slope_p = comp_slope.detach().clone()
    comp_slope_p[0] += eps
    comp_slope_m = comp_slope.detach().clone()
    comp_slope_m[0] -= eps
    Lp = run_loss(_comp_slope=comp_slope_p)
    Lm = run_loss(_comp_slope=comp_slope_m)
    gnum = (Lp - Lm) / (2 * eps)
    gaut = float(comp_slope.grad[0].item())
    assert math.isfinite(gaut)
    assert abs(gnum - gaut) <= atol + rtol * max(
        1.0, abs(gnum)
    ), f"comp_slope grad mismatch: num={gnum} aut={gaut}"

    # Check feedback coeff grad
    fb_p = fb.detach().clone()
    fb_p[0] += eps
    fb_m = fb.detach().clone()
    fb_m[0] -= eps
    Lp = run_loss(_fb=fb_p)
    Lm = run_loss(_fb=fb_m)
    gnum = (Lp - Lm) / (2 * eps)
    gaut = float(fb.grad[0].item())
    assert math.isfinite(gaut)
    assert abs(gnum - gaut) <= atol + rtol * max(
        1.0, abs(gnum)
    ), f"fb grad mismatch: num={gnum} aut={gaut}"

    # Kernel-FD cross-check: recompute gradients of T_* by asking the kernel to
    # use its internal finite differences (env flag), then compare to autograd grads.
    # This avoids any drift in a_t or masks and matches the internal debug path.
    # Prepare a fresh graph and set the env flag just for this run.
    os.environ["SSL_USE_FD_TCONST_GRADS"] = "1"
    os.environ["SSL_TCONST_FD_FIXED_MASK"] = "1"
    try:
        T_af_fd = T_af.detach().clone().requires_grad_(True)
        T_as_fd = T_as.detach().clone().requires_grad_(True)
        T_sf_fd = T_sf.detach().clone().requires_grad_(True)
        T_ss_fd = T_ss.detach().clone().requires_grad_(True)
        comp_slope_fd = comp_slope.detach().clone().requires_grad_(True)
        comp_thresh_fd = comp_thresh.detach().clone().requires_grad_(True)
        fb_fd = fb.detach().clone().requires_grad_(True)
        y_db_fd = ssl2_smoother(
            x_peak_dB=x_peak_dB,
            T_af=T_af_fd,
            T_as=T_as_fd,
            T_sf=T_sf_fd,
            T_ss=T_ss_fd,
            comp_slope=comp_slope_fd,
            comp_thresh=comp_thresh_fd,
            feedback_coeff=fb_fd,
            k=k,
            fs=fs,
            soft_gate=soft_gate,
        )
        L_fd = y_db_fd.sum()
        L_fd.backward()
        # Compare kernel FD grads to analytic grads from the first run
        for name, g_fd, g_aut in (
            ("T_af", float(T_af_fd.grad[0].item()), float(T_af.grad[0].item())),
            ("T_as", float(T_as_fd.grad[0].item()), float(T_as.grad[0].item())),
            ("T_sf", float(T_sf_fd.grad[0].item()), float(T_sf.grad[0].item())),
            ("T_ss", float(T_ss_fd.grad[0].item()), float(T_ss.grad[0].item())),
        ):
            assert math.isfinite(g_aut) and math.isfinite(g_fd)
            if os.getenv("SSL_DEBUG_TCONST_CHECK"):
                print(f"[test_dbg] kernel FD vs aut: {name} {g_fd} vs {g_aut}")
            assert abs(g_fd - g_aut) <= atol + rtol_tconst * max(1.0, abs(g_fd)), (
                f"{name} grad mismatch (kernel FD vs aut): fd={g_fd} aut={g_aut}"
            )
    finally:
        # Clean up env flag to not affect other tests
        os.environ.pop("SSL_USE_FD_TCONST_GRADS", None)
        os.environ.pop("SSL_TCONST_FD_FIXED_MASK", None)

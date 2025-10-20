import os
import math
import torch
import pytest

from differential_dynamics.benchmarks.bench_utilities import gain_db
from differential_dynamics.backends.torch.gain import SSL_comp_gain
from differential_dynamics.benchmarks.signals import (
    step as step_sig,
    probe_p8_attack_doublet,
    probe_p2b_release_drop_near_threshold,
)


def _set_env_analytic_no_salt():
    os.environ["SSL_USE_FD_TCONST_GRADS"] = "0"
    os.environ["SSL_USE_ANALYTIC_JAC"] = "1"
    os.environ["SSL_USE_SALTATION"] = "0"


def _set_env_analytic_with_salt():
    os.environ["SSL_USE_FD_TCONST_GRADS"] = "0"
    os.environ["SSL_USE_ANALYTIC_JAC"] = "1"
    os.environ["SSL_USE_SALTATION"] = "1"
    os.environ.setdefault("SSL_SALTATION_MAX_FLIPS", "4096")
    os.environ.setdefault("SSL_SALTATION_EPS_REL", "1e-3")
    os.environ.setdefault("SSL_SALTATION_MAX_BACKOFF", "6")


def _fd_grad_T(loss_fn, Tvals, which, rel_eps=1e-3):
    T = list(Tvals)
    base = float(T[which])
    e = rel_eps * (abs(base) if abs(base) > 1e-12 else 1.0)
    T[which] = base * math.exp(+e)
    Lp = loss_fn(*T)
    T[which] = base * math.exp(-e)
    Lm = loss_fn(*T)
    g_lnT = (Lp - Lm) / (2.0 * e)
    return float(g_lnT / base)


def _loss_for_T(x_peak_db, fs, theta):
    def f(Taf, Tas, Tsf, Tss):
        y = SSL_comp_gain(
            x_peak_dB=x_peak_db,
            comp_thresh=theta["ct"],
            comp_ratio=theta["cr"],
            T_attack_fast=Taf,
            T_attack_slow=Tas,
            T_shunt_fast=Tsf,
            T_shunt_slow=Tss,
            feedback_coeff=theta["fb"],
            k=0.0,
            fs=fs,
            soft_gate=False,
        )
        L = 0.5 * torch.mean(y * y)
        return float(L.item())
    return f


@pytest.mark.parametrize("level", [0.05, 0.12, 0.3])
def test_analytic_vs_fd_no_saltation_no_flip(level):
    _set_env_analytic_no_salt()
    fs = 44100
    T = int(2.0 * fs)
    x = torch.full((1, T), float(level), dtype=torch.float32)
    x_peak_db = gain_db(x.abs().to(torch.float32))
    theta = dict(ct=-24.0, cr=4.0, fb=0.999)
    Taf = torch.tensor(5e-3, dtype=torch.float32, requires_grad=True)
    Tas = torch.tensor(300e-3, dtype=torch.float32, requires_grad=True)
    Tsf = torch.tensor(200e-3, dtype=torch.float32, requires_grad=True)
    Tss = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

    y = SSL_comp_gain(
        x_peak_dB=x_peak_db,
        comp_thresh=theta["ct"],
        comp_ratio=theta["cr"],
        T_attack_fast=Taf,
        T_attack_slow=Tas,
        T_shunt_fast=Tsf,
        T_shunt_slow=Tss,
        feedback_coeff=theta["fb"],
        k=0.0,
        fs=fs,
        soft_gate=False,
    )
    L = 0.5 * torch.mean(y * y)
    L.backward()

    g_an = dict(Taf=float(Taf.grad.item()), Tas=float(Tas.grad.item()), Tsf=float(Tsf.grad.item()), Tss=float(Tss.grad.item()))

    loss_fn = _loss_for_T(x_peak_db, fs, theta)
    T0 = [float(Taf.item()), float(Tas.item()), float(Tsf.item()), float(Tss.item())]
    g_fd = dict(Taf=_fd_grad_T(loss_fn, T0, 0), Tas=_fd_grad_T(loss_fn, T0, 1), Tsf=_fd_grad_T(loss_fn, T0, 2), Tss=_fd_grad_T(loss_fn, T0, 3))

    for k in ["Taf", "Tas", "Tsf", "Tss"]:
        a, f = g_an[k], g_fd[k]
        abs_err = abs(a - f); rel_err = abs_err / max(1e-8, abs(f))
        assert abs_err <= 1e-5 or rel_err <= 0.02, f"no-flip {k}: analytic {a} vs fd {f}"


@pytest.mark.parametrize("probe_name", ["single_step_up", "attack_doublet", "drop_with_gap"])
def test_analytic_plus_saltation_vs_fd_flip_heavy(probe_name):
    _set_env_analytic_with_salt()
    fs = 44100
    T = int(2.0 * fs)
    if probe_name == "single_step_up":
        x = step_sig(fs=fs, T=T, B=1, at=0.5, amp_before=0.08, amp_after=0.80)
    elif probe_name == "attack_doublet":
        x = probe_p8_attack_doublet(fs=fs, T=T, B=1)
    elif probe_name == "drop_with_gap":
        x = probe_p2b_release_drop_near_threshold(fs=fs, T=T, B=1)
    else:
        raise RuntimeError("bad probe")
    x_peak_db = gain_db(x.abs().to(torch.float32))
    theta = dict(ct=-24.0, cr=4.0, fb=0.999)
    Taf = torch.tensor(5e-3, dtype=torch.float32, requires_grad=True)
    Tas = torch.tensor(300e-3, dtype=torch.float32, requires_grad=True)
    Tsf = torch.tensor(200e-3, dtype=torch.float32, requires_grad=True)
    Tss = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

    y = SSL_comp_gain(
        x_peak_dB=x_peak_db,
        comp_thresh=theta["ct"],
        comp_ratio=theta["cr"],
        T_attack_fast=Taf,
        T_attack_slow=Tas,
        T_shunt_fast=Tsf,
        T_shunt_slow=Tss,
        feedback_coeff=theta["fb"],
        k=0.0,
        fs=fs,
        soft_gate=False,
    )
    L = 0.5 * torch.mean(y * y)
    L.backward()

    g_an = dict(Taf=float(Taf.grad.item()), Tas=float(Tas.grad.item()), Tsf=float(Tsf.grad.item()), Tss=float(Tss.grad.item()))

    loss_fn = _loss_for_T(x_peak_db, fs, theta)
    T0 = [float(Taf.item()), float(Tas.item()), float(Tsf.item()), float(Tss.item())]
    g_fd = dict(Taf=_fd_grad_T(loss_fn, T0, 0), Tas=_fd_grad_T(loss_fn, T0, 1), Tsf=_fd_grad_T(loss_fn, T0, 2), Tss=_fd_grad_T(loss_fn, T0, 3))

    for k in ["Taf", "Tas", "Tsf", "Tss"]:
        a, f = g_an[k], g_fd[k]
        abs_err = abs(a - f); rel_err = abs_err / max(1e-8, abs(f))
        assert abs_err <= 5e-5 or rel_err <= 0.05, f"{probe_name} {k}: analytic+salt {a} vs fd {f}"


def _backward_T_grads(x_peak_db, fs, theta, Taf, Tas, Tsf, Tss, use_fd: bool):
    if use_fd:
        os.environ["SSL_USE_FD_TCONST_GRADS"] = "1"
    else:
        os.environ["SSL_USE_FD_TCONST_GRADS"] = "0"
        os.environ["SSL_USE_ANALYTIC_JAC"] = "1"
    y = SSL_comp_gain(
        x_peak_dB=x_peak_db,
        comp_thresh=theta["ct"],
        comp_ratio=theta["cr"],
        T_attack_fast=Taf,
        T_attack_slow=Tas,
        T_shunt_fast=Tsf,
        T_shunt_slow=Tss,
        feedback_coeff=theta["fb"],
        k=0.0,
        fs=fs,
        soft_gate=False,
    )
    # Backward scalar is L = sum_t grad_y[t] * y[t]; using grad_y = y here compares the two modes consistently
    L = 0.5 * torch.mean(y * y)
    L.backward()
    return dict(Taf=float(Taf.grad.item()), Tas=float(Tas.grad.item()), Tsf=float(Tsf.grad.item()), Tss=float(Tss.grad.item()))


def test_backward_analytic_vs_fd_fixedmask_no_flip():
    # Apples-to-apples: compare analytic vs in-kernel FD for the exact backward scalar and fixed hard mask.
    _set_env_analytic_no_salt()
    fs = 44100
    T = int(2.0 * fs)
    x = torch.full((1, T), 0.12, dtype=torch.float32)
    x_peak_db = gain_db(x.abs().to(torch.float32))
    theta = dict(ct=-24.0, cr=4.0, fb=0.999)
    Taf_a = torch.tensor(5e-3, dtype=torch.float32, requires_grad=True)
    Tas_a = torch.tensor(300e-3, dtype=torch.float32, requires_grad=True)
    Tsf_a = torch.tensor(200e-3, dtype=torch.float32, requires_grad=True)
    Tss_a = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

    # Analytic
    g_an = _backward_T_grads(x_peak_db, fs, theta, Taf_a, Tas_a, Tsf_a, Tss_a, use_fd=False)

    # Reset params for FD
    Taf_f = torch.tensor(5e-3, dtype=torch.float32, requires_grad=True)
    Tas_f = torch.tensor(300e-3, dtype=torch.float32, requires_grad=True)
    Tsf_f = torch.tensor(200e-3, dtype=torch.float32, requires_grad=True)
    Tss_f = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

    g_fd = _backward_T_grads(x_peak_db, fs, theta, Taf_f, Tas_f, Tsf_f, Tss_f, use_fd=True)

    for k in ["Taf", "Tas", "Tsf", "Tss"]:
        a, f = g_an[k], g_fd[k]
        abs_err = abs(a - f); rel_err = abs_err / max(1e-8, abs(f))
        assert abs_err <= 1e-5 or rel_err <= 0.02, f"fixedmask backward {k}: analytic {a} vs fd {f}"

import os
import torch
from differential_dynamics.backends.torch.ssl_smoother_ext import _get_ext

def test_dbg_dab_matches_numeric():
    ext = _get_ext()
    # Typical rates from tests (attack branch)
    fs = 44100.0
    Ts = 1.0 / fs
    Raf, Ras, Rsf, Rss = 1.0/0.010, 1.0/0.050, 1.0/0.030, 1.0/0.200

    # Ensure integral method is used by default for this check
    os.environ.pop("SSL_ANALYTIC_BD_METHOD", None)

    an = ext.dbg_dab_analytic(Raf, Ras, Rsf, Rss, Ts)
    nu = ext.dbg_dab_numeric(Raf, Ras, Rsf, Rss, Ts)

    # Compare per-rate and per-field; tolerate small float32 diffs
    atol, rtol = 5e-4, 1e-2
    diffs = (an - nu).abs()
    max_diff = float(diffs.max().item())
    if max_diff > atol + rtol * float(nu.abs().max().item()):
        print("analytic vs numeric d(Ad,Bd)/drates diffs (attack):\n", diffs)
        print("analytic:\n", an)
        print("numeric:\n", nu)
    assert torch.allclose(an, nu, atol=atol, rtol=rtol)


def test_dbg_dab_release_matches_numeric():
    ext = _get_ext()
    # Release branch rates (series=0)
    fs = 44100.0
    Ts = 1.0 / fs
    Raf, Ras, Rsf, Rss = 0.0, 0.0, 1.0/0.030, 1.0/0.200

    os.environ.pop("SSL_ANALYTIC_BD_METHOD", None)

    an = ext.dbg_dab_analytic(Raf, Ras, Rsf, Rss, Ts)
    nu = ext.dbg_dab_numeric(Raf, Ras, Rsf, Rss, Ts)

    atol, rtol = 5e-4, 1e-2
    diffs = (an - nu).abs()
    max_diff = float(diffs.max().item())
    if max_diff > atol + rtol * float(nu.abs().max().item()):
        print("analytic vs numeric d(Ad,Bd)/drates diffs (release):\n", diffs)
        print("analytic:\n", an)
        print("numeric:\n", nu)
    assert torch.allclose(an, nu, atol=atol, rtol=rtol)

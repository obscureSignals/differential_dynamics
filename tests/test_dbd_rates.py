import os
import importlib
from pathlib import Path

import numpy as np
import torch
import pytest

# Ensure project root is importable
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in os.sys.path:
    os.sys.path.insert(0, str(proj_root))


def _load_ext():
    # Set env for analytic path and Frechet implementation
    os.environ.setdefault("SSL_ANALYTIC_BD_METHOD", "phi")  # linear-solve analytic dBd
    # Use robust quadrature Frechet (default); do not force block method
    if os.environ.get("SSL_FRECHET_METHOD", "").lower().startswith("b"):
        del os.environ["SSL_FRECHET_METHOD"]
    import differential_dynamics.backends.torch.ssl_smoother_ext as sm
    sm._ext = None
    importlib.reload(sm)
    return sm._get_ext()


def _cases():
    # Deterministic set of rate tuples and sample rates
    rng = np.random.default_rng(12345)
    cases = []
    # Reference case from docs/debug
    cases.append((200.0, 50.0, 10.0, 1.0, 48000.0))
    # A few random but well-conditioned cases
    for _ in range(4):
        Raf = float(rng.uniform(50.0, 400.0))
        Ras = float(rng.uniform(20.0, 200.0))
        Rsf = float(rng.uniform(1.0, 30.0))
        Rss = float(rng.uniform(0.3, 5.0))
        fs  = float(rng.choice([44100.0, 48000.0, 96000.0]))
        cases.append((Raf, Ras, Rsf, Rss, fs))
    return cases


@pytest.mark.parametrize("Raf,Ras,Rsf,Rss,fs", _cases())
def test_dbd_analytic_matches_double_fd(Raf, Ras, Rsf, Rss, fs):
    ext = _load_ext()
    Ts = 1.0 / fs
    out = ext.dbg_attack_dbd_compare(float(Raf), float(Ras), float(Rsf), float(Rss), float(Ts))
    arr = out.to(torch.float64).cpu().numpy()  # rows: [an_Bd1, an_Bd2, fd_Bd1, fd_Bd2]

    # Tolerances: float32-level; allow small absolute error.
    atol = 5e-6
    rtol = 1e-2

    for i, rate_name in enumerate(["R_af", "R_as", "R_sf", "R_ss"]):
        an = arr[i, 0:2]
        fd = arr[i, 2:4]
        diff = np.abs(an - fd)
        ok = np.all(diff <= atol + rtol * np.maximum(1.0, np.abs(fd)))
        if not ok:
            raise AssertionError(
                f"dBd mismatch for {rate_name} @ (Raf={Raf:.3g}, Ras={Ras:.3g}, Rsf={Rsf:.3g}, Rss={Rss:.3g}, fs={fs:.0f})\n"
                f"analytic: {an} fd: {fd} diff: {diff}"
            )

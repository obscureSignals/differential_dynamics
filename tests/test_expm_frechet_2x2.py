import math
import random
import torch
import pytest


def make_A_from_rates(r_af: float, r_as: float, r_sf: float, r_ss: float) -> torch.Tensor:
    # SSL-style 2x2 continuous-time A from series/shunt rates
    # A = [[-(r_af + r_sf), -r_af], [-r_as, -(r_as + r_ss)]]
    A = torch.tensor(
        [
            [-(r_af + r_sf), -r_af],
            [-r_as, -(r_as + r_ss)],
        ],
        dtype=torch.float64,
    )
    return A


def expm_2x2_torch(A: torch.Tensor, Ts: float) -> torch.Tensor:
    return torch.linalg.matrix_exp(A * Ts)


def frechet_expm_2x2_torch(A: torch.Tensor, H: torch.Tensor, Ts: float) -> torch.Tensor:
    # Block-matrix trick: exp([[A, H],[0, A]]) has upper-right block equal to d exp(A)[H]
    # Apply scaling Ts inside.
    Z = torch.zeros_like(A)
    top = torch.cat([A * Ts, H * Ts], dim=1)
    bot = torch.cat([Z, A * Ts], dim=1)
    M = torch.cat([top, bot], dim=0)
    EM = torch.linalg.matrix_exp(M)
    dE = EM[:2, 2:4]
    return dE


def dBd_analytic(A: torch.Tensor, H: torch.Tensor, B: torch.Tensor, Ts: float) -> torch.Tensor:
    # Bd = A^{-1} (Ad - I) B, with Ad = expm(A Ts)
    # dBd = (dX) Y B + X (dY) B + X Y (dZ), here dZ=0 when varying A only
    # X = A^{-1}, Y = (Ad - I)
    X = torch.linalg.inv(A)
    Ad = torch.linalg.matrix_exp(A * Ts)
    Y = Ad - torch.eye(2, dtype=A.dtype)
    # dX = -X H X
    dX = -X @ H @ X
    # dY via Frechet derivative
    dAd = frechet_expm_2x2_torch(A, H, Ts)
    dY = dAd
    # Assemble
    term1 = dX @ Y @ B
    term2 = X @ dY @ B
    dBd = term1 + term2
    return dBd


def dBd_numeric(A: torch.Tensor, H: torch.Tensor, B: torch.Tensor, Ts: float, eps: float = 1e-6) -> torch.Tensor:
    def Bd(A_: torch.Tensor) -> torch.Tensor:
        X = torch.linalg.inv(A_)
        Ad = torch.linalg.matrix_exp(A_ * Ts)
        Y = Ad - torch.eye(2, dtype=A.dtype)
        return X @ Y @ B

    Bp = Bd(A + eps * H)
    Bm = Bd(A - eps * H)
    return (Bp - Bm) / (2.0 * eps)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("Ts", [1 / 44100.0, 1 / 48000.0, 1 / 22050.0])
def test_frechet_expm_matches_fd(seed: int, Ts: float):
    torch.manual_seed(seed)
    random.seed(seed)
    # Generate stable, positive rates in a realistic SSL range (per second)
    # Attack rates ~ 1/0.001..1/0.1; shunt rates ~ 1/0.01..1/2.0
    r_af = 1.0 / (10 ** random.uniform(-3.0, -1.0))
    r_as = 1.0 / (10 ** random.uniform(-2.0, -0.5))
    r_sf = 1.0 / (10 ** random.uniform(-2.0, -0.5))
    r_ss = 1.0 / (10 ** random.uniform(-0.3, 0.4))

    A = make_A_from_rates(r_af, r_as, r_sf, r_ss)

    # Random direction H
    H = torch.randn(2, 2, dtype=torch.float64) * 0.1

    dE = frechet_expm_2x2_torch(A, H, Ts)

    # FD baseline for d exp(A Ts)[H Ts] / (Ts) -> compare raw dE
    eps = 1e-6
    Ep = torch.linalg.matrix_exp((A + eps * H) * Ts)
    Em = torch.linalg.matrix_exp((A - eps * H) * Ts)
    dE_fd = (Ep - Em) / (2.0 * eps)

    # Tolerances for double precision
    atol = 1e-9
    rtol = 5e-7
    assert torch.allclose(dE, dE_fd, atol=atol, rtol=rtol), f"Frechet vs FD mismatch\nA={A}\nH={H}\ndE={dE}\ndE_fd={dE_fd}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("Ts", [1 / 44100.0, 1 / 48000.0])
def test_dBd_analytic_matches_fd(seed: int, Ts: float):
    torch.manual_seed(seed)
    random.seed(seed)
    r_af = 1.0 / (10 ** random.uniform(-3.0, -1.0))
    r_as = 1.0 / (10 ** random.uniform(-2.0, -0.5))
    r_sf = 1.0 / (10 ** random.uniform(-2.0, -0.5))
    r_ss = 1.0 / (10 ** random.uniform(-0.3, 0.4))

    A = make_A_from_rates(r_af, r_as, r_sf, r_ss)

    # Ensure A is well-conditioned for inversion
    if torch.linalg.det(A).abs() < 1e-9:
        pytest.skip("Near-singular A; resample")

    H = torch.randn(2, 2, dtype=torch.float64) * 0.1
    B = torch.randn(2, 1, dtype=torch.float64)

    dBd_a = dBd_analytic(A, H, B, Ts)
    dBd_fd = dBd_numeric(A, H, B, Ts, eps=1e-6)

    atol = 1e-8
    rtol = 1e-6
    assert torch.allclose(dBd_a, dBd_fd, atol=atol, rtol=rtol), f"dBd analytic vs FD mismatch\nA={A}\nH={H}\nB={B}\ndBd_a={dBd_a}\ndBd_fd={dBd_fd}"

import torch
import math


def expm_block_frechet(A, E):
    # Compute d exp(A)[E] via block matrix exponential
    # Build M = [[A, E], [0, A]]
    n = A.shape[0]
    M = torch.zeros((2*n, 2*n), dtype=A.dtype)
    M[:n, :n] = A
    M[:n, n:] = E
    M[n:, n:] = A
    EM = torch.linalg.matrix_exp(M)
    return EM[:n, n:]


def phi1_from_solve(A, Ts):
    # Phi1(A,Ts) defined by A F = Ad - I
    n = A.shape[0]
    Ad = torch.linalg.matrix_exp(A * Ts)
    I = torch.eye(n, dtype=A.dtype)
    RHS = Ad - I
    # Solve A F = RHS column-wise
    # Use lstsq to be robust even if nearly singular
    F = torch.linalg.lstsq(A, RHS).solution
    return F


def dphi1_via_sylvester(A, Ts, dA):
    n = A.shape[0]
    F = phi1_from_solve(A, Ts)
    dAd = expm_block_frechet(A * Ts, dA * Ts)
    RHS = dAd - dA @ F
    dF = torch.linalg.lstsq(A, RHS).solution
    return dF


def bd_from_phi(A, B, Ts):
    F = phi1_from_solve(A, Ts)
    return F @ B


def dbd_via_phi(A, B, Ts, dA, dB):
    dF = dphi1_via_sylvester(A, Ts, dA)
    return dF @ B + phi1_from_solve(A, Ts) @ dB


def fd_grad(fun, X, dX, eps=1e-6):
    # Central difference directional derivative of matrix-valued fun at X along dX
    return (fun(X + eps * dX) - fun(X - eps * dX)) / (2.0 * eps)


def test_dphi1_matches_fd():
    torch.manual_seed(0)
    dtype = torch.float64
    A = torch.randn(2, 2, dtype=dtype) * 0.2
    # Shift to ensure stability (negative real parts)
    A = A - 0.5 * torch.eye(2, dtype=dtype)
    dA = torch.randn(2, 2, dtype=dtype) * 0.1
    Ts = 1.0 / 44100.0

    def F_of_A(Amat):
        return phi1_from_solve(Amat, Ts)

    dF_fd = fd_grad(F_of_A, A, dA)
    dF_an = dphi1_via_sylvester(A, Ts, dA)

    assert torch.allclose(dF_an, dF_fd, atol=1e-9, rtol=1e-6), f"dPhi1 mismatch\nanalytical=\n{dF_an}\nfd=\n{dF_fd}"


def test_dbd_phi_matches_fd():
    torch.manual_seed(1)
    dtype = torch.float64
    A = torch.randn(2, 2, dtype=dtype) * 0.2
    A = A - 0.5 * torch.eye(2, dtype=dtype)
    B = torch.randn(2, 1, dtype=dtype) * 0.3
    dA = torch.randn(2, 2, dtype=dtype) * 0.1
    dB = torch.randn(2, 1, dtype=dtype) * 0.1
    Ts = 1.0 / 48000.0

    def Bd_of_A(Amat):
        return bd_from_phi(Amat, B, Ts)

    def Bd_of_B(Bmat):
        return bd_from_phi(A, Bmat, Ts)

    # Directional derivative w.r.t A using fd
    dBd_fd_A = fd_grad(Bd_of_A, A, dA)
    # Directional derivative w.r.t B using fd
    dBd_fd_B = fd_grad(Bd_of_B, B, dB)

    dBd_an = dbd_via_phi(A, B, Ts, dA, dB)
    dBd_fd = dBd_fd_A + dBd_fd_B

    assert torch.allclose(dBd_an, dBd_fd, atol=1e-9, rtol=1e-6), f"dBd mismatch\nanalytical=\n{dBd_an}\nfd=\n{dBd_fd}"
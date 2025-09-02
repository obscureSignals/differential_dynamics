import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from numba import njit, prange, cuda
from typing import Tuple, Any, Optional, Callable
from torchlpc import sample_wise_lpc


@cuda.jit
def compressor_cuda_kernel(
    x: np.ndarray,
    zi: np.ndarray,
    at: np.ndarray,
    rt: np.ndarray,
    y: np.ndarray,
    at_mask: np.ndarray,
    B: int,
    T: int,
):
    """
    CUDA kernel implementing a one-pole, per-sample smoothing with a hard
    attack/release switch. For simplicity and determinism, this launches one
    thread per batch item (threads_per_block=1) and iterates over time in Python.

    Parameters
    - x: input sequence to be smoothed (B, T)
    - zi: initial state per batch (B,)
    - at: attack coefficient per batch (B,) in (0, 1)
    - rt: release coefficient per batch (B,) in (0, 1)
    - y: output buffer (B, T)
    - at_mask: output boolean mask (B, T) indicating where attack branch was taken
    - B, T: batch size and time length (ints)

    Update rule per time t for batch b:
      if x[b, t] < g:  # envelope wants to fall -> attack branch
          coeff = at[b]
      else:
          coeff = rt[b]
      g = (1 - coeff) * g + coeff * x[b, t]
    """
    b: int = cuda.blockIdx.x
    i: int = cuda.threadIdx.x

    # Only one thread per block is used (i==0). Others return immediately.
    if b >= B or i > 0:
        return

    g = zi[b]
    at_b = at[b]
    rt_b = rt[b]
    for t in range(T):
        f = x[b, t]
        # Hard if/else: non-differentiable at the boundary f == g.
        if f < g:
            coeff = at_b
            at_mask[b, t] = 1
        else:
            coeff = rt_b
        # Exponential smoothing update (stable for 0 < coeff < 1).
        g *= 1 - coeff
        g += coeff * f
        y[b, t] = g


@njit(parallel=True)
def compressor_kernel(
    x: np.ndarray, zi: np.ndarray, at: np.ndarray, rt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CPU implementation (Numba JIT) of the same hard-switched one-pole smoother
    as compressor_cuda_kernel. Returns both the smoothed output and a boolean
    attack mask that records where the attack branch was taken. The mask enables
    a custom backward pass that routes gradients through the chosen branch.
    """
    B, T = x.shape
    y = np.empty_like(x)
    at_mask = np.zeros_like(x, dtype=np.bool_)

    for b in prange(B):
        g = zi[b]
        at_b = at[b]
        rt_b = rt[b]
        for t in range(T):
            f = x[b, t]
            flag = f < g
            # TODO: make if-else differentiable
            if flag:
                coeff = at_b
                at_mask[b, t] = True
            else:
                coeff = rt_b
            g *= 1 - coeff
            g += coeff * f
            y[b, t] = g

    return y, at_mask


def compressor_cuda(
    x: torch.Tensor, zi: torch.Tensor, at: torch.Tensor, rt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Thin wrapper to launch the CUDA kernel on torch Tensors. The boolean mask is
    returned (as torch.bool) to be used by the custom autograd Function.

    Note: This kernel is intentionally simple (one thread per batch item). It is
    correct but not throughput-optimized; suitable for research workloads.
    """
    B, T = x.shape
    y = torch.empty_like(x)
    at_mask = torch.zeros_like(x, dtype=torch.uint8)

    threads_per_block = 1
    blocks_per_grid = B

    compressor_cuda_kernel[blocks_per_grid, threads_per_block](
        cuda.as_cuda_array(x),
        cuda.as_cuda_array(zi),
        cuda.as_cuda_array(at),
        cuda.as_cuda_array(rt),
        cuda.as_cuda_array(y),
        cuda.as_cuda_array(at_mask),
        B,
        T,
    )
    return y, at_mask.bool()


class CompressorFunction(Function):
    """
    Custom autograd function for the hard-switched one-pole envelope.

    Forward: runs CUDA or Numba CPU path and returns (y, at_mask).
    Backward: uses the recorded at_mask to pick per-sample coefficients and
    propagate gradients efficiently via sample_wise_lpc (a batched IIR solver).

    The hard switch makes the function piecewise differentiable. Gradients are
    well-defined away from the switching boundary; at the boundary (measure-zero)
    the subgradient is chosen implicitly by the branch taken during forward.
    """

    @staticmethod
    def forward(
        x: torch.Tensor, zi: torch.Tensor, at: torch.Tensor, rt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.is_cuda:
            y, at_mask = compressor_cuda(
                x.detach(), zi.detach(), at.detach(), rt.detach()
            )
        else:
            y, at_mask = compressor_kernel(
                x.detach().cpu().numpy(),
                zi.detach().cpu().numpy(),
                at.detach().cpu().numpy(),
                rt.detach().cpu().numpy(),
            )
            y = torch.from_numpy(y).to(x.device)
            at_mask = torch.from_numpy(at_mask).to(x.device)
        return y, at_mask

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
        """Save tensors for backward and mark at_mask as non-differentiable."""
        x, zi, at, rt = inputs
        y, at_mask = output
        ctx.mark_non_differentiable(at_mask)
        ctx.save_for_backward(x, y, zi, at, rt, at_mask)
        ctx.save_for_forward(x, y, zi, at, rt, at_mask)
        return ctx

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor, _
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backpropagate gradients through the one-pole recursion. The difference
        equation is:
            y_t = (1 - beta_t) * y_{t-1} + beta_t * x_t,
        where beta_t = at if at_mask[t] else rt.

        The backward pass solves the transposed system using sample_wise_lpc,
        taking care to handle the initial condition (zi) correctly. Gradients
        with respect to x are scaled by beta_t (local sensitivity), while
        gradients for at/rt are accumulated only on the time steps where that
        branch was active.
        """
        x, y, zi, at, rt, at_mask = ctx.saved_tensors
        grad_x = grad_zi = grad_at = grad_rt = None

        coeffs = torch.where(at_mask, at.unsqueeze(1), rt.unsqueeze(1))
        # IIR in LPC form expects 'a' coefficients for y_t + a1*y_{t-1} = input.
        # Our recursion rearranged gives a1 = (beta_t - 1).
        lpc_a = coeffs.unsqueeze(2) - 1
        padded_lpc_a = F.pad(lpc_a.transpose(1, 2), (0, 1)).transpose(1, 2)

        if not ctx.needs_input_grad[1]:
            # If zi gradient not needed, don't prepend the extra sample.
            padded_grad_y = grad_y
            padded_lpc_a = padded_lpc_a[:, 1:]
        else:
            # Prepend one step for zi handling.
            padded_grad_y = F.pad(grad_y.unsqueeze(1), (1, 0)).squeeze(1)

        # Solve transposed IIR to propagate grad_y back through time.
        grad_x_unscaled = sample_wise_lpc(
            padded_grad_y.flip(1), padded_lpc_a.flip(1)
        ).flip(1)

        if ctx.needs_input_grad[1]:
            # Split off the zi gradient from the time sequence.
            grad_zi, grad_x_unscaled = grad_x_unscaled[:, 0], grad_x_unscaled[:, 1:]

        if ctx.needs_input_grad[0]:
            # Local sensitivity dy/dx_t = beta_t
            grad_x = grad_x_unscaled * coeffs

        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            # d y_t / d beta_t = x_t - y_{t-1}
            grad_combined = grad_x_unscaled * (
                x - torch.cat([zi.unsqueeze(1), y[:, :-1]], dim=1)
            )
            if ctx.needs_input_grad[2]:
                grad_at = torch.where(at_mask, grad_combined, 0.0).sum(1)
            if ctx.needs_input_grad[3]:
                grad_rt = torch.where(~at_mask, grad_combined, 0.0).sum(1)

        return grad_x, grad_zi, grad_at, grad_rt

    @staticmethod
    def jvp(
        ctx: Any,
        grad_x: torch.Tensor,
        grad_zi: torch.Tensor,
        grad_at: torch.Tensor,
        grad_rt: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward-mode (Jacobian-vector product). Mirrors the backward logic but
        computes directional derivatives in the forward sweep.
        """
        x, y, zi, at, rt, at_mask = ctx.saved_tensors
        coeffs = torch.where(at_mask, at.unsqueeze(1), rt.unsqueeze(1))

        fwd_x = 0 if grad_x is None else grad_x * coeffs

        fwd_combined: torch.Tensor
        if grad_at is None and grad_rt is None:
            fwd_combined = fwd_x
        else:
            grad_beta = torch.where(
                at_mask,
                0.0 if grad_at is None else grad_at.unsqueeze(1),
                0.0 if grad_rt is None else grad_rt.unsqueeze(1),
            )
            fwd_combined = fwd_x + grad_beta * (
                x - torch.cat([zi.unsqueeze(1), y[:, :-1]], dim=1)
            )
        return (
            sample_wise_lpc(
                fwd_combined,
                coeffs.unsqueeze(2) - 1,
                grad_zi if grad_zi is None else grad_zi.unsqueeze(1),
            ),
            None,
        )

    @staticmethod
    def vmap(info, in_dims, *args):
        """Vectorized mapping support for torch.func.vmap."""
        def maybe_expand_bdim_at_front(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        x, zi, at, rt = tuple(
            map(
                lambda x: x.reshape(-1, *x.shape[2:]),
                map(maybe_expand_bdim_at_front, args, in_dims),
            )
        )

        y, at_mask = CompressorFunction.apply(x, zi, at, rt)
        return (
            y.reshape(info.batch_size, -1, *y.shape[1:]),
            at_mask.reshape(info.batch_size, -1, *at_mask.shape[1:]),
        ), 0


def compressor_core(*args, **kwargs) -> torch.Tensor:
    """
    Public entrypoint that returns only the smoothed sequence (y). The attack
    mask is intentionally dropped; it is used internally for autograd.
    Signature matches the upstream API: compressor_core(x, zi, at, rt).
    """
    return CompressorFunction.apply(*args, **kwargs)[0]

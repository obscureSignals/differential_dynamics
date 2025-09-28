import os
from pathlib import Path
import warnings
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load as load_ext
import differential_dynamics.backends.torch.gain as gain_mod

"""
Autograd-capable sigmoid smoother that prefers a compiled C++ CPU extension
for forward/backward, with a pure-Python fallback when the extension is
unavailable. The op implements only the gated one-pole smoother; detector and
static curve stay in Python.

Design
- Lazy build on import via torch.utils.cpp_extension.load. If build fails or
  sources are missing, we fall back to a Python implementation (correct but slow).
- The C++ op expects float32, CPU, contiguous tensors. We .contiguous() inputs
  opportunistically before calling into the extension.
- Only ar_mode="sigmoid" uses this path. Hard mode continues to use torchcomp.

Shapes
- g:        (B, T) linear gain targets (0,1]
- alpha_a:  (B,)   attack coefficients in (0,1)
- alpha_r:  (B,)   release coefficients in (0,1)
- k:        float  gate sharpness (scalar)

See docs/CPU_SIGMOID_SMOOTHER_PLAN.md for math and details.
"""

# Try to build/load the C++ extension lazily on import
_ext = None
_force_py = os.getenv("DD_FORCE_TS_SMOOTHER") in {"1", "true", "True"}
try:
    this_dir = Path(__file__).resolve()
    proj_root = this_dir.parents[3]  # project root
    src = proj_root / "csrc" / "sigmoid_smoother.cpp"
    if _force_py:
        _ext = None
        warnings.warn("DD_FORCE_TS_SMOOTHER is set; using Python/TorchScript fallback.")
    elif src.exists():
        _ext = load_ext(
            name="sigmoid_smoother_cpu",
            sources=[str(src)],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    else:
        warnings.warn(
            f"Sigmoid smoother extension source not found at {src}; using Python fallback."
        )
except Exception as e:
    warnings.warn(
        f"Failed to build/load sigmoid smoother extension: {e}. Using Python fallback."
    )
    _ext = None


class _SigmoidSmootherCoefFn(Function):
    """Custom autograd Function wrapping the C++ extension.

    Saves (g, y, alpha_a, alpha_r, k) for backward and calls into the compiled
    kernels when available. Falls back to Python loops if not.
    """

    @staticmethod
    def forward(ctx, g, alpha_a, alpha_r, k: float):
        if _ext is None:
            # Fallback: use TorchScript reference implementation
            y = gain_mod._var_alpha_smooth_sigmoid(g, alpha_a, alpha_r, float(k))
            # Save inputs for backward; we will call autograd.grad on the TS op
            ctx.save_for_backward(
                g.contiguous(), alpha_a.contiguous(), alpha_r.contiguous()
            )
            ctx.k = float(k)
            ctx.use_fallback = True
            return y
        y = _ext.forward(g, alpha_a, alpha_r, float(k))
        ctx.save_for_backward(g, y, alpha_a, alpha_r)
        ctx.k = float(k)
        ctx.use_fallback = False
        return y

    @staticmethod
    def backward(ctx, grad_out):
        k = ctx.k
        if ctx.use_fallback:
            # Recompute grads via PyTorch autograd on the TorchScript op.
            g_in, alpha_a, alpha_r = ctx.saved_tensors
            # During backward(), grad tracking is disabled by default. Re-enable it
            # to build a local graph for computing gradients of the TS function.
            with torch.enable_grad():
                g_var = g_in.detach().requires_grad_(True)
                aa_var = alpha_a.detach().requires_grad_(True)
                ar_var = alpha_r.detach().requires_grad_(True)
                y_ts = gain_mod._var_alpha_smooth_sigmoid(
                    g_var, aa_var, ar_var, float(k)
                )
                gg, gaa, gar = torch.autograd.grad(
                    outputs=y_ts,
                    inputs=(g_var, aa_var, ar_var),
                    grad_outputs=grad_out.contiguous(),
                    allow_unused=False,
                    retain_graph=False,
                    create_graph=False,
                )
            # No gradient for k in TS fallback
            return gg, gaa, gar, None
        # Extension path
        g, y, alpha_a, alpha_r = ctx.saved_tensors
        gg, gaa, gar, _gk = _ext.backward(grad_out.contiguous(), g, y, alpha_a, alpha_r, float(k))
        # k is a Python float in forward; return None for its gradient
        return gg, gaa, gar, None


def sigmoid_smoother(
    g: torch.Tensor, alpha_a: torch.Tensor, alpha_r: torch.Tensor, k: float
) -> torch.Tensor:
    """Autograd-capable sigmoid smoother calling into C++ extension when available.
    Falls back to a Python loop if extension is not available (slow).
    """
    assert g.dim() == 2, "g must be (B,T)"
    assert alpha_a.dim() == 1 and alpha_r.dim() == 1, "alpha_a/alpha_r must be (B,)"
    # Ensure CPU float tensors for extension
    if _ext is not None:
        return _SigmoidSmootherCoefFn.apply(
            g.contiguous(), alpha_a.contiguous(), alpha_r.contiguous(), float(k)
        )
    warnings.warn(
        "Using TorchScript fallback for sigmoid smoother (slower). Consider building the C++ extension."
    )
    return _SigmoidSmootherCoefFn.apply(g, alpha_a, alpha_r, float(k))

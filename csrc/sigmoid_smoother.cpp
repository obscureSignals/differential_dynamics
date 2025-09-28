// CPU implementation of a sigmoid-gated one-pole smoother with a custom autograd
// forward/backward designed to mirror torchcomp's single reverse-scan strategy.
//
// Purpose
// - Replace the Python/TorchScript recurrence (and its large autograd graph) with a
//   fused C++ kernel that performs a single O(B·T) forward pass and a single O(B·T)
//   reverse pass. This avoids graph materialization and reduces Python overhead,
//   bringing performance closer to torchcomp's custom backward for the hard A/R case.
//
// API (exposed via PyBind):
// - forward(g, alpha_a, alpha_r, k)  -> y
// - backward(grad_out, g, y, alpha_a, alpha_r, k) -> grads for (g, alpha_a, alpha_r, k)
//
// Semantics (per batch b, time t):
//   y[-1] = 1
//   Δ_db(t) = db(g_t) - db(y_{t-1}),   db(x) = 20 * log10(max(x, eps))
//   s_t = sigmoid(k * Δ_db(t))
//   α_t = s_t * α_r + (1 - s_t) * α_a
//   y_t = y_{t-1} + α_t * (g_t - y_{t-1})
//
// Backward (reverse scan):
//   See docs/CPU_SIGMOID_SMOOTHER_PLAN.md for the full derivation. We recompute s_t,
//   α_t, and Δ_db on the fly in the reverse pass from saved (g, y) to minimize
//   memory usage while keeping excellent cache behavior.
//
// Parallelism & layout:
// - Parallelized over batch (B) using at::parallel_for. The time loop (T) is serial
//   per batch element because the recurrence is inherently sequential.
// - All tensors are float32, CPU, contiguous. We use raw pointers for tight inner
//   loops and avoid allocations inside the loops.

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>

namespace {

// Match TorchScript math exactly:
// - Forward uses 20*log10(max(x, eps)) in float precision
// - Backward uses derivative zeroed when x <= eps: d/dx db(x) = 20/(ln(10)) * 1/x for x > eps, else 0
inline float db_safe(float x, float eps) {
  const float x_clamped = (x > eps) ? x : eps;
  return 20.0f * std::log10f(x_clamped);
}

inline float ddb_safe(float x, float eps) {
  // Zero grad through the clamp region to mirror autograd on clamp(min=eps)
  if (x <= eps) return 0.0f;
  const float C = 20.0f / std::logf(10.0f);
  return C / x;
}

} // anonymous namespace

// Forward pass
// Inputs:
//   g(B,T)        : target gain (linear), (0, 1]
//   alpha_a(B)    : attack coefficient in (0,1)
//   alpha_r(B)    : release coefficient in (0,1)
//   k (scalar)    : gate sharpness (dB-domain)
// Output:
//   y(B,T)        : smoothed gain (linear)
// Notes:
//   - Uses y_prev initialized to 1.0 as in the Python implementation.
//   - eps = 1e-7 used in db to match Python implementation.
//   - Parallelization over batch only; T loop is serial per batch element.
torch::Tensor sigmoid_smoother_forward(
    const torch::Tensor& g_in,
    const torch::Tensor& alpha_a_in,
    const torch::Tensor& alpha_r_in,
    double k) {
  TORCH_CHECK(g_in.device().is_cpu(), "sigmoid_smoother_forward: CPU tensor expected for g");
  TORCH_CHECK(g_in.dtype() == torch::kFloat, "sigmoid_smoother_forward: float32 expected for g");
  TORCH_CHECK(alpha_a_in.device().is_cpu() && alpha_r_in.device().is_cpu(), "alpha tensors must be on CPU");
  TORCH_CHECK(alpha_a_in.dtype() == torch::kFloat && alpha_r_in.dtype() == torch::kFloat, "alpha tensors must be float32");
  TORCH_CHECK(g_in.dim() == 2, "g must be (B,T)");
  TORCH_CHECK(alpha_a_in.dim() == 1 && alpha_r_in.dim() == 1, "alpha tensors must be (B,)");

  const auto B = static_cast<int64_t>(g_in.size(0));
  const auto T = static_cast<int64_t>(g_in.size(1));
  TORCH_CHECK(alpha_a_in.size(0) == B && alpha_r_in.size(0) == B, "alpha size mismatch with batch");

  auto g = g_in.contiguous();
  auto alpha_a = alpha_a_in.contiguous();
  auto alpha_r = alpha_r_in.contiguous();
  auto y = torch::empty_like(g);

  const float eps = 1e-7f;
  const float kf = static_cast<float>(k);

  at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {
    for (int64_t b = b_begin; b < b_end; ++b) {
      const float aa = alpha_a.data_ptr<float>()[b];
      const float ar = alpha_r.data_ptr<float>()[b];
      const float* g_ptr = g.data_ptr<float>() + b * T;
      float* y_ptr = y.data_ptr<float>() + b * T;

      // Initialize recurrence state for this batch element
      float y_prev = 1.0f;
      for (int64_t t = 0; t < T; ++t) {
        const float gt = g_ptr[t];
        const float delta_db = db_safe(gt, eps) - db_safe(y_prev, eps);
        const float s = 1.0f / (1.0f + std::expf(-kf * delta_db));
        const float alpha_t = s * ar + (1.0f - s) * aa;
        const float y_t = (1.0f - alpha_t) * y_prev + alpha_t * gt;
        y_ptr[t] = y_t;
        y_prev = y_t;
      }
    }
  });

  return y;
}

// Backward pass (custom autograd)
// Inputs:
//   grad_out(B,T) : dL/dy_t from upstream
//   g(B,T)        : saved forward target gains
//   y(B,T)        : saved forward outputs
//   alpha_a/r(B)  : coefficients
//   k (scalar)
// Outputs:
//   grad_g(B,T), grad_alpha_a(B), grad_alpha_r(B), grad_k(scalar)
//
// Strategy:
//   - Single reverse scan over time per batch element.
//   - Recompute s_t, α_t, and Δ_db(t) in-place to avoid saving full per-time state.
//   - Accumulate grad_k in a thread-local and reduce across threads atomically.
std::vector<torch::Tensor> sigmoid_smoother_backward(
    const torch::Tensor& grad_out_in,
    const torch::Tensor& g_in,
    const torch::Tensor& y_in,
    const torch::Tensor& alpha_a_in,
    const torch::Tensor& alpha_r_in,
    double k) {
  TORCH_CHECK(grad_out_in.device().is_cpu() && g_in.device().is_cpu() && y_in.device().is_cpu(), "CPU tensors expected");
  TORCH_CHECK(grad_out_in.dtype() == torch::kFloat && g_in.dtype() == torch::kFloat && y_in.dtype() == torch::kFloat, "float32 tensors expected");
  TORCH_CHECK(alpha_a_in.device().is_cpu() && alpha_r_in.device().is_cpu(), "alpha tensors must be on CPU");
  TORCH_CHECK(alpha_a_in.dtype() == torch::kFloat && alpha_r_in.dtype() == torch::kFloat, "alpha tensors must be float32");

  const auto B = static_cast<int64_t>(g_in.size(0));
  const auto T = static_cast<int64_t>(g_in.size(1));

  auto grad_out = grad_out_in.contiguous();
  auto g = g_in.contiguous();
  auto y = y_in.contiguous();
  auto alpha_a = alpha_a_in.contiguous();
  auto alpha_r = alpha_r_in.contiguous();

  auto grad_g = torch::zeros_like(g);
  auto grad_alpha_a = torch::zeros_like(alpha_a);
  auto grad_alpha_r = torch::zeros_like(alpha_r);
  auto grad_k = torch::zeros({}, torch::dtype(torch::kFloat));

  const float eps = 1e-7f;
  const float kf = static_cast<float>(k);

  // Parallelize over batch; accumulate grad_k per-thread then reduce
  std::atomic<float> grad_k_atomic(0.0f);

  at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {
    float grad_k_local = 0.0f;
    for (int64_t b = b_begin; b < b_end; ++b) {
      const float aa = alpha_a.data_ptr<float>()[b];
      const float ar = alpha_r.data_ptr<float>()[b];
      const float* g_ptr = g.data_ptr<float>() + b * T;
      const float* y_ptr = y.data_ptr<float>() + b * T;
      const float* go_ptr = grad_out.data_ptr<float>() + b * T;
      float* gg_ptr = grad_g.data_ptr<float>() + b * T;
      float gaa = 0.0f;
      float gar = 0.0f;

      float grad_y_prev = 0.0f;

      for (int64_t t = T - 1; t >= 0; --t) {
        const float upstream = go_ptr[t] + grad_y_prev;
        const float y_prev = (t == 0) ? 1.0f : y_ptr[t - 1];
        const float yt = y_ptr[t];
        const float gt = g_ptr[t];

        const float delta_db = db_safe(gt, eps) - db_safe(y_prev, eps);
        const float s = 1.0f / (1.0f + std::expf(-kf * delta_db));
        const float s_prime = s * (1.0f - s);
        const float alpha_t = s * ar + (1.0f - s) * aa;
        const float delta_y = gt - y_prev;

        // dL/dalpha_t
        const float dL_dalpha = upstream * delta_y;

        // grads to alpha_a / alpha_r
        gaa += dL_dalpha * (1.0f - s);
        gar += dL_dalpha * s;

        // gate derivatives
        const float dalpha_dk = (ar - aa) * s_prime * delta_db;
        grad_k_local += dL_dalpha * dalpha_dk;

        const float ddb_g = ddb_safe(gt, eps);
        const float ddb_y = ddb_safe(y_prev, eps);

        const float dalpha_dg = (ar - aa) * s_prime * kf * ddb_g;
        const float dalpha_dyp = -(ar - aa) * s_prime * kf * ddb_y;

        const float dy_dg = alpha_t + delta_y * dalpha_dg;
        const float dy_dyp = 1.0f - alpha_t + delta_y * dalpha_dyp;

        gg_ptr[t] += upstream * dy_dg;
        grad_y_prev = upstream * dy_dyp;
      }

      grad_alpha_a.data_ptr<float>()[b] += gaa;
      grad_alpha_r.data_ptr<float>()[b] += gar;
    }
    // Reduce local grad_k
    float old = grad_k_atomic.load(std::memory_order_relaxed);
    float desired;
    do {
      desired = old + grad_k_local;
    } while (!grad_k_atomic.compare_exchange_weak(old, desired, std::memory_order_release, std::memory_order_relaxed));
  });

  grad_k.fill_(grad_k_atomic.load());

  return {grad_g, grad_alpha_a, grad_alpha_r, grad_k};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sigmoid_smoother_forward, "Sigmoid smoother forward (CPU)");
  m.def("backward", &sigmoid_smoother_backward, "Sigmoid smoother backward (CPU)");
}

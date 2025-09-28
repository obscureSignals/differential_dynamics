// CPU SSL 2-state sigmoid-gated smoother (forward, backward stub)
//
// Forward implements a per-sample Zero-Order Hold (ZOH) discretization of a
// 2x2 continuous-time system with gate-blended series rates. The gate operates
// in the dB domain: s = sigmoid(k * (g_db - y_prev_db)). Output is y_db.
//
// Backward is intentionally left unimplemented for now and will fail loudly.
// A single reverse scan accumulating gradients w.r.t. inputs and parameters
// will be added next.

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <atomic>
#include <cmath>
#include <vector>

namespace {

inline float stable_sigmoid(float x) {
  // Clamp to avoid exp overflow; ~1e-26 tails in float32
  if (x < -60.0f) x = -60.0f;
  if (x > 60.0f) x = 60.0f;
  return 1.0f / (1.0f + std::expf(-x));
}

// Closed-form expm for 2x2 using Higham’s formula
// exp(A) = e^mu * (cosh(s) I + (sinh(s)/s) (A - mu I)), with s^2 = mu^2 - det(A)
inline void expm2x2(float A11, float A12, float A21, float A22,
                    float& E11, float& E12, float& E21, float& E22) {
  const float tr = A11 + A22;
  const float mu = 0.5f * tr;
  const float D11 = A11 - mu;
  const float D12 = A12;
  const float D21 = A21;
  const float D22 = A22 - mu;
  const float detA = A11 * A22 - A12 * A21;
  const float s2 = mu * mu - detA;

  float c, kappa;
  if (s2 < 0.0f) {
    const float r = std::sqrt(-s2);
    c = std::cos(r);
    if (r > 1e-12f) {
      kappa = std::sin(r) / r;
    } else {
      kappa = 1.0f;
    }
  } else {
    const float s = std::sqrt(s2);
    c = std::cosh(s);
    if (s > 1e-12f) {
      kappa = std::sinh(s) / s;
    } else {
      kappa = 1.0f;
    }
  }
  const float emu = std::exp(mu);
  E11 = emu * (c + kappa * D11);
  E12 = emu * (kappa * D12);
  E21 = emu * (kappa * D21);
  E22 = emu * (c + kappa * D22);
}

inline bool solve2x2(float a11, float a12, float a21, float a22,
                     float b1, float b2, float& x1, float& x2) {
  const float det = a11 * a22 - a12 * a21;
  if (std::fabs(det) < 1e-20f) return false;
  const float inv11 = a22 / det;
  const float inv12 = -a12 / det;
  const float inv21 = -a21 / det;
  const float inv22 = a11 / det;
  x1 = inv11 * b1 + inv12 * b2;
  x2 = inv21 * b1 + inv22 * b2;
  return true;
}

}  // namespace

// Forward API
// Inputs (CPU float32 contiguous):
//   g_raw_db: (B, T) target gain in dB (<= 0 typically)
//   T_af, T_as, T_sf, T_ss: (B,) positive time constants (seconds)
//   k: gate sharpness (scalar)
//   fs: sample rate (scalar Hz)
// Output: y_db: (B, T) smoothed gain in dB
torch::Tensor ssl_smoother_forward(
    const torch::Tensor& g_raw_db_in,
    const torch::Tensor& T_af_in,
    const torch::Tensor& T_as_in,
    const torch::Tensor& T_sf_in,
    const torch::Tensor& T_ss_in,
    double k,
    double fs) {
  TORCH_CHECK(g_raw_db_in.device().is_cpu(), "ssl_smoother_forward: CPU tensor expected for g_raw_db");
  TORCH_CHECK(g_raw_db_in.dtype() == torch::kFloat, "ssl_smoother_forward: float32 expected for g_raw_db");
  TORCH_CHECK(g_raw_db_in.dim() == 2, "g_raw_db must be (B,T)");
  TORCH_CHECK(T_af_in.device().is_cpu() && T_as_in.device().is_cpu() &&
                  T_sf_in.device().is_cpu() && T_ss_in.device().is_cpu(),
              "time constant tensors must be on CPU");
  TORCH_CHECK(T_af_in.dtype() == torch::kFloat && T_as_in.dtype() == torch::kFloat &&
                  T_sf_in.dtype() == torch::kFloat && T_ss_in.dtype() == torch::kFloat,
              "time constant tensors must be float32");
  TORCH_CHECK(T_af_in.dim() == 1 && T_as_in.dim() == 1 && T_sf_in.dim() == 1 && T_ss_in.dim() == 1,
              "time constant tensors must be (B,)");

  const auto B = static_cast<int64_t>(g_raw_db_in.size(0));
  const auto T = static_cast<int64_t>(g_raw_db_in.size(1));
  TORCH_CHECK(T_af_in.size(0) == B && T_as_in.size(0) == B && T_sf_in.size(0) == B && T_ss_in.size(0) == B,
              "time constant batch sizes must match B");

  auto g_raw_db = g_raw_db_in.contiguous();
  auto T_af = T_af_in.contiguous();
  auto T_as = T_as_in.contiguous();
  auto T_sf = T_sf_in.contiguous();
  auto T_ss = T_ss_in.contiguous();
  auto y_db = torch::empty_like(g_raw_db);

  const float Ts = 1.0f / static_cast<float>(fs);
  const float kf = static_cast<float>(k);

  at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {
    for (int64_t b = b_begin; b < b_end; ++b) {
      const float* g_ptr = g_raw_db.data_ptr<float>() + b * T;
      float* y_ptr = y_db.data_ptr<float>() + b * T;

      // Convert time constants to rates (positive)
      const float Taf = std::max(T_af.data_ptr<float>()[b], 1e-12f);
      const float Tas = std::max(T_as.data_ptr<float>()[b], 1e-12f);
      const float Tsf = std::max(T_sf.data_ptr<float>()[b], 1e-12f);
      const float Tss = std::max(T_ss.data_ptr<float>()[b], 1e-12f);
      const float af_r = 1.0f / Taf;
      const float as_r = 1.0f / Tas;
      const float sf_r = 1.0f / Tsf;
      const float ss_r = 1.0f / Tss;

      // Two-state vector (x1, x2) in dB-domain representation
      float x1 = 0.0f;
      float x2 = 0.0f;
      float y_prev = 0.0f;  // 0 dB = unity

      for (int64_t t = 0; t < T; ++t) {
        const float g_now = g_ptr[t];
        const float s = stable_sigmoid(kf * (g_now - y_prev));

        const float series_f = (1.0f - s) * af_r;
        const float series_s = (1.0f - s) * as_r;

        // Continuous-time A and B for this step
        const float a11 = -(series_f + sf_r);
        const float a12 = -series_f;
        const float a21 = -series_s;
        const float a22 = -(series_s + ss_r);
        const float b1 = series_f;
        const float b2 = series_s;

        // Discretize with exact ZOH using closed-form expm for 2x2
        float Ad11, Ad12, Ad21, Ad22;
        expm2x2(Ts * a11, Ts * a12, Ts * a21, Ts * a22, Ad11, Ad12, Ad21, Ad22);

        // Bd = A^{-1} (Ad - I) B, with fallback series if singular
        const float rhs1 = (Ad11 - 1.0f) * b1 + Ad12 * b2;
        const float rhs2 = Ad21 * b1 + (Ad22 - 1.0f) * b2;
        float Bd1, Bd2;
        if (!solve2x2(a11, a12, a21, a22, rhs1, rhs2, Bd1, Bd2)) {
          // Series fallback: Bd ≈ Ts*b + 0.5*Ts^2*A*b
          const float AB1 = a11 * b1 + a12 * b2;
          const float AB2 = a21 * b1 + a22 * b2;
          Bd1 = Ts * b1 + 0.5f * Ts * Ts * AB1;
          Bd2 = Ts * b2 + 0.5f * Ts * Ts * AB2;
        }

        // State update
        const float nx1 = Ad11 * x1 + Ad12 * x2 + Bd1 * g_now;
        const float nx2 = Ad21 * x1 + Ad22 * x2 + Bd2 * g_now;
        x1 = nx1;
        x2 = nx2;
        const float y_t = x1 + x2;  // [1 1] x
        y_ptr[t] = y_t;
        y_prev = y_t;
      }
    }
  });

  return y_db;
}

// Backward API (stub that fails loudly)
std::vector<torch::Tensor> ssl_smoother_backward(
    const torch::Tensor& /*grad_out_in*/,
    const torch::Tensor& /*g_raw_db_in*/,
    const torch::Tensor& /*y_db_in*/,
    const torch::Tensor& /*T_af_in*/,
    const torch::Tensor& /*T_as_in*/,
    const torch::Tensor& /*T_sf_in*/,
    const torch::Tensor& /*T_ss_in*/,
    double /*k*/, double /*fs*/) {
  TORCH_CHECK(false, "ssl_smoother_backward not implemented yet");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ssl_smoother_forward, "SSL 2-state smoother forward (CPU)");
  m.def("backward", &ssl_smoother_backward, "SSL 2-state smoother backward (CPU, stub)");
}


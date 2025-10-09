// CPU SSL 2-state sigmoid-gated smoother (forward, backward stub)
//
// Forward implements a per-sample Zero-Order Hold (ZOH) discretization of a
// 2x2 continuous-time system with gate-blended series rates. The gate operates
// in the dB domain: s = sigmoid(k * (g_db - y_prev_db)). Output is y_db.

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <atomic>
#include <cmath>
#include <vector>
#include <csignal>

namespace {

  // 2x2 Ad and 2x1 Bd, all scalars; trivially inlinable/pass-by-value
struct DiscreteAB {
  float Ad11, Ad12, Ad21, Ad22;
  float Bd1, Bd2;
};

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

// From series/shunt rates and Ts, build continuous A,B then ZOH → Ad,Bd.
// Handles near-singular A with a series fallback.
inline DiscreteAB zoh_discretize_from_rates(
    float series_fast, float series_slow,  // attack series rates
    float shunt_fast,  float shunt_slow,   // shunt rates
    float Ts)
{
  // Continuous-time A and B
  const float a11 = -(series_fast + shunt_fast);
  const float a12 = -series_fast;
  const float a21 = -series_slow;
  const float a22 = -(series_slow + shunt_slow);
  const float b1  = series_fast;
  const float b2  = series_slow;

  // expm(A*Ts) via 2x2 closed-form
  float Ad11, Ad12, Ad21, Ad22;
  expm2x2(Ts * a11, Ts * a12, Ts * a21, Ts * a22, Ad11, Ad12, Ad21, Ad22);

  // Bd = A^{-1} (Ad - I) B, with safe fallback
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
  return {Ad11, Ad12, Ad21, Ad22, Bd1, Bd2};
}

}  // namespace

// Forward API
// Inputs (CPU float32 contiguous):
//   x_peak_dB: (B, T) input signal - 20log10(abs(x))
//   T_af, T_as, T_sf, T_ss: (B,) positive time constants (seconds)
//   comp_slope: (B,)  compressor slope
//   comp_thresh: (B,) compressor threshold
//   feedback_coeff: (B,) feedback coefficient
//   k: (B,) gate sharpness
//   fs: sample rate (scalar Hz)
// Output: y_db: (B, T) smoothed gain in dB

torch::Tensor ssl_smoother_forward(
    const torch::Tensor& x_peak_dB_in,
    const torch::Tensor& T_attack_fast_in,
    const torch::Tensor& T_attack_slow_in,
    const torch::Tensor& T_shunt_fast_in,
    const torch::Tensor& T_shunt_slow_in,
    const torch::Tensor& comp_slope_in,
    const torch::Tensor& comp_thresh_in,
    const torch::Tensor& feedback_coeff_in,
    const torch::Tensor& k_in,
    const double fs,
    const bool soft_gate) {
  TORCH_CHECK(x_peak_dB_in.device().is_cpu(), "ssl_smoother_forward: CPU tensor expected for x_peak_dB");
  TORCH_CHECK(x_peak_dB_in.dtype() == torch::kFloat, "ssl_smoother_forward: float32 expected for x_peak_dB");
  TORCH_CHECK(x_peak_dB_in.dim() == 2, "x_peak_dB must be (B,T)");
  TORCH_CHECK(T_attack_fast_in.device().is_cpu() && T_attack_slow_in.device().is_cpu() &&
                  T_shunt_fast_in.device().is_cpu() && T_shunt_slow_in.device().is_cpu() && comp_slope_in.device().is_cpu() && comp_thresh_in.device().is_cpu() && feedback_coeff_in.device().is_cpu() && k_in.device().is_cpu(),
              "time constant tensors must be on CPU");
  TORCH_CHECK(T_attack_fast_in.dtype() == torch::kFloat && T_attack_slow_in.dtype() == torch::kFloat &&
                  T_shunt_fast_in.dtype() == torch::kFloat && T_shunt_slow_in.dtype() == torch::kFloat && comp_slope_in.dtype() == torch::kFloat && comp_thresh_in.dtype() == torch::kFloat && feedback_coeff_in.dtype() == torch::kFloat && k_in.dtype() == torch::kFloat,
              "time constant tensors must be float32");
  TORCH_CHECK(T_attack_fast_in.dim() == 1 && T_attack_slow_in.dim() == 1 && T_shunt_fast_in.dim() == 1 && T_shunt_slow_in.dim() == 1 && comp_slope_in.dim() == 1 && comp_thresh_in.dim() == 1 && feedback_coeff_in.dim() == 1 && k_in.dim() == 1,
              "time constant tensors must be (B,)");

  const auto B = static_cast<int64_t>(x_peak_dB_in.size(0));
  const auto T = static_cast<int64_t>(x_peak_dB_in.size(1));
  TORCH_CHECK(T_attack_fast_in.size(0) == B && T_attack_slow_in.size(0) == B && T_shunt_fast_in.size(0) == B && T_shunt_slow_in.size(0) == B && comp_slope_in.size(0) == B && comp_thresh_in.size(0) == B && feedback_coeff_in.size(0) == B && k_in.size(0) == B,
              "time constant batch sizes must match B");

  const auto x_peak_dB = x_peak_dB_in.contiguous();
  const auto T_attack_fast = T_attack_fast_in.contiguous();
  const auto T_attack_slow = T_attack_slow_in.contiguous();
  const auto T_shunt_fast = T_shunt_fast_in.contiguous();
  const auto T_shunt_slow = T_shunt_slow_in.contiguous();
  const auto comp_slope = comp_slope_in.contiguous();
  const auto comp_thresh = comp_thresh_in.contiguous();
  const auto feedback_coeff = feedback_coeff_in.contiguous();
  const auto k = k_in.contiguous();
  auto y_db = torch::empty_like(x_peak_dB);

  const float Ts = 1.0f / static_cast<float>(fs);

  at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {
    for (int64_t b = b_begin; b < b_end; ++b) {
      const float* x_ptr = x_peak_dB.data_ptr<float>() + b * T;
      float* y_ptr = y_db.data_ptr<float>() + b * T;

      const float comp_slope_f = comp_slope.data_ptr<float>()[b];
      const float comp_thresh_f = comp_thresh.data_ptr<float>()[b];
      const float fb_f = feedback_coeff.data_ptr<float>()[b];
      const float kf = k.data_ptr<float>()[b];

      // Convert time constants to rates (positive)
      const float T_attack_fast_f = std::max(T_attack_fast.data_ptr<float>()[b], 1e-12f);
      const float T_attack_slow_f = std::max(T_attack_slow.data_ptr<float>()[b], 1e-12f);
      const float T_shunt_fast_f = std::max(T_shunt_fast.data_ptr<float>()[b], 1e-12f);
      const float T_shunt_slow_f = std::max(T_shunt_slow.data_ptr<float>()[b], 1e-12f);
      const float R_attack_fast = 1.0f / T_attack_fast_f;
      const float R_attack_slow = 1.0f / T_attack_slow_f;
      const float R_shunt_fast = 1.0f / T_shunt_fast_f;
      const float R_shunt_slow = 1.0f / T_shunt_slow_f;

      // Two-state vector (x1, x2) in dB-domain representation
      float x1 = 0.0f;
      float x2 = 0.0f;
      float y_prev = 0.0f;  // 0 dB = unity

      DiscreteAB AB = {};
      DiscreteAB AB_attack = {};
      DiscreteAB AB_release = {};

      if (!soft_gate)
      {
        AB_attack = zoh_discretize_from_rates(R_attack_fast, R_attack_slow, R_shunt_fast, R_shunt_slow, Ts);
        AB_release = zoh_discretize_from_rates(0.f, 0.f, R_shunt_fast, R_shunt_slow, Ts);
      }

      for (int64_t t = 0; t < T; ++t) {
        const float x_dB_now = x_ptr[t] + y_prev * fb_f; // apply feedback
        float gain_raw_db = comp_slope_f * (comp_thresh_f - x_dB_now);
        if (gain_raw_db > 0.0f) {
            gain_raw_db = 0.0f;
        }

        const auto delta = gain_raw_db - y_prev;

        if (soft_gate)
        {
          const float s = stable_sigmoid(kf * delta);
          const float series_fast = (1.0f - s) * R_attack_fast;
          const float series_slow = (1.0f - s) * R_attack_slow;

          AB = zoh_discretize_from_rates(series_fast, series_slow, R_shunt_fast, R_shunt_slow, Ts);
        }
        else
        {
          if (delta < 0.0f)
          {
            AB = AB_attack;
          }
          else
          {
            AB = AB_release;
          }
        }

        // State update
        const float nx1 = AB.Ad11 * x1 + AB.Ad12 * x2 + AB.Bd1 * gain_raw_db;
        const float nx2 = AB.Ad21 * x1 + AB.Ad22 * x2 + AB.Bd2 * gain_raw_db;
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

// Numeric sensitivity of DiscreteAB wrt a single rate parameter q using central differences
inline DiscreteAB zoh_disc_rates_eps(float series_fast, float series_slow, float shunt_fast, float shunt_slow, float Ts, int which, float eps) {
  float sf = series_fast, ss = series_slow, hf = shunt_fast, hs = shunt_slow;
  // Apply +eps to the selected parameter
  switch (which) {
    case 0: sf += eps; break; // R_attack_fast
    case 1: ss += eps; break; // R_attack_slow
    case 2: hf += eps; break; // R_shunt_fast
    case 3: hs += eps; break; // R_shunt_slow
    default: break;
  }
  return zoh_discretize_from_rates(sf, ss, hf, hs, Ts);
}

inline void numeric_dAB_drates(
  float series_fast, float series_slow, float shunt_fast, float shunt_slow,
  float Ts,
  // outputs: d(Ad,Bd)/dq for q in {R_af,R_as,R_sf,R_ss}
  DiscreteAB& d_q0, DiscreteAB& d_q1, DiscreteAB& d_q2, DiscreteAB& d_q3) {

  const float eps_base = 1e-6f;
  // Use relative epsilon based on magnitude
  float q[4] = {series_fast, series_slow, shunt_fast, shunt_slow};
  for (int i = 0; i < 4; ++i) {
    float eps = eps_base * std::max(std::fabs(q[i]), 1.0f);
    if (eps == 0.0f) eps = eps_base;
    DiscreteAB plus = zoh_disc_rates_eps(series_fast, series_slow, shunt_fast, shunt_slow, Ts, i, +eps);
    DiscreteAB minus = zoh_disc_rates_eps(series_fast, series_slow, shunt_fast, shunt_slow, Ts, i, -eps);
    DiscreteAB d{};
    d.Ad11 = (plus.Ad11 - minus.Ad11) / (2.0f * eps);
    d.Ad12 = (plus.Ad12 - minus.Ad12) / (2.0f * eps);
    d.Ad21 = (plus.Ad21 - minus.Ad21) / (2.0f * eps);
    d.Ad22 = (plus.Ad22 - minus.Ad22) / (2.0f * eps);
    d.Bd1  = (plus.Bd1  - minus.Bd1 ) / (2.0f * eps);
    d.Bd2  = (plus.Bd2  - minus.Bd2 ) / (2.0f * eps);
    if (i == 0) d_q0 = d;
    else if (i == 1) d_q1 = d;
    else if (i == 2) d_q2 = d;
    else d_q3 = d;
  }
}

// Backward API (hard gate). Computes grads w.r.t x_peak_dB, T_* (via rates), comp_slope, comp_thresh, feedback.
std::vector<torch::Tensor> ssl_smoother_backward(
    const torch::Tensor& grad_y_in,           // (B,T)
    const torch::Tensor& x_peak_dB_in,        // (B,T)
    const torch::Tensor& T_attack_fast_in,    // (B,)
    const torch::Tensor& T_attack_slow_in,    // (B,)
    const torch::Tensor& T_shunt_fast_in,     // (B,)
    const torch::Tensor& T_shunt_slow_in,     // (B,)
    const torch::Tensor& comp_slope_in,       // (B,)
    const torch::Tensor& comp_thresh_in,      // (B,)
    const torch::Tensor& feedback_coeff_in,   // (B,)
    const torch::Tensor& k_in,                // (B,) (ignored in hard)
    double fs,
    bool soft_gate) {
  TORCH_CHECK(grad_y_in.device().is_cpu() && x_peak_dB_in.device().is_cpu(), "backward: CPU tensors expected");
  TORCH_CHECK(!soft_gate, "ssl_smoother_backward: soft gate not supported yet in backward (hard-only)");
  TORCH_CHECK(grad_y_in.dtype() == torch::kFloat && x_peak_dB_in.dtype() == torch::kFloat, "float32 expected");
  const auto B = static_cast<int64_t>(x_peak_dB_in.size(0));
  const auto T = static_cast<int64_t>(x_peak_dB_in.size(1));
  TORCH_CHECK(grad_y_in.sizes() == x_peak_dB_in.sizes(), "grad_y and x_peak_dB must have same shape");

  auto grad_y = grad_y_in.contiguous();
  auto x_peak_dB = x_peak_dB_in.contiguous();
  auto T_af = T_attack_fast_in.contiguous();
  auto T_as = T_attack_slow_in.contiguous();
  auto T_sf = T_shunt_fast_in.contiguous();
  auto T_ss = T_shunt_slow_in.contiguous();
  auto comp_slope = comp_slope_in.contiguous();
  auto comp_thresh = comp_thresh_in.contiguous();
  auto feedback = feedback_coeff_in.contiguous();

  // Outputs
  auto gx = torch::zeros_like(x_peak_dB);
  auto gT_af = torch::zeros_like(T_af);
  auto gT_as = torch::zeros_like(T_as);
  auto gT_sf = torch::zeros_like(T_sf);
  auto gT_ss = torch::zeros_like(T_ss);
  auto g_slope = torch::zeros_like(comp_slope);
  auto g_thresh = torch::zeros_like(comp_thresh);
  auto g_fb = torch::zeros_like(feedback);
  auto g_k = torch::zeros_like(k_in); // zeros (hard)

  const float Ts = 1.0f / static_cast<float>(fs);

  // Per-batch processing
  for (int64_t b = 0; b < B; ++b) {
    const float* x_ptr = x_peak_dB.data_ptr<float>() + b * T;
    const float* gy_ptr = grad_y.data_ptr<float>() + b * T;
    float* gx_ptr = gx.data_ptr<float>() + b * T;

    const float slope = comp_slope.data_ptr<float>()[b];
    const float thresh = comp_thresh.data_ptr<float>()[b];
    const float fb = feedback.data_ptr<float>()[b];

    const float Taf = std::max(T_af.data_ptr<float>()[b], 1e-12f);
    const float Tas = std::max(T_as.data_ptr<float>()[b], 1e-12f);
    const float Tsf = std::max(T_sf.data_ptr<float>()[b], 1e-12f);
    const float Tss = std::max(T_ss.data_ptr<float>()[b], 1e-12f);

    const float Raf = 1.0f / Taf;
    const float Ras = 1.0f / Tas;
    const float Rsf = 1.0f / Tsf;
    const float Rss = 1.0f / Tss;

    // Precompute hard-gate discrete systems
    DiscreteAB AB_attack = zoh_discretize_from_rates(Raf, Ras, Rsf, Rss, Ts);
    DiscreteAB AB_release = zoh_discretize_from_rates(0.f, 0.f, Rsf, Rss, Ts);

    // We intentionally skip per-step numeric sensitivities for time constants here.
    // Time-constant gradients will be computed via finite differences on the
    // scalar loss defined by grad_y later in this function.

    // Forward recompute to capture states and branch mask
    std::vector<float> x1_prev(T), x2_prev(T), y_prev_arr(T), g_arr(T);
    std::vector<uint8_t> at_mask(T);
    float x1 = 0.f, x2 = 0.f, y_prev = 0.f;

    for (int64_t t = 0; t < T; ++t) {
      x1_prev[t] = x1; x2_prev[t] = x2; y_prev_arr[t] = y_prev;
      const float xdb = x_ptr[t] + fb * y_prev;
      float a = slope * (thresh - xdb);
      if (a > 0.f) a = 0.f;
      g_arr[t] = a;
      const float delta = a - y_prev;
      const bool is_attack = (delta < 0.f);
      at_mask[t] = static_cast<uint8_t>(is_attack);
      const DiscreteAB& AB = is_attack ? AB_attack : AB_release;
      const float nx1 = AB.Ad11 * x1 + AB.Ad12 * x2 + AB.Bd1 * a;
      const float nx2 = AB.Ad21 * x1 + AB.Ad22 * x2 + AB.Bd2 * a;
      x1 = nx1; x2 = nx2;
      y_prev = x1 + x2;
    }

    // Reverse scan
    float l1 = 0.f, l2 = 0.f; // adjoint for x (2-vector)

    for (int64_t t = T - 1; t >= 0; --t) {
      // Seed from y_t
      l1 += gy_ptr[t];
      l2 += gy_ptr[t];

      const bool is_attack = (at_mask[t] != 0);
      const DiscreteAB& AB = is_attack ? AB_attack : AB_release;

      const float a = g_arr[t];
      const float x1_t = x1_prev[t];
      const float x2_t = x2_prev[t];
      const float y_prev_t = y_prev_arr[t];

      // dL/dg = (Bd^T) * lambda
      const float dL_dg = AB.Bd1 * l1 + AB.Bd2 * l2;

      // Static curve chain (hard clamp mask m = [a < 0])
      const float m = (a < 0.f) ? 1.f : 0.f;
      const float gamma = dL_dg * m;

      // Grads for slope, thresh, x_peak_dB, fb and feedback into y_{t-1}
      // a = slope * (thresh - (x_peak_dB + fb * y_{t-1}))
      g_slope.data_ptr<float>()[b] += gamma * (thresh - (x_peak_dB.data_ptr<float>()[b*T + t] + fb * y_prev_t));
      g_thresh.data_ptr<float>()[b] += gamma * slope;
      gx_ptr[t] += gamma * (-slope);
      g_fb.data_ptr<float>()[b] += gamma * (-slope) * y_prev_t;
      const float add_y_prev_scalar = gamma * (-slope) * fb; // to y_{t-1}

      // Propagate adjoint to previous state: lambda_t = Ad^T * lambda_{t+1} + [1,1]*add_y_prev_scalar
      const float nl1 = AB.Ad11 * l1 + AB.Ad21 * l2 + add_y_prev_scalar;
      const float nl2 = AB.Ad12 * l1 + AB.Ad22 * l2 + add_y_prev_scalar;
      l1 = nl1; l2 = nl2;
    }

    // Finite-difference gradients for time constants using scalar loss L = sum_t grad_y[t] * y[t]
    auto scalar_loss_for_Ts = [&](float Taf_v, float Tas_v, float Tsf_v, float Tss_v) -> float {
      Taf_v = std::max(Taf_v, 1e-12f);
      Tas_v = std::max(Tas_v, 1e-12f);
      Tsf_v = std::max(Tsf_v, 1e-12f);
      Tss_v = std::max(Tss_v, 1e-12f);
      const float Raf_v = 1.0f / Taf_v;
      const float Ras_v = 1.0f / Tas_v;
      const float Rsf_v = 1.0f / Tsf_v;
      const float Rss_v = 1.0f / Tss_v;
      DiscreteAB ABa = zoh_discretize_from_rates(Raf_v, Ras_v, Rsf_v, Rss_v, Ts);
      DiscreteAB ABr = zoh_discretize_from_rates(0.f, 0.f, Rsf_v, Rss_v, Ts);
      float x1l = 0.f, x2l = 0.f, yprevl = 0.f;
      float L = 0.f;
      for (int64_t t = 0; t < T; ++t) {
        const float xdb = x_ptr[t] + fb * yprevl;
        float a = slope * (thresh - xdb);
        if (a > 0.f) a = 0.f;
        const float delta = a - yprevl;
        const bool is_attack = (delta < 0.f);
        const DiscreteAB& AB = is_attack ? ABa : ABr;
        const float nx1 = AB.Ad11 * x1l + AB.Ad12 * x2l + AB.Bd1 * a;
        const float nx2 = AB.Ad21 * x1l + AB.Ad22 * x2l + AB.Bd2 * a;
        x1l = nx1; x2l = nx2;
        const float y_t = x1l + x2l;
        L += gy_ptr[t] * y_t;
        yprevl = y_t;
      }
      return L;
    };
    const float eps = 1e-3f;
    // T_attack_fast
    {
      const float Lp = scalar_loss_for_Ts(Taf + eps, Tas, Tsf, Tss);
      const float Lm = scalar_loss_for_Ts(Taf - eps, Tas, Tsf, Tss);
      gT_af.data_ptr<float>()[b] = (Lp - Lm) / (2.0f * eps);
    }
    // T_attack_slow
    {
      const float Lp = scalar_loss_for_Ts(Taf, Tas + eps, Tsf, Tss);
      const float Lm = scalar_loss_for_Ts(Taf, Tas - eps, Tsf, Tss);
      gT_as.data_ptr<float>()[b] = (Lp - Lm) / (2.0f * eps);
    }
    // T_shunt_fast
    {
      const float Lp = scalar_loss_for_Ts(Taf, Tas, Tsf + eps, Tss);
      const float Lm = scalar_loss_for_Ts(Taf, Tas, Tsf - eps, Tss);
      gT_sf.data_ptr<float>()[b] = (Lp - Lm) / (2.0f * eps);
    }
    // T_shunt_slow
    {
      const float Lp = scalar_loss_for_Ts(Taf, Tas, Tsf, Tss + eps);
      const float Lm = scalar_loss_for_Ts(Taf, Tas, Tsf, Tss - eps);
      gT_ss.data_ptr<float>()[b] = (Lp - Lm) / (2.0f * eps);
    }
  }

  // Return grads in the order of inputs to forward wrapper
  return {
    gx,        // grad x_peak_dB
    gT_af,     // grad T_attack_fast
    gT_as,     // grad T_attack_slow
    gT_sf,     // grad T_shunt_fast
    gT_ss,     // grad T_shunt_slow
    g_slope,   // grad comp_slope
    g_thresh,  // grad comp_thresh
    g_fb,      // grad feedback_coeff
    g_k        // grad k (zeros in hard)
  };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ssl_smoother_forward, "SSL 2-state smoother forward (CPU)");
  m.def("backward", &ssl_smoother_backward, "SSL 2-state smoother backward (CPU, hard gate)");
}


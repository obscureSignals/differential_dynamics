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
#include <cstdlib>
#include <cstdio>

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

// Closed-form expm for 2x2 using Higham’s parametrization.
// Reference: N. J. Higham, Functions of Matrices: Theory and Computation,
// SIAM, 2008 (Sec. 10.2, 2x2 case). Expresses exp(A) via
// mu = tr(A)/2 and s^2 = mu^2 - det(A) with cosh/sinh (or cos/sin) branches.
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

// ------------------ 4x4 helpers for exact Frechet derivative (float) ------------------
inline void mat4_identity(float* M) {
  for (int i = 0; i < 16; ++i) M[i] = 0.0f;
  M[0] = M[5] = M[10] = M[15] = 1.0f;
}
inline void mat4_copy(const float* A, float* B) {
  for (int i = 0; i < 16; ++i) B[i] = A[i];
}
inline void mat4_add(const float* A, const float* B, float* C) {
  for (int i = 0; i < 16; ++i) C[i] = A[i] + B[i];
}
inline void mat4_sub(const float* A, const float* B, float* C) {
  for (int i = 0; i < 16; ++i) C[i] = A[i] - B[i];
}
inline void mat4_scale(float* A, float s) {
  for (int i = 0; i < 16; ++i) A[i] *= s;
}
inline void mat4_mul(const float* A, const float* B, float* C) {
  float R[16];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      float s = 0.0f;
      for (int k = 0; k < 4; ++k) s += A[r*4 + k] * B[k*4 + c];
      R[r*4 + c] = s;
    }
  }
  mat4_copy(R, C);
}
inline float mat4_norm1(const float* A) {
  float maxc = 0.0f;
  for (int c = 0; c < 4; ++c) {
    float s = 0.0f;
    for (int r = 0; r < 4; ++r) s += std::fabs(A[r*4 + c]);
    if (s > maxc) maxc = s;
  }
  return maxc;
}
// Solve (A) X = B for 4x4 A and 4x4 B (X overwritten into B) using Gauss-Jordan
inline bool mat4_solve(float* A, float* B) {
  // Augment A|B into 4x8
  float aug[4][8];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) aug[r][c] = A[r*4 + c];
    for (int c = 0; c < 4; ++c) aug[r][4 + c] = B[r*4 + c];
  }
  for (int i = 0; i < 4; ++i) {
    // pivot
    int piv = i;
    float amax = std::fabs(aug[i][i]);
    for (int r = i+1; r < 4; ++r) {
      float v = std::fabs(aug[r][i]);
      if (v > amax) { amax = v; piv = r; }
    }
    if (amax < 1e-30f) return false;
    if (piv != i) {
      for (int c = 0; c < 8; ++c) std::swap(aug[i][c], aug[piv][c]);
    }
    float diag = aug[i][i];
    float invd = 1.0f / diag;
    for (int c = 0; c < 8; ++c) aug[i][c] *= invd;
    for (int r = 0; r < 4; ++r) if (r != i) {
      float f = aug[r][i];
      if (f != 0.0f) {
        for (int c = 0; c < 8; ++c) aug[r][c] -= f * aug[i][c];
      }
    }
  }
  // Extract X into B
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) B[r*4 + c] = aug[r][4 + c];
  return true;
}
// Pade(6) scaling and squaring for 4x4 expm
inline void expm4x4(const float* A_in, float* E_out) {
  // Scaling
  float A[16]; mat4_copy(A_in, A);
  float norm1 = mat4_norm1(A);
  const float theta6 = 3.0f; // conservative threshold
  int s = 0;
  if (norm1 > 0.0f) {
    float t = norm1 / theta6;
    while (t > 1.0f) { t *= 0.5f; ++s; }
  }
  if (s > 0) {
    float scale = std::ldexp(1.0f, -s); // 2^{-s}
    mat4_scale(A, scale);
  }
  // Powers
  float A2[16], A4[16], A6[16];
  mat4_mul(A, A, A2);
  mat4_mul(A2, A2, A4);
  mat4_mul(A4, A2, A6);
  // Coefficients for Pade(6)
  const float c0 = 1.0f;
  const float c1 = 1.0f;
  const float c2 = 1.0f/2.0f;
  const float c3 = 1.0f/6.0f;
  const float c4 = 1.0f/24.0f;
  const float c5 = 1.0f/120.0f;
  const float c6 = 1.0f/720.0f;
  // U = A*(c1 I + c3 A2 + c5 A4)
  float I[16]; mat4_identity(I);
  float T1[16], T2[16], U[16], V[16];
  float C13[16]; // c1 I + c3 A2 + c5 A4
  for (int i = 0; i < 16; ++i) C13[i] = c1*I[i] + c3*A2[i] + c5*A4[i];
  mat4_mul(A, C13, U);
  // V = c0 I + c2 A2 + c4 A4 + c6 A6
  for (int i = 0; i < 16; ++i) V[i] = c0*I[i] + c2*A2[i] + c4*A4[i] + c6*A6[i];
  // Compute (V - U)^{-1} (V + U)
  float Vinv_arg[16], RHS[16];
  mat4_sub(V, U, Vinv_arg);
  mat4_add(V, U, RHS);
  // Solve (V-U) X = (V+U)
  float Vinv_arg_copy[16]; mat4_copy(Vinv_arg, Vinv_arg_copy);
  float X[16]; mat4_copy(RHS, X);
  bool ok = mat4_solve(Vinv_arg_copy, X);
  if (!ok) {
    // Fallback to series: E_out = I + A (poor man's fallback)
    mat4_add(I, A, E_out); // acceptable since this is extremely rare with our A
  } else {
    mat4_copy(X, E_out);
  }
  // Squaring
  for (int i = 0; i < s; ++i) {
    mat4_mul(E_out, E_out, E_out);
  }
}

// Compute exact Frechet derivative d expm(A Ts)[Ts*dA] via block matrix exponential
inline void frechet_expm_2x2(float A11, float A12, float A21, float A22,
                             float dA11, float dA12, float dA21, float dA22,
                             float Ts,
                             float& dE11, float& dE12, float& dE21, float& dE22) {
  // Build 4x4 M = [[A*Ts, dA*Ts]; [0, A*Ts]]
  float M[16];
  for (int i = 0; i < 16; ++i) M[i] = 0.0f;
  M[0] = Ts*A11; M[1] = Ts*A12; M[4] = Ts*A21; M[5] = Ts*A22;
  M[2] = Ts*dA11; M[3] = Ts*dA12; M[6] = Ts*dA21; M[7] = Ts*dA22;
  M[10] = Ts*A11; M[11] = Ts*A12; M[14] = Ts*A21; M[15] = Ts*A22;
  // expm(M)
  float EM[16];
  expm4x4(M, EM);
  // Upper-right 2x2 block is the Frechet derivative
  dE11 = EM[2];  dE12 = EM[3];
  dE21 = EM[6];  dE22 = EM[7];
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

// ------------------ double-precision variants for phi-based dBd ------------------
inline void mat4_identity(double* M) {
  for (int i = 0; i < 16; ++i) M[i] = 0.0;
  M[0] = M[5] = M[10] = M[15] = 1.0;
}
inline void mat4_copy(const double* A, double* B) {
  for (int i = 0; i < 16; ++i) B[i] = A[i];
}
inline void mat4_add(const double* A, const double* B, double* C) {
  for (int i = 0; i < 16; ++i) C[i] = A[i] + B[i];
}
inline void mat4_sub(const double* A, const double* B, double* C) {
  for (int i = 0; i < 16; ++i) C[i] = A[i] - B[i];
}
inline void mat4_scale(double* A, double s) {
  for (int i = 0; i < 16; ++i) A[i] *= s;
}
inline void mat4_mul(const double* A, const double* B, double* C) {
  double R[16];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      double s = 0.0;
      for (int k = 0; k < 4; ++k) s += A[r*4 + k] * B[k*4 + c];
      R[r*4 + c] = s;
    }
  }
  mat4_copy(R, C);
}
inline double mat4_norm1(const double* A) {
  double maxc = 0.0;
  for (int c = 0; c < 4; ++c) {
    double s = 0.0;
    for (int r = 0; r < 4; ++r) s += std::fabs(A[r*4 + c]);
    if (s > maxc) maxc = s;
  }
  return maxc;
}
inline bool mat4_solve(double* A, double* B) {
  double aug[4][8];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) aug[r][c] = A[r*4 + c];
    for (int c = 0; c < 4; ++c) aug[r][4 + c] = B[r*4 + c];
  }
  for (int i = 0; i < 4; ++i) {
    int piv = i;
    double amax = std::fabs(aug[i][i]);
    for (int r = i+1; r < 4; ++r) {
      double v = std::fabs(aug[r][i]);
      if (v > amax) { amax = v; piv = r; }
    }
    if (amax < 1e-300) return false;
    if (piv != i) {
      for (int c = 0; c < 8; ++c) std::swap(aug[i][c], aug[piv][c]);
    }
    double diag = aug[i][i];
    double invd = 1.0 / diag;
    for (int c = 0; c < 8; ++c) aug[i][c] *= invd;
    for (int r = 0; r < 4; ++r) if (r != i) {
      double f = aug[r][i];
      if (f != 0.0) {
        for (int c = 0; c < 8; ++c) aug[r][c] -= f * aug[i][c];
      }
    }
  }
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) B[r*4 + c] = aug[r][4 + c];
  return true;
}
inline void expm4x4_double(const double* A_in, double* E_out) {
  double A[16]; mat4_copy(A_in, A);
  double norm1 = mat4_norm1(A);
  const double theta6 = 3.0;
  int s = 0;
  if (norm1 > 0.0) {
    double t = norm1 / theta6;
    while (t > 1.0) { t *= 0.5; ++s; }
  }
  if (s > 0) {
    double scale = std::ldexp(1.0, -s);
    mat4_scale(A, scale);
  }
  double A2[16], A4[16], A6[16];
  mat4_mul(A, A, A2);
  mat4_mul(A2, A2, A4);
  mat4_mul(A4, A2, A6);
  const double c0 = 1.0;
  const double c1 = 1.0;
  const double c2 = 1.0/2.0;
  const double c3 = 1.0/6.0;
  const double c4 = 1.0/24.0;
  const double c5 = 1.0/120.0;
  const double c6 = 1.0/720.0;
  double I[16]; mat4_identity(I);
  double U[16], V[16];
  double C13[16];
  for (int i = 0; i < 16; ++i) C13[i] = c1*I[i] + c3*A2[i] + c5*A4[i];
  mat4_mul(A, C13, U);
  for (int i = 0; i < 16; ++i) V[i] = c0*I[i] + c2*A2[i] + c4*A4[i] + c6*A6[i];
  double Vinv_arg[16], RHS[16];
  mat4_sub(V, U, Vinv_arg);
  mat4_add(V, U, RHS);
  double Vinv_arg_copy[16]; mat4_copy(Vinv_arg, Vinv_arg_copy);
  double X[16]; mat4_copy(RHS, X);
  bool ok = mat4_solve(Vinv_arg_copy, X);
  if (!ok) {
    for (int i = 0; i < 16; ++i) E_out[i] = I[i] + A[i];
  } else {
    mat4_copy(X, E_out);
  }
  for (int i = 0; i < s; ++i) {
    mat4_mul(E_out, E_out, E_out);
  }
}
// Forward declaration for 2x2 double expm used by quadrature Frechet
inline void expm2x2_double(double A11, double A12, double A21, double A22,
                           double& E11, double& E12, double& E21, double& E22);

inline void frechet_expm_2x2_double_quad(double A11, double A12, double A21, double A22,
                                           double dA11, double dA12, double dA21, double dA22,
                                           double Ts,
                                           double& dE11, double& dE12, double& dE21, double& dE22) {
  // Frechet derivative of the matrix exponential via the integral representation:
  //   L_exp(A Ts)[Ts dA] = \int_0^{Ts} exp((Ts - t) A) (dA) exp(t A) dt
  // We evaluate the integral with a 12-point Gauss–Legendre rule on [0, Ts],
  // calling the closed-form expm2x2_double for the 2x2 exponentials. Double precision throughout.
  // This avoids numerical artifacts of the 4x4 block-expm for small Ts and stiff A.
  // References:
  // - Higham, N. J., Functions of Matrices, SIAM, 2008: integral form of the
  //   Fréchet derivative of exp.
  // - Al-Mohy, A. H., & Higham, N. J., SIAM J. Sci. Comput., 2011: efficient
  //   computation via phi-functions and quadrature ideas.
  // - Gauss–Legendre nodes/weights are standard; see Golub & Welsch (1969) for derivation.
  // 12-point Gauss-Legendre on [0, Ts]
  static const double xi[12] = {
    -0.981560647, -0.904117286, -0.769902674, -0.587317954,
    -0.367831498, -0.125233406,  0.125233406,  0.367831498,
     0.587317954,  0.769902674,  0.904117286,  0.981560647 };
  static const double wi[12] = {
     0.047175336,  0.106939326,  0.160078328,  0.203167427,
     0.233492537,  0.249147046,  0.249147046,  0.233492537,
     0.203167427,  0.160078328,  0.106939326,  0.047175336 };
  dE11 = dE12 = dE21 = dE22 = 0.0;
  // Precompute Ts*A once as scalars to scale tau efficiently
  for (int i = 0; i < 12; ++i) {
    const double w = 0.5 * wi[i] * Ts;
    const double tau = 0.5 * (xi[i] + 1.0) * Ts;
    const double left = Ts - tau;
    double L11,L12,L21,L22, R11,R12,R21,R22;
    expm2x2_double(left*A11, left*A12, left*A21, left*A22, L11,L12,L21,L22);
    expm2x2_double(tau*A11,  tau*A12,  tau*A21,  tau*A22,  R11,R12,R21,R22);
    // mid = dA * R
    const double m11 = dA11*R11 + dA12*R21;
    const double m12 = dA11*R12 + dA12*R22;
    const double m21 = dA21*R11 + dA22*R21;
    const double m22 = dA21*R12 + dA22*R22;
    // add L * mid
    dE11 += w * (L11*m11 + L12*m21);
    dE12 += w * (L11*m12 + L12*m22);
    dE21 += w * (L21*m11 + L22*m21);
    dE22 += w * (L21*m12 + L22*m22);
  }
}

inline void frechet_expm_2x2_double(double A11, double A12, double A21, double A22,
                                    double dA11, double dA12, double dA21, double dA22,
                                    double Ts,
                                    double& dE11, double& dE12, double& dE21, double& dE22) {
  // Dispatch to the quadrature implementation by default.
  // This matches the definition and is robust for small Ts and ill-conditioned A.
  // See Higham (2008), Al-Mohy & Higham (2011).
  frechet_expm_2x2_double_quad(A11,A12,A21,A22,dA11,dA12,dA21,dA22,Ts,dE11,dE12,dE21,dE22);
}
inline bool solve2x2_double(double a11, double a12, double a21, double a22,
                            double b1, double b2, double& x1, double& x2) {
  const double det = a11 * a22 - a12 * a21;
  if (std::fabs(det) < 1e-300) return false;
  const double inv11 =  a22 / det;
  const double inv12 = -a12 / det;
  const double inv21 = -a21 / det;
  const double inv22 =  a11 / det;
  x1 = inv11 * b1 + inv12 * b2;
  x2 = inv21 * b1 + inv22 * b2;
  return true;
}

inline void expm2x2_double(double A11, double A12, double A21, double A22,
                           double& E11, double& E12, double& E21, double& E22) {
  const double tr = A11 + A22;
  const double mu = 0.5 * tr;
  const double D11 = A11 - mu;
  const double D12 = A12;
  const double D21 = A21;
  const double D22 = A22 - mu;
  const double detA = A11 * A22 - A12 * A21;
  const double s2 = mu * mu - detA;
  double c, kappa;
  if (s2 < 0.0) {
    const double r = std::sqrt(-s2);
    c = std::cos(r);
    if (r > 1e-18) kappa = std::sin(r) / r; else kappa = 1.0;
  } else {
    const double s = std::sqrt(s2);
    c = std::cosh(s);
    if (s > 1e-18) kappa = std::sinh(s) / s; else kappa = 1.0;
  }
  const double emu = std::exp(mu);
  E11 = emu * (c + kappa * D11);
  E12 = emu * (kappa * D12);
  E21 = emu * (kappa * D21);
  E22 = emu * (c + kappa * D22);
}

// Generic N x N double-precision matrix helpers and expm (Pade(6))
inline void matN_identity(std::vector<double>& M, int N) {
  M.assign(N*N, 0.0);
  for (int i = 0; i < N; ++i) M[i*N + i] = 1.0;
}
inline void matN_copy(const std::vector<double>& A, std::vector<double>& B) {
  B = A;
}
inline void matN_add(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int N) {
  C.resize(N*N);
  for (int i = 0; i < N*N; ++i) C[i] = A[i] + B[i];
}
inline void matN_sub(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int N) {
  C.resize(N*N);
  for (int i = 0; i < N*N; ++i) C[i] = A[i] - B[i];
}
inline void matN_scale(std::vector<double>& A, double s) {
  for (double& v : A) v *= s;
}
inline void matN_mul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int N) {
  C.assign(N*N, 0.0);
  for (int r = 0; r < N; ++r) {
    for (int c = 0; c < N; ++c) {
      double s = 0.0;
      for (int k = 0; k < N; ++k) s += A[r*N + k] * B[k*N + c];
      C[r*N + c] = s;
    }
  }
}
inline double matN_norm1(const std::vector<double>& A, int N) {
  double maxc = 0.0;
  for (int c = 0; c < N; ++c) {
    double s = 0.0;
    for (int r = 0; r < N; ++r) s += std::fabs(A[r*N + c]);
    if (s > maxc) maxc = s;
  }
  return maxc;
}
inline bool matN_solve(std::vector<double> A, std::vector<double>& B, int N) {
  // Solve A X = B, where B is N x N (multiple RHS), Gauss-Jordan
  std::vector<double> aug(N * 2 * N, 0.0);
  for (int r = 0; r < N; ++r) {
    for (int c = 0; c < N; ++c) aug[r*(2*N) + c] = A[r*N + c];
    for (int c = 0; c < N; ++c) aug[r*(2*N) + (N + c)] = B[r*N + c];
  }
  for (int i = 0; i < N; ++i) {
    int piv = i;
    double amax = std::fabs(aug[i*(2*N) + i]);
    for (int r = i+1; r < N; ++r) {
      double v = std::fabs(aug[r*(2*N) + i]);
      if (v > amax) { amax = v; piv = r; }
    }
    if (amax < 1e-300) return false;
    if (piv != i) {
      for (int c = 0; c < 2*N; ++c) std::swap(aug[i*(2*N) + c], aug[piv*(2*N) + c]);
    }
    double diag = aug[i*(2*N) + i];
    double invd = 1.0 / diag;
    for (int c = 0; c < 2*N; ++c) aug[i*(2*N) + c] *= invd;
    for (int r = 0; r < N; ++r) if (r != i) {
      double f = aug[r*(2*N) + i];
      if (f != 0.0) {
        for (int c = 0; c < 2*N; ++c) aug[r*(2*N) + c] -= f * aug[i*(2*N) + c];
      }
    }
  }
  for (int r = 0; r < N; ++r) for (int c = 0; c < N; ++c) B[r*N + c] = aug[r*(2*N) + (N + c)];
  return true;
}
inline void expmN_double(const std::vector<double>& A_in, std::vector<double>& E_out, int N) {
  std::vector<double> A = A_in;
  double norm1 = matN_norm1(A, N);
  const double theta6 = 3.0;
  int s = 0;
  if (norm1 > 0.0) {
    double t = norm1 / theta6;
    while (t > 1.0) { t *= 0.5; ++s; }
  }
  if (s > 0) {
    double scale = std::ldexp(1.0, -s);
    matN_scale(A, scale);
  }
  std::vector<double> A2, A4, A6;
  matN_mul(A, A, A2, N);
  matN_mul(A2, A2, A4, N);
  matN_mul(A4, A2, A6, N);
  const double c0 = 1.0, c1 = 1.0, c2 = 0.5, c3 = 1.0/6.0, c4 = 1.0/24.0, c5 = 1.0/120.0, c6 = 1.0/720.0;
  std::vector<double> I; I.resize(N*N); matN_identity(I, N);
  std::vector<double> C13(N*N,0.0), U, V;
  // C13 = c1 I + c3 A2 + c5 A4
  for (int i = 0; i < N*N; ++i) C13[i] = c1*I[i] + c3*A2[i] + c5*A4[i];
  matN_mul(A, C13, U, N);
  V.resize(N*N);
  for (int i = 0; i < N*N; ++i) V[i] = c0*I[i] + c2*A2[i] + c4*A4[i] + c6*A6[i];
  std::vector<double> Vinv_arg, RHS, X;
  matN_sub(V, U, Vinv_arg, N);
  matN_add(V, U, RHS, N);
  X = RHS;
  bool ok = matN_solve(Vinv_arg, X, N);
  if (!ok) {
    E_out.resize(N*N);
    for (int i = 0; i < N*N; ++i) E_out[i] = I[i] + A[i];
  } else {
    E_out = X;
  }
  for (int i = 0; i < s; ++i) {
    std::vector<double> tmp;
    matN_mul(E_out, E_out, tmp, N);
    E_out.swap(tmp);
  }
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

// Numeric sensitivity of DiscreteAB wrt a single rate parameter q using central differences
inline DiscreteAB zoh_disc_rates_eps(float series_fast, float series_slow, float shunt_fast, float shunt_slow, float Ts, int which, float eps);

inline void numeric_dAB_drates(
  float series_fast, float series_slow, float shunt_fast, float shunt_slow,
  float Ts,
  DiscreteAB& d_q0, DiscreteAB& d_q1, DiscreteAB& d_q2, DiscreteAB& d_q3) {
  const float eps_base_default = 1e-6f;
  const float eps_base_sf = 1e-4f;
  const float eps_base_ss = 1e-4f;
  float q[4] = {series_fast, series_slow, shunt_fast, shunt_slow};
  for (int i = 0; i < 4; ++i) {
    float base = (i == 2) ? eps_base_sf : (i == 3) ? eps_base_ss : eps_base_default;
    float eps = base * std::max(std::fabs(q[i]), 1.0f);
    if (eps == 0.0f) eps = base;
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

// Analytic sensitivities of (Ad, Bd) w.r.t. the four rate parameters
// R = [Raf, Ras, Rsf, Rss] for the 2x2 continuous-time system under ZOH.
//
// ZOH identities:
//   Ad = exp(A Ts)
//   Bd = \int_0^{Ts} exp(\tau A) B d\tau = F B, where F = Phi1(A Ts)
//   A F = Ad - I   (Sylvester-like relation)
//
// References:
// - Van Loan, C. (1978). Computing integrals involving the matrix exponential.
//   IEEE Trans. Automatic Control, 23(3), 395–404. (establishes block-exponential and ZOH identities)
// - Higham, N. J., & Al-Mohy, A. H. (2011). Computing matrix functions.
//   Acta Numerica, 20, 209–287. (phi functions and Fréchet derivatives)
// - Al-Mohy, A. H., & Higham, N. J. (2011). Computing the action of the matrix
//   exponential, with an application to exponential integrators. SIAM J. Sci. Comput.
//   (phi-function evaluation context)
//
// Differentiate A F = Ad - I:
//   A dF + dA F = dAd
//   Solve A dF = dAd - dA F for each column of dF
// Then dBd = dF B + F dB.
//
// We implement this literally using 2x2 linear solves and a robust Frechet for dAd
// (12-point Gauss–Legendre quadrature); everything in double internally, cast to float at end.
inline void analytic_dAB_drates(
  float series_fast, float series_slow, float shunt_fast, float shunt_slow,
  float Ts,
  DiscreteAB& d_q0, DiscreteAB& d_q1, DiscreteAB& d_q2, DiscreteAB& d_q3) {
  // Base continuous A and B
  const float a11 = -(series_fast + shunt_fast);
  const float a12 = -series_fast;
  const float a21 = -series_slow;
  const float a22 = -(series_slow + shunt_slow);
  const float b1  = series_fast;
  const float b2  = series_slow;

  // dAd via exact Frechet derivative helper
  auto dAd_via_frechet = [&](float dA11,float dA12,float dA21,float dA22,
                             float& dAd11,float& dAd12,float& dAd21,float& dAd22){
    frechet_expm_2x2(a11, a12, a21, a22, dA11, dA12, dA21, dA22, Ts, dAd11, dAd12, dAd21, dAd22);
  };

  // Compute Ad(Ts) and Phi1(A,Ts) by solving A * F = (Ad - I)
  float Ad11_T, Ad12_T, Ad21_T, Ad22_T;
  expm2x2(Ts*a11, Ts*a12, Ts*a21, Ts*a22, Ad11_T, Ad12_T, Ad21_T, Ad22_T);
  // Solve for F columns: A * F = Ad - I
  float F11, F21, F12, F22; // columns [F11 F12; F21 F22]
  {
    float rhs1 = (Ad11_T - 1.0f);
    float rhs2 = Ad21_T;
    if (!solve2x2(a11,a12,a21,a22, rhs1,rhs2, F11,F21)) { F11 = Ts; F21 = 0.0f; }
  }
  {
    float rhs1 = Ad12_T;
    float rhs2 = (Ad22_T - 1.0f);
    if (!solve2x2(a11,a12,a21,a22, rhs1,rhs2, F12,F22)) { F12 = 0.0f; F22 = Ts; }
  }

  auto dBd_via_phi = [&](float dA11,float dA12,float dA21,float dA22,
                         float dB1,float dB2,
                         float& oBd1,float& oBd2){
    // ZOH-consistent analytic derivative of Bd using linear solves.
    // We use the identity Bd = F B with F = Phi1(A Ts), and A F = Ad - I.
    // Differentiating A F = Ad - I gives A dF + dA F = dAd, so A dF = dAd - dA F.
    // This reduces to two 2x2 linear solves (columnwise) for F and dF.
    // Implementation steps:
    // References:
    // - Van Loan (1978): block-exp identity behind ZOH and Phi1.
    // - Higham & Al-Mohy (2011): phi-functions; Phi1(Z) = Z^{-1} (exp(Z) - I) when Z invertible.
    //  1) Compute Ad = exp(A Ts)
    //  2) Solve A F = (Ad - I) for F (two 2x2 solves)
    //  3) Compute dAd via Frechet (quadrature)
    //  4) Solve A dF = dAd - dA F for dF (two 2x2 solves)
    //  5) dBd = dF B + F dB
    // Everything here is in double precision internally for stability and cast to float at exit.
    double A11=a11, A12=a12, A21=a21, A22=a22, Ts_d=Ts;
    // Ad(Ts)
    double Ad11,Ad12,Ad21,Ad22; expm2x2_double(Ts_d*A11,Ts_d*A12,Ts_d*A21,Ts_d*A22, Ad11,Ad12,Ad21,Ad22);
    // Y = (Ad - I)
    double Y11 = Ad11 - 1.0, Y12 = Ad12, Y21 = Ad21, Y22 = Ad22 - 1.0;
    // Solve A * F = Y for columns
    double F11,F21,F12,F22;
    if (!solve2x2_double(A11,A12,A21,A22, Y11,Y21, F11,F21)) { F11 = Ts_d; F21 = 0.0; }
    if (!solve2x2_double(A11,A12,A21,A22, Y12,Y22, F12,F22)) { F12 = 0.0; F22 = Ts_d; }
    // dAd via double Frechet or FD
    double dAd11,dAd12,dAd21,dAd22;
    bool use_fd_ad = false;
    if (const char* t = std::getenv("SSL_AD_USE_FD")) { use_fd_ad = (t[0]=='1'||t[0]=='t'||t[0]=='T'||t[0]=='y'||t[0]=='Y'); }
    if (!use_fd_ad) {
      frechet_expm_2x2_double(A11,A12,A21,A22, (double)dA11,(double)dA12,(double)dA21,(double)dA22, Ts_d,
                              dAd11,dAd12,dAd21,dAd22);
    } else {
      double eps = 1e-6;
      double Ap11=A11+eps*dA11, Ap12=A12+eps*dA12, Ap21=A21+eps*dA21, Ap22=A22+eps*dA22;
      double Am11=A11-eps*dA11, Am12=A12-eps*dA12, Am21=A21-eps*dA21, Am22=A22-eps*dA22;
      double Ep11,Ep12,Ep21,Ep22, Em11,Em12,Em21,Em22;
      expm2x2_double(Ts_d*Ap11,Ts_d*Ap12,Ts_d*Ap21,Ts_d*Ap22, Ep11,Ep12,Ep21,Ep22);
      expm2x2_double(Ts_d*Am11,Ts_d*Am12,Ts_d*Am21,Ts_d*Am22, Em11,Em12,Em21,Em22);
      dAd11=(Ep11-Em11)/(2*eps); dAd12=(Ep12-Em12)/(2*eps);
      dAd21=(Ep21-Em21)/(2*eps); dAd22=(Ep22-Em22)/(2*eps);
    }
    // R = dAd - dA F
    double dAF11 = dA11*F11 + dA12*F21;
    double dAF12 = dA11*F12 + dA12*F22;
    double dAF21 = dA21*F11 + dA22*F21;
    double dAF22 = dA21*F12 + dA22*F22;
    double R11 = dAd11 - dAF11;
    double R12 = dAd12 - dAF12;
    double R21 = dAd21 - dAF21;
    double R22 = dAd22 - dAF22;
    // Solve A * dF = R for columns
    double dF11,dF21,dF12,dF22;
    if (!solve2x2_double(A11,A12,A21,A22, R11,R21, dF11,dF21)) { dF11=0.0; dF21=0.0; }
    if (!solve2x2_double(A11,A12,A21,A22, R12,R22, dF12,dF22)) { dF12=0.0; dF22=0.0; }
    // dBd = dF * B + F * dB
    double dFB1 = dF11*b1 + dF12*b2;
    double dFB2 = dF21*b1 + dF22*b2;
    double FdB1 = F11*dB1 + F12*dB2;
    double FdB2 = F21*dB1 + F22*dB2;
    oBd1 = (float)(dFB1 + FdB1);
    oBd2 = (float)(dFB2 + FdB2);
  };

  // Fallback methods retained for A/B testing
  // Removed legacy integral dBd (kept phi linear-solve only)

  // Removed legacy inverse-form dBd (kept phi linear-solve only)

  auto deriv_for = [&](float dA11,float dA12,float dA21,float dA22,
                       float dB1,float dB2, DiscreteAB& out) {
    float dAd11,dAd12,dAd21,dAd22;
    dAd_via_frechet(dA11,dA12,dA21,dA22, dAd11,dAd12,dAd21,dAd22);
    float dBd1,dBd2;
    // Use the verified ZOH-consistent linear-solve formulation for dBd (phi)
    dBd_via_phi(dA11,dA12,dA21,dA22, dB1,dB2, dBd1,dBd2);
    out = {dAd11, dAd12, dAd21, dAd22, dBd1, dBd2};
  };

  // q0 = R_af: dA11=-1, dA12=-1, dB1=1
  deriv_for(-1.f, -1.f, 0.f, 0.f, 1.f, 0.f, d_q0);
  // q1 = R_as: dA21=-1, dA22=-1, dB2=1
  deriv_for(0.f, 0.f, -1.f, -1.f, 0.f, 1.f, d_q1);
  // q2 = R_sf: dA11=-1
  deriv_for(-1.f, 0.f, 0.f, 0.f, 0.f, 0.f, d_q2);
  // q3 = R_ss: dA22=-1
  deriv_for(0.f, 0.f, 0.f, -1.f, 0.f, 0.f, d_q3);
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
  // Backward implements hard gate only (analytic adjoint with fixed mask)
  TORCH_CHECK(!soft_gate, "ssl_smoother_backward: backward supports hard gate only (soft_gate=False)");
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

  // No surrogate blending in backward; use fixed hard mask only

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

    if (b == 0) {
      if (const char* dbgA = std::getenv("SSL_DEBUG_AD_RAW")) {
        bool onA = (dbgA[0]=='1'||dbgA[0]=='t'||dbgA[0]=='T'||dbgA[0]=='y'||dbgA[0]=='Y');
        if (onA) {
          // Print dAd (Frechet) vs numeric FD for attack A
          const float a11 = -(Raf + Rsf);
          const float a12 = -Raf;
          const float a21 = -Ras;
          const float a22 = -(Ras + Rss);
          auto print_dAd = [&](const char* name, float dA11,float dA12,float dA21,float dA22, int which){
            // Frechet dAd
            float dAd11,dAd12,dAd21,dAd22;
            frechet_expm_2x2(a11, a12, a21, a22, dA11, dA12, dA21, dA22, Ts,
                              dAd11, dAd12, dAd21, dAd22);
            // Numeric FD on Ad for rate which
            auto Ad_from_rates = [&](float Raf_v,float Ras_v,float Rsf_v,float Rss_v){
              float E11,E12,E21,E22;
              expm2x2(Ts * (-(Raf_v+Rsf_v)), Ts * (-Raf_v), Ts * (-Ras_v), Ts * (-(Ras_v+Rss_v)), E11,E12,E21,E22);
              return std::array<float,4>{E11,E12,E21,E22};
            };
            float Raf_p=Raf, Ras_p=Ras, Rsf_p=Rsf, Rss_p=Rss;
            float Raf_m=Raf, Ras_m=Ras, Rsf_m=Rsf, Rss_m=Rss;
            float q = (which==0?Raf:(which==1?Ras:(which==2?Rsf:Rss)));
            float eps = ((which==2||which==3)?1e-4f:1e-6f) * ((std::fabs(q)>1.f)?std::fabs(q):1.f);
            if (which==0) { Raf_p+=eps; Raf_m-=eps; }
            else if (which==1) { Ras_p+=eps; Ras_m-=eps; }
            else if (which==2) { Rsf_p+=eps; Rsf_m-=eps; }
            else { Rss_p+=eps; Rss_m-=eps; }
            auto Ap = Ad_from_rates(Raf_p,Ras_p,Rsf_p,Rss_p);
            auto Am = Ad_from_rates(Raf_m,Ras_m,Rsf_m,Rss_m);
            float n11=(Ap[0]-Am[0])/(2.f*eps), n12=(Ap[1]-Am[1])/(2.f*eps), n21=(Ap[2]-Am[2])/(2.f*eps), n22=(Ap[3]-Am[3])/(2.f*eps);
            std::printf("[ssl_ad_raw] ATT dAd_%s (Frechet vs FD): (%.9g, %.9g; %.9g, %.9g) vs (%.9g, %.9g; %.9g, %.9g)\n",
                        name, dAd11,dAd12,dAd21,dAd22, n11,n12,n21,n22);
          };
          print_dAd("Raf", -1.f,-1.f, 0.f,0.f, 0);
          print_dAd("Ras", 0.f,0.f, -1.f,-1.f, 1);
          print_dAd("Rsf", -1.f,0.f, 0.f,0.f, 2);
          print_dAd("Rss", 0.f,0.f, 0.f,-1.f, 3);
        }
      }
      if (const char* dbg = std::getenv("SSL_DEBUG_BD_RAW")) {
        bool on = (dbg[0]=='1'||dbg[0]=='t'||dbg[0]=='T'||dbg[0]=='y'||dbg[0]=='Y');
        if (on) {
          // Build continuous A and B for attack branch
          const float a11 = -(Raf + Rsf);
          const float a12 = -Raf;
          const float a21 = -Ras;
          const float a22 = -(Ras + Rss);
          const float b1  = Raf;
          const float b2  = Ras;
          // Ad(Ts)
          float Ad11, Ad12, Ad21, Ad22;
          expm2x2(Ts*a11, Ts*a12, Ts*a21, Ts*a22, Ad11, Ad12, Ad21, Ad22);
          // Phi1 via solve A F = (Ad - I)
          float F11, F21, F12, F22;
          {
            float rhs1 = (Ad11 - 1.0f);
            float rhs2 = Ad21;
            if (!solve2x2(a11,a12,a21,a22, rhs1,rhs2, F11,F21)) { F11 = Ts; F21 = 0.0f; }
          }
          {
            float rhs1 = Ad12;
            float rhs2 = (Ad22 - 1.0f);
            if (!solve2x2(a11,a12,a21,a22, rhs1,rhs2, F12,F22)) { F12 = 0.0f; F22 = Ts; }
          }
          // Bd via phi and exact (from zoh)
          const float Bd_phi_1 = F11*b1 + F12*b2;
          const float Bd_phi_2 = F21*b1 + F22*b2;
          std::printf("[ssl_bd_raw] ATT Bd phi vs exact: (%.9g, %.9g) vs (%.9g, %.9g)\n",
                      Bd_phi_1, Bd_phi_2, AB_attack.Bd1, AB_attack.Bd2);

          // Directional dBd per-rate via phi vs FD on Bd
          auto dBd_phi = [&](float dA11,float dA12,float dA21,float dA22,
                              float dB1,float dB2, float& o1, float& o2){
            // dAd via Frechet at Ts
            float dAd11,dAd12,dAd21,dAd22;
            frechet_expm_2x2(a11, a12, a21, a22, dA11, dA12, dA21, dA22, Ts,
                              dAd11, dAd12, dAd21, dAd22);
            // dA*F
            const float dAF11 = dA11*F11 + dA12*F21;
            const float dAF12 = dA11*F12 + dA12*F22;
            const float dAF21 = dA21*F11 + dA22*F21;
            const float dAF22 = dA21*F12 + dA22*F22;
            const float R11 = dAd11 - dAF11;
            const float R12 = dAd12 - dAF12;
            const float R21 = dAd21 - dAF21;
            const float R22 = dAd22 - dAF22;
            float dF11, dF21, dF12, dF22;
            if (!solve2x2(a11,a12,a21,a22, R11,R21, dF11,dF21)) { dF11=0.f; dF21=0.f; }
            if (!solve2x2(a11,a12,a21,a22, R12,R22, dF12,dF22)) { dF12=0.f; dF22=0.f; }
            // dBd = dF*B + F*dB
            const float dFB1 = dF11*b1 + dF12*b2;
            const float dFB2 = dF21*b1 + dF22*b2;
            const float FdB1 = F11*dB1 + F12*dB2;
            const float FdB2 = F21*dB1 + F22*dB2;
            o1 = dFB1 + FdB1;
            o2 = dFB2 + FdB2;
          };
          auto Bd_from_rates = [&](float Raf_v,float Ras_v,float Rsf_v,float Rss_v, float& o1,float& o2){
            DiscreteAB ABv = zoh_discretize_from_rates(Raf_v, Ras_v, Rsf_v, Rss_v, Ts);
            o1 = ABv.Bd1; o2 = ABv.Bd2;
          };
          auto fd_rate = [&](int which, float scale)->std::pair<float,float>{
            float Raf_p=Raf, Ras_p=Ras, Rsf_p=Rsf, Rss_p=Rss;
            float Raf_m=Raf, Ras_m=Ras, Rsf_m=Rsf, Rss_m=Rss;
            float q = (which==0?Raf:(which==1?Ras:(which==2?Rsf:Rss)));
            float eps = scale * ((std::fabs(q) > 1.f) ? std::fabs(q) : 1.f);
            if (which==0) { Raf_p+=eps; Raf_m-=eps; }
            else if (which==1) { Ras_p+=eps; Ras_m-=eps; }
            else if (which==2) { Rsf_p+=eps; Rsf_m-=eps; }
            else { Rss_p+=eps; Rss_m-=eps; }
            float p1,p2,m1,m2; Bd_from_rates(Raf_p,Ras_p,Rsf_p,Rss_p,p1,p2); Bd_from_rates(Raf_m,Ras_m,Rsf_m,Rss_m,m1,m2);
            return { (p1-m1)/(2.f*eps), (p2-m2)/(2.f*eps) };
          };
          auto print_d = [&](const char* name, float dA11,float dA12,float dA21,float dA22,
                             float dB1,float dB2, int which){
            float an1,an2; dBd_phi(dA11,dA12,dA21,dA22, dB1,dB2, an1,an2);
            auto fd = fd_rate(which, (which>=2)?1e-4f:1e-6f);
            std::printf("[ssl_bd_raw] ATT dBd_%s (phi vs FD): (%.9g, %.9g) vs (%.9g, %.9g)\n",
                        name, an1, an2, fd.first, fd.second);
          };
          // q0=R_af: dA11=-1, dA12=-1, dB=[1,0]
          print_d("Raf", -1.f,-1.f, 0.f,0.f, 1.f,0.f, 0);
          // q1=R_as: dA21=-1, dA22=-1, dB=[0,1]
          print_d("Ras", 0.f,0.f, -1.f,-1.f, 0.f,1.f, 1);
          // q2=R_sf: dA11=-1, dB=[0,0]
          print_d("Rsf", -1.f,0.f, 0.f,0.f, 0.f,0.f, 2);
          // q3=R_ss: dA22=-1, dB=[0,0]
          print_d("Rss", 0.f,0.f, 0.f,-1.f, 0.f,0.f, 3);
        }
      }
    }

    // We intentionally skip per-step numeric sensitivities for time constants here.
    // Time-constant gradients will be computed via finite differences on the
    // scalar loss defined by grad_y later in this function.

    // Forward recompute to capture states and branch mask
    std::vector<float> x1_prev(T), x2_prev(T), y_prev_arr(T), g_arr(T);
    std::vector<uint8_t> at_mask(T);
    float x1 = 0.f, x2 = 0.f, y_prev = 0.f;
    bool step_dbg = false;
    if (const char* sbd = std::getenv("SSL_DEBUG_STEP_BD")) {
      step_dbg = (sbd[0]=='1'||sbd[0]=='t'||sbd[0]=='T'||sbd[0]=='y'||sbd[0]=='Y');
    }

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

    // Reverse scan with operator adjoint accumulation
    float l1 = 0.f, l2 = 0.f; // adjoint for x (2-vector)
    std::vector<float> lam1_hist, lam2_hist;
    if (step_dbg) { lam1_hist.resize(T); lam2_hist.resize(T); }
    // Accumulate dL/dAd and dL/dBd per branch
    float gAd_att_11=0.f, gAd_att_12=0.f, gAd_att_21=0.f, gAd_att_22=0.f;
    float gBd_att_1=0.f, gBd_att_2=0.f;
    float gAd_rel_11=0.f, gAd_rel_12=0.f, gAd_rel_21=0.f, gAd_rel_22=0.f;
    float gBd_rel_1=0.f, gBd_rel_2=0.f;

    for (int64_t t = T - 1; t >= 0; --t) {
      // Seed from y_t (y_t = [1 1] x_{t+1})
      l1 += gy_ptr[t];
      l2 += gy_ptr[t];
      if (step_dbg) { lam1_hist[t] = l1; lam2_hist[t] = l2; }

      const bool is_attack = (at_mask[t] != 0);
      const DiscreteAB& AB = is_attack ? AB_attack : AB_release;

      const float a = g_arr[t];
      const float x1_t = x1_prev[t];
      const float x2_t = x2_prev[t];
      const float y_prev_t = y_prev_arr[t];

      // Operator adjoint contributions for this step: dL/dAd += lambda_{t+1} * x_t^T, dL/dBd += lambda_{t+1} * a_t
      if (is_attack) {
        gAd_att_11 += l1 * x1_t; gAd_att_12 += l1 * x2_t;
        gAd_att_21 += l2 * x1_t; gAd_att_22 += l2 * x2_t;
        gBd_att_1  += l1 * a;    gBd_att_2  += l2 * a;
      } else {
        gAd_rel_11 += l1 * x1_t; gAd_rel_12 += l1 * x2_t;
        gAd_rel_21 += l2 * x1_t; gAd_rel_22 += l2 * x2_t;
        gBd_rel_1  += l1 * a;    gBd_rel_2  += l2 * a;
      }

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
      float add_y_prev_scalar = gamma * (-slope) * fb; // to y_{t-1}

      // No surrogate gate path; fixed-mask only

      // Propagate adjoint to previous state: lambda_t = Ad^T * lambda_{t+1} + [1,1]*add_y_prev_scalar
      const float nl1 = AB.Ad11 * l1 + AB.Ad21 * l2 + add_y_prev_scalar;
      const float nl2 = AB.Ad12 * l1 + AB.Ad22 * l2 + add_y_prev_scalar;
      l1 = nl1; l2 = nl2;
    }

// Decide gradient mode: analytic (default) or FD if env requests
    bool use_fd = false;
    int subsample = 1;
    if (const char* v = std::getenv("SSL_USE_FD_TCONST_GRADS")) {
      if (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y') use_fd = true;
    }
    if (const char* s = std::getenv("SSL_TCONST_FD_SUBSAMPLE")) {
      int tmp = std::atoi(s);
      if (tmp >= 1) subsample = tmp;
    }

    if (!use_fd) {
      // Analytic vs numeric operator Jacobians with split toggles for Ad and Bd
      // Flags precedence: SSL_USE_ANALYTIC_JAC (master) -> SSL_USE_ANALYTIC_JAC_AD / _BD (override if set)
      auto env_truthy = [](const char* v){ return v && (v[0]=='1'||v[0]=='t'||v[0]=='T'||v[0]=='y'||v[0]=='Y'); };
      bool use_analytic_ad = false, use_analytic_bd = false;
      if (const char* aj = std::getenv("SSL_USE_ANALYTIC_JAC")) {
        if (env_truthy(aj)) { use_analytic_ad = true; use_analytic_bd = true; }
      }
      if (const char* aj_ad = std::getenv("SSL_USE_ANALYTIC_JAC_AD")) {
        use_analytic_ad = env_truthy(aj_ad);
      }
      if (const char* aj_bd = std::getenv("SSL_USE_ANALYTIC_JAC_BD")) {
        use_analytic_bd = env_truthy(aj_bd);
      }

      auto mix_fields = [&](const DiscreteAB& ana, const DiscreteAB& num){
        DiscreteAB m{};
        // Select Ad
        m.Ad11 = use_analytic_ad ? ana.Ad11 : num.Ad11;
        m.Ad12 = use_analytic_ad ? ana.Ad12 : num.Ad12;
        m.Ad21 = use_analytic_ad ? ana.Ad21 : num.Ad21;
        m.Ad22 = use_analytic_ad ? ana.Ad22 : num.Ad22;
        // Select Bd
        m.Bd1  = use_analytic_bd ? ana.Bd1  : num.Bd1;
        m.Bd2  = use_analytic_bd ? ana.Bd2  : num.Bd2;
        return m;
      };

      // Derivatives of (Ad,Bd) w.r.t rates for attack branch
      DiscreteAB d_att_q0, d_att_q1, d_att_q2, d_att_q3; // q0=R_af, q1=R_as, q2=R_sf, q3=R_ss
      if (use_analytic_ad || use_analytic_bd) {
        // Compute both numeric and analytic, then mix
        DiscreteAB an_att_q0, an_att_q1, an_att_q2, an_att_q3;
        DiscreteAB nu_att_q0, nu_att_q1, nu_att_q2, nu_att_q3;
        analytic_dAB_drates(Raf, Ras, Rsf, Rss, Ts, an_att_q0, an_att_q1, an_att_q2, an_att_q3);
        numeric_dAB_drates(Raf, Ras, Rsf, Rss, Ts, nu_att_q0, nu_att_q1, nu_att_q2, nu_att_q3);
        d_att_q0 = mix_fields(an_att_q0, nu_att_q0);
        d_att_q1 = mix_fields(an_att_q1, nu_att_q1);
        d_att_q2 = mix_fields(an_att_q2, nu_att_q2);
        d_att_q3 = mix_fields(an_att_q3, nu_att_q3);
      } else {
        numeric_dAB_drates(Raf, Ras, Rsf, Rss, Ts, d_att_q0, d_att_q1, d_att_q2, d_att_q3);
      }
      // Derivatives for release branch (series rates = 0)
      DiscreteAB d_rel_q0, d_rel_q1, d_rel_q2, d_rel_q3;
      if (use_analytic_ad || use_analytic_bd) {
        DiscreteAB an_rel_q0, an_rel_q1, an_rel_q2, an_rel_q3;
        DiscreteAB nu_rel_q0, nu_rel_q1, nu_rel_q2, nu_rel_q3;
        analytic_dAB_drates(0.f, 0.f, Rsf, Rss, Ts, an_rel_q0, an_rel_q1, an_rel_q2, an_rel_q3);
        numeric_dAB_drates(0.f, 0.f, Rsf, Rss, Ts, nu_rel_q0, nu_rel_q1, nu_rel_q2, nu_rel_q3);
        d_rel_q0 = mix_fields(an_rel_q0, nu_rel_q0);
        d_rel_q1 = mix_fields(an_rel_q1, nu_rel_q1);
        d_rel_q2 = mix_fields(an_rel_q2, nu_rel_q2);
        d_rel_q3 = mix_fields(an_rel_q3, nu_rel_q3);
      } else {
        numeric_dAB_drates(0.f, 0.f, Rsf, Rss, Ts, d_rel_q0, d_rel_q1, d_rel_q2, d_rel_q3);
      }

      auto dotAB = [](const DiscreteAB& d, float gAd11, float gAd12, float gAd21, float gAd22, float gBd1, float gBd2) -> float {
        float s = 0.f;
        s += gAd11 * d.Ad11 + gAd12 * d.Ad12 + gAd21 * d.Ad21 + gAd22 * d.Ad22;
        s += gBd1  * d.Bd1  + gBd2  * d.Bd2;
        return s;
      };

      float dL_dR_af = dotAB(d_att_q0, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
      float dL_dR_as = dotAB(d_att_q1, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
      float dL_dR_sf = dotAB(d_att_q2, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                           + dotAB(d_rel_q2, gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
      float dL_dR_ss = dotAB(d_att_q3, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                           + dotAB(d_rel_q3, gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
      if (use_analytic_ad || use_analytic_bd) {
        if (const char* dbg = std::getenv("SSL_DEBUG_ANALYTIC_JAC")) {
          bool on = (dbg[0] == '1' || dbg[0] == 't' || dbg[0] == 'T' || dbg[0] == 'y' || dbg[0] == 'Y');
          if (on) {
            DiscreteAB n_att_q0, n_att_q1, n_att_q2, n_att_q3;
            DiscreteAB n_rel_q0, n_rel_q1, n_rel_q2, n_rel_q3;
            numeric_dAB_drates(Raf, Ras, Rsf, Rss, Ts, n_att_q0, n_att_q1, n_att_q2, n_att_q3);
            numeric_dAB_drates(0.f, 0.f, Rsf, Rss, Ts, n_rel_q0, n_rel_q1, n_rel_q2, n_rel_q3);
            auto diff = [](const DiscreteAB& a, const DiscreteAB& b){
              return std::max({std::fabs(a.Ad11-b.Ad11), std::fabs(a.Ad12-b.Ad12), std::fabs(a.Ad21-b.Ad21), std::fabs(a.Ad22-b.Ad22), std::fabs(a.Bd1-b.Bd1), std::fabs(a.Bd2-b.Bd2)});
            };
            std::printf("[ssl_smoother_dbg] analytic vs numeric d(Ad,Bd)/dR deltas: af %.3g as %.3g sf %.3g ss %.3g\n",
                        diff(d_att_q0,n_att_q0), diff(d_att_q1,n_att_q1), diff(d_att_q2,n_att_q2), diff(d_att_q3,n_att_q3));
            // Also compare resulting dL/dR values (analytic operator Jacobians vs numeric operator Jacobians)
            auto dotAB = [](const DiscreteAB& d, float gAd11, float gAd12, float gAd21, float gAd22, float gBd1, float gBd2) -> float {
              float s = 0.f;
              s += gAd11 * d.Ad11 + gAd12 * d.Ad12 + gAd21 * d.Ad21 + gAd22 * d.Ad22;
              s += gBd1  * d.Bd1  + gBd2  * d.Bd2;
              return s;
            };
            // Recompute analytic d(Ad,Bd)/dR explicitly for comparison
            DiscreteAB a_att_q0, a_att_q1, a_att_q2, a_att_q3;
            DiscreteAB a_rel_q0, a_rel_q1, a_rel_q2, a_rel_q3;
            analytic_dAB_drates(Raf, Ras, Rsf, Rss, Ts, a_att_q0, a_att_q1, a_att_q2, a_att_q3);
            analytic_dAB_drates(0.f, 0.f, Rsf, Rss, Ts, a_rel_q0, a_rel_q1, a_rel_q2, a_rel_q3);
            // Full contractions
            float dL_dR_af_num = dotAB(n_att_q0, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dL_dR_as_num = dotAB(n_att_q1, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dL_dR_sf_num = dotAB(n_att_q2, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(n_rel_q2, gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dL_dR_ss_num = dotAB(n_att_q3, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(n_rel_q3, gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dL_dR_af_an  = dotAB(a_att_q0, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dL_dR_as_an  = dotAB(a_att_q1, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dL_dR_sf_an  = dotAB(a_att_q2, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(a_rel_q2, gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dL_dR_ss_an  = dotAB(a_att_q3, gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(a_rel_q3, gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            std::printf("[ssl_smoother_dbg] dL/dR (analytic vs numeric): af %.6g vs %.6g | as %.6g vs %.6g | sf %.6g vs %.6g | ss %.6g vs %.6g\n",
                        dL_dR_af_an, dL_dR_af_num, dL_dR_as_an, dL_dR_as_num, dL_dR_sf_an, dL_dR_sf_num, dL_dR_ss_an, dL_dR_ss_num);
            // Also compare contraction (numeric Jacobians) vs fixed-mask scalar FD in rate space
            auto scalar_loss_fixed = [&](float Raf_v, float Ras_v, float Rsf_v, float Rss_v) -> float {
              DiscreteAB ABa = zoh_discretize_from_rates(Raf_v, Ras_v, Rsf_v, Rss_v, Ts);
              DiscreteAB ABr = zoh_discretize_from_rates(0.f, 0.f, Rsf_v, Rss_v, Ts);
              float x1f=0.f, x2f=0.f, Lf=0.f;
              for (int64_t tt = 0; tt < T; ++tt) {
                const float a = g_arr[tt];
                const bool is_att = (at_mask[tt] != 0);
                const DiscreteAB& ABf = is_att ? ABa : ABr;
                const float nx1f = ABf.Ad11 * x1f + ABf.Ad12 * x2f + ABf.Bd1 * a;
                const float nx2f = ABf.Ad21 * x1f + ABf.Ad22 * x2f + ABf.Bd2 * a;
                x1f = nx1f; x2f = nx2f;
                const float yf = x1f + x2f;
                Lf += gy_ptr[tt] * yf;
              }
              return Lf;
            };
            auto fd_rate = [&](float Raf_c, float Ras_c, float Rsf_c, float Rss_c, int which)->float{
              const float base[4] = {Raf_c, Ras_c, Rsf_c, Rss_c};
              float plus[4] = {Raf_c, Ras_c, Rsf_c, Rss_c};
              float minus[4] = {Raf_c, Ras_c, Rsf_c, Rss_c};
              const float q = base[which];
              float scale = (which == 2 || which == 3) ? 1e-4f : 1e-6f;
              float epsr = scale * ((std::fabs(q) > 1.f) ? std::fabs(q) : 1.f);
              plus[which] = base[which] + epsr;
              minus[which] = base[which] - epsr;
              float Lp = scalar_loss_fixed(plus[0], plus[1], plus[2], plus[3]);
              float Lm = scalar_loss_fixed(minus[0], minus[1], minus[2], minus[3]);
              return (Lp - Lm) / (2.f * epsr);
            };
            float dR_af_fd = fd_rate(Raf, Ras, Rsf, Rss, 0);
            float dR_as_fd = fd_rate(Raf, Ras, Rsf, Rss, 1);
            float dR_sf_fd = fd_rate(Raf, Ras, Rsf, Rss, 2);
            float dR_ss_fd = fd_rate(Raf, Ras, Rsf, Rss, 3);
            std::printf("[ssl_smoother_dbg] dL/dR (numeric contraction vs FD): af %.6g vs %.6g | as %.6g vs %.6g | sf %.6g vs %.6g | ss %.6g vs %.6g\n",
                        dL_dR_af_num, dR_af_fd, dL_dR_as_num, dR_as_fd, dL_dR_sf_num, dR_sf_fd, dL_dR_ss_num, dR_ss_fd);
            // Breakdown Ad-only and Bd-only contributions
            auto onlyAd = [](DiscreteAB d){ d.Bd1 = 0.f; d.Bd2 = 0.f; return d; };
            auto onlyBd = [](DiscreteAB d){ d.Ad11 = d.Ad12 = d.Ad21 = d.Ad22 = 0.f; return d; };
            float dR_af_num_Ad = dotAB(onlyAd(n_att_q0), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_af_an_Ad  = dotAB(onlyAd(a_att_q0), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_af_num_Bd = dotAB(onlyBd(n_att_q0), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_af_an_Bd  = dotAB(onlyBd(a_att_q0), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_as_num_Ad = dotAB(onlyAd(n_att_q1), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_as_an_Ad  = dotAB(onlyAd(a_att_q1), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_as_num_Bd = dotAB(onlyBd(n_att_q1), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_as_an_Bd  = dotAB(onlyBd(a_att_q1), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_sf_num_Ad = dotAB(onlyAd(n_att_q2), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(onlyAd(n_rel_q2), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_sf_an_Ad  = dotAB(onlyAd(a_att_q2), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(onlyAd(a_rel_q2), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_sf_num_Bd = dotAB(onlyBd(n_att_q2), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(onlyBd(n_rel_q2), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_sf_an_Bd  = dotAB(onlyBd(a_att_q2), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(onlyBd(a_rel_q2), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_ss_num_Ad = dotAB(onlyAd(n_att_q3), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(onlyAd(n_rel_q3), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_ss_an_Ad  = dotAB(onlyAd(a_att_q3), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(onlyAd(a_rel_q3), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_ss_num_Bd = dotAB(onlyBd(n_att_q3), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(onlyBd(n_rel_q3), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_ss_an_Bd  = dotAB(onlyBd(a_att_q3), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2)
                               + dotAB(onlyBd(a_rel_q3), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            std::printf("[ssl_smoother_dbg] Ad-only (an vs num):   af %.6g vs %.6g | as %.6g vs %.6g | sf %.6g vs %.6g | ss %.6g vs %.6g\n",
                        dR_af_an_Ad, dR_af_num_Ad, dR_as_an_Ad, dR_as_num_Ad, dR_sf_an_Ad, dR_sf_num_Ad, dR_ss_an_Ad, dR_ss_num_Ad);
            std::printf("[ssl_smoother_dbg] Bd-only (an vs num):   af %.6g vs %.6g | as %.6g vs %.6g | sf %.6g vs %.6g | ss %.6g vs %.6g\n",
                        dR_af_an_Bd, dR_af_num_Bd, dR_as_an_Bd, dR_as_num_Bd, dR_sf_an_Bd, dR_sf_num_Bd, dR_ss_an_Bd, dR_ss_num_Bd);
            // Attack-only vs Release-only Bd contributions per rate
            float dR_af_num_Bd_att = dotAB(onlyBd(n_att_q0), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_af_an_Bd_att  = dotAB(onlyBd(a_att_q0), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_af_num_Bd_rel = dotAB(onlyBd(n_rel_q0), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_af_an_Bd_rel  = dotAB(onlyBd(a_rel_q0), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);

            float dR_as_num_Bd_att = dotAB(onlyBd(n_att_q1), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_as_an_Bd_att  = dotAB(onlyBd(a_att_q1), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_as_num_Bd_rel = dotAB(onlyBd(n_rel_q1), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_as_an_Bd_rel  = dotAB(onlyBd(a_rel_q1), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);

            float dR_sf_num_Bd_att = dotAB(onlyBd(n_att_q2), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_sf_an_Bd_att  = dotAB(onlyBd(a_att_q2), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_sf_num_Bd_rel = dotAB(onlyBd(n_rel_q2), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_sf_an_Bd_rel  = dotAB(onlyBd(a_rel_q2), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);

            float dR_ss_num_Bd_att = dotAB(onlyBd(n_att_q3), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_ss_an_Bd_att  = dotAB(onlyBd(a_att_q3), gAd_att_11, gAd_att_12, gAd_att_21, gAd_att_22, gBd_att_1, gBd_att_2);
            float dR_ss_num_Bd_rel = dotAB(onlyBd(n_rel_q3), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);
            float dR_ss_an_Bd_rel  = dotAB(onlyBd(a_rel_q3), gAd_rel_11, gAd_rel_12, gAd_rel_21, gAd_rel_22, gBd_rel_1, gBd_rel_2);

            std::printf("[ssl_smoother_dbg] Bd-only ATT (an vs num): af %.6g vs %.6g | as %.6g vs %.6g | sf %.6g vs %.6g | ss %.6g vs %.6g\n",
                        dR_af_an_Bd_att, dR_af_num_Bd_att, dR_as_an_Bd_att, dR_as_num_Bd_att, dR_sf_an_Bd_att, dR_sf_num_Bd_att, dR_ss_an_Bd_att, dR_ss_num_Bd_att);
            std::printf("[ssl_smoother_dbg] Bd-only REL (an vs num): af %.6g vs %.6g | as %.6g vs %.6g | sf %.6g vs %.6g | ss %.6g vs %.6g\n",
                        dR_af_an_Bd_rel, dR_af_num_Bd_rel, dR_as_an_Bd_rel, dR_as_num_Bd_rel, dR_sf_an_Bd_rel, dR_sf_num_Bd_rel, dR_ss_an_Bd_rel, dR_ss_num_Bd_rel);

            if (const char* sbd = std::getenv("SSL_DEBUG_STEP_BD")) {
              bool sbd_on = (sbd[0]=='1'||sbd[0]=='t'||sbd[0]=='T'||sbd[0]=='y'||sbd[0]=='Y');
              if (sbd_on) {
                // Per-step Bd-only attack contributions, summed over steps, using analytic and numeric dBd
                double w1_sum = 0.0, w2_sum = 0.0;
                for (int64_t tt = 0; tt < T; ++tt) if (at_mask[tt]) {
                  const double a_t = static_cast<double>(g_arr[tt]);
                  const double l1_t = static_cast<double>(lam1_hist[tt]);
                  const double l2_t = static_cast<double>(lam2_hist[tt]);
                  w1_sum += l1_t * a_t;
                  w2_sum += l2_t * a_t;
                }
                auto bd_only = [&](const DiscreteAB& d)->std::pair<double,double>{ return {static_cast<double>(d.Bd1), static_cast<double>(d.Bd2)}; };
                auto dot_w = [&](std::pair<double,double> v)->double{ return v.first * w1_sum + v.second * w2_sum; };
                // Attack branch only
                double step_af_an = dot_w(bd_only(a_att_q0));
                double step_as_an = dot_w(bd_only(a_att_q1));
                double step_sf_an = dot_w(bd_only(a_att_q2));
                double step_ss_an = dot_w(bd_only(a_att_q3));
                double step_af_num = dot_w(bd_only(n_att_q0));
                double step_as_num = dot_w(bd_only(n_att_q1));
                double step_sf_num = dot_w(bd_only(n_att_q2));
                double step_ss_num = dot_w(bd_only(n_att_q3));
                std::printf("[ssl_smoother_dbg] Bd-only ATT step-sum (an vs num): af %.6g vs %.6g | as %.6g vs %.6g | sf %.6g vs %.6g | ss %.6g vs %.6g\n",
                            step_af_an, step_af_num, step_as_an, step_as_num, step_sf_an, step_sf_num, step_ss_an, step_ss_num);

                // Optional per-step trace of contributions for first up to 8 steps
                if (const char* ptr = std::getenv("SSL_DEBUG_PHI_TRACE")) {
                  bool trace_on = (ptr[0]=='1'||ptr[0]=='t'||ptr[0]=='T'||ptr[0]=='y'||ptr[0]=='Y');
                  if (trace_on) {
                    // Build attack-only Bd (analytic and numeric) vectors per rate
                    auto bd_pair = [&](const DiscreteAB& d){ return std::array<double,2>{static_cast<double>(d.Bd1), static_cast<double>(d.Bd2)}; };
                    auto A_bd_af_an = bd_pair(a_att_q0);
                    auto A_bd_as_an = bd_pair(a_att_q1);
                    auto A_bd_sf_an = bd_pair(a_att_q2);
                    auto A_bd_ss_an = bd_pair(a_att_q3);
                    auto N_bd_af = bd_pair(n_att_q0);
                    auto N_bd_as = bd_pair(n_att_q1);
                    auto N_bd_sf = bd_pair(n_att_q2);
                    auto N_bd_ss = bd_pair(n_att_q3);
                    int printed = 0;
                    for (int64_t tt = 0; tt < T && printed < 8; ++tt) {
                      if (!at_mask[tt]) continue;
                      const double a_t = static_cast<double>(g_arr[tt]);
                      const double l1_t = static_cast<double>(lam1_hist[tt]);
                      const double l2_t = static_cast<double>(lam2_hist[tt]);
                      auto contrib = [&](const std::array<double,2>& v){ return l1_t * a_t * v[0] + l2_t * a_t * v[1]; };
                      double c_af_an = contrib(A_bd_af_an), c_af_num = contrib(N_bd_af);
                      double c_as_an = contrib(A_bd_as_an), c_as_num = contrib(N_bd_as);
                      double c_sf_an = contrib(A_bd_sf_an), c_sf_num = contrib(N_bd_sf);
                      double c_ss_an = contrib(A_bd_ss_an), c_ss_num = contrib(N_bd_ss);
                      std::printf("[ssl_phi_trace] t=%lld a_t=%.6g l1=%.6g l2=%.6g | af an/num %.6g/%.6g | as %.6g/%.6g | sf %.6g/%.6g | ss %.6g/%.6g\n",
                                  (long long)tt, a_t, l1_t, l2_t,
                                  c_af_an, c_af_num, c_as_an, c_as_num, c_sf_an, c_sf_num, c_ss_an, c_ss_num);
                      printed++;
                    }
                  }
                }
              }
            }
          }
        }
      }


    // Chain rule from rates to time constants: R = 1/T => dL/dT = -dL/dR / T^2
      gT_af.data_ptr<float>()[b] = - dL_dR_af / (Taf * Taf);
      gT_as.data_ptr<float>()[b] = - dL_dR_as / (Tas * Tas);
      gT_sf.data_ptr<float>()[b] = - dL_dR_sf / (Tsf * Tsf);
      gT_ss.data_ptr<float>()[b] = - dL_dR_ss / (Tss * Tss);

      }

    else {
      // Finite-difference gradients for time constants using scalar loss L = sum_t grad_y[t] * y[t]
      // Optionally use a fixed-mask replay consistent with the analytic path.
      bool fixed_mask = false;
      if (const char* fm = std::getenv("SSL_TCONST_FD_FIXED_MASK")) {
        if (fm[0] == '1' || fm[0] == 't' || fm[0] == 'T' || fm[0] == 'y' || fm[0] == 'Y') fixed_mask = true;
      }

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
        for (int64_t t = 0; t < T; t += subsample) {
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
      if (!fixed_mask) {
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
      } else {
        // Fixed-mask FD: do FD in rate domain with recorded at_mask/g_arr and chain to T.
        auto scalar_loss_fixed = [&](float Raf_v, float Ras_v, float Rsf_v, float Rss_v) -> float {
          DiscreteAB ABa = zoh_discretize_from_rates(Raf_v, Ras_v, Rsf_v, Rss_v, Ts);
          DiscreteAB ABr = zoh_discretize_from_rates(0.f, 0.f, Rsf_v, Rss_v, Ts);
          float x1f=0.f, x2f=0.f, Lf=0.f;
          for (int64_t t = 0; t < T; ++t) {
            const float a = g_arr[t];
            const bool is_attack = (at_mask[t] != 0);
            const DiscreteAB& ABf = is_attack ? ABa : ABr;
            const float nx1f = ABf.Ad11 * x1f + ABf.Ad12 * x2f + ABf.Bd1 * a;
            const float nx2f = ABf.Ad21 * x1f + ABf.Ad22 * x2f + ABf.Bd2 * a;
            x1f = nx1f; x2f = nx2f;
            const float yf = x1f + x2f;
            Lf += gy_ptr[t] * yf;
          }
          return Lf;
        };
        auto fd_rate = [&](float Raf_c, float Ras_c, float Rsf_c, float Rss_c, int which)->float{
          const float base[4] = {Raf_c, Ras_c, Rsf_c, Rss_c};
          float plus[4] = {Raf_c, Ras_c, Rsf_c, Rss_c};
          float minus[4] = {Raf_c, Ras_c, Rsf_c, Rss_c};
          const float q = base[which];
          // Use a larger epsilon for shunt_fast (which==2) and shunt_slow (which==3)
          // to avoid float32 cancellation
          float scale = (which == 2 || which == 3) ? 1e-4f : 1e-6f;
          float epsr = scale * ((std::fabs(q) > 1.f) ? std::fabs(q) : 1.f);
          plus[which] = base[which] + epsr;
          minus[which] = base[which] - epsr;
          float Lp = scalar_loss_fixed(plus[0], plus[1], plus[2], plus[3]);
          float Lm = scalar_loss_fixed(minus[0], minus[1], minus[2], minus[3]);
          return (Lp - Lm) / (2.f * epsr);
        };
        const float dR_af = fd_rate(Raf, Ras, Rsf, Rss, 0);
        const float dR_as = fd_rate(Raf, Ras, Rsf, Rss, 1);
        const float dR_sf = fd_rate(Raf, Ras, Rsf, Rss, 2);
        const float dR_ss = fd_rate(Raf, Ras, Rsf, Rss, 3);
        gT_af.data_ptr<float>()[b] = - dR_af / (Taf * Taf);
        gT_as.data_ptr<float>()[b] = - dR_as / (Tas * Tas);
        gT_sf.data_ptr<float>()[b] = - dR_sf / (Tsf * Tsf);
        gT_ss.data_ptr<float>()[b] = - dR_ss / (Tss * Tss);
      }
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
  m.def("dbg_dab_analytic", [](float series_fast, float series_slow, float shunt_fast, float shunt_slow, float Ts){
      DiscreteAB q0,q1,q2,q3;
      analytic_dAB_drates(series_fast, series_slow, shunt_fast, shunt_slow, Ts, q0,q1,q2,q3);
      auto out = torch::empty({4,6}, torch::TensorOptions().dtype(torch::kFloat));
      auto acc = out.accessor<float,2>();
      DiscreteAB arr[4] = {q0,q1,q2,q3};
      for (int i=0;i<4;++i){ acc[i][0]=arr[i].Ad11; acc[i][1]=arr[i].Ad12; acc[i][2]=arr[i].Ad21; acc[i][3]=arr[i].Ad22; acc[i][4]=arr[i].Bd1; acc[i][5]=arr[i].Bd2; }
      return out;
    }, "Debug: analytic d(Ad,Bd)/drates");
  m.def("dbg_dab_numeric", [](float series_fast, float series_slow, float shunt_fast, float shunt_slow, float Ts){
      DiscreteAB q0,q1,q2,q3;
      numeric_dAB_drates(series_fast, series_slow, shunt_fast, shunt_slow, Ts, q0,q1,q2,q3);
      auto out = torch::empty({4,6}, torch::TensorOptions().dtype(torch::kFloat));
      auto acc = out.accessor<float,2>();
      DiscreteAB arr[4] = {q0,q1,q2,q3};
      for (int i=0;i<4;++i){ acc[i][0]=arr[i].Ad11; acc[i][1]=arr[i].Ad12; acc[i][2]=arr[i].Ad21; acc[i][3]=arr[i].Ad22; acc[i][4]=arr[i].Bd1; acc[i][5]=arr[i].Bd2; }
      return out;
    }, "Debug: numeric d(Ad,Bd)/drates");
  m.def("dbg_attack_dbd_compare", [](double Raf, double Ras, double Rsf, double Rss, double Ts){
      // Build base A,B
      float a11 = -(float)(Raf + Rsf);
      float a12 = -(float)Raf;
      float a21 = -(float)Ras;
      float a22 = -(float)(Ras + Rss);
      float b1  = (float)Raf;
      float b2  = (float)Ras;
      // Helper: compute Bd via zoh
      auto Bd_from = [&](double Raf_v,double Ras_v,double Rsf_v,double Rss_v){
        DiscreteAB AB = zoh_discretize_from_rates((float)Raf_v,(float)Ras_v,(float)Rsf_v,(float)Rss_v,(float)Ts);
        return std::array<float,2>{AB.Bd1, AB.Bd2};
      };
      // Analytic dBd via robust linear-solve formulation per rate
      auto dBd_phi_rate = [&](int which){
        float dA11=0, dA12=0, dA21=0, dA22=0, dB1=0, dB2=0;
        if (which==0){ dA11=-1.f; dA12=-1.f; dB1=1.f; }
        else if (which==1){ dA21=-1.f; dA22=-1.f; dB2=1.f; }
        else if (which==2){ dA11=-1.f; }
        else { dA22=-1.f; }
        double A11d=a11, A12d=a12, A21d=a21, A22d=a22, Ts_d=Ts;
        // Ad and Y
        double Ad11,Ad12,Ad21,Ad22; expm2x2_double(Ts_d*A11d,Ts_d*A12d,Ts_d*A21d,Ts_d*A22d, Ad11,Ad12,Ad21,Ad22);
        double Y11=Ad11-1.0,Y12=Ad12,Y21=Ad21,Y22=Ad22-1.0;
        // F via solves
        double F11,F21,F12,F22;
        if (!solve2x2_double(A11d,A12d,A21d,A22d, Y11,Y21, F11,F21)) { F11 = Ts_d; F21 = 0.0; }
        if (!solve2x2_double(A11d,A12d,A21d,A22d, Y12,Y22, F12,F22)) { F12 = 0.0; F22 = Ts_d; }
        // dAd
        double dAd11,dAd12,dAd21,dAd22;{
          bool use_fd_ad=false; if (const char* t=getenv("SSL_AD_USE_FD")) use_fd_ad=(t[0]=='1'||t[0]=='t'||t[0]=='T'||t[0]=='y'||t[0]=='Y');
          if (!use_fd_ad) {
            frechet_expm_2x2_double(A11d,A12d,A21d,A22d,dA11,dA12,dA21,dA22,Ts_d,dAd11,dAd12,dAd21,dAd22);
          } else {
            double eps=1e-6;
            double Ap11=A11d+eps*dA11, Ap12=A12d+eps*dA12, Ap21=A21d+eps*dA21, Ap22=A22d+eps*dA22;
            double Am11=A11d-eps*dA11, Am12=A12d-eps*dA12, Am21=A21d-eps*dA21, Am22=A22d-eps*dA22;
            double Ep11,Ep12,Ep21,Ep22, Em11,Em12,Em21,Em22;
            expm2x2_double(Ts_d*Ap11,Ts_d*Ap12,Ts_d*Ap21,Ts_d*Ap22, Ep11,Ep12,Ep21,Ep22);
            expm2x2_double(Ts_d*Am11,Ts_d*Am12,Ts_d*Am21,Ts_d*Am22, Em11,Em12,Em21,Em22);
            dAd11=(Ep11-Em11)/(2*eps); dAd12=(Ep12-Em12)/(2*eps);
            dAd21=(Ep21-Em21)/(2*eps); dAd22=(Ep22-Em22)/(2*eps);
          }
        }
        // R = dAd - dA F
        double dAF11 = dA11*F11 + dA12*F21;
        double dAF12 = dA11*F12 + dA12*F22;
        double dAF21 = dA21*F11 + dA22*F21;
        double dAF22 = dA21*F12 + dA22*F22;
        double R11 = dAd11 - dAF11;
        double R12 = dAd12 - dAF12;
        double R21 = dAd21 - dAF21;
        double R22 = dAd22 - dAF22;
        // dF via solves
        double dF11,dF21,dF12,dF22;
        if (!solve2x2_double(A11d,A12d,A21d,A22d, R11,R21, dF11,dF21)) { dF11=0.0; dF21=0.0; }
        if (!solve2x2_double(A11d,A12d,A21d,A22d, R12,R22, dF12,dF22)) { dF12=0.0; dF22=0.0; }
        // dBd
        double dFB1 = dF11*b1 + dF12*b2;
        double dFB2 = dF21*b1 + dF22*b2;
        double FdB1 = F11*dB1 + F12*dB2;
        double FdB2 = F21*dB1 + F22*dB2;
        return std::array<double,2>{dFB1 + FdB1, dFB2 + FdB2};
      };
      // Finite-difference on Bd for each rate (double-precision Bd baseline)
      auto dBd_fd_rate = [&](int which){
        auto Bd_from_d = [&](double r0,double r1,double r2,double r3){
          double A11d = -(r0 + r2), A12d = -r0, A21d = -r1, A22d = -(r1 + r3);
          double B1d = r0, B2d = r1;
          double Ad11,Ad12,Ad21,Ad22; expm2x2_double(Ts*A11d,Ts*A12d,Ts*A21d,Ts*A22d, Ad11,Ad12,Ad21,Ad22);
          double rhs1 = (Ad11 - 1.0) * B1d + Ad12 * B2d;
          double rhs2 = Ad21 * B1d + (Ad22 - 1.0) * B2d;
          double Bd1, Bd2;
          if (!solve2x2_double(A11d,A12d,A21d,A22d, rhs1,rhs2, Bd1,Bd2)) {
            double AB1 = A11d*B1d + A12d*B2d, AB2 = A21d*B1d + A22d*B2d;
            Bd1 = Ts*B1d + 0.5*Ts*Ts*AB1; Bd2 = Ts*B2d + 0.5*Ts*Ts*AB2;
          }
          return std::array<double,2>{Bd1,Bd2};
        };
        double q_base[4] = {Raf,Ras,Rsf,Rss};
        double plus[4] = {Raf,Ras,Rsf,Rss};
        double minus[4]= {Raf,Ras,Rsf,Rss};
        double q = q_base[which];
        double eps = ((which==2||which==3)?1e-4:1e-6) * ((std::fabs(q)>1.0)?std::fabs(q):1.0);
        plus[which] += eps; minus[which] -= eps;
        auto P = Bd_from_d(plus[0],plus[1],plus[2],plus[3]);
        auto M = Bd_from_d(minus[0],minus[1],minus[2],minus[3]);
        return std::array<double,2>{(P[0]-M[0])/(2.0*eps), (P[1]-M[1])/(2.0*eps)};
      };
      auto out = torch::empty({4,4}, torch::TensorOptions().dtype(torch::kFloat64));
      auto acc = out.accessor<double,2>();
      for (int i=0;i<4;++i){
        auto an = dBd_phi_rate(i);
        auto fd = dBd_fd_rate(i);
        acc[i][0]=an[0]; acc[i][1]=an[1]; acc[i][2]=fd[0]; acc[i][3]=fd[1];
      }
      return out.to(torch::kFloat32);
    }, "Debug: attack dBd analytic vs FD per rate (columns: an_Bd1, an_Bd2, fd_Bd1, fd_Bd2)");

  // Removed legacy per-term breakdown debug (dbg_attack_dbd_terms) to reduce surface area

  // Internal: compute Bd in double precision from rates
  auto bd_from_rates_double = [](double Raf, double Ras, double Rsf, double Rss, double Ts){
    double A11 = -(Raf + Rsf), A12 = -Raf, A21 = -Ras, A22 = -(Ras + Rss);
    double B1 = Raf, B2 = Ras;
    double Ad11,Ad12,Ad21,Ad22; expm2x2_double(Ts*A11,Ts*A12,Ts*A21,Ts*A22, Ad11,Ad12,Ad21,Ad22);
    double rhs1 = (Ad11 - 1.0) * B1 + Ad12 * B2;
    double rhs2 = Ad21 * B1 + (Ad22 - 1.0) * B2;
    double Bd1, Bd2; if (!solve2x2_double(A11,A12,A21,A22, rhs1,rhs2, Bd1,Bd2)) {
      // Series fallback
      double AB1 = A11*B1 + A12*B2; double AB2 = A21*B1 + A22*B2;
      Bd1 = Ts*B1 + 0.5*Ts*Ts*AB1; Bd2 = Ts*B2 + 0.5*Ts*Ts*AB2;
    }
    return std::array<double,2>{Bd1,Bd2};
  };

  // Debug: return Bd from rates (float32)
  m.def("dbg_bd_from_rates", [bd_from_rates_double](double Raf, double Ras, double Rsf, double Rss, double Ts){
      auto BD = bd_from_rates_double(Raf,Ras,Rsf,Rss,Ts);
      auto out = torch::empty({2}, torch::TensorOptions().dtype(torch::kFloat));
      auto acc = out.accessor<float,1>();
      acc[0] = static_cast<float>(BD[0]); acc[1] = static_cast<float>(BD[1]); return out;
    }, "Debug: Bd from (Raf,Ras,Rsf,Rss,Ts)");

  // Debug: FD dBd at a specific epsilon (double precision)
  m.def("dbg_dbd_fd_with_eps", [bd_from_rates_double](double Raf, double Ras, double Rsf, double Rss, double Ts, int which, double eps){
      double plus[4] = {Raf,Ras,Rsf,Rss};
      double minus[4]= {Raf,Ras,Rsf,Rss};
      plus[which] += eps; minus[which] -= eps;
      auto P = bd_from_rates_double(plus[0],plus[1],plus[2],plus[3],Ts);
      auto M = bd_from_rates_double(minus[0],minus[1],minus[2],minus[3],Ts);
      auto out = torch::empty({2}, torch::TensorOptions().dtype(torch::kFloat64));
      auto acc = out.accessor<double,1>();
      acc[0] = (P[0]-M[0])/(2.0*eps);
      acc[1] = (P[1]-M[1])/(2.0*eps);
      return out.to(torch::kFloat32);
    }, "Debug: FD dBd with custom epsilon (which: 0=Raf,1=Ras,2=Rsf,3=Rss)");

  // Debug: sweep eps over decades for FD dBd stability (returns [eps, dBd1, dBd2] rows)
  m.def("dbg_dbd_fd_sweep", [bd_from_rates_double](double Raf, double Ras, double Rsf, double Rss, double Ts,
                                int which, double eps0, int num_decades, int steps_per_decade){
      int N = num_decades * steps_per_decade + 1;
      auto out = torch::empty({N,3}, torch::TensorOptions().dtype(torch::kFloat64));
      auto acc = out.accessor<double,2>();
      for (int i=0;i<N;++i){
        double factor = std::pow(10.0, (double)i / (double)steps_per_decade);
        double eps = eps0 * factor;
        double plus[4] = {Raf,Ras,Rsf,Rss};
        double minus[4]= {Raf,Ras,Rsf,Rss};
        plus[which] += eps; minus[which] -= eps;
        auto P = bd_from_rates_double(plus[0],plus[1],plus[2],plus[3],Ts);
        auto M = bd_from_rates_double(minus[0],minus[1],minus[2],minus[3],Ts);
        double d1 = (P[0]-M[0])/(2.0*eps);
        double d2 = (P[1]-M[1])/(2.0*eps);
        acc[i][0] = eps; acc[i][1] = d1; acc[i][2] = d2;
      }
      return out.to(torch::kFloat32);
    }, "Debug: FD dBd sweep over eps (columns: eps, dBd1, dBd2)");
}


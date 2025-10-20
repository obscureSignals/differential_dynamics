# Analytic dBd for the SSL 2‑state smoother: plan, criteria, evidence, and math

This note documents the ongoing effort to make the analytic Jacobian of the ZOH discretization (Ad, Bd) with respect to the time‑constant rates match finite differences (FD) for the SSL 2‑state smoother attack branch. It is written both for future reference and to keep the debugging tight and falsifiable.


## What we are trying to do and why

Goal: Implement a numerically robust, correct analytic derivative of Bd (and Ad) with respect to the four rate parameters R = [Raf, Ras, Rsf, Rss] for the 2×2 continuous system under Zero‑Order Hold (ZOH) discretization:

- Continuous time:
  - A = [[-(Raf + Rsf),  -Raf], [ -Ras,  -(Ras + Rss)]]
  - B = [[Raf], [Ras]]
- Discrete time (per sample Ts = 1/fs):
  - Ad = exp(A Ts)
  - Bd = ∫₀^{Ts} exp(τ A) B dτ

We need dBd/dR and dAd/dR for the hard‑gate (attack/release) branches to make the autograd backward stable and accurate. Correct analytic derivatives let us avoid FD in the training loop (less noise, faster) while retaining fidelity comparable to the FD “ground truth.”

This matters because the smoother’s gradients propagate into time‑constant parameters, and mismatched derivatives lead to poor convergence or divergence in training.


## How we will know we have done it

We declare success when the following conditions are met for the attack branch (release is already consistent):

1. Raw derivative parity (most important): for each rate q ∈ {Raf, Ras, Rsf, Rss}, the analytic dBd(q) matches Bd FD computed directly on Bd via zoh_discretize_from_rates with a central difference. Concretely,
   - ||dBd_analytic(q) − dBd_FD(q)||∞ ≤ atol + rtol · max(1, ||dBd_FD(q)||∞)
   - With float32 tolerances; e.g., atol ~ 5e-6, rtol ~ 1e-2 is realistic given sums over T.

2. Contraction parity: the adjoint-weighted contraction of (dAd, dBd) against recorded per-step weights equals both:
   - the contraction using numeric operator Jacobians, and
   - the scalar FD on the loss with a fixed mask.

3. Unit test passes in analytic mode:
   - tests/test_ssl_smoother_backward.py::test_hard_gate_backward_small with SSL_USE_ANALYTIC_JAC_BD=1

4. Training sanity: with analytic mode enabled, training curves behave comparably to the FD baseline (within expected stochastic noise). This is optional as a criterion; the raw and contraction checks are decisive.


## What we think we know and evidence

Current evidence (from kernel debug prints and unit tests):

- Bd itself is correct: Bd(phi) = Bd(exact ZOH) to float precision on the attack branch.
- The mismatch is exclusively in dBd for the attack branch rates — it persists even when:
  - Using double-precision Frechet and solves,
  - Using an integral‑based analytic formula,
  - Using a block‑Frechet (6×6) formulation on the lifted 3×3 M = [[A, B],[0, 0]] to produce dF·B + F·dB directly.
- The mismatch appears at the raw derivative level (analytic dBd vs Bd FD), before any adjoint contraction. Therefore it is not due to the adjoint chain or event sensitivity; it is squarely in the operator derivative mapping used for attack A,B.
- Ad Frechet vs Ad FD at float32 is noisy at tiny Ts; float64 tests show the Frechet path is correct for Ad. This supports that dAd is not the root cause of the large dBd mismatch.

Interpretation: The math is correct in isolation, but the in‑kernel mapping from rate directions to (dA, dB) in the derivative chain used for Bd is inconsistent with the mapping embodied by zoh_discretize_from_rates for the attack branch (or the exact ZOH lifting used in the block construction doesn’t mirror the zoh path’s conventions).


## The math (practical derivations)

Let Ts = 1/fs.

- ZOH discretization:
  - Ad = exp(A Ts)
  - Bd = ∫₀^{Ts} exp(τ A) B dτ = Φ₁(A Ts) · B
  - Φ₁(Z) = ∑_{k≥0} Z^{k} / (k+1)! (matrix phi function)
  - Equivalent identity (when A is invertible): Φ₁(A Ts) = A⁻¹ (Ad − I)

- Derivative of Ad (Frechet):
  - dAd = L_exp(A Ts)[Ts·dA]
  - Numerically, via a block exponential (2n×2n):
    - Let H = [[A Ts, Ts·dA], [0, A Ts]]
    - Then exp(H) = [[Ad, L_exp(A Ts)[Ts dA]], [0, Ad]]

- Derivative of Bd:
  - dBd = ∫₀^{Ts} L_exp(τ A)[τ·dA] B dτ + ∫₀^{Ts} exp(τ A) dB dτ
  - Using Φ₁: set F = Φ₁(A Ts) (so Bd = F·B). Differentiate:
    - A·F = Ad − I  (Sylvester-like relation)
    - Differentiate: A·dF + dA·F = dAd
    - Solve the 2×2 linear system for each column of dF
    - Then dBd = dF·B + F·dB

- Alternative block‑lift (for verification and a constructive implementation):
  - Define M = [[A, B],[0, 0]] (3×3). Then exp(M Ts) has the structure:
    - exp(M Ts) = [[Ad, F·B], [0, I]]  with F = Φ₁(A Ts)
  - The Frechet derivative at M along dM = [[dA, dB],[0, 0]] gives:
    - Upper‑right block of exp([[M,dM],[0,M]] Ts) equals d(F·B) = dF·B + F·dB = dBd
  - This block construction is mathematically exact and is a good way to compute dBd in practice.

- 2×2 closed form (basis used by expm2x2):
  - Let tr = A₁₁ + A₂₂, μ = tr/2, det = A₁₁A₂₂ − A₁₂A₂₁, s² = μ² − det
  - exp(A Ts) = e^{μ Ts} [cosh(s Ts) I + (sinh(s Ts)/s) (A − μ I)]
  - Φ₁(A Ts) has a corresponding basis expression (obtainable by integrating the series or via block-lift). Differentiating Φ₁ in this basis yields a closed form for dΦ₁ that is consistent with expm2x2’s parametrization.


## Debug apparatus in the kernel

The kernel exposes several toggles to make this falsifiable:

- SSL_DEBUG_BD_RAW=1: at the start of backward for b=0 (attack branch), print
  - Bd(phi/exact) parity
  - dBd per rate (analytic vs Bd FD on zoh_discretize_from_rates)
- SSL_DEBUG_AD_RAW=1: print dAd per rate (Frechet vs Ad FD) — informative only; Ad FD at f32 is tiny and can be zero.
- SSL_DEBUG_STEP_BD=1 and SSL_DEBUG_PHI_TRACE=1: print first few attack steps with a_t, λ, and per‑step contributions for each rate (analytic vs numeric) and the step‑sum.

These diagnostics proved that the mismatch is at the raw dBd level, not caused by adjoint contraction.


## Implementation plan (to completion)

1) Keep the block‑Frechet implementation for dBd (via expmN_double on W = [[M, dM],[0, M]] Ts). This is mathematically exact. Ensure that:
   - We construct M exactly as the zoh path implies (ordering and Ts scaling),
   - We take dBd from the correct indices (upper‑right (0..1, 2) block),
   - We use the exact same (dA, dB) mapping per rate as zoh_discretize_from_rates FD.

2) For transparency and performance, introduce a 2×2 closed‑form dΦ₁ in the same μ/s/κ basis as expm2x2, and compute dBd = dΦ₁·B + Φ₁·dB. Validate this closed‑form equals the block‑Frechet and FD at float32.

3) Success criteria (see above) — all three parity checks pass. Then enable the analytic path by default and remove FD fallbacks from training runs.


## Practical tolerances

- For raw Bd derivative parity: atol ~ 5e-6, rtol ~ 1e-2 (float32, single-frame checks).
- For scalar loss FD parity (sum over T with masks): atol ~ 5e-3, rtol ~ 5e-2; time-constant gradients allow rtol up to ~2e-1 due to clamp subgradients and O(T) accumulation.


## Known pitfalls and mitigations

- Float32 sensitivity at tiny Ts: prefer double internally for Frechet/block‑expm solves; cast at the end.
- M/dM layout drift: the block construction MUST mirror the exact way zoh discretization composes A and B under ZOH. Testing with raw Bd and Bd FD is the right falsification point.
- Rate‑direction mapping: Raf→(dA11=-1,dA12=-1; dB=[1,0]), Ras→(dA21=-1,dA22=-1; dB=[0,1]), Rsf→(dA11=-1), Rss→(dA22=-1). Ensure this is used consistently across all implementations (numeric, integral, block, closed‑form).


## Summary (before fix)

- We must match FD at the raw derivative level (dBd vs Bd FD) for the attack branch; that’s our primary correctness test.
- Bd itself is already correct; the remaining issue is dBd mapping for the attack branch.
- We have robust diagnostic prints and a mathematically exact block‑Frechet path. The next steps reconcile the M/dM construction with the zoh path to close the gap, then replace it with a performant, closed‑form 2×2 basis derivative consistent with expm2x2.

---

# 2025‑10‑12 Resolution: Analytic dBd matches double‑precision FD; Frechet fixed

This section captures the exact changes and mathematics that brought analytic dBd into parity with a double‑precision finite‑difference (FD) baseline, and corrected the Frechet derivative used for dAd.

## What we tried (and why it failed)

1) Direct closed‑form via Φ₁(Z) = Z⁻¹ (e^Z − I)
- Built Φ₁(A Ts) and differentiated it using linear solves with dE = L_exp(A Ts)[Ts dA].
- Multiple variants were attempted (left‑ and right‑associated inverse orderings, sign checks on (Ad − I) vs (I − Ad)).
- Despite careful ordering, analytic dBd disagreed with Bd FD for attack rates, while Bd itself matched perfectly.

2) Block‑lift Frechet on 3×3 M = [[A, B],[0, 0]]
- Theoretically exact for d(F B) = dF B + F dB.
- Our early versions diverged from Bd FD, pointing to a mismatch in how (dA, dB) were mapped from rate directions in the discretization path compared to the block path.

3) Integral (Gauss–Legendre) over τ for dBd
- Implemented ∫₀^{Ts} (L_exp(τA)[τ dA] B + exp(τA) dB) dτ.
- This also disagreed at the raw dBd level for attack rates. Later we discovered the numeric Bd FD baseline we used for comparison was float32 and too coarse, masking or exaggerating errors depending on eps.

Conclusion from 1–3: The formulas themselves are sound, but small policy differences (inverse ordering, sign, or FD baseline quality) hid the true culprit and misled diagnostics.

## Two decisive changes that solved it

A) ZOH‑consistent linear‑solve formulation for dBd (robust and simple)
- Use the Sylvester‑like identity A F = Ad − I to compute F = Φ₁(A Ts) by solves:
  - Solve A F = (Ad − I) for each column (2×2 solve)
- Differentiate A F = Ad − I:
  - A dF + dA F = dAd
  - Solve A dF = dAd − dA F for each column
- Then dBd = dF B + F dB
- This matches the ZOH definition exactly and avoids ambiguity about inverse ordering.

B) Fix Frechet dAd by using a quadrature definition (12‑point Gauss–Legendre)
- Replace the block 4×4 expm Frechet for 2×2 with a robust quadrature implementation:
  - dAd = ∫₀^{Ts} exp((Ts−t) A) (dA) exp(t A) dt, numerically via 12‑point Gauss–Legendre.
  - Implementation uses expm2x2_double, all in double precision; results cast to float at the end.
- This yields small, stable dAd that leads to the correct small dBd for attack rates (~2e−5 in our reference case), instead of the spurious ~3e−4 magnitudes observed earlier.

With these two changes, analytic dBd now matches a double‑precision Bd FD baseline across all four rates within float32 tolerance.

## Verified results (reference case)

- Parameters: Raf=200.0, Ras=50.0, Rsf=10.0, Rss=1.0, fs=48 kHz (Ts ≈ 2.0833e−5)
- Analytic vs FD (both in double internally; reported as float):
  - R_af: an [+2.0734e−05, −1.0812e−08] vs fd [+2.0734e−05, −1.0805e−08]
  - R_as: an [−4.3249e−08, +2.0768e−05] vs fd [−4.3238e−08, +2.0768e−05]
  - R_sf: an [−4.3261e−08, +1.5030e−11] vs fd [−4.3262e−08, +1.5344e−11]
  - R_ss: an [+1.5030e−11, −1.0828e−08] vs fd [+1.1514e−11, −1.0825e−08]

## Implementation details (as shipped)

- Analytic dBd path (phi): ZOH‑consistent linear‑solve formulation:
  - Compute Ad via expm2x2_double
  - Solve A F = (Ad − I)
  - Compute dAd via Frechet quadrature (12‑point Gauss–Legendre), then solve A dF = dAd − dA F
  - dBd = dF B + F dB
- Analytic dAd path (Frechet): 12‑point Gauss–Legendre quadrature (default)
  - Optional debug: set SSL_AD_USE_FD=1 to compute dAd via finite differences instead of Frechet
- Numeric Bd FD (baseline for debug): uses double‑precision Bd (expm2x2_double + solve2x2_double) when comparing raw dBd against FD

### 2025‑10‑13 Cleanup and public controls

- The analytic dBd backend is hardwired to the phi linear‑solve method; legacy integral and inverse variants were removed.
- Two supported gradient modes in the kernel:
  - Analytic operator‑Jacobian contraction (fixed mask, subgradient): uses phi linear‑solve for dBd and Frechet quadrature for dAd.
  - Scalar‑loss finite difference (FD) for time‑constant gradients: event‑sensitive by default; optionally fixed‑mask replay.
- Trainer interface: scripts/train_param_recovery_ssl.py exposes --solver={analytic,fd}
  - --solver fd: sets SSL_USE_FD_TCONST_GRADS=1 and unsets analytic toggles.
  - --solver analytic: sets SSL_USE_FD_TCONST_GRADS=0 and SSL_USE_ANALYTIC_JAC=1.
- Note on event sensitivity: the analytic path is fixed‑mask (no event sensitivity to gate flips). FD with variable mask captures event sensitivity; FD with SSL_TCONST_FD_FIXED_MASK=1 replays a fixed mask consistent with the analytic subgradient.

## Debug utilities exposed by the extension

- dbg_attack_dbd_compare(Raf,Ras,Rsf,Rss,Ts): returns a 4×4 matrix with per‑rate columns [an_Bd1, an_Bd2, fd_Bd1, fd_Bd2]; FD uses a double‑precision Bd baseline
- dbg_bd_from_rates(Raf,Ras,Rsf,Rss,Ts): Bd in float32 (computed in double internally)
- dbg_dbd_fd_with_eps(Raf,Ras,Rsf,Rss,Ts, which, eps): FD dBd at custom epsilon (double precision)
- dbg_dbd_fd_sweep(Raf,Ras,Rsf,Rss,Ts, which, eps0, decades, steps_per_decade): rows [eps, dBd1, dBd2] to locate stable FD plateaus

Removed during cleanup:
- dbg_attack_dbd_terms: per‑term breakdown removed to reduce surface area; retained higher‑level compares above

## Environment toggles (current)

Build and caching:
- SSL_SMOOTHER_DEBUG=1, SSL_SMOOTHER_FORCE_REBUILD=1

Gradient mode selection:
- SSL_USE_FD_TCONST_GRADS=1: use scalar‑loss FD for time‑constant gradients (default 0 = analytic operator Jacobians)
- SSL_TCONST_FD_FIXED_MASK=1: replay fixed mask in FD mode (default 0 = event‑sensitive)
- SSL_TCONST_FD_SUBSAMPLE=N: subsample steps in FD scalar‑loss evaluation (default 1)

Analytic operator Jacobians (debug mixing):
- SSL_USE_ANALYTIC_JAC=1: master toggle to enable analytic operator Jacobians for both Ad and Bd
- SSL_USE_ANALYTIC_JAC_AD=0|1, SSL_USE_ANALYTIC_JAC_BD=0|1: override per‑operator to mix numeric/analytic for A/B
- SSL_AD_USE_FD=1: compute dAd via finite differences instead of Frechet (debug only)

Saltation (hard gate):
- SSL_USE_SALTATION=1: enable saltation corrections at gate flips
- SSL_SALTATION_EPS_REL, SSL_SALTATION_MAX_BACKOFF: numeric knobs for dα/d ln T
- SSL_SALTATION_MAX_FLIPS: limit processed flips per backward
- SSL_SALTATION_LOG_SUMMARY=1: one‑line smooth vs salt totals per batch 0
- SSL_SALTATION_LOG_EVERY, SSL_SALTATION_LOG_MAX: throttle per‑flip logs
- SSL_SALTATION_SIGN_TEST=1: invert sign of saltation to sanity‑check conventions

Debug printing:
- SSL_DEBUG_BD_RAW=1, SSL_DEBUG_AD_RAW=1, SSL_DEBUG_STEP_BD=1, SSL_DEBUG_PHI_TRACE=1
- SSL_DEBUG_ONEHOT_T=t: print per‑step analytic decomposition and one‑hot FD at timestep t

Notes:
- The analytic dBd backend is hardwired to the phi linear‑solve. SSL_ANALYTIC_BD_METHOD and SSL_FRECHET_METHOD are ignored.

## Takeaways

- Always validate at the raw operator level (dBd vs Bd FD) in double precision; float32 FD can be misleading at tiny eps.
- ZOH‑consistent linear‑solve for dBd is simple, fast for 2×2, and unambiguous.
- Quadrature‑based Frechet is robust and matches FD in practice; keep block‑expm as an optional fallback for experiments.

---

# 2025‑10‑20 Update: Clean analytic contraction, release‑branch structure, and saltation

- No empirical scaling in gradients: no k_db or mean‑reduction factors. dL/dR is accumulated directly in the dB domain and chained to time constants via −dL/dR/T².
- Per‑step one‑hot check: with SSL_DEBUG_ONEHOT_T set, the kernel prints the analytic per‑term dot products (Ad, Bd, Gamma) for a chosen timestep and the matching per‑step FD dR using L_step = y_t. This validated the Ad‑term contraction and exposed remaining issues.
- Release‑branch structural exclusion: during release steps, series rates (Raf, Ras) do not affect the dynamics. Their contributions are excluded from dL/dR accumulation on release; only shunt rates (Rsf, Rss) contribute. This closed the residual gap in gT_as vs kernel FD.
- Saltation (hard gate) implemented and optional (SSL_USE_SALTATION=1): computes flip location α via linear interpolation on δ = a − y_prev, contracts J = λ · d x_{t+1}/dα via a split‑step composition, and multiplies by dα/d ln T from a central‑difference with adaptive backoff. Summaries/log throttles are available (see toggles above).
- Status: backward tests pass; forward vs FD dL/dR agrees; saltation contributions are small but non‑zero in typical cases.

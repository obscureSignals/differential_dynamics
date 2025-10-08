# SSL Auto-Release (series of parallel RC sections): hard-gate state-space ground truth

Current implementation status (2025-10-08)
- Domain: end-to-end dB domain (VCA control is linear in dB). Static curve computed in dB and clamped to <= 0 dB.
- Feedback topology: t-1 feedback (detector sees previous-sample output gain in dB), matching MATLAB ground truth.
- Forward path: implemented as a fused CPU kernel; ZOH discretization per mode (attack/release) with shared state basis.
- Backward path: analytic reverse-scan for comp_slope, comp_thresh, feedback coefficient, and input; finite-difference gradients for the four time constants (T_af, T_as, T_sf, T_ss), epsilon tuned to tests.
- Tests: gradient parity vs finite differences in tests/test_ssl_smoother_backward.py.
- Code entry points: differential_dynamics/backends/torch/ssl_smoother_ext.py (autograd wrapper), csrc/ssl_smoother.cpp (CPU kernel), and differential_dynamics/backends/torch/gain.py::SSL_comp_gain.
- MATLAB provenance: see the example loop and zoh_discretize_step pattern below; our implementation matches the structure and timing.

Next steps
- Build a parameter-recovery script that uses SSL_comp_gain with a dB-domain loss on gain traces.
- Add a configurable switch for time-constant gradient method and epsilon.
- Explore analytic gradients for time constants to replace FD for efficiency and stability.

This document specifies the exact state-space model and per-sample hard-gate algorithm that reproduce the analog SSL Auto release behavior. It matches the proven MATLAB reference and is the source of truth for subsequent implementations (sigmoid gate, Python forward/backward).

Sections
- Circuit topology (attack vs release)
- Physical states and continuous-time state-space (shared basis)
- ZOH discretization
- Hard-gate per-sample algorithm
- Release topology options (reverse-biased diode vs true open)
- Validation against the transfer functions
- Practical notes

---

## Circuit topology (attack vs release)

Observation node Vout is above two sections connected in series via a middle node Vm:

```text path=null start=null
Vin --[ Ra ]-- Vout --[ Rf || Cf ]-- Vm --[ Rs || Cs ]-- GND
```

- Attack: Ra = Rattack (diode forward-biased)
- Release: Ra = Release (very large, diode reverse-biased). For a true open diode, take Ra → ∞.

For each mode, the small-signal transfer function from Vin to Vout is:

```text path=null start=null
Zfast = Rf / (1 + s Rf Cf)
Zslow = Rs / (1 + s Rs Cs)
Vout/Vin = (Zfast + Zslow) / (Ra + Zfast + Zslow)
```

These TFs are the ground truth when the mode is fixed.

---

## Physical states and continuous-time state-space (shared basis)

We use the same two physical states in both modes so that states can be passed between them without scaling:
- x1 = Vout − Vm  (capacitor drop across the fast section)
- x2 = Vm         (capacitor drop across the slow section)
- Output y = Vout = x1 + x2

Let Ra be the series input resistor (Rattack during attack, Release during release). Define the series current I = (Vin − y)/Ra.

Capacitor KCL (currents positive from left to right):
- Cf dx1/dt = I − x1/Rf
- Cs dx2/dt = I − x2/Rs

This yields the continuous-time state-space, shared across modes:

```text path=null start=null
dx/dt = A(Ra) x + B(Ra) u,     y = C x + D u

A(Ra) = [ -(1/(Cf Ra) + 1/(Cf Rf))   -1/(Cf Ra)
          -1/(Cs Ra)                  -(1/(Cs Ra) + 1/(Cs Rs)) ]
B(Ra) = [ 1/(Cf Ra); 1/(Cs Ra) ]
C     = [ 1  1 ]
D     = 0
```

Special cases:
- Attack: Ra = Rattack → driven 2×2 system with input u = Vin
- Release (reverse-biased diode): Ra = Release ≫ 1 → weak drive; for a true open diode set Ra = ∞ ⇒ A = diag(−1/(Rf Cf), −1/(Rs Cs)), B = 0

This state basis corresponds exactly to the ladder physics and guarantees state continuity across mode switches.

---

## ZOH discretization

Discretize each mode with exact zero-order hold at Ts = 1/fs:

```text path=null start=null
(Ad_att, Bd_att, C_att, D_att) = c2d(A(Rattack), B(Rattack), C, D, Ts)
(Ad_rel, Bd_rel, C_rel, D_rel) = c2d(A(Release),  B(Release),  C, D, Ts)
```

We keep C_att = C_rel = [1 1] and D_att, D_rel from the ZOH, ensuring equality with the fixed-mode transfer functions above.

---

## Hard-gate per-sample algorithm

Let u[n] be the scalar drive (e.g., gain target in dB). Maintain a shared state x[n] ∈ R^2 that is used by both filters.

Per sample n:

1) Read left-continuous observation from the current state
- y[n] = C x[n]

2) Decide mode by hard gate (in dB compressor usage, “more negative” means attack)
- attack if u[n] < y[n]
- else release

3) Advance state over the sample with the chosen mode
- If attack: x[n+1] = Ad_att x[n] + Bd_att u[n]
- If release: choose one of the two release topologies (see next section):
  - reverse-biased diode approx: x[n+1] = Ad_rel x[n] + Bd_rel u[n]
  - true open diode: x[n+1] = Ad_open x[n]  (Bd_open = 0)

4) Output
- For control/metering, return y[n] (left-continuous). If you prefer right-continuous display, compute y[n] after the update instead; the dynamics are unchanged.

State sharing rule
- Always copy the same x between modes. Because both use the shared basis [x1; x2], state passing requires no scaling.

---

## Release topology options (reverse-biased diode vs true open)

- Reverse-biased diode (what the MATLAB ground truth uses): set a very large Release (e.g., 1e8 Ω). Keep D_rel and Bd_rel as produced by ZOH; drive with the current u[n]. This approximates a finite, tiny source conductance in release.
- True open diode: set Ra = ∞ when building the release operator. Then Bd_open = 0 and D_open = 0. During release, update with x[n+1] = Ad_open x[n] and drive the filter with u_rel = 0.

Both options are supported by the same shared-basis formulation.

---

## Validation against the transfer functions

Per mode (attack or release with a fixed Ra), the discrete-time state-space obtained by ZOH exactly matches the discrete-time transfer function of

```text path=null start=null
Vout/Vin = (Zfast + Zslow) / (Ra + Zfast + Zslow)
```

Use step or bode overlays to verify numerically for your parameters. Any mismatch indicates an implementation error.

---

## Practical notes

- Left-continuous readout avoids apparent instantaneous jumps at the gate boundary in plots. If you read out with the new topology’s D term before updating the state, the node voltage can jump (which is physical), but the envelope defined as C x is continuous.
- Do not attempt to share states between unrelated realizations (e.g., df2sos); only the shared state-space basis allows safe state passing.
- Numerical stability is best with ZOH; matched-pole formulas are acceptable but do not preserve the exact step shape of the ladder.
- Typical values (example): Rf = 91 kΩ, Cf = 0.47 µF (τf ≈ 43 ms); Rs = 750 kΩ, Cs = 6.8 µF (τs ≈ 5.1 s); Rattack ≈ 820 Ω; Release ≫ 1 MΩ.

---

### MATLAB reference (structure only; see repository scripts for full example)

```matlab path=null start=null
% Build shared-basis CT operators
[A_att_ct, B_att_ct, C_ct, D_ct] = series_parallel_rc_ct(Rattack, Rfast, Rslow, Cfast, Cslow);
[A_rel_ct, B_rel_ct, ~,     ~  ] = series_parallel_rc_ct(Release, Rfast, Rslow, Cfast, Cslow);

% ZOH discretization
Vout_in_ss_z_attack  = c2d(ss(A_att_ct, B_att_ct, C_ct, D_ct), Ts, 'zoh');
Vout_in_ss_z_release = c2d(ss(A_rel_ct, B_rel_ct, C_ct, D_ct), Ts, 'zoh');

% State-space filters (States are x = [Vout-Vm; Vm])
Hd_Vout_in_attack  = dfilt.statespace(Vout_in_ss_z_attack.A,  Vout_in_ss_z_attack.B,  Vout_in_ss_z_attack.C,  Vout_in_ss_z_attack.D);
Hd_Vout_in_release = dfilt.statespace(Vout_in_ss_z_release.A, Vout_in_ss_z_release.B, Vout_in_ss_z_release.C, Vout_in_ss_z_release.D);
Hd_Vout_in_attack.PersistentMemory  = true;
Hd_Vout_in_release.PersistentMemory = true;

% Per-sample hard gate (sharing States between modes)
if gain_raw_db < prev_gain_smoothed_dB
    Hd_Vout_in_attack.States = prevStates;
    y = filter(Hd_Vout_in_attack, gain_raw_db);
    prevStates = Hd_Vout_in_attack.States;
else
    Hd_Vout_in_release.States = prevStates;
    % For true open diode, drive 0.0 instead
    y = filter(Hd_Vout_in_release, gain_raw_db);
    prevStates = Hd_Vout_in_release.States;
end
```

Helper (continuous-time operators in shared basis):

```matlab path=null start=null
function [A, B, C, D] = series_parallel_rc_ct(Ra, Rf, Rs, Cf, Cs)
    invRa = 1/Ra;
    A = [ -(invRa/(Cf) + 1/(Cf*Rf))   -(invRa/Cf);
          -(invRa/(Cs))               -(invRa/(Cs) + 1/(Cs*Rs)) ];
    B = [ invRa/Cf; invRa/Cs ];
    C = [1 1];
    D = 0;
end
```

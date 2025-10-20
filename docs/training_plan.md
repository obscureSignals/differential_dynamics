# Parameter Identification: Data + Learning Regime

Objective

- Create a dataset and learning regime that recovers correct parameters for arbitrary (but reasonable) ground truths.

What we can do

- Design any probe or combination of probes we want (e.g., plateaus, dB-linear ramps, AB steps, ABA with echoes).
- Schedule learning to freeze subsets of parameters while training on specific probes.
- Other non-leaky engineering choices (batching, LR schedules, early stop, etc.).

What we cannot do (non-leakage policy)

- Do not allow anything besides processor behavior to influence training.
- No per-clip ground-truth parameter usage in data synthesis or masking; only global, processor-level bounds are permitted.

Requirements for success

- Probes must produce a smooth error surface with an unambiguous global minimum at the true parameters.
- The learning regime must be able to navigate that surface reliably (stable gradients, sensible parameterization, reasonable LR).

Paper goal

- Demonstrate the backward is correct and that learning succeeds under plausible conditions that remain useful for target applications.

Directive (2025-10-16): Drop staged curriculum

- We will not use staged (CT/R first, then TCs) training. Rationale:
  - With fb > 0 and SSL2’s closed-loop smoother, there is no truly TC-agnostic probe. Even constant plateaus have a TC- and state-dependent fixed point.
  - Cropping targets to steady tails does not remove TC dependence in the student forward pass, which starts from a reset state and re-enters a transient.
  - Staging therefore biases CT/R when TCs differ and crosses our "fewer shenanigans" bar.

Fewer-shenanigans ladder (preferred progression)

- scheduled curriculum with loss masks or similar < scheduled curriculum by itself < single phase synthetic data < training with real audio < training with real audio and learning fb

Recommended probe presets (non-leaky)

- Statics (plateaus): uniform levels in [−45, 0] dB with sufficient dwell; optionally fixed, theta-independent burn-in and crop to steady tail.
- dB-linear ramps: slow up/down ramps over [−40, 0] dB; use identical detector semantics; avoid normalization when isolating statics.
- AB steps: A < CT, B ≫ CT, long B plateau; separate up/down for attack/release identification when needed.
- ABA + echoes: for auto-release and feedback studies (later stages only).

Learning regime (single-phase, no staging)

- Jointly learn CT, R, and at least T_af (optionally T_sf). Freeze T_as/T_ss large; keep fb fixed per dataset (fb > 0 allowed).
- Probes: dB-linear ramps over [−40, 0] dB and long plateaus (no masks, no burn-in in trainer). Detector/teacher identical to student.
- Loss: plain RMSE on gain in dB (g_dB), consistent across probes. Early stopping on train RMSE.
- Validation: FD vs autograd checks on selected params; 1D/2D sweeps over R and/or CT to confirm a single basin under the probe mix.

Validation and evidence

- Gradient correctness: compare autograd vs central-difference FD on scalar loss for selected parameters.
- Identifiability scans: 2D loss heatmaps (e.g., CT vs R) with other params frozen—show single basin.
- Metrics: RMSE in dB on gain trajectories; report per-probe and aggregate.
- Reproducibility: record seeds, processor version tags, and probe metadata in each example.

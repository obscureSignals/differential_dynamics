# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository: differential_dynamics — research code for differentiable dynamic range processing. The package includes a PyTorch backend for compressor/expander gain with switchable attack/release gating modes, classical baselines, benchmarking utilities, and a vendored third_party torchcomp_core.

State of the union (2025-10-08)
- Current focus: Differentiable SSL-style compressor operating entirely in the dB domain, with auto-release (dual time constants) and t-1 sidechain feedback.
- Implementation status: Forward path done; CPU backward implemented with analytic adjoints for most parameters and finite-difference gradients for time constants. Verified by unit tests.
- Key files:
  - differential_dynamics/backends/torch/gain.py (SSL_comp_gain entry)
  - differential_dynamics/backends/torch/ssl_smoother_ext.py (autograd wrapper)
  - csrc/ssl_smoother.cpp (CPU kernel)
  - tests/test_ssl_smoother_backward.py (gradient tests)
- Reference/ground truth: docs/ssl_auto_release.md and the MATLAB formulation retained there.

Commands

- Environment and install (Python >= 3.9)
  - Create venv and install the package in editable mode
    ```bash path=null start=null
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -e .
    ```
  - Optional/needed for scripts and faster benchmarks
    - Bench plotting and audio I/O: matplotlib, torchaudio
    - Numba-based smoothers: numba
    ```bash path=null start=null
    python -m pip install matplotlib torchaudio numba
    ```

- Build wheels/distribution
  - Build a wheel into ./dist using setuptools
    ```bash path=null start=null
    python -m pip wheel . -w dist
    ```

- Tests
  - Repo tests (SSL smoother gradients):
    ```bash path=null start=null
    pytest -q tests/test_ssl_smoother_backward.py
    ```
  - Vendored tests (torchcomp_core):
    - Install the vendored torchcomp_core as a package so torchcomp is importable:
      ```bash path=null start=null
      python -m pip install -e third_party/torchcomp_core
      pytest -q third_party/torchcomp_core/tests
      ```
    - Run a single vendored test file or function:
      ```bash path=null start=null
      pytest -q third_party/torchcomp_core/tests/test_grad.py
      pytest -q third_party/torchcomp_core/tests/test_grad.py::test_low_order_cpu
      ```

- Benchmark script
  - Compare classical vs differentiable gating envelopes and plot gain traces. Requires matplotlib, torchaudio.
    ```bash path=null start=null
    python scripts/bench_forward.py --test-signal-type step \
      --comp-thresh -24 --comp-ratio 4.0 --exp-thresh -1000 --exp-ratio 1.0 \
      --attack-time-ms 10 --release-time-ms 100
    ```
  - Use a file input (ensure the path exists):
    ```bash path=null start=null
    python scripts/bench_forward.py --test-signal-type file --test-file-path /path/to/audio.wav
    ```

Notes on tools/config

- Linting/formatting/type-checking: No repo-configured tools found (e.g., ruff/flake8/black/mypy). If you intend to add them, prefer failing loudly on CI.
- PyTorch 2.1+ and numpy are core dependencies (from pyproject.toml). torchaudio and matplotlib are required only for the benchmark script.
- Numba is optional: the code paths in differential_dynamics/benchmarks handle absence of numba at runtime, but some backends (e.g., smoother_backend="numba") require it to be installed.

High-level architecture

- Differential gain core (PyTorch backend)
  - File: differential_dynamics/backends/torch/gain.py
  - Entry: compexp_gain_mode(x_rms, comp_thresh, comp_ratio, exp_thresh, exp_ratio, at_ms, rt_ms, fs, ar_mode, smoother_backend)
  - Purpose: Computes compressor/expander gain traces with identical detector semantics and static dB curve to the baseline (torchcomp), while swapping only the attack/release gating and smoothing policy.
  - Modes and backends:
    - ar_mode="hard": delegates entirely to third_party.torchcomp_core.torchcomp.compexp_gain to preserve exact baseline behavior and custom backward.
    - ar_mode="sigmoid": uses a TorchScript loop or a Numba kernel to implement a differentiable, gate-blended one-pole smoother. Gate operates in the dB domain; k_db controls sharpness. No beta smoothing.
  - Key invariants:
    - Static curve is computed in dB and clamped to never exceed 0 dB (only reduce or keep unity gain), matching torchcomp.
    - Time constants (ms2coef) use the 10–90% convention across implementations.
  - Near-term plan: simplify API to expose k_db directly (CLI flag) and remove gate_cfg/beta; dB-domain gating is standard.

- Baseline (classical compressor/expander)
  - File: differential_dynamics/baselines/classical_compressor.py
  - Class: ClassicalCompressor
  - Purpose: Reference classical A/R implementation. Computes static dB curve and applies a two-coefficient A/R smoother (Numba-backed forward recurrence). Used for benchmarking against differentiable variants.

- Benchmarks and utilities
  - Files: differential_dynamics/benchmarks/bench_utilities.py, differential_dynamics/benchmarks/signals.py
  - Purpose: Helpers for gain/log conversions, active-region masks, RMSE in dB, simple signal generators (tone/step/burst/ramp), and an efficient lfilter-based EMA detector.
  - Design choice: bench_utilities tries to import numba; when unavailable, functions fall back so that non-Numba paths still run (with reduced speed).

- Scripted benchmark entry point
  - File: scripts/bench_forward.py
  - Purpose: Standalone comparison of baseline vs differentiable gating envelopes. Generates or loads a test signal, computes detector envelopes, runs gain functions, reports RMSE in dB on active regions, and plots traces.
  - Dependencies: matplotlib (plotting), torchaudio (file I/O), PyTorch/torchaudio for tensors and DSP utilities.

- Vendored dependency: torchcomp_core (MIT)
  - Path: third_party/torchcomp_core
  - Provides: torchcomp (fast, efficient compressor/expander/limiter primitives), tests, and packaging. This repository includes it as a subtree for baseline parity and reference tests.
  - Important detail: Vendored tests import torchcomp as a top-level module. Ensure you install the third_party subpackage (pip install -e third_party/torchcomp_core) or have torchcomp available in your environment when running those tests.

Documentation and attribution

- README.md: Summarizes scope (research for differentiable dynamic range processing), includes contributions (switchable differentiable gating variants, benchmarks, paper code), baselines (torchcomp_core subtree), and attribution to DiffAPF/torchcomp (MIT). See THIRD_PARTY_NOTICES.md if present for licensing details.


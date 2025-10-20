import torch
import math
import random
from typing import Iterable, List


def tone(freq: float, fs: int, T: int, B: int = 1, amp: float = 0.5) -> torch.Tensor:
    """Sine tone.
    Returns (B, T)
    """
    t = torch.arange(T).float() / fs
    x = amp * torch.sin(2 * math.pi * freq * t)[None, :].repeat(B, 1)
    return x


def step(
    fs: int,
    T: int,
    B: int = 1,
    at: float = 0.5,
    amp_before: float = 0.1,
    amp_after: float = 0.7,
) -> torch.Tensor:
    """Amplitude step at fraction 'at' of the clip.
    Returns (B, T)
    """
    t0 = int(T * at)
    x = torch.full((B, T), amp_before)
    x[:, t0:] = amp_after
    return x


def burst(
    fs: int,
    T: int,
    B: int = 1,
    start: float = 0.2,
    dur: float = 0.1,
    amp: float = 0.8,
    freq: float = 1000.0,
) -> torch.Tensor:
    """Sine burst.
    Returns (B, T)
    """
    s = int(T * start)
    L = min(T - s, int(T * dur))
    x = torch.zeros((B, T))
    if L > 0:
        n = torch.arange(L)
        x[:, s : s + L] = amp * torch.sin(2 * math.pi * freq * n / fs)[None, :]
    return x


def ramp(
    fs: int,
    T: int,
    B: int = 1,
    start: float = 0.2,
    dur: float = 0.4,
    a0: float = 0.1,
    a1: float = 0.8,
) -> torch.Tensor:
    """Linear amplitude ramp.
    Returns (B, T)
    """
    s = int(T * start)
    L = max(1, int(T * dur))
    slope = torch.linspace(a0, a1, L)
    x = torch.zeros((B, T)) + a0
    x[:, s : s + L] = slope
    x[:, s + L :] = a1
    return x


def am_tone(
    fs: int,
    T: int,
    B: int = 1,
    carrier_hz: float = 1000.0,
    am_hz: float = 4.0,
    depth: float = 0.5,
    amp: float = 0.5,
) -> torch.Tensor:
    """Amplitude-modulated tone. depth in [0,1] controls modulation index.
    Returns (B, T)
    """
    t = torch.arange(T).float() / fs
    carrier = torch.sin(2 * math.pi * carrier_hz * t)
    mod = 1.0 + depth * torch.sin(2 * math.pi * am_hz * t)
    mod = torch.clamp(mod, min=0.0)
    x = amp * (carrier * mod)[None, :].repeat(B, 1)
    return x


def white_noise(fs: int, T: int, B: int = 1, amp: float = 0.1) -> torch.Tensor:
    """White noise clip, amplitude scaled.
    Returns (B, T)
    """
    return amp * torch.randn(B, T)


def beating_tones(
    fs: int,
    T: int,
    B: int = 1,
    base_hz: float = 440.0,
    beat_hz: float = 3.0,
    amp: float = 0.5,
) -> torch.Tensor:
    """Two sine tones with small frequency separation to produce beating in the envelope.
    Returns (B, T)
    """
    t = torch.arange(T).float() / fs
    x = 0.5 * (
        torch.sin(2 * math.pi * base_hz * t)
        + torch.sin(2 * math.pi * (base_hz + beat_hz) * t)
    )
    return amp * x[None, :].repeat(B, 1)


essentially_zero = 1e-12


def am_noise(
    fs: int,
    T: int,
    B: int = 1,
    am_hz: float = 3.0,
    depth: float = 0.5,
    amp: float = 0.2,
) -> torch.Tensor:
    """Amplitude-modulated white noise (slow AM for near-CT fluctuation).
    Returns (B, T)
    """
    t = torch.arange(T).float() / fs
    mod = 1.0 + depth * torch.sin(2 * math.pi * am_hz * t)
    mod = torch.clamp(mod, min=0.0)
    n = torch.randn(B, T)
    return amp * (n * mod[None, :])


def composite_program(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    min_segments: int = 4,
    max_segments: int = 12,
) -> torch.Tensor:
    """Deterministic 2s-style program to maximize identifiability (API unchanged).

    Layout (fractions of T):
      - 45% quasi-static staircase (pins static curve CT/CR)
      - 35% A→B→A with long B plateau (feedback/disambiguation, dynamics)
      - 10% two-pulse echo (release/feedback interaction)
      - 10% near-threshold flicker (gate boundary, event sensitivity)

    Notes:
      - No randomness; rng/min_segments/max_segments are ignored for compatibility.
      - If T != 2s*fs, durations scale proportionally and the last value is padded/truncated to fit T.
    Returns (B, T).
    """
    # Allocate canvas
    x = torch.zeros(B, T)
    t_cur = 0

    def paste(seg: torch.Tensor):
        nonlocal t_cur
        L = min(seg.shape[1], T - t_cur)
        if L > 0:
            x[:, t_cur : t_cur + L] = seg[:, :L]
            t_cur += L

    # Helper builders (deterministic)
    def build_static(dur_samp: int) -> torch.Tensor:
        # 16 steps, evenly spaced amplitudes 0.02..0.90, per-step length >= 1
        n_steps = 16
        step_len = max(1, dur_samp // n_steps)
        levels = torch.linspace(0.02, 0.90, n_steps)
        segs = [torch.full((B, step_len), float(a)) for a in levels]
        seg = torch.cat(segs, dim=1)
        if seg.shape[1] < dur_samp:
            seg = torch.cat([seg, torch.full((B, dur_samp - seg.shape[1]), float(levels[-1]))], dim=1)
        return seg[:, :dur_samp]

    def build_aba(dur_samp: int) -> torch.Tensor:
        # A(20%) -> B(60%) -> A2(20%), A≈0.08, B≈0.80, A2≈0.10
        a = 0.08; b = 0.80; a2 = 0.10
        dA = max(1, int(0.20 * dur_samp))
        dB = max(1, int(0.60 * dur_samp))
        dA2= max(1, dur_samp - dA - dB)
        return torch.cat([
            torch.full((B, dA), a),
            torch.full((B, dB), b),
            torch.full((B, dA2), a2),
        ], dim=1)

    def build_echo(dur_samp: int) -> torch.Tensor:
        # Two pulses, width ≈ 4 ms, separation ≈ 40% of segment, centered
        seg = torch.zeros(B, dur_samp)
        w = max(1, int(0.004 * fs))
        dt = max(w + 1, int(0.40 * dur_samp))
        span = 2 * w + dt
        t0 = max(0, (dur_samp - span) // 2)
        a = 0.85
        seg[:, t0 : t0 + w] = a
        t1 = min(dur_samp, t0 + w + dt)
        seg[:, t1 : min(dur_samp, t1 + w)] = a
        return seg

    def build_flicker(dur_samp: int) -> torch.Tensor:
        # Toggle every 20 ms between a_lo=0.10 and a_hi=0.16
        seg = torch.zeros(B, dur_samp)
        a_lo, a_hi = 0.10, 0.16
        p = max(2, int(0.020 * fs))
        toggle = False
        t = 0
        while t < dur_samp:
            L = min(p, dur_samp - t)
            seg[:, t : t + L] = a_hi if toggle else a_lo
            toggle = not toggle
            t += L
        return seg

    # Durations
    static_len = int(0.45 * T)
    aba_len    = int(0.35 * T)
    echo_len   = int(0.10 * T)
    flick_len  = T - static_len - aba_len - echo_len  # ensure exact fill

    # Build and paste segments in deterministic order
    paste(build_static(static_len))
    paste(build_aba(aba_len))
    paste(build_echo(echo_len))
    paste(build_flicker(flick_len))

    # If not filled (rounding), pad last value; if over, truncate already handled
    if t_cur < T and t_cur > 0:
        x[:, t_cur:] = x[:, t_cur - 1 : t_cur]
    return x
    return x


def composite_program_jitter(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Composite program with controlled per-clip randomness in envelope domain.

    Goals:
    - Preserve probe semantics (static staircase, ABA, echo, flicker) for identifiability
    - Introduce bounded jitter so each clip is unique but gradients remain stable
    - Optional light envelope "wobble" to mimic residual ripple after rectifier+smoother

    This operates purely on positive envelopes (no AC carriers), consistent with
    x_peak_dB = gain_db(abs(x)) in the builder.
    """
    if rng is None:
        rng = random.Random()

    x = torch.zeros(B, T)
    t_cur = 0

    def paste(seg: torch.Tensor):
        nonlocal t_cur
        L = min(seg.shape[1], T - t_cur)
        if L > 0:
            x[:, t_cur : t_cur + L] = seg[:, :L]
            t_cur += L

    # Durations around the 45/35/10/10 split with small jitter
    base = [0.45, 0.35, 0.10, 0.10]
    jitter = [rng.uniform(-0.05, 0.05) for _ in base]
    weights = [max(0.01, b + j) for b, j in zip(base, jitter)]
    s = sum(weights)
    weights = [w / s for w in weights]
    static_len = int(round(weights[0] * T))
    aba_len = int(round(weights[1] * T))
    echo_len = int(round(weights[2] * T))
    flick_len = max(0, T - static_len - aba_len - echo_len)

    # Helper builders with jitter
    def build_static_jitter(dur_samp: int) -> torch.Tensor:
        n_steps = rng.randint(12, 18)
        step_len = max(1, dur_samp // n_steps)
        # Monotone levels with small jitter
        a0, a1 = 0.02, 0.90
        base_levels = torch.linspace(a0, a1, n_steps)
        jittered = []
        for lvl in base_levels:
            dl = rng.uniform(-0.03, 0.03)
            jittered.append(float(min(max(lvl.item() + dl, a0), a1)))
        levels = torch.tensor(sorted(jittered))
        segs = [torch.full((B, step_len), float(a)) for a in levels]
        seg = torch.cat(segs, dim=1)
        if seg.shape[1] < dur_samp:
            seg = torch.cat([seg, torch.full((B, dur_samp - seg.shape[1]), float(levels[-1]))], dim=1)
        return seg[:, :dur_samp]

    def build_aba_jitter(dur_samp: int) -> torch.Tensor:
        # Sample levels and dwell ratios
        a = rng.uniform(0.05, 0.15)
        b = rng.uniform(0.60, 0.90)
        a2 = min(0.95, max(0.01, a + rng.uniform(-0.04, 0.04)))
        # Dwell proportions ~ Dirichlet-like by sampling and normalizing
        rA = max(0.05, rng.uniform(0.15, 0.30))
        rB = max(0.30, rng.uniform(0.50, 0.70))
        rC = max(0.05, 1.0 - rA - rB)
        denom = rA + rB + rC
        rA, rB, rC = rA / denom, rB / denom, rC / denom
        dA = max(1, int(round(rA * dur_samp)))
        dB = max(1, int(round(rB * dur_samp)))
        dC = max(1, dur_samp - dA - dB)
        return torch.cat([
            torch.full((B, dA), a),
            torch.full((B, dB), b),
            torch.full((B, dC), a2),
        ], dim=1)

    def build_echo_jitter(dur_samp: int) -> torch.Tensor:
        seg = torch.zeros(B, dur_samp)
        # Pulse width 2–8 ms, separation 30–60% of segment, 2 or 3 pulses
        w = max(1, int(rng.uniform(0.002, 0.008) * fs))
        dt = max(w + 1, int(rng.uniform(0.30, 0.60) * dur_samp))
        n_pulses = 3 if rng.random() < 0.25 else 2
        a0 = rng.uniform(0.70, 0.90)
        span = n_pulses * w + (n_pulses - 1) * dt
        t0 = max(0, (dur_samp - span) // 2)
        for k in range(n_pulses):
            s = t0 + k * (w + dt)
            if s >= dur_samp:
                break
            e = min(dur_samp, s + w)
            amp = a0 * (0.7 ** k)
            seg[:, s:e] = amp
        return seg

    def build_flicker_jitter(dur_samp: int) -> torch.Tensor:
        seg = torch.zeros(B, dur_samp)
        a_lo = max(0.01, rng.uniform(0.08, 0.12))
        a_hi = min(0.99, rng.uniform(0.14, 0.20))
        p = max(2, int(rng.uniform(0.010, 0.030) * fs))  # 10–30 ms period
        toggle = rng.random() < 0.5
        t = 0
        while t < dur_samp:
            L = min(p, dur_samp - t)
            seg[:, t : t + L] = a_hi if toggle else a_lo
            toggle = not toggle
            t += L
        return seg

    paste(build_static_jitter(static_len))
    paste(build_aba_jitter(aba_len))
    paste(build_echo_jitter(echo_len))
    paste(build_flicker_jitter(flick_len))

    # Optional light envelope wobble to mimic residual ripple without AC carriers
    if rng.random() < 0.35:
        a = rng.uniform(0.02, 0.06)  # wobble depth
        f = rng.uniform(20.0, 180.0)  # Hz
        t = torch.arange(T).float() / fs
        wob = (1.0 + a * torch.sin(2 * math.pi * f * t)).clamp_min(0.0)
        x = (x * wob[None, :]).clamp(0.0, 1.0)

    # If not filled (rounding), pad last value
    if t_cur < T and t_cur > 0:
        x[:, t_cur:] = x[:, t_cur - 1 : t_cur]
    return x


# --- New parameter-agnostic probe generators (P0-P3) ---

def _cosine_edge(a: float, b: float, n: int) -> torch.Tensor:
    """Half-cosine edge from a to b over n samples (n>=1)."""
    if n <= 1:
        return torch.tensor([b], dtype=torch.float32)
    t = torch.arange(n, dtype=torch.float32)
    w = 0.5 * (1.0 - torch.cos(math.pi * (t + 1) / n))
    return torch.tensor(a, dtype=torch.float32) + (torch.tensor(b, dtype=torch.float32) - a) * w


def probe_p0_quasi_static(fs: int, T: int, B: int = 1, levels: Iterable[float] | None = None) -> torch.Tensor:
    """P0: Slow quasi-static staircase over fixed absolute envelope levels, long dwells.
    levels in [0,1]."""
    if levels is None:
        levels = [0.02, 0.05, 0.08, 0.12, 0.18, 0.26, 0.36, 0.50, 0.70, 0.90]
    levels = list(levels)
    # Equal dwell per level
    step_len = max(1, T // max(1, len(levels)))
    segs = [torch.full((B, step_len), float(a)) for a in levels]
    x = torch.cat(segs, dim=1)
    if x.shape[1] < T:
        x = torch.cat([x, torch.full((B, T - x.shape[1]), float(levels[-1]))], dim=1)
    return x[:, :T]


essential_levels: List[float] = [0.03, 0.06, 0.12, 0.24, 0.48, 0.84]


def probe_p1_smoothed_pairs(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    n_pairs: int = 3,
    edge_fast_ms: float = 25.0,
    edge_slow_ms: float = 200.0,
    edge_ms_grid: List[float] | None = None,
) -> torch.Tensor:
    """P1: Multiple absolute up/down smoothed steps with long dwells.
    Chooses pairs from essential_levels. Edge durations are C1 via cosine.

    If edge_ms_grid is provided, choose edge durations per pair from this grid
    (alternating/random) to cover a broad spectrum of attack/release times.
    """
    rng = rng or random.Random()
    levels = essential_levels.copy()
    # Choose pairs ensuring up/down variety
    pairs = [(0.06, 0.30), (0.06, 0.80), (0.12, 0.48), (0.24, 0.80), (0.12, 0.84), (0.03, 0.24)]
    rng.shuffle(pairs)
    pairs = pairs[: max(1, n_pairs)]

    def dur(ms: float) -> int:
        return max(1, int(round(ms * 1e-3 * fs)))

    e_fast = dur(edge_fast_ms)
    e_slow = dur(edge_slow_ms)

    # Allocate time slots per pair (up, dwell_hi, down, dwell_lo)
    per_pair = max(1, T // (len(pairs)))
    x = torch.zeros(B, 0)
    for i, (a, b) in enumerate(pairs):
        if edge_ms_grid is not None and len(edge_ms_grid) > 0:
            # Alternate through grid then random after
            idx = i if i < len(edge_ms_grid) else rng.randrange(len(edge_ms_grid))
            edge = dur(float(edge_ms_grid[idx]))
        else:
            edge = e_fast if (i % 2 == 0) else e_slow
        dwell_hi = max(1, int(0.35 * per_pair))
        dwell_lo = max(1, int(0.20 * per_pair))
        edge = min(edge, max(1, per_pair // 6))
        up = _cosine_edge(a, b, edge)[None, :].repeat(B, 1)
        hi = torch.full((B, dwell_hi), float(b))
        down = _cosine_edge(b, a, edge)[None, :].repeat(B, 1)
        lo = torch.full((B, dwell_lo), float(a))
        seg = torch.cat([up, hi, down, lo], dim=1)
        x = torch.cat([x, seg], dim=1)
    # Fit to T
    if x.shape[1] < T:
        pad = torch.full((B, T - x.shape[1]), float(pairs[-1][0]))
        x = torch.cat([x, pad], dim=1)
    return x[:, :T]


def probe_p2_echo_and_silence(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """P2: Echo pulses with smooth edges, plus trailing silence to expose release tails."""
    rng = rng or random.Random()
    x = torch.zeros(B, T)
    # Decide number of pulses and spacing
    n_pulses = 2 if rng.random() < 0.7 else 3
    # Choose total active window ~ 30-60% of T
    win = int(rng.uniform(0.3, 0.6) * T)
    t0 = (T - win) // 2
    # Pulse width 2–20 ms; spacing 10–40% of window
    w = max(1, int(rng.uniform(0.002, 0.020) * fs))
    gap = max(w + 1, int(rng.uniform(0.10, 0.40) * win))
    amp0 = rng.uniform(0.6, 0.9)
    for k in range(n_pulses):
        s = t0 + k * (w + gap)
        if s >= T:
            break
        e = min(T, s + w)
        edge = max(1, w)
        up = _cosine_edge(0.0, amp0 * (0.7 ** k), edge)
        down = _cosine_edge(amp0 * (0.7 ** k), 0.0, edge)
        # Place up then down; if overlaps, just saturate
        seg = torch.cat([up, down])
        e2 = min(T, s + seg.numel())
        L = e2 - s
        if L > 0:
            x[:, s:e2] = seg[:L][None, :]
    return x


def probe_p3_db_dither(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    bases: Iterable[float] | None = None,
    freqs_hz: Iterable[float] | None = None,
    depth_db: float = 2.5,
) -> torch.Tensor:
    """P3: Small-signal dB dither around fixed absolute bases at multiple low frequencies."""
    rng = rng or random.Random()
    if bases is None:
        bases = [0.05, 0.15, 0.35, 0.70]
    if freqs_hz is None:
        freqs_hz = [0.5, 1.0, 2.0, 4.0, 8.0]
    t = torch.arange(T, dtype=torch.float32) / float(fs)
    # Build segments sequentially across bases and freqs
    n_segments = len(list(bases)) * len(list(freqs_hz))
    seg_len = max(1, T // max(1, n_segments))
    x = torch.zeros(B, 0)
    for b in bases:
        for f in freqs_hz:
            n = seg_len
            tt = t[:n]
            d_db = depth_db * torch.sin(2 * math.pi * float(f) * tt)
            scale = torch.pow(10.0, d_db / 20.0)
            seg = torch.clamp(float(b) * scale, 0.0, 1.0)[None, :].repeat(B, 1)
            x = torch.cat([x, seg], dim=1)
    if x.shape[1] < T:
        x = torch.cat([x, x[:, -1:].repeat(1, T - x.shape[1])], dim=1)
    return x[:, :T]


def probe_p1b_preconditioned_pairs(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    n_pairs: int = 2,
) -> torch.Tensor:
    """P1B: Preconditioned step pairs to disambiguate feedback vs ratio.

    Structure per pair:
      1) Reset: low-level dwell to bring y_prev ≈ 0
      2) Up step to L_hi with a moderate cosine edge, long dwell
      3) Precondition: long high dwell to drive y_prev ≪ 0
      4) Down step to the SAME L_hi target, long dwell

    Both events end at identical L_hi but start with different y_prev, so
    the trajectories differ due to feedback only. Levels are absolute (no param leakage).
    """
    rng = rng or random.Random()
    # Partition time across pairs
    per_pair = max(1, T // max(1, n_pairs))
    x = torch.zeros(B, 0)

    def dur(frac: float) -> int:
        return max(1, int(round(frac * per_pair)))

    for _ in range(n_pairs):
        # Choose levels
        L_lo = rng.uniform(0.03, 0.10)
        L_hi = rng.uniform(0.60, 0.85)
        # Dwell/edge allocations
        d_reset = dur(0.20)
        edge = min(dur(0.10), max(1, int(0.015 * fs)))
        d_hi = dur(0.25)
        d_precond = dur(0.25)
        d_post = per_pair - (d_reset + edge + d_hi + d_precond + edge + d_hi)
        d_post = max(1, d_post)
        # Build segments
        reset = torch.full((B, d_reset), float(L_lo))
        up = _cosine_edge(L_lo, L_hi, edge)[None, :].repeat(B, 1)
        hi1 = torch.full((B, d_hi), float(L_hi))
        precond = torch.full((B, d_precond), float(L_hi))
        down_to_same_hi = _cosine_edge(max(L_hi, 0.95), L_hi, edge)[None, :].repeat(B, 1)
        hi2 = torch.full((B, d_hi), float(L_hi))
        post = torch.full((B, d_post), float(L_lo))
        seg = torch.cat([reset, up, hi1, precond, down_to_same_hi, hi2, post], dim=1)
        x = torch.cat([x, seg], dim=1)

    # Fit to T
    if x.shape[1] < T:
        pad = torch.full((B, T - x.shape[1]), float(x[:, -1:].mean().item()))
        x = torch.cat([x, pad], dim=1)
    return x[:, :T]


essential_drop_levels: List[float] = [0.10, 0.14, 0.20]


def probe_p2b_release_drop_near_threshold(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    n_blocks: int = 3,
) -> torch.Tensor:
    """P2B: Release drop tests near the gate boundary with history/no-history contrast.

    For each block we do:
      - Long high dwell (drive y_prev negative)
      - Drop to a mid level in {0.10, 0.14, 0.20} and dwell
      - Optionally insert a very long gap (reset y_prev ≈ 0), then repeat a similar drop
    This produces matched regions at similar absolute levels with/without prehistory.
    """
    rng = rng or random.Random()
    per = max(1, T // max(1, n_blocks))
    x = torch.zeros(B, 0)

    def dur(frac: float) -> int:
        return max(1, int(round(frac * per)))

    for _ in range(n_blocks):
        L_hi = rng.uniform(0.65, 0.90)
        L_mid = float(rng.choice(essential_drop_levels))
        # First sequence: preconditioned
        d_hi1 = dur(0.35)
        edge = min(dur(0.08), max(1, int(0.010 * fs)))
        d_mid = dur(0.22)
        d_gap = dur(rng.uniform(0.08, 0.20))  # history reset window
        d_tail = per - (d_hi1 + edge + d_mid + d_gap + edge + d_mid)
        d_tail = max(1, d_tail)
        hi1 = torch.full((B, d_hi1), L_hi)
        drop = _cosine_edge(L_hi, L_mid, edge)[None, :].repeat(B, 1)
        mid = torch.full((B, d_mid), L_mid)
        gap = torch.full((B, d_gap), 0.0)
        # Second sequence: no prehistory (from near silence)
        drop2 = _cosine_edge(0.0, L_mid, edge)[None, :].repeat(B, 1)
        mid2 = torch.full((B, d_mid), L_mid)
        tail = torch.full((B, d_tail), L_mid)
        seg = torch.cat([hi1, drop, mid, gap, drop2, mid2, tail], dim=1)
        x = torch.cat([x, seg], dim=1)

    if x.shape[1] < T:
        x = torch.cat([x, x[:, -1:].repeat(1, T - x.shape[1])], dim=1)
    return x[:, :T]


def probe_p5_transient_churn(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    min_step: float = 0.05,
    edge_ms_range: tuple[float, float] = (4.0, 120.0),
    gap_prob: float = 0.25,
) -> torch.Tensor:
    """P5: Transient churn – continuous sequence of short cosine edges between random levels.

    Goals:
    - Minimize steady dwell; keep envelope moving most of the time.
    - Random absolute levels in [0.02, 0.90] with minimum step size to avoid tiny changes.
    - Random short edges (8–40 ms) to excite both attack and release repeatedly.
    - Occasional short gaps (silence) to reset history and expose feedback vs ratio via contrast.

    No parameter conditioning; entirely absolute levels and durations.
    """
    rng = rng or random.Random()
    x = torch.zeros(B, 0)

    def edge_len_ms() -> int:
        lo, hi = edge_ms_range
        return max(1, int(round(rng.uniform(lo, hi) * 1e-3 * fs)))

    cur = rng.uniform(0.02, 0.20)
    while x.shape[1] < T:
        # Next target level far enough away
        for _ in range(8):  # try a few times
            nxt = rng.uniform(0.02, 0.90)
            if abs(nxt - cur) >= min_step:
                break
        else:
            nxt = min(0.90, max(0.02, cur + (min_step if rng.random() < 0.5 else -min_step)))
        L = edge_len_ms()
        seg = _cosine_edge(cur, nxt, L)[None, :].repeat(B, 1)
        x = torch.cat([x, seg], dim=1)
        cur = nxt
        # Occasionally inject a very short gap to drop history
        if rng.random() < gap_prob:
            gapL = max(1, int(round(rng.uniform(4.0, 20.0) * 1e-3 * fs)))
            x = torch.cat([x, torch.zeros(B, gapL)], dim=1)
            # resume from a small level to vary direction
            cur = rng.uniform(0.02, 0.20)
    if x.shape[1] > T:
        x = x[:, :T]
    else:
        # pad final value minimally (should be rare given loop condition)
        if x.shape[1] < T and x.shape[1] > 0:
            x = torch.cat([x, x[:, -1:].repeat(1, T - x.shape[1])], dim=1)
    return x


# Additional parameter-agnostic probes (P6-P10)

def probe_p6_slope_sweep(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    durations_ms: List[float] | None = None,
    n_segments: int = 6,
) -> torch.Tensor:
    """P6: Sequence of ramps with a grid of edge durations (both up and down),
    separated by silences to reset history.
    """
    rng = rng or random.Random()
    if durations_ms is None:
        durations_ms = [5, 10, 20, 50, 100, 200, 400]
    per = max(1, T // max(1, n_segments))
    x = torch.zeros(B, 0)
    cur = rng.uniform(0.03, 0.20)
    for i in range(n_segments):
        L = max(1, int(round((rng.choice(durations_ms)) * 1e-3 * fs)))
        # ensure edge fits segment
        L = min(L, max(1, per - 2))
        nxt = rng.uniform(0.60, 0.90) if (i % 2 == 0) else rng.uniform(0.05, 0.25)
        ramp = _cosine_edge(cur, nxt, L)[None, :].repeat(B, 1)
        dwell = torch.full((B, max(1, per - L)), float(nxt))
        # occasionally insert silence gap instead of dwell to reset
        if rng.random() < 0.35:
            dwell = torch.zeros_like(dwell)
        seg = torch.cat([ramp, dwell], dim=1)
        x = torch.cat([x, seg], dim=1)
        cur = nxt
    if x.shape[1] < T:
        x = torch.cat([x, x[:, -1:].repeat(1, T - x.shape[1])], dim=1)
    return x[:, :T]


def probe_p7_release_staircase(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    levels_mid: List[float] | None = None,
    n_blocks: int = 3,
) -> torch.Tensor:
    """P7: High precondition, then multiple drops to mid levels with gaps between blocks.
    Contrasts history/no-history for release identification.
    """
    rng = rng or random.Random()
    if levels_mid is None:
        levels_mid = [0.10, 0.14, 0.20, 0.30]
    per = max(1, T // max(1, n_blocks))
    x = torch.zeros(B, 0)
    for _ in range(n_blocks):
        L_hi = rng.uniform(0.70, 0.90)
        d_hi = max(1, int(0.30 * per))
        edge = max(1, int(0.010 * fs))
        d_mid = max(1, int(0.25 * per))
        d_gap = max(1, int(0.15 * per))
        hi = torch.full((B, d_hi), float(L_hi))
        mids = []
        for m in rng.sample(levels_mid, k=min(2, len(levels_mid))):
            drop = _cosine_edge(L_hi, float(m), edge)[None, :].repeat(B, 1)
            mid = torch.full((B, d_mid), float(m))
            mids.append(torch.cat([drop, mid], dim=1))
        gap = torch.zeros(B, d_gap)
        seg = torch.cat([hi] + mids + [gap], dim=1)
        x = torch.cat([x, seg], dim=1)
    if x.shape[1] < T:
        x = torch.cat([x, x[:, -1:].repeat(1, T - x.shape[1])], dim=1)
    return x[:, :T]


def probe_p8_attack_doublet(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    n_blocks: int = 4,
) -> torch.Tensor:
    """P8: Two successive up-steps to increasing targets with short dwell between.
    Emphasizes attack dynamics and interaction across closely spaced events.
    """
    rng = rng or random.Random()
    per = max(1, T // max(1, n_blocks))
    x = torch.zeros(B, 0)
    for _ in range(n_blocks):
        a = rng.uniform(0.04, 0.12)
        b = rng.uniform(0.35, 0.65)
        c = min(0.98, b + rng.uniform(0.10, 0.25))
        edge1 = max(1, int(0.008 * fs))
        edge2 = max(1, int(0.020 * fs))
        d1 = max(1, int(0.15 * per))
        d2 = max(1, int(0.20 * per))
        up1 = _cosine_edge(a, b, edge1)[None, :].repeat(B, 1)
        hi1 = torch.full((B, d1), float(b))
        up2 = _cosine_edge(b, c, edge2)[None, :].repeat(B, 1)
        hi2 = torch.full((B, d2), float(c))
        tail = torch.full((B, per - (edge1 + d1 + edge2 + d2)), float(c))
        seg = torch.cat([up1, hi1, up2, hi2, tail], dim=1)
        x = torch.cat([x, seg], dim=1)
    if x.shape[1] < T:
        x = torch.cat([x, x[:, -1:].repeat(1, T - x.shape[1])], dim=1)
    return x[:, :T]


def probe_p9_am_sweep(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    base_levels: List[float] | None = None,
    f_lo: float = 0.5,
    f_hi: float = 30.0,
) -> torch.Tensor:
    """P9: AM frequency sweep around fixed bases to probe gating dynamics at many rates."""
    rng = rng or random.Random()
    if base_levels is None:
        base_levels = [0.08, 0.20, 0.50]
    t = torch.arange(T, dtype=torch.float32) / float(fs)
    x = torch.zeros(B, 0)
    n_bases = len(base_levels)
    seg_len = max(1, T // max(1, n_bases))
    for b in base_levels:
        tt = t[:seg_len]
        # log sweep from f_lo to f_hi
        k = math.log(f_hi / max(1e-6, f_lo)) / max(1e-12, tt[-1].item() if tt.numel() > 1 else 1.0)
        f = f_lo * torch.exp(k * tt)
        phase = 2 * math.pi * torch.cumsum(f / fs, dim=0)
        d_db = 3.0 * torch.sin(phase)
        scale = torch.pow(10.0, d_db / 20.0)
        seg = torch.clamp(float(b) * scale, 0.0, 1.0)[None, :].repeat(B, 1)
        x = torch.cat([x, seg], dim=1)
    if x.shape[1] < T:
        x = torch.cat([x, x[:, -1:].repeat(1, T - x.shape[1])], dim=1)
    return x[:, :T]


def probe_p10_prbs_amp(
    fs: int,
    T: int,
    B: int = 1,
    rng: random.Random | None = None,
    levels: List[float] | None = None,
    min_dwell_ms: float = 12.0,
) -> torch.Tensor:
    """P10: PRBS-like amplitude switching among fixed absolute levels with min dwell."""
    rng = rng or random.Random()
    if levels is None:
        levels = [0.04, 0.08, 0.16, 0.32, 0.64, 0.90]
    x = torch.zeros(B, 0)
    while x.shape[1] < T:
        a = float(rng.choice(levels))
        dwell = max(1, int(round(min_dwell_ms * 1e-3 * fs)))
        if x.shape[1] + dwell > T:
            dwell = T - x.shape[1]
        seg = torch.full((B, dwell), a)
        x = torch.cat([x, seg], dim=1)
    return x

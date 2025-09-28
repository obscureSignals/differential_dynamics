import torch
import math
import random


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
        torch.sin(2 * math.pi * base_hz * t) + torch.sin(2 * math.pi * (base_hz + beat_hz) * t)
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
    """Stitch a sequence of heterogeneous segments into a program-like clip.

    Segment palette:
      - step: piecewise-constant amplitude with a single transition
      - ramp: linear amplitude ramp
      - burst: short sinusoidal bursts
      - am   : amplitude-modulated tone (beats/tremolo-like)
      - noise: white noise plateaus
      - silence: short rests

    Notes:
      - Parameter ranges are intentionally broad to cover diverse dynamics.
      - rng controls reproducibility; pass a split-specific RNG to avoid leakage.

    Returns (B, T)
    """
    rng = rng or random.Random()
    x = torch.zeros(B, T)
    t_cur = 0
    n_segs = rng.randint(min_segments, max_segments)
    while t_cur < T and n_segs > 0:
        n_segs -= 1
        seg_type = rng.choices(
            population=["step", "ramp", "burst", "am", "noise", "silence"],
            weights=[0.2, 0.2, 0.2, 0.2, 0.15, 0.05],
            k=1,
        )[0]
        # Duration 30..300 ms (except silence 10..100 ms)
        if seg_type == "silence":
            dur_samp = int(rng.uniform(0.01, 0.1) * fs)
            seg = torch.zeros(B, dur_samp)
        elif seg_type == "step":
            dur_samp = int(rng.uniform(0.05, 0.4) * fs)
            at = rng.uniform(0.2, 0.8)
            a0 = rng.uniform(0.01, 0.2)
            a1 = rng.uniform(0.4, 0.9)
            seg = step(fs=fs, T=dur_samp, B=B, at=at, amp_before=a0, amp_after=a1)
        elif seg_type == "ramp":
            dur_samp = int(rng.uniform(0.05, 0.6) * fs)
            a0 = rng.uniform(0.01, 0.4)
            a1 = rng.uniform(max(a0 + 0.05, 0.1), 0.95)
            seg = ramp(fs=fs, T=dur_samp, B=B, start=0.0, dur=1.0, a0=a0, a1=a1)
        elif seg_type == "burst":
            dur_samp = int(rng.uniform(0.05, 0.3) * fs)
            freq = rng.choice([250.0, 500.0, 1000.0, 2000.0])
            ampv = rng.uniform(0.3, 0.9)
            # one burst at random place inside seg
            start = rng.uniform(0.1, 0.6)
            seg = burst(fs=fs, T=dur_samp, B=B, start=start, dur=0.3, amp=ampv, freq=freq)
        elif seg_type == "am":
            dur_samp = int(rng.uniform(0.2, 0.8) * fs)
            carrier = rng.uniform(200.0, 3000.0)
            rate = rng.uniform(1.0, 8.0)
            depth = rng.uniform(0.2, 0.8)
            ampv = rng.uniform(0.2, 0.8)
            seg = am_tone(fs=fs, T=dur_samp, B=B, carrier_hz=carrier, am_hz=rate, depth=depth, amp=ampv)
        elif seg_type == "noise":
            dur_samp = int(rng.uniform(0.1, 0.6) * fs)
            ampv = rng.uniform(0.05, 0.3)
            seg = white_noise(fs=fs, T=dur_samp, B=B, amp=ampv)
        else:
            seg = torch.zeros(B, 0)
            dur_samp = 0
        # Paste segment without exceeding T
        L = min(seg.shape[1], T - t_cur)
        if L > 0:
            x[:, t_cur : t_cur + L] = seg[:, :L]
            t_cur += L
    # If not filled, pad last value
    if t_cur < T and t_cur > 0:
        x[:, t_cur:] = x[:, t_cur - 1 : t_cur]
    return x

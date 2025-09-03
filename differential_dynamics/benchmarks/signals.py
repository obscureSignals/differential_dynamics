import torch
import math

def tone(freq: float, fs: int, T: int, B: int = 1, amp: float = 0.5) -> torch.Tensor:
    """Sine tone.
    Returns (B, T)
    """
    t = torch.arange(T).float() / fs
    x = amp * torch.sin(2 * math.pi * freq * t)[None, :].repeat(B, 1)
    return x

def step(fs: int, T: int, B: int = 1, at: float = 0.5, amp_before: float = 0.1, amp_after: float = 0.7) -> torch.Tensor:
    """Amplitude step at fraction 'at' of the clip.
    Returns (B, T)
    """
    t0 = int(T * at)
    x = torch.full((B, T), amp_before)
    x[:, t0:] = amp_after
    return x

def burst(fs: int, T: int, B: int = 1, start: float = 0.2, dur: float = 0.1, amp: float = 0.8, freq: float = 1000.0) -> torch.Tensor:
    """Sine burst.
    Returns (B, T)
    """
    s = int(T * start)
    L = min(T - s, int(T * dur))
    x = torch.zeros((B, T))
    if L > 0:
        n = torch.arange(L)
        x[:, s:s+L] = amp * torch.sin(2 * math.pi * freq * n / fs)[None, :]
    return x

def ramp(fs: int, T: int, B: int = 1, start: float = 0.2, dur: float = 0.4, a0: float = 0.1, a1: float = 0.8) -> torch.Tensor:
    """Linear amplitude ramp.
    Returns (B, T)
    """
    s = int(T * start)
    L = max(1, int(T * dur))
    slope = torch.linspace(a0, a1, L)
    x = torch.zeros((B, T)) + a0
    x[:, s:s+L] = slope
    x[:, s+L:] = a1
    return x


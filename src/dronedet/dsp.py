"""Small DSP primitives implemented directly with NumPy array operations."""

from __future__ import annotations

import numpy as np


def hann_periodic(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("window length must be at least 2")
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)).astype(np.float32)


def fft_radix2(x: np.ndarray) -> np.ndarray:
    """Iterative radix-2 DIT FFT along the final axis; no NumPy FFT calls."""
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim == 0:
        raise ValueError("FFT input must have at least one dimension")
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("FFT length must be a positive power of two")
    bits = n.bit_length() - 1
    indexes = np.arange(n, dtype=np.uint64)
    reversed_indexes = np.zeros(n, dtype=np.uint64)
    for bit in range(bits):
        reversed_indexes |= ((indexes >> bit) & 1) << (bits - bit - 1)
    out = x[..., reversed_indexes.astype(np.intp)].copy()
    size = 2
    while size <= n:
        leading_shape = out.shape[:-1]
        half = size // 2
        twiddles = np.exp(-2j * np.pi * np.arange(half) / size)
        blocks = out.reshape(*leading_shape, n // size, size)
        even = blocks[..., :half]
        odd = blocks[..., half:] * twiddles
        out = np.concatenate((even + odd, even - odd), axis=-1)
        out = out.reshape(*leading_shape, n)
        size *= 2
    return out


def ifft_radix2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.conjugate(fft_radix2(np.conjugate(x))) / x.shape[-1]


def rfft_scratch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return fft_radix2(x)[..., : x.shape[-1] // 2 + 1]


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """HTK mel triangular filters with constant-area normalization."""
    if sample_rate <= 0 or n_fft < 2 or n_fft & (n_fft - 1):
        raise ValueError("sample_rate must be positive and n_fft a power of two")
    if n_mels <= 0 or not 0 <= fmin < fmax <= sample_rate / 2:
        raise ValueError("invalid mel count or frequency range")
    frequencies = np.arange(n_fft // 2 + 1, dtype=np.float64) * sample_rate / n_fft
    points = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bank = np.zeros((n_mels, frequencies.size), dtype=np.float32)
    for index in range(n_mels):
        low, center, high = points[index:index + 3]
        if center <= low or high <= center:
            raise ValueError("mel points are not strictly increasing")
        rising = (frequencies - low) / (center - low)
        falling = (high - frequencies) / (high - center)
        bank[index] = np.maximum(0.0, np.minimum(rising, falling)) * (2.0 / (high - low))
    if np.any(bank.sum(axis=1) == 0):
        raise ValueError("empty mel filter; increase n_fft or lower n_mels")
    return bank
# reference only; not imported by production code

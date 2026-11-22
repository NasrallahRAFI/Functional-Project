"""Human-review plots for raw waveforms and native-resolution spectrograms."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .legacy_frontend import FrontEnd


def save_spectrogram(
    audio: np.ndarray,
    frontend: FrontEnd,
    output_path: str | Path,
    *,
    representation: str = "log_mel",
    title: str = "Audio spectrogram",
) -> None:
    """Render a spectrogram without resizing/interpolating the feature matrix."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to render spectrograms") from exc
    x = np.asarray(audio, dtype=np.float32)
    if representation == "log_mel":
        matrix = frontend.offline(x)
        y_label = "Mel bin"
    elif representation == "linear_log_stft":
        matrix = frontend.linear_log_stft(x)
        y_label = "Frequency bin"
    elif representation == "pcen":
        from .legacy_normalization import pcen

        matrix, _ = pcen(frontend.power(x))
        y_label = "Mel bin"
    else:
        raise ValueError("representation must be log_mel, linear_log_stft, or pcen")
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("audio is too short to produce a complete spectrogram frame")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    image = axis.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest")
    axis.set(title=title, xlabel="Frame", ylabel=y_label)
    figure.colorbar(image, ax=axis, label="dB" if representation != "pcen" else "PCEN")
    figure.savefig(target, dpi=150)
    plt.close(figure)

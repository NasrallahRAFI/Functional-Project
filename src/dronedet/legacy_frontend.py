"""Library based audio front end shared by file and windowed inference paths."""

from __future__ import annotations

import hashlib
import json
import warnings
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

warnings.warn("dronedet.legacy_frontend is deprecated; use dronedet.features", DeprecationWarning, stacklevel=2)
_log = logging.getLogger(__name__)

import librosa
import numpy as np


@dataclass(frozen=True)
class FrontEndConfig:
    """Resolved front-end settings. ``clip_length_s`` stays unset until the audit."""

    sample_rate: int = 16_000
    channel: int = 0
    n_fft: int = 1024
    hop: int = 256
    win_length: int = 1024
    window: str = "hann"
    center: bool = False
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 7000.0
    mel_scale: str = "htk"
    mel_norm: str = "area"
    power: float = 2.0
    amin: float = 1e-12
    top_db: float | None = None
    resample_quality: str = "HQ"
    clip_length_s: float | None = 0.5
    mode: str = "global_stats"
    pcen_time_constant: float = 0.4
    pcen_gain: float = 0.98
    pcen_bias: float = 2.0
    pcen_power: float = 0.5
    pcen_eps: float = 1e-6
    pcen_input_scale: float = float(2**31)
    measured_passband_hz: float | None = None
    preset: str = "project"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.channel < 0:
            raise ValueError("sample_rate must be positive and channel nonnegative")
        if self.n_fft < 2 or not 0 < self.hop <= self.n_fft:
            raise ValueError("n_fft must be >= 2 and hop must be within n_fft")
        if self.win_length != self.n_fft or self.window != "hann" or self.center:
            raise ValueError("project front end requires a full-length Hann window and center=False")
        if self.n_mels <= 0 or not 0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("invalid mel range or fmax exceeds Nyquist")
        if self.measured_passband_hz is not None and self.fmax > self.measured_passband_hz:
            raise ValueError("fmax exceeds the measured resampler passband")
        if self.mel_scale not in {"htk", "slaney"} or self.mel_norm not in {"area", "peak"}:
            raise ValueError("mel_scale must be htk/slaney and mel_norm area/peak")
        if self.power != 2.0 or self.amin <= 0 or self.top_db is not None:
            raise ValueError("project log-mel uses power=2, amin>0, and top_db=None")
        if self.resample_quality not in {"HQ", "VHQ", "MQ", "LQ"}:
            raise ValueError("unsupported soxr quality")
        if self.clip_length_s is not None and self.n_samples < self.n_fft:
            raise ValueError("clip length produces fewer samples than n_fft")
        if self.mode not in {"global_stats", "per_window_mean", "pcen", "logstft"}:
            raise ValueError("unsupported front-end mode")
        if self.preset not in {"project", "reference"}:
            raise ValueError("preset must be project or reference")
        if min(self.pcen_time_constant, self.pcen_gain, self.pcen_bias, self.pcen_power, self.pcen_eps) <= 0:
            raise ValueError("PCEN parameters must be positive")
        self.validate_filterbank()

    @property
    def n_samples(self) -> int:
        if self.clip_length_s is None:
            raise ValueError("clip_length_s awaits the dataset duration audit")
        return int(round(self.clip_length_s * self.sample_rate))

    @property
    def n_win(self) -> int:
        return 1 + (self.n_samples - self.n_fft) // self.hop

    @property
    def win_samples(self) -> int:
        return self.n_fft + (self.n_win - 1) * self.hop

    @property
    def sample_rate_hz(self) -> int:
        return self.sample_rate

    @property
    def db_floor(self) -> float:
        return 10.0 * np.log10(self.amin)

    @property
    def eps(self) -> float:
        return self.amin

    @property
    def unresolved_checks(self) -> tuple[str, ...]:
        pending = []
        if self.measured_passband_hz is None:
            pending.append("measure soxr HQ -1 dB passband and finalize fmax")
        if self.clip_length_s is None:
            pending.append("audit duration distribution and select clip_length_s")
        return tuple(pending)

    @classmethod
    def reference(cls, **overrides) -> "FrontEndConfig":
        values = {
            "preset": "reference", "sample_rate": 16_000, "n_fft": 1024,
            "win_length": 1024, "hop": 256, "n_mels": 64,
            "fmin": 50.0, "fmax": 8000.0, "measured_passband_hz": None,
            "mel_scale": "htk", "mel_norm": "area", "amin": 1e-12,
            "clip_length_s": 33_536 / 16_000, "mode": "global_stats",
        }
        values.update(overrides)
        return cls(**values)

    def validate_filterbank(self) -> dict[str, float | int]:
        library_norm = np.inf if self.mel_norm == "peak" else "slaney"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            bank = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                htk=self.mel_scale == "htk",
                norm=library_norm,
                dtype=np.float32,
            )
        counts = np.count_nonzero(bank, axis=1)
        if np.any(counts < 2):
            raise ValueError("every mel filter must have at least two nonzero FFT bins")
        edges = librosa.mel_frequencies(
            n_mels=self.n_mels + 2, fmin=self.fmin, fmax=self.fmax,
            htk=self.mel_scale == "htk",
        )
        spacing = self.sample_rate / self.n_fft
        half_width = float(edges[1] - edges[0])
        if half_width < spacing:
            raise ValueError("lowest mel filter half-width is below one FFT bin")
        return {"minimum_nonzero_bins": int(counts.min()), "lowest_half_width_hz": half_width,
                "fft_bin_spacing_hz": spacing, "margin_bins": half_width / spacing}


class FrontEnd:
    """Compute power mel, log-mel and the reference linear log-STFT."""

    def __init__(self, config: FrontEndConfig | None = None) -> None:
        _log.warning("legacy front end, output differs from dronedet.features, not for results")
        self.config = config or FrontEndConfig()
        self.mel = librosa.filters.mel(
            sr=self.config.sample_rate, n_fft=self.config.n_fft,
            n_mels=self.config.n_mels, fmin=self.config.fmin, fmax=self.config.fmax,
            htk=self.config.mel_scale == "htk",
            norm=np.inf if self.config.mel_norm == "peak" else "slaney", dtype=np.float32,
        )
        self.reset()

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)

    def _mono(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("front end expects selected mono one-dimensional audio")
        if not np.all(np.isfinite(x)):
            raise ValueError("audio contains NaN or infinite values")
        return x

    @staticmethod
    def quality_flags(audio: np.ndarray) -> dict[str, bool]:
        """Report silence and clipping without rejecting finite clipped audio."""
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("quality checks expect mono one-dimensional audio")
        if not np.all(np.isfinite(x)):
            raise ValueError("audio contains NaN or infinite values")
        silent = bool(x.size == 0 or np.all(x == 0))
        clipped = bool(x.size and np.count_nonzero(np.abs(x) >= 0.999) / x.size > 0.01)
        return {"silent": silent, "clipped": clipped}

    def power(self, audio: np.ndarray) -> np.ndarray:
        x = self._mono(audio)
        if x.size < self.config.n_fft:
            return np.empty((self.config.n_mels, 0), dtype=np.float32)
        mel = librosa.feature.melspectrogram(
            y=x, sr=self.config.sample_rate, n_fft=self.config.n_fft,
            hop_length=self.config.hop, win_length=self.config.win_length,
            window="hann", center=False, power=2.0,
            n_mels=self.config.n_mels, fmin=self.config.fmin, fmax=self.config.fmax,
            htk=self.config.mel_scale == "htk", norm=np.inf if self.config.mel_norm == "peak" else "slaney", dtype=np.float32,
        )
        return np.asarray(mel, dtype=np.float32)

    def offline(self, audio: np.ndarray) -> np.ndarray:
        power = self.power(audio)
        db = librosa.power_to_db(power, ref=1.0, amin=self.config.amin, top_db=None)
        return np.asarray(db, dtype=np.float32)

    def linear_log_stft(self, audio: np.ndarray) -> np.ndarray:
        x = self._mono(audio)
        if x.size < self.config.n_fft:
            return np.empty((self.config.n_fft // 2 + 1, 0), dtype=np.float32)
        spectrum = np.abs(librosa.stft(
            x, n_fft=self.config.n_fft, hop_length=self.config.hop,
            win_length=self.config.win_length, window="hann", center=False,
        )) ** 2
        return np.asarray(librosa.power_to_db(spectrum, ref=1.0, amin=self.config.amin, top_db=None), dtype=np.float32)

    def process_window(self, audio: np.ndarray, *, stats: dict | None = None) -> np.ndarray:
        """Return a configured window tensor shaped (1, mel, time)."""
        cfg = self.config
        if cfg.clip_length_s is None:
            raise ValueError("clip_length_s awaits the dataset duration audit")
        x = self._mono(audio)
        if x.size < cfg.win_samples:
            if x.size < cfg.win_samples / 2:
                raise ValueError("audio is shorter than half a usable window")
            x = np.pad(x, (0, cfg.win_samples - x.size), mode="wrap")
        else:
            start = (x.size - cfg.win_samples) // 2
            x = x[start:start + cfg.win_samples]
        power = self.power(x)
        if cfg.mode == "pcen":
            magnitude = np.sqrt(power) * np.float32(cfg.pcen_input_scale)
            smoother = 1.0 - np.exp(-cfg.hop / (cfg.sample_rate * cfg.pcen_time_constant))
            zi = ((1.0 - smoother) * magnitude.mean(axis=-1, keepdims=True)).astype(np.float32)
            result = librosa.pcen(magnitude, sr=cfg.sample_rate, hop_length=cfg.hop,
                                  time_constant=cfg.pcen_time_constant, gain=cfg.pcen_gain,
                                  bias=cfg.pcen_bias, power=cfg.pcen_power, eps=cfg.pcen_eps,
                                  b=smoother, zi=zi, axis=-1)
            if stats is None:
                raise ValueError("PCEN scalar training statistics are required")
            result = (result - float(stats["mean"])) / float(stats["std"])
        else:
            db = self.offline(x)
            if cfg.mode == "per_window_mean":
                if stats is None:
                    raise ValueError("training global standard deviation is required")
                result = (db - float(db.mean())) / float(stats["std"])
            elif cfg.mode == "global_stats":
                if stats is None:
                    raise ValueError("training per-bin statistics are required")
                result = (db - np.asarray(stats["mean"], np.float32)[:, None]) / np.asarray(stats["std"], np.float32)[:, None]
            else:
                result = self.linear_log_stft(x)
        return np.asarray(result[None, ...], dtype=np.float32)

    def push(self, pcm16: np.ndarray) -> np.ndarray:
        """Compatibility adapter; callers should resample then provide window-sized audio."""
        pcm = np.asarray(pcm16)
        if pcm.ndim != 1 or pcm.dtype != np.int16:
            raise ValueError("push expects a one-dimensional int16 PCM chunk")
        audio = np.concatenate((self._buffer, pcm.astype(np.float32) / 32768.0))
        frames = 0 if audio.size < self.config.n_fft else 1 + (audio.size - self.config.n_fft) // self.config.hop
        consumed = frames * self.config.hop
        self._buffer = audio[consumed:].copy()
        return self.offline(audio[:self.config.n_fft + max(frames - 1, 0) * self.config.hop])


def load_frontend_config(path: str | Path) -> FrontEndConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        text = stream.read()
    if Path(path).suffix.lower() in {".yaml", ".yml"}:
        import yaml
        raw = yaml.safe_load(text)
        raw = raw.get("frontend", raw)
    else:
        raw = json.loads(text)
    aliases = {"mel_normalization": "mel_norm", "sample_rate_hz": "sample_rate"}
    if "sr" in raw and "sample_rate" not in raw:
        raw["sample_rate"] = raw["sr"]
    if "L" in raw and "clip_length_s" not in raw:
        raw["clip_length_s"] = raw["L"]
    for old, new in aliases.items():
        if old in raw and new not in raw:
            raw[new] = {"constant_area": "area", "slaney": "area", "peak": "peak"}.get(raw[old], raw[old]) if old == "mel_normalization" else raw[old]
    preset = raw.get("preset", "project")
    allowed = set(FrontEndConfig.__dataclass_fields__)
    if preset == "reference":
        values = asdict(FrontEndConfig.reference())
    else:
        values = asdict(FrontEndConfig())
    values.update({key: value for key, value in raw.items() if key in allowed})
    values["preset"] = preset
    if "db_floor" in raw:
        values["amin"] = 10.0 ** (float(raw["db_floor"]) / 10.0)
    if "fmax" not in values:
        values["fmax"] = 7000.0
    return FrontEndConfig(**values)


def frontend_hash(config: FrontEndConfig, stats: dict | None = None) -> str:
    """Hash canonical parameters and float32 normalization-stat bytes."""
    payload = asdict(config)
    if config.clip_length_s is not None:
        payload.update(n_win=config.n_win, win_samples=config.win_samples)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    if stats is not None:
        for key in ("mean", "std"):
            if key in stats:
                digest.update(np.asarray(stats[key], dtype=np.float32).tobytes())
        for key in ("scalar_mean", "scalar_std"):
            if key in stats:
                digest.update(np.asarray([stats[key]], dtype=np.float32).tobytes())
    return digest.hexdigest()

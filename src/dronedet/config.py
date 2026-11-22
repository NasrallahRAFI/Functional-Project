"""Validated YAML configuration for the shared drone-audio front end."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import yaml


@dataclass(frozen=True)
class FrontEndConfig:
    preset: str = "project"
    sr: int = 16_000
    channel: int = 0
    n_fft: int = 1024
    win_length: int = 1024
    hop: int = 256
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
    L: float = 0.5
    mode: str = "global_stats"
    measured_passband_hz: float | None = None
    passband_status: str = "interim"
    pcen_time_constant: float = 0.4
    pcen_gain: float = 0.98
    pcen_bias: float = 2.0
    pcen_power: float = 0.5
    pcen_eps: float = 1e-6
    pcen_input_scale: float = float(2**31)
    gain_range_db: tuple[float, float] = (-12.0, 12.0)
    near_silent_dbfs: float = -80.0
    clipped_peak: float = 0.999
    clipped_fraction: float = 0.01
    manifest_seed: int = 42
    decision_interval_s: float = 0.5
    smoothing_windows: int = 3
    passband_override_reason: str | None = None

    def __post_init__(self) -> None:
        if self.preset not in {"project", "reference"}:
            raise ValueError("preset must be 'project' or 'reference'")
        if self.sr <= 0 or self.channel < 0:
            raise ValueError("sr must be positive and channel nonnegative")
        if self.n_fft < 2 or self.win_length != self.n_fft or not 0 < self.hop <= self.n_fft:
            raise ValueError("n_fft/win_length must match and hop must be in (0, n_fft]")
        if self.window != "hann" or self.center:
            raise ValueError("the project front end requires window='hann' and center=False")
        if self.L <= 0 or self.n_samples < self.n_fft:
            raise ValueError("L*sr must provide at least one FFT window")
        if self.n_mels < 2 or not 0 <= self.fmin < self.fmax <= self.nyquist_hz:
            raise ValueError("invalid mel range or fmax above Nyquist")
        if self.measured_passband_hz is not None and self.fmax > self.measured_passband_hz:
            raise ValueError("fmax exceeds measured resampler passband")
        if self.passband_status not in {"interim", "measured"}:
            raise ValueError("passband_status must be 'interim' or 'measured'")
        if self.passband_status == "measured" and self.measured_passband_hz is None:
            raise ValueError("measured passband status requires measured_passband_hz")
        if self.mel_scale not in {"htk", "slaney"} or self.mel_norm not in {"area", "peak"}:
            raise ValueError("mel_scale must be htk/slaney and mel_norm area/peak")
        if self.power != 2.0 or self.amin != 1e-12 or self.top_db is not None:
            raise ValueError("V8 uses power=2, amin=1e-12 and top_db=None")
        if self.resample_quality not in {"HQ", "VHQ", "MQ", "LQ"}:
            raise ValueError("unsupported soxr quality")
        if self.mode not in {"global_stats", "per_window_mean", "pcen", "logstft"}:
            raise ValueError("unsupported front-end mode")
        if min(self.pcen_time_constant, self.pcen_gain, self.pcen_bias, self.pcen_power, self.pcen_eps) <= 0:
            raise ValueError("PCEN parameters must be positive")
        if len(self.gain_range_db) != 2 or self.gain_range_db[0] >= self.gain_range_db[1]:
            raise ValueError("gain_range_db must be an increasing pair")
        if not 0 < self.clipped_peak <= 1 or not 0 < self.clipped_fraction <= 1:
            raise ValueError("invalid clipping thresholds")
        if self.passband_override_reason is not None and not self.passband_override_reason.strip():
            raise ValueError("passband override reason must not be blank")
        self.validate_filterbank()
        if self.passband_status == "interim":
            warnings.warn("fmax is interim until the soxr passband measurement is recorded", UserWarning, stacklevel=2)

    @property
    def n_samples(self) -> int:
        return int(round(self.L * self.sr))

    @property
    def n_win(self) -> int:
        return 1 + (self.n_samples - self.n_fft) // self.hop

    @property
    def win_samples(self) -> int:
        return self.n_fft + (self.n_win - 1) * self.hop

    @property
    def nyquist_hz(self) -> float:
        return self.sr / 2

    @property
    def db_floor(self) -> float:
        return 10.0 * np.log10(self.amin)

    @classmethod
    def reference(cls, **overrides: Any) -> "FrontEndConfig":
        """Resolve V8's 128-frame reference preset, then apply explicit overrides."""
        values = _preset_values("reference")
        values.update(overrides)
        return cls(**values)

    # Compatibility aliases used by the old drone_audio package.
    @property
    def sample_rate(self) -> int:
        return self.sr

    @property
    def sample_rate_hz(self) -> int:
        return self.sr

    @property
    def clip_length_s(self) -> float:
        return self.L

    @property
    def eps(self) -> float:
        return self.amin

    def validate_filterbank(self) -> dict[str, float | int]:
        lib_norm: str | float = np.inf if self.mel_norm == "peak" else "slaney"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            bank = librosa.filters.mel(
                sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                fmin=self.fmin, fmax=self.fmax, htk=self.mel_scale == "htk",
                norm=lib_norm, dtype=np.float32,
            )
        counts = np.count_nonzero(bank, axis=1)
        edges = librosa.mel_frequencies(
            n_mels=self.n_mels + 2, fmin=self.fmin, fmax=self.fmax,
            htk=self.mel_scale == "htk",
        )
        # V8 rule 6b.6-3(c) uses m1: centre minus lower edge.
        lowest_half_width = float(edges[1] - edges[0])
        bin_spacing = self.sr / self.n_fft
        diagnostics = (f"minimum_nonzero_bins={int(counts.min())}, "
                       f"filters_with_fewer_than_2={int(np.count_nonzero(counts < 2))}, "
                       f"filters_with_fewer_than_3={int(np.count_nonzero(counts < 3))}, "
                       f"margin_bins={lowest_half_width / bin_spacing:.3f}")
        if np.any(counts == 0):
            raise ValueError(f"empty mel filter; {diagnostics}")
        if np.any(counts < 2):
            raise ValueError(f"mel starvation: fewer than 2 bins in a filter; {diagnostics}")
        if lowest_half_width < bin_spacing:
            raise ValueError("lowest mel-filter half-width is below FFT-bin spacing")
        return {
            "minimum_nonzero_bins": int(counts.min()),
            "filters_with_fewer_than_3_bins": int(np.count_nonzero(counts < 3)),
            "lowest_half_width_hz": lowest_half_width,
            "fft_bin_spacing_hz": bin_spacing,
            "margin_bins": lowest_half_width / bin_spacing,
        }

    def validate_cache_write(self, *, logged_override: str | None = None) -> None:
        if self.passband_status == "measured":
            return
        reason = logged_override or self.passband_override_reason
        if not reason:
            raise ValueError("cache writes require passband_status='measured' or an explicit logged override")


def _preset_values(preset: str) -> dict[str, Any]:
    if preset == "project":
        return {"preset": "project"}
    if preset == "reference":
        return {
            "preset": "reference", "sr": 16_000, "channel": 0,
            "n_fft": 1024, "win_length": 1024, "hop": 256,
            "window": "hann", "center": False, "n_mels": 64,
            "fmin": 50.0, "fmax": 8000.0, "mel_scale": "htk",
            "mel_norm": "area", "power": 2.0, "amin": 1e-12,
            "top_db": None, "resample_quality": "HQ",
            "L": 33_536 / 16_000, "mode": "global_stats",
            "passband_status": "interim", "measured_passband_hz": None,
        }
    raise ValueError(f"unknown front-end preset: {preset}")


def config_from_mapping(raw: dict[str, Any], *, preset: str | None = None) -> FrontEndConfig:
    section = raw.get("frontend", raw)
    selected = preset or str(section.get("preset", raw.get("preset", "project")))
    values = _preset_values(selected)
    aliases = {
        "sample_rate": "sr", "sample_rate_hz": "sr", "clip_length_s": "L",
        "mel_normalization": "mel_norm", "quality": "resample_quality",
    }
    normalized: dict[str, Any] = {}
    for key, value in section.items():
        normalized[aliases.get(key, key)] = value
    if "mel_norm" in normalized:
        normalized["mel_norm"] = {"constant_area": "area", "slaney": "area"}.get(normalized["mel_norm"], normalized["mel_norm"])
    values.update(normalized)
    if "gain_range_db" in values:
        values["gain_range_db"] = tuple(values["gain_range_db"])
    known = set(FrontEndConfig.__dataclass_fields__)
    return FrontEndConfig(**{key: value for key, value in values.items() if key in known})


def load_config(path: str | Path, *, preset: str | None = None) -> FrontEndConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a YAML mapping")
    return config_from_mapping(raw, preset=preset)


def load_project_config(path: str | Path) -> dict[str, Any]:
    """Load the data-preparation paths and cleaning settings beside the front-end config."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    profile = "colab" if "google.colab" in __import__("sys").modules else "local"
    selected = {key: Path(value) for key, value in raw["paths"][profile].items()}
    root = selected["project_root"]
    return {
        **raw,
        "P": {
            "root": root,
            "raw_root": selected["raw_root"],
            "interim": root / "data" / "interim",
            "splits": root / "data" / "processed" / "splits",
            "tab": root / "data" / "processed" / "features_tabular",
            "spec": root / "data" / "processed" / "spectrograms",
            "golden": root / "data" / "processed" / "golden",
            "tables": root / "results" / "tables",
            "figures": root / "results" / "figures",
        },
        "fe": load_config(path),
    }


def validate_frontend(config: FrontEndConfig, passband_limit_hz: float | None = None) -> dict[str, float | int]:
    if passband_limit_hz is not None and config.fmax > passband_limit_hz:
        raise ValueError("fmax exceeds the measured resampler passband")
    details = config.validate_filterbank()
    return {"min_nonzero_bins": details["minimum_nonzero_bins"],
            "lowest_filter_margin": round(float(details["margin_bins"]), 3),
            "n_win": config.n_win, "win_samples": config.win_samples}


def params_hash(config: FrontEndConfig) -> str:
    """Stable front-end parameter hash, including derived window dimensions."""
    return config_hash(config)


def cache_content_hash(config: FrontEndConfig) -> str:
    """Hash only settings that change the stored full-clip mel-power cache.

    Mode and PCEN settings are intentionally excluded: they transform cached
    power after it has been read and therefore do not change cache contents.
    Keep this list aligned with ``cache_split_power`` and its audio/front-end
    helpers.
    """
    keys = (
        "sr", "channel", "n_fft", "win_length", "hop", "window", "center",
        "n_mels", "fmin", "fmax", "mel_scale", "mel_norm", "power", "L",
        "resample_quality",
    )
    payload = {key: getattr(config, key) for key in keys}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_not_raw(path: str | Path, config: dict[str, Any]) -> Path:
    """Create an output parent only after confirming the target is outside raw_root."""
    target = Path(path).resolve()
    raw_root = Path(config["P"]["raw_root"]).resolve()
    if target == raw_root or raw_root in target.parents:
        raise PermissionError(f"refusing to write inside the raw dataset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def config_hash(config: FrontEndConfig, stats: dict[str, Any] | None = None) -> str:
    payload = asdict(config)
    payload.update(n_win=config.n_win, win_samples=config.win_samples, F_MAX=config.fmax, mode=config.mode)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if stats is not None:
        for key in ("mean", "std"):
            if key in stats:
                digest.update(np.asarray(stats[key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def write_environment_log(path: str | Path, *, dataset_sha256: str | None = None,
                         dataset_zip_sha256: str | None = None,
                         train_manifest_sha256: str | None = None,
                         frontend_hash: str | None = None,
                         commit: str | None = None) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version
    names = ("numpy", "scipy", "librosa", "soxr", "soundfile", "scikit-learn", "torch", "pandas", "matplotlib", "PyYAML")
    packages = {}
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = {"python": __import__("sys").version, "platform": __import__("platform").platform(),
              "packages": packages, "dataset_sha256": dataset_sha256,
              "dataset_zip_sha256": dataset_zip_sha256,
              "frozen_train_manifest_sha256": train_manifest_sha256,
              "frontend_config_hash": frontend_hash, "repository_commit": commit}
    Path(path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

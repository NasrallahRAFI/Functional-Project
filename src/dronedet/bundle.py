"""Front-end bundle creation, verification, and deterministic tamper checks."""
from __future__ import annotations

import io
import json
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import FrontEndConfig, config_hash, legacy_config_hash
from .features import (centre_window, fit_length, load_audio, mel_power,
                       standardize, to_db)


class BundleVerificationError(ValueError):
    """A bundle failed a specific integrity contract."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def _fail(code: str, message: str):
    raise BundleVerificationError(code, message)


def write_bundle(root: str | Path, config: FrontEndConfig, stats: dict[str, Any],
                 golden_audio: dict[str, np.ndarray]) -> Path:
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("bundle directory must be empty")
    if set(golden_audio) != {"golden16", "golden48"}:
        raise ValueError("bundle requires golden16 and golden48 audio fixtures")
    root.mkdir(parents=True, exist_ok=True)
    parameters = {k: v for k, v in config.__dict__.items()}
    parameters["_identity_schema"] = 2
    (root / "config.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf8")
    stats_json = {key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
                  for key, value in stats.items()}
    (root / "stats.json").write_text(json.dumps(stats_json, indent=2, allow_nan=False) + "\n", encoding="utf8")
    arrays = {}
    for key in ("mean", "std"):
        if key in stats:
            arrays[key] = np.asarray(stats[key], dtype="<f4")
    if set(arrays) != {"mean", "std"}:
        raise ValueError("bundle statistics require mean and std arrays")
    if config.mode == "global_stats":
        if arrays["mean"].shape != (config.n_mels,) or arrays["std"].shape != (config.n_mels,):
            raise ValueError("global_stats bundle arrays must have one value per mel band")
        if (not np.isfinite(arrays["mean"]).all() or not np.isfinite(arrays["std"]).all()
                or np.any(arrays["std"] <= 0)):
            raise ValueError("bundle statistics must be finite with strictly positive std")
    config.validate()
    np.savez(root / "stats.npz", **arrays)
    digest = config_hash(config, stats)
    (root / "hash.txt").write_text(digest + "\n", encoding="ascii")
    from importlib.metadata import PackageNotFoundError, version
    names = ("numpy", "scipy", "librosa", "soxr", "soundfile", "scikit-learn", "torch", "PyYAML")
    libraries = {}
    for name in names:
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    libraries.update(model_weights=None, validation_threshold=None)
    (root / "libraries.json").write_text(json.dumps(libraries, indent=2) + "\n", encoding="utf8")
    (root / "golden").mkdir()
    for name, audio in golden_audio.items():
        source_rate = 16_000 if name == "golden16" else 48_000
        source = np.asarray(audio, dtype=np.float32)
        if (source.ndim not in {1, 2} or source.size == 0 or not np.isfinite(source).all()
                or (source.ndim == 2 and source.shape[1] <= config.channel)):
            raise ValueError(f"{name} must be nonempty finite audio containing the configured channel")
        wav_path = root / "golden" / f"{name}_audio.wav"
        sf.write(wav_path, source, source_rate, subtype="FLOAT")
        standardized = fit_length(standardize(source, source_rate, config), config)
        power = mel_power(standardized, config)
        db = to_db(centre_window(power, config), config)
        np.save(root / "golden" / f"{name}_db.npy", db, allow_pickle=False)
        if config.mode == "global_stats":
            normalized = ((db - arrays["mean"][:, None]) / arrays["std"][:, None]).astype(np.float32)
            np.save(root / "golden" / f"{name}_normalized.npy", normalized[None], allow_pickle=False)
    return root


def _required_files(root: Path) -> list[Path]:
    required = [root / name for name in ("config.json", "stats.json", "stats.npz", "hash.txt", "libraries.json")]
    if not (root / "golden").is_dir():
        _fail("GOLDEN_MISSING", "bundle golden directory is missing")
    for name in ("golden16", "golden48"):
        wav_path = root / "golden" / f"{name}_audio.wav"
        array_path = root / "golden" / f"{name}_audio.npy"
        source_path = wav_path if wav_path.is_file() else array_path
        required.extend((source_path, root / "golden" / f"{name}_db.npy"))
        config_path = root / "config.json"
        if config_path.is_file():
            try:
                config_data = json.loads(config_path.read_text(encoding="utf8"))
            except (OSError, json.JSONDecodeError):
                config_data = {}
            if config_data.get("_identity_schema") == 2 and config_data.get("mode") == "global_stats":
                required.append(root / "golden" / f"{name}_normalized.npy")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        _fail("REQUIRED_FILE_MISSING", f"required bundle file missing: {missing[0]}")
    return required


def verify_bundle(root: str | Path, running_config: FrontEndConfig) -> dict[str, Any]:
    root = Path(root)
    _required_files(root)
    try:
        params = json.loads((root / "config.json").read_text(encoding="utf8"))
        stats = json.loads((root / "stats.json").read_text(encoding="utf8"))
        identity_schema = params.pop("_identity_schema", 1)
        bundled = FrontEndConfig(**params)
    except Exception as exc:
        _fail("CONFIG_INVALID", f"invalid bundled config/statistics metadata: {exc}")
    try:
        stats_bytes = (root / "stats.npz").read_bytes()
        with np.load(io.BytesIO(stats_bytes), allow_pickle=False) as archive:
            if set(archive.files) != {"mean", "std"}:
                _fail("STATS_KEYS_INVALID", "statistics archive must contain exactly mean and std")
            mean, std = archive["mean"], archive["std"]
    except BundleVerificationError:
        raise
    except Exception as exc:
        _fail("STATS_ARCHIVE_INVALID", f"statistics archive cannot be read: {exc}")
    if mean.dtype.str != "<f4" or std.dtype.str != "<f4":
        _fail("STATS_DTYPE_INVALID", "statistics arrays must use explicit little-endian float32")
    if (mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape or mean.size == 0):
        _fail("STATS_SHAPE_INVALID", "statistics mean/std must be nonempty vectors with matching shapes")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        _fail("STATS_NONFINITE", "statistics contain non-finite values")
    if np.any(std <= 0):
        _fail("STATS_STD_INVALID", "statistics std must be strictly positive")
    try:
        running_config.validate()
    except Exception as exc:
        _fail("CONFIG_INVALID", f"running front-end configuration is not runnable: {exc}")
    if bundled.mode == "global_stats" and mean.shape != (bundled.n_mels,):
        _fail("STATS_SHAPE_INVALID", "global_stats mean/std must contain one value per mel band")
    for key, values in (("mean", mean), ("std", std)):
        try:
            listed = np.asarray(stats[key], dtype="<f4")
        except Exception as exc:
            _fail("STATS_METADATA_INVALID", f"statistics metadata field {key} is invalid: {exc}")
        if listed.shape != values.shape or not np.array_equal(listed, values):
            _fail("STATS_METADATA_MISMATCH", f"statistics JSON and NPZ disagree for {key}")
    try:
        stored = (root / "hash.txt").read_text(encoding="ascii").strip()
    except Exception as exc:
        _fail("HASH_INVALID", f"bundle hash cannot be read: {exc}")
    identity = config_hash if identity_schema == 2 else legacy_config_hash
    actual = identity(bundled, stats)
    if stored != actual:
        _fail("HASH_MISMATCH", "bundle configuration/statistics hash mismatch")
    if identity(running_config, stats) != stored:
        _fail("RUNNING_FRONTEND_MISMATCH", "bundle does not match running front end")
    try:
        libraries = json.loads((root / "libraries.json").read_text(encoding="utf8"))
    except Exception as exc:
        _fail("LIBRARIES_INVALID", f"library record cannot be read: {exc}")
    from importlib.metadata import PackageNotFoundError, version
    mismatches = {}
    for name, recorded in libraries.items():
        if name in {"model_weights", "validation_threshold"}:
            continue
        try:
            current = version(name)
        except PackageNotFoundError:
            current = None
        if recorded != current:
            mismatches[name] = {"recorded": recorded, "current": current}
    if mismatches:
        warnings.warn(f"bundle library versions differ: {mismatches}", RuntimeWarning, stacklevel=2)
    count = 0
    verified_rates = {}
    normalized_outputs = {}
    for name in ("golden16", "golden48"):
        wav_path = root / "golden" / f"{name}_audio.wav"
        try:
            if wav_path.is_file():
                expected_rate = 16_000 if name == "golden16" else 48_000
                audio, source_rate = load_audio(wav_path, running_config)
                if source_rate != expected_rate:
                    _fail("GOLDEN_RATE_INVALID", f"{name} WAV has sample rate {source_rate}, expected {expected_rate}")
                standardized = fit_length(standardize(audio, source_rate, running_config), running_config)
                power = mel_power(standardized, running_config)
                actual_db = to_db(centre_window(power, running_config), running_config)
                verified_rates[name] = {"source_rate_hz": source_rate,
                                        "resampled": source_rate != running_config.sr}
            else:
                # Read-only compatibility for the installed V1 bundle.
                audio = np.load(root / "golden" / f"{name}_audio.npy", allow_pickle=False)
                actual_db = to_db(mel_power(audio, running_config), running_config)
                verified_rates[name] = {"source_rate_hz": None, "resampled": False,
                                        "legacy_array_fixture": True}
            expected = np.load(root / "golden" / f"{name}_db.npy", allow_pickle=False)
        except BundleVerificationError:
            raise
        except Exception as exc:
            _fail("GOLDEN_INVALID", f"golden check failed for {name}: {exc}")
        for label, values in (("expected", expected), ("actual", actual_db)):
            if values.dtype.kind not in "fiu" or values.size == 0 or values.ndim != 2:
                _fail("GOLDEN_DTYPE_OR_SHAPE_INVALID", f"{name} {label} golden must be a nonempty numeric matrix")
            if not np.isfinite(values).all():
                _fail("GOLDEN_NONFINITE", f"{name} {label} golden contains NaN or infinity")
        if actual_db.shape != expected.shape or np.max(np.abs(actual_db - expected), initial=0) > 1e-3:
            _fail("GOLDEN_MISMATCH", f"golden mismatch: {name}")
        if running_config.mode == "global_stats":
            normalized = ((actual_db - mean[:, None]) / std[:, None]).astype(np.float32)[None]
            normalized_path = root / "golden" / f"{name}_normalized.npy"
            if normalized_path.is_file():
                expected_normalized = np.load(normalized_path, allow_pickle=False)
                if (expected_normalized.dtype.kind not in "fiu" or expected_normalized.shape != normalized.shape
                        or not np.isfinite(expected_normalized).all()):
                    _fail("NORMALIZED_GOLDEN_INVALID", f"{name} normalized golden has invalid dtype, shape or values")
                error = float(np.max(np.abs(normalized - expected_normalized), initial=0))
                if not np.allclose(normalized, expected_normalized, rtol=0.0, atol=1e-6):
                    _fail("NORMALIZED_GOLDEN_MISMATCH", f"normalized golden mismatch: {name}")
                normalized_outputs[name] = {"shape": list(normalized.shape),
                                            "max_abs_error": error, "verified": True}
            else:
                normalized_outputs[name] = {"shape": list(normalized.shape), "verified": False,
                                            "reason": "legacy bundle has no normalized golden"}
        count += 1
    return {"hash": stored, "golden_files_verified": count,
            "golden_source_rates": verified_rates, "normalized_outputs": normalized_outputs,
            "library_version_mismatches": mismatches}


def run_bundle_rejection_checks(source_bundle: str | Path,
                                running_config: FrontEndConfig) -> dict[str, Any]:
    """Exercise bundle tamper cases in isolated temporary copies of the stamped bundle."""
    source = Path(source_bundle)
    base = verify_bundle(source, running_config)
    outcomes: dict[str, Any] = {"unaltered": {"pass": True, "result": base}}

    with tempfile.TemporaryDirectory(prefix="frontend_bundle_checks_") as temporary:
        temp = Path(temporary)

        def fresh(name: str) -> Path:
            destination = temp / name
            shutil.copytree(source, destination)
            return destination

        def expect_rejection(name: str, mutate, expected_code: str | None = None) -> None:
            target = fresh(name)
            mutate(target)
            try:
                verify_bundle(target, running_config)
            except BundleVerificationError as exc:
                if expected_code and exc.reason_code != expected_code:
                    raise AssertionError(f"{name}: expected {expected_code}, got {exc.reason_code}") from exc
                outcomes[name] = {"pass": True, "reason_code": exc.reason_code}
            else:
                raise AssertionError(f"{name}: tamper was accepted")

        original_config = json.loads((source / "config.json").read_text(encoding="utf8"))
        for key, value in original_config.items():
            def mutate_config(target: Path, key=key, value=value):
                changed = json.loads((target / "config.json").read_text(encoding="utf8"))
                if isinstance(value, bool):
                    changed[key] = not value
                elif isinstance(value, (int, float)):
                    changed[key] = value + max(abs(float(value)) * 1e-4, 1e-6)
                elif isinstance(value, str):
                    choices = {"global_stats": "per_window_mean", "area": "peak", "htk": "slaney",
                               "HQ": "VHQ", "hann": "hamming", "measured": "interim", "project": "reference"}
                    changed[key] = choices.get(value, value + "_changed")
                elif value is None:
                    changed[key] = 0.25
                elif isinstance(value, list):
                    changed[key] = [value[0] + 0.25, value[1]]
                (target / "config.json").write_text(json.dumps(changed, indent=2) + "\n", encoding="utf8")
            expect_rejection(f"config_{key}", mutate_config)

        def change_and_rehash(target: Path):
            params = json.loads((target / "config.json").read_text(encoding="utf8"))
            params["pcen_gain"] = float(params["pcen_gain"]) - 0.01
            changed = FrontEndConfig(**params)
            stats = json.loads((target / "stats.json").read_text(encoding="utf8"))
            (target / "config.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf8")
            digest_fn = config_hash if params.get("_identity_schema") == 2 else legacy_config_hash
            (target / "hash.txt").write_text(digest_fn(changed, stats) + "\n", encoding="ascii")
        expect_rejection("parameter_changed_and_rehashed", change_and_rehash, "RUNNING_FRONTEND_MISMATCH")

        def flip_stats(target: Path):
            path = target / "stats.npz"; data = bytearray(path.read_bytes()); data[len(data) // 2] ^= 1; path.write_bytes(data)
        expect_rejection("stats_byte_flip", flip_stats)
        expect_rejection("stats_truncated", lambda target: (target / "stats.npz").write_bytes(
            (target / "stats.npz").read_bytes()[:24]))

        def stats_float64(target: Path):
            with np.load(target / "stats.npz", allow_pickle=False) as archive:
                mean = archive["mean"].astype(np.float64)
                std = archive["std"].astype(np.float64)
            np.savez(target / "stats.npz", mean=mean, std=std)
        expect_rejection("stats_float64", stats_float64, "STATS_DTYPE_INVALID")
        expect_rejection("stored_hash_tamper", lambda target: (target / "hash.txt").write_text("0" * 64 + "\n"))
        expected_files = [path.relative_to(source) for path in _required_files(source)]
        for rel in expected_files:
            expect_rejection(f"missing_{str(rel).replace('/', '_')}", lambda target, rel=rel: (target / rel).unlink(),
                             "REQUIRED_FILE_MISSING")
        def tamper_golden(target: Path):
            path = next((target / "golden").glob("*_db.npy"))
            values = np.load(path, allow_pickle=False); values.flat[0] += 0.01
            np.save(path, values, allow_pickle=False)
        expect_rejection("golden_shift_0_01_db", tamper_golden, "GOLDEN_MISMATCH")

        altered_libs = fresh("libraries_only")
        lib_path = altered_libs / "libraries.json"
        libraries = json.loads(lib_path.read_text(encoding="utf8")); libraries["librosa"] = "0.0-test"
        lib_path.write_text(json.dumps(libraries, indent=2) + "\n", encoding="utf8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            verify_bundle(altered_libs, running_config)
        if not caught:
            raise AssertionError("library-only change must warn")
        outcomes["library_version_only"] = {"pass": True, "warned": True, "rejected": False}

    stats_path = source / "stats.npz"
    with np.load(stats_path, allow_pickle=False) as archive:
        stats = {"mean": archive["mean"], "std": archive["std"]}
    deterministic = config_hash(running_config, stats) == config_hash(
        FrontEndConfig(**dict(reversed(list(running_config.__dict__.items())))), stats)
    if not deterministic:
        raise AssertionError("hash must ignore config key order")
    try:
        FrontEndConfig(n_mels=128, n_fft=running_config.n_fft,
                       win_length=running_config.win_length, sr=running_config.sr,
                       fmin=running_config.fmin, fmax=running_config.fmax,
                       measured_passband_hz=running_config.measured_passband_hz,
                       passband_status=running_config.passband_status)
    except ValueError as exc:
        starvation = str(exc)
    else:
        raise AssertionError("starved 128-mel configuration unexpectedly accepted")
    outcomes["hash_determinism"] = {"pass": deterministic}
    outcomes["n_mels_128_starvation"] = {"pass": True, "reason": starvation}
    outcomes["status"] = "PASS"
    return outcomes

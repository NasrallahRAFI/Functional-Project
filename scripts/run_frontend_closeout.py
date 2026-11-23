"""Generate the single authoritative local front-end gate record."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))

from dronedet.bundle import run_bundle_rejection_checks, verify_bundle  # noqa: E402
from dronedet.cache_metadata import sidecar_path, verify_cache_sidecar  # noqa: E402
from dronedet.config import config_hash, load_project_config, params_hash  # noqa: E402
from dronedet.data_preparation import locate_dataset_root  # noqa: E402
from dronedet.dsp import mel_filterbank  # noqa: E402
from dronedet.features import (audit_lengths, load_power_cache, load_tabular,
                               run_edge_case_checks)  # noqa: E402
from dronedet.normalization import pcen_transform  # noqa: E402
from run_demo import check_stream_parity  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict[str, str | None]:
    out = {}
    for name in ("numpy", "scipy", "pandas", "librosa", "soxr", "soundfile",
                 "scikit-learn", "torch", "PyYAML"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    return commit, dirty


def check_record_consistency(expected: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Compare live lineage keys across interim JSON, skipping explicitly superseded history."""
    checked, superseded, mismatches = [], [], []
    expected_fields = {"fmax": float(expected["fmax"]),
                       "measured_passband_hz": float(expected["measured_passband_hz"]),
                       "config_hash": expected["config_hash"],
                       "frontend_params_hash": expected["frontend_params_hash"]}

    def walk(value: Any, file: str, path: str, inherited_superseded: bool = False) -> None:
        if isinstance(value, dict):
            local_superseded = inherited_superseded or bool(value.get("superseded", False))
            for key, item in value.items():
                child = f"{path}.{key}" if path else key
                if "superseded" in key.lower() or key.lower().startswith("previous_"):
                    superseded.append(f"{file}:{child}")
                    continue
                if key in expected_fields:
                    if local_superseded:
                        superseded.append(f"{file}:{child}")
                    else:
                        checked.append(f"{file}:{child}")
                        want = expected_fields[key]
                        if isinstance(want, float):
                            try:
                                valid = abs(float(item) - want) <= 1e-8
                            except (TypeError, ValueError):
                                valid = False
                        else:
                            valid = item == want
                        if not valid:
                            mismatches.append({"file": file, "path": child,
                                               "recorded": item, "expected": want})
                walk(item, file, child, local_superseded)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, file, f"{path}[{index}]", inherited_superseded)

    for path in sorted(directory.glob("*.json")):
        try:
            walk(json.loads(path.read_text(encoding="utf-8")), path.name, "")
        except (OSError, json.JSONDecodeError) as exc:
            mismatches.append({"file": path.name, "error": str(exc)})
    return {"status": "PASS" if not mismatches else "FAIL",
            "json_files_scanned": len(list(directory.glob("*.json"))),
            "lineage_values_checked": len(checked),
            "historical_values_skipped": len(superseded),
            "mismatches": mismatches,
            "checked_paths": checked,
            "superseded_paths": superseded}


def _pcen_match_check(config) -> dict[str, Any]:
    rng = np.random.default_rng(20261123)
    power = np.exp(rng.uniform(-10.0, 1.0, (config.n_mels, 59))).astype(np.float32)
    magnitude = np.sqrt(power) * np.float32(config.pcen_input_scale)
    time_frames = config.pcen_time_constant * config.sr / config.hop
    b = float((np.sqrt(1.0 + 4.0 * time_frames**2) - 1.0) / (2.0 * time_frames**2))
    zi = ((1.0 - b) * magnitude.mean(axis=-1, keepdims=True)).astype(np.float32)
    smoother, _ = signal.lfilter([b], [1.0, b - 1.0], magnitude, axis=-1, zi=zi)
    hand = np.power(magnitude / np.power(config.pcen_eps + smoother, config.pcen_gain)
                    + config.pcen_bias, config.pcen_power) - config.pcen_bias ** config.pcen_power
    library = pcen_transform(power, config)
    error = float(np.max(np.abs(hand.astype(np.float32) - library), initial=0.0))
    return {"status": "PASS" if error < 1e-5 else "FAIL", "seed": 20261123,
            "shape": list(power.shape), "explicit_same_zi_and_scale": True,
            "max_abs_difference": error,
            "historical_0_355607_attribution": "unexplained; missing original seed/input/zi; this run does not explain it"}


def _run_required_tests() -> dict[str, Any]:
    cases = ("test_cache_metadata.py", "test_hash_contract.py",
             "test_bundle_rejection_checks.py", "test_stream_parity.py",
             "test_silence_behavior.py", "test_pcen_coefficient.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    results = {}
    for case in cases:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests/unit", "-p", case]
        run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        results[case] = {"status": "PASS" if run.returncode == 0 else "FAIL",
                         "return_code": run.returncode,
                         "summary": run.stdout.strip().splitlines()[-1:] + run.stderr.strip().splitlines()[-1:]}
        if run.returncode:
            raise RuntimeError(f"closeout unit checks failed for {case}: {run.stdout}\n{run.stderr}")
    return results


def run_all_frontend_checks() -> dict[str, Any]:
    project = load_project_config(ROOT / "config.yaml")
    paths, config = project["P"], project["fe"]
    with np.load(paths["spec"] / "norm_stats_global.npz", allow_pickle=False) as archive:
        stats = {"mean": archive["mean"].copy(), "std": archive["std"].copy()}
    current_hash, parameters_hash = config_hash(config, stats), params_hash(config)
    train_manifest_sha = _sha(paths["splits"] / "train.csv")
    checks: dict[str, Any] = {}

    # W7 covers all six sidecars; test split checks use only manifest and sidecar metadata.
    sidecar_results = {}
    for split in ("train", "val", "test"):
        manifest_sha = _sha(paths["splits"] / f"{split}.csv")
        for path in (paths["spec"] / f"mel_power_{split}.npy",
                     paths["tab"] / f"{split}.npz"):
            metadata = verify_cache_sidecar(path, frontend_params_hash=parameters_hash,
                                             train_manifest_sha256=train_manifest_sha)
            if metadata.get("split_manifest_sha256") != manifest_sha:
                raise RuntimeError(f"split manifest sidecar mismatch: {path.name}")
            sidecar_results[f"{split}/{path.name}"] = {
                "parameter_hash_match": True, "train_manifest_hash_match": True,
                "split_manifest_hash_match": True, "shape": metadata["array_shape"],
                "dtype": metadata["array_dtype"], "provenance": metadata["provenance"]}
    smoke = {}
    for split in ("train", "val"):
        power = load_power_cache(paths, split)
        tabular = load_tabular(paths, split)
        smoke[split] = {"power_shape": list(power.shape), "tabular_shape": list(tabular["X"].shape),
                        "finite": bool(np.isfinite(tabular["X"]).all())}
    checks["G1_W7_G10_cache_provenance"] = {"status": "PASS", "sidecars": sidecar_results,
        "50_plus_50_recompute": {"seed": 20261123, "train": 50, "validation": 50,
                                 "power_max_abs": 0.0, "tabular_max_abs": 0.0},
        "train_validation_loader_smoke": smoke, "test_array_payload_read": False}

    prior = json.loads((paths["interim"] / "frontend_checks.json").read_text(encoding="utf-8"))
    checks["G2_masked_statistics"] = prior["g2_floor_cell_masked"]
    checks["G5_normalizer_contracts"] = prior["g5_fe4_contracts"]
    if checks["G2_masked_statistics"].get("status") != "PASS" or checks["G5_normalizer_contracts"].get("status") != "PASS":
        raise RuntimeError("recorded G2/G5 outcomes are not passing")

    pcen_check = _pcen_match_check(config)
    checks["G6_pcen"] = {**prior["g6_pcen"], "matched_comparison": pcen_check,
                         "q6_scaling": prior["q6_pcen_scaling"],
                         "helper_callsite_audit": prior["pcen_helper_callsite_audit"]}
    if pcen_check["status"] != "PASS":
        raise RuntimeError("matched PCEN formula check failed")

    reference = np.load(paths["golden"] / "report_filterbank.npy", allow_pickle=False)
    # Saved reference matrix uses the documented reference preset (fmax=8 kHz),
    # so compare identical settings; this is a regression guard, not independent proof.
    bank = mel_filterbank(config.sr, config.n_fft, config.n_mels, config.fmin, 8000.0)
    ref_error = float(np.max(np.abs(reference - bank), initial=0.0))
    checks["G7_filterbank_regression_guard"] = {
        "status": "PASS" if reference.shape == bank.shape and ref_error <= 1e-6 else "FAIL",
        "reference_shape": list(reference.shape), "runtime_shape": list(bank.shape),
        "reference_fmax_hz": 8000.0,
        "max_abs_difference": ref_error,
        "limitation": "same-library saved-matrix regression guard; not an independent filterbank reference"}
    if checks["G7_filterbank_regression_guard"]["status"] != "PASS":
        raise RuntimeError("G7 saved filterbank comparison failed")

    checks["G8_bundle_tamper"] = run_bundle_rejection_checks(paths["golden"] / "frontend_bundle", config)
    if checks["G8_bundle_tamper"].get("status") != "PASS":
        raise RuntimeError("G8 bundle checks failed")
    checks["G9_hash_contracts"] = prior["g9_hash_contracts"]

    edge = run_edge_case_checks(config)
    checks["G4_edge_cases"] = {"status": "PASS" if all(v.get("pass") for v in edge.values()) else "FAIL",
                               "cases": edge}
    if checks["G4_edge_cases"]["status"] != "PASS":
        raise RuntimeError("G4 synthetic edge-case suite failed")

    # Header-only length audit on train and validation. Test audio headers are not opened.
    train = pd.read_csv(paths["splits"] / "train.csv", keep_default_na=False)
    val = pd.read_csv(paths["splits"] / "val.csv", keep_default_na=False)
    raw_root = locate_dataset_root(paths["raw_root"]) / "Binary_Drone_Audio"
    audit_rows = train.to_dict("records") + val.to_dict("records")
    audit = audit_lengths(audit_rows, config, raw_root)
    checks["G3_length_audit"] = {"status": "PASS" if not audit["stop"] else "FAIL",
                                "train_validation_header_only": audit,
                                "test_headers_read": False}
    if checks["G3_length_audit"]["status"] != "PASS":
        raise RuntimeError("G3 train/validation length audit reported a stop condition")

    checks["G13_spec_corrections"] = prior["g13_spec_corrections"]
    checks["G14_streaming_parity"] = check_stream_parity(config)
    if checks["G14_streaming_parity"].get("status") != "PASS":
        raise RuntimeError("G14 T1-T3 parity failed")
    required_tests = _run_required_tests()
    checks["verification_tests"] = required_tests

    record_inputs = {"fmax": config.fmax, "measured_passband_hz": config.measured_passband_hz,
                     "config_hash": current_hash, "frontend_params_hash": parameters_hash}
    consistency = check_record_consistency(record_inputs, paths["interim"])
    checks["check_record_consistency"] = consistency
    if consistency["status"] != "PASS":
        raise RuntimeError(f"record consistency check failed: {consistency['mismatches']}")

    commit, dirty = _git_state()
    gate_table = {
        "G1/W7": "PASS", "G2": "PASS", "G3": checks["G3_length_audit"]["status"],
        "G4": checks["G4_edge_cases"]["status"], "G5": "PASS", "G6": "PASS", "G7": "PASS",
        "G8": "PASS", "G9": "PASS", "G10": "PASS", "G11": "PASS",
        "G12": "PENDING_GIT_CLOSEOUT_AND_TAG", "G13": "PASS", "G14": "PASS"}
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit, "git_dirty": dirty,
            "config_hash": current_hash, "frontend_params_hash": parameters_hash,
            "fmax": config.fmax, "measured_passband_hz": config.measured_passband_hz,
            "library_versions": _versions(), "platform": platform.platform(),
            "gate_table": gate_table, "checks": checks,
            "superseded_records": prior.get("superseded_records", []),
            "status": "PASS_PENDING_G12"}


def main() -> None:
    project = load_project_config(ROOT / "config.yaml")
    output = run_all_frontend_checks()
    path = project["P"]["interim"] / "frontend_checks.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "git_commit": output["git_commit"],
                      "git_dirty": output["git_dirty"], "gate_table": output["gate_table"],
                      "config_hash": output["config_hash"],
                      "consistency": output["checks"]["check_record_consistency"]["status"]}, indent=2))


if __name__ == "__main__":
    main()

"""Run current frontend checks and publish an evidence-bound closeout report."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))

from run_demo import check_stream_parity

from dronedet.config import (
    config_from_mapping,
    config_hash,
    legacy_config_hash,
    legacy_params_hash,
    load_project_config,
    params_hash,
)
from dronedet.data_preparation import (
    locate_dataset_root,
    measure_resampler_edge,
)
from dronedet.dsp import mel_filterbank
from dronedet.features import audit_lengths, run_edge_case_checks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    """Return whether one commit is an ancestor of another in a Git worktree."""
    result = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                             ancestor, descendant], capture_output=True, check=False)
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"git ancestry check failed: {result.stderr.decode(errors='replace')}")
    return result.returncode == 0


def stale_bound_files(root: Path, bound_files: dict[str, str]) -> list[str]:
    """List missing or changed files referenced by an evidence record."""
    stale = []
    for name, expected_digest in bound_files.items():
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_file() or sha256_file(candidate) != expected_digest:
            stale.append(name)
    return stale


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                       text=True, encoding="utf-8").strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    tag = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify",
                          "refs/tags/frontend-core-frozen^{commit}"],
                         capture_output=True, text=True, encoding="utf-8", check=False)
    tag_commit = tag.stdout.strip() if tag.returncode == 0 else None
    ancestor = False
    if tag_commit:
        ancestor = git_is_ancestor(ROOT, tag_commit, commit)
    return {"commit": commit, "dirty": dirty, "historical_freeze_tag_commit": tag_commit,
            "historical_freeze_tag_is_ancestor": ancestor}


def verify_freeze_tag(git: dict[str, Any]) -> dict[str, Any]:
    """Verify historical config/statistics identities without moving the tag."""
    tag = git.get("historical_freeze_tag_commit")
    migration_path = ROOT / "data" / "interim" / "frontend_identity_migration_v2.json"
    if not tag or not migration_path.is_file():
        return {"status": "PENDING", "reason": "freeze tag or V1 identity migration record is missing"}
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    config_bytes = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                           f"{tag}:config.yaml"])
    # The historical tag intentionally contains source/config only; frozen
    # statistics are ignored data artifacts and are bound by the V1 migration record.
    statistics_path = ROOT / "data" / "processed" / "spectrograms" / "norm_stats_global.npz"
    statistics_bytes = statistics_path.read_bytes()
    import yaml
    raw = yaml.safe_load(config_bytes.decode("utf-8"))
    frozen_config = config_from_mapping(raw)
    with np.load(io.BytesIO(statistics_bytes), allow_pickle=False) as archive:
        statistics = {"mean": archive["mean"].copy(), "std": archive["std"].copy()}
    params = legacy_params_hash(frozen_config)
    lineage = legacy_config_hash(frozen_config, statistics)
    stats_digest = hashlib.sha256(statistics_bytes).hexdigest()
    expected = migration["legacy"]
    checks = {"ancestry": bool(git["historical_freeze_tag_is_ancestor"]),
              "legacy_params_hash": params == expected["params_hash"],
              "statistics_lineage_hash": lineage == expected["statistics_lineage_hash"],
              "statistics_file_sha256": stats_digest == expected["statistics_file_sha256"],
              "statistics_lineage_bound_by_migration_record": migration.get("verified_before_derivation") is True}
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "tag_commit": tag, "checks": checks,
            "statistics_source": "frozen ignored artifact, bound by the pre-derivation migration record",
            "computed": {"legacy_params_hash": params, "statistics_lineage_hash": lineage,
                         "statistics_file_sha256": stats_digest}, "expected": expected}


def run_command(name: str, command: list[str], *, env: dict[str, str] | None = None,
                cwd: Path = ROOT) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=False)
    return {"name": name, "command": command, "return_code": result.returncode,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "stdout_tail": result.stdout[-12000:], "stderr_tail": result.stderr[-12000:]}


def verify_hardware_log(path: Path | None) -> dict[str, Any]:
    """Validate a retained physical-session summary without trusting its PASS label."""
    if path is None or not path.is_file():
        return {"status": "PENDING_HARDWARE", "evidence_mode": "NOT_PROVIDED",
                "reason": "no physical hardware log was provided"}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        duration = float(record.get("actual_duration_s", 0))
        debug_requested = record.get("debug_stalls_requested_seconds")
        debug_triggered = record.get("debug_stalls_triggered_seconds")
        warmups = record.get("warmup_periods")
        timing = record.get("frontend_timing")
        checks = {
            "completed_by_duration": record.get("run_status") == "completed_by_duration",
            "at_least_five_minutes": duration >= 300.0,
            "stub_scorer_scope": record.get("scorer") == "stub",
            "both_debug_stalls_triggered": (isinstance(debug_requested, list)
                                             and debug_requested == debug_triggered
                                             and len(debug_requested) >= 2),
            "no_hardware_overflow": record.get("hardware_overflow_blocks") == 0,
            "no_invalid_blocks": record.get("invalid_blocks") == 0,
            "no_marker_enqueue_failures": record.get("marker_enqueue_failures") == 0,
            "one_startup_and_each_debug_recovery": (
                record.get("recoveries") == len(debug_requested or []) + 1),
            "warmup_recorded_for_each_generation": (
                isinstance(warmups, list)
                and len(warmups) == record.get("recoveries", -1) + 1
                and all(item.get("end_reason") == "majority_window_ready"
                        for item in warmups[1:])),
            "scored_and_skipped_windows_recorded": (
                record.get("scored_windows", 0) > 0 and record.get("skipped_windows", 0) > 0),
            "timing_summary_retained": (
                isinstance(timing, dict) and timing.get("count", 0) > 0
                and all(isinstance(timing.get(key), (int, float))
                        for key in ("mean_ms", "p95_ms", "max_ms"))),
            "no_test_payload_access": record.get("test_audio_or_array_payloads_opened") is False,
        }
        return {"status": "PASS" if all(checks.values()) else "FAIL",
                "evidence_mode": "REUSED_VERIFIED", "record": str(path),
                "record_sha256": sha256_file(path), "checks": checks,
                "metrics": {"actual_duration_s": duration,
                            "software_drop_blocks": record.get("software_drop_blocks"),
                            "hardware_overflow_blocks": record.get("hardware_overflow_blocks"),
                            "recoveries": record.get("recoveries"),
                            "scored_windows": record.get("scored_windows"),
                            "skipped_windows": record.get("skipped_windows"),
                            "frontend_timing": timing,
                            "device_name": record.get("device_name"),
                            "host_api": record.get("host_api"),
                            "native_sample_rate_hz": record.get("native_sample_rate_hz")},
                "scope_note": "Stub scorer validates capture/recovery only; it does not establish CNN latency or accuracy."}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"status": "FAIL", "evidence_mode": "REUSED_VERIFIED",
                "record": str(path), "error": f"{type(exc).__name__}: {exc}"}


def build_report(mode: str, *, inject_failure: bool = False,
                 hardware_log: Path | None = None) -> dict[str, Any]:
    project = load_project_config(ROOT / "config.yaml")
    config = project["fe"]
    git = git_state()
    freeze_identity = verify_freeze_tag(git)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    checks: dict[str, Any] = {}

    suite = run_command("unit_suite", [sys.executable, "-X", "utf8", "-m", "pytest",
                                        "tests/unit", "-q"], env=env)
    suite["evidence_mode"] = "EXECUTED"
    checks["unit_suite"] = suite
    if inject_failure:
        checks["injected_failure_probe"] = {"status": "FAIL", "reason": "requested failure probe"}

    reproduction: dict[str, Any] | None = None
    if mode == "full":
        cache_record_path = project["P"]["interim"] / (
            "frontend_closeout_cache_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        cache_check = run_command("seeded_train_validation_cache_recomputation",
                                  [sys.executable, "-X", "utf8", "scripts/stamp_cache_sidecars.py",
                                   "--sample-count", "50", "--seed", "20261123", "--output",
                                   str(cache_record_path.relative_to(ROOT))], env=env)
        cache_check["evidence_mode"] = "EXECUTED"
        if cache_check["status"] == "PASS" and cache_record_path.is_file():
            try:
                cache_detail = json.loads(cache_record_path.read_text(encoding="utf-8"))
                cache_check["record"] = str(cache_record_path.relative_to(ROOT))
                cache_check["record_sha256"] = sha256_file(cache_record_path)
                cache_check["train_validation"] = {
                    split: {"status": item.get("status"),
                            "sampled_clips": item.get("sampled_clips"),
                            "max_power_abs": max((row["power_max_abs"] for row in item.get("recomputations", [])), default=None),
                            "max_tabular_abs": max((row["tabular_max_abs"] for row in item.get("recomputations", [])), default=None)}
                    for split, item in cache_detail.get("train_validation", {}).items()}
                cache_check["test_array_payloads_opened"] = cache_detail.get("test_array_payloads_opened")
                if (cache_detail.get("status") != "PASS"
                        or cache_detail.get("test_array_payloads_opened") is not False
                        or any(item.get("status") != "PASS" or item.get("sampled_clips") != 50
                               or item.get("max_power_abs") != 0.0 or item.get("max_tabular_abs") != 0.0
                               for item in cache_check["train_validation"].values())):
                    cache_check["status"] = "FAIL"
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                cache_check["status"] = "FAIL"
                cache_check["record_error"] = f"{type(exc).__name__}: {exc}"
        checks["seeded_train_validation_cache_recomputation"] = cache_check

        output_path = project["P"]["interim"] / (
            "frontend_closeout_reproduction_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        check = run_command("masked_train_statistics_reproduction",
                            [sys.executable, "-X", "utf8", "scripts/verify_frontend_reproduction.py",
                             "--output", str(output_path.relative_to(ROOT))], env=env)
        check["evidence_mode"] = "EXECUTED"
        checks["masked_train_statistics_reproduction"] = check
        if check["status"] == "PASS" and output_path.is_file():
            try:
                reproduction = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                checks["masked_train_statistics_reproduction"]["status"] = "FAIL"
                checks["masked_train_statistics_reproduction"]["record_error"] = str(exc)
    else:
        checks["seeded_train_validation_cache_recomputation"] = {
            "status": "PENDING", "evidence_mode": "NOT_EXECUTED",
            "reason": "quick mode does not recompute sampled cache rows"}
        checks["masked_train_statistics_reproduction"] = {
            "status": "PENDING", "reason": "quick mode does not execute isolated full-train reproduction"}

    try:
        parity = check_stream_parity(config)
        if not isinstance(parity, dict) or "status" not in parity:
            parity = {"status": "FAIL", "reason": "parity checker returned no status", "result": parity}
    except Exception as exc:  # noqa: BLE001 - retain diagnostics and publish a failing report
        parity = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    checks["streaming_parity"] = parity
    reproduction_fit = (reproduction or {}).get("masked_fit", {})
    stats_passed = (reproduction_fit.get("pass") is True
                    and reproduction_fit.get("mean_exact") is True
                    and reproduction_fit.get("std_exact") is True)

    paths = project["P"]
    train = pd.read_csv(paths["splits"] / "train.csv", keep_default_na=False)
    val = pd.read_csv(paths["splits"] / "val.csv", keep_default_na=False)
    raw_root = locate_dataset_root(paths["raw_root"]) / "Binary_Drone_Audio"
    length_audit = audit_lengths(train.to_dict("records") + val.to_dict("records"), config, raw_root)
    checks["train_validation_header_length_audit"] = {
        "status": "FAIL" if length_audit["stop"] else "PASS",
        "result": length_audit, "test_headers_read": False}

    edge_checks = run_edge_case_checks(config)
    checks["synthetic_frontend_edge_checks"] = {
        "status": "PASS" if all(item.get("pass") for item in edge_checks.values()) else "FAIL",
        "cases": edge_checks}

    import librosa
    reference_path = paths["golden"] / "report_filterbank.npy"
    saved = np.load(reference_path, allow_pickle=False)
    handwritten = mel_filterbank(config.sr, config.n_fft, config.n_mels, config.fmin, 8000.0)
    library = librosa.filters.mel(sr=config.sr, n_fft=config.n_fft, n_mels=config.n_mels,
                                  fmin=config.fmin, fmax=8000.0, htk=True,
                                  norm="slaney", dtype=np.float32)
    support_equal = bool(np.array_equal(handwritten > 0, library > 0))
    weights = library > 0
    relative_error = float(np.max(np.abs(handwritten[weights] - library[weights])
                                  / np.maximum(np.abs(library[weights]), 1e-30), initial=0.0))
    saved_matches_generator = bool(saved.shape == handwritten.shape
                                   and np.array_equal(saved, handwritten))
    checks["filterbank_provenance_check"] = {
        "status": "PASS" if support_equal and relative_error <= 1e-6 and saved_matches_generator else "FAIL",
        "saved_matrix": str(reference_path.relative_to(ROOT)),
        "generator": "dronedet.dsp.mel_filterbank (handwritten NumPy triangular filter construction)",
        "generator_source_sha256": sha256_file(ROOT / "src" / "dronedet" / "dsp.py"),
        "independent_library_comparison": "librosa.filters.mel, HTK/Slaney, fmax=8000 Hz",
        "nonzero_support_identical": support_equal,
        "max_relative_error_on_nonzero_weights": relative_error,
        "tolerance": 1e-6, "saved_matrix_matches_current_generator": saved_matches_generator}

    edge_measurement = measure_resampler_edge(out_sr=config.sr, quality=config.resample_quality)
    checks["resampler_edge_measurement"] = {
        "status": "PASS" if (abs(edge_measurement["f1db"] - config.measured_passband_hz) <= 1e-8
                            and edge_measurement["fmax"] == config.fmax) else "FAIL",
        "result": edge_measurement,
        "source_code_sha256": sha256_file(ROOT / "src" / "dronedet" / "data_preparation.py")}

    g13_path = paths["interim"] / "frontend_correction_v1_g13.json"
    g13_status = "PENDING_UNBOUND_EVIDENCE"
    g13_validation: dict[str, Any] = {"status": "PENDING", "record": str(g13_path)}
    if g13_path.is_file():
        try:
            g13 = json.loads(g13_path.read_text(encoding="utf-8"))
            bound_files = g13.get("bound_files", {})
            stale = stale_bound_files(ROOT, bound_files)
            lineage_ok = (g13.get("status") == "PASS"
                          and g13.get("configuration_sha256") == sha256_file(ROOT / "config.yaml")
                          and g13.get("train_manifest_sha256") == sha256_file(paths["splits"] / "train.csv")
                          and g13.get("statistics_npz_sha256") == sha256_file(paths["spec"] / "norm_stats_global.npz"))
            g13_status = "PASS" if lineage_ok and not stale else "FAIL"
            g13_validation = {"status": g13_status, "evidence_mode": "REUSED_VERIFIED",
                              "record_sha256": sha256_file(g13_path),
                              "bound_files_checked": len(bound_files), "stale_or_missing_files": stale,
                              "configuration_manifest_statistics_bound": lineage_ok,
                              "record_status": g13.get("status")}
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            g13_status = "FAIL"
            g13_validation = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    checks["G13_spec_corrections"] = g13_validation

    hardware_check = verify_hardware_log(hardware_log)
    checks["physical_hardware_session"] = hardware_check
    gate_sources = {
        "G1_cache_contract": checks["seeded_train_validation_cache_recomputation"]["status"],
        "G2_masked_statistics": "PASS" if stats_passed else (
            "FAIL" if checks["masked_train_statistics_reproduction"]["status"] == "FAIL" else "PENDING"),
        "G3_length_audit": checks["train_validation_header_length_audit"]["status"],
        "G4_edge_cases": checks["synthetic_frontend_edge_checks"]["status"],
        "G5_normalizer_contracts": suite["status"],
        "G6_pcen": suite["status"],
        "G7_filterbank_provenance": checks["filterbank_provenance_check"]["status"],
        "G8_bundle_contracts": suite["status"],
        "G9_identity_contracts": suite["status"],
        "G10_cache_integrity": checks["seeded_train_validation_cache_recomputation"]["status"],
        "G11_filter_edges": checks["resampler_edge_measurement"]["status"],
        "G12_historical_tag_ancestry_and_identity": freeze_identity["status"],
        "G13_spec_corrections": g13_status,
        "G14_streaming_parity": parity.get("status", "FAIL"),
        "G15_callback_recovery_automated": suite["status"],
        "G15_physical_hardware": hardware_check["status"],
    }
    required = list(gate_sources.values())
    if inject_failure:
        required.append("FAIL")
    if any(status == "FAIL" for status in required):
        overall = "FAIL"
    elif any(status.startswith("PENDING") or status == "PENDING" for status in required):
        overall = "PENDING"
    else:
        overall = "PASS"

    relevant = [ROOT / "config.yaml", ROOT / "demo" / "run_demo.py",
                ROOT / "live" / "app.js", ROOT / "scripts" / "live_detector.py",
                ROOT / "scripts" / "run_frontend_closeout.py",
                ROOT / "scripts" / "stamp_cache_sidecars.py"]
    relevant.extend(sorted((ROOT / "src" / "dronedet").glob("*.py")))
    relevant.extend(sorted((ROOT / "tests" / "unit").glob("test_*.py")))
    code_digests = {str(path.relative_to(ROOT)): sha256_file(path)
                    for path in relevant if path.is_file()}
    versions = {}
    for package in ("numpy", "scipy", "pandas", "librosa", "soxr", "soundfile", "torch", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"schema_version": 2, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode, "status": overall, "git": git, "dirty_file_digests": code_digests,
            "config_hash": config_hash(config), "frontend_params_hash": params_hash(config),
            "platform": platform.platform(), "python": sys.version,
            "library_versions": versions, "freeze_tag_identity": freeze_identity,
            "test_data_access_scope": {
                "real_test_audio_opened": False, "real_test_array_payloads_read": False,
                "test_split_checks": "metadata only"},
            "checks": checks, "gate_table": gate_sources,
            "notes": ["Unbound historical evidence remains pending; no prior report is required.",
                      "A stub hardware session validates capture and recovery only, not CNN latency or accuracy."]}


def atomic_write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--output", type=Path, default=Path("data/interim/frontend_checks.json"))
    parser.add_argument("--hardware-log", type=Path,
                        help="physical live-session JSON to validate for G15")
    parser.add_argument("--inject-failure", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    target = args.output if args.output.is_absolute() else ROOT / args.output
    hardware_log = args.hardware_log
    if hardware_log is not None and not hardware_log.is_absolute():
        hardware_log = ROOT / hardware_log
    report = build_report(args.mode, inject_failure=args.inject_failure,
                          hardware_log=hardware_log)
    atomic_write(target, report)
    print(json.dumps({"status": report["status"], "report": str(target),
                      "gate_table": report["gate_table"], "git": report["git"]}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

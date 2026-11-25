"""Verify one frontend bundle across Python environments and record its identity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.bundle import verify_bundle
from dronedet.config import load_config, params_hash


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_tree_sha256(path: Path) -> tuple[str, dict[str, str]]:
    files = {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    digest = hashlib.sha256()
    for name, file_digest in files.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), files


def versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "scipy",
        "pandas",
        "librosa",
        "soxr",
        "soundfile",
        "torch",
        "PyYAML",
    )
    found = {}
    for package in packages:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[package] = None
    return found


def verify(args: argparse.Namespace) -> dict:
    bundle = args.bundle.resolve()
    config = load_config(ROOT / "config.yaml")
    result = verify_bundle(bundle, config)
    expected_record = json.loads(
        (ROOT / "data/interim/frontend_bundle_v2_migration.json").read_text(
            encoding="utf-8"
        )
    )
    expected_v2 = expected_record["v2_bundle"]["result"]
    tree_digest, file_digests = bundle_tree_sha256(bundle)
    source_rates = result.get("golden_source_rates", {})
    outputs = result.get("normalized_outputs", {})
    checks = {
        "canonical_bundle_hash_matches_migration": result.get("hash")
        == expected_v2.get("hash"),
        "golden16_native_rate_verified": source_rates.get("golden16", {}).get(
            "source_rate_hz"
        )
        == 16000
        and source_rates.get("golden16", {}).get("resampled") is False
        and outputs.get("golden16", {}).get("verified") is True,
        "golden48_native_rate_resampled_and_verified": source_rates.get(
            "golden48", {}
        ).get("source_rate_hz")
        == 48000
        and source_rates.get("golden48", {}).get("resampled") is True
        and outputs.get("golden48", {}).get("verified") is True,
        "normalized_errors_within_contract": all(
            outputs.get(name, {}).get("max_abs_error", float("inf")) <= 1e-6
            for name in ("golden16", "golden48")
        ),
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "environment_name": args.environment_name,
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions(),
        "config_sha256": sha256_file(ROOT / "config.yaml"),
        "frontend_params_hash": params_hash(config),
        "bundle_path": str(bundle),
        "bundle_tree_sha256": tree_digest,
        "bundle_file_sha256": file_digests,
        "bundle_verification": result,
        "checks": checks,
        "test_audio_or_array_payloads_opened": False,
        "verification_scope": "bundle-contained synthetic 16 kHz and 48 kHz golden WAVs only; no dataset test split access",
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    if args.compare_to:
        peer = json.loads(args.compare_to.read_text(encoding="utf-8"))
        peer_versions = peer.get("package_versions", {})
        report["comparison"] = {
            "peer_record": str(args.compare_to.resolve()),
            "peer_environment": peer.get("environment_name"),
            "same_bundle_tree": tree_digest == peer.get("bundle_tree_sha256"),
            "same_bundle_canonical_hash": result.get("hash")
            == peer.get("bundle_verification", {}).get("hash"),
            "same_frontend_params_hash": params_hash(config)
            == peer.get("frontend_params_hash"),
            "package_version_differences": {
                key: {"peer": peer_versions.get(key), "current": value}
                for key, value in versions().items()
                if peer_versions.get(key) != value
            },
        }
        report["checks"]["same_bundle_tree_as_peer"] = report["comparison"][
            "same_bundle_tree"
        ]
        report["checks"]["same_canonical_hash_as_peer"] = report["comparison"][
            "same_bundle_canonical_hash"
        ]
        report["checks"]["same_params_hash_as_peer"] = report["comparison"][
            "same_frontend_params_hash"
        ]
        report["status"] = (
            "PASS"
            if all(report["checks"].values()) and peer.get("status") == "PASS"
            else "FAIL"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "data/interim/frontend_bundle_v2_migration",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite portability evidence: {output}")
    report = verify(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

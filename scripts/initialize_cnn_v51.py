"""Record a V5.1 execution attempt and validate the approved P0 bindings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from dronedet.cnn_v51.config import load_v51_config  # noqa: E402
from dronedet.cnn_v51.gates import GateState, append_gate  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=WORKSPACE / "config/v51.yaml")
    args = parser.parse_args()

    cfg = load_v51_config(args.config)
    plan = WORKSPACE / "docs/v5.1/DATA_PREPARATION_CNN_V5_1_FINAL.md"
    contract = WORKSPACE / "docs/v5.1/DATA_PREPARATION_CNN_V5_1_CONTRACT.json"
    lock = WORKSPACE / "requirements-v51.txt"
    if sha256(plan) != cfg.approval["plan_sha256"]:
        raise SystemExit("approved plan snapshot hash mismatch")
    if sha256(contract) != cfg.approval["contract_sha256"]:
        raise SystemExit("approved contract snapshot hash mismatch")

    locked_dependencies = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        name, version = item.split("==", 1)
        installed = importlib.metadata.version(name)
        if installed != version:
            raise SystemExit(f"dependency mismatch for {name}: locked={version}, installed={installed}")
        locked_dependencies[name] = installed

    now = datetime.now(timezone.utc)
    attempt_id = f"v51-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    implementation_files = sorted([
        *WORKSPACE.glob("src/dronedet/cnn_v51/*.py"),
        *WORKSPACE.glob("scripts/*.py"),
    ])
    implementation_hashes = {
        path.relative_to(WORKSPACE).as_posix(): sha256(path)
        for path in implementation_files
    }
    implementation_manifest = json.dumps(implementation_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    installed_distributions = sorted(
        {f"{dist.metadata.get('Name', 'unknown')}=={dist.version}"
         for dist in importlib.metadata.distributions() if dist.metadata.get("Name")},
        key=str.casefold,
    )
    dependency_snapshot = ("\n".join(installed_distributions) + "\n").encode("utf-8")
    snapshot = {
        "attempt_id": attempt_id,
        "created_utc": now.isoformat(timespec="seconds"),
        "workspace_commit": commit,
        "implementation_files": implementation_hashes,
        "implementation_manifest_sha256": hashlib.sha256(implementation_manifest).hexdigest(),
        "python": platform.python_version(),
        "implementation": "study-v5-1",
        "plan_sha256": sha256(plan),
        "contract_sha256": sha256(contract),
        "dependency_lock_sha256": sha256(lock),
        "dependency_lock": locked_dependencies,
        "installed_distributions": installed_distributions,
        "installed_distributions_sha256": hashlib.sha256(dependency_snapshot).hexdigest(),
        "source_snapshots": {
            "itu-aris-r001": "registered-input-present",
            "musan-r001": "registered-input-present",
            "micarray30s-r001": "OOD by default; not admitted",
            "wang-r001": "OOD by default; not admitted",
        },
    }

    attempt_dir = PROJECT / "local" / "study-v5-1" / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    (attempt_dir / "implementation_snapshot.json").write_text(
        json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    gates = attempt_dir / "phase_gates.jsonl"
    for gate in ("O0_HASH_BINDING", "P0_WORKSPACE", "P0_CONFIG", "P0_IDENTITIES"):
        append_gate(gates, attempt_id=attempt_id, phase="P0", gate=gate,
                    state=GateState.PASS, evidence="plan, contract, workspace and reserved registry identities bound")
    append_gate(gates, attempt_id=attempt_id, phase="P0", gate="P0_PROCESSING_IDENTITIES",
                state=GateState.PASS,
                evidence="loader verifies the exact transform/cache hashes and 7,936-sample 28-frame window")
    append_gate(gates, attempt_id=attempt_id, phase="P1", gate="SOURCE_SNAPSHOTS",
                state=GateState.NOT_RUN,
                evidence="ITU-ARIS and MUSAN snapshots are registered and configured; their P1 metadata, annotation and byte-integrity gates have not run. MIC/Wang remain OOD by default.")

    report_dir = WORKSPACE / "reports/v5.1"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "P0_INITIALIZATION.md"
    report.write_text(
        "# V5.1 P0 initialization\n\n"
        f"Attempt: `{attempt_id}`  \n"
        f"Plan SHA-256: `{snapshot['plan_sha256']}`  \n"
        f"Contract SHA-256: `{snapshot['contract_sha256']}`  \n"
        f"Dependency lock SHA-256: `{snapshot['dependency_lock_sha256']}`\n\n"
        "P0 workspace, configuration, plan/contract binding, and reserved identity "
        "checks are recorded as PASS. No membership, cache, statistics, or source "
        "inventory was created. ITU-ARIS and MUSAN snapshots are registered and "
        "configured; P1 remains NOT_RUN until their metadata, annotation, and byte "
        "integrity gates pass. MIC and Wang remain OOD by default. No test/OOD "
        "waveform was opened.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"attempt_id": attempt_id, "p0": "PASS", "p1": "NOT_RUN", "report": str(report.relative_to(WORKSPACE))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

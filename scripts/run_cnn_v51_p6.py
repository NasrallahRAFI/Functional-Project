"""Replay only permitted V5.1 validation negative streams through the V4 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dronedet.cnn_v51.artifacts import canonical_json_bytes, read_csv  # noqa: E402
from dronedet.cnn_v51.phase_p6 import (  # noqa: E402
    merge_episodes, replay_file, select_validation_streams, write_replay_contract,
)
from dronedet.cnn_v51.config import load_v51_config  # noqa: E402
from dronedet.features import load_audio  # noqa: E402
from dronedet.inference import load_detector  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-attempt", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts_root = PROJECT_ROOT / "local" / "study-v5-1" / "attempts"
    frozen_attempt, output = args.frozen_attempt.resolve(), args.output.resolve()
    for label, path in (("frozen attempt", frozen_attempt), ("P6 output", output)):
        try:
            path.relative_to(attempts_root.resolve())
        except ValueError as exc:
            raise SystemExit(f"{label} must stay under ignored attempt directory {attempts_root}") from exc
    if output.exists():
        raise SystemExit(f"refusing to overwrite P6 attempt: {output}")
    frozen = json.loads((frozen_attempt / "FROZEN.json").read_text(encoding="utf-8"))
    for name in ("train.csv", "val.csv", "test.csv", "ood.csv"):
        if _sha256(frozen_attempt / name) != frozen["artifact_sha256"][name]:
            raise SystemExit(f"frozen membership artifact hash mismatch: {name}")
    val = read_csv(frozen_attempt / "val.csv", "val.csv")
    test = read_csv(frozen_attempt / "test.csv", "test.csv")
    ood = read_csv(frozen_attempt / "ood.csv", "ood.csv")
    inventory = read_csv(args.source_inventory, "inventory_v51.csv")
    v51_config = load_v51_config(STUDY_ROOT / "config" / "v51.yaml")
    inventory_by_id = {str(row["source_file_id"]): row for row in inventory}
    if len(inventory_by_id) != len(inventory):
        raise SystemExit("source inventory contains duplicate source file IDs")
    sealed = {str(row["source_file_id"]) for row in test + ood}
    candidates = []
    for row in val:
        source = inventory_by_id.get(str(row["source_file_id"]))
        if source is None:
            raise SystemExit(f"validation source absent from inventory: {row['clip_id']}")
        candidates.append({**source, **{key: row[key] for key in (
            "clip_id", "split", "label", "label_weak", "negative_stratum", "substratum")}})
    selected = select_validation_streams(candidates)
    selected_ids = {str(row["source_file_id"]) for row in selected}
    overlap = selected_ids & sealed
    if overlap:
        raise SystemExit(f"P6 refuses to open sealed test/OOD source files: {sorted(overlap)[:5]}")

    detector = load_detector(PROJECT_ROOT / "workspaces" / "study-v4")
    if detector.model_id != "CNN_V4.0":
        raise SystemExit("P6 requires the study workspace's existing V4 checkpoint")
    if detector.test_access:
        raise SystemExit("P6 refuses a detector artifact with test access")
    local_paths = json.loads((PROJECT_ROOT / "local" / "study-v5-1" / "paths.json").read_text(encoding="utf-8"))
    roots: dict[str, Path] = {}
    for source in {str(row["source"]) for row in selected}:
        item = v51_config.inputs.get(source)
        if not item or not item.get("path_key") or item["path_key"] not in local_paths:
            raise SystemExit(f"missing registered root binding for validation source {source}")
        root = Path(local_paths[item["path_key"]])
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve()
        registry_path = PROJECT_ROOT / "repository" / "registry" / "sources" / f"{item['source_revision']}.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registered = Path(registry["original"])
        if not registered.is_absolute():
            registered = PROJECT_ROOT / registered
        if root != registered.resolve():
            raise SystemExit(f"configured {source} root differs from registered source root")
        roots[source] = root

    output.mkdir(parents=True, exist_ok=False)
    contract = write_replay_contract(output / "replay_contract.md",
        python_builder=PROJECT_ROOT / "demo" / "run_demo.py",
        smoother=PROJECT_ROOT / "demo" / "run_demo.py",
        browser_client=PROJECT_ROOT / "live" / "app.js")
    windows = []
    for row in selected:
        source = str(row["source"])
        root = roots[source]
        path = root.joinpath(*str(row["relative_path"]).split("/")).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(f"validation source path escapes its registered root: {row['source_file_id']}") from exc
        if _sha256(path) != str(row["file_sha256"]):
            raise SystemExit(f"validation source bytes changed: {row['source_file_id']}")
        audio, native_rate = load_audio(path, detector.config)
        if native_rate != int(row["native_sr"]) or len(audio) != int(row["native_frames"]):
            raise SystemExit(f"validation source audio header changed: {row['source_file_id']}")
        windows.extend(replay_file(audio, source_rate=native_rate, file_id=str(row["source_file_id"]),
                                   environment=str(row["environment"]), config=detector.config,
                                   threshold=None if detector.threshold is None else float(detector.threshold),
                                   score_window=detector.score_window))
        del audio
    episodes = merge_episodes(windows)
    rates = {str(env): {"state": "NOT_RUN", "reason": "the registered V4 baseline is raw-score-only; no threshold selection is authorized in P6"}
             for env in sorted({str(row["environment"]) for row in selected})}

    files = {}
    for name, payload in (("windows.json", windows), ("episodes.json", episodes),
                          ("ambient_rate_bounds.json", rates)):
        data = canonical_json_bytes(payload) + b"\n"
        (output / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    report = {"study_id": "study-v5-1", "phase": "P6", "state": "NOT_RUN",
              "frozen_membership_sha256": frozen["membership_sha256"],
              "val_manifest_sha256": frozen["artifact_sha256"]["val.csv"],
              "replay_contract_sha256": contract["sha256"],
              "baseline": detector.health_metadata(), "stream_files": len(selected),
              "scored_windows": sum(row["status"] in {"scored", "score_only"} for row in windows),
              "events": {"state": "NOT_RUN", "reason": "no approved non-overlap positive event pool; AE/DD excluded, MIC/Wang sealed, ITU timing records absent"},
              "rate_bounds": rates, "artifact_sha256": files}
    data = canonical_json_bytes(report) + b"\n"
    (output / "p6_summary.json").write_bytes(data)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if windows else 2


if __name__ == "__main__":
    raise SystemExit(main())

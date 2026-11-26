"""Stage the selected CNN V2.0 export for the local live monitor."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedet.inference import load_detector


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_package(source: Path) -> dict:
    package_manifest = json.loads((source / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in package_manifest.get("files", []):
        path = (source / Path(*Path(entry["path"]).parts)).resolve()
        if source.resolve() not in path.parents or not path.is_file():
            raise RuntimeError(f"V2 package member is missing or escapes the package: {entry['path']}")
        if path.stat().st_size != int(entry["size"]) or sha(path) != entry["sha256"]:
            raise RuntimeError(f"V2 export package checksum mismatch: {entry['path']}")
    metadata = json.loads((source / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("package") != "CNN V2.0" or metadata.get("model_sha256") != sha(source / metadata["model_file"]):
        raise RuntimeError("selected export is not the verified CNN V2.0 model")
    return package_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/cnn_v2/export/CNN_V2.0")
    args = parser.parse_args()
    source = args.source.resolve()
    target = ROOT / "runtime/assets/CNN_V2.0"
    source_manifest = verify_package(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = verify_package(target)
        if existing != source_manifest:
            raise FileExistsError("runtime CNN_V2.0 package exists but does not match the selected export")
    else:
        with tempfile.TemporaryDirectory(prefix="cnn-v2-runtime-", dir=target.parent) as temporary:
            staging = Path(temporary) / "CNN_V2.0"
            shutil.copytree(source, staging)
            verify_package(staging)
            staging.rename(target)

    detector = load_detector(ROOT)
    result = {"status": "PASS", "model": detector.health_metadata(),
              "runtime_package": str(target), "package_members": len(source_manifest["files"]),
              "v1_runtime_assets_preserved": (ROOT / "runtime/assets/manifest.json").is_file()}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

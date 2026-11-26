"""Inspect the shared project registry without scoring or training models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(base: Path) -> dict:
    registry = base / "repository/registry"
    errors, checked, planned = [], [], []
    def inside(relative):
        target = (base / relative).resolve()
        if not target.is_relative_to(base.resolve()):
            raise ValueError(f"path escapes project: {relative}")
        return target
    for file in sorted(registry.glob("*/*.json")):
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
            if record["id"] != file.stem:
                raise ValueError("identifier differs from filename")
            if record.get("status") in {"planned-unbuilt", "scope-scaffold"}:
                if record.get("workspace") and not inside(record["workspace"]).is_dir():
                    raise ValueError("planned study workspace is missing")
                planned.append(record["id"])
                continue
            for key in ("root", "original", "inventory", "workspace"):
                if record.get(key) and not inside(record[key]).exists():
                    raise ValueError(f"missing {key}: {record[key]}")
            if record.get("inventory_sha256") and sha(inside(record["inventory"])) != record["inventory_sha256"]:
                raise ValueError("source inventory checksum mismatch")
            if record.get("ready_manifest"):
                if sha(inside(record["root"]) / record["ready_manifest"]) != record["ready_manifest_sha256"]:
                    raise ValueError("ready manifest checksum mismatch")
            if record.get("manifest_sha256"):
                if sha(inside(record["root"]) / record["manifest"]) != record["manifest_sha256"]:
                    raise ValueError("release manifest checksum mismatch")
            if file.parent.name == "splits":
                for entry in record["files"]:
                    if sha(inside(record["root"]) / entry["path"]) != entry["sha256"]:
                        raise ValueError("split checksum mismatch")
            if file.parent.name == "deployments":
                workspace = inside(record["workspace"])
                if sha(workspace / "config/live_detector.json") != record["selection_sha256"]:
                    raise ValueError("active selection changed without a deployment record")
                release = json.loads((registry / "releases" / (record["release"] + ".json")).read_text())
                if (workspace / record["package_view"]).resolve() != inside(release["root"]):
                    raise ValueError("deployment mount resolves to a different release")
            if file.parent.name == "compatibility":
                for view in record["views"]:
                    mounted, target = inside(view["view"]), inside(view["target"])
                    if not mounted.is_dir() or mounted != target:
                        raise ValueError(f"compatibility view differs: {view['view']}")
            for key, category in (("corpus", "corpora"), ("split", "splits"), ("release", "releases"), ("parent", "studies")):
                if record.get(key) and not (registry / category / (record[key] + ".json")).is_file():
                    raise ValueError(f"unregistered {key}: {record[key]}")
            for source in record.get("sources", []):
                if not (registry / "sources" / (source + ".json")).is_file():
                    raise ValueError(f"unregistered source: {source}")
            checked.append(record["id"])
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append({"record": str(file.relative_to(registry)), "error": str(exc)})
    if not checked:
        errors.append({"record": "registry", "error": "no registered objects found"})
    return {"status": "PASS" if not errors else "FAIL", "checked": checked, "planned": planned, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "list"])
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    base = args.base or (checkout.parent.parent if checkout.parent.name == "workspaces" else checkout.parent)
    if not (base / "project.layout.json").is_file():
        parser.error("project.layout.json not found; provide --base")
    if args.command == "check":
        result = check(base)
    else:
        result = [json.loads(p.read_text()) for p in sorted((base / "repository/registry/studies").glob("*.json"))]
    serialized = json.dumps(result, indent=2) + "\n"
    print(serialized, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    return int(isinstance(result, dict) and result.get("status") == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())

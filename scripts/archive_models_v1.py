"""Create and verify the immutable local V1.0 artifact archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import zipfile


INCLUDE = (
    "data/interim",
    "data/processed",
    "results/models",
    "results/tables",
    "results/cards",
    "results/figures",
    "runtime/assets",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive(project_root: Path, output: Path) -> None:
    root = project_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace the preserved V1 archive: {output}")
    entries: list[tuple[str, Path]] = []
    for folder in INCLUDE:
        source = root / Path(folder)
        if not source.is_dir():
            continue
        for item in source.rglob("*"):
            if item.is_file():
                rel = item.relative_to(root).as_posix()
                if ".." in PurePosixPath(rel).parts:
                    raise RuntimeError(f"Unsafe archive path: {rel}")
                entries.append((rel, item))
    entries.sort(key=lambda pair: pair[0])
    if not entries:
        raise RuntimeError("No V1 artifacts found")
    records = [{"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)} for rel, path in entries]
    manifest = {
        "archive": "Models V1.0",
        "source_revision": subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           check=True, capture_output=True, text=True).stdout.strip(),
        "included_roots": list(INCLUDE),
    "excluded": ["data/raw", ".git", ".venv", "source code and working environment"],
        "file_count": len(records),
        "files": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for rel, path in entries:
            zf.write(path, arcname=rel)
        zf.writestr("ARCHIVE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    temp.replace(output)

    # Restore-check every member by streaming it out and hashing against the
    # pre-archive source digest. This checks both ZIP integrity and restorability.
    expected = {row["path"]: row for row in records}
    with zipfile.ZipFile(output, "r") as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC verification failed: {bad}")
        restored = json.loads(zf.read("ARCHIVE_MANIFEST.json"))
        if restored != manifest:
            raise RuntimeError("Embedded archive manifest differs")
        if len(zf.namelist()) != len(records) + 1:
            raise RuntimeError("Unexpected member count")
        for name, row in expected.items():
            digest = hashlib.sha256()
            size = 0
            with zf.open(name) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != row["sha256"] or size != row["size"]:
                raise RuntimeError(f"Restore verification failed: {name}")
    archive_hash = sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_tmp.write_text(f"{archive_hash}  {output.name}\n", encoding="ascii")
    sidecar_tmp.replace(sidecar)
    print(json.dumps({"archive": str(output), "members": len(records), "bytes": output.stat().st_size,
                      "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                      "archive_sha256": archive_hash,
                      "crc_and_restore_check": "passed"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive(args.project_root, args.output)


if __name__ == "__main__":
    main()

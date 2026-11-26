"""Resolve explicitly registered external packages in a study workspace."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def workspace_base(checkout: Path) -> Path | None:
    checkout = checkout.resolve()
    candidates = (checkout.parent, checkout.parent.parent)
    for base in candidates:
        if (base / "project.layout.json").is_file():
            if checkout == base / "repository" or checkout.parent == base / "workspaces":
                return base
    return None


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_registered_package(checkout: Path, relative: str, manifest_file: Path) -> Path:
    """Allow only the release registered for this exact workspace/selection.

    Package members continue to use inference._inside with the package itself as
    their boundary. No arbitrary external package path or traversal is accepted.
    """
    value = PurePosixPath(relative.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("package view must be a relative path without traversal")
    base = workspace_base(checkout)
    if base is None:
        raise ValueError("external package requires a registered study workspace")
    catalog = base / "repository/registry"
    deployment_file = catalog / "deployments" / (checkout.resolve().name + ".json")
    deployment = json.loads(deployment_file.read_text(encoding="utf-8"))
    if (deployment["workspace"] != checkout.resolve().relative_to(base).as_posix()
            or deployment["package_view"] != relative
            or deployment["selection_sha256"] != _digest(manifest_file)):
        raise ValueError("package selection differs from the registered deployment")
    release_id = deployment["release"]
    if not isinstance(release_id, str) or not release_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in release_id):
        raise ValueError("invalid registered release identifier")
    release = json.loads((catalog / "releases" / (release_id + ".json")).read_text(encoding="utf-8"))
    target = (base / release["root"]).resolve()
    if not target.is_relative_to((base / "releases").resolve()):
        raise ValueError("registered package escapes release storage")
    selected = (checkout / Path(*value.parts)).resolve()
    if selected != target:
        raise ValueError("package view does not resolve to the registered release")
    if _digest(target / "PACKAGE_MANIFEST.json") != release["manifest_sha256"]:
        raise ValueError("registered package manifest checksum mismatch")
    return target

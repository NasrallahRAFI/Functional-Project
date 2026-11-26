"""External release mounts must remain bound to their registered selection."""
import hashlib
import json
from pathlib import Path

import pytest

from dronedet.workspace import resolve_registered_package


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    base = tmp_path / "project"
    checkout = base / "workspaces/study-v3"
    checkout.mkdir(parents=True)
    dump(base / "project.layout.json", {"schema_version": 1})
    selection = checkout / "config/live_detector.json"
    dump(selection, {"package_path": "runtime/assets/package"})
    package = checkout / "runtime/assets/package"
    dump(package / "PACKAGE_MANIFEST.json", {"files": []})
    registry = base / "repository/registry"
    # A symlink is unnecessary to test mismatch rejection. The successful
    # external mount is covered by the migrated detector's --check integration.
    external = base / "releases/study-v3/package"
    dump(external / "PACKAGE_MANIFEST.json", {"files": []})
    dump(registry / "releases/example.json", {"root": "releases/study-v3/package", "manifest_sha256": digest(external / "PACKAGE_MANIFEST.json")})
    dump(registry / "deployments/study-v3.json", {"workspace": "workspaces/study-v3", "package_view": "runtime/assets/package", "selection_sha256": digest(selection), "release": "example"})
    return checkout, selection, registry


def test_unregistered_external_package_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="registered study"):
        resolve_registered_package(tmp_path, "runtime/assets/package", tmp_path / "selection.json")


def test_path_traversal_is_rejected_before_registry_access(tmp_path):
    with pytest.raises(ValueError, match="traversal"):
        resolve_registered_package(tmp_path, "../other", tmp_path / "selection.json")


def test_changed_selection_is_rejected(tmp_path):
    checkout, selection, _ = fixture(tmp_path)
    dump(selection, {"package_path": "runtime/assets/other"})
    with pytest.raises(ValueError, match="registered deployment"):
        resolve_registered_package(checkout, "runtime/assets/package", selection)


def test_mount_to_other_package_is_rejected(tmp_path):
    checkout, selection, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="registered release"):
        resolve_registered_package(checkout, "runtime/assets/package", selection)


def test_release_registry_cannot_escape_release_storage(tmp_path):
    checkout, selection, registry = fixture(tmp_path)
    dump(registry / "releases/example.json", {"root": "../elsewhere", "manifest_sha256": "invalid"})
    with pytest.raises(ValueError, match="release storage"):
        resolve_registered_package(checkout, "runtime/assets/package", selection)

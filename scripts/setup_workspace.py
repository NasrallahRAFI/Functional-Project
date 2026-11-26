"""Recreate registered directory views after restoring shared payload storage.

Preview is the default. --apply creates missing junctions only; existing entries
are never replaced. Restore source/artifact payloads before applying the views.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="study-v3")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.study or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.study):
        parser.error("invalid study identifier")
    checkout = Path(__file__).resolve().parents[1]
    base = (args.base or (checkout.parent.parent if checkout.parent.name == "workspaces" else checkout.parent)).resolve()
    record = base / "repository/registry/compatibility" / (args.study + ".json")
    if not record.is_file():
        parser.error("this study has no registered compatibility views")
    views = json.loads(record.read_text(encoding="utf-8"))["views"]
    plan = []
    for view in views:
        source, target = base / view["view"], base / view["target"]
        if not source.resolve().is_relative_to(base) or not target.resolve().is_relative_to(base):
            raise ValueError("registered view escapes the workspace")
        if source.exists():
            if source.resolve() != target.resolve():
                raise FileExistsError(f"existing entry differs from its registered target: {source}")
            state = "already-mounted"
        elif not target.is_dir():
            raise FileNotFoundError(f"restore the registered payload first: {target}")
        else:
            state = "create-junction"
            if args.apply:
                if os.name != "nt":
                    raise OSError("registered views currently require Windows directory junctions")
                source.parent.mkdir(parents=True, exist_ok=True)
                quote = lambda value: "'" + str(value).replace("'", "''") + "'"
                command = f"$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path {quote(source)} -Target {quote(target)} | Out-Null"
                subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True)
        plan.append({**view, "state": state})
    print(json.dumps({"study": args.study, "applied": args.apply, "views": plan}, indent=2))


if __name__ == "__main__":
    main()

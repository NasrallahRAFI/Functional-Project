"""Derive fresh P1 inputs by applying the approved metadata-only MUSAN cap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from dronedet.cnn_v51.artifacts import contract_columns, read_csv, write_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-input", type=Path, required=True,
                        help="complete P1 attempt directory for the same immutable inventory")
    parser.add_argument("--selection", type=Path, required=True,
                        help="O0-A3 MUSAN byte selection manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite capped P1 output: {output}")
    with args.selection.open("r", encoding="utf-8", newline="") as stream:
        selection = list(csv.DictReader(stream))
    if not selection or any(row.get("selected") not in {"true", "false"} for row in selection):
        raise SystemExit("selection manifest is empty or has malformed boolean values")
    selected = {str(row["source_file_id"]) for row in selection if row["selected"] == "true"}
    if not selected:
        raise SystemExit("MUSAN selection is empty")
    inventory = read_csv(args.p1_input / "inventory_v51.csv", "inventory_v51.csv")
    labels = read_csv(args.p1_input / "label_register.csv", "label_register.csv")
    specs = json.loads((args.p1_input / "candidate_specs.json").read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        raise SystemExit("candidate_specs.json must contain a JSON array")
    musan_ids = {str(row["source_file_id"]) for row in inventory if row["source"] == "musan"}
    selection_ids = {str(row["source_file_id"]) for row in selection}
    if musan_ids != selection_ids:
        raise SystemExit("selection manifest does not cover the complete P1 MUSAN inventory")
    if not selected <= musan_ids:
        raise SystemExit("selection contains a MUSAN source file absent from P1 inventory")

    keep = lambda row: row["source"] != "musan" or str(row["source_file_id"]) in selected
    kept_inventory = [row for row in inventory if keep(row)]
    kept_labels = [row for row in labels if keep(row)]
    kept_specs = [row for row in specs if keep(row)]
    if sum(row["source"] == "musan" for row in kept_inventory) != len(selected):
        raise SystemExit("filtered MUSAN inventory count does not match selected file IDs")

    output.mkdir(parents=True)
    write_csv(output / "inventory_v51.csv", contract_columns("inventory_v51.csv"), kept_inventory,
              sort_by=None)
    write_csv(output / "label_register.csv", contract_columns("label_register.csv"), kept_labels,
              sort_by=None)
    (output / "candidate_specs.json").write_text(
        json.dumps(kept_specs, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (output / "legacy_reference.json").write_bytes((args.p1_input / "legacy_reference.json").read_bytes())
    (output / "selection_summary.json").write_bytes(args.selection.with_name("musan_budget_selection.json").read_bytes())
    print(json.dumps({"inventory_rows": len(kept_inventory), "label_rows": len(kept_labels),
                      "candidate_rows": len(kept_specs), "musan_selected_files": len(selected),
                      "output": str(output)}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

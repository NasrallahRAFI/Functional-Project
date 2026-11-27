"""O0-A4 metadata-only MUSAN selection under the 1 GiB cap.

Reserve group coverage for the two MUSAN strata constrained by B0, then fill
the remaining byte budget with deterministic rank order. The earlier O0-A3
selection remains immutable evidence and is not read as a selection input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from dronedet.cnn_v51.randomness import canonical_key, keyed_sha1  # noqa: E402
from select_musan_budget_v51 import OUTPUT_COLUMNS, read_csv, sha256, write_csv  # noqa: E402

TOTAL_CAP_BYTES = 1_073_741_824
SCOPE = "budget_1g_a4_v1"
CORE_GROUPS = {"speech": 20, "music": 10}


def select(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len({row["source_file_id"] for row in rows}) != len(rows):
        raise ValueError("source inventory contains duplicate source_file_id values")
    non_musan_bytes = sum(int(row["bytes"]) for row in rows if row["source"] != "musan")
    budget = TOTAL_CAP_BYTES - non_musan_bytes
    if budget <= 0:
        raise ValueError("non-MUSAN sources meet or exceed the 1 GiB cap")
    musan = [row for row in rows if row["source"] == "musan"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in musan:
        grouped[row["group_key"]].append(row)
    groups = []
    for group_key, members in grouped.items():
        classes = {row["source_class"] for row in members}
        strata = {row["negative_stratum"] for row in members}
        if len(classes) != 1 or len(strata) != 1:
            raise ValueError(f"MUSAN group metadata conflict: {group_key}")
        groups.append({"id": group_key, "members": members,
                       "source_class": next(iter(classes)),
                       "negative_stratum": next(iter(strata)),
                       "bytes": sum(int(row["bytes"]) for row in members)})

    chosen: set[str] = set()
    selected_bytes = 0
    reason: dict[str, str] = {}
    rank_digest: dict[str, str] = {}
    for source_class, minimum in CORE_GROUPS.items():
        candidates = [group for group in groups if group["source_class"] == source_class]
        if len(candidates) < minimum:
            raise ValueError(f"not enough {source_class} groups for O0-A4 core")
        candidates.sort(key=lambda group: (
            int(group["bytes"]),
            keyed_sha1("cap", ["musan", f"{SCOPE}_core", group["id"]]),
            canonical_key(["musan", f"{SCOPE}_core", group["id"]]),
        ))
        for group in candidates[:minimum]:
            if selected_bytes + int(group["bytes"]) > budget:
                raise ValueError("mandatory group-coverage core exceeds the MUSAN byte budget")
            selected_bytes += int(group["bytes"])
            chosen.add(str(group["id"]))
            reason[str(group["id"])] = "minimum_group_coverage_core"
            rank_digest[str(group["id"])] = keyed_sha1(
                "cap", ["musan", f"{SCOPE}_core", group["id"]])

    remaining = [group for group in groups if group["id"] not in chosen]
    remaining.sort(key=lambda group: (
        keyed_sha1("cap", ["musan", f"{SCOPE}_fill", group["id"]]),
        canonical_key(["musan", f"{SCOPE}_fill", group["id"]]),
    ))
    for group in remaining:
        if selected_bytes + int(group["bytes"]) <= budget:
            selected_bytes += int(group["bytes"])
            chosen.add(str(group["id"]))
            reason[str(group["id"])] = "deterministic_budget_fill"
            rank_digest[str(group["id"])] = keyed_sha1(
                "cap", ["musan", f"{SCOPE}_fill", group["id"]])

    output_rows: list[dict[str, object]] = []
    for row in sorted(musan, key=lambda item: item["source_file_id"]):
        is_selected = row["group_key"] in chosen
        output_rows.append({
            **{key: row[key] for key in OUTPUT_COLUMNS[:9]},
            "selection_rank_sha1": rank_digest.get(row["group_key"], ""),
            "selected": "true" if is_selected else "false",
            "selection_reason": reason.get(row["group_key"], "outside_deterministic_budget_subset"),
        })
    selected_files = [row for row in musan if row["group_key"] in chosen]
    musan_bytes = sum(int(row["bytes"]) for row in selected_files)
    summary = {
        "selection_scope": SCOPE,
        "selection_purpose": "cap",
        "selection_key": {"core": ["musan", f"{SCOPE}_core", "group_key"],
                          "fill": ["musan", f"{SCOPE}_fill", "group_key"]},
        "allocation": "minimum deterministic byte-efficient speech/music group core, then deterministic hash-ranked fill",
        "grouping": "retain all exact-byte aliases sharing group_key",
        "core_group_minima": CORE_GROUPS,
        "total_cap_bytes": TOTAL_CAP_BYTES,
        "non_musan_bytes": non_musan_bytes,
        "musan_budget_bytes": budget,
        "musan_selected_bytes": musan_bytes,
        "musan_unused_budget_bytes": budget - musan_bytes,
        "combined_selected_bytes": non_musan_bytes + musan_bytes,
        "musan_inventory_files": len(musan),
        "musan_selected_files": len(selected_files),
        "musan_selected_groups": len(chosen),
        "class_selected_groups": {name: sum(g["id"] in chosen and g["source_class"] == name for g in groups)
                                   for name in sorted({g["source_class"] for g in groups})},
        "class_selected_bytes": {name: sum(int(r["bytes"]) for r in selected_files if r["source_class"] == name)
                                   for name in sorted({g["source_class"] for g in groups})},
        "inventory_sha256": None,
    }
    if summary["combined_selected_bytes"] > TOTAL_CAP_BYTES:
        raise AssertionError("O0-A4 subset exceeds 1 GiB")
    return output_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    rows = read_csv(args.inventory)
    selected_rows, summary = select(rows)
    summary["inventory_sha256"] = sha256(args.inventory)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, selected_rows)
    summary["selection_manifest_sha256"] = sha256(args.output_csv)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("xb") as stream:
        stream.write((json.dumps(summary, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

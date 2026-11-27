"""Select a deterministic, group-safe MUSAN subset under the 1 GiB data cap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from dronedet.cnn_v51.randomness import canonical_key, keyed_sha1  # noqa: E402

TOTAL_CAP_BYTES = 1_073_741_824
SELECTION_SCOPE = "budget_1g_v1"
OUTPUT_COLUMNS = [
    "source_file_id", "source", "source_revision", "relative_path", "file_sha256",
    "bytes", "group_key", "source_class", "negative_stratum", "selection_rank_sha1",
    "selected", "selection_reason",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len({row["source_file_id"] for row in rows}) != len(rows):
        raise ValueError("source inventory contains duplicate source_file_id values")
    non_musan_bytes = sum(int(row["bytes"]) for row in rows if row["source"] != "musan")
    if non_musan_bytes >= TOTAL_CAP_BYTES:
        raise ValueError("non-MUSAN sources already meet or exceed the 1 GiB cap")
    budget = TOTAL_CAP_BYTES - non_musan_bytes
    musan = [row for row in rows if row["source"] == "musan"]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in musan:
        groups[row["group_key"]].append(row)
    group_records: list[dict[str, object]] = []
    for group_key, members in groups.items():
        classes = {row["source_class"] for row in members}
        strata = {row["negative_stratum"] for row in members}
        if len(classes) != 1 or len(strata) != 1:
            raise ValueError(f"MUSAN alias group has inconsistent metadata: {group_key}")
        key = ["musan", SELECTION_SCOPE, group_key]
        group_records.append({
            "group_key": group_key,
            "source_class": next(iter(classes)),
            "negative_stratum": next(iter(strata)),
            "bytes": sum(int(row["bytes"]) for row in members),
            "members": members,
            "rank": keyed_sha1("cap", key),
            "canonical": canonical_key(key),
        })

    class_bytes: dict[str, int] = defaultdict(int)
    for record in group_records:
        class_bytes[str(record["source_class"])] += int(record["bytes"])
    classes = sorted(class_bytes)
    total_musan_bytes = sum(class_bytes.values())
    quotas = {name: budget * class_bytes[name] // total_musan_bytes for name in classes}
    quotas[classes[-1]] += budget - sum(quotas.values())

    chosen_groups: set[str] = set()
    used_by_class: dict[str, int] = defaultdict(int)
    for source_class in classes:
        ranked = sorted((g for g in group_records if g["source_class"] == source_class),
                         key=lambda g: (g["rank"], g["canonical"]))
        for group in ranked:
            size = int(group["bytes"])
            if used_by_class[source_class] + size <= quotas[source_class]:
                chosen_groups.add(str(group["group_key"]))
                used_by_class[source_class] += size

    rank_by_group = {str(group["group_key"]): str(group["rank"]) for group in group_records}
    output_rows: list[dict[str, object]] = []
    for row in sorted(musan, key=lambda item: item["source_file_id"]):
        chosen = row["group_key"] in chosen_groups
        output_rows.append({
            **{key: row[key] for key in OUTPUT_COLUMNS[:9]},
            "selection_rank_sha1": rank_by_group[row["group_key"]],
            "selected": "true" if chosen else "false",
            "selection_reason": "selected_group_within_class_byte_quota" if chosen else "outside_deterministic_budget_subset",
        })
    selected_bytes = sum(int(row["bytes"]) for row in musan if row["group_key"] in chosen_groups)
    if non_musan_bytes + selected_bytes > TOTAL_CAP_BYTES:
        raise AssertionError("selected corpus exceeds total byte cap")
    summary: dict[str, object] = {
        "selection_scope": SELECTION_SCOPE,
        "selection_key": ["musan", SELECTION_SCOPE, "group_key"],
        "selection_purpose": "cap",
        "grouping": "retain all exact-byte aliases sharing group_key",
        "allocation": "MUSAN byte budget apportioned by original source_class byte shares; ranked groups greedily accepted within each class quota",
        "total_cap_bytes": TOTAL_CAP_BYTES,
        "non_musan_bytes": non_musan_bytes,
        "musan_budget_bytes": budget,
        "musan_selected_bytes": selected_bytes,
        "musan_unused_budget_bytes": budget - selected_bytes,
        "combined_selected_bytes": non_musan_bytes + selected_bytes,
        "inventory_sha256": None,
        "musan_inventory_files": len(musan),
        "musan_selected_files": sum(row["group_key"] in chosen_groups for row in musan),
        "musan_selected_groups": len(chosen_groups),
        "class_byte_quotas": quotas,
        "class_selected_bytes": dict(used_by_class),
        "class_selected_groups": {
            name: sum(g["group_key"] in chosen_groups and g["source_class"] == name for g in group_records)
            for name in classes
        },
    }
    return output_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="immutable P1 inventory_v51.csv")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    rows = read_csv(args.inventory)
    selected_rows, summary = select(rows)
    summary["inventory_sha256"] = sha256(args.inventory)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, selected_rows)
    summary["selection_manifest_sha256"] = sha256(args.output_csv)
    summary_bytes = (json.dumps(summary, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("xb") as stream:
        stream.write(summary_bytes)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

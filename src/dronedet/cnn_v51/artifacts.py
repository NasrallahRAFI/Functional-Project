"""Canonical, closed-schema CSV and JSON serialization helpers."""

from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import math
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable

CONTRACT_PATH = Path(__file__).resolve().parents[3] / "docs" / "v5.1" / "DATA_PREPARATION_CNN_V5_1_CONTRACT.json"


@lru_cache(maxsize=1)
def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _schema_for_columns(columns: list[str]) -> tuple[str, list[dict[str, Any]]]:
    for name, schema in _contract()["artifact_schemas"].items():
        if [column["name"] for column in schema] == columns:
            return name, schema
    raise ValueError("column list does not match a closed V5.1 artifact schema")


def _validate_row(row: dict[str, Any], schema: list[dict[str, Any]], artifact: str, row_number: int) -> None:
    enums = _contract()["enums"]
    for column in schema:
        name, kind, nullable = column["name"], column["type"], column["nullable"]
        value = row[name]
        if value is None:
            if nullable:
                continue
            raise ValueError(f"{artifact} row {row_number} has null required {name}")
        invalid = False
        if kind == "bool":
            invalid = not isinstance(value, bool)
        elif kind == "binary":
            invalid = isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
        elif kind == "int64":
            invalid = isinstance(value, bool) or not isinstance(value, int) or not -(2**63) <= value < 2**63
        elif kind == "float64":
            invalid = isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value))
        elif kind == "sha256":
            invalid = not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        elif kind == "relative_path":
            invalid = (
                not isinstance(value, str) or not value or "\\" in value or value.startswith("/")
                or any(part in {"", ".", ".."} for part in value.split("/"))
                or unicodedata.normalize("NFC", value) != value
            )
        elif kind in enums:
            invalid = not isinstance(value, str) or value not in enums[kind]
        elif kind == "string":
            invalid = not isinstance(value, str) or value == ""
        else:
            raise ValueError(f"unhandled V5.1 schema type {kind!r} in {artifact}.{name}")
        if invalid:
            raise ValueError(f"{artifact} row {row_number} has invalid {kind} value in {name}: {value!r}")


def contract_columns(artifact_name: str) -> list[str]:
    contract = _contract()
    try:
        schema = contract["artifact_schemas"][artifact_name]
    except KeyError as exc:
        raise ValueError(f"unknown V5.1 artifact schema: {artifact_name}") from exc
    return [column["name"] for column in schema]


def read_csv(path: str | Path, artifact_name: str) -> list[dict[str, Any]]:
    """Read a closed V5.1 artifact using only its declared schema and types."""
    contract = _contract()
    try:
        schema = contract["artifact_schemas"][artifact_name]
    except KeyError as exc:
        raise ValueError(f"unknown V5.1 artifact schema: {artifact_name}") from exc
    columns = [column["name"] for column in schema]
    output: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"{artifact_name} header differs from its closed schema")
        for row_number, raw in enumerate(reader, start=2):
            parsed: dict[str, Any] = {}
            for column in schema:
                name, kind, nullable = column["name"], column["type"], column["nullable"]
                value = raw[name]
                if value == "" and nullable:
                    parsed[name] = None
                    continue
                if value == "" and not nullable:
                    raise ValueError(f"{artifact_name}:{row_number} has empty required {name}")
                if kind == "bool":
                    if value not in {"true", "false"}:
                        raise ValueError(f"{artifact_name}:{row_number} has invalid bool in {name}")
                    parsed[name] = value == "true"
                elif kind in {"int64", "binary"}:
                    try:
                        integer = int(value, 10)
                    except ValueError as exc:
                        raise ValueError(f"{artifact_name}:{row_number} has invalid integer in {name}") from exc
                    if kind == "binary" and integer not in {0, 1}:
                        raise ValueError(f"{artifact_name}:{row_number} has non-binary value in {name}")
                    parsed[name] = integer
                elif kind in {"float64", "float"}:
                    try:
                        number = float(value)
                    except ValueError as exc:
                        raise ValueError(f"{artifact_name}:{row_number} has invalid float in {name}") from exc
                    if not math.isfinite(number):
                        raise ValueError(f"{artifact_name}:{row_number} has non-finite float in {name}")
                    parsed[name] = number
                else:
                    parsed[name] = value
            output.append(parsed)
            _validate_row(parsed, schema, artifact_name, row_number)
    return output


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_csv(path: str | Path, columns: list[str], rows: Iterable[dict[str, Any]], *,
              sort_by: str | None = "clip_id") -> str:
    """Write UTF-8/LF CSV with a fixed schema and deterministic row order."""
    materialized = list(rows)
    artifact_name, schema = _schema_for_columns(columns)
    expected = set(columns)
    for index, row in enumerate(materialized):
        if set(row) != expected:
            raise ValueError(f"row {index} has schema mismatch: missing={sorted(expected-set(row))}, extra={sorted(set(row)-expected)}")
        _validate_row(row, schema, artifact_name, index + 1)
    if sort_by is not None:
        if sort_by not in expected:
            raise ValueError(f"sort column {sort_by!r} is not in schema")
        materialized.sort(key=lambda row: (str(row[sort_by]), canonical_json_bytes(row)))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("CSV cannot contain NaN or infinity")
        return format(value, ".17g")
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported CSV value type: {type(value).__name__}")

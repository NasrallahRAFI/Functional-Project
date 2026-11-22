"""Dataset inventory API used by notebooks 00 and 01."""
from __future__ import annotations

from .data_preparation import (
    CLASS_DIRS,
    build_inventory,
    clip_stats,
    environment_log,
    list_binary_wavs,
    locate_dataset_root,
    l_rule_table,
    measure_resampler_edge,
    parse_clip,
)

GATES: list[dict] = []


def gate(name, got, expected, tol: float = 0.0) -> bool:
    """Record and print one numeric/equality audit gate."""
    ok = got == expected if tol == 0 else abs(got - expected) <= tol
    GATES.append({"gate": name, "expected": expected, "got": got, "ok": bool(ok)})
    print("PASS" if ok else "FAIL", name, "| expected", expected, "| got", got)
    return bool(ok)


__all__ = ["CLASS_DIRS", "locate_dataset_root", "list_binary_wavs", "clip_stats", "build_inventory",
           "parse_clip", "l_rule_table", "gate", "measure_resampler_edge", "environment_log", "GATES"]

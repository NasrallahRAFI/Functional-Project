"""Cleaning, group-aware split creation, and deterministic manifest serialization."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MANIFEST_COLUMNS = ["clip_id", "label", "label_name", "drone_type", "family", "style", "recording", "group", "segment",
                    "n_samples", "sr", "duration_s", "rms_dbfs", "peak", "zero_frac", "dc", "clipped_frac",
                    "all_zero", "silent", "near_silent", "partial_zero", "clipped", "clipped_flag", "dc_flag", "pcm_sha1", "dup_count",
                    "kept", "exclusion_reason", "pinned", "split"]


class TestAccessError(PermissionError):
    """Raised when a test artifact is requested outside the final scoring pass."""


def guard_test_access(name: str, *, allow_test: bool = False) -> None:
    if str(name).strip().lower() == "test" and not allow_test:
        raise TestAccessError("test data is locked until the final comparison pass")


def assign_group(m: pd.DataFrame) -> pd.Series:
    """Derive the final group key, including the 13 approved B_S2_D1 blocks."""
    groups = m["group_raw"].copy()
    b = m["family"] == "B_S2_D1"
    if b.any():
        if m.loc[b, "rec_no"].min() != 67 or m.loc[b, "rec_no"].max() != 133:
            raise ValueError("B_S2_D1 recording range differs from the approved 67..133 rule")
        start = np.minimum(67 + 5 * ((m.loc[b, "rec_no"] - 67) // 5), 127)
        groups.loc[b] = "B_S2_D1_blk" + start.astype(int).astype(str).str.zfill(3)
    return groups


def flag_columns(inv: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
    m = inv.copy()
    m["label"] = (m.folder == "yes_drone").astype(np.int8)
    m["label_name"] = np.where(m.label == 1, "drone", "unknown")
    m["duration_s"] = m.n_samples / m.sr
    parsed_columns = {"style", "family", "drone_type", "recording", "rec_no", "segment", "group_raw"}
    if not parsed_columns.issubset(m.columns):
        parsed = pd.DataFrame([__import__("dronedet.data_preparation", fromlist=["parse_clip"]).parse_clip(cid) for cid in m.clip_id])
        m = pd.concat([m.reset_index(drop=True), parsed], axis=1)
    nz = ~m.all_zero
    m["near_silent"] = nz & (m.rms_dbfs < cleaning["near_silent_dbfs"])
    m["partial_zero"] = nz & (m.zero_frac > cleaning["partial_zero_frac"])
    m["clipped_flag"] = m.clipped_frac > cleaning["clipped_frac"]
    m["silent"] = m["all_zero"]
    m["clipped"] = m["clipped_flag"]
    m["dc_flag"] = m.dc.abs() > cleaning["dc_abs"]
    m["dup_count"] = m.groupby("pcm_sha1").clip_id.transform("size")
    m["group"] = assign_group(m)
    return m


def choose_zero_representatives(m: pd.DataFrame) -> list[str]:
    zero = m[m.all_zero & m.kept]
    reps = []
    for _, group in zero.groupby("pcm_sha1"):
        silence = group[group["style"] == "silence"]
        reps.append(sorted((silence if len(silence) else group).clip_id)[0])
    return sorted(reps)


def apply_cleaning(m: pd.DataFrame, win_samples: int):
    m = m.copy()
    m["kept"], m["exclusion_reason"] = True, ""
    if "error" in m and not (m.error == "").all():
        raise ValueError("inventory contains unreadable files")
    short = m.n_samples < win_samples / 2
    m.loc[short, ["kept", "exclusion_reason"]] = [False, "too_short"]
    reps = choose_zero_representatives(m)
    duplicate = m.all_zero & m.kept & ~m.clip_id.isin(reps)
    m.loc[duplicate, ["kept", "exclusion_reason"]] = [False, "exact_zero_duplicate"]
    m["pinned"] = False
    pinned_groups = set(m.loc[m.clip_id.isin(reps), "group"]) | {"extra_membo_D2"}
    m.loc[m.kept & m["group"].isin(pinned_groups), "pinned"] = True
    return m, reps


def two_stage_split(pool: pd.DataFrame, seed: int, n_test: int = 7, n_val: int = 6) -> pd.Series:
    pool = pool.sort_values("clip_id")
    y, groups = pool.label.to_numpy(), pool.group.to_numpy()
    first = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(iter(first.split(np.zeros(len(pool)), y, groups)))
    rest = pool.iloc[rest_idx]
    second = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    _, val_idx = next(iter(second.split(np.zeros(len(rest)), rest.label.to_numpy(), rest.group.to_numpy())))
    out = pd.Series("train", index=pool.index)
    out.iloc[test_idx] = "test"
    out.loc[rest.index[val_idx]] = "val"
    return out


def split_gates(m: pd.DataFrame, expected_kept: int = 10623) -> pd.DataFrame:
    kept = m[m.kept]
    gates = []
    def add(name, ok, detail=""):
        gates.append({"gate": name, "ok": bool(ok), "detail": str(detail)})
    add("S1 group in one split", (kept.groupby("group")["split"].nunique() == 1).all())
    add("S2 PCM identity in one split", (kept.groupby("pcm_sha1").split.nunique() == 1).all())
    add("S3 both classes in every split", all(set(kept[kept.split == s].label) == {0, 1} for s in ("train", "val", "test")))
    add("S4 kept pool size", len(kept) == expected_kept and set(kept.split) == {"train", "val", "test"}, len(kept))
    add("S5 no exact-zero outside train", not kept[kept.split != "train"].all_zero.any())
    extra = kept[kept.family == "extra_membo_D2"]
    add("S6 extra_membo_D2 train only", len(extra) == 56 and (extra.split == "train").all(), len(extra))
    blocks = kept[kept.family == "B_S2_D1"]
    add("S7 B_S2_D1 blocks", blocks.group.nunique() == 13 and (blocks.groupby("group").split.nunique() == 1).all())
    add("S8 pinned groups train only", (kept[kept.pinned]["split"] == "train").all())
    return pd.DataFrame(gates)


def write_manifest(df: pd.DataFrame, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("clip_id")[MANIFEST_COLUMNS].to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.8g")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(paths: dict, name: str, *, allow_test: bool = False) -> pd.DataFrame:
    guard_test_access(name, allow_test=allow_test)
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {name}")
    return pd.read_csv(Path(paths["splits"]) / f"{name}.csv", keep_default_na=False)

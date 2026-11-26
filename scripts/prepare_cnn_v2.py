"""Build or verify the persistent CNN-only V2.0 mel-power dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from dronedet.config import config_from_mapping
from dronedet.data_preparation import build_inventory, list_binary_wavs, locate_dataset_root
from dronedet.features import file_to_power, to_db
from dronedet.normalization import Normalizer


SCHEMA = 4
REPRESENTATION = "cnn_mel_power"
SPLITS = ("train", "val", "test")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError(f"invalid boolean manifest value: {value!r}")


def load_recipe(path: Path) -> tuple[dict, object]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    fe = config_from_mapping(doc["frontend"])
    if doc.get("version") != "CNN_V2.0" or doc["seed"] != 42:
        raise ValueError("unsupported CNN V2 recipe or seed")
    fe.validate()
    if fe.n_mels != 64 or fe.n_win != 28 or fe.fmax != 7500.0:
        raise ValueError("CNN V2 frontend identity differs from the frozen recipe")
    return doc, fe


def _fresh_source_manifest(root: Path, reference_path: Path, split_root: Path,
                           expected_split_hashes: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    base = locate_dataset_root(root)
    ids = list_binary_wavs(base)
    if not ids:
        raise RuntimeError(f"no WAV files under {base}")
    fresh = build_inventory(base, ids)
    ref = pd.read_csv(reference_path, keep_default_na=False)
    if len(fresh) != len(ref) or set(fresh.clip_id) != set(ref.clip_id):
        raise RuntimeError("fresh raw inventory membership differs from the reviewed V1 source inventory")
    combined = ref.merge(fresh, on="clip_id", suffixes=("_ref", "_fresh"), validate="one_to_one")
    for field in ("n_samples", "sr", "channels", "pcm_sha1", "truncated", "error"):
        left, right = combined[f"{field}_ref"], combined[f"{field}_fresh"]
        if field == "error":
            if (left.astype(str) != right.astype(str)).any():
                raise RuntimeError("fresh inventory contains an error or differs from the reviewed inventory")
        elif field == "pcm_sha1":
            if (left.astype(str) != right.astype(str)).any():
                raise RuntimeError("raw waveform PCM identities differ from the reviewed inventory")
        elif field == "truncated":
            if (left.astype(str).str.lower() != right.astype(str).str.lower()).any():
                raise RuntimeError("raw WAV truncation flags differ from the reviewed inventory")
        elif (pd.to_numeric(left) != pd.to_numeric(right)).any():
            raise RuntimeError(f"raw inventory field changed: {field}")
    ref = ref.copy()
    ref["kept"] = ref["kept"].astype(str).str.lower().isin(("true", "1"))
    if len(ref) != 11704 or int(ref.kept.sum()) != 10623:
        raise RuntimeError("reviewed V1 cleaning membership does not match the approved 11,704/10,623 counts")
    exclusions = ref.loc[~ref.kept, "exclusion_reason"].value_counts().to_dict()
    if exclusions != {"exact_zero_duplicate": 1079, "too_short": 2}:
        raise RuntimeError(f"reviewed V1 cleaning exclusions differ from the approved rules: {exclusions}")
    kept = ref[ref.kept].copy()
    if kept.clip_id.duplicated().any() or kept.group.isna().any():
        raise RuntimeError("kept source manifest has duplicate clips or missing groups")
    if int(kept.groupby("group").split.nunique().max()) != 1:
        raise RuntimeError("group membership leaks across the frozen train/validation/test split")
    counts = kept.groupby("split").agg(clips=("clip_id", "size"), positives=("label", "sum"), groups=("group", "nunique"))
    positive_groups = kept[kept.label.astype(int) == 1].groupby("split").group.nunique()
    expected = {"train": (7607, 967, 1249, 146), "val": (1510, 185, 247, 29), "test": (1506, 180, 247, 28)}
    for split, (n, pos, groups, pos_groups) in expected.items():
        actual = tuple(map(int, counts.loc[split])) + (int(positive_groups.get(split, 0)),) if split in counts.index else ()
        if actual != (n, pos, groups, pos_groups):
            raise RuntimeError(f"approved split identity changed for {split}: {counts.loc[split].to_dict() if split in counts.index else 'missing'}")
    for split in SPLITS:
        split_path = split_root / f"{split}.csv"
        if digest_file(split_path) != expected_split_hashes[split]:
            raise RuntimeError(f"frozen V1 {split} manifest checksum mismatch")
        frozen = pd.read_csv(split_path, keep_default_na=False)
        fresh_rows = kept[kept.split == split]
        for field in ("clip_id", "label", "group"):
            left = frozen[field].astype(str).sort_values().reset_index(drop=True)
            right = fresh_rows[field].astype(str).sort_values().reset_index(drop=True)
            if not left.equals(right):
                raise RuntimeError(f"fresh {split} {field} assignments differ from the frozen V1 manifest")
    assignment = kept[["clip_id", "label", "group", "split"]].copy().sort_values("clip_id")
    assignment_bytes = assignment.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return kept, {"raw_root": str(base), "inventory_count": len(fresh),
                  "inventory_sha256": digest_bytes(fresh.to_csv(index=False, lineterminator="\n").encode()),
                  "assignment_sha256": digest_bytes(assignment_bytes), "counts": counts.reset_index().to_dict("records")}


def _build_split(manifest: pd.DataFrame, raw_root: Path, fe, stage: Path, split: str, allow_test: bool) -> dict:
    if split == "test" and not allow_test:
        raise PermissionError("test cache build requires the explicit preparation build operation")
    rows: list[dict] = []
    chunks: list[np.ndarray] = []
    cursor = 0
    audio_root = locate_dataset_root(raw_root) / "Binary_Drone_Audio"
    for row in manifest.to_dict("records"):
        power, flags = file_to_power(audio_root / row["clip_id"], fe)
        if power.dtype != np.float32 or not np.isfinite(power).all() or (power < 0).any() or power.shape[0] != 64 or power.shape[1] < 28:
            raise ValueError(f"{row['clip_id']}: invalid full-clip mel power")
        frames = int(power.shape[1])
        center = (frames - 28) // 2
        crop_zero = bool(np.all(power[:, center:center + 28] == 0))
        silent = as_bool(row.get("silent", row.get("all_zero", False)))
        near = as_bool(row.get("near_silent", False))
        eligible = not (silent or near or crop_zero)
        if flags["silent"] != silent or flags["near_silent"] != near:
            raise RuntimeError(f"fresh quality flags differ from reviewed cleaning policy: {row['clip_id']}")
        rows.append({"clip_id": row["clip_id"], "start_frame": cursor, "n_frames": frames,
                     "center_start_frame": center, "silent": silent, "near_silent": near,
                     "clipped": as_bool(row.get("clipped", False)), "partial_zero": as_bool(row.get("partial_zero", False)),
                     "crop_all_zero": crop_zero, "stats_eligible": eligible,
                     "length_status": flags["length_status"]})
        chunks.append(power)
        cursor += frames
        if len(rows) % 1000 == 0:
            print(f"{split}: rebuilt {len(rows):,}/{len(manifest):,} full-clip spectrograms", flush=True)
    power_cache = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    index = pd.DataFrame(rows)
    if len(index) != len(manifest) or int(index.start_frame.iloc[0]) != 0:
        raise RuntimeError(f"{split}: cache row/index mismatch")
    if int(index.iloc[-1].start_frame + index.iloc[-1].n_frames) != power_cache.shape[1]:
        raise RuntimeError(f"{split}: cache index does not span payload")
    out = stage / "cache"
    out.mkdir(parents=True, exist_ok=True)
    power_path = out / f"mel_power_{split}.npy"
    index_path = out / f"index_{split}.csv"
    np.save(power_path, power_cache, allow_pickle=False)
    index.to_csv(index_path, index=False, lineterminator="\n")
    return {"power": power_cache, "index": index, "paths": (power_path, index_path)}


def build(config_path: Path) -> None:
    config_path = config_path.resolve()
    doc, fe = load_recipe(config_path)
    project = config_path.parent.parent
    raw_value = os.environ.get("DRONE_AUDIO_DATASET_ROOT", doc["raw_root"])
    raw_path = Path(raw_value).expanduser()
    raw_root = (project / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
    ref_manifest = project / doc["reference_manifest"]
    output = project / doc["output_root"] / "data"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite CNN V2 dataset: {output}")
    split_root = project / doc["reference_splits"]
    kept, source = _fresh_source_manifest(raw_root, ref_manifest, split_root,
                                           doc["reference_split_hashes"])
    print(f"Fresh raw inventory validated: {source['inventory_count']:,} files; rebuilding {len(kept):,} retained clips.", flush=True)
    stage = Path(tempfile.mkdtemp(prefix="cnn_v2_stage_", dir=output.parent if output.parent.exists() else project))
    started = time.time()
    try:
        split_records = {}
        train_meta = None
        for split in SPLITS:
            frame = kept[kept.split == split].reset_index(drop=True)
            record = _build_split(frame, raw_root, fe, stage, split, allow_test=True)
            print(f"{split}: cache ready with {len(frame):,} clips.", flush=True)
            split_records[split] = record
            if split == "train":
                train_meta = (frame, record)
        train_frame, tr = train_meta
        stats_rows = []
        power_by_file = {}
        for manifest_row, idx_row in zip(train_frame.to_dict("records"), tr["index"].to_dict("records")):
            stats_rows.append({**manifest_row, **idx_row, "split": "train"})
            lo, n = int(idx_row["start_frame"]), int(idx_row["n_frames"])
            power_by_file[str(manifest_row["clip_id"])] = tr["power"][:, lo:lo + n]
        normalizer = Normalizer.fit(stats_rows, fe, power_by_file)
        if normalizer.mean.shape != (64,) or normalizer.std.shape != (64,) or not np.isfinite(normalizer.mean).all() or not np.isfinite(normalizer.std).all() or np.any(normalizer.std <= 0):
            raise RuntimeError("CNN V2 train-only statistics are invalid")
        if len(stats_rows) != 7607 or int(normalizer.fit_record.get("eligible_clips", -1)) != 7519:
            raise RuntimeError(f"CNN V2 normalizer train/eligible counts differ: {normalizer.fit_record}")
        stats_json = stage / "normalization.json"
        normalizer.save(stats_json)
        np.savez_compressed(stage / "normalization_stats.npz", mean=normalizer.mean, std=normalizer.std)
        # Bind all output files and split manifests into one complete schema-4 transaction.
        manifests = stage / "manifests"
        manifests.mkdir()
        file_records = []
        for split in SPLITS:
            frame = kept[kept.split == split].reset_index(drop=True)
            manifest_path = manifests / f"{split}.csv"
            frame.to_csv(manifest_path, index=False, lineterminator="\n")
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p.name != "DATASET_READY.json":
                file_records.append({"path": p.relative_to(stage).as_posix(), "sha256": digest_file(p), "size": p.stat().st_size})
        ready = {"schema_version": SCHEMA, "representation": REPRESENTATION, "status": "READY",
                 "source": source, "config_sha256": digest_file(config_path), "files": file_records,
                 "stats_policy": "train_stats_eligible_per_band_floor_mask_v1",
                 "stats_fit": normalizer.fit_record, "duration_seconds": round(time.time() - started, 3),
                 "test_payload_built_for": "mechanical cache generation only; no model/test scoring"}
        ready_path = stage / "DATASET_READY.json"
        ready_path.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite CNN V2 dataset: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, output)
        print(json.dumps({"status": "READY", "output": str(output), "files": len(file_records),
                          "source": source, "stats_fit": normalizer.fit_record}, indent=2))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify(config_path: Path) -> None:
    config_path = config_path.resolve()
    doc, _ = load_recipe(config_path)
    root = config_path.parent.parent / doc["output_root"] / "data"
    ready = json.loads((root / "DATASET_READY.json").read_text(encoding="utf-8"))
    if ready.get("schema_version") != SCHEMA or ready.get("representation") != REPRESENTATION or ready.get("status") != "READY":
        raise RuntimeError("CNN V2 dataset readiness marker has unsupported schema/status")
    safe_records = [rec for rec in ready["files"] if not (Path(rec["path"]).name.endswith("_test.npy")
                    or Path(rec["path"]).name.endswith("_test.csv")
                    or Path(rec["path"]).as_posix() == "manifests/test.csv")]
    skipped_test_records = len(ready["files"]) - len(safe_records)
    for rec in safe_records:
        path = root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["size"] or digest_file(path) != rec["sha256"]:
            raise RuntimeError(f"CNN V2 dataset transaction member failed verification: {rec['path']}")
    expected_shapes = {"train": (64, 457897), "val": (64, 91761)}
    for split in ("train", "val"):
        x = np.load(root / "cache" / f"mel_power_{split}.npy", mmap_mode="r", allow_pickle=False)
        idx = pd.read_csv(root / "cache" / f"index_{split}.csv")
        if x.dtype != np.float32 or x.shape != expected_shapes[split] or not np.isfinite(x).all() or (x < 0).any():
            raise RuntimeError(f"{split} CNN cache payload is invalid")
        if len(idx) != (7607 if split == "train" else 1510):
            raise RuntimeError(f"{split} CNN cache row count is incorrect")
        if split == "train" and int(idx.stats_eligible.sum()) != 7519:
            raise RuntimeError("train statistics eligibility count is incorrect")
    print(json.dumps({"status": "VERIFIED", "schema_version": SCHEMA,
                      "scope": "cnn_v2_preparation_safe_no_test_payload_opened",
                      "dataset": str(root), "verified_files": len(safe_records),
                      "test_payload": "not read", "test_records_skipped": skipped_test_records}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("build", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "build":
        build(args.config)
    else:
        verify(args.config)


if __name__ == "__main__":
    main()

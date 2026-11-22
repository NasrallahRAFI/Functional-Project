"""Dataset audit plots. Call from notebook 00 before the split is frozen."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def audit_figures(inv: pd.DataFrame, output_dir: Path, *, window_lengths=(7936, 15872, 23808, 31744)) -> list[Path]:
    """Write length, nonzero RMS, and raw group-size plots for the pre-split audit."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for folder, label in (("unknown", "Unknown"), ("yes_drone", "Drone")):
        values = inv.loc[inv.folder == folder, "n_samples"]
        ax.hist(values, bins=40, alpha=0.55, label=label)
    for samples in window_lengths:
        ax.axvline(samples, linestyle="--", alpha=0.45, label=f"Window {samples:,} samples")
    ax.set(xlabel="Decoded clip length (samples)", ylabel="Clips", title="Clip lengths by label")
    ax.legend(fontsize="small")
    fig.tight_layout(); path = output_dir / "audit_clip_lengths.png"; fig.savefig(path, dpi=160); plt.close(fig); outputs.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    for folder, label in (("unknown", "Unknown"), ("yes_drone", "Drone")):
        values = inv.loc[(inv.folder == folder) & ~inv.all_zero, "rms_dbfs"]
        ax.hist(values, bins=45, alpha=0.55, label=label)
    ax.set(xlabel="RMS level (dBFS)", ylabel="Clips", title="Nonzero clip levels")
    ax.legend(); fig.tight_layout(); path = output_dir / "audit_rms_dbfs.png"; fig.savefig(path, dpi=160); plt.close(fig); outputs.append(path)

    sizes = inv.groupby("group_raw").size()
    counts = sizes.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set(xlabel="Clips in raw group", ylabel="Number of groups", title="Group size distribution")
    fig.tight_layout(); path = output_dir / "audit_group_sizes.png"; fig.savefig(path, dpi=160); plt.close(fig); outputs.append(path)
    return outputs


def train_only_audit_figures(inventory_path: Path, train_manifest_path: Path, output_dir: Path,
                             *, window_lengths=(7936, 15872, 23808, 32000)) -> list[Path]:
    """Write the N00 audit figures from inventory rows joined only to the frozen train manifest."""
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    train_manifest = pd.read_csv(train_manifest_path, keep_default_na=False)
    if "split" not in train_manifest or not train_manifest["split"].eq("train").all():
        raise ValueError("the supplied frozen manifest must contain train rows only")
    train_rows = train_manifest.loc[train_manifest["split"].eq("train"), ["clip_id", "split", "group"]]
    train_inv = train_rows.merge(inventory, on="clip_id", how="left", validate="one_to_one")
    if len(train_inv) != len(train_rows) or train_inv["error"].ne("").any():
        raise ValueError("train manifest clips must all match successfully decoded inventory rows")
    if not train_inv["split"].eq("train").all():
        raise AssertionError("non-train rows reached the figure data")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_count = len(train_inv)
    group_count = int(train_rows["group"].nunique())
    caption = f"Train split only ({clip_count} clips, {group_count} groups), generated after split freeze"
    outputs = []

    fig, ax = plt.subplots(figsize=(9, 5))
    for folder, label in (("unknown", "Unknown"), ("yes_drone", "Drone")):
        values = train_inv.loc[train_inv["folder"].eq(folder), "n_samples"]
        ax.hist(values, bins=40, alpha=0.55, label=label)
    for samples in window_lengths:
        ax.axvline(samples, linestyle="--", alpha=0.45, label=f"Window {samples:,} samples")
    ax.set(xlabel="Decoded clip length (samples)", ylabel="Clips")
    ax.legend(fontsize="small")
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "audit_clip_lengths.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    nonzero = ~train_inv["all_zero"].astype(str).str.lower().isin(("true", "1"))
    fig, ax = plt.subplots(figsize=(9, 5))
    for folder, label in (("unknown", "Unknown"), ("yes_drone", "Drone")):
        values = train_inv.loc[train_inv["folder"].eq(folder) & nonzero, "rms_dbfs"]
        ax.hist(values, bins=45, alpha=0.55, label=label)
    ax.set(xlabel="RMS level (dBFS)", ylabel="Clips")
    ax.legend()
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "audit_rms_dbfs.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    raw_group_sizes = train_inv.groupby("group_raw", sort=True).size()
    group_size_counts = raw_group_sizes.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(group_size_counts.index.astype(str), group_size_counts.values)
    ax.set(xlabel="Clips in raw group", ylabel="Number of groups")
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "audit_group_sizes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)
    return outputs

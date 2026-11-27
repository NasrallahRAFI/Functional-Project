# O0-A4 — 1 GiB MUSAN subset with minimum B0 group support

**Authorization basis:** The owner instructed continuation under the new 1 GiB decision and requested updates to the study sections and Colab files. O0-A4 preserves the O0-A3 failure and creates a new selection identity; it does not retry or edit corpus-006/split-006.

## Unchanged owner constraints

- Combined inventoried raw-audio payload remains capped at **1,073,741,824 bytes**.
- All AE/DD/ITU files remain included; the MUSAN allowance is recomputed from their byte total.
- O0-A2 source/stratum shares, group/source/clip caps, descriptor margins, solver profile, and all P3/P4 gates remain unchanged.
- corpus-006/split-006 and its P4 B0 failure remain immutable history.

## Revised deterministic selection for corpus-007

O0-A3 demonstrated that byte-proportional quotas combined with hash-first inclusion produced too few training speech groups for the fixed N3 share and 1.5% per-group cap. O0-A4 changes only the MUSAN subset selection order:

1. Use the complete P1 metadata inventory; never decode audio or infer labels while selecting.
2. Group MUSAN exact-byte aliases by `group_key` and keep each group indivisible.
3. Reserve a core of 20 `speech` groups and 10 `music` groups. These minima are derived from the fixed split fractions and B0 group cap: they are intended to leave at least 15 N3 speech groups and 6 N4 music groups in train after the specified MUSAN split. They do not change any gate threshold.
4. Within each required source class, choose the smallest-byte groups first; break equal-byte ties by the existing `cap` purpose, key `["musan", "budget_1g_a4_v1_core", group_key]`.
5. Fill the remaining MUSAN byte allowance by existing `cap` rank order with key `["musan", "budget_1g_a4_v1_fill", group_key]`, skipping groups that do not fit. The total must remain at or below the exact MUSAN allowance.
6. Record every selected/excluded file and group, byte count, rank digest, reason, P1 inventory hash, selection-manifest hash, and combined-byte total.

The minimum-byte core makes recording duration less representative than a random-file subset; that limitation is explicit. P3 and P4 must still pass without changing fixed rules. A failed gate ends corpus-007's attempt.

## New identity boundary

Use planned corpus-007/split-007 and derived-build-v51-005/cnn-v51-005. Build fresh P1/P2/P3 artifacts from the O0-A4 selection, run independent P3 reproduction, and rerun P4. Do not reuse O0-A3 or corpus-005 P4 artifacts. P5 may proceed only if the new P4 passes.

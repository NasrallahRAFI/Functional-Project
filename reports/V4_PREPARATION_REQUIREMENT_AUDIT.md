# CNN V4.0 preparation requirement audit

Date: 2026-11-26

Authoritative protocol: `docs/v4/DATA_PREPARATION_CNN_V4_0.md` (SHA-256 `2bb6e2bc3ad13f5d5ede122982375b27fe851a8a38d04a39f2f6207674f5222d`)
Execution result: **PASS**; the split was frozen after all pre-split and split gates passed.

## Requirement-by-requirement evidence

| Plan requirement | Evidence and result | Audit status |
|---|---|---|
| Scope is exactly Al-Emadi binary and thesis DD audio; exclude Salford | `audits/status.json` records included counts, `salford_processed: false`; environment log names only `ae` and `dd`. | Passed |
| A1 interpreter, package versions, source paths, date, config/plan/code identity | `storage/corpora/corpus-002/environment_log_v4.json`; exact working modules and config are snapshotted under `audits/implementation_source/`. A1 records the absent Al-Emadi ZIP as `exists: false`, hash null. | Passed |
| A2 Al-Emadi inventory and expected formats/subfamilies | `audits/A2_ae_inventory.csv`; gate record reports 11,704 files, 10,372 unknown, 1,332 yes_drone, subfamilies 10,000/301/71 and 16 kHz mono PCM16. | Passed |
| A3 DD class counts and measured audio metadata | `audits/A3_dd_inventory.csv`; 30 each BACKGROUND, DRONE, HELICOPTER; metadata is recorded per file. | Passed |
| A4 read local license/source documents and preserve clauses/attribution | `source_licences_v4.md`, including Al-Emadi D031 clauses, DD CC0/README attribution, and hashes of both source README/LICENSE pairs. | Passed |
| B1 inherited Al-Emadi labels, rejects, zero representatives, groups and flags | `audits/B1_ae_cleaning.csv`; gates confirm 2 prescribed short rejects, 1,089 zeros / 10 contents / 9 representatives, 203 positive groups, 1,528 ESC groups, and 10,623 retained (9,291 negative, 1,332 positive). | Passed |
| B2 DD labeling, HQ 16 kHz conversion, non-overlapping 1-second segments, weak-positive flag and source-file groups | `audits/C_dd_segmentation.csv`, `audits/C_dd_written_clips.csv`, and `audits/B1_ae_cleaning.csv`; 900 output clips were written from 90 recordings, with 48 trailing samples discarded in total and 25 weak-positive DRONE clips retained. | Passed |
| C2/C4 float32 clip write/readback, length, finite samples, no write clipping | `audits/gates.json` reports all 900 readbacks exact; `audits/C_dd_written_clips.csv` records per-output validation. | Passed |
| D float32-sample SHA-256 duplicate analysis, cross-class stop, silence handling | `audits/D_exact_duplicates.csv` and `audits/D_exclusions.csv`; no cross-class or cross-source duplicate, 10 non-zero exact duplicates removed, 1,079 prescribed AE zero duplicates removed, and only the 9 required AE zero representatives retained. | Passed |
| E pre-split descriptors, source-only audit and shortcut warnings | `audits/E_source_class_summary.csv`, `audits/E_source_only_baseline.csv`, `audits/E_shortcut_warnings.csv`; pooled-majority source baseline balanced accuracy is 0.5 (gate ≤0.75). Both sources have rolloff-difference warnings above 1 kHz; neither has a level-difference warning above 6 dB. | Passed |
| Raw sources read-only | `audits/raw_read_only.json` records 11,794 included files unchanged. Hash verification occurred before split assignment. No write was directed under source roots. | Passed |
| F pinned rows, group-aware split, all split gates and frozen manifests | `audits/F_class_fractions.csv`, `F_split_counts.csv`, `F_stratum_groups.csv`, `audits/gates.json`, `storage/splits/split-002/FROZEN.json`; 68 clips pinned; G-GRP/G-DUP/G-FRAC/G-POS/G-STR all passed. | Passed |
| F frontend smoke on training rows only | `audits/F_training_frontend_smoke.csv`; 20 train clips per source, shape `(1,64,28)`, finite. Identity placeholders exercised the canonical input interface; no fitted normalization statistics were computed or saved. | Passed |
| Do not open test audio after freeze | `audits/status.json` records `test_audio_opened_after_freeze: false`; the preparation process ended after its train-only smoke check. This audit rechecked manifest bytes and metadata only, and did not open test audio. | Passed |
| Do not produce later-stage artifacts under this plan | Status evidence records no model loaded/trained, no feature cache, fitted normalization statistics, threshold, bundle, or runtime activation. | Passed |

## Frozen output integrity

The four manifest hashes below were independently recomputed from file bytes after preparation. Split CSV row counts were read as metadata; audio paths were not followed.

| Manifest | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `storage/corpora/corpus-002/manifest_v4_full.csv` | 12,604 | 10,591,642 | `2407e2a3f31beebd86134b4423f14a4a3b17b65d64ce9bfaeb01156a064d4a39` |
| `storage/splits/split-002/train.csv` | 8,246 | 6,940,345 | `6b895c0d6ce420aa7452bed72c06f0db14eb472490ab82bd99de91276ef24d81` |
| `storage/splits/split-002/val.csv` | 1,631 | 1,369,700 | `6fc5134f8faa3875e088d7f84fdc3d03debb2de0466cb73c5624410e754430e9` |
| `storage/splits/split-002/test.csv` | 1,636 | 1,375,433 | `11606d8fb9d597d579b53c996980422d2cb8ecec7dc2df9eca776dd775da813c` |

Manifest rows total 12,604 because the full ledger includes all 11,704 Al-Emadi inventory entries (including rejected rows) plus 900 generated DD segments. The 11,513 rows assigned to splits equal 10,623 retained Al-Emadi files plus 890 DD segments after the 10 exact-duplicate removals.

## Approved implementation definitions and preserved failed attempt

The owner approved (a) the pooled pre-split source-majority population for G-SRC and (b) the weak-positive calculation: absolute band-level difference ≤3 dB from the BACKGROUND median, using SciPy's one-sided periodogram, boxcar window, no detrending, 100–4,000 Hz inclusive, and amplitude-1 reference. These definitions are recorded in `docs/v4/decisions/V4-IMPL-001.md`.

An earlier attempt stopped on a Python column-access implementation error before any statistical gate failed. Its output and STOP report remain under `runs/study-v4/preparation-attempts/001-column-access-error/`; the corrected execution passed and produced the frozen outputs listed above.

## Scope boundary

This audit closes only the data-preparation plan. It does not claim V4 model training, model performance, generalization to unseen datasets, threshold selection, cache generation, or runtime readiness. Al-Emadi test-use history remains a limitation recorded in the report and frozen split metadata.

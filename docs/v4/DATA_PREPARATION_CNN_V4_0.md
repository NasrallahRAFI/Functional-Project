# CNN V4.0 — Data preparation plan (from raw)

**Owner revision:** Salford excluded; only Al-Emadi and Drone-detection-dataset are included in this plan. The plan identifier remains V4.0.

Date: 2026-11-26. Audience: Luna (coding worker, decides nothing) and the owner.
Read with: `PROJECT_CONTEXT_V8.md` (sections 3, 6, 6b; D023, D024, D028, D039, D040, D041, D043, D047) and the frontend correction plan V1.0. Where this plan and those files disagree on a data-preparation rule for this version, this plan wins.

## 0. How to read this plan

- Every value is fixed here. Luna implements it as written and chooses nothing.
- **[ASSUMPTION]** marks a value chosen by the plan's author, not measured. Luna still implements it; the owner may change it before the build starts.
- **STOP** means: write the finding to the report, do not repair or improvise, and wait for the owner. A STOP is triggered by any failed gate and by any measured fact that contradicts an "Expected" value.
- Raw folders are read-only. Nothing is ever written under a raw folder.
- Reuse existing code in `src/dronedet/` (loader, resampler, flag functions in `data_preparation.py`, `splits.py`). New logic goes into existing modules or existing scripts. No new package, no new test tree, no new notebook folder.
- The test split is never scored, tuned on, or inspected beyond counts. Audit statistics in phase F are computed once on the pooled data from the two included sources before the split exists, from audio descriptors and the included sources’ existing labels only (no model).
- Every non-trivial choice has a plan-decision ID `P4-nn` (section 12) so the owner can paste it into the decision log.

## 1. Goal and scope

Build one pooled, labelled, group-aware corpus from exactly two included downloaded sources (Al-Emadi and Drone-detection-dataset), then freeze train / validation / test manifests. All positives are mixed into one class, all negatives into one class, then split.

Produced by this plan: standardised clips, one full manifest, three split manifests, audits, a report.
Not produced here: mel caches, normalisation statistics, models, thresholds, the model bundle. The frontend is unchanged (16 kHz, FFT 1024, hop 256, 64 HTK area mels, fmax 7500, `L` = 0.5 s, `win_samples` 7,936, tensor `(1, 64, 28)`). Section 11 lists the steps that follow.

## 2. Sources and expected raw facts

| Id | Source | Role | Expected raw facts (from the dataset comparison) |
|---|---|---|---|
| ae | Al-Emadi DroneAudioDataset, `binary` folder | positives (`yes_drone`), negatives (`unknown`) | 11,704 WAV; `yes_drone` 1,332 (bebop 666, membo 666); `unknown` 10,372 (10,000 ESC-style names, 301 `silence###`, 71 other noise); 16,000 Hz, mono, PCM16; about 384 MB |
| dd | Drone-detection-dataset, audio only | positives (DRONE), negatives (HELICOPTER, BACKGROUND) | 90 WAV: DRONE 30, HELICOPTER 30, BACKGROUND 30; sample rate, channels, bit depth and duration not stated by its README: measure. Videos and `.mat` labels are not used |


**Explicit exclusion:** The University of Salford DroneNoise Database is not part of CNN V4.0. It has no reliable segment-level drone/non-drone labels in the supplied dataset structure. The proposed energy/activity gate would infer source identity from acoustic level and could create incorrect labels (for example, loud non-drone events as positives or quiet drone flyovers as negatives). Do not inventory it, process it, label it, include it in audits, or include it in any manifest or split. Reconsider only if independently verified event annotations become available and the owner approves a new plan version.

Local path known from the comparison: Al-Emadi `C:\Users\Nasrallah\Downloads\DroneAudioDataset-master\DroneAudioDataset-master\`. The Drone-detection-dataset folder path is recorded by Luna in the environment log (step A1).

## 3. Fixed constants

- Target: 16,000 Hz, mono (channel 0), float32 in [-1, 1]. No level normalisation of any kind.
- Resampling: `soxr`, quality `HQ`, through the project's canonical loader or resampler (the one `features.py` uses). If it does not expose a waveform-level call, call `soxr.resample(..., quality="HQ")` directly and record that fact.
- New clips are 16,000 samples (1.0 s), non-overlapping. The frontend crops its 0.5 s window from them.
- `win_samples` = 7,936. Rejection threshold = `win_samples / 2` = 3,968 samples (D024).
- Flags (use the project's existing definitions and functions; if a flag function is missing, STOP): `silent` (all samples zero), `near_silent` (clip RMS below -80 dBFS), `clipped` (more than 1 % of samples at |x| >= 0.999), `dc_offset`, `crop_all_zero`, and `stats_eligible` = not (`silent` or `near_silent` or `crop_all_zero`).
- Seed: the split seed key already in `config.yaml`. If no such key exists, STOP.

## 4. Output layout

```
data/interim/v4/
  clips/                       Drone-detection-dataset clips only (float32 WAV, 16 kHz, 16,000 samples)
  manifest_v4_full.csv         every clip, every column
  audits/                      CSVs and PNGs of phases A to F
  environment_log_v4.json      D043 record
  source_licences_v4.md
data/processed/splits/v4/
  train.csv  val.csv  test.csv
reports/DATA_PREPARATION_CNN_V4_0_REPORT.md
```

Al-Emadi clips are not copied: the manifest points to the raw file. Existing columns of the project's manifests (`file`, `label`, `split`, `group` and any others `splits.py` reads) keep their names and meaning; new columns are appended after them.

## 5. Phase A — Inventory and verification

- **A1** Environment log (D043): Python and library versions (`librosa`, `soundfile`, `soxr`, `scikit-learn`, `numpy`, `scipy`, `pandas`), the two included source paths, SHA-256 of the Al-Emadi zip if it still exists, the date.
- **A2** Al-Emadi: count files per class and sub-family, and read format with `soundfile`. Expected as in section 2. Any difference: STOP.
- **A3** Drone-detection-dataset: count 30 / 30 / 30. For every file record sample rate, channels, subtype, duration (CSV in `audits/`). Gate: counts match. Durations and rates are recorded, not gated.
- **A4** Licences: read the README, LICENSE or record page for each included dataset (Al-Emadi and Drone-detection-dataset) and write `source_licences_v4.md` (licence, attribution text, redistribution clause). Do not assume a licence that was not read. For Al-Emadi copy the clauses of D031. Do not add Salford to this included-source licence inventory; its exclusion is documented in section 2.

## 6. Phase B — Per-source labelling and grouping rules

### B1 Al-Emadi (rules carried over unchanged from V8)

1. Label from the folder: `yes_drone` = 1, `unknown` = 0. Sub-family from the name (bebop, membo; ESC-style; `silence###`; other noise).
2. Reject clips shorter than 3,968 samples. Expected: exactly 2 (`silence300.wav`, `doing_the_dishes019.wav`). Log them.
3. Exact-zero clips (D039): find all clips whose samples are all zero. Expected: 1,089 in 10 distinct contents. Keep one representative per distinct content, except the content of `silence300` (already rejected). Representative = first file name in sorted order, preferring `silence###` names. Expected: 9 representatives. Drop the other exact-zero clips and log each dropped file with its representative. All representatives are pinned to train, and so are their groups.
4. Group keys (expected counts in brackets):
   - drone clips: file name up to the last `_###_`; the 67 `B_S2_D1` recordings (numbers 67 to 133) are merged into 13 blocks of consecutive recordings, keys `B_S2_D1_blk067` ... `_blk127` (blocks 67-71, 72-76, ..., 122-126 and 127-133); [203 drone groups in all];
   - `extra_membo_D2_####` (56 clips): one group, pinned to train, reported as not evaluated (D040);
   - ESC-style clips: first two fields of the name, `fold-freesoundID`, e.g. `1-100032` [1,528 groups];
   - `silence###`: one group per file; other noise: one group per stem plus one group for `1-7973-A-7_1002`.
5. Flag every clip. The `near_silent` clips stay in the manifest, flagged, with their folder label (D041).
6. Expected result: 10,623 clips = 1,332 positives + 9,291 negatives. Any other number: STOP.

### B2 Drone-detection-dataset audio

1. Label from the file-name class: DRONE = 1; HELICOPTER = 0; BACKGROUND = 0. Sub-class column keeps the three names (helicopter is a deliberate hard negative).
2. Resample each file to 16 kHz (channel 0). Cut non-overlapping 1.0 s segments from the start; drop the remainder shorter than 1.0 s.
3. No activity gate (the dataset gives no ambient reference). Label noise at long range is accepted and reported: flag `weak_positive` when a DRONE segment's band level (100 to 4,000 Hz, RMS in dB) is within 3 dB of the median band level of all BACKGROUND segments. Flagged segments are kept. [ASSUMPTION]
4. Group = one source file (`dd:<file stem>`). The segments of one file never cross splits.

### B3 Salford DroneNoise — excluded

No Salford labelling, segmentation, grouping, or activity-gate code is to be implemented for V4.0. The dataset lacks reliable segment-level class labels, and the proposed energy-based gate cannot establish that a detected acoustic event is a drone. This source is excluded as a data-quality decision, not treated as an empty or failed source. See the exclusion rationale in section 2 and decision P4-03 in section 12.

## 7. Phase C — Standardise and write clips

- **C1** Al-Emadi: no copy, no resample (already 16 kHz mono PCM16). The manifest path is the raw path.
- **C2** dd: resample and segment as in B2, then write each segment as 32-bit float WAV, 16 kHz, mono, 16,000 samples, to `clips/dd__<recording>__<start_ms:07d>.wav`. Do not dither. Do not change the level.
- **C3** Record per clip: source, class, sub-class, recording id, group id, start sample in the recording, native sample rate, `n_samples`, RMS dBFS, 99 %-power rolloff frequency, all flags, `weak_positive`, `pinned`.
- **C4** Check every written clip: reads back, 16,000 samples, finite, not clipped by writing. Failure: STOP.

## 8. Phase D — Cleaning and duplicates

1. Hash every clip (SHA-256 of the float32 samples at 16 kHz; for Al-Emadi decode the PCM16 file to float32 first). Report exact duplicates within and across sources. Within Al-Emadi they are handled by B1.3. Any other duplicate group: keep the first (sorted by clip id), drop the rest, log them. If a duplicate group spans both classes: STOP.
2. Clips with `silent` outside the nine Al-Emadi representatives: drop and log (this can only happen in the included Drone-detection-dataset source).
3. Nothing else is deleted. Flags are reported, not acted on (D041 principle).

## 9. Phase E — Shortcut audit (before the split, no model)

Write these tables to `audits/`:

1. Per included source x class: clips, seconds, groups, median and inter-quartile RMS dBFS, median and inter-quartile 99 %-power rolloff, fraction `clipped`, `dc_offset`, `silent`, `near_silent`, `weak_positive`.
2. Source-only baseline: predict the class of each clip from its source id alone (the training-majority class of that source), report balanced accuracy on the pooled data from the two included sources. **Gate G-SRC: balanced accuracy <= 0.75** [ASSUMPTION]. Above: STOP.
3. Flag, in the report, every included source whose class-conditional median level differs by more than 6 dB or whose rolloff differs by more than 1 kHz between its two classes. These are warnings for the owner, not stops.

## 10. Phase F — Pool and split

1. Pool all surviving clips. Class 1 = all positives, class 0 = all negatives. Stratum label = `<class>|<source>`.
2. Pinned to train: the nine Al-Emadi representatives with their groups, and `extra_membo_D2`. Remove them from the splitting step; they stay in train.
3. Two-stage `StratifiedGroupKFold` on the remaining clips, `groups` = group id, `y` = stratum label, fixed seed (section 3):
   - stage 1: 7 folds; fold 0 = test;
   - stage 2: 6 folds on the rest; fold 0 = validation; the remaining folds = train.
   About 71 / 14.3 / 14.3 by clips.
4. Gates:
   - **G-GRP** no group id appears in more than one split;
   - **G-DUP** no SHA-256 appears in more than one split;
   - **G-FRAC** validation and test each hold 12 % to 17 % of the clips of each class;
   - **G-POS** validation and test each hold at least 25 positive groups;
   - **G-STR** every stratum with at least 6 groups has at least one group in validation and one in test.
   Any gate fails: STOP; do not change the seed or fold counts.
5. Write `train.csv`, `val.csv`, `test.csv` plus `manifest_v4_full.csv`. Record SHA-256 of each manifest in the report. Test audio is not opened after this point.
6. Frontend smoke check (training split only): load 20 random training clips from each included source (Al-Emadi and Drone-detection-dataset) through the canonical frontend path and confirm each yields a window of shape `(1, 64, 28)` and finite values. Failure: STOP.

## 11. After this plan (not Luna's decisions, not gated here)

Mel-power caches through the validated writers with sidecars bound to the new manifest digests; statistics from `stats_eligible` training rows (D047); CNN retraining with the D049 recipe and three seeds; threshold re-chosen on the new validation set (D044); bundle, goldens and the live-detector runtime manifest regenerated; evaluation with per-source breakdown and group bootstrap (D051). Results from the earlier split are not comparable with this version's test set.

## 12. Plan decisions

| Id | Decision | Why | Alternative |
|---|---|---|---|
| P4-01 | Pool all positives and all negatives, stratify the split by class and source | the owner's pooling goal; each source appears in every split | split each source separately |
| P4-02 | Al-Emadi rules of V8 reused unchanged | already approved (D023, D024, D039, D040, D041) | new rules |
| P4-03 | Exclude Salford from V4.0 | no reliable segment-level labels are supplied; energy-based activity detection cannot establish drone identity and risks label noise | include only after independent event-level annotation and owner approval |
| P4-04 | Helicopter and background audio are negatives | hard negatives with rotor-like spectra | drop helicopter |
| P4-05 | 1.0 s non-overlapping segments, float32 WAV, no dither, no normalisation | gives the frontend a cached clip to crop; keeps natural level | 0.5 s clips; PCM16 |
| P4-06 | Group = source file (dd), V8 keys (ae) | prevents leakage through related recordings | clip-level split |
| P4-08 | Source-only baseline bar of 0.75 balanced accuracy | a source that predicts the class makes shortcuts likely | no bar |

## 13. Known limitations (state them in the final report)

- Salford is excluded from V4.0 because the available files do not provide reliable segment-level drone/non-drone labels. Its energy-based labelling proposal was rejected to avoid introducing unverified ground truth.
- The Drone-detection-dataset contributes only 30 original recordings per class; its sample rate, channels, bit depth and duration are measured during inventory rather than taken from its README.
- Al-Emadi remains indoor and its positive recordings cover only Bebop and Membo drones.

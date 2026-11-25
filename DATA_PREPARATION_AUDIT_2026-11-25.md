# Data preparation implementation and readiness audit

Assessment date: 25 November 2026. Code inspected at commit `f09f4f8f0288c4950d326025d817037c233c32bb`, with the existing working tree preserved. Scope: the raw dataset, notebooks 00–02, current audio frontend, frozen manifests, prepared arrays, and their handover to model training. This is an assessment; no preparation logic, frozen data, model weights, or historical reports were corrected during it.

**Overall finding: the installed prepared data is usable and numerically well supported, but the current preparation workflow cannot be certified as fully reproducible from raw files.** Cleaning, grouping, split integrity, feature extraction, and train-only normalization are substantially implemented. Fresh checks reproduced cached features and normalization exactly. A confirmed missing import breaks raw inventory measurements for nonzero audio. The present notebook 02 is an isolated train/validation verification workflow, rather than a complete persistent three-split build workflow. Several older reports also describe states that have since changed.

**Assessment basis and strength of evidence**

The review used `PROJECT_CONTEXT_V8.md`; all four documents in the companion Data folder; the active frontend `correction plan V1.0.md` and its archived current contract, decisions, amendments, and relevant evidence; both model specifications and the dated model status reports; and the actual repository, configuration, manifests, arrays, and executable helpers. Later explicit decisions supersede earlier proposals: frame-aligned centering, the stored statistics eligibility mask, per-band floor masking, local CPU execution, and separate identity domains are the current contracts.

Fresh verification covered the following:

| Check performed for this audit | Result |
|---|---|
| Unit suite, `python -X utf8 -m pytest tests/unit -q` | 67 passed, 18 warnings, 22 subtests passed; 43.20 seconds |
| Four frozen manifest SHA-256 checks | All matched the recorded freeze |
| Structural split checks recomputed from the full manifest | All eight structural checks passed; the ninth recorded gate concerns freeze/hash persistence |
| In-memory cleaning and seed-42 split replay from the inventory | Kept-clip assignments matched the frozen manifest exactly |
| Current train and validation tabular/power/index loaders | Passed their provenance, checksum, ordering, shape, and content checks |
| Header inspection of every training and validation WAV | 9,117 headers; all 16 kHz mono PCM16, matching recorded sample counts; no retained clip too short |
| Seeded raw-audio recomputation | 20 train and 20 validation clips, seed 20261125; maximum mel-power error 0.0 and maximum tabular-feature error 0.0 |
| Approved normalization refit over the training cache | Mean and standard deviation bit-identical to installed float32 statistics; 7,519 eligible, 88 excluded |
| Swept passband measurement | 7,540.278697934931 Hz; selected `fmax=7500` agrees with configuration |
| Native 16 kHz and 48 kHz synthetic golden WAV recomputation | Both full log-mel outputs matched exactly, maximum error 0.0 |
| Existing V1 and migration V2 frontend bundle verification | Both accepted within their declared scopes; V2 native-rate normalized tensors matched exactly |
| Default test-access guards | Tabular, power, index, and manifest loaders all rejected test access without explicit permission |
| Raw inventory function probe | Failed on a nonzero training WAV with `NameError: name 'math' is not defined`; an exact-zero training WAV passed |

Real test audio and real test feature-array payloads were not opened or scored in this audit. Test-related observations use the frozen manifest/index metadata and existing preparation/evaluation records. Dataset-wide descriptive measurements below are derived from the saved inventory and file sizes, rather than a new decode of held-out audio. There was no training, threshold selection, or repeat final evaluation.

The fresh normalization/passband evidence is saved locally as `data/interim/data_preparation_audit_2026-11-25_reproduction.json`. Its protected-artifact verification passed for 143 protected paths, with only the previously authorized configuration metadata migration distinguished from the original baseline.

**What the plan requires and what exists**

| Planned stage | Current implementation and result | Assessment |
|---|---|---|
| Read-only source dataset and auditable provenance | Raw WAVs remain external; preparation writes manifests/caches separately; archive hash and environment records exist | Implemented; download time and exact source commit remain qualified |
| Notebook 00 inventory, format, duplicate, artifact, and length audit | An 11,704-row inventory and 25 passing historical audit rows exist | Installed evidence complete; current nonzero-audio inventory function is broken |
| Notebook 01 minimal cleaning | Two short clips removed, 1,079 redundant exact-zero clips excluded; all other anomalies flagged | Matches approved plan and replayed correctly |
| Leakage-aware groups and pinned groups | Recording groups, ESC source groups, five noise stems, 13 B_S2_D1 blocks, and train-pinned groups are represented | Implemented; current group/PCM separation checks pass |
| Frozen two-stage split with seed 42 | Three manifests, hash records, and `FROZEN.txt`; assignments reproduce exactly | Complete for the installed dataset |
| One configurable library-based audio frontend | SoundFile, soxr, librosa; shared canonical helpers in `src/dronedet/` | Implemented; compatibility code remains documented |
| 84 acoustic tabular features per retained clip | Three NPZ archives, correct feature catalog and finite train/validation arrays | Complete installed artifacts; historical gates cover test finiteness |
| Full-clip mel-power cache and index per split | Three float32 NPY arrays and ordered CSV indices | Complete installed artifacts; train/validation independently sampled and exact |
| Train-only normalization and parity assets | Eligible-train fit, per-band floor mask, saved mean/std, goldens and bundles | Strong current implementation; exact fresh reproduction |
| Notebook 02 as a reusable build/handover | Current notebook builds train/validation in a temporary output tree and verifies them | Useful verification path; persistent cold-start preparation remains incomplete |
| Model-facing loaders, scaling, and fixed evaluation crops | Validating loaders, guarded test access, eligible-train scaler, training-only random crops | Implemented in local model runners |
| Per-family/flag reporting and group-aware uncertainty | Required metadata exists and downstream CSV/evaluation records are present | Handover supported; the dataset's scientific limits remain |

Historical gate totals are N00 25/25, N01 9/9, and N02 25/25. These totals describe saved executions and do not override a newly reproduced failure in today's raw-audit function.

**Raw file format and dataset composition**

The source archive is `DroneAudioDataset-master.zip`, 580,430,228 bytes, with recorded SHA-256 `90d09a04ce08bd08f0e76d92d0aebd394346317460dad4bdc8344c2ea9683f37`. The configured dataset locator handles the nested extracted-directory layout. Training uses only `Binary_Drone_Audio`; the multiclass tree contains copies and must not be added as extra training data.

| Raw binary corpus property | Value |
|---|---|
| Total WAV files | 11,704 |
| Negative folder and label | `unknown`, label 0; 10,372 clips |
| Positive folder and label | `yes_drone`, label 1; 1,332 clips |
| Positive drone names | 666 Bebop, 666 Membo, inferred from filenames |
| Audio encoding | WAV, signed 16-bit PCM, mono, 16,000 Hz |
| Actual binary-folder file bytes | 384,021,184 bytes, approximately 384.0 MB decimal |
| Aggregate duration from the inventory | 11,971.512875 seconds, approximately 3.3254 hours |
| Positive aggregate duration | 1,332.81825 seconds, approximately 22.21 minutes |
| Negative aggregate duration | 10,638.694625 seconds, approximately 177.31 minutes |
| Raw negative-to-positive clip ratio | Approximately 7.79:1 |
| Original recorded decode errors/truncation | Zero; confirmed headers across train/validation remain consistent |

The 10,372 negative clips comprise 10,000 ESC-style names, 301 `silence###` files, and 71 other noise files. The latter are 70 files from five stems plus `1-7973-A-7_1002.wav`. These are naming/source categories; they do not establish perceived audio content.

Lengths vary systematically by source:

| Source | Important lengths |
|---|---|
| Drone clips | 1,038 clips have 16,384 samples, or 1.024 seconds; minimum 10,399 samples; many others have 14,464 or 14,336 samples |
| ESC-style negatives | 6,000 have 16,347 samples; 2,000 have 15,604; 2,000 have 15,356 |
| Named silence | 24–17,833 samples |
| Other noise | Includes the 1,266-sample rejected file and 67 clips around five seconds; maximum 81,920 samples |

WAV headers have different sizes, including previously inspected 78-byte and 108-byte examples. Length must come from decoding/header frame counts, not a fixed subtraction from file size. File sizes, headers, durations, filenames, labels, and group identifiers are metadata and are not model feature columns.

SoundFile float32 decoding corresponds to PCM16 divided by 32768. Labels come from the binary folder, not acoustic inference. The exceptional `2-102581-A-291.wav` is readable: the original exploration error occurred in descriptor calculation. It is retained and flagged as near-silent/partial-zero.

The synthetic golden assets are a separate format: the current native 16/48 kHz WAV fixtures are mono FLOAT WAVs of 0.9 seconds, not raw PCM16 dataset clips. Their full log-mel arrays have shape `(64,53)`. V2 bundle checks subsequently select and normalize the model-sized `(1,64,28)` window. This differs from the original illustrative plan's two-second PCM16 golden-generation snippet, while still exercising the intended native-rate/resampling parity paths.

**Cleaning implementation and the meaning of the manifests**

The pipeline performs conservative selection and flagging; it does not rewrite cleaned audio, repair clipping, remove DC, denoise, or alter labels. That matches the approved minimal-cleaning plan.

| Action | Clips |
|---|---:|
| Original inventory | 11,704 |
| Exclude shorter than half the usable window | 2 |
| Exclude additional exact-zero duplicate copies | 1,079 |
| Total exclusions | 1,081 |
| Retained model examples | 10,623 |
| Retained negatives | 9,291 |
| Retained positives | 1,332 |

The rejected short files are `unknown/silence300.wav` (24 samples) and `unknown/doing_the_dishes019.wav` (1,266 samples). Applying this rule before zero deduplication avoids counting `silence300.wav` twice.

The 1,089 exact-zero raw clips represent ten distinct decoded contents, where length is part of content identity. After the 24-sample case is rejected, nine representatives remain. Selection is deterministic: sorted clip identifiers, preferring a named silence file when present. Their eight whole groups are pinned to train. The additional `extra_membo_D2` group brings the total to nine pinned groups and 68 retained pinned clips. Validation and test contain no completely zero full clips.

An important schema clarification: **`manifest_full.csv` is an 11,704-row audit ledger, not a 10,623-row filtered training table.** It contains 10,623 rows with `kept=True` and 1,081 exclusions. The three split CSVs contain retained examples only. `manifest_with_split.csv` also preserves all 11,704 source rows and additional inventory fields. Counting all rows of the full manifest as examples would therefore be incorrect. The older implementation report's phrase “10,623 retained rows” needs this qualification.

Flags are retained for reporting and normalization eligibility:

| Flag | Definition | Train | Validation | Test |
|---|---|---:|---:|---:|
| `silent` | Whole clip exactly zero | 9 | 0 | 0 |
| `near_silent` | Nonzero whole clip, RMS below −80 dBFS | 41 | 8 | 10 |
| `partial_zero` | Nonzero whole clip, more than half its samples zero | 173 | 36 | 37 |
| `clipped_flag` | More than 1% of samples have absolute amplitude at least 0.999 | 163 | 22 | 21 |
| `dc_flag` | Absolute sample mean above 0.01 | 95 | 12 | 9 |
| `crop_all_zero` | Selected centered window has zero power throughout | 69 | 8 | 19 |

The first five are represented in manifests; the crop-specific flag belongs in cache indices and NPZ archives because it is established at feature extraction. Flags overlap and must not be added as disjoint counts. A nonzero clip can still have a completely zero centered crop, explaining why evaluation indices contain crop-zero cases despite excluding whole-clip digital silence.

Both near-silent drone clips remain label 1 and are in validation, as approved. Their identities are `mixed_membo_17-membo_003_.wav` and `mixed_membo_17-membo_004_.wav`. Their existence makes per-flag reporting necessary; it does not justify relabeling them after model results.

**Grouping, leakage control, and the frozen split**

The split code implements the specified two-stage `StratifiedGroupKFold`: seven folds with fold 0 reserved for test, then six folds on the remainder with fold 0 for validation, seed 42. Pinned groups are removed from the free split pool and added to train.

Groups use recording names for ordinary drone recordings; 13 consecutive-recording blocks for B_S2_D1 recordings 67–133; one group for all 56 extra Membo clips; the first two filename fields for ESC sources; one group per other-noise stem; and individual silence groups before deduplication. The final B_S2_D1 block covers 127–133 and contains 31 clips; the first twelve blocks contain 25 each.

| Split | Clips | Negative | Positive | All groups | Positive groups |
|---|---:|---:|---:|---:|---:|
| Train | 7,607 | 6,640 | 967 | 1,249 | 146 |
| Validation | 1,510 | 1,325 | 185 | 247 | 29 |
| Test | 1,506 | 1,326 | 180 | 247 | 28 |
| Total | 10,623 | 9,291 | 1,332 | 1,743 | 203 |

No retained group or PCM identity crosses splits. Both classes are present in each split. All blocks remain intact, all pinned groups are train-only, and all 56 extra Membo clips are in train. Replaying the implemented cleaning and split logic from the saved inventory reproduces the frozen assignments exactly, without rewriting the manifests.

| Positive family | Train clips | Validation clips | Test clips |
|---|---:|---:|---:|
| B_S2_D1 | 231 | 50 | 50 |
| mixed_bebop | 235 | 55 | 45 |
| mixed_membo | 240 | 40 | 55 |
| Membo_0 | 90 | 20 | 15 |
| Membo_1 | 70 | 15 | 5 |
| Membo_2 | 45 | 5 | 10 |
| extra_membo_D2 | 56 | 0 | 0 |

Family coverage is uneven: validation Membo_2 has one independent group; test Membo_1 also has one. Extra Membo has no evaluation coverage at all. Overall clip counts must not be mistaken for independent sample counts, and confidence intervals should bootstrap groups.

The four freshly checked SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| Full manifest | `bd2d9059ceda181b0877d9fb371f1d00760d057e2546da79873187832a625142` |
| Train manifest | `60b14ff205bdd4e950ae0642d5ae886c8b8c4c291cabfedc26ee936d9a3e1afc` |
| Validation manifest | `c1c0170fddbd6dfa4d3095eab12abef6572ac45036278ba79cc4c71794dc23fc` |
| Test manifest | `350356a0b89a84bf6cf77fc4a1ffa36dea662151fbd5d79e66989cdfeec14848` |

**Implemented transformation from a WAV to model input**

The operational chain is:

```text
Binary WAV and frozen manifest row
    -> SoundFile float32, configured channel 0
    -> soxr HQ to 16 kHz only if source rate differs
    -> reject/wrap-pad only where the approved length rules require it
    -> full-clip librosa mel power, stored in a split NPY cache
         -> centered 28-frame window -> 84 descriptors -> model's train-fitted scaler
         -> 28-frame window -> training-only gain where applicable
                            -> dB -> training per-band normalization -> CNN tensor
```

The active settings are 16,000 Hz, FFT/window 1,024, hop 256, periodic Hann, `center=False`, power 2, 64 HTK mel filters with area normalization, and 50–7,500 Hz. dB conversion uses reference 1, `amin=1e-12`, and `top_db=None`.

The FFT frame is 64 ms long, the hop is 16 ms, and bin spacing is 15.625 Hz. Nominal `L=0.5` gives:

```text
n_win       = 1 + floor((0.5 * 16000 - 1024) / 256) = 28
win_samples = 1024 + (28 - 1) * 256                 = 7936
usable time = 7936 / 16000                         = 0.496 seconds
```

Retained dataset clips need neither resampling nor padding. General helpers support other rates and wrap padding for accepted short inputs. Shorter than half-window inputs are rejected; the below-FFT guard also rejects unusable input.

Fixed centering uses `(n_frames - 28) // 2`, then converts frame position to samples through the hop. This is an approved change from the data recipe's sample-level centering expression. For a 16,384-sample clip, the chosen crop starts at sample 4,096 rather than 4,224. Frame alignment allows the cached crop and direct waveform transform to agree.

The mel cache stores full clips to support random training offsets. It is not a pre-normalized image dataset. CNN training chooses a new random valid window per clip per epoch and applies gain in power with `10**(g/10)`, with `g` sampled from −12 to +12 dB. Tabular features, validation, and test use the fixed center. Evaluation calls now reject an RNG. These training differences affect comparisons between MLP and CNN; the clean fixed-window/no-gain CNN control is the relevant companion comparison.

**Final files ready to feed models**

Paths in this section are relative to the Functional Project repository.

| Purpose | Files | Content |
|---|---|---|
| Frozen examples and metadata | `data/processed/splits/train.csv`, `val.csv`, `test.csv` | One row per retained clip; identifiers, labels, groups, family/type, quality flags, split |
| Classical/MLP feature inputs | `data/processed/features_tabular/train.npz`, `val.npz`, `test.npz` | Float32 `X`, integer labels and aligned identifiers/flags |
| CNN reusable acoustic representation | `data/processed/spectrograms/mel_power_<split>.npy` | Full-clip float32 power, concatenated along time |
| Power-to-clip mapping | `data/processed/spectrograms/index_<split>.csv` | `clip_id`, global start frame, clip frame count, centered offset, flags, eligibility, length status |
| CNN normalization | `norm_stats_global.npz`, `normalization_stats.npz`, and JSON record in the spectrogram folder | 64 float32 means/stds, eligibility/floor-mask accounting and policy |
| Cache provenance | `*.npz.meta.json`, `*.npy.meta.json`, `frontend_hash.json` | Installed identities, manifest links, shape/dtype, provenance and artifact records |
| Numerical parity | `data/processed/golden/` | Native 16/48 kHz synthetic WAVs, expected log-mels and reference filterbank |

| Split | Tabular X | Full-clip power cache | Power payload bytes |
|---|---|---|---:|
| Train | `(7607,84)` | `(64,457897)` | 117,221,632 |
| Validation | `(1510,84)` | `(64,91761)` | 23,490,816 |
| Test | `(1506,84)` | `(64,91494)` | 23,422,464 |
| Total | 10,623 rows | 641,152 time frames | 164,134,912 |

The total power payload is about 164.1 MB decimal or 156.5 MiB; each NPY also has a small header. Actual installed NPY file lengths are 117,221,760, 23,490,944, and 23,422,592 bytes. NPZ archives are compressed and retain values independently of their compressed file size.

Each tabular NPZ has `X`, `y`, `clip_id`, `group`, `feature_names`, `silent`, `near_silent`, `crop_all_zero`, `partial_zero`, `clipped_flag`, and `dc_flag`. `X` is float32, `y` is int8, identifiers/catalog are string arrays, and flags are boolean. The stored NPZ does not directly contain every returned model field: `load_tabular()` joins family, drone type, recording, and labels from the manifest and obtains `stats_eligible` from the index. Consumers should use that loader rather than assume direct `np.load()` gives the complete handover contract.

The 84 features are:

| Feature component | Aggregation | Columns |
|---|---|---:|
| MFCC coefficients c1–c19 | Mean, population std, min, max across 28 frames | 76 |
| Mel-domain spectral centroid | Mean and population std | 2 |
| Mel-domain spectral bandwidth, p=2 | Mean and population std | 2 |
| Mel-domain spectral rolloff, 85% | Mean and population std | 2 |
| Mel-domain spectral flatness, `power=1.0` | Mean and population std | 2 |
| Total | | 84 |

These spectral descriptors are defined on mel power with mel center frequencies, rather than the usual linear-FFT spectrum. MFCC c0, waveform RMS, zero-crossing rate, clip length, file size, source identifiers, and quality flags are not columns of `X`. Flags and identifiers support selection/reporting, not predictive input.

The NPZ `X` is **not StandardScaler-normalized**. The current local model runner fits the scaler on `train['X'][train['stats_eligible']]`, then transforms all training and validation rows. The excluded 88 training examples stay in model fitting; they are omitted from statistics estimation only. Labels are aligned with the same row order. Classical/MLP input is a batch `(B,84)`, with labels `(B,)`.

For CNN use, a selected power window is `(64,28)`. `features.transform()` with the selected `global_stats` mode returns float32 `(1,64,28)`; batching gives `(B,1,64,28)`. Its values are:

```text
x[k,t] = (10*log10(max(power[k,t],1e-12)) - training_mean[k]) / training_std[k]
```

Passing the full concatenated NPY as if it were a batch of fixed clips, passing raw mel power directly to the trained CNN, or applying a newly fitted validation/test scaler would violate the implemented contract.

**Normalization quality and what changed after data preparation v5**

The current fit uses stored boolean `stats_eligible = not(silent or near_silent or crop_all_zero)`. Train has 9 silent, 41 near-silent, and 69 crop-zero rows; the overlapping union excludes 88, leaving 7,519. It uses full-clip frames from eligible training clips at 0 dB, not only the centered crop and not augmented windows.

Within each eligible clip, individual cells at or below −119.999999 dB are omitted independently in each mel band. Other cells in a partly floored frame remain usable. This is a substantive refinement beyond the original simple concatenated-mean/std recipe, and is explicitly recorded by the later frontend contract.

The fit record counts 452,765 eligible clip frames before band-specific cell masking. Each band retains between 448,045 and 448,139 cells and omits between 4,626 and 4,720 floor cells. Thus “452,765 frames” does not mean every band uses that many values in its moments.

The installed 64-element means range from approximately −39.119 to −15.533 dB and standard deviations from 18.842 to 20.537 dB; all are finite and stds are positive. The fresh approved refit reproduced both arrays exactly. It uses per-clip/per-band moment merging in float64 before producing float32 outputs, reducing large temporary allocations.

The frontend benchmark records 480,059,392 versus 5,718,016 bytes of incremental peak RSS, with runtime 2.952 versus 14.825 seconds on the measured host. This is a memory/runtime tradeoff; the time ratio is approximately 5.02, so the archived narrative's “roughly 2.9×” statement is an arithmetic error. The fresh audit refit took 16.963 seconds. These measurements are tied to this host and cache.

Dropping c0 and using scale-normalized spectral descriptors makes the tabular representation approximately gain-independent away from numerical floors. It does not establish unconditional invariance at arbitrary attenuation. The same qualification applies to area/peak normalization equivalence: floor-clamped values prevent an unrestricted algebraic equivalence claim. The later frontend document correctly narrows these claims.

**Cache integrity, schemas, and bundle readiness**

Current writers implement cache schema 3 and identity schema 2. They validate power values, dimensions, index spans, centered offsets, flags, catalog order, manifest correspondence, and measured-passband status; refuse unintended overwrite; stage the array/index/metadata set; and publish a transaction marker last. Raw destination paths are guarded.

The installed production sidecars are nevertheless **unversioned historical metadata**, intentionally preserved. They use the verified V1 compatibility path plus the original `frontend_hash.json` artifact digests. They are not already schema-3 transactions. Current train/validation loaders accepted them, so this is a supported compatibility state, not evidence that caches need rebuilding. Descriptions of the new writer contract should make this installed-versus-new distinction explicit.

The separate V2 identity domains are appropriate: power settings control power-cache identity; the feature catalog controls tabular identity; mode/statistics/fit policy control normalized-input lineage; decision cadence and smoothing control decision identity. Different hashes from different domains do not by themselves indicate data drift.

| Identity | Current verified value |
|---|---|
| Historical V1 parameter hash | `97544883461de11a1ee51cc17d7028431d6d0ea8f175effc8e63bc18595f28fc` |
| Historical V1 statistics lineage | `327b5e0c54632f3c7e40533d2d0b521faa876cbaf58b8dd8c5bfa57fe9f6b8f0` |
| V2 power content | `026ff2ecaa8a240383af8666f61bbfcd9ea7ef3d4e9a51c73e803d82259a43b0` |
| V2 tabular content | `cf945d56312b83dab20a6ad401f6d7273346aeab9483a19fd9b1c7e6f222ea0b` |
| V2 transform | `bb7a21a1e5fb317c43e1e214bc04af6e46e04d75053383d1c62ef6849c9466c7` |
| V2 normalized-input lineage | `dab196aee59c482e2938d90ea0c55d33eb5606f8aa04d11ff8d441fb6cd31678` |

The installed V1 frontend bundle passes verification, but its array fixtures do not establish native source rates or verify a normalized model window. The newer bundle at `data/interim/frontend_bundle_v2_migration` verifies native 16 kHz and resampled 48 kHz paths and both `(1,64,28)` normalized outputs with zero error. It is still an interim frontend migration bundle. These findings establish frontend numerical readiness; they do not certify a final CNN deployment bundle with weights, threshold, and runtime policy.

**Confirmed defects and completeness gaps**

| Priority | Finding | Evidence and consequence | Recommended closeout |
|---|---|---|---|
| High | Raw inventory fails for nonzero audio | `src/dronedet/data_preparation.py:54` calls `math.log10` without importing `math`. Direct `clip_stats()` fails; `build_inventory()` catches it and stores the NameError. Notebook 00 calls that builder. Zero audio bypasses the expression and succeeds. | Add the missing import and a meaningful nonzero/zero inventory regression check, then verify notebook 00 safely without rewriting frozen artifacts. Existing caches are not invalidated by this defect. |
| Medium | Current notebook 02 is not the original persistent preparation workflow | `BUILD_SPLITS=('train','val')`, `TemporaryDirectory`, and `OUTPUT_PATHS` direct writes to temporary storage. It checks against the installed project and never creates the production test cache. | Document it as verification and supply a distinct guarded cold-start build procedure that persists all required outputs. Keep post-freeze verification separate from test cache creation. |
| Medium | Notebook 02 can finish while its required closeout is pending | Its final command passes no `--hardware-log`. The runner therefore treats physical acceptance as pending. The notebook raises only for a nonzero process with status `FAIL`, permitting `PENDING` to complete without an exception; absent output with another process failure is also insufficiently checked. | Make notebook success depend on explicitly named required statuses; either pass validated hardware evidence or label the invocation as software-only verification. Use a distinct evidence output instead of silently replacing the generic record. |
| Medium | Saved green gates and the unit suite do not cover full preparation reproducibility | All 67 unit tests pass despite the directly reproduced inventory error. The current test modules focus on audio/frontend/cache/hash/bundle contracts. | Cover the actual preparation entry points and an isolated minimal pipeline, especially nonzero inventory input and persistent output expectations. |
| Medium | Model environment is not fully described by the preparation requirements file | Current environment records include `torch 2.15.0+cpu`; model code imports it, but root `requirements.txt` does not pin/install torch. The old “PyTorch absent” statement describes preparation time. | Provide a reproducible model-consumer environment specification alongside the preparation-only requirements. Do not treat the historical preparation log as the current environment. |
| Low | Documentation mixes historical and current state | V5 report still lists resolved passband/provenance/test issues; V8 status boxes remain early-stage; November 23 model status says no final test scores despite November 24 evaluation artifacts. | Add dated supersession/current-state pointers without erasing historical evidence. |
| Low | Schema and data-count descriptions can mislead consumers | Full manifest retains exclusions; installed sidecars are legacy; eligibility/family metadata is enriched by loaders. | Publish the installed file contract with row counts, schemas, loader enrichment, and identity domains distinguished. |

The first defect is an executable failure. The remaining workflow findings concern reproducibility and acceptance semantics; they are not evidence that the installed feature values are wrong. No source repair was included in this assessment.

**Historical issues that are now resolved or superseded**

| Older claim or outstanding item | Current evidence |
|---|---|
| YAML measured passband 7,550 Hz differs from refined measurement | YAML now has 7,540.278697934931; fresh swept measurement and bundle verification agree |
| `environment_log.json` has only stale hash `9733737f…` | It now records `97544883…` and explicitly labels the older value superseded |
| Conflicting bundle hashes `42f639f9…` and `cf0e159b…` | Current records retain them as history and record verified V1 statistics lineage `327b5e0c…`; V2 mapping is explicit |
| Legacy unit failures remain unresolved | Fresh current suite passes 67 tests; warnings are recorded |
| CNN normalization is only a simple eligible-frame mean/std | Current contract adds the per-band floor mask and an exact-reproducing streaming fit |
| No final test evaluation exists | `results/tables/run_report_notebook90_local.json` records PASS and an authorized scoring lock dated 2026-11-24, covering 1,506 clips, 28 fits and 42,168 score rows |
| Fully portable across platforms | Verification covers two Python environments on the same Windows host; cross-OS behavior remains unverified |

The final-test record was inspected as existing evidence. It was not rerun. The dated model report is a historical snapshot, rather than the latest completion state.

**Scientific data quality and limits that preparation cannot remove**

The preparation is considerably stronger than a random clip split: it explicitly handles identity duplication, correlated segments, train-only estimation, and metadata shortcuts. Those controls preserve the meaning of model comparison. They do not make this corpus representative of real-world detection.

The important limits are two indoor drone models; artificially mixed positive recordings; a predominantly ESC-derived negative class; source-dependent clipping, DC, padding and silence; uncertain recording-session meanings; few independent positive evaluation groups; and no evaluation of extra Membo. Similarity between neighboring B_S2_D1 recordings is reduced by blocking, but true session independence is not established.

The two classes have similar median nonzero RMS, so the old claim that simple overall loudness separates them is not supported. Source artifacts can still be predictive. All-zero centered windows are particularly clear negative cues. A headline score should therefore be accompanied by the already-planned per-family, per-type and flag-sensitivity results, with group-bootstrap intervals and small-group qualifications.

The positive class is only about 12.54% of the retained dataset; an always-negative classifier would achieve approximately 87.46% aggregate accuracy while missing every positive. F1, recall, false-alarm and miss rates, threshold selection on validation, and group uncertainty are necessary to interpret performance.

Fixed centers discard substantial audio: approximately half of a typical one-second recording and about 90% of a five-second negative. Whole-clip labels can therefore be imperfect labels for the selected crop. Random CNN windows increase training coverage, but also inherit the whole-clip label, including quiet stretches. Neither behavior establishes event-level annotation quality.

Train statistics weight full-clip frames, so long retained clips contribute more cells than short ones. This matches the approved full-clip statistics contract, but differs from equal weighting of one center window per example and should remain explicit in methodology.

The five-minute physical frontend record establishes capture and recovery under the recorded protocol. Because its scorer was a stub, it does not validate CNN accuracy, CNN latency, or speech/music/room-noise negative performance. Those are separate from preparation correctness.

**Readiness judgment and recommended order of work**

| Question | Judgment |
|---|---|
| Can the installed tabular data feed classical models and the MLP? | Yes, through the current validated loader and eligible-train scaler contract |
| Can the installed power caches feed the frozen CNN frontend? | Yes, after selecting a 28-frame window and applying the stored global statistics |
| Is the split implementation faithful to the approved design? | Yes; hashes, structural checks and exact in-memory replay support it |
| Are existing numerical outputs supported? | Yes; current loader validation, exact sampled feature recomputation, exact full-train statistics reproduction, and golden checks agree |
| Is the current raw-to-ready workflow fully reproducible from a clean setup? | No; inventory has a confirmed failure, and the current notebook is an isolated verification workflow |
| Is portable final deployment completeness established here? | No; V2 frontend parity is verified, while final model packaging and cross-OS acceptance require their own evidence |

The next work should first repair the inventory import and close its coverage gap. Then clarify and verify a persistent cold-start preparation procedure without changing any frozen assignment or trained model. Tighten notebook closeout status handling and record the actual environment needed by model consumers. Finally, reconcile current documentation and prepare the intended final export from the verified V2 frontend contract.

There is no measured reason in this audit to re-split, relabel, retrain, or regenerate the installed caches. The supported result is a usable prepared dataset with strong numerical evidence, plus specific defects in reproducibility and completion reporting that should be corrected before declaring the whole preparation implementation finished.

# DATA_PREPARATION_CNN_V5_1_FINAL

**Prepared:** 27 November 2026

**Status:** final document for owner approval; implementation and preparation remain unexecuted.

**Scope:** source admission, deterministic preparation, frozen membership, sampling weights, augmentation interfaces, and validation-stream evidence for CNN V5.1. Architecture, loss, optimizer, training schedule, V5.1 training, operating-point activation, and held-out evaluation require separate authorization.

**Authority:** this revision supersedes both pasted proposals. It becomes the execution plan only when O0 is recorded. Publication of this document is not O0.

**Convention:** **Fixed** means implement literally; **Default** means use the stated value unless changed before O0; **Open** identifies an owner-controlled action with an inactive default. Missing rules or failed blocking gates require a recorded stop, not a workaround.

The companion `DATA_PREPARATION_CNN_V5_1_CONTRACT.json` supplies the exact artifact column sets, canonical serialization, random-draw namespaces, and gate parameters. Both files form the proposed contract. If they disagree, stop before execution. Current execution gates are `NOT_RUN`; missing implementation is `NOT_IMPLEMENTED`.

## 0. Authorization and study boundaries

O0 authorizes only the preparation phases A0 and P0-P7 described here, including the mechanical-pass relaxation and the revised effective-size policy. O0 does not authorize V5.1 model training, a threshold sweep for a new candidate model, local recording, downloads, changes to V4, or test evaluation. P6 may replay the existing V4 checkpoint on the specifically permitted V5.1 validation inputs.

1. Raw sources and historical source revisions remain read-only. Audio derivatives and caches belong under canonical `storage/` roots. Machine paths, recordings, environments, caches, weights and checkpoints stay outside Git.
2. Create an independent `workspaces/study-v5-1/` only after O0. Read its own `AGENTS.md` and `STUDY.md`. Do not connect V3 compatibility mounts or modify V4 source, frozen records, caches, model, statistics, runtime selection or threshold.
3. No listening and no description of inferred audio content. Labels come only from the stated metadata basis. Model scores cannot admit a source, file or label. Mining may select windows already in the frozen train reservoir.
4. Gate states are exactly `PASS`, `FAIL`, `NOT_RUN`, `NOT_IMPLEMENTED`. A report must not convert a missing implementation, authoring calculation or inherited test into a V5.1 execution `PASS`.
5. Pin dependency versions, source snapshots and numeric settings before P1. A changed source identity or dependency profile starts a new recorded attempt. Never overwrite an immutable completed attempt or frozen split.

### 0.1 Test-sealing zones

| Zone | Material and timing | Permitted work | Prohibited work |
|---|---|---|---|
| Z0 | Admitted inputs before S9, including prospective test/OOD | File inventory/hash, headers, text metadata and one logical mechanical pass per attempt. Decode, standardize and enumerate approved candidate windows in memory. Retain only the closed ledger columns, including mechanical booleans and identities. | Retained RMS/band/rolloff/tilt descriptors, plots, feature caches, derived audio files, weak-positive levels, model scores or listening. |
| Z1 | Frozen train/validation candidates and train reservoir | Re-derive and verify ledger identities; write approved derivatives; descriptors, weights, training statistics, caches and validation streams. | Opening a frozen test/OOD waveform. |
| Z2 | Frozen test/OOD after S9 | Byte-integrity checks and metadata reads only. At a separately authorized final evaluation, re-derive and match the ledger before scoring. | Decoding, writing derived test audio, descriptors, fitted statistics, caches, streams or scoring before that evaluation. |

The mechanical pass may compute temporary RMS, zero fraction, DC, peak fractions and centre-crop mel power solely to produce the fixed booleans. It must discard those intermediate values; no amplitude or spectral descriptor column may enter the ledger. This is an explicit allowance, not a reusable pre-split descriptor cache. Before S9, counts may be reported by source and cleaning reason, never by a provisional test partition.

### 0.2 Reuse boundary

Use unchanged numerical decode/resample, AE/DD cleaning/group semantics, frontend, normalization and streaming primitives. Do not call V4 orchestration, `split_gates`, training entry points, its identity-bound replay script, or its ten-second recording endpoint as V5.1 entry points.

The V5.1 loader reads its own configuration, validates `inputs`, `sampling`, `augmentation`, `replay` and `approval`, then passes only `frontend` and `decision` to the unchanged frontend mapping helper. The V4 parser currently rejects the extra root keys. Implement this adapter in V5.1, without changing the V4 parser or its protected hashes.

## 1. Evidence and hypotheses

V4 uses AE binary clips and DD audio. Its provisional browser threshold is raw logit -3.00 with two-of-three voting; the Python service remains raw-score-only. The threshold came from a 9.4-minute ambient log with zero crossings but an approximately 19.4/hour upper false-alarm bound. The five-crop validation summary had 581/1,406 negative clips above that threshold. These observations do not establish a deployment operating point.

V4's class-conditional rolloff gaps are descriptor observations, not proof that its model uses a spectral shortcut. H1 concerns level mismatch; H2 concerns logit association with spectral descriptors; H3 concerns ambient representativeness; H4 concerns insufficient measured exposure. A0 is descriptive evidence from existing score logs. P6 examines raw-logit relationships on permitted validation waveforms. Neither alone establishes causation.

The existing V4 ambient log stores no waveform and has missing statistics/frontend identity fields. It can support a labeled descriptive A0 report, not identity-verified V5.1 calibration. MUSAN noise has previous V4 diagnostic replay and candidate-threshold exposure. It was not V4 training data, and it was excluded from the selected ambient-only frontend candidate. Its prior evaluation history must remain visible.

## 2. Sources and project binding

| Code | Source identity and evidence | Label and default role |
|---|---|---|
| ae | `al-emadi-r001`, registered. 11,704 binary files; V4 rules retain 10,623, including 1,332 positives. | Folder classes; pooled source. Preserve prior test exposure. |
| dd | `thesis-r001`, registered. 90 recordings, 30 per filename class; 900 derivatives, 890 retained including 300 positives. | DRONE=1; BACKGROUND/HELICOPTER=0; pooled source. |
| itu | Proposed `itu-aris-r001`: extracted Zenodo tree only; 4,491 drone/1,000 background segments; 35 declared drone intervals and 20 background blocks. | Drone `interval_annotation`; background `unannotated_stretch`, always weak. Pooled source. |
| mic | Proposed `micarray30s-r001`: 163 real eight-channel 48-kHz recordings, with coordinate CSVs; eight conservative keys in authoring analysis. | Folder-only positives, always weak. OOD by default. |
| musan | Proposed `musan-r001`: 2,016 mono 16-kHz files: 930 noise, 660 music, 426 speech. | Dataset-definition negatives; pooled clips, train reservoir and held-out streams. |
| wang | Proposed `wang-r001`: 31 WAVs; manifest lists 32 drones with one missing audio asset. | `manifest_record` positive basis, weak/provenance-limited; OOD only. |
| cochl | Optional `cochl-r001`: exact recovered local input, if supplied. Published corpus has 76,115 ten-second files and participant/submission/segment names. | Scene-class negatives after the admission gate; otherwise unused. |
| loc | Optional registered revision of completed owner-confirmed ambient sessions. | Owner-confirmed negative; inactive unless O1 selects recording. |

The published Cochl record does not prove a local download or a recovered license. Record the actual local archive hash, tool version, CRC outcomes and license text before admission.

### 2.1 Quarantine

Keep A/B/C and MLSP quarantined without their missing documentation/labels. Salford remains excluded. DroneAudioSet/FSD50K metadata without audio, duplicate ITU download locations, and optional unlocated sources are not admitted. Hash/list quarantine; never decode it. A documented O2 amendment must precede P1 to admit a formerly quarantined source. This final default has no A/B/C or MLSP admission path active; changed membership requires a revised pre-freeze plan and registered identities.

### 2.2 P0 registration

After O0, create the V5.1 workspace, copy this exact approved contract, and record its hash in `STUDY.md`. Add new global source/corpus/split/cache records under `repository/registry`, initially `planned-unbuilt`. Reserve `corpus-003`, `split-003`, V5.1 derived-build and cache identities without presenting empty files as completed artifacts. Register prospective source lineage and the OOD identities. Optional sources are recorded as absent/tier D rather than as completed imports. Existing catalog integrity must continue to pass.

The source register binds original trees and byte checksum inventories. Local path resolution lives in ignored local configuration. Every execution receives an attempt ID and implementation/dependency snapshot. Runtime selection stays V4.

## 3. Frontend, identity and artifact schemas

### 3.1 Processing and normalization

Retain 16-kHz mono float32, channel 0, soxr HQ, 1,024 FFT/Hann, hop 256, no centering, 64 HTK mel bands from 50 to 7,500 Hz, area normalization, power 2, `amin=1e-12`, no top-dB, and a 7,936-sample/28-frame input. Decision cadence is the existing 31 hops = 7,936 samples = 0.496 seconds.

* `transform_hash` must remain `bb7a21a1e5fb317c43e1e214bc04af6e46e04d75053383d1c62ef6849c9466c7`.
* `cache_content_hash` for identical power processing remains `026ff2ecaa8a240383af8666f61bbfcd9ea7ef3d4e9a51c73e803d82259a43b0`. New caches receive new membership/lineage records, not a falsely changed processing hash.
* Fit new D047 per-band mean/std from eligible training rows only. Bind statistics hash, training membership hash, policy and source implementation. Do not copy V4 statistics.
* Compute the actual new normalized-input hash; never relabel V4's `028d682f79e0099f265bf9194e11ba3250d16fc8ef5377944e7d7671ab60fdd2` as V5.1. Hash equality after genuinely identical fitted arrays is not itself a failure; the independent membership/statistics lineage is mandatory.
* Put A1-A5 in `augmentation`, with its own canonical hash. Keep the historical frontend gain field at its V4 value for identity compatibility; the V5.1 trainer must consume only the new augmentation block for actual augmentation.

### 3.2 Unique IDs and sample coordinates

Canonical source-relative paths use forward slashes, no leading slash or traversal, Unicode NFC, case preserved. Reject normalization/case collisions before MP. `source_file_id = sha256(UTF8(source_revision + NUL + canonical_relative_path))`, full lowercase 64 hex digits. Store file-content SHA-256 separately.

For new sources:

`clip_id = code + "__" + source_file_id + "__s" + start_sample_16k:012d + "__n" + n_samples:08d`.

Offsets are zero-based in that physical file's whole-recording standardized waveform. ITU offsets restart in each segment file, which is safe because the file ID differs. `group_key` is a separate source-qualified field and never replaces the file ID.

AE/DD are explicit exceptions: preserve their exact V4 `clip_id`, `group` value as `group_key`, and duplicate representative IDs. DD's legacy suffix is seven-digit milliseconds. Do not convert it to a sample suffix. The independent `start_sample_16k` column always stores the true sample coordinate. `legacy_clip_id=clip_id` for AE/DD and null for new sources. R3 compares these exact legacy maps and retained standardized sample identities.

Reject duplicate `clip_id` values even if hashes match. Derived filenames for legacy IDs use a separate safe SHA-256 of `clip_id`; preserve the scientific ID in metadata rather than writing colons/slashes into Windows filenames.

### 3.3 Complete schemas and storage rules

The companion contract lists every required column and type. The closed artifacts are:

| Artifact | Identity and permitted contents |
|---|---|
| `inventory_v51.csv` | File ID, source/revision/path, file hash, bytes, native header fields and text-metadata-derived recording/group/class/label basis. No decoded descriptor. |
| `label_register.csv` | Source file ID, source class, binary label, label basis/weakness and source-metadata identity; one row per inventoried admitted file. No sample-derived label. |
| `mechanical_ledger.csv` | Clip/file/source identity, standardized coordinates, sample hash, mechanical flags, label metadata, candidate role, cleaning decision, representative and pin state. No split column or retained RMS/spectral values. |
| `train.csv`, `val.csv`, `test.csv`, `ood.csv` | All retained ledger columns plus `split` and deterministic logical `derived_relpath` (null for AE). This is a future path for sealed test/OOD, not permission to write it. |
| `reservoir_train.csv` | Same schema as split manifests, `split=train`, `candidate_role=reservoir`; membership restricted to train source-file groups. |
| `quarantine_manifest.csv` | File ID, source name, relative path, file SHA, bytes, reason; no decoded columns. |
| `descriptors_v51.csv` | Clip ID, split-manifest hash, named descriptors, descriptor flags, weak-positive/distance results, and nullable train weight. Only train/val rows. |

CSV is UTF-8 without BOM, LF, fixed header order, rows sorted by `clip_id` (inventory/quarantine by source/file ID). Booleans are `true`/`false`; null is an empty field; integers are decimal; finite floats use `.17g`. NaN/Inf are prohibited. SHA fields are lowercase hex. JSON uses UTF-8, sorted keys and fixed separators for hashed content. Timestamps/machine paths never enter membership-hash domains. No frozen CSV is edited after S9.

### 3.4 Mechanical flags

On finite standardized float32 samples, compute `exact_zero`/`silent` as all samples zero; near-silent as nonzero whole-waveform RMS below -80 dBFS; partial-zero as nonzero waveform with zero fraction >0.5; clipped as fraction with absolute amplitude >=0.999 greater than 0.01; DC flag as absolute mean >0.01. Keep the V4 AE/DD numerical definitions and source-identity checks unchanged.

Compute `crop_all_zero` and `stats_eligible` with unchanged V4 fix-length/centre-mel-crop semantics. Temporary centre-crop power is allowed in Z0 for these booleans only. `stats_eligible = not(silent or near_silent or crop_all_zero)`. Neither the temporary numerical descriptors nor power are retained. A rejected-too-short row has null crop/eligibility booleans and is excluded before S9. New non-AE candidates have exactly 16,000 samples; AE preserves native length and V4 eligibility rules.

## 4. Per-source grids, grouping and labels

### 4.1 AE and DD

Apply V4 AE rejection, nine exact-zero representatives, 1,079 zero-duplicate exclusions, grouping and pinning. Expected retained: 10,623 = 9,291 negative + 1,332 positive. Preserve `extra_membo_D2` as its one train-pinned group. Original AE files remain referenced in place, with no rewritten one-second copy.

Resample each whole DD recording to 16 kHz with channel 0/HQ; enumerate 16,000-sample non-overlapping windows at starts 0, 16,000, ...; count the remainder. Preserve V4's 900 candidates, ten within-source duplicate exclusions and 890 retained: 300 positive, 290 helicopter, 300 background. Group by original recording and preserve V4 representative IDs. R3 compares hashes and maps; it does not rerun V4 orchestration.

### 4.2 ITU

Use the exact extracted Zenodo tree; do not read duplicate snapshot/partial payloads. Drone=1 with `interval_annotation`; background=0 with `unannotated_stretch` and `label_weak=true` for all background rows. Every file must resolve to its declared metadata group.

Merge drone intervals that overlap or have gaps <=1.0 second, transitively in ascending start/group order. Name the merged group by the smallest original group number. Keep each declared background block intact. Background timestamps are absent: apply no proximity filter and invent no coordinates. Report possible overlap between background blocks as L-ITU-2; do not claim an independent-session or proven sample-disjoint background test.

Resample each one-second file separately to exactly 16,000 samples. Record its unique file-based clip ID. ITU event eligibility requires an explicit per-segment `start_s`; current segment records lack it, so current ITU event count is zero. All ITU metrics carry the single-session caveat.

### 4.3 MIC and Wang

Exclude `__MACOSX`, `.DS_Store`, and `._*`. MIC real audio is 163 files. Strip all trailing numeric underscore tokens from the containing folder to form a conservative flight key; expected eight keys. These names are not independently verified flight identities. Label all MIC candidates positive with `folder_name_only`, `label_weak=true`; default all MIC groups to OOD.

Only a recorded O5 decision before P1 may admit MIC to the pool. Admission needs at least 12 conservative keys or documented flight independence. Use whole-recording channel-0 resampling, then one-second windows. If pooled, distance is informational: match CSV time to `[start,end)` at 16-kHz coordinates, use finite X/Y/Z and median Euclidean magnitude. Missing/non-numeric/constant position rows yield unknown. Train-only valid values define terciles; no row is dropped on distance. OOD distance is deferred to evaluation.

Wang is positive-only OOD. Keep the missing manifest asset explicitly absent; never fabricate it. Register local provenance/license as unestablished. Its five-second recordings are mechanically enumerated at one-second starts, but no derivative or descriptor is written until final evaluation.

### 4.4 MUSAN and the reservoir

Group by physical file. Canonical exact-byte duplicate files within one source share a single canonical file group; alias paths remain recorded, not separately split. Record every ANNOTATIONS/LICENSE schema. Associate each row to its named file; scan the filename and every associated text field with lowercase ASCII-alphanumeric tokenization. Exact tokens X1={drone,uav,quadcopter,quadrotor,multirotor} exclude the file. X2={helicopter,propeller,rotor,aircraft,airplane,jet,engine,motor} assign N5. X1 takes precedence. Missing annotation is recorded, not filled by content inference.

For a standardized file with N samples, budget M0=12 for noise and 4 for music/speech. Let K=floor(N/16000), M=min(M0,K). If M=0 there is no clip; if M=1 use start 0. If K<M0 use starts `16000*k` for k=0..M-1. Otherwise use integer starts `floor(k*(N-16000)/(M-1))`. These fractional-time, integer-sample positions are an explicit exception to the generic one-second grid.

Before assignment, MP also enumerates reservoir-candidate starts at strides 16,000 samples for noise and 160,000 for music/speech, while start+16,000<=N. Exact same file/start/length candidates appear once with role `base_and_reservoir`; all other roles are `base` or `reservoir_candidate`. After file assignment, only unused candidates from train files may become reservoir rows. Exclude base-identity overlaps, apply zero/duplicate rules, then cap at 200,000 by reservoir rank. Reservoir candidates from val/test never become training rows. No reservoir audio files are written until a separately authorized mining/training stage needs them; P4 may verify them in memory.

### 4.5 CochlScene and local ambient

`inputs.cochl_z01_path=null` by default. Missing input, failed recovery, fewer than 1,000 CRC-valid WAVs, or fewer than four scene classes means tier D and the fixed fallback. Record the outcome and continue. No download is implied. If supplied, record tool/version and input hash; validate recovered entries without original-source overwrite. Read the actual license. Preserve supplier membership as metadata but split by participant/UserID.

Base clips use whole-file resampling then starts 0, 64,000 and 128,000 samples where a full second fits. Per-user cap 12 base clips, then global caps 6,000 train /1,200 val /1,200 test by cap rank after assignment. The ledger retains candidates that are later capped, with decisions explicit. Whole-file stream eligibility is independent of the base-clip cap but obeys the same UserID partition.

O1 defaults to no local sessions. If B/C is separately approved, collect >=3 h validation and >=3 h test in >=30-minute sessions over >=4 places/times and >=2 devices. Save source-rate float32 chunks <=10 minutes with requested constraints, observed track settings, UTC, device/place, owner-confirmed drone absence, frame/drop/recovery metadata. A new V5.1 recorder is required; the old ten-second endpoint is not reused. Session is the group. Assign alternating complete sessions to val/test by deterministic rank until both duration targets are met; C extras go to train. Newly collected sessions are separately registered before their MP.

## 5. Cleaning and frozen membership

1. Validate IDs, source mappings and finite decoding. Apply the exact V4 AE/DD rules first.
2. Drop new-source exact-zero candidates before the cross-source duplicate guard; retain only the nine V4 zero representatives. Log all exclusions.
3. Compare standardized float32 little-endian sample identities across retained candidate classes/sources. A cross-class or cross-source identity is a stop. Within-source AE/DD preserve V4 representatives. For new sources, choose base candidates before reservoir-only candidates, then file-order rank, then clip ID; log aliases/exclusions. No score determines retention.
4. Preserve near-silent/clipped/DC/partial-zero flags; no descriptor-based rejection. Descriptor-bound weak-positive and distance outputs are deferred until after S9.
5. Run S1/S2 on the retained ledger-derived memberships; they establish declared group and exact candidate-identity exclusivity, not perceptual independence or absence of overlap at unenumerated continuous-stream offsets.

### 5.1 Group splits

Use seed 51 and the unchanged two-stage SGKF helper for pooled AE/DD/ITU and approved MIC: 7 folds, fold 0 test, then 6 folds, fold 0 validation; local `random_state=51`, rows sorted by exact `clip_id`. Put AE pinned groups in train first.

Strata are `source|class|substratum`: AE positive uses drone_type, AE negative style; DD uses filename class; ITU uses folder label; MIC uses `folder_positive`. Pool source-qualified group keys. Record the exact scikit-learn version and input-order digest. No retry with a different seed, order or fold count after a failed gate.

For each MUSAN subcollection, rank files by `split_musan` with key `[source_revision,relative_path]`. Noise fractions are .60/.20/.20, music/speech .80/.10/.10. Train count is floor(train_fraction*n), val count floor(val_fraction*n), remainder test. Rank Cochl UserIDs .60/.20/.20 with the same rounding; all classes of a participant remain together. Unavailable sources add no empty membership. Loc follows its explicit session rule. MIC/Wang default to `ood`.

### 5.2 S9 freeze

Freeze train/val/test/ood, train reservoir, quarantine and the complete ledger. Sort and serialize exactly as section 3.3. `FROZEN.json` binds each file hash, source/inventory hashes, approved plan/contract hashes, dependencies, seed, group/ID/grid rules, source list, L, transform hash, and B2 reference/floor policy. No runtime timestamp enters the membership digest.

Record `test_cache_created=false`. The freeze contains no post-freeze descriptor or weight results. Those are separately hashed artifacts bound back to the frozen memberships. R-DET runs preparation twice in distinct attempt directories and compares membership hashes; preserve both attempts. Do not register an alternate split or replace a freeze to make a gate pass.

## 6. Descriptors and sample eligibility

Compute `v51_descriptor` on train/val only. Use finite float64 accumulations. RMS over nonzero samples has a -240 dBFS sentinel for exact-zero; `rms_missing=false`, `all_zero_descriptor=true` makes its meaning explicit. Rolloff is the first frequency reaching 99% of mean frontend-STFT power; exact-zero rolloff=0 with the zero flag. Noise floor is the 10th percentile (linear quantile) over per-frame mean log-mel dB; tilt is the OLS slope over integer mel indices. Band power is the V4-approved inclusive 100-4,000-Hz boxcar/no-detrend periodogram integral, named separately from STFT rolloff.

Report V4 periodogram/whole-waveform descriptors under their own names. Do not compare numerical gaps as if these estimators were identical.

Weak-positive uses the absolute band-level difference <=3 dB from its own source's train-background median: AE unknown, DD BACKGROUND, ITU background. Use all retained base train-background rows, including quality-flagged rows and finite exact-zero sentinels; normalization eligibility is not an extra background-selection rule. A missing train reference yields `weak_positive=null` plus a reason; never pool a validation or test reference. For approved pooled MIC use each recording's train-clip tenth percentile. OOD/test descriptors are evaluation-only.

B5 reports per-source class-median gaps in named v51 RMS >6 dB, rolloff >1,000 Hz or noise floor >6 dB. One-class sources have undefined class gaps, explicitly recorded rather than substituted. These warnings do not delete rows or change weights automatically.

Compute RMS/rolloff decile edges once from unweighted train rows with linear quantiles; bind edges and hash. Exact zero uses its stated finite sentinels. Duplicate edges create explicitly empty bins; do not merge or perturb them silently. Use searchsorted-right and clamp the maximum to bin 9. Report every empty/one-class bin; B1 evaluates nonempty bins and stops if their achieved class fraction is out of range. Do not fit edges on val/test.

## 7. Sampling weights and effective sample size

### 7.1 Fixed class and source margins

Default positive class prior is .35; negative is .65. The sampler uses weights as draw probabilities with replacement. A future training spec must not add V4 `pos_weight` without a separate decision.

Source/group/clip caps apply to expected probability mass, not the fluctuating counts of a finite replacement-draw sequence. Report realized draw frequencies separately; do not mark a probability-cap failure from sampling noise alone.

| Case | AE positive | ITU positive | DD positive | MIC positive |
|---|---:|---:|---:|---:|
| MIC OOD default | .35 | .35 | .30 | 0 |
| Approved MIC pool | .35 | .25 | .20 | .20 |

Negative targets are N1 .30 steady/noise, N2 .15 AE ESC, N3 .10 speech, N4 .08 music, N5 .15 rotor/vehicle class metadata, N6 .10 open-air scenes, N7 .12 crowd/indoor scene metadata. Cochl mapping follows the supplied scene classes. With Cochl absent: N1 .40, N2 .15, N3 .22, N4 .08, N5 .15; N6/N7 inactive. Assign MUSAN X2 to N5 instead of its original stratum.

### 7.2 Structural reference sampler

Use within-class probabilities q, each class summing to one. Start the reference cap scale at equal weights within the fixed positive-source or negative-stratum margin: q0_i=margin_share/n_margin. Fix the clip upper bound at 20 times that class's median q0; it is not recomputed while fitting.

For P4-P5, the reference, weights, decile edges and normalization population is the frozen base `train.csv` only. Reservoir candidates do not increase its counts or effective size and do not enter training-statistics fitting. Their descriptors may be computed later for authorized train-only mining. M1 creates a new bound sampling artifact for base plus selected reservoir rows; it does not rewrite the base freeze.

Structural constraints: exact M1 margins; nonnegative probabilities; clip bounds; each source <=.35 of its class; each group <=.015 of its class; aggregate positive-only sources <=.25 of positive draws; aggregate negative-only sources <=.35 of negative draws. Source-only classification depends on the admitted pool, not whether a source has a missing split class.

First solve, separately per class, the strictly convex problem minimize sum(q_i^2) subject to those structural constraints. This gives q_ref and `n_ref=1/sum(q_ref_i^2)`, the maximum feasible Kish size under the fixed structural design. Infeasibility or a missing source/stratum is B0 failure, with an acquisition/constraint gap report. Never change source shares or caps automatically.

Equivalent group/cell aggregation is allowed only when clips in a cell have identical structural coefficients and bounds; objective contribution is mass_cell^2/n_cell. Expand uniformly and verify every original clip constraint afterward.

### 7.3 Suitable B2 requirement for this data

**Explicit change from the prior proposals:** 3,000 per class is a reported acquisition target, not a blocking requirement that the available positives cannot meet. Blocking B2 retains at least 80% of the best structurally feasible Kish size, without quantizing the floor:

`B2_min_class = min(n_ref_class, 0.80 * n_ref_class)`.

O0 amendment O0-A2 (owner-approved 2026-11-27) removes the earlier round-up-to-100 step. It preserves the 80% retention requirement and all fitted shares, margins, constraints, and solver settings.

Compute this once after freeze from q_ref, before fitting descriptor margins. Freeze its output in `weight_reference.json`, bound to membership, solver and contract. Do not recompute a lower reference after descriptor balancing, change the 80% rule, or lower the threshold after a failure. A reference below 100 gets the exact reference floor and is explicitly a severe diversity gap; it does not imply scientific adequacy.

The metadata-only seed-51 authoring calculation gives:

| Class | Candidate train rows | Best structural Kish size | Proposed B2 minimum |
|---|---:|---:|---:|
| Positive | 4,396 (AE 972, DD 220, ITU 3,204) | 1,667.85 | 1,334.28 |
| Negative | 15,890 | 3,537.75 | 2,830.20 |

These are sizing estimates, not a frozen split or execution gate result. The companion `DATA_PREPARATION_CNN_V5_1_EFFECTIVE_SIZE_BASIS.json` records the metadata assumptions, prospective counts, source/reference hashes and residuals. The estimates exclude unrun new-source zero/dedup and K1 effects; Cochl/loc are absent and MIC is OOD. The actual gate is calculated by the fixed formula on cleaned, frozen training rows. Report absolute Kish size, ratio to reference, count-only ceiling, group/session counts and the shortfall to 3,000 separately. Kish size measures weight concentration, not independent recording conditions, power, accuracy or deployment readiness. Per-stratum Kish size below 150 is a flagged gap, not an automatic class-level failure.

### 7.4 Descriptor-balanced weights

Add M2: positive draw fraction .35 in each nonempty rolloff decile; M3: the same in RMS deciles. In joint probabilities w, enforce class sums .35/.65 and structural caps scaled by the class prior. The equalities are linear: `.65*positive_mass - .35*negative_mass = 0` within a decile.

Minimize `sum_pos((w_i/.35)^2) + sum_neg((w_i/.65)^2)` under M1/M2/M3 and the structural constraints. This explicitly replaces the earlier underspecified IPF loop with a uniquely defined minimum-concentration quadratic program. If the solver proves infeasibility, drop M3, then M2; never M1. A numerical failure is a stop, not evidence allowing a margin drop. Report dropped margins and all residuals. B1 still requires every nonempty RMS and rolloff decile's positive share in [.21,.56]; dropping a margin does not waive B1. B2 compares final within-class Kish sizes with the fixed reference floors.

Use OSQP 1.1.3 builtin direct solver; record NumPy/SciPy/OSQP/binary platform identities. Settings: float64 CSC matrices, eps_abs/eps_rel=1e-11, max_iter=200000, adaptive_rho=false, rho=.1, warm_starting=false, polishing=true, scaled_termination=false, check_termination=25. Sort variables by clip ID and constraints by the contract order. Require status solved, finite nonnegative-within-1e-10 weights, and original-unit max constraint violation <=1e-8. Zero only negative roundoff in [-1e-10,0); recheck every constraint and class sum. No solver fallback or random restart. Reference and final solver records are separate. O0 amendment O0-A1 (owner-approved 2026-11-27) tightens eps_abs/eps_rel from 1e-9 to 1e-11 after the original profile returned a weight below the fixed nonnegativity tolerance. The solver remains OSQP 1.1.3 builtin direct; this is a numerical-tolerance amendment only, not a change to margins, constraints, solver, or post-solve rules. The solver is an authoring dependency choice covered by O0, not an installed V5.1 environment claim.

## 8. Deterministic training augmentation interface

All draws use section 10's functions, with epoch, draw index and clip ID in the key. Training operates on cached power. Evaluation gets no augmentation.

Order: choose frame-aligned 28-frame base/donor crops; A2 mix; A1 common gain; A3 response; A4 self-noise; log transform and fitted normalization. Crop selection is A5 and precedes operations so donor/base dimensions agree. For training SNR scales, P5 precomputes train-only band power and full-window RMS for each eligible waveform crop start, using the periodogram/whole-waveform contracts. Bind this scalar lookup to sample/manifest/crop identity; its offset domain is the unchanged V4 fixed-length waveform, it contains no test/val values, and it requires no training-time waveform read.

* A1: gain uniform [-36,+6) dB, converted to power factor 10^(gain/10).
* A2: positive mix probability .6 with eligible train negatives N1/N5/N6/N7, nonzero band power; donor gain [-30,0) dB, target nominal SNR [-12,+20) dB. Negative mix probability .4 with another train negative, target [-5,+20) dB. Donor sampling is weighted uniformly by eligible file group, then candidate clip; do not reuse the base group. Rank/keys decide ties. A missing nonzero donor or zero base band power skips mixing and records the reason. The probabilities differ by class: report that design rather than claiming mixture presence cannot be a cue. SNR uses the bound crop band-power lookup; donor gain is applied before scaling base to target SNR. Add power matrices; A-chk audits the approximation.
* A3: probability .5. Evaluate the second-order Butterworth high-pass magnitude-squared response at mel-band centres, corner [0,300) Hz (corner zero means unity). Independently with probability .3 inside A3 apply analogous low-pass corner [4000,7500). Apply linear mel-index tilt with total end-to-end span uniform [-6,+6) dB; endpoints are -span/2,+span/2. Add three independently drawn ripple values [-2,+2) dB at indices 0,31.5,63 with linear interpolation. All offsets are power dB.
* A4: probability .5. Add a fixed frontend white-noise power template scaled to RMS uniform [-90,-55) dBFS. Build the template once from a bound 16-kHz, 64-second, zero-mean/unit-RMS synthetic normal sequence; mean its mel power by band. Template keys and source/array hashes are recorded. Its generation is synthetic, not a source admission operation.
* A5: choose an integer crop uniformly from 0 through available_frames-28 inclusive, independently for base and donor. Use unchanged V4 fix-length behavior before feature construction.

A-chk compares waveform addition and power addition on exactly 200 deterministically selected eligible train crop pairs at 0 dB, before A1/A3/A4. Use the same gains, crop starts, frontend and log floor. Report mean absolute dB error across all pairs/bands/frames and per-pair quantiles. Mean >3 dB is a stop; do not reselect easier pairs.

## 9. Continuous streams, events and future threshold protocol

### 9.1 Replay and exposure

Create a versioned `replay_contract.md` that binds LiveWindowBuilder, VoteSmoother and the current browser reset rules by source hashes. Use three fresh scored windows before decisions. Reset votes on standby, buffering, recovered capture/generation changes, new files/sessions and stop. Do not reset the waveform builder merely because a quiet window is emitted; mirror the production branch. A real gap resets builder and votes. A timing/reset mismatch that changes decisions is a blocking parity failure, not an accepted deviation for certification.

Replay whole files at native rates, flush complete tail windows through `finish`, and timestamp by standardized source-sample window end, not HTTP receipt time. A scored window represents its half-open 7,936-sample exposure interval. Exposure is the union of scored intervals excluding gap/recovered windows; never sum overlapping intervals or count quiet/buffering time. Store absolute file/session time for episode merging. Merge active episodes only within the same uninterrupted source file/session when the elapsed inactive gap is <5 seconds. Never merge across a file or capture reset.

Environment classes: E1 remaining MUSAN noise plus DD BACKGROUND and admitted Cochl N1; E2 speech; E3 music; E4 Cochl N6/N7; E5 approved loc; E6 DD HELICOPTER/Cochl N5/MUSAN X2, informational only. Assign X2 files to E6 instead of also counting them in E1/E2/E3. Exclude AE, ITU background and quarantine. Every stream/bed must have recording label 0, label_weak=false and the correct split. Rank files and retain whole files until cumulative duration would exceed four hours per class/split; omit the exceeding file and continue to later shorter eligible files. Measured scored exposure, not nominal duration, controls inference.

### 9.2 Exact event construction

Beds have 320,000 samples at 16 kHz: a contiguous region of one >=20-second MUSAN/loc file, or two verified consecutive Cochl segments from one submission. For Cochl concatenate native samples only after verifying identifiers, index adjacency, lengths, rate and same split, then resample the 20-second pair. No unrelated concatenation, loops or padding. DD/unpaired short files still supply alarm streams but not beds. Allocate events as evenly as possible across available eligible bed classes; integer remainders go by class code. Select source groups uniformly, then eligible files/regions by deterministic draws.

Events are exactly 64,000 samples. DD/MIC/Wang use contiguous whole-recording standardized regions with start chosen from one-second boundaries where four seconds fit. ITU requires explicit per-segment times and four non-overlapping consecutive seconds of one group; absent times mean no events.

**AE rule:** order clips by verified parsed numeric segment index within one original recording; do not bridge recording names merely because they share a V4 block group. Require consecutive numeric indices, concatenate the unchanged standardized waveforms, and extend with the next consecutive clip until >=64,000 samples exist. Take the first 64,000 samples; if the sequence ends earlier, reject that event candidate and record insufficient length. Never pad, stretch, repeat or rewrite training AE clips. Mark every such event `contiguity_verified=false` and report AE recall separately. At retained original clip joins within the selected 64,000 samples, apply a non-overlapping 80-sample fade-out before and 80-sample fade-in after the join. Intersect a join's fade mask with the available event sample positions; never extend the event to complete a fade. Trimmed/outside joins receive no dip.

Apply a 4,000-sample raised-cosine outer ramp. Measure event and the bed's placed-region band RMS over event samples [4000,60000), using SciPy Butterworth order=4 bandpass [100,4000] with SOS zero-phase filtering. Scale event to nominal SNR, then place at sample 112,000 (7 seconds). If mixed peak>.99, scale the entire mix by .99/peak and record it. SNR labels refer to the pre-common-scaling added-noise ratio, not true acoustic SNR. Exact sample counts and boundary/dip indices are recorded.

For event candidates with zero band RMS <=1e-6, skip deterministically to the next candidate. SNR bins are +20,+15,+10,+5,0,-5,-10 dB. Target >=120 events per bin/split from >=25 eligible positive groups. If fewer groups exist, retain the count, label low diversity and do not manufacture groups. No eligible group means no recall curve, not zero recall. A detection is a true alarm window whose source-time end is in [7,10] seconds. Report pre-event alarms separately so an already-open false episode cannot be presented as event-triggered detection; the primary event result requires a new active episode in that interval. Latency is its first alarm-window end minus 7 seconds.

### 9.3 Rate estimation and future candidate selection

Report episode rate per class and the larger of the one-sided exact Poisson 95% upper bound and the file/session block-bootstrap 95th percentile from 2,000 replacement resamples. Recompute merged episode counts and valid exposure for the sampled independent file/session blocks; do not merge blocks across their boundaries. A zero-exposure replicate is invalid and reported; if no positive-exposure estimate can be formed, report unestimable rather than pass. Resampling uses fixed bootstrap keys.

For zero episodes the Poisson bound is -log(.05)/T, approximately 3/T. >=3 hours of zero-alarm scored exposure is needed for an approximately <=1/hour class-level claim; short classes remain uncertified. Kish sizes and nominal source hours do not substitute for exposure.

The following is a **future authorized model stage**, not P0-P7 preparation: replay validation logits once; evaluate thresholds -10.0 through +6.0 inclusive using integer tenths; select the smallest threshold with empirical episode rate <=.5/hour in every binding class (exposure>=.5h), excluding E6. If no binding class exists or no threshold satisfies it, stop without a selected operating point. Record that empirical selection is not 95%-confidence certification. Report bounds, per-SNR recall/latency, ambient ceilings and clearance. Recall at +10 dB below .5 is an insufficient-model/data result, never a reason to retune the threshold. No test score participates.

After separate final-evaluation approval, verify the ledger, build sealed test/OOD streams once, and use the frozen candidate threshold without retuning. A failed test result remains a failure with its exposure/limits; do not recycle that test as untouched evidence in a replacement study.

### 9.4 P6 and mining boundaries

P6 replays the existing V4 checkpoint only on permitted V5.1 validation inputs. AE/DD events are excluded because their groups may have V4 training overlap; MIC/Wang remain sealed. Current ITU timing records admit zero events, so default P6 is ambient-only and cannot report a new-source positive sensitivity curve. MUSAN retains its prior V4 diagnostic history. Logit/descriptor associations are reported descriptively, not as a causal shortcut proof.

Future M1 mining is separately authorized with the V5.1 training specification. Score only the frozen train reservoir; add the top 2% per stratum, count rounded down, ties by mining rank and clip ID. Labels/files/sources remain admitted independently of scores. Refit weights under this same contract and train a new run from scratch. Record the parent/round without editing the original frozen base split or its artifacts. No validation/test/OOD mining.

T1 uses 20 ranked train clips per test rate 16/22.05/44.1/48 kHz. Convert each standardized train waveform to the chosen source rate; compare streaming output against an offline resample of that same source-rate waveform back to 16 kHz, at identical emitted starts. Do not compare against the pre-roundtrip original. Use the production fix-length/frontend; flush both paths consistently. Max absolute mel-dB difference <=.05; no test/OOD fixtures. Assert Python/browser-compatible reset sequences and median-of-three equality on ready windows.

## 10. Exact random-draw contract

Seed is integer 51. Canonical key strings are JSON arrays of explicitly typed components: NFC strings and integers only, with nested arrays permitted; booleans, floats and null are rejected in random keys. Rates are integer Hz and SNR bins are integer dB. Use UTF-8 and no spaces. Define H(purpose,key)=SHA1(UTF8("51:"+purpose+":"+canonical_JSON(key))). Finite ranking is ascending (full digest hex, canonical key). No Python hash or global RNG is used.

Uniform U uses the first 13 hex digits (52 bits): `(int(prefix,16)+.5)/2^52`, strictly between 0 and 1. Uniform [a,b) is a+(b-a)*U; Bernoulli(p) is U<p; integer 0..n-1 is floor(n*U). Categorical uses candidates in canonical ID order, normalizes their finite probabilities deterministically, sets the final cumulative entry to exactly 1.0, and chooses the first cumulative value strictly greater than U. Weighted draws are with replacement unless the operation is expressly a finite rank/cap.

Augmentation keys are `[epoch,draw_index,clip_id,parameter,component_index]`, epoch/draw zero-based. Branch flags and every continuous/component draw have distinct parameter strings. Purposes are `aug_gain`, `aug_mix_flag`, `aug_donor_group`, `aug_donor_clip`, `aug_bed_gain`, `aug_snr`, `aug_response_flag`, `aug_hp`, `aug_lp_flag`, `aug_lp`, `aug_tilt`, `aug_ripple`, `aug_floor_flag`, `aug_floor_rms`, `crop_base`, `crop_donor`. Future sampler draws use `sampler` with `[epoch,draw_index]`; every replacement draw has a distinct draw index.

Other purposes and keys are enumerated in the companion contract: split_musan, split_cochl, split_loc, cap, reservoir, file_order, bed_class, bed_file, bed_start, event_group, event_start, bootstrap, parity, mix_check and mining. Bootstrap keys include environment, split, replicate and draw index. Template normal pairs use purpose `synthetic_noise`, key `[pair_index,component]` and Box-Muller; for each pair, z0=sqrt(-2ln U0)cos(2pi U1), z1=sqrt(-2ln U0)sin(2pi U1). Normalize the complete fixed sequence once before template feature extraction.

Only SGKF uses its explicitly permitted local seeded library randomness. Pin the library version and sorted input digest; no retries. Numerical solvers are deterministic algorithms with pinned settings, not random sampling. Different hardware/library results outside the stated tolerances require a stopped reproduction investigation.

## 11. Gates and deliverables

| Gate | Blocking pass condition |
|---|---|
| I1/I2 | Source hashes unchanged; zero writes to source roots. |
| I3/I4 | Label basis/weakness recorded; quarantine complete and absent from admitted memberships, weights and streams; quarantine never decoded. |
| M-P | Exact closed ledger schema; IDs unique; temporary descriptor data discarded; no split report before freeze. |
| P-AE/R3 | AE counts and AE/DD identity/group/representative maps match the declared V4 reference. |
| R2a/R2b | Processing hash preserved; actual new normalized-input/statistics/membership lineage recorded without reusing a false V4 identity. |
| R-DET/R-LED | Two attempts have identical membership hashes; every permitted re-derivation matches ledger sample hash. Test R-LED waits for final evaluation. |
| S1/S2 | No group or retained candidate sample identity crosses splits. |
| S3/S4/S5 | Both labels in train/val/test; >=25 positive groups in val and test; each pooled source with >=6 groups appears in val/test. |
| S6/S7/S8/S9 | Pins train-only; OOD absent from train/val/weights/reservoir; reservoir train files only; all freeze bindings present. |
| K1 | X1 files excluded; X2 remapped and reported; no inferred content labels. |
| B0 | Structurally feasible reference with verified original-unit residuals. |
| B1/B2 | Both descriptor balance checks within [.21,.56]; class Kish sizes meet precomputed reference floors. |
| B3/B4 | Source/group/clip/single-class aggregate caps satisfied. |
| B5/B6 | B5 descriptor gaps are reported, nonblocking. B6 checks A1-only full-window levels: precomputed full-window RMS dBFS plus its sampled A1 gain, before A2/A3/A4. There must be >=5% per class in each 10-dB bin [-70,-60),...,[-20,-10], measured on fixed 100,000 weighted train draws/epoch 0; exact zero stays at its sentinel and out-of-range mass is reported. This is not a measured waveform RMS of the later power-domain mix. B6 failure is a stop, not a retuned gain range. |
| A-chk | The fixed 200-pair mean mixing error <=3 dB. |
| T1/T2 | Four-rate streaming/frontend parity and ready-window/reset decision parity. |
| T3/T4/T5 | Measured exposure, recording-level negative purity, correct split, eligible event groups and sample/timing integrity. No positive training group in validation/test event sources. |
| T6 | No sealed waveform access or derived test/OOD output after S9 before final evaluation. |
| MIC-DATA | Pooled-only distance results, unknowns and reference edges recorded. OOD MIC remains undecoded after freeze. |

Report all gate states by phase; future-stage gates are NOT_RUN, not a preparation failure. Low event diversity, missing optional sources, zero P6 events, the 3,000 Kish acquisition target and sub-stratum 150 target are explicit limitations, not hidden gate waivers. A failed blocking gate ends that attempt with its report. Preserve locks and do not modify the protocol to turn the same attempt green.

Deliver inventory/label/quarantine/ledger files; frozen memberships and FROZEN.json; train/val derivatives; train-only reservoir metadata; descriptors and weight-reference/weight-fit records; train/val power caches and fitted normalization; bound train crop-band lookup; augmentation template/hash; validation stream/event manifests and replay contract; A0 and P6 reports; gap report; implementation/environment snapshot; phase gate ledger; `DATA_PREPARATION_CNN_V5_1_REPORT.md`. A continuous recorder is delivered only under O1-B/C. Full artifacts remain local; Git contains plans, source code, sanitized registry and reports only.

## 12. Run order

| Phase | Authorized work after O0 | Stop or limitation |
|---|---|---|
| A0 | Describe existing V4 score/RMS logs against recorded training levels, clearly stating missing waveform/identity. | Missing logs: report unavailable; do not invent data. No V5.1 calibration claim. |
| P0 | Independent workspace, planned identities, V5.1 loader, approved contract snapshot and dependency lock. | Missing O0 or broken catalog integrity: no execution. |
| P1 | Read-only inventory, text label/K1 register, quarantine, optional-input outcomes. | Identity/admission errors stop; Cochl absence follows tier D. |
| P2 | MP, mechanical cleaning, exact identities and V4-reference comparisons. | Schema/identity/cleaning gate failure stops. |
| P3 | Fixed grouping, assignment, caps and S9 freeze; two-attempt membership reproduction. | S-gate/reproduction failure stops. |
| P4 | Re-derive/verify train/val; descriptors; structural reference; fixed B2 floors; balanced weights and gap report. | B0/B1/B2/cap failure stops. |
| P5 | Train/val caches, train-only statistics and band lookup, augmentation template, A-chk/B6. | Numerical/identity/augmentation gate failure stops. |
| P6 | New validation replay harness, T1/T2 and permitted V4 baseline; no new candidate-model threshold selection. | Parity/purity failures stop; zero eligible event groups are reported. |
| P7 | Preparation report, decisions, gate ledger, immutable artifact identities and handoff. | Future model/test stages remain NOT_RUN. |

## 13. Owner decisions and final authoring choices

* O0: approve execution of this exact plan/contract and MP relaxation. Default no execution. Approval includes the explicitly revised B2 reference policy, quadratic-program weights, fixed random schema, ID exceptions and AE event extraction.
* O1: local recording A=none (default); B=validation/test only; C=B plus separately grouped training sessions. No recording implementation or capture starts before B/C is selected.
* O2: missing source documentation. Default quarantine remains.
* O3: Cochl input/download. Default no input, tier D. Downloads are separate actions; no invented local archive path.
* O5: MIC pool admission. Default OOD; document the eligible independent/conservative group decision before P1.

The user requested an effective size suited to the data. O0-A2 sets the blocking floor to exactly 80% of the fixed structural reference: the metadata sizing illustration gives 1,334.28 positive and 2,830.20 negative, rather than a ceiling-rounded floor. The absolute 3,000-per-class value remains a reported acquisition target, not a blocking requirement. This is a design tolerance for weight concentration, not a power analysis. The solver objective and reference calculation are explicit so the floor cannot be made easier by first producing a poor reference sampler.

Remaining scientific limits are source-level labels, the ITU single session and unknown background-block overlap, AE prior test history, MIC/Wang provenance/label weakness, class-dependent training mixture probabilities, nominal rather than true acoustic SNR, power-mixing approximation, and default absence of device ambient. A completed preparation run does not establish a valid operating point.

## 14. Review traceability and references

| Review issue | Final resolution |
|---|---|
| First F1 and corrected C3 | Explicit MP allowance and complete schemas, including identity fields, crop/eligibility booleans and discarded intermediates. |
| First F2 | Unchanged processing hash, separate cache lineage/new training statistics, V5.1-only config adapter. |
| First F3 | Weak ITU background; no invented times; overlap caveat; explicit event ineligibility without segment times. |
| First F4 and corrected C4 | Verified contiguous beds; exact 64,000-sample event; variable-length AE sequence/extension/trim rule, no padding. |
| First F5 and corrected C1 | B3-compatible shares; MIC OOD; data-sized B2 reference/floors and group/session caveats. |
| First F6/F7 | Recording-label stream purity; unambiguous optional Cochl fallback. |
| First F8 and corrected C2/C5 | File-based unique IDs with exact legacy exceptions; full random sampling/key contract and split exception. |
| First F9 | New identity-parameterized replay; browser parity; recorder inactive by default; sealed evaluation boundary. |
| Additional notes | Named zero-safe descriptors, explicit grids, measured exposure, prior MUSAN replay history, zero default P6 events. |

Source authority: project catalog and existing source/corpus/split/run registries; both reviews under `docs/sources/`; supplied proposal digests `3d2769f37a2a4d8cb059be59dfb0eea22b4116fa1d19e6e9ba5dbe6d338ad90e` and `f5fe98c8ba651a81f73e3118465e87f467070b37948c586009bde4e6d92af112`; V4 scientific/helper source; the 27 November scout; recorded V4 training/live/MUSAN reports. The independent metadata sizing record is an authoring calculation, not registered membership.

Primary technical references: [CochlScene paper, section II.D](https://arxiv.org/pdf/2211.02289), [OSQP Python interface](https://osqp.org/docs/interfaces/python.html), [OSQP settings](https://osqp.org/docs/interfaces/solver_settings.html). Pin the actual implementation dependencies before execution rather than relying on a moving documentation page.

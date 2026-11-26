# CNN V2.0 implementation plan

Date: 25 November 2026. Status: implementation specification; archiving, preparation, training, evaluation, and export below are pending.

Build one CNN V2.0 using the architecture already selected in V1.0. Rebuild its acoustic inputs from the original raw WAVs through a corrected, persistent, CNN-only preparation pipeline. Train new weights from scratch, choose a new operating threshold on validation, evaluate the frozen checkpoint once, and export one self-contained CNN package. Preserve the old models and treated data in `archive/Models V1.0.zip` before starting the transition.

Canonical plan: `docs/CNN_V2_0_IMPLEMENTATION_PLAN.md`. Identical reading copy: `Project documents/Models/Implementation plan CNN V2.0.md` in the companion documentation folder. Repository paths in this plan are relative to Functional Project. This is a plan-only deliverable; its unchecked requirements do not describe completed work.

**Scope and decisions fixed by this plan**

The governing assessment is `DATA_PREPARATION_AUDIT_2026-11-25.md`. The selected architecture comes from `results/models/30/metadata.json` and the implemented `DroneCNN` builder, rather than from a comparison of old test scores. The existing data correction plan supplies the inventory, persistence, status, provenance, and regression requirements. This new plan supersedes its multi-model preparation/handover scope for CNN V2.0.

| Topic | V2.0 decision |
|---|---|
| Model scope | One fixed CNN architecture, one fresh training run, one exported checkpoint |
| Architecture | Selected Stage C `depth_4`, channels 16–32–64–64, global-average head, dropout 0.3, 60,641 trainable parameters |
| Initialization | Fresh random weights; no V1 checkpoint, optimizer state, threshold, or resume record imported |
| Training seed | 42, fixed before training; no seed search or ensemble |
| Source data | Original binary raw WAV corpus, read-only |
| Fresh prepared data | Newly built inventory, manifests, CNN power caches, indices, masked training statistics, provenance, and readiness evidence in V2-only locations |
| Cleaning | Reproduce the approved minimal cleaning and quality flags; do not silently denoise, repair clipping, remove DC, relabel, or discard all flagged clips |
| Split | Preserve V1 retained clip membership and groups for train/validation/test; rebuild and verify the protocol from raw measurements |
| Frontend | Fixed selected 16 kHz, 64-band HTK area-normalized log-mel frontend and corrected global-statistics fit |
| Model comparisons | None; no classical models, perceptron, MLP, clean-control CNN, architecture variants, or frontend sweep |
| Prepared features | CNN mel power only; no 84-column features, MFCC table, StandardScaler, tabular archive, or tabular handover |
| Training/testing | V2-only runners and records; no invocation of the old all-model training or evaluation scripts |
| Archive | Local `archive/Models V1.0.zip`, verified with per-file hashes and a restore check |
| Runtime integration | Export and verify the new package; switching an existing live application is a separate consumer change after acceptance |

One training run is an explicit reduction from V1's three-seed recipe. Report uncertainty over evaluation groups, but do not present seed variance as measured. The old designated seed 44 is historical provenance and is not a V2 seed-selection rule.

“Clean new data” means freshly prepared and validated representations of the existing corpus. It does not mean new recordings or automatic removal of every imperfect sample. Under unchanged rules and dependencies, many correctly rebuilt arrays and statistics should remain identical to V1. New file identities are not evidence of better audio or better accuracy.

**Evaluation validity and version boundary**

The V1 final-test record shows that the existing test split was scored on 24 November 2026. V2 must retain that fact in its protocol. Rebuilding caches, changing version names, archiving the lock, or reshuffling the same corpus cannot make the data an unseen test set.

Use the same held-out membership to prevent former test clips from entering training. Report V2 results as a new CNN evaluation on a reused historical benchmark, with the architecture previously selected on V1 validation. Do not claim an independent first-use test or use V2 test outcomes to revise cleaning, thresholds, training settings, or the architecture. A genuinely unseen evaluation requires separately collected data and a separately frozen protocol; acquiring that data is outside this plan.

This plan intentionally authorizes a new V2 mechanical test-cache build and a single V2 scoring pass in the future implementation. That differs from the older V1 correction plan's archive-only restoration restriction. Keep the old lock and results immutable; create a new study ID and evaluation record that explicitly references the old evaluation. Mechanical preparation may decode held-out WAVs to create the fixed cache, compute required integrity flags, and check shapes/finite power, but must not invoke a model, fit statistics, listen, plot test examples, or optimize any policy. Default training and verification remain unable to open test feature payloads.

**Fixed CNN architecture**

Use the existing standard PyTorch layers and preserve the complete topology, including the third max-pooling stage. Four blocks with pooling after only two blocks would not reproduce the selected model.

| Stage | Operation | Shape excluding batch |
|---|---|---|
| Input | Normalized float32 log-mel | `(1,64,28)` |
| Block 1 | Conv 1→16, 3×3, padding 1; BatchNorm; ReLU | `(16,64,28)` |
| Pool 1 | MaxPool 2×2, stride 2 | `(16,32,14)` |
| Block 2 | Conv 16→32, 3×3, padding 1; BatchNorm; ReLU | `(32,32,14)` |
| Pool 2 | MaxPool 2×2, stride 2 | `(32,16,7)` |
| Block 3 | Conv 32→64, 3×3, padding 1; BatchNorm; ReLU | `(64,16,7)` |
| Pool 3 | MaxPool 2×2, stride 2 | `(64,8,3)` |
| Block 4 | Conv 64→64, 3×3, padding 1; BatchNorm; ReLU | `(64,8,3)` |
| Head | AdaptiveAvgPool 1×1; flatten; dropout 0.3 | `(64,)` |
| Output | Linear 64→1 | One raw logit per example |

The count must be exactly 60,641 trainable parameters. Convolution biases, BatchNorm settings, pooling floor behavior, and layer initialization retain the selected builder's defaults. Validate shape in evaluation mode without modifying BatchNorm running state; reset any state changed by a construction-time probe before training. Seed all generators before model construction. No flatten head, width change, depth change, pretrained weights, imported network, or search space is part of V2.

**V2 layout and separation from V1**

```text
archive/
    Models V1.0.zip
    Models V1.0.sha256
    Models V1.0.manifest.json
    Models V1.0.verification.json
config/
    cnn_v2.yaml
requirements-cnn-v2.txt
scripts/
    archive_models_v1.py
    prepare_cnn_v2.py
    train_cnn_v2.py
    evaluate_cnn_v2.py
    export_cnn_v2.py
src/dronedet/cnn_v2/
    preparation.py
    dataset.py
    experiment.py
data/cnn_v2/
    interim/
    splits/
    power/
    normalization/
    golden/
    records/
    DATASET_READY.json
results/cnn_v2/
    training/
    evaluation/
    export/
docs/
    CNN_V2_0_IMPLEMENTATION_PLAN.md
    CNN_V2_0_DATA_CONTRACT.md
    CNN_V2_0_IMPLEMENTATION_REPORT.md
```

These paths and filenames are proposed implementation targets. Code may consolidate a thin wrapper if it preserves the command contract. Reuse canonical audio loading, resampling, mel power, normalization, metrics, and CNN code. The new package coordinates the CNN-only workflow; it must not introduce another DSP implementation.

Resolve all paths from the explicit V2 config and project root. Current generic loaders read the legacy root `config.yaml`; do not let a V2 command silently follow that route. Preserve the existing root config and concurrent live-detector work. An explicit V2 activation/configuration must select only the new paths.

**V0 Inventory the transition and archive V1**

Primary implementation: `scripts/archive_models_v1.py`. Archive all V1 model families and old treated data, including intermediate variant caches, even though V2 will use only the CNN.

| Archive member prefix | Source and purpose |
|---|---|
| `project/data/interim/` | Full old inventory, audit/cleaning records, manifests, frontend evidence and migration fixtures |
| `project/data/processed/` | All old splits, tabular features, power caches, variant caches, indices, statistics, goldens, bundles and sidecars |
| `project/results/models/` | Every old algorithm's checkpoints, all seeds, controls, metadata, and training artifacts |
| `project/results/tables/` | Ledgers, freezes, validation/test scores, final-test lock, metrics, reports and judgments |
| `project/results/cards/` and `project/results/figures/` | Old cards and diagnostic figures |
| `project/runtime/assets/` | Existing packaged V1 CNN and frontend, if present and attributable to V1 |
| `reference_documents/` | Audit, V1 preparation/model/frontend specifications, selected CNN metadata and relevant report copies |
| `source_snapshot/` | Committed source/config/notebook/dependency snapshot at the recorded revision; separately recorded task-relevant working-tree diffs where needed |
| `ARCHIVE_MANIFEST.json` and `RESTORE.md` | Contents, hashes, provenance, missing files and restore instructions |

At plan preparation, the six main data/results roots contain approximately 1.846 GB before compression, excluding the source snapshot and most runtime material. This is a planning observation, not a promised ZIP size. Recalculate counts, bytes, and available disk space at execution. Use Python ZIP64 support and streaming reads; do not assume a two-gigabyte limit or load the archive into memory.

Required archive procedure:

1. Record branch/commit, source-file digests, relevant uncommitted state, dataset/archive provenance, all frozen manifest hashes, selected architecture metadata, model/statistics digests, and original test lock. Inventory source paths before copying. Distinguish committed snapshot files from the actual working tree and record already-missing outputs honestly.
2. Use an allowlist for project artifacts. Do not recursively include all of `results/` or `runtime/`: they also contain browser profiles, environments, recordings and unrelated runtime state. Exclude credentials, cookies, private sessions, `.git`, virtual environments, caches of installed packages, raw source WAVs, incidental recordings, and the destination archive itself. Include a task-relevant model file found outside the expected roots through an explicit inventory entry; do not silently miss it.
3. Resolve absolute paths and verify containment under the intended project or explicitly named companion-document source. Reject destination/source aliasing and unexpected symlink/reparse traversal. Refuse to overwrite an existing `Models V1.0.zip`; verify and reuse an identical completed archive or stop on a conflicting archive.
4. Quiesce writers to the allowlisted V1 artifact roots for the snapshot. Preserve unrelated active work; if a selected artifact changes during the archive, reject that attempt and retry from a consistent inventory. Do not interrupt unrelated jobs or discard their changes.
5. Write a sibling partial ZIP, preserve relative paths, include the manifest, and compare source digests before/after capture. Hash every extracted member stream and check it against the manifest; run ZIP CRC validation. Reject duplicate/unsafe member paths or missing required selected-CNN/statistics/split/evaluation records.
6. Perform an extraction/restore check in a temporary directory: verify all restored member digests and metadata, reconstruct the selected CNN in evaluation mode, load its archived state dictionary, and run a synthetic input. Do not score the real V1 test set. Distinguish source snapshot restoration from support for installing an archived environment.
7. Publish the final filename atomically only after verification. Write the external SHA-256, manifest, and verification report. Keep these files local; exclude `archive/` from Git in a task-scoped edit that preserves other ignore rules.
8. Mark V1 retired for V2 orchestration and route all new work to V2 paths. Keep original V1 files available through the transition. Archive creation does not by itself authorize deleting source artifacts; disk cleanup is outside the default implementation. No V2 runner may fall back to V1 weights or caches.

Acceptance: `archive/Models V1.0.zip` exists, its exact name is preserved, per-file and outer hashes pass, restore validation passes, and the original raw/derived/model files are unchanged. V1 retirement must not silently switch a live detector that is being worked on separately.

**V1 Repair inventory and define the CNN protocol**

Fix the missing `math` import in `data_preparation.clip_stats()` and add a shared inventory validator immediately after raw measurement. The validator rejects read errors, truncated inputs, missing measured fields, nonfinite invalid values, repeated identifiers, and unexpected format. Keep exact-zero RMS's existing finite sentinel. Exercise the public route used by the notebooks as well as the underlying function.

Create `config/cnn_v2.yaml` with `study_id: cnn_v2_0`, the archive digest/reference, explicit source and output paths, fixed cleaning/group rules, seed 42, frontend settings, architecture, training recipe, validation threshold policy, and evaluation scope. Freeze the protocol and code/dependency digests before training. Record all later changes as new configuration revisions; a trained checkpoint cannot resume under a different revision.

V2 initially audits all binary files as a mechanical preparation operation. Fail on a mismatch to the measured source identity or expected counts; investigate the source or code instead of adjusting gates or selecting a favorable split. Unknown source download time and inferred repository revision remain explicitly qualified. Do not treat archive creation time as a verified download timestamp.

Acceptance: nonzero and zero audio measurements pass, corrupt input stops downstream work, source identities are recorded, and the V2 protocol is complete without any tabular or model-ladder dependency.

**V2 Rebuild cleaning, flags, and frozen membership**

Use fresh raw measurements to recreate the full ledger, not a copied V1 inventory. Validate against the archived baseline as a compatibility check. Retain all raw rows in the ledger and write separate retained split manifests.

| Rule | Required behavior and expected current-corpus result |
|---|---|
| Labels | From `unknown`/`yes_drone` folders only |
| Short clips | Reject fewer than 3,968 samples; expected 2 |
| Exact-zero contents | After the short rule, keep one sorted representative per content, preferring named silence; expected 9 retained representatives and 1,079 additional exclusions |
| Other quality issues | Preserve and flag near-silence, partial zeros, clipping and DC offset |
| Grouping | Original recording/ESC/noise-stem rules, 13 B_S2_D1 blocks, one extra Membo group |
| Pinning | Zero-representative whole groups and extra Membo group are train-only |
| Split | Seed-42 two-stage grouped split must reproduce archived kept-clip memberships |
| Expected retained total | 10,623: 9,291 negatives and 1,332 positives |

Expected train/validation/test counts are 7,607/1,510/1,506, with 967/185/180 positives and 146/29/28 positive groups. These are integrity expectations, not numbers to force by modifying data. Store a canonical sorted `(clip_id,label,group,split)` assignment digest and compare it with the archived V1 mapping. V2 manifests may add version/provenance columns and have different file hashes while the scientific assignments remain identical.

The two near-silent drone clips remain positive and flagged; extra Membo stays train-only and is reported as not evaluated. No blanket artifact exclusion, sample relabeling, recropping to find the loudest event, or new seed is allowed as an incidental cleanup. Any future alternative cleaning policy would require a separate predeclared protocol, rather than an adjustment after V2 validation/test results.

Acceptance: ledger exclusions reconcile exactly, no identifier/group/PCM identity leaks across splits, pinned/block rules pass, and archived split membership is preserved.

**V3 Build persistent CNN-only caches**

Use canonical SoundFile/soxr/librosa functions with 16 kHz, channel 0, FFT/window 1,024, hop 256, periodic Hann, `center=False`, 64 HTK area-normalized mels, 50–7,500 Hz, power 2, `amin=1e-12`, and `top_db=None`. Remeasure the soxr HQ passband using the registered sweep. Require agreement with the established edge/cutoff; report a mismatch without rewriting the frozen V2 config.

Nominal `L=0.5` means 28 frames and 7,936 waveform samples, or 0.496 seconds. Keep full-clip power for training's random offsets; store the fixed center `(n_frames-28)//2` for validation/test. Current retained corpus inputs need no padding/resampling. The general length/resampling paths remain tested on synthetic fixtures.

Existing cache assembly/publication is coupled to computing and publishing 84 tabular features. Refactor a reusable power/index assembly primitive or introduce a CNN-only orchestration over the same DSP helpers. Do not call the tabular builder and discard its result. A V2 preparation run must never execute MFCC extraction, a tabular feature catalog, a scaler fit, or a non-CNN model import.

Define a distinct CNN-only cache schema, proposed `schema_version=4` and `representation=cnn_mel_power`, with existing domain identity schema 2 where applicable. Do not label a three-file CNN cache as the old schema-3 transaction requiring tabular files. For each split publish:

- `mel_power_<split>.npy`: float32 `(64,total_frames)`.
- `index_<split>.csv`: ordered IDs, frame spans, canonical window offsets, quality flags, `crop_all_zero`, `stats_eligible`, length status.
- A sidecar binding cache/index hashes, split and training-manifest hashes, source/inventory identity, config/content identity, code/dependency provenance, shape/dtype and operation.
- A commit marker identifying the complete transaction. No incomplete set is usable.

Labels, groups, family and drone type belong in the manifest and are joined by a validating CNN dataset loader. Model input is only the normalized tensor. Default loaders refuse test payload access; only the restricted mechanical-build entry point and final evaluation command can request it with their declared operation scopes.

Expected unchanged-corpus cache shapes are `(64,457897)`, `(64,91761)`, and `(64,91494)`; total power payload is approximately 164.1 MB. Recompute rather than copy. Use staging and atomic publication to a persistent destination; refuse unintended overwrite and write `DATASET_READY.json` only after all required data checks pass. A temporary directory is reserved for verification and failure-safe staging.

Acceptance: every retained row maps to exactly one contiguous cache span; powers are finite/nonnegative float32; center offsets and all flags agree; output survives process exit; no tabular files or features are produced; synthetic interrupted publication cannot leave a ready dataset.

**V4 Fit training statistics and verify numerical parity**

Recompute `crop_all_zero` for the canonical window and define stored boolean `stats_eligible = not(silent or near_silent or crop_all_zero)`. Fit on eligible train rows only, over all full-clip frames at 0 dB. Excluded rows stay in the training dataset.

Within each mel band independently omit cells at or below −119.999999 dB; retain non-floor cells in partly floored frames. Use the corrected stable per-clip/per-band float64 moment accumulation, save float32 mean/std, and record the policy `train_stats_eligible_per_band_floor_mask_v1`.

Current-corpus expectations are 7,519 eligible training clips, 88 excluded, and 452,765 eligible clip frames before cell masking. Record per-band retained and masked cell counts; frame count is not the count of cells used in every band. Means/stds must be finite and each std positive. Compare rebuilt train/validation powers and training float32 statistics with the archived baseline in the approved same environment. Expected equality demonstrates correctness; do not introduce changes just to make V2 numerically different.

Use trusted native 16/48 kHz synthetic WAVs and saved expected outputs from the verified frontend V2 migration bundle. The current standalone reference WAVs are 0.9-second FLOAT files; preserve their true rates/encoding and hashes. Verify the existing expectations before training; do not regenerate an expected output from the current code and claim independent agreement.

Keep the 1e-3 dB golden limit, 1e-6 absolute normalized-tensor limit with zero relative tolerance, and filterbank support/1e-6 relative-weight checks. Gain invariance and area/peak equivalence remain conditional on avoiding numerical-floor effects. Performance optimizations must preserve approved numerical values.

Acceptance: correct train-only fit, exact approved-environment reproduction, existing goldens/parity pass, synthetic validation/test fitting is rejected, and generated normalized batches have shape `(B,1,64,28)` and finite float32 values.

**V5 Close preparation regressions and environment setup**

Add meaningful tests under the existing tests tree for nonzero inventory, invalid-input propagation, membership/flag integrity, persistent CNN-only publication and reload, rejected partial transactions, raw-output containment, train-only statistics, deterministic evaluation crops, test-access scopes, and process/result failure handling. Use explicit small fixture contracts rather than weakening production count gates. Test orchestration so a hidden call to the tabular path fails the CNN-only regression.

Provide a pinned CPU training environment in `requirements-cnn-v2.txt` or an equivalent recorded lock. Include the required audio/data stack, PyTorch, grouped-split and metrics dependencies, plotting and validation tools. Scikit-learn utilities for splitting/metrics are allowed; its estimators are not part of the V2 runtime graph. Verify the actual CPU PyTorch distribution/index and wheel compatibility before publishing installation instructions. The recorded working build is `2.15.0+cpu`; a local version string alone does not prove availability from another index.

Validate a separate environment without replacing the existing `.venv` or concurrent live environment. Record Python, platform, exact dependency/build versions, CPU threads, config and source digests. Same-Windows acceptance does not establish cross-OS reproducibility. A tiny synthetic forward/backward smoke check is permitted; it is not a second real-data model experiment.

Default software verification must report `scope=cnn_v2_preparation`, independent of microphone/hardware acceptance. Require exit zero, fresh parseable evidence, all named required gates PASS, matching run/config/artifact IDs and a complete package. Missing evidence, stale PASS, malformed output, any required PENDING/FAIL, or nonzero exit prevents readiness. Use new evidence paths and never overwrite V1 reports to claim V2 completion.

Acceptance: focused and applicable existing tests pass, clean-environment setup and persistent loader handover pass, full train/validation reconstruction passes once, and preparation succeeds without invoking hardware acceptance or model training.

**V6 Train the one CNN from scratch**

Primary command: `scripts/train_cnn_v2.py`. It accepts only a V2-ready dataset and the frozen architecture/config. It must have no model-list option and no dependency on old experiment ledgers, old best weights or the old all-model runner.

| Training setting | Fixed value |
|---|---|
| Seed | 42, applied before model initialization |
| Device | CPU, float32; fixed recorded thread count |
| Input | `(B,1,64,28)` normalized using V2 training statistics |
| Window sampling | One fresh uniform valid frame offset per training clip per epoch |
| Augmentation | Power-domain gain uniform in [−12,+12] dB, training only |
| Loss | `BCEWithLogitsLoss(pos_weight=n_negative_train/n_positive_train)` |
| Optimizer | AdamW, learning rate 1e-3, weight decay 1e-2 |
| Batch size | 64 |
| Scheduler | ReduceLROnPlateau on validation loss, factor 0.5, patience 3 |
| Maximum epochs | 100 |
| Early stopping | Validation loss, strict improvement, min_delta 0, patience 8 |
| Checkpoint | Restore epoch with lowest validation loss |
| Validation | Fixed center, no augmentation, `model.eval()` and no gradients |

Keep all 7,607 retained train examples, including statistics-ineligible rows. Compute the positive weight from train labels only, approximately 6.867 for unchanged counts. Do not combine weighted loss with a balanced sampler by default. Do not skip quiet random windows based on labels or retune augmentation from evaluation results.

Save loss/LR history, epoch and stopping reason, best state, optimizer/scheduler/RNG state for exact interrupted-run resume, training time, warnings, class weight, CPU threads, architecture count, data/statistics/code identities, and checkpoint digest. Resume only the same run with matching settings and artifacts. Starting again under different settings is a new protocol, not an unnoticed retry.

Acceptance: exactly one real-data CNN training run completes or resumes its interrupted execution; 60,641 parameters; finite losses/logits; no test payload read; best-loss checkpoint restored; no other model, seed search or hyperparameter sweep runs.

**V7 Freeze validation threshold and diagnostics**

For the restored checkpoint choose the raw-logit threshold maximizing validation F1, drone positive. Ties go to the larger threshold. Store the threshold's score space explicitly. Do not reuse the V1 logit threshold or apply 0.5 to logits.

Before any V2 test scoring, write an immutable freeze record with model/checkpoint digest, data/split/statistics identities, architecture, training seed/recipe, selected epoch, threshold, validation metrics, operating thresholds for validation FAR targets of 1% and 5%, environment, and reused-benchmark disclosure.

Produce CNN-only loss curves, confusion matrix, precision/recall/F1, accuracy, ROC-AUC, PR-AUC, false-alarm and miss rates, per-type/family recall and group counts, and metrics with/without each relevant quality flag. Extra Membo is “not evaluated.” Small-family results below three independent groups are descriptive without family confidence intervals.

Validation robustness diagnostics may report the existing fixed gain/filter conditions after training, with the same clean-validation threshold held fixed. They do not choose a new frontend, architecture, training setting or checkpoint, and do not trigger retraining. No promise of improvement over V1 is an acceptance gate.

Acceptance: a complete frozen CNN package/threshold exists before test access; all diagnostic labels refer only to this CNN and clearly identify validation data.

**V8 Evaluate the frozen CNN once**

Primary command: `scripts/evaluate_cnn_v2.py`, requiring the V2 freeze record. Verify archive reference, dataset readiness, test membership, cache/statistics/model identities and the absence of a completed V2 scoring lock before inference.

Run one fixed centered window per test clip, no augmentation, no gradients, and the frozen validation threshold. Produce scores for this one checkpoint only. Save score rows atomically with `clip_id`, group, true label, logit and frozen prediction. Track scoring state so interruption does not leave a false completed lock. Exact recovery may reuse verified saved score rows and compute only missing rows; it must not perform tuning or overwrite a completed result. After completion, regenerating tables/bootstrap from saved scores is allowed; rerunning model inference is not the default.

Report the same CNN metrics, confusion matrix, realized test rates at validation-selected FAR operating points, per-family/type and flag sensitivity, and 95% percentile intervals from 1,000 group-bootstrap draws with a fixed evaluation RNG seed. Resample entire groups, record class-degenerate skipped draws and require 1,000 accepted draws or mark intervals incomplete. Do not bootstrap clips as independent observations or reselect thresholds inside bootstrap draws.

The evaluation record must say “reused V1 benchmark; not an independent unseen test.” Preserve the original V1 test lock and cite its digest. A new V2 lock does not erase the previous evaluation. Test outcomes cannot reopen the freeze. A failed accuracy target is a measured result, not a reason to run another seed or modify the cleaning.

Acceptance: one frozen CNN's test scores, complete lock, group intervals and sensitivity tables exist; no other algorithm is scored and no test-derived choice changes the model.

**V9 Export one CNN V2.0 package and document the result**

Export only the selected V2 checkpoint with its architecture, frontend settings, training mean/std and policy, label mapping, validation-selected threshold/score space, input shape, source/data/config/code identities, environment, model card and native-rate golden fixtures/expected normalized tensors. The package must load in a fresh process without any path into V1 or the archive.

Verify bundle checksums, native 16/48 kHz preprocessing and tensor parity, and synthetic CNN inference. Record real model inference latency separately from frontend time; a stub scorer cannot certify CNN latency or accuracy. File/bundle software acceptance does not require physical microphone acceptance. Do not silently activate the new package in an existing live detector; provide its explicit model version/path for the consumer integration task.

Write the V2 operational data contract and implementation report. Record executed/reused/not-applicable evidence truthfully, actual counts, cleaning/flag policy, model recipe and outcomes, archive verification, known data confounds, and limitations of the reused benchmark and single seed. Preserve historical reports and add short retirement/supersession pointers only for material that belongs to this transition.

Acceptance: one independently loadable CNN V2.0 bundle and report, no dependency on old models or tabular data, and no unsupported claim of unseen-test generalization, cross-OS portability or live acceptance.

**Commands to implement and execution order**

These are proposed interfaces. The implementation must expose and verify their `--help` before executing this runbook. The output root and record paths below are versioned and local.

```powershell
# V0: archive and verify before changing artifact routing
python scripts/archive_models_v1.py --project-root . --output "archive/Models V1.0.zip"

# V1–V5: persistent fresh preparation; restricted test cache build, no model calls
python scripts/prepare_cnn_v2.py build --config config/cnn_v2.yaml

# Safe default verification excludes real test payloads
python scripts/prepare_cnn_v2.py verify --config config/cnn_v2.yaml

# V6–V7: one training run; validation threshold and freeze
python scripts/train_cnn_v2.py --config config/cnn_v2.yaml

# V8: one checkpoint, one frozen threshold, one V2 inference pass
python scripts/evaluate_cnn_v2.py --config config/cnn_v2.yaml --freeze results/cnn_v2/training/freeze.json

# V9: independent CNN bundle and parity verification
python scripts/export_cnn_v2.py --config config/cnn_v2.yaml --freeze results/cnn_v2/training/freeze.json
```

Each command writes a unique invocation record and returns nonzero on failed or pending required gates. The final publish command paths must reference already-committed protocol/code digests; the runner records the relevant source digests when unrelated working-tree changes exist.

**Acceptance gates and definition of done**

| Gate | Required result |
|---|---|
| C0 Archive | Exact ZIP name/path, manifest, outer/member hashes and restore check pass; source artifacts preserved |
| C1 Scope | Only one CNN architecture/run/checkpoint; V2 graph has no tabular extraction or other model execution |
| C2 Inventory | Fresh valid source inventory; missing-import regression fixed; invalid input stops |
| C3 Data protocol | Approved cleaning, original group/split membership and pinned rules reproduced; counts reconcile |
| C4 Persistence | CNN-only caches, indices and metadata publish atomically and load after process exit |
| C5 Statistics | Eligible-train/floor policy, finite 64-band statistics, counts and same-environment reproduction pass |
| C6 Parity | Existing native-rate expectations and normalized tensor limits pass without circular golden regeneration |
| C7 Reliability | Status/process failures prevent readiness; tests, default test guards and raw path protection pass |
| C8 Environment | Reproducible CPU setup and explicit V2 path/config resolution pass |
| C9 Training | Fresh seed-42 fixed depth_4 training completes; recorded checkpoint/count/history; no test access |
| C10 Freeze | Validation-selected threshold and complete identities fixed before test inference |
| C11 Evaluation | One V2 checkpoint scored once; reused-test disclosure, group intervals and flag/family results present |
| C12 Export | Self-contained single-CNN package, fresh-process loading and frontend/model software checks pass |
| C13 Delivery | Current contract/report and task-owned code/tests published; local archives/data/weights excluded from Git |

Stop at the failing gate if archive integrity fails, a source artifact changes unexpectedly, raw paths are targeted for writes, split membership drifts, a new producer executes tabular extraction, required parity fails, a model identity changes during resume/evaluation, or test information is used to change the protocol. Preserve failure evidence. Do not relax tolerance, silently change counts, or train another model to obtain a pass.

Before the training run, commit the V2 protocol, preparation/model code and required tests after software checks pass. After training/evaluation/export, publish the final task-owned documentation and remaining verified code on the current branch using the existing GitHub remote. Never push ZIP archives, generated datasets, raw audio, checkpoints, local environments, private records, credentials, or unrelated work. Use a scoped commit; do not force-push or replace remote history.

- [x] V1 archive and restoration verification complete.
- [x] Corrected fresh CNN-only data package complete and persistent.
- [x] One fixed 60,641-parameter CNN trained from scratch.
- [x] Validation threshold and model/data identities frozen.
- [x] One V2 test pass and honest benchmark report complete.
- [x] One self-contained CNN V2.0 export verified.
- [x] Task-only implementation and results documentation committed and pushed.

# Data preparation correction implementation plan

Date: 25 November 2026. Version: 1.0. Status: implementation specification; all correction work and acceptance gates below are pending.

Make the existing data preparation implementation reproducible and its completion claims reliable, while preserving the frozen experiment. The installed features, split, and normalization have strong numerical evidence. The work should repair the raw inventory failure, provide a persistent build workflow alongside verification, enforce explicit completion criteria, cover preparation entry points with meaningful regression checks, and document the actual model input and environment contracts.

The governing assessment is [Data preparation implementation and readiness audit](../DATA_PREPARATION_AUDIT_2026-11-25.md), published in commit `40fb9e5`. Repository state was checked again at `9ec4ce0` while writing this plan. New detector integration files and a runtime requirements file are also present in the working tree; they belong to separate ongoing work. This plan defines future changes and does not represent them as implemented.

Canonical plan: `docs/DATA_PREPARATION_CORRECTION_PLAN.md`. An identical reading copy belongs in the companion Data reports folder as `Implementation plan data preparation corrections V1.0.md`. Repository paths below are relative to Functional Project. Generated datasets, execution evidence, and machine-specific configuration remain local.

**Scope and required outcome**

Implement the corrections in the audit without changing the scientific experiment. A successful result has a working raw inventory function; a documented persistent reconstruction path; a safe default verification path; accurate status and evidence handling; a reproducible environment for preparation and model consumers; and an installed-data contract that explains exactly what the models receive.

This work does not include new feature experiments, alternative cleaning, split optimization, normalization selection, model tuning, training, repeat final-test evaluation, browser UI changes, live microphone acceptance, or final CNN deployment packaging. Those activities have separate protocols or the existing local detector integration plan. Frontend golden verification is in scope because it validates preparation output; redesigning the deployment package is not required to close this plan.

**Frozen values and protected artifacts**

Preserve seed 42, folder-derived labels, approved groups, exclusions, fixed evaluation crops, and the selected frontend. The preparation corrections must not replace the existing dataset or reinterpret old identity hashes as new domains.

| Contract | Required value |
|---|---|
| Raw binary clips | 11,704: 10,372 unknown and 1,332 drone |
| Retained examples | 10,623: 9,291 unknown and 1,332 drone |
| Exclusions | 2 short clips and 1,079 additional exact-zero copies |
| Train, validation, test examples | 7,607; 1,510; 1,506 |
| Positive examples by split | 967; 185; 180 |
| All groups by split | 1,249; 247; 247 |
| Positive groups by split | 146; 29; 28 |
| Exact-zero representatives | 9, all in train |
| Pinned groups | 9, including extra Membo; 68 retained clips |
| Frontend | 16 kHz; channel 0; FFT/window 1,024; hop 256; periodic Hann; no centered STFT padding |
| Mel representation | 64 HTK area-normalized filters, 50–7,500 Hz, power 2 |
| Window | Nominal 0.5 s; 7,936 samples; 28 frames; usable duration 0.496 s |
| dB conversion | Reference 1, `amin=1e-12`, `top_db=None` |
| Normalization | `global_stats`, eligible training clips, independent floor-cell masking per band |
| Eligible training clips | 7,519 used, 88 excluded from statistics estimation |
| Eligible clip frames before cell masking | 452,765 |
| Tabular features | 84 float32 acoustic columns; no metadata columns |

Freeze hashes to verify before and after implementation:

| Artifact | SHA-256 |
|---|---|
| `data/interim/manifest_full.csv` | `bd2d9059ceda181b0877d9fb371f1d00760d057e2546da79873187832a625142` |
| `data/processed/splits/train.csv` | `60b14ff205bdd4e950ae0642d5ae886c8b8c4c291cabfedc26ee936d9a3e1afc` |
| `data/processed/splits/val.csv` | `c1c0170fddbd6dfa4d3095eab12abef6572ac45036278ba79cc4c71794dc23fc` |
| `data/processed/splits/test.csv` | `350356a0b89a84bf6cf77fc4a1ffa36dea662151fbd5d79e66989cdfeec14848` |

Also protect the source WAVs; installed inventory and manifest ledgers; split freeze record; feature/power/index files and their sidecars; original normalization archives and records; existing goldens and bundles; model checkpoints and metadata; and the existing final-test scores and lock. Record their digests before edits. Existing config path profiles may be copied and rebased into a separate build destination, but the installed configuration and transform values must remain unchanged.

Use a new evidence record for this correction. Do not overwrite the audit, earlier gate CSVs, old frontend reports, or the original model evaluation record to make them appear current.

**Finding coverage and order of work**

| Audit finding | Work package | Acceptance evidence |
|---|---|---|
| Missing `math` import breaks nonzero inventory | DP1 | Direct nonzero/zero WAV tests and an isolated real train inventory check |
| Current notebook 02 only builds temporary train/validation outputs | DP2 and DP3 | Persistent reconstruction command, explicit mode behavior, notebook integration |
| Notebook can finish with pending or missing closeout | DP4 | Process/status failure matrix and correctly scoped success |
| Unit suite misses preparation entry points | DP1 and DP5 | Synthetic pipeline regression and negative-path checks |
| Model environment is not described by root preparation requirements | DP6 | Separate environment contract and clean-environment checks |
| Historical documents appear to describe the current state | DP7 | Dated current contract and supersession pointers |
| Full-manifest, sidecar, and loader-enrichment descriptions are ambiguous | DP3 and DP7 | File/field catalog and loader handover check |

Implement DP0 before any mutation, then DP1, DP2, DP3, DP4, and DP5. DP6 and DP7 can follow once the interfaces are settled. DP8 closes the task after all required checks pass. These are dependencies, not instructions to delegate work to other agents.

**DP0 Record the baseline and enforce operating boundaries**

Primary files: the new preparation entry point and existing path/configuration helpers. Reuse `ensure_not_raw()`, cache checksum helpers, `FROZEN.txt`, and the existing identity migration record.

1. Record the current branch, commit, remote, staged changes, and unrelated working-tree changes. Preserve concurrent detector integration work, deleted report files, and untracked files. Do not assume an initially clean repository.
2. Create a correction-specific baseline with source digests, configuration digest, all four manifest hashes, cache/statistics/golden digests, and model/evaluation digests. The audit's 143 protected files are a starting reference, not a fixed count to copy into a new result.
3. Resolve all source and destination paths before creating output directories. Refuse writes inside the raw dataset, the installed protected dataset, model folders, or their descendants. Reject a build destination that aliases the source through a resolved path.
4. Find the evaluation lock in the authoritative source project, not just the proposed destination. A new empty output directory must not bypass the already-evaluated status of this experiment.
5. Record test access by operation: metadata inspection, opaque file checksum/copy, decoded WAV access, NumPy array access, or scoring. Do not claim that no bytes were read when a file was hashed or copied. The relevant protection is that verification never decodes or exposes held-out feature values and never scores them.

Acceptance: the baseline is saved under a unique local run identifier; path protection and test-access decisions are explicit; no protected artifact has changed.

**DP1 Repair the raw inventory function and stop on invalid inventories**

Files: `src/dronedet/data_preparation.py`, `src/dronedet/data.py` if a new validator needs exporting, notebook 00, and a new preparation test module under the existing `tests/unit/` directory.

1. Add the missing `math` import used by `clip_stats()`. Preserve PCM16 scaling, SHA-1 byte construction, RMS calculation, exact-zero sentinel, clipping threshold, and output field names. Do not replace the numeric recipe while fixing the import.
2. Keep per-file error records for diagnosis, but add a shared inventory validation step before parsing/cleaning or accepting success. It must reject any nonempty error field, missing required measured fields, unreadable/truncated input, duplicate identifiers, and invalid required numeric values. Exact-zero RMS uses the established finite sentinel rather than an artificial failure.
3. In notebook 00, run that validation immediately after `build_inventory()`, including the post-freeze training-only branch. A failed nonzero decode must stop before later cells write a success record or treat the inventory as usable.
4. Keep full-corpus initial audit behavior distinct from verification after freeze. For this correction's real-data acceptance, decode only the frozen training clips. Save the result to the new evidence destination and compare it with the corresponding rows of the installed inventory; do not replace `inventory.csv`.
5. Check that notebook 01 refuses an invalid inventory before grouping. Fix project-root/import resolution so documented execution from the project root or the notebooks folder works without relying on a previous notebook's `sys.path` state.

Required regressions: a deterministic nonzero PCM16 WAV produces finite RMS and the expected independently calculated sample statistics; an exact-zero WAV produces the existing sentinel and zero flags; a missing/corrupt input records an error and fails validation; a valid input cannot disappear because the inventory loop caught an exception. Test the public `dronedet.data` route used by notebooks as well as the underlying helper.

Acceptance: the previously failing nonzero probe passes; training-only inventory reproduction has no decode failures and agrees with recorded PCM identities and measured values; invalid input prevents downstream success.

**DP2 Add explicit verification and persistent build modes**

Proposed new entry point: `scripts/prepare_data.py`. Put reusable orchestration in existing preparation modules; keep the command-line wrapper thin. Reuse `load_project_config`, cleaning/grouping helpers, `write_split_cache`, validating loaders, `fit_cached_normalizer`, and bundle verification. Do not implement another frontend.

| Mode | Inputs | Output behavior | Real test behavior |
|---|---|---|---|
| `verify` | Existing frozen project and installed artifacts | Temporary train/validation reconstruction plus a persistent new evidence record | Metadata/checksums only; no decoded test audio, loaded test feature arrays, or scoring |
| `build` with archived test artifacts | Raw train/validation WAVs and a trusted frozen protocol/artifact source | Complete package in a new persistent destination | Copy the archived test cache/index/tabular/sidecars as opaque files and verify digests; do not recompute test values |
| Initial mechanical test-cache creation | Frozen protocol in a study that has never been evaluated, with an explicit preparation-only flag | Create the initial test artifacts before evaluation | Restricted to preparation; no model calls, threshold selection, plotting, statistics fitting, or scoring |

`verify` is the default mode. `build` must require an explicit output root and must never target the installed dataset. Existing files cause refusal; an exact completed package may be verified for reuse through an explicit option, but it is not silently rebuilt.

The current experiment already has a final-test lock. Its correction acceptance must use the archived-test route. An initial-test-build flag must reject this source project even when the destination lacks a lock. If the archived held-out artifacts are missing, report an incomplete restoration; do not silently recreate them from raw test audio. Any later change to that restriction requires a separate explicit protocol decision.

Distinguish a clean software installation from a new scientific experiment. Reconstructing this frozen experiment requires the private raw data and its trusted frozen protocol/reference assets. A Git checkout alone does not contain those assets. A raw-only reinitialization that establishes a new split or new golden expectations is outside this correction plan.

For each mode, define required inputs before starting work. A trusted frozen source must include the original manifest ledgers, all three split manifests and hashes, freeze record, identity mapping, reference normalization files, and native-rate golden expectations. The archived-test route additionally requires both test caches, index, sidecars, and the original lineage record needed by legacy loaders. Validate the complete set before expensive processing.

Acceptance: mode selection is explicit; the default is non-destructive verification; an evaluated experiment cannot enter initial test-cache creation; missing restoration inputs fail clearly before partial output is accepted.

**DP3 Build a complete persistent dataset package using canonical producers**

Files: `scripts/prepare_data.py`, existing preparation/configuration helpers, and existing cache metadata/feature functions where integration requires changes.

1. Create a staging directory next to the requested new destination. Bind the run to the source configuration, frozen manifest digests, code digests, environment, and intended final path. All processing uses an explicit path map rather than relying on the working directory.
2. Copy the verified frozen manifests and control records. Replay cleaning/group assignment and the seed-42 split in memory from the saved inventory to check the assignments. Do not select a new seed or rewrite the source manifests. Preserve the difference between the 11,704-row full ledger and the 10,623 retained examples.
3. Build train and validation power/index/tabular outputs through `write_split_cache()`. Load them back through the canonical validators. Require aligned IDs/labels/flags, contiguous spans, 84 finite columns, finite nonnegative power, canonical centered offsets, correct schema-3 sidecars, and a committed cache transaction.
4. Acquire test artifacts only through the route permitted by DP2. Opaque adoption retains the original file bytes and truthful provenance. Preserve required legacy lineage records rather than labeling a copied old cache as newly built schema 3. The package catalog must state if new train/validation files and historical test metadata use different supported schemas.
5. Fit normalization using the generated train cache and its stored boolean eligibility. Preserve `train_stats_eligible_per_band_floor_mask_v1`, all full-clip eligible train frames, the per-band −119.999999 dB threshold, and float32 output arrays. Compare refitted values exactly with the trusted frozen statistics before claiming compatibility.
6. Preserve the original normalization archives byte-for-byte in the restored package once their values have been reproduced. A newly compressed NPZ can have a different file digest despite identical arrays; existing model metadata binds the original statistics file. Put the new fit accounting in a separate run record rather than replacing that archive.
7. Copy and verify existing native-rate golden expectations. Do not generate expected values from the current code and present their immediate agreement as independent parity evidence. The current reference WAVs are 0.9-second FLOAT files at 16/48 kHz; preserve their actual format and hashes.
8. Validate the reference filterbank and both native-rate normalized golden outputs through canonical paths. Use the V2 frontend bundle for native-rate/model-window acceptance. Keep the historical V1 bundle's narrower scope explicit.
9. Materialize a destination configuration/path profile that resolves to the new package. Rebase only path fields in this detached copy; preserve transform/normalization identities and retain the original source configuration digest. Test loading after publication, not just while the staging directory exists.
10. Write a package inventory listing relative paths, roles, checksums, shapes/dtypes, metadata schema, provenance operation, and identity domain. Publish the directory only after all package checks pass and a completion record is written. An interrupted build remains visibly incomplete and cannot be treated as a ready dataset.

Required package content:

| Output group | Required files and checks |
|---|---|
| Audit/control | Inventory, full manifest, cleaning log, freeze record, manifest hashes and provenance/source references |
| Splits | `train.csv`, `val.csv`, `test.csv`; all four frozen hashes preserved |
| Tabular | Three `<split>.npz` archives; aligned identifiers and catalog; matrices `(7607,84)`, `(1510,84)`, `(1506,84)` |
| Power/index | Three NPY/CSV pairs; shapes `(64,457897)`, `(64,91761)`, `(64,91494)` |
| Metadata | Complete sidecars and any required transaction/legacy lineage records |
| Normalization | Approved original archives, policy record, eligibility and per-band floor accounting |
| Parity | Existing native 16/48 kHz golden expectations and verified V2 frontend bundle |
| Completion | Rebased paths, package file catalog, run evidence, and unambiguous completion marker |

Test dimensions and finiteness in the archived-test route come from trusted prior evidence and opaque checksum identity, not a fresh held-out array scan. Represent that evidence as reused, with its source digest. Do not mark it executed by the correction run.

Acceptance: outputs survive process exit; a consumer can load the published train/validation package through its documented path configuration; frozen assignments and float32 statistics agree; archived test artifacts retain their original digests; source artifacts remain unchanged.

**DP4 Enforce truthful completion and unique evidence output**

Files: notebook 02, `scripts/verify_frontend_reproduction.py`, `scripts/run_frontend_closeout.py` only if an explicit scope extension is needed, and the new preparation orchestration.

1. Keep notebook 02 as a thin verification entry point by default. Expose persistent build as a clearly separate action using the same orchestration. Avoid notebook-only global variables as the durable API.
2. Remove the reproduction script's dependence on `exec`-ing notebook cells and recovering `BUILD_SPLITS`, `OUTPUT_ROOT`, and `_output_directory`. Let notebook and script call one reusable verification function that returns a typed/structured result and owns temporary-resource cleanup.
3. Give every execution a unique report path. Refuse overwriting an existing record. Bind each report to the current command, code revision/source digests, configuration, input digests, versions, and scope. Record actual gates rather than literal passing values or copied historical labels.
4. Distinguish a preparation result from frontend hardware acceptance. Preparation success requires its data/software gates. A full frontend closeout still requires valid hardware evidence; preserve that runner's existing strict behavior. If notebook 02 invokes full closeout, it must pass the retained hardware-log path explicitly and require the full result to pass. Otherwise it should report a preparation-only result without claiming G15 acceptance.
5. Treat nonzero process exit, absent output, invalid JSON, missing required fields, unexpected status, and any required `FAIL`/`PENDING` gate as unsuccessful. Do not accept a stale report left by a previous invocation after a child process fails.
6. Use `PASS`, `FAIL`, `PENDING`, and scope explicitly. Record whether a check was freshly executed, reused after verifying bound evidence, or not applicable. A required check cannot become not applicable because it was inconvenient to run.
7. Record exceptions with the failing stage and exit nonzero. Close memmaps and temporary directories on Windows before publication or cleanup. Preserve useful failure evidence without leaving a ready marker.

Status contract:

| Event | Required behavior |
|---|---|
| All required preparation checks pass | Exit 0; `status=PASS`, `scope=preparation`; explicit hardware scope field |
| Child process fails but an old PASS report exists | Fail; reject the old invocation's report |
| Missing, malformed or incomplete report | Fail; no success message or ready marker |
| A required gate is pending | Nonzero exit; overall result cannot be PASS |
| Full frontend closeout lacks accepted hardware evidence | Remain pending/nonzero; do not weaken full closeout |
| Opaque test files adopted with verified digests | Mark adopted/reused; never claim fresh test recomputation |

Acceptance: tests exercise each status branch; notebook success accurately names its scope; no old evidence file is silently replaced.

**DP5 Cover preparation behavior and integration failures**

Use the existing `tests/unit/` directory, adding preparation tests there. The tests should catch the reported failures and cross-module errors, rather than repeating feature formulas already tested elsewhere.

| Test | What it must establish |
|---|---|
| Nonzero and zero raw inventory | The actual notebook-facing API handles both paths; the missing-import regression fails before the repair |
| Invalid input propagation | A corrupt/missing WAV cannot produce an accepted manifest or green completion report |
| Synthetic persistent package | Small generated PCM16 fixtures pass through inventory, cleaning/metadata, canonical cache writers, train-only fit and loader handover; files remain usable after the build process exits |
| Eligibility enforcement | Validation/test rows cannot fit statistics; excluded train rows stay in model examples and are omitted only from fit estimation |
| Split and identity tampering | Altered manifest, reordered IDs or features, wrong lineage, and missing provenance are rejected |
| Atomic/incomplete publication | An injected failure leaves no ready package and changes no preexisting artifact |
| Test-access policy | Default verification never requests real test payload loaders; evaluated-source lock blocks initial creation even with an empty destination |
| Opaque adoption | Copied artifacts keep source digests and adoption provenance; no feature decode or model call occurs |
| Process/result matrix | FAIL, PENDING, nonzero exit, stale PASS, absent JSON and malformed JSON all prevent false success |
| Working-directory behavior | Documented root/notebooks invocation paths resolve the intended project and imports |

Synthetic fixture counts and hashes belong to an explicit fixture contract. Do not weaken production count/hash gates to accommodate a small test dataset. Exercise synthetic test-cache creation through a dedicated fixture study with no evaluation lock, never by opening the real held-out arrays.

After these checks pass, run a single real train/validation verification under the approved environment. Reproduce the full train/validation arrays in isolation, rather than relying only on the audit's 20 clips per split. Refit normalization once and verify approved float32 arrays exactly. Reuse that result for related gates so notebook integration does not recursively launch the same full check.

Acceptance: new focused tests and the existing applicable suite pass; real isolated train/validation outputs match the installed values; no real test scoring or payload-based diagnostics were performed.

**DP6 Specify preparation and model environments**

Files: existing `requirements.txt`, a proposed `requirements-models.txt`, environment/checking documentation, and the new preparation command's preflight. Coordinate with the existing detector integration requirements rather than editing its untracked files as part of this task.

1. Keep the root preparation environment explicit: Python version, pinned audio/data/scikit-learn packages, and validation tooling where needed. Record the actual versions and platform in every new preparation report.
2. Define the model-consumer environment separately. It must include preparation imports and an installable CPU PyTorch specification consistent with the working environment. The audit recorded `2.15.0+cpu`; do not assume a local version label is available from an arbitrary package index. Verify the intended source, wheel availability for the target Python/platform, and installed build before documenting the install command.
3. A new `requirements-live.txt` is now present as part of separate detector work. Reconcile shared versions after that work is reviewed; do not create incompatible torch pins or duplicate an uncontrolled runtime lock. The training/model environment may have additional dependencies beyond the minimal live runtime.
4. Test the documented setup in a separate environment, preserving the working `.venv`. First validate imports and a synthetic `(B,84)` tabular batch and `(B,1,64,28)` CNN batch. Use model inference in evaluation mode only where this handover check requires it; never train or score real test data.
5. Record clean-install acceptance separately from numerical portability. If installation cannot be verified, retain that gate as pending. Same-Windows verification does not establish Linux/Colab compatibility.

Acceptance: preparation and model consumers each have a documented install route with the necessary imports and CPU behavior verified; the current working environment remains usable.

**DP7 Publish the operational data contract and historical pointers**

Add `docs/DATA_PREPARATION_CONTRACT.md` after interfaces settle. Keep the audit immutable. Add short dated supersession notes to companion reports only when implementation closes their findings; do not rewrite their historical measured results.

The contract must explain raw PCM16 format, folder-derived labels, exclusion counts, the full ledger versus retained split rows, the exact fixed frame center, and the current statistics policy. Include this consumer mapping:

| Consumer | Required input contract |
|---|---|
| Classical models and MLP | NPZ `X` is unscaled float32 `(N,84)`; fit scaler on eligible train rows, then transform all retained examples; labels `(N,)` |
| CNN | Select `(64,28)` power window from full-clip cache, transform with approved stats to `(1,64,28)`, then batch to `(B,1,64,28)` |
| Metadata/reporting | Loader enriches NPZ data with `stats_eligible`, family, type and recording fields from indices/manifests; these are not extra columns of `X` |
| Evaluation crops | Fixed center for validation/test; supplying an RNG is rejected; random offsets are train-only |

State that installed sidecars use supported historical metadata while new producers use schema 3/identity schema 2. Separate power, tabular, transform, normalized-input and decision identities. File digests and normalized numeric lineage are different checks.

Keep gain-invariance claims conditional on avoiding floor effects. Explain that per-band masked cell counts differ from the eligible full-clip frame count. Correct the archived benchmark interpretation: 14.825/2.952 is approximately 5.02 times the runtime, not 2.9 times.

Add current-state pointers for the resolved passband/environment-hash issues and the existing November 24 final evaluation. Do not present final test evaluation as pending or rerun it. Retain the scientific limitations: indoor/two-model scope, source artifacts, small independent family samples, extra Membo not evaluated, and crop/whole-clip label mismatch.

Acceptance: a model consumer can identify the correct files, shapes, preprocessing, metadata and mode without reconciling several historical plans; current and historical completion states are clearly dated.

**DP8 Final acceptance and publication**

The following gates are required to close the data preparation correction implementation. Their statuses must come from recorded checks, not from this plan's expected values.

| Gate | Required result |
|---|---|
| A1 Inventory | Nonzero/zero regression passes; invalid input stops the pipeline |
| A2 Protected artifacts | Source raw data, manifests, caches, statistics, goldens, models and final evaluation are unchanged |
| A3 Frozen protocol | All four manifest hashes and all structural split gates pass |
| A4 Mode behavior | Default verification is isolated; persistent build outputs survive exit; source evaluation lock cannot be bypassed |
| A5 Cache handover | Canonical writers/loaders round-trip, ordered rows/catalog/flags agree, publication is complete |
| A6 Numerical reproduction | Full train/validation power and feature arrays match exactly in the approved environment |
| A7 Statistics | Mean/std match exactly; 7,519 eligible, 88 excluded; per-band floor policy and counts agree |
| A8 Goldens | Existing 16/48 kHz log-mels within 1e-3 dB; normalized V2 tensors within 1e-6 absolute; no regenerated expectations |
| A9 Failure handling | Process/report failure matrix rejects false success; evidence paths are unique |
| A10 Environment | Documented preparation and model environments pass their stated installation/import checks |
| A11 Documentation | Operational data contract, supersession pointers and supported restoration procedure agree with implementation |
| A12 Delivery | Scoped changes committed/pushed after required checks; unrelated changes preserved |

Tolerance rules: same-environment array/statistics reproduction requires exact equality because that is the established baseline. Filterbank comparison retains identical nonzero support and relative error at most 1e-6. Cross-environment drift is reported against the existing golden limits, not accepted by loosening them. If a discrepancy appears, isolate its cause and affected consumers before considering any protocol change.

Stop implementation acceptance if a frozen digest changes, any new path writes into protected data, new caches disagree numerically, a legacy identity cannot be reconstructed, a gate needs real held-out payloads after evaluation, or an environment change lacks a verified installation/numerical check. Preserve the failing evidence and report the actual cause. Do not rebuild installed artifacts, relax tolerances, or retrain to remove the failure.

**Commands to expose and validation sequence**

The following preparation commands are proposed interfaces, not commands already available in the repository. Validate the implemented `--help` against this contract before using the runbook. Output paths use placeholders so each run can receive a new identifier.

```powershell
# Proposed default verification command after implementation
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_data.py verify `
  --project-root . `
  --output data/interim/preparation_<run-id>.json

# Proposed persistent restoration using the current frozen experiment
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_data.py build `
  --project-root . `
  --output-root <new-empty-destination> `
  --test-artifact-source <trusted-frozen-artifact-source> `
  --output data/interim/preparation_build_<run-id>.json
```

The ordinary verification command does not perform microphone acceptance or repeat model evaluation. Full frontend acceptance, if requested as a separate check, must use the existing runner with its validated hardware log and a new report path.

Implementation validation order is: focused preparation regressions; the existing applicable unit suite; one full isolated train/validation verification; a persistent synthetic all-split build and consumer reload; a real persistent reconstruction with opaque archived test adoption if the trusted source is available; final protected-artifact comparison; and document/command consistency review. Installation checks use a separate environment and do not replace the original environment.

Publish task-owned code, tests, dependency specifications, and documentation only after the required checks pass. Use the current branch and existing GitHub remote without force-pushing. Keep source audio, generated arrays, local evidence, model weights, credentials, personal paths, and unrelated changes out of the commit. Report the commit, push result, executed gates, any reused evidence, and any genuinely incomplete acceptance item.

**Definition of done**

- [ ] Raw inventory works for nonzero and zero audio, and invalid inventories cannot proceed.
- [ ] Verification and persistent reconstruction have explicit, documented interfaces over shared canonical code.
- [ ] Current evaluated test artifacts are protected and restored only through the allowed archive route.
- [ ] A completed persistent package survives process exit and loads through the documented consumer path.
- [ ] Notebook and script status handling rejects pending, stale, missing and failed required results.
- [ ] Preparation regressions and relevant existing tests pass.
- [ ] Full train/validation numerical reproduction, statistics and native-rate parity pass without changing frozen artifacts.
- [ ] Preparation/model environment setup and the operational data contract are verified.
- [ ] Historical reports point to the current correction result while retaining their original evidence.
- [ ] Task-only changes are committed and pushed, with actual verification results reported.

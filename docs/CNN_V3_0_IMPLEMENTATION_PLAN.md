# CNN V3.0 implementation plan

Date: 26 November 2026. Status: final planning specification. Archive creation, restore testing, V3 implementation, training, evaluation and deployment are pending.

Improve robustness to phone playback, rooms, microphone noise and low recording levels using the existing binary corpus. Preserve a complete, restorable V2 baseline first. Start with waveform augmentation and the existing CNN; allow saved training derivatives, frontend changes, fine-tuning and architecture changes through separately logged experiments. A successful simulation result does not establish performance on new microphones, rooms or drone types.

Canonical plan: `Functional Project/docs/CNN_V3_0_IMPLEMENTATION_PLAN.md`. Identical reading copy: `Project documents/Models/Implementation plan CNN V3.0.md`. All repository paths below are relative to **Functional Project**, the Git root, rather than its parent workspace. This plan supersedes the supplied proposed V3 plan. Section 12 explains the changes.

## 1. Baseline and scope

The frozen reference is CNN V2.0, depth_4, seed 42, 60,641 parameters. Its TorchScript SHA-256 is `0e97e3194e42f8fa3a2541e37dcff9fe33cdc1246d5a260ab65c6d157af6f4fb`; its raw-logit threshold is `2.823960542678833`. The frontend uses 16 kHz mono, 64 HTK area-normalized mel bands, **50–7,500 Hz**, a 1,024-sample FFT/window, 256-sample hop, no centering, and train-fitted global statistics. A window has 7,936 samples and produces `(1,64,28)`; batches have shape `(B,1,64,28)`. The live decision cadence is 0.496 s with a three-window majority and the existing standby/reset policy.

V2's saved replay found zero threshold crossings in 80 windows from four microphone WAVs, with logits from −8.256 to −0.905 and whole-file RMS around −45.2 to −47.1 dBFS. Lower validation thresholds at the 1% and 5% FAR operating points also missed all 80 windows. A labeled validation Bebop clip passed the direct digital streaming control. These findings support investigating acoustic-channel coverage; they do not establish the cause of the phone-playback failure.

The supplied proposal additionally reports a 99-window browser trace, maximum −3.002, median −6.796, browser processing reported off and no drops/resets. Its original JSON was not located during this inventory. Preserve these numbers as **reported in the supplied proposal**, pending the source JSON. Browser settings do not rule out Windows or device processing. The earlier brief detection remains unaligned with the saved evidence.

The owner describes drone presence throughout the original four recordings. Earlier reports recorded their event labels as unconfirmed. Preserve both the owner's description and that provenance difference; do not invent sample-accurate annotations. A fifth microphone WAV now exists locally and has no established role in the earlier four-file replay.

V2's trainer used random frame-aligned cropping and ±12 dB gain, without dedicated speaker, room or noise-mixing augmentation. The source dataset itself already contains artificial noise mixtures. Do not describe it as having no noise or room content at all. Negative-class clipping, DC and zero-padding are plausible shortcuts, not demonstrated explanations of this failure.

The following boundaries apply:

- Keep the same original source membership, labels, reviewed exclusions and group assignments. There are 11,704 original binary clips, 10,623 retained clips, and 7,607/1,510/1,506 train/validation/test clips. Positives number 967/185/180; negatives number 6,640/1,325/1,326.
- Use one source dataset and remain under the project's 1 GB data limit. Use 1,000,000,000 bytes as the conservative budget for source binary WAVs plus persisted derived audio. Report caches, checkpoints and backups separately. The pre-existing V1 backup being larger than 1 GB does not make it a second training dataset.
- Use the existing local Windows/CPU workflow. Record any later hardware or execution change. Library-first still applies; no external pretrained network or new training corpus enters this plan. A warm start uses only this project's own V2 weights.
- No recording, trace, test clip or validation clip enters training or normalization fitting. Noise donors come from eligible training negatives only. Downloaded impulse responses or noise libraries would add source data and are outside this scope; use procedural room responses instead.
- V3 outputs have new paths and identities. Do not overwrite V2 caches, weights, statistics, thresholds, locks or packages.

## 2. Archive inventory checked against the project

### 2.1 Existing backup and current gaps

`archive/Models V1.0.zip` already exists with its `.zip.sha256` sidecar. Its outer size is **1,847,016,558 bytes**. It contains 703 payload files plus `ARCHIVE_MANIFEST.json`, under `data/`, `results/` and `runtime/`. It is a V1 artifact backup, not the complete V2 baseline. It excludes the raw corpus and source code. Preserve it unchanged, verify its sidecar and member manifest again before relying on it, and retain it as a separate backup rather than nesting it in a new ZIP.

The actual V2 prepared-data root is **`results/cnn_v2/data/`**, not `data/cnn_v2/` or `data_manifests/`. Generated data and runtime packages are ignored by Git. A Git tag alone cannot restore them. There is no separate V2 operating-point CSV: the authoritative V2 threshold and alternative operating points are in `training/freeze.json` and `evaluation/report.json`. `results/tables/operating_points.csv` belongs to the older multi-model study.

The active V2 runtime selection is **`config/live_detector.json`**. `runtime/assets/manifest.json`, `runtime/assets/metadata/model_selection.json`, `runtime/assets/models/` and `runtime/assets/bundle/` describe the preserved V1 seed-44 runtime. Do not use those legacy files to identify or restore the active V2 model.

The current `config/cnn_v2.yaml` has SHA-256 `0cf5f934cd1e9a736d87f786132dc422dc7a9b541d4c73e7ba42a30e87fd7998`, while the training freeze and ready marker bind `e4aef0b2ba55c2c5b0d10db5b1dcf36342d8562b5472703215e64d4c855e0bee`. The difference was resolved during this review: restoring the original absolute raw-data path and removing the later portability comment reconstructs the exact frozen bytes. That hash-matching copy is preserved locally at `archive/CNN_V2.0_recovered_provenance/config/cnn_v2.training-frozen.yaml`. Keep both versions in the source archive, label the current one as the portable configuration, and use the frozen copy to explain the original training identity. Do not overwrite the current YAML or edit the historical freeze to match it. The recovered copy contains a local filesystem path and stays out of GitHub.

### 2.2 Create these separate archives before V3 implementation

The names below use the inventory date. If the freeze occurs later, use its actual date and record it in all manifests. Size figures are current uncompressed payload sizes, not promised ZIP sizes.

| Archive | Required contents | Current size or reason |
|---|---|---|
| `archive/CNN_V2.0_source_20261126.zip` | The V2 source snapshot, relevant current working-file changes, configuration, tests, requirements, notebooks, documentation, source revision and environment records described below | Supplies the code needed to reproduce and run the artifacts |
| `archive/CNN_V2.0_artifacts_20261126.zip` | The explicit V2 results/runtime allowlist below and the original V1 test lock referenced by V2 | Approximately 172 MB before source/provenance additions |
| `archive/DroneAudioDataset_binary_frozen_20261126.zip` | The original `Binary_Drone_Audio/unknown/` and `Binary_Drone_Audio/yes_drone/` trees, source `README.md`, `LICENSE.md`, and an inventory of every raw WAV with SHA-256 | **11,704 WAVs; 384,021,184 bytes** of WAV files |
| `archive/CNN_V2.0_diagnostics_private_20261126.zip` | The five local microphone WAVs, the two replay/control CSVs, the derived reference WAV, source trace JSON if located, and provenance/label notes | Known local payload: **9,776,788 bytes**, before notes and any recovered JSON |

The raw dataset currently resides outside the repository. Resolve it through `DRONE_AUDIO_DATASET_ROOT`, then check the `Binary_Drone_Audio` child. Preserve relative dataset paths in the ZIP and document how to set the environment variable after restoration. Do not archive the full parent download: its binary and multiclass trees together have 23,408 WAVs and duplicate the binary task's scope. Keep source URL `https://github.com/saraalemadi/DroneAudioDataset`, the license, and the observed source revision if available; otherwise record revision as unknown and use the per-file hashes as the content identity.

**Raw audio is required for a complete backup.** An inventory, SHA-256 and URL are useful provenance, but cannot replace the WAVs or guarantee that a future download has the same bytes. A backup without them must be labeled incomplete for retraining.

### 2.3 Exact V2 artifact allowlist

Include these paths with their repository-relative structure:

| Path | What must be preserved | Checked local payload |
|---|---|---:|
| `results/cnn_v2/data/` | All 12 files: ready marker, three mel-power arrays, three indices, three split manifests, `normalization.json`, `normalization_stats.npz` | 168,494,072 bytes |
| `results/cnn_v2/training/` | `cnn_v2_best.pt`, `checkpoint_epoch.pt`, `freeze.json`, `history.json`; these include best weights and resume evidence | 1,284,849 bytes |
| `results/cnn_v2/evaluation/` | `scores.csv`, `report.json`, `scoring.lock.json` | 160,337 bytes |
| `results/cnn_v2/export/CNN_V2.0/` | All 16 package files, including `PACKAGE_MANIFEST.json`, TorchScript, statistics, metadata, model card, software verification and native-rate golden fixtures | 554,559 bytes |
| `runtime/assets/CNN_V2.0/` | All 16 currently staged runtime package files, even though they match the export; retain evidence of what was deployed | 554,559 bytes |
| `results/cnn_v2/frontend_verification/` | Six evidence files, including desktop/mobile screenshots and browser review/logs, after checking for private session details | 978,489 bytes |
| `results/cnn_v2/verify_frontend_browser.py` | The local verification driver, absent from a source-only Git export because `results/` is ignored | 3,842 bytes |
| `results/tables/test_scored.lock` | The historical V1 test lock referenced by V2's lock; preserve exact bytes | 659 bytes |
| `data/interim/inventory.csv`, `data/interim/manifest_full.csv`, `data/interim/manifest_with_split.csv`, `data/interim/cleaning_log.csv`, `data/interim/decision_log.md` | Original inventory, cleaning, grouping and decision provenance used by the preparation workflow | Explicit files only |
| `data/processed/splits/` | `train.csv`, `val.csv`, `test.csv`, `FROZEN.txt`; required V1 split references for reproducing V2 preparation | Explicit files only |

Also preserve the raw inventory identity recorded in V2's ready marker: `cdfa077215c174ba3a88ce89f4095908685dcc4b1b3aa43e7d38a243d581bf24`. This identifies the source inventory recorded by V2; it is not a substitute for hashing the current raw WAVs. If a current provenance file differs from the state bound by the freeze, retain both versions with their roles identified rather than silently substituting one for the other.

The model-file hash was checked in both export and runtime locations during this review and matches the V2 reference in section 1. All package members still require verification at the actual archive freeze.

### 2.4 Source and environment contents

Snapshot the relevant current versions of `src/dronedet/`, V2 preparation/training/evaluation/export scripts, local-detector staging/server/launcher scripts, associated audio inspection and acceptance helpers, `tests/`, `notebooks/`, `demo/`, `live/`, `start_detector.bat`, `config.yaml`, `config/cnn_v2.yaml`, `config/live_detector.json`, applicable requirements files, `.gitignore`, the preparation audit and relevant `docs/`. Include `docs/CAPTURE_RECOVERY_FIX.md` and other untracked files only when they belong to the baseline being preserved. Include the companion `PROJECT_CONTEXT_V8.md`, V2 implementation-plan reading copy, relevant data/front-end plans and this V3 plan under a documented `project_documents/` prefix.

Record Git HEAD, branch, remotes, dirty status, the selected source-file manifest and its hashes. A Git source ZIP at HEAD is insufficient when required code has uncommitted changes. Preserve selected working versions and a record of their differences. Do not commit, stash, reset or include unrelated work just to manufacture a clean freeze. The current tree contains unrelated modifications/deletions; the archive builder must identify the selected baseline explicitly. An optional Git bundle supplements the snapshot; it does not replace selected working files or ignored artifacts.

Include the recovered training YAML from section 2.1 under `provenance/config/cnn_v2.training-frozen.yaml` in the source ZIP, alongside the current `config/cnn_v2.yaml`. Its reconstructed bytes match the original config digest; preserve that evidence and the reconstruction note. Set a portable raw-root override when running restored code. A rebuild using a relocated source or newly written config may have a different config identity even when the numerical recipe is identical; do not rewrite old dataset/freeze identities to conceal relocation.

Capture Python version/executable, Windows/platform, package versions, `pip freeze`, `pip check`, Torch version, thread count and CUDA availability. Preserve `requirements-cnn-v2.txt` and the live/dev requirement files separately from the freeze output. Record the PyTorch CPU index/install recipe already documented in `requirements-cnn-v2.txt`. An installed-package list is environment evidence, not a guarantee that all wheels will remain downloadable; a wheel cache is an optional separate offline backup. Do not ZIP `.venv/` as the environment restoration method.

The V2 training entry point is `scripts/train_cnn_v2.py`. The four existing notebooks do not constitute a V2 training notebook; retain them as preparation/history evidence without inventing a notebook that produced the V2 model.

### 2.5 Private diagnostic allowlist

Preserve these five `runtime/recordings/` files in the private diagnostics ZIP:

- `microphone_20261125_215246_4ffd885f.wav`
- `microphone_20261125_215302_f1361f85.wav`
- `microphone_20261125_215321_68533d41.wav`
- `microphone_20261125_215348_8025424d.wav`
- `microphone_20261126_080355_e7e848a7.wav`

Each is a 10-second, 48 kHz mono FLOAT WAV, 1,920,080 bytes. The first four are the historical replay set; the fifth is an additional diagnostic, not part of the reported 80-window result. Also preserve:

- `runtime/diagnostics/cnn_v2_saved_window_trace.csv`
- `runtime/diagnostics/cnn_v2_validation_bebop_trace.csv`
- `runtime/diagnostics/training_drone_reference.wav`, labeled as a derived corpus reference rather than a new independent recording
- The downloaded session-trace JSON described in the proposal, when recovered, with its original filename and SHA-256. It was not found in the inspected project or attached JSON files; record it as missing until supplied or located.

Add a diagnostics index listing file hash, source, recording settings, known playback/label information, historical use and missing metadata. Keep private recordings, traces, local filesystem paths and archives out of GitHub. Their archive destination may remain under ignored local storage or outside the repository; never stage the archive directory indiscriminately.

## 3. Freeze and restore procedure

Implement the archive operation as a dedicated allowlist-based tool, such as a new `scripts/archive_cnn_v2.py`. The existing `scripts/archive_models_v1.py` is a pattern for member manifests and hashes, **not** a V2 command: its roots select the older artifacts. This plan does not assert that the new tool already exists.

1. Inventory the selected source, artifacts, external raw corpus and diagnostic files. Stop V2 processes that could write checkpoints or diagnostics while copying; leave unrelated working changes intact. Record missing evidence explicitly. Missing source trace JSON does not prevent software restoration, but it prevents claiming that diagnostic evidence is complete.
2. Create archives from a stable staging snapshot or detect and reject files that change during reading. Use repository-relative paths and an explicit prefix for external dataset/document files. Reject unsafe paths and symlink/junction escapes. Exclude the output archive itself.
3. Embed `ARCHIVE_MANIFEST.json` in each ZIP. For every payload member, record path, byte size, SHA-256 and role. Record source revision, dirty-source coverage, included/excluded roots and missing items. Write to a new `.partial` destination, verify it, then publish under the final name; refuse to overwrite an existing freeze.
4. Prefer Python `zipfile` with `allowZip64=True` and deliberate compression settings. Do not rely on the draft's broad `Compress-Archive` command: it references a nonexistent `data_manifests` directory, omits source/raw/active configuration, and risks selecting the wrong folder level or incidental state.
5. Write a SHA-256 sidecar for every ZIP and a small backup-set index linking the four new ZIPs and the preserved V1 ZIP. Hash all members and run ZIP CRC checks. A read-only file flag can discourage accidental edits but does not make the backup immutable.
6. Extract the source, artifact and raw archives into a fresh temporary directory. Keep private diagnostics in a separate restoration folder. Compare every extracted member's size/hash against its manifest, including all test-cache bytes as an integrity check only. Verify the export and runtime TorchScript hashes against section 1.
7. Restore the documented environment in a fresh environment, set `PYTHONPATH=src`, configure `DRONE_AUDIO_DATASET_ROOT`, verify the package, load the model, and run the focused software checks below. Capture command, exit status and environment in a restore report. Do not invoke the test evaluator, score the locked test or start training during restoration.
8. Confirm that `config/live_detector.json` loads V2 seed 42, threshold 2.823960542678833, the correct hash and three-window policy. The copied V1 legacy runtime manifest must not select the active model.
9. Record `*.verification.json` beside each ZIP and an overall result distinguishing software restoration, raw-data restoration and diagnostic completeness. Copy the ZIPs, sidecars, manifests and reports to a second physical device or independent remote location and verify hashes there. A second folder on the same drive is not an independent backup.
10. Only after successful restoration, create an annotated baseline tag such as `cnn-v2.0-frozen-20261126` on the appropriate source commit. If baseline code has selected uncommitted changes, record that the tag alone is incomplete and bind the working-source ZIP hash in the freeze record. Do not mislabel unrelated HEAD contents as the complete runtime baseline. Begin V3 implementation on a separate branch when that work starts.

Focused restored-project checks, run **inside the restored project**, using its fresh environment:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m pytest tests/unit/test_live_detector.py tests/unit/test_live_launcher.py tests/unit/test_cnn_v2_protocol.py -q
node --test tests/browser/*.test.mjs
.venv\Scripts\python.exe scripts/prepare_local_detector.py
```

Exclude `.git/`, virtual environments, package caches, browser profiles, cookies, credentials, session ownership tokens, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `results/cnn_v2/pytest-tmp/`, incidental logs, unrelated recordings and unrelated generated report sites. Do not ZIP all of `results/`, `runtime/`, the workspace or Downloads. Preserve the V1 ZIP separately. No archive creation or restore test was performed as part of finalizing this plan.

## 4. Preregistration and output layout

Use the next unused decision-log IDs after inspecting the current log. `PROJECT_CONTEXT_V8.md` already contains D044–D058, so **do not restart at D044**. Use a V3 namespace such as `V3-001` if a shared next ID cannot be reserved unambiguously. Log the hypothesis, archive identities, experiment graph, parameter ranges, training budgets, threshold rule, selection rule, guards and final-test authorization before fitting.

Keep the following versioned layout:

```text
config/cnn_v3.yaml
requirements-cnn-v3.txt
src/dronedet/cnn_v3/augment.py
src/dronedet/cnn_v3/dataset.py
scripts/archive_cnn_v2.py
scripts/prepare_cnn_v3.py
scripts/train_cnn_v3.py
scripts/evaluate_cnn_v3.py
scripts/export_cnn_v3.py
results/cnn_v3/data/
results/cnn_v3/experiments/<run_id>/
results/cnn_v3/evaluation/
results/cnn_v3/export/CNN_V3.0/
runtime/assets/CNN_V3.0/
```

These are implementation targets, not existing files. Reuse canonical audio I/O, resampling, mel extraction, normalization and CNN/metric code. A shared frontend change must be selected by a versioned config and preserve V2 behavior; adding a V3 feature must not silently change the default V2 contract. New preparation/training verification must retain the V2 test-access guards.

Each run record binds source commit/selected source hash, parent configuration, data/split/statistics hashes, dependency lock, seed, augmentation config hash, sampler/loss policy, training budget and complete metrics. Store optimizer, scheduler and RNG/resume states and timings. Resume only a matching configuration; rerun under a new ID after a material change.

## 5. Channel augmentation and label checks

### 5.1 Processing order and clean-data exposure

The physical path is source codec → phone speaker response/distortion → propagation and room → microphone/background/electronics → capture codec or processing if present → canonical resampling/frontend. Source-codec and capture-codec simulations occupy different positions; record which one is used. All source clips here are already 16 kHz, so simulation cannot reconstruct missing high-frequency content or faithfully reproduce the entire native 48 kHz path.

For the main channel runs, retain a 20% branch using the V2 random aligned crop and ±12 dB gain recipe. Apply the channel branch to the other 80%, with the conditional stage probabilities below. Apply both branches independently of label. This preserves exposure to the original distribution while broadening channels; it is part of the registered recipe, not a parameter tuned on the recordings.

Augment full clips with adequate filter/reverb context, then select a 7,936-sample crop, then compute the versioned frontend. Applying a transform to the full clip does **not** automatically eliminate its startup/ending transients from a random crop. Carry proper filter state/context where available, define padding/tail handling for short clips, and check boundary versus interior windows. Do not use circular FFT filtering or hard looping that creates artificial wrap boundaries.

### 5.2 Initial channel ranges

These are fixed starting settings to calibrate with **training-only** energy/survival checks before the first fit. Record any resulting adjustment before freezing the validation recipe; do not repeatedly tune ranges against canary outcomes.

| Stage | Conditional probability | Starting range and implementation requirement |
|---|---:|---|
| Speaker band limit | 0.7 | HP 150–600 Hz, LP 3,500–7,500 Hz; log-uniform cutoffs, explicit filter order and measured response |
| Smooth EQ | 0.5 | Six smooth control points, ±6 dB; use a proper filter response rather than circular FFT wrap |
| Speaker soft clipping | 0.3 | Drive 0–18 dB; retain audibility and record peak/RMS change |
| Gain | 1.0 | −30 to +6 dB; any coupled distance low-pass is a separate logged option, not an unmeasured physical law |
| Synthetic room | 0.5 | RT60 0.2–1.0 s, DRR −5 to +10 dB; normalize direct and reflected energy explicitly |
| Background mix | 0.6 | Target added-noise SNR 0–30 dB; procedural white/pink/brown or eligible training-negative donor |
| Microphone floor | 1.0 | Sample absolute −75 to −45 dBFS independently of label; account for it in total added-noise SNR |
| DC offset | 0.3, late ablation | Symmetric, bounded, label-independent amplitude; ±0.01 is a maximum trial range requiring low-level audibility/DC checks |
| Hard clipping | 0.1, late ablation | Explicit threshold 30–90% of peak; log fraction clipped and prevent a second unintended final saturation |
| Codec | 0.3, optional late ablation | Specify exact codec, supported rate, quality/bitrate, latency and output alignment; no undefined “medium to low” setting |
| Compressor/limiter | 0.3, optional late ablation | Threshold −30 to −15 dB, ratio 2–4; specify attack/release/makeup gain and stage position |
| Pitch/time stretch | Optional separate ablation | ±1 semitone and ±5%; preserve duration/crop validity; these do not create a new drone type |
| SpecAugment | 0.5, separate rung | At most six frequency bins and four time frames, after normalization, zero fill in normalized space |

Implement library-supported transforms with NumPy/SciPy and existing audio libraries first. Add and pin optional libraries only for a selected rung; torchaudio, pedalboard and room libraries are not implied existing dependencies. Do not copy the proposed pedalboard call without checking installed API/rate support and an actual impulse/alignment test.

### 5.3 SNR, artifacts and noise-bank rules

Construct the training noise bank only from training non-drone clips. Reject exact-zero and below −70 dBFS donors, remove donor DC, and preserve donor clip/group/hash in each augmentation record. Use valid random offsets including the last legal offset; handle short donors with smooth joins or other documented boundary handling, not `np.resize` repetition.

Define SNR using the transformed positive component and the **sum of added background and microphone-floor components**, after gain/channel effects and on the actual scored crop. A target background SNR of 0 dB plus a further noise floor can produce total SNR below 0 dB. Reduce/resample the nuisance draw, or retain the unaugmented source, when an eligible positive would violate the floor; log how often this occurs. Do not discard or relabel the existing near-silent positive clips, and do not claim that a zero-dB floor guarantees recognizability. Existing source mixtures have unknown original SNR; report the constraint as **added-noise SNR**, not the true total acoustic SNR.

The microphone floor must not be capped in a label-dependent way or always track the signal at exactly −6 dB, which can create a new cue. Use the same sampling policy for both classes, then enforce the explicit survival constraints. Adding floor noise can reduce exact-zero cues but cannot erase all duration/padding/clipping artifacts. Retain artifact-slice evaluation and compare DC removal as its own frontend change.

Use procedural nuisance signals and separate deterministic seeds for augmented validation/test. Do not mix recorded training donors into held-out evaluation waveforms: that would introduce familiar audio and complicate group provenance. The evaluation room/noise recipe is independent of the candidate's enabled training transforms.

### 5.4 Checks before training

- Check mel/spectral survival across individual training positives and families under the most severe settings, within the actual 50–7,500 Hz frontend band. Fix a minimum energy/spectral-retention criterion from training measurements before fitting; an average spectrum alone can hide failure on a weak family.
- Listen to 20 augmented training positives and 20 negatives; inspect spectrograms, RMS/peak/DC, clipping, added-noise SNR and crop-boundary behavior. Log failures and calibrate before freezing the recipe.
- Verify finite float32 output, length/rate, valid crop size, determinism, silence/near-silence handling and no unintended final saturation. Check actual frequency response, achieved DRR/decay and achieved noise level, not just range bounds.
- An exponential synthetic response is a coarse simulation. Construct `h = direct_impulse + scaled_tail` with the tail's zero-lag handling explicit; replacing its first sample after energy scaling does not enforce the requested DRR. Use context/padding and a documented truncation policy.
- Avoid a large constant DC offset dominating a weak drone or driving a limiter while carrying no useful drone energy. Recheck added-noise/spectral survival after nonlinear stages.

## 6. Dataset integration and saved derivatives

The V2 training loop reads cached mel power. V3 waveform augmentation requires a raw-waveform loader that reproduces the original retained split and preprocessing, not augmentation of the cached mel array alone.

First create a no-channel raw-loader control. Match V2's valid **frame-aligned** training crop offsets, centered deterministic validation/test crops, gain semantics, short-clip handling and normalization. Validation/test must never use the proposal's unconditional `random_crop`. Do not assume a `features.to_model_input` helper exists; reuse or factor the actual canonical frontend and prove numerical parity.

Use a stable seed derived from run seed, epoch, source clip ID, derivative ID and draw/occurrence ID. A seed based only on `(seed, epoch, index)` gives identical augmentation to repeated weighted-sampler draws in one epoch. Seed NumPy, Python and Torch/masking generators; preserve the sampler's state. Begin with `num_workers=0`, then add workers only with demonstrated reproducibility and Windows-safe entry points. A parent assignment to `dataset.epoch` does not update persistent worker copies automatically. PyTorch documents separate worker replicas and cross-library RNG considerations in its [data-loading guidance](https://docs.pytorch.org/docs/2.15/data.html).

Use on-the-fly augmentation by default. Saved copies are an optional storage/reproducibility experiment using the same versioned transform engine, not a prerequisite for V3:

- Suggested fixed trial: four derivatives per training positive and one per training negative. This creates **10,508 additional WAVs**; originals plus copies total **18,115**, comprising **4,835 positives and 13,280 negatives**. They remain 7,607 original training clips and 1,249 original groups.
- Using actual train sample counts, 16-bit mono WAVs with unchanged lengths and 44-byte headers require approximately **342,587,944 extra bytes**. Raw binary audio plus these derivatives is approximately **726,609,128 bytes**. Recompute actual sizes, including manifests and any context/tails, before rendering; stop before the 1 GB budget is exceeded.
- Persist below `results/cnn_v3/data/augmented/train/` with a manifest containing original clip/group/split, source SHA-256, derivative ID, run/config hash, seed, all parameters, donor identity if any, output SHA-256 and measured levels/SNR. Reload PCM16 copies through the canonical loader so quantization is part of the experiment. Do not compare them with unquantized float data while attributing all effects to persistence.
- Apply crop and optional SpecAugment during training; do not apply another channel chain to already augmented saved copies. Any extra chain requires a separately registered experiment.
- Keep all derivatives in training. Verify source groups, hashes and ready markers against frozen manifests; no source or derivative may cross into validation/test.
- Compare with on-the-fly augmentation at matched optimizer-step and class-exposure budgets. More copies otherwise change epoch length and training effort as well as the augmentation method.

Baseline loss is `BCEWithLogitsLoss(pos_weight=6640/967)` with ordinary shuffled original sampling. The suggested saved set changes its class ratio to `13280/4835`, about 2.75. Recompute class weighting for the selected exposure policy, or retain the original prior through explicit per-source sampling weights. Do not silently keep the old approximately 6.87 positive weight after oversampling positives. Do not combine a balanced sampler with inverse-frequency positive loss weighting by default. These are separate weighting choices; PyTorch's [loss documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BCEWithLogitsLoss.html) defines the effect of `pos_weight`.

A weighted sampler can match class exposure without stored audio. It does not provide the same fixed waveform diversity as saved derivatives, and repeated draws need distinct augmentation seeds. Treat storage, class exposure and loss weighting as distinct factors when interpreting results.

## 7. Experiment graph and frontend options

### 7.1 Ordered main experiments

Each row has an explicit parent and adds one named group. These are grouped ablations, so they attribute a result to a group, not to each transform inside it. Split a useful group further if that distinction matters. Archive/restore completion and the preregistration record precede every row.

| ID | Parent | Change |
|---|---|---|
| A0a | Frozen V2 | Verify loaded checkpoint/golden/validation-score parity and reproduce the original cached-power V2 training recipe, including clean-validation-loss stopping. No test scoring. |
| A0b | A0a | Raw-waveform loader with matched offsets, gain and the original clean-validation-loss stopping; no new channel augmentation |
| A0c | A0b | Change only the stopping/scheduler monitor to the fixed weighted validation loss used by the rest of V3; no new channel augmentation |
| A1 | A0c | Wider gain, with the fixed 20% V2-recipe branch |
| A2 | A1 | Speaker band limit, EQ and soft clipping |
| A3 | A2 | Background mixing and microphone floor |
| A4 | A3 | Synthetic room response |
| A5a | Best eligible A1–A4 | DC and clipping stress, applied to both classes |
| A5b | Best eligible parent | Optional codec/compressor group; compare against its exact parent without it |
| A6 | Best eligible A1–A5 | SpecAugment |
| A7 | Best eligible A configuration | Optional pitch/stretch ablation |
| A8 | Best eligible A configuration | Saved derivatives versus on-the-fly augmentation; matched exposure and step budgets |
| B1 | Best eligible A configuration | Window DC removal |
| B2 | Same A parent; also test B1+B2 only if registered | Scalar per-window log-mel mean subtraction |
| B3 | Registered A/B parent | Train-only re-estimation of normalization statistics, including paired with/without B2 if registered |
| C1 | Finalist with unchanged tensor topology | V2-weight warm start versus fresh initialization |
| C2 | Best eligible A/B/C1 configuration | Optional small registered architecture or recipe grid, one factor/group at a time |

Record the parent when it is chosen. If a later rung fails, do not silently continue as if it improved the chain; subsequent rows branch from the best eligible configuration as stated. All candidates use the same frozen evaluation suite. Keep optional extensions bounded to the configurations written before their results are inspected.

A0 separates saved-inference reproduction, loader refactoring and the changed validation monitor so none is mistaken for an augmentation gain. Match cached versus raw-loader tensors at identical aligned offsets before comparing trained models: require normalized-tensor maximum absolute error at most 1e-4 and logit error at most 1e-4, with identical frozen-threshold votes. Reproducing the original cached training in the same pinned environment should reproduce its deterministic result; allow at most 0.005 absolute clean-validation F1 difference, one positive decision and one negative decision of count difference as the reproduction guard. Archive both score sets and explain any nonzero difference. A0c is a new control, not a claim to reproduce the original stopping outcome. If A0a or A0b fails its parity/reproduction requirements, resolve the refactor or environment difference before interpreting augmentation gains.

### 7.2 Training and initialization

Use the V2 architecture and recipe for the main A ladder: batch 64, AdamW 1e-3, weight decay 1e-2, the recorded ReduceLROnPlateau settings, maximum 100 epochs, early-stopping patience 8. A0a/A0b retain the original clean-validation-loss monitor. From A0c onward, use one fixed validation-loss monitor built from the condition weights in section 8 for early stopping and the plateau scheduler. Restore the best checkpoint. For comparisons with a changed epoch length, match the optimizer-step cap and scheduler/validation frequency as well.

Run seed 42 for screening. Refit up to two eligible finalists with seeds 42, 43 and 44 under the identical recipe, report mean/std, and retain fixed seed 42 as the deployable seed. Rank finalized configurations by mean selection score under the guards, with seed-42 eligibility also required. Do not choose a lucky seed or add an ensemble. V2's historical benchmark remains a one-seed reference; distinguish that from V3's seed-variance assessment.

Use from-scratch initialization as the controlled default for frontend ablations. **A changed frontend does not make a warm start mathematically meaningless.** Where weight shapes remain compatible, C1 may compare adaptation of V2 weights with a fresh model using the identical V3 frontend and statistics. Reset optimizer/scheduler state for fine-tuning; document BatchNorm state treatment, use all layers trainable by default, and start with a preregistered lower learning rate such as 1e-4. Any LR comparison is itself part of the registered grid. Staging mild-to-full augmentation or freezing layers is optional C2 work, not an unlogged adjustment.

### 7.3 Frontend contracts

B1 subtracts the **scored window's** waveform mean before spectral extraction in both training and runtime. For B2, specify one scalar mean over the whole log-mel window; do not subtract a separate time mean per frequency band, which would remove stationary spectral shape. Document the exact operation order relative to B1 and per-band standardization. Gain invariance is approximate in the presence of floors, clipping or nonlinear transforms.

B3 fits statistics from a deterministic, versioned training-only mixture matching the declared clean/channel distribution. Freeze the sampled realizations/seeds and original-source weighting before fitting; do not estimate them anew every epoch or count extra positive copies as unrelated evidence. Apply the reviewed statistics eligibility and per-band floor policy unless a separately logged experiment changes it.

Each selected frontend variant needs a new config/statistics identity, golden fixtures, sample-rate tests and package hash. Test training, offline inference and live-window processing at 16/44.1/48 kHz through the shared code. Keep quiet-gate, downmix, window cadence, smoothing and reset policies fixed in the main ladder. If evidence requires a capture/downmix fix, version and evaluate it as a separate processing change before attributing its effect to the CNN.

## 8. Validation selection and acceptance

### 8.1 Fixed evaluation sets

Use unchanged clean validation plus three fixed augmented versions of each validation clip: mild, moderate and severe. Freeze recipe, per-clip seed/parameters, crop positions, procedural noise/room realization, library versions and waveform hashes before A1. Include both labels at every severity. Generate the held-out test variants only under the already frozen recipe, through a separate evaluator after selection; training and selection may not load their feature payloads.

Weight clean validation at **1/2**, and each augmented severity at **1/6**, with equal clip weighting within a condition. Compute weighted confusion totals and then F1; do not average condition F1 values and call that pooled F1. This avoids making three correlated augmented copies implicitly outweigh clean validation 3:1. Report every condition separately, with unweighted counts as well as pooled metrics. They are repeated conditions on the same 247 validation groups, not four independent datasets.

Before fitting, score frozen V2 on this validation suite and save those reference results. The evaluation simulation stays fixed even when a rung does not train on a corresponding transform. Report recall/FAR/F1/ROC-AUC/PR-AUC and paired condition/severity curves at fixed thresholds. Use validation artifact slices for clean, clipped, DC, partial-zero and silent/near-silent clips, allowing overlapping flags and giving group counts. Type/family slices remain descriptive when sparsely represented; extra Membo remains train-only.

### 8.2 One explicit V3 selection rule

V2 and D044 selected the threshold by **maximum clean-validation F1**, ties to the higher threshold. V2 did not use a primary fixed-FAR threshold. The proposal's “same false-alarm target V2 used” therefore cannot be reused literally.

For V3, preserve the maximum-F1 objective and tie direction, but **explicitly amend the calibration population to the weighted clean-plus-augmented validation pool above**. For each checkpoint, choose one raw-logit threshold maximizing weighted pooled F1; ties go to the higher threshold. Freeze that threshold across clean, all severities, artifact slices and live replay. This amendment must appear in the V3 decision log; historical V2 scores keep their original threshold.

Eligibility at that threshold requires:

1. Clean-validation recall is at least V2's `164/185` minus 0.03: at least **159/185** positive clips detected.
2. Clean-validation FAR is no worse than V2's `5/1325`: at most **five false positives**.
3. FAR is at most **1% in each augmented validation condition**. A severely corrupted positive result is not useful if augmented negatives also cross the threshold.
4. Mean recall across the three augmented conditions improves by at least **five percentage points** over frozen V2 on those same conditions at its frozen threshold. No severity may lose more than three recall points against its V2 reference.

Rank eligible finalized configurations by mean weighted pooled validation F1 across seeds 42/43/44. Require the seed-42 package to meet all guards; report every seed's guards and metrics. Ties within 0.005 pooled F1 prefer fewer optional transform groups, then lower measured runtime cost, then earlier preregistered configuration order. With no eligible candidate, retain V2 as the deployment and report that V3 did not meet the target. Do not relax guards after observing results.

For comparability with D044, also report each finalist's clean-validation-optimal F1 threshold and the validation-derived 1%/5% FAR operating points as **secondary analyses**. They do not replace the selected deployment threshold. Show V2 and V3 at matched validation-FAR targets as a secondary robustness comparison. Never tune a separate threshold per severity or use test FAR to set one.

The three-point/five-point/1% values are the fixed planning choices adopted here, not measured V3 outcomes. A later owner-requested change requires a dated protocol amendment before the affected experiments.

### 8.3 Final benchmark and uncertainty

Freeze one configuration, seed-42 checkpoint, frontend/statistics, threshold and complete decision policy before the final evaluator can read held-out payloads. Evaluate that one candidate on clean test and the three prespecified augmented test conditions in one locked evaluation job. Existing saved V2 clean-test rows can supply the historical clean comparison; one preregistered frozen-V2 inference pass on augmented test may supply paired robustness references. No other candidate enters the final job.

Preserve the original V1 and V2 test locks. Create a separate V3 lock binding checkpoint/config/threshold, test membership, augmentation-suite identity, score rows and code version. A reused benchmark is not made unseen by rebuilding caches or renaming the experiment. Test outcomes cannot initiate new tuning on that same study.

The final clean-test acceptance target remains recall within three points of V2's `159/180`, with FAR no worse than `5/1326`: **at least 154/180 positives and no more than five false positives**. Report failure if this target is missed; do not use it to choose another configuration. The prespecified augmented-test results measure whether validation robustness transfers; report all severities even if they fail.

Use at least 1,000 paired group-bootstrap draws for intervals and V3-minus-V2 differences. Resample original groups and keep every clean/augmented condition of a source together. Report the small positive-group counts: validation 29, test 28. Intervals describe these represented groups, not new devices or sites. Multiple ablations and repeated validation use make selected validation estimates optimistic; three seeds and bootstraps do not remove that limitation.

## 9. Live diagnostics and limits

Before model work, inspect existing evidence for capture/channel issues: per-channel levels and correlation if original two-channel samples exist, actual browser downmix code, resampler state, alignment, quiet/reset behavior and the timing of the brief detection. The saved WAVs are mono; they cannot reveal the original channel correlation or recover a lost channel. A trace with settings but no PCM cannot be replayed as audio or establish the true noise floor/SNR.

The four historical microphone files and additional fifth file remain development diagnostics, never training, statistics-fitting, threshold or model-selection data. Replay the selected frozen V3 once through the actual window builder, gate and smoother. Report per-window logits, raw votes, majority decisions, warm-up/standby/reset counts and sustained detection over owner-described drone intervals. A canary failure is a reported limit, not permission to pick another candidate.

If their mixed spectra/levels informed the augmentation design, log that before training. Their whole-file RMS is not a noise-floor estimate: only a known background-only interval can support that measurement. After such use, they are not an independent final test. Avoid repeated canary checks during the ladder.

Add prespecified validation-only continuous streams through the real runtime to assess decision delay, fraction of positive time detected, negative false-alarm events and the effect of three-window smoothing. Define contiguous event and reset policies before measuring. Report raw observation duration and event counts; short or looped streams cannot establish a reliable real-world false-alarm rate per hour.

Synthetic channels approximate phones and rooms. Severe filtering may erase usable harmonics; procedural reverberation lacks measured device/room responses; gain alone changes level without improving SNR. There are still only two drone types and a reused indoor benchmark. Per-window normalization already has historical evidence in D058 against the earlier robustness suite, so retain that result as context while allowing a new, explicitly different augmentation experiment.

A separate future evaluation-only recording session, with both drone-present and drone-absent periods, varied devices/placements and documented intervals, would provide stronger evidence. Collect it only under a separately requested session protocol after the model/policy is frozen. It adds no training data if never used for fitting or tuning. It is optional future work and is not silently included in this implementation plan.

## 10. Execution order and completion gates

| Gate | Required result before advancing |
|---|---|
| G0 Archive | Source, V2 artifacts, binary raw audio and private diagnostics inventories complete; missing trace status explicit; ZIP member hashes/CRC, fresh restoration and second-location hashes recorded |
| G1 Protocol | New decision IDs reserved; experiment graph, clean/augmented weighting, threshold/selection rules, acceptance guards and budgets recorded before training |
| G2 Prechecks | Existing capture/downmix evidence checked; training-only spectral/SNR/label checks pass; unresolved channel evidence clearly recorded |
| G3 Preparation | Original splits and cleaning preserved; raw loader parity demonstrated; train-only nuisance bank/statistics provenance verified; fixed validation suite ready; test-access guard passes |
| G4 A0 | Saved inference and cached/raw-loader controls satisfy parity/reproduction requirements; the changed validation monitor has its own no-channel control; no test scoring |
| G5 Main ladder | A1–A6 logged with parent/run/config hashes and identical validation suite; optional A7/A8/B/C experiments stay within their registered scope |
| G6 Selection | Finalists assessed with seeds 42/43/44; one eligible configuration and seed-42 package selected entirely on validation, or a documented no-candidate outcome |
| G7 Freeze and test | Threshold/package/policy frozen; one V3 final scoring job completed with historical-test disclosure and paired grouped uncertainty |
| G8 Export | Self-contained `CNN_V3.0` bundle includes model, frontend, statistics, threshold, labels, provenance, golden fixtures, model card, software verification and member hashes |
| G9 Runtime | Training/offline/live parity, 16/44.1/48 kHz tests and package checks pass; selected-model hash/version/threshold/window/cadence/smoothing browser contract updated and verified |
| G10 Delivery | Final report states gains, failures, all guard results and diagnostic limits; next context version reflects actual results; task changes alone committed and pushed |

Verify every augmentation stage with meaningful DSP/property tests, deterministic replay and split/provenance tests; verify the training/raw path at matching offsets; test that selection cannot access test payloads. Do not use a range/dtype test alone as proof of acoustic correctness. Run the focused Python/browser regression suites after export integration. Any user-facing UI edits must also follow the project's Impeccable and motion instructions, including desktop/mobile Chrome DevTools screenshots and an audit; no UI edit is part of this plan-finalization task.

Stage V3 beside V2 and retain a rollback configuration. Activate it only after package/runtime checks and the declared acceptance gates pass; export a failed research candidate only with its failed status explicit and keep V2 selected. Publish source/config/tests/aggregate reports to the existing GitHub remote after required verification, following the current task branch. Keep raw audio, per-clip private diagnostics, generated caches/checkpoints and local ZIPs out of commits. Do not include unrelated uncommitted work or force-push.

## 11. Evidence references

- `docs/CNN_V2_0_IMPLEMENTATION_PLAN.md`: version boundary, archive policy, original data/split/frontend constraints.
- `docs/CNN_V2_0_IMPLEMENTATION_REPORT.md`: retained counts, data layout, training recipe, threshold rule, benchmark metrics and environment evidence.
- `docs/CNN_V2_FRONTEND_VERIFICATION.md`: package identity, software parity, browser checks and original four-recording replay.
- `docs/CNN_V2_DEBUG_TEST_TUNING_PLAN.md`: digital Bebop control, saved replay paths, capture limitations and live evaluation requirements.
- `results/cnn_v2/data/DATASET_READY.json`, `training/freeze.json`, `evaluation/report.json`, `evaluation/scoring.lock.json`: generated local source identities and frozen measurements; the latter three paths are relative to `results/cnn_v2/`.
- `config/live_detector.json`: active V2 runtime contract. The legacy seed-44 runtime manifest is a separate preserved artifact.
- `Project documents/PROJECT_CONTEXT_V8.md`: one-dataset/size/library-first rules, existing D044 threshold protocol and D058 frontend result. This companion document is outside the Git root.
- Supplied proposed V3 plan: proposed transform ranges, owner-described recording labels and the unlocated 99-window trace summary. Those trace numbers retain this limited provenance.

## 12. Additions and modifications to the proposal

| Change | Reason |
|---|---|
| Replace the broad ZIP command with four explicit archives and preserve V1 separately | Separates reproducible source, V2 artifacts, indispensable raw WAVs and private diagnostics; avoids browser/session state and duplicate multiclass audio |
| Use actual V2 paths and active `config/live_detector.json` | The proposed data path does not exist; older runtime selection files still describe V1 seed 44 |
| Preserve the recovered hash-matching training YAML as well as the current portable YAML | The current YAML no longer has the config digest recorded in the training freeze; exact original bytes were reconstructed and verified without changing the active config |
| Require the raw WAV backup and per-member restore verification | CSVs/URLs and a tag cannot restore the dataset, ignored model weights or the deployed software state |
| Handle a dirty working tree without “commit or stash first” | Preserves unrelated changes and records exactly which working files form the V2 baseline |
| Inventory five recordings and mark the session JSON missing | Four recordings support the historical replay; the fifth is additional evidence with a distinct role |
| Reserve fresh decision IDs | D044 through D058 are already used and must retain their history |
| Replace executable-looking untested skeletons with implementation contracts | The proposed validation crop was random, helper APIs were assumed, RNG draws could repeat, FFT/noise joins could wrap, and DRR/SNR were not fully enforced |
| Separate original training, raw-loader refactoring and validation-monitor controls | A changed input path or early-stopping monitor must not be credited as an augmentation improvement |
| Add a fixed 20% V2-recipe training branch and explicit added-noise survival checks | Preserves clean-distribution exposure while preventing background plus floor noise from silently exceeding the stated corruption budget |
| Keep held-out nuisance generation procedural and candidate-independent | Avoids mixing training donor audio into evaluation and makes every rung face the same fixed conditions |
| Separate saved-copy diversity, sampler exposure and loss weights | Copies change class ratio and epoch length; retaining inverse-frequency weighting unchanged can compound rebalancing |
| Replace approximate copy totals/storage with manifest-derived estimates | The proposed K values yield 18,115 total training entries and approximately 342.6 MB additional PCM16 audio, not independent new samples |
| Permit compatible warm starts after frontend changes | Scratch training is a useful controlled default, but frontend changes alone do not invalidate reuse of compatible weights |
| State one explicit weighted-pool F1 rule and numeric validation guards | V2 selected maximum F1, not a primary FAR target; the proposal combined inconsistent rules and left its clean-validation constraint unspecified |
| Keep test acceptance separate from selection, with numeric counts and a new lock | A failed final test must be reported without using it to choose another candidate; historical test reuse remains disclosed |
| Add source-group bootstrap pairing and continuous decision metrics | Repeated augmented clips are correlated, and clip F1 alone does not assess the live gate/smoother behavior |
| Retain optional extensions and stronger real-world evaluation as bounded work | Broad fine-tuning is allowed, while architecture/codec/storage choices and new recording sessions remain separately justified |

The archive, rules and measurements above make V3 reviewable without claiming that augmentation will fix the phone test. Implementation is complete only when the corresponding gates have evidence; finalizing this plan does not mark those gates as passed.

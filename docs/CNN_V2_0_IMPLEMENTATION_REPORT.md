# CNN V2.0 implementation report

**Run date:** 2026-11-26  
**Protocol:** [CNN V2.0 implementation plan](CNN_V2_0_IMPLEMENTATION_PLAN.md)  
**Scope:** one depth_4 CNN, CNN-only dataset preparation, one real-data training run, one frozen test-scoring pass, and one deployable CNN bundle.

## Status

The implementation gates C0–C12 were completed on the current Windows CPU environment. The task-specific code and report still need a final scoped commit and push (C13). Generated datasets, checkpoints, score rows, and the V1 archive are local artifacts and are excluded from Git.

| Gate | Result | Evidence |
|---|---|---|
| C0 V1 archive | Pass | `archive/Models V1.0.zip`, embedded member manifest, CRC and restore-hash verification; outer SHA-256 sidecar |
| C1 CNN-only scope | Pass | V2 producer writes only full-clip mel power, indices, split manifests, and statistics; one architecture/run |
| C2 inventory | Pass | Fresh inventory of 11,704 raw WAVs; missing `math` import fixed |
| C3 cleaning and split identity | Pass | 10,623 retained rows; cleaning exclusions reconcile; all frozen V1 split CSV checksums and clip/label/group assignments match |
| C4 persistent package | Pass | Schema 4 `cnn_mel_power`, transaction hashes and ready marker; safe reload verifier passes |
| C5 statistics | Pass | 7,519 eligible train clips, 88 excluded, 452,765 eligible frames; 64 finite positive standard deviations |
| C6 frontend parity | Pass | Train/validation caches and stats match the existing baseline exactly; both saved native-rate goldens have zero error |
| C7 preparation regressions | Pass for implemented checks | 13 CNN V2 focused tests pass, including safe no-test-payload verification, interrupted publication, model count, and test-scope checks |
| C8 environment | Pass | Separate clean CPU environment installs pinned packages; `pip check` clean; fresh model and frontend verification pass |
| C9 training | Pass | One seed-42 run, 31 epochs; best epoch 23 restored and recorded |
| C10 freeze | Pass | Raw-logit validation threshold and identities written before V2 test inference |
| C11 evaluation | Pass | One scoring pass, 1,506 saved score rows, complete lock, 1,000 group-bootstrap draws |
| C12 export | Pass | One TorchScript CNN plus frontend/statistics bundle; golden, parity, fresh-process, and clean-environment checks pass |
| C13 delivery | Pending | Scoped report/code commit and GitHub push remain |

## V1 archive

The archive is [Models V1.0.zip](../archive/Models%20V1.0.zip) and has 703 files totaling 1,847,016,558 bytes. It contains the old model artifacts, processed/interim treated data, results tables/cards/figures, and runtime model assets. `ARCHIVE_MANIFEST.json` lists each member’s path, size, and SHA-256. ZIP CRC verification and a complete member-by-member restore hash check passed. The outer digest is in [Models V1.0.zip.sha256](../archive/Models%20V1.0.zip.sha256).

Raw source recordings, `.git`, source code, and local environments are outside the archive. Archive creation preserved the original files in their existing locations.

## Raw files and cleaning

The source corpus was discovered under `DroneAudioDataset-master/DroneAudioDataset-master/Binary_Drone_Audio`. Its 11,704 inputs are all mono, 16 kHz, PCM 16-bit WAV files. The inventory contains 10,372 `unknown` clips and 1,332 `yes_drone` clips. Source length ranges from 24 to 81,920 samples; the median is 16,347 samples. The two extremely short clips are excluded by the reviewed cleaning ledger.

The cleaned corpus has 10,623 clips: 9,291 negatives and all 1,332 positives. The reviewed ledger excludes 1,079 redundant exact-zero copies and two too-short clips. Quiet drone clips remain positive and flagged; the implementation does not remove or relabel them. The fresh source inventory SHA-256 is `cdfa077215c174ba3a88ce89f4095908685dcc4b1b3aa43e7d38a243d581bf24`.

The fixed split is the archived V1 group assignment. Fresh V2 manifests were compared with all three frozen split CSVs by their file digests and by clip ID, label, and group. No group crosses splits.

| Split | Clips | Negative | Positive | Groups | Positive groups |
|---|---:|---:|---:|---:|---:|
| Train | 7,607 | 6,640 | 967 | 1,249 | 146 |
| Validation | 1,510 | 1,325 | 185 | 247 | 29 |
| Test | 1,506 | 1,326 | 180 | 247 | 28 |

Other retained-corpus flags include 9 silent clips, 59 near-silent clips, 246 partial-zero clips, 206 clipped clips, and 116 DC-offset flags. Flags are preserved in the manifests and indices. Extra Membo remains train-only and is not reported as evaluated.

## Fresh V2 data format and frontend

The persistent output is `results/cnn_v2/data/`. It is a new representation and does not reuse the old tabular feature cache or run any tabular feature extractor.

| File pattern | Format and purpose |
|---|---|
| `cache/mel_power_<split>.npy` | Float32 power mel array shaped `(64, total_frames)`; full clip frames are concatenated in manifest order |
| `cache/index_<split>.csv` | One row per clip, contiguous frame start/count, centered 28-frame offset, length status, and quality/statistics flags |
| `manifests/<split>.csv` | Clip labels, groups, family/type and reviewed cleaning metadata |
| `normalization_stats.npz` | Float32 `mean` and `std` vectors, each length 64 |
| `normalization.json` | Statistics policy, counts, and per-band retained/floor-masked cell counts |
| `DATASET_READY.json` | Schema-4 transaction marker, split/source identities, and member hashes |

The cache dimensions are train `(64, 457897)`, validation `(64, 91761)`, and test `(64, 91494)`. The payloads are approximately 164 MB in total. A model sample is a centered validation/test window or a random train window of 28 frames, corresponding to 7,936 waveform samples (0.496 seconds), transformed to float32 `(B, 1, 64, 28)` after normalization.

The frontend uses channel 0, 16 kHz, FFT/window 1,024, hop 256, periodic Hann, no centering, 64 HTK area-normalized mel bands, 50–7,500 Hz, power 2, and HQ soxr resampling. The measured passband edge remains 7,540.2787 Hz. The full training cache and validation cache are element-for-element identical to the V1 power caches in this environment. The 64 means and standard deviations are exactly identical to the archived V1 statistics.

Statistics use eligible train clips only, omit cells at or below −119.999999 dB per mel band, and retain the 88 statistics-ineligible clips in model training. Each band has its own retained/masked counts. The training statistics reproduce 7,519 eligible clips and 452,765 full-clip frames before cell-level floor masking.

## CNN V2 training and freeze

The only model is the existing best depth_4 CNN: four convolution/BatchNorm/ReLU blocks with channels 16, 32, 64, 64; 3×3 kernels; 2×2 pooling after the first three blocks; global average pooling; dropout 0.3; one logit. It has 60,641 trainable parameters.

Training used seed 42, CPU float32 with four Torch threads, batch size 64, fresh valid train offsets each epoch, power gain augmentation uniformly from −12 to +12 dB, weighted BCE with positive weight `6640/967`, AdamW at 1e-3 with 1e-2 weight decay, and ReduceLROnPlateau. Early stopping used validation loss, patience 8, and restored the best checkpoint. The one run stopped after 31 epochs and selected epoch 23 with validation loss 0.25395. Epoch history and epoch-boundary optimizer/scheduler/RNG resume state are preserved locally.

The freeze record selected raw-logit threshold 2.82396054 by maximum validation F1, with ties going to the higher threshold. Validation F1 is 0.9266; validation ROC-AUC is 0.9880. The freeze record binds checkpoint, config, manifests, statistics, and ready-dataset digests. It was written before test scoring.

## Single V2 benchmark evaluation

The V2 test split is the same group-held-out benchmark that had already been scored in V1. It is explicitly **not an independent unseen test**. The original V1 test lock remains unchanged; the V2 scoring lock cites its SHA-256 and verifies the same frozen test manifest. The evaluator scored one frozen checkpoint once and stored 1,506 clip-level rows. The score file digest and checkpoint/freeze identities are in `results/cnn_v2/evaluation/scoring.lock.json`.

At the validation-F1 threshold, test metrics are F1 0.9244, accuracy 0.9827, ROC-AUC 0.9916, and PR-AUC 0.9685. Confusion counts are TN 1,321, FP 5, FN 21, TP 159; false-alarm rate is 0.00377 and miss rate is 0.11667. The group bootstrap used seed 20261126, sampled entire groups, and completed 1,000 accepted draws with no class-degenerate skips. The 95% percentile intervals are F1 [0.8466, 0.9745], accuracy [0.9697, 0.9939], recall [0.7703, 0.9644], FAR [0.0000, 0.0109], and miss rate [0.0356, 0.2297].

For the validation FAR 1% and 5% thresholds, the realized test FARs are 0.00679 and 0.03695 respectively. Test drone recall is 0.9684 for Bebop and 0.7882 for Membo overall; the `mixed_membo` family recall is lower (0.6909). Family rows with fewer than three groups are marked descriptive. These are observed outcomes on a reused benchmark and do not establish unseen-test generalization.

## Export and environment verification

The single export is `results/cnn_v2/export/CNN_V2.0/`. It contains one TorchScript CNN, frontend config and normalization statistics, validation threshold/label mapping, preserved 16/48 kHz FLOAT golden WAVs and expected tensors, a model card, metadata, software verification, and a package manifest. It has no V1 checkpoint dependency and no raw-data path. Eager-to-TorchScript max absolute logit difference was zero. Both native-rate frontend goldens passed with zero normalized-tensor error. A fresh process and the separate clean environment both loaded and ran the model.

The clean environment used CPython 3.12.10 and pinned NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1, SoundFile 0.14.0, soxr 1.1.0, librosa 1.0.1, scikit-learn 1.9.1, PyYAML 6.0.4, matplotlib 3.11.2, and CPU PyTorch 2.15.0. `pip check` reported no broken requirements and CUDA was unavailable. Reported timing is about 4.13 ms per CNN forward on a synthetic normalized sample and 5.37 ms per 0.9-second golden frontend input; these are separate timings and are not microphone acceptance measurements.

## Remaining limits and files

- The V2 test benchmark is reused, so the reported test metrics are not independent confirmation of generalization.
- This is one seed and one model; no seed search or alternative model was run.
- The package verification covers software loading and frontend parity, not a physical microphone or live detector activation.
- Cross-OS portability was not measured. The clean environment check was a separate Windows CPU environment.
- The current V2 config uses `DRONE_AUDIO_DATASET_ROOT` or `data/raw/DroneAudioDataset-master` for future builds. The raw corpus was not copied into Git or into the model bundle.

Primary artifacts: `results/cnn_v2/data/`, `results/cnn_v2/training/freeze.json`, `results/cnn_v2/evaluation/report.json`, `results/cnn_v2/evaluation/scoring.lock.json`, and `results/cnn_v2/export/CNN_V2.0/`. These are ignored local project artifacts; the protocol/code/report are the publishable deliverables.

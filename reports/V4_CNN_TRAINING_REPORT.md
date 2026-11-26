# CNN V4.0 single-model training report

Date: 2026-11-26

Run: `cnn_a0_seed42`
Status: **FROZEN_BEFORE_TEST**

## Scope

One CNN was trained from the frozen V4 corpus using seed 42. No architecture or hyperparameter search, additional seed, threshold selection, test scoring, or test-audio access took place. A failed preprocessing attempt stopped before model initialization; its error is preserved in the run's `attempts.jsonl`. The one completed CNN training run is recorded as `training_runs: 1`.

## Fixed model and training recipe

- Architecture: D048 A0, three 3×3 Conv/BatchNorm/ReLU blocks with 16, 32, and 64 channels; 2×2 max pooling after the first two blocks; global average pooling; dropout 0.3; one logit. Parameter count: 23,585.
- Seed: 42. Device: CPU, one PyTorch thread.
- Loss: `BCEWithLogitsLoss` with training positive weight `7,059 / 1,187`.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.01; batch size 64.
- Scheduler: `ReduceLROnPlateau`, factor 0.5, patience 3.
- Fixed stop procedure: validation loss, patience 8, maximum 100 epochs, restore the lowest-validation-loss state.
- Training input: one fresh random frame-aligned 0.5-second window per training clip per epoch; random gain from −12 dB to +12 dB in training only.

Validation loss was used only as prescribed by this fixed training recipe for scheduling and early stopping. It was not used to compare models or tune parameters.

## Run result

- Epochs completed: 61.
- Restored epoch: 53.
- Best weighted validation loss: 0.2021571547.
- Stop: eight consecutive epochs without improved validation loss.
- Training duration: 1,802.996 seconds.
- Checkpoint SHA-256: `f4372da59644395c174514944aa6387901ed483f64c137fd1ae4e31c83092049`.
- Train manifest SHA-256: `6b895c0d6ce420aa7452bed72c06f0db14eb472490ab82bd99de91276ef24d81`.
- Validation manifest SHA-256: `6fc5134f8faa3875e088d7f84fdc3d03debb2de0466cb73c5624410e754430e9`.

The best validation loss describes the fixed early-stopping procedure; it is not a thresholded detection metric or a test result.

## Training data artifacts

- Canonical full-clip mel-power and tabular caches were created for train and validation only. No test cache exists.
- Global per-band normalization was fit from 8,159 `stats_eligible` training rows; 87 training rows were excluded from fitting but remain in training.
- Cache sidecars bind artifacts to the V4 train/validation manifest digests and frontend content identity.
- Cache and statistics location: `storage/caches/corpus-002/processed/`.
- Checkpoint, history, freeze record, run configuration, and source-hash provenance: `runs/study-v4/cnn_a0_seed42/`.

## Boundary

The model and training artifacts remain in local project storage. The Al-Emadi source licence restricts redistribution of the dataset and its derived caches; no audio, cache, checkpoint, or local run configuration is included in GitHub publication. This training report, training script, and non-sensitive registry metadata are versioned.

The V4 test split remains frozen and unaccessed. No threshold, bundle, deployment, or runtime activation was produced by this training request.

# V4-TRAIN-001 — One CNN training run, one seed, no tuning

Owner instruction: “one CNN, one seed, no tuning or multiple runs.” This overrides the D049 recommendation to train three seeds and the D050 tuning procedure for this V4 training request only.

The single model was the fixed D048 A0 baseline: channels 16/32/64, three convolution blocks, global average pooling and dropout 0.3. Seed 42 was used, matching the V4 split/config seed. The fixed D049 CPU training recipe, scheduler and validation-loss early stopping were retained. Validation loss was used only by that recipe; no model/parameter comparison, threshold selection, test scoring, or test access was performed.

Run record: `registry/runs/cnn_a0_seed42.json`. Detailed output: `reports/V4_CNN_TRAINING_REPORT.md`. The checkpoint and training artifacts remain in local storage.

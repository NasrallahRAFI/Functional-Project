# V4-LIVE-001 — Activate the frozen A0 model for local raw-score monitoring

Date: 26 November 2026.

The owner requested server fixes and frontend integration after the single V4 run
completed. Asked how to handle the absent decision threshold, the owner selected
"Live scores; threshold pending". No operating threshold is selected by this task.

The monitor reads the existing local checkpoint from run cnn_a0_seed42 and the
normalization fitted on 8,159 eligible training rows. Runtime loading verifies the
registry binding, checkpoint, freeze record, normalization archive, training config,
and source identities for the training frontend and model. No new training,
normalization fit, validation scoring, or test access is performed.

V4 uses its own workspace and port 8766. V3 remains separate on port 8765. No V3
data/results/runtime compatibility mount is introduced in V4. Registration is in
repository/registry/runtime-selections/study-v4-score-only.json and synchronized
explicitly into the V4 registry snapshot. This is direct checkpoint loading; a
new portable bundle or model export is not required or claimed.

The UI and health metadata identify CNN V4.0, A0, seed 42, 23,585 parameters and
raw-score-only-v1. Threshold and binary verdict are null; voting is disabled.
Quiet input is standby. Logits are not displayed as probabilities.

The server fixes cover consistent numeric/localhost Host and Origin checks, an
IPv4-only CLI, bounded socket reads, early sample-rate/duration validation, strict
finite JSON serialization, and explicit JavaScript MIME mapping. The same boundary
fixes apply to V3. Existing V3 model selection and quiet/voting policies are retained.

Model weights, caches, audio, local run config, browser profiles and screenshots
stay outside Git. The test split remains frozen and unaccessed.

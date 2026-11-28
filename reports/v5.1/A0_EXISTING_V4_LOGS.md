# A0 — existing V4 ambient score and RMS log

**A0 gate: PASS.** Descriptive only; not identity-verified V5.1 calibration.

Input: `ambient-v4-20261126T212021Z-c7cf5a35.jsonl` (existing V4 log; no waveform read).
Scored windows: 1121. Recorded span: 564.266 seconds.
Raw logit min / p25 / median / p75 / max: -7.5818, -5.45229, -5.101, -4.6686, -3.14819.
Recorded RMS dBFS min / p25 / median / p75 / max: -45.008, -43.345, -42.78, -42.277, -31.055.
Missing identity header fields: statistics_sha256, frontend_hash.

## Training-level comparison

A verified V4 training-window RMS distribution is not available in the registered report artifacts. The V4 training report records the historical random-gain range, but that does not establish the RMS distribution of training inputs. Therefore this log cannot be compared quantitatively with recorded training levels. No levels were inferred from audio, and no score threshold or calibration claim was produced.

This log is retained as owner-confirmed no-drone context in the V4 provisional-threshold report. Its missing frontend/statistics identity fields mean it is descriptive evidence only. The short exposure and absent training-level reference leave H1/H3/H4 unresolved; the score/RMS association is not causal evidence for H2.

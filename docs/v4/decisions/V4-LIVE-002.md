# V4-LIVE-002: Supplemental MUSAN negative replay

Date: 2026-11-26

## Decision

Use the local MUSAN `noise` subset only as a supplemental negative-audio stress check for the frozen V4 A0 checkpoint. It is external to corpus-002 and split-002. Do not add its files or scores to V4 training, validation, or test membership, and do not treat its result as the live operating false-alarm rate or as a deployment threshold.

The owner’s requested operating targets remain at most 1 false-alarm event per hour and at least 80% pooled validation recall. V4 activation stays raw-score-only with threshold pending until representative ambient-only live logs have been collected and evaluated.

## Implementation and result

`scripts/score_musan_noise_supplement.py` scores WAV files under `MUSAN/noise` with the registered V4 model loader, `LiveWindowBuilder`, canonical feature transform, and the configured three-window majority `VoteSmoother`. It resets smoothing at each source-file boundary and at quiet windows. It reports the exact one-sided 95% Poisson upper rate bound over total input duration. Waveforms are read in place, never copied or written; score and inventory artifacts go under the ignored local output directory.

The 26 November 2026 local run scored 930 WAVs, 6.2263 hours, 44,741 windows (43,792 scored and 949 quiet), with no skipped files. On a predeclared 0.25-logit grid, 5.00 was the lowest MUSAN-only candidate meeting the 1/hour upper-bound target: one smoothed alarm, 0.1606 observed events/hour, 0.7619/hour 95% upper bound. At 4.75 there were three events and the upper bound was 1.2453/hour. These corpus-specific figures do not certify performance in live environments.

`scripts/score_v4_validation_windows.py` scores only the registered frozen validation manifest at frame offsets 0, 8, 16, 24 and 31. It verifies the manifest, source lineage, and derived DD segment identities before scoring. It computes grouped-bootstrap intervals by resampling recording groups, and writes clip-level and window-level ROC/PR curves to the ignored local output directory. The clip-level result uses the maximum logit across the five requested crops. Those overlapping crops are not a live smoother sequence.

At the 5.00 MUSAN-only candidate, pooled clip-level validation recall was 54.7% (123/225; 95% group-bootstrap interval 37.3%–70.9%). This is below the 80% floor. The result therefore fails the recall floor at this candidate operating point. No threshold was selected or activated; no test manifest or test audio was opened.

The V4 monitor now has an explicit ambient-only score logger. Starting it requires an active monitor session and an in-browser confirmation that no drone is present. It writes timestamped JSONL metadata only (raw logits, RMS, sequence/source-frame positions, capture drops, generations and reset reasons) to `local/reports/study-v4/ambient-score-logs/`. It writes no audio and closes the log when logging or monitoring stops. While it is active, the browser does not retain a duplicate unbounded in-memory score trace.

`scripts/replay_v4_ambient_score_logs.py` validates the V4 ambient-log contract, resets the production VoteSmoother at session/gap/quiet boundaries, evaluates the fixed 0.25-logit grid, computes exact one-sided Poisson upper bounds, and refuses to mark a candidate eligible before 24 hours of observed-window exposure. It reports a candidate without changing runtime selection; validation recall must still pass before any freeze or one-time test evaluation.

## Scope and limits

MUSAN’s noise recordings combine varied environmental and technical sounds, including unrelated noise classes, and are separate recordings rather than a continuous sample of the intended platform’s operation. They do not provide the required 24-hour live coverage or platform engine/vehicle conditions. The candidate is grid-dependent and applies to this supplemental corpus only. Any threshold decision must use representative no-drone V4 live score logs, then pass the validation recall floor before freezing a bundle.

The local score CSVs, per-recording inventory, and JSON run outputs are not versioned. They remain under `local/reports/study-v4/musan-noise-supplement/`; the source recordings remain at the owner-provided `D:\musan` path.

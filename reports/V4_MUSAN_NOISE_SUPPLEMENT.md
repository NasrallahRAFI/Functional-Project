# V4 MUSAN noise supplement

Date: 2026-11-26

This is an external negative-audio supplement for CNN V4.0 A0. It is separate from frozen corpus-002 and split-002. The analysis did not change the model, training run, validation labels, test lock, or runtime threshold. The selected runtime remains **raw scores; threshold pending**.

## MUSAN noise replay

The local `MUSAN/noise` directory contained 930 WAV recordings. The V4 live window builder and canonical feature pipeline produced 44,741 windows over 6.2263 hours; 43,792 windows received logits and 949 were treated as quiet standby windows. The configured VoteSmoother uses three windows and was reset at each recording boundary and on quiet windows. No audio was copied or saved.

The 0.25-logit candidate grid gave the following neighboring results:

| Raw-logit threshold | Smoothed alarm events | Observed events/hour | One-sided 95% Poisson upper bound/hour |
|---:|---:|---:|---:|
| 4.75 | 3 | 0.482 | 1.245 |
| 5.00 | 1 | 0.161 | 0.762 |

So 5.00 is the lowest candidate on this grid whose MUSAN-only upper bound is at most 1/hour. The bound is based on this corpus’s accumulated duration; it does not substitute for live ambient coverage.

## Validation at the MUSAN-only candidate

Each of 1,631 frozen validation clips was verified against its recorded source identity and scored at the available starts among frame offsets 0, 8, 16, 24 and 31. For the clip-level table, the highest of those raw logits represents the clip. The 95% intervals resample positive recording groups; clip crops are not counted as independent observations. ROC AUC and average precision use those clip-maximum scores. Window-level score curves are also saved locally.

| Validation slice | Clips (positive) | Positive groups | ROC AUC | Average precision | Recall at 5.00 (95% group interval) |
|---|---:|---:|---:|---:|---:|
| Pooled | 1,631 (225) | 33 | 0.9948 | 0.9716 | 54.7% (37.3%–70.9%) |
| Al-Emadi | 1,511 (185) | 29 | 0.9951 | 0.9725 | 65.4% (46.2%–79.6%) |
| Drone-detection-dataset | 120 (40) | 4 | 1.0000 | 1.0000 | 5.0% (0%–10.0%) |
| Bebop | 95 (95) | 11 | Not defined for a positive-only slice | Not defined for a positive-only slice | 65.3% (25.5%–84.3%) |
| Membo | 90 (90) | 18 | Not defined for a positive-only slice | Not defined for a positive-only slice | 65.6% (45.6%–83.3%) |
| DD DRONE | 40 (40) | 4 | Not defined for a positive-only slice | Not defined for a positive-only slice | 5.0% (0%–10.0%) |

Hard negatives had no clip-level scores at or above 5.00: HELICOPTER 0/50 across 5 groups and BACKGROUND 0/30. Al-Emadi ESC was 0/1,313 and `other` was 0/13. ROC AUC and average precision are undefined for one-class subclasses.

The pooled recall is below the requested 80% floor at this candidate. **This point does not satisfy both requested targets.** The threshold remains pending; no decision was activated and the test split remains locked and unaccessed.

## Reproducibility and boundaries

Use `scripts/score_musan_noise_supplement.py` for the MUSAN replay and `scripts/score_v4_validation_windows.py` for validation-window scoring. Default score artifacts are local and ignored by Git:

- `local/reports/study-v4/musan-noise-supplement/musan_noise_supplement_report.json`
- `local/reports/study-v4/musan-noise-supplement/musan_noise_threshold_replay.csv`
- `local/reports/study-v4/musan-noise-supplement/v4_validation_candidate_report.json`
- `local/reports/study-v4/musan-noise-supplement/v4_validation_roc_pr_curves.csv`

For ambient collection, start monitoring, enter an optional environment label, check **I confirm no drone is present**, then choose **Start ambient-only log**. Leave the monitor open and stop the log after the session. Run `python scripts/replay_v4_ambient_score_logs.py` to replay completed logs; it refuses to mark a candidate eligible before 24 hours of observed-window exposure. Then pass that candidate to `python scripts/score_v4_validation_windows.py --threshold <logit>` and compare pooled group-bootstrap recall with the 80% floor. Neither script changes runtime selection or accesses test.

MUSAN is a varied noise corpus, not 24 hours of continuous representative live ambience. Platform engine and vehicle conditions still need to be collected in no-drone sessions. The V4 interface now provides an explicit ambient-only logger that writes timestamped per-window logits, RMS, session boundaries, source-frame positions and reset/gap markers to `local/reports/study-v4/ambient-score-logs/`; it does not retain audio. Once sufficient logs exist, `scripts/replay_v4_ambient_score_logs.py` replays the actual sequences through the production VoteSmoother and applies the 24-hour duration gate. The MUSAN candidate is not a deployment threshold.

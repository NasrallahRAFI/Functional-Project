# V4 provisional threshold candidate

Date: 2026-11-26
Status: **PROVISIONAL CLIENT-SIDE LABEL — NOT VALIDATED**
Candidate: **raw logit −3.00**
False-alarm target under consideration: **at most 1 alarm per hour**

## Decision record

Per the user's direction, this candidate is based only on the current ambient-only recording. Its highest observed raw logit was −3.148. On the existing 0.25-logit grid, −3.00 is the lowest value above that maximum; there were no score crossings at −3.00 in this recording. This is a provisional reference from one short session, not a certified operating threshold. MUSAN results were excluded from this selection.

The frontend now applies this provisional threshold locally to raw logits and displays a two-of-three drone/no-drone label. The Python detector service remains raw-score-only with no selected deployment threshold. No test data was accessed. The validation recall floor is met at this provisional candidate, but its high validation-negative clip rate and the short ambient exposure leave the false-alarm requirement unmet.

## Evidence available

### Ambient-only live log

Log: `local/reports/study-v4/ambient-score-logs/ambient-v4-20261126T212021Z-c7cf5a35.jsonl` (local and ignored by Git)

- The user confirmed no drone was present through the frontend.
- Session ended after 564.3 seconds (9.4 minutes), with 1,121 scored windows.
- Raw logits ranged from −7.582 to −3.148. At candidate −3.00 there were zero raw threshold crossings, therefore zero three-window majority alarms.
- Observed exposure was 0.1544 hours. With zero alarms, the one-sided exact 95% Poisson upper bound is approximately **19.4 alarms/hour**. This short session cannot establish a rate at or below 1/hour.
- The log header has null `statistics_sha256` and `frontend_hash`, so the strict ambient replay utility rejects it. The session is summarized here for provisional tracking, but it is not accepted as a fully identity-bound replay artifact.
- The environment label was unspecified. The log includes capture-drop increments; the replay would reset across those gaps if the required identities were present.

### Frozen validation clips

At logit −3.00, the five-crop clip-maximum validation summary found:

- Pooled recall: **225/225 (100%)**.
- Validation negative clips above threshold: **581/1,406 (41.3%)**.

The five-crop clip maximum is an optimistic validation summary and is not a direct live smoother sequence. The large negative clip rate is a warning that −3.00 may produce excessive false alarms outside this one room recording. The candidate does not meet the false-alarm evidence requirement because its ambient exposure is too short.

## Limits and next update

This candidate is useful only as a provisional reference while ambient collection continues. The short ambient log does not materially tighten the false-alarm bound, and its missing identity hashes prevent strict replay. No MUSAN result was used to select this candidate.

Before replacing this provisional frontend label with a validated operating point, correct the logger identity fields, collect at least 24 hours of representative ambient-only scores (including relevant engine and vehicle noise), replay the completed logs through the production `VoteSmoother`, and report validation recall at the resulting false-alarm candidate. If recall remains below 80%, retain the failure finding rather than lowering the false-alarm target. Test audio remains locked until the operating point is frozen.

Source artifacts and detailed per-window outputs remain under ignored local storage and are not published.

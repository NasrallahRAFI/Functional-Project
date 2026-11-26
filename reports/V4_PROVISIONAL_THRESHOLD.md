# V4 provisional threshold candidate

Date: 2026-11-26  
Status: **PROVISIONAL CANDIDATE — NOT ACTIVE**  
Candidate: **raw logit 5.00**  
False-alarm target under consideration: **at most 1 alarm per hour**

## Decision record

Logit 5.00 is recorded as the current provisional candidate because it was the lowest tested 0.25-logit grid value meeting the false-alarm target on the supplemental MUSAN noise replay. A subsequent short ambient-only session also had no scores at or above 5.00. This is a working candidate for continued collection, not a certified operating threshold.

The V4 frontend and runtime remain in raw-score-only, threshold-pending mode. No verdict threshold was activated, no test data was accessed, and the validation recall floor has not been met.

## Evidence available

### Ambient-only live log

Log: `local/reports/study-v4/ambient-score-logs/ambient-v4-20261126T212021Z-c7cf5a35.jsonl` (local and ignored by Git)

- The user confirmed no drone was present through the frontend.
- Session ended after 564.3 seconds (9.4 minutes), with 1,121 scored windows.
- Raw logits ranged from −7.582 to −3.148. At candidate 5.00 there were zero raw threshold crossings, therefore zero three-window majority alarms.
- Observed exposure was 0.1544 hours. With zero alarms, the one-sided exact 95% Poisson upper bound is approximately **19.4 alarms/hour**. This short session cannot establish a rate at or below 1/hour.
- The log header has null `statistics_sha256` and `frontend_hash`, so the strict ambient replay utility rejects it. The session is summarized here for provisional tracking, but it is not accepted as a fully identity-bound replay artifact.
- The environment label was unspecified. The log includes capture-drop increments; the replay would reset across those gaps if the required identities were present.

### Supplemental MUSAN noise replay

- 930 noise WAV files, 6.2263 measured hours; no music or speech subset was used.
- At logit 5.00, one smoothed alarm event was observed: 0.1606 alarms/hour, with one-sided exact 95% Poisson upper bound **0.7619/hour**.
- At the next lower tested grid value, 4.75, three alarm events gave an upper bound of 1.2453/hour. Thus 5.00 was the lowest tested grid value meeting the fixed target on MUSAN alone.
- MUSAN is supplemental downloaded noise and does not represent all deployment environments, especially platform engine and vehicle noise.

### Frozen validation clips

At logit 5.00, the five-crop clip-maximum validation summary found:

- Pooled recall: **123/225 (54.7%)**, group-bootstrap 95% interval 37.3–70.9%.
- Validation negative clips above threshold: **0/1,406**.
- Helicopter hard negatives above threshold: **0/50**.

The 54.7% recall is below the proposed 80% floor. Five-crop clip maximum is an optimistic validation summary and is not a direct live smoother sequence. The candidate therefore does not currently satisfy the full operating-point requirements.

## Limits and next update

This candidate is useful as a provisional reference while ambient collection continues. The short ambient log does not materially tighten the false-alarm bound, and its missing identity hashes prevent strict replay. The 6.23-hour MUSAN result is not pooled with that live session because the environments and sampling process differ.

Before any threshold is activated, correct the logger identity fields, collect at least 24 hours of representative ambient-only scores (including relevant engine and vehicle noise), replay the completed logs through the production `VoteSmoother`, and report validation recall at the resulting false-alarm candidate. If recall remains below 80%, retain the failure finding rather than lowering the false-alarm target. Test audio remains locked until the operating point is frozen.

Source artifacts and detailed per-window outputs remain under ignored local storage and are not published.

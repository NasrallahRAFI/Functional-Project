# V4-LIVE-003: Provisional frontend drone/no-drone labels

Date: 2026-11-26

## Decision

Apply a clearly marked provisional client-side threshold of raw logit −3.00 to V4 live scores and display a drone/no-drone label after a two-of-three window majority. This value is the lowest 0.25-grid value above the maximum score (−3.148) in the current 9.4-minute user-confirmed ambient-only recording. MUSAN results are excluded from the selection.

This change is limited to the frontend readout. The verified detector service still returns raw scores with `threshold: null`; no runtime package threshold or test-evaluation state is changed. The UI labels are explicitly provisional, because the ambient exposure has a 95% Poisson upper false-alarm bound of about 19.4/hour and validation negatives frequently exceed −3.00 (581/1,406 five-crop clip maxima). No test data was accessed.

## Behavior

- Compare each finite raw logit with −3.00 and retain the most recent three votes.
- Display “Drone detected · provisional” for at least two above-threshold votes; otherwise display “No drone detected · provisional.”
- Wait for three fresh windows before labeling; reset votes after quiet standby, buffering, capture gaps, and monitor stop.
- Continue to show quiet input as standby, not as a no-drone vote.
- Keep the threshold and validation limitations visible in the monitor.

## Follow-up

Correct the ambient logger's missing statistics/frontend identity fields, collect representative ambient-only data over at least 24 hours, replay it through the production smoother, and revisit the client-side candidate against the false-alarm target and validation recall floor. The provisional frontend behavior does not authorize a frozen deployment bundle or test access.

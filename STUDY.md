# Study V5.1

Status: O0-A2 corpus-005/split-005 and O0-A3/O0-A4 capped failures remain immutable history. O0-A5 retains the exact 1 GiB byte ceiling and full AE/DD/ITU inputs, and raises the B0 reference-median clip cap from 20x to 105x for one new attempt.

The O0-A5 target corpus-007/split-008 reuses byte-identical P1/P2/P3 artifacts from O0-A4 and binds them to the amended contract. P4 passed R-LED, B0, B1, B2, B3/B4, and B5. P5-P7 are NOT_RUN; preparation remains incomplete. The 105x cap is confirmed feasible for this membership but is a documented relaxation of the clip-level concentration limit.
Preparation phases A0 and P0–P7 are authorized. Model training, new-model
threshold selection, local recording, downloads, and held-out evaluation are
not authorized. V4 runtime selection remains active and unchanged.

The authoritative execution plan is
[`DATA_PREPARATION_CNN_V5_1_FINAL.md`](docs/v5.1/DATA_PREPARATION_CNN_V5_1_FINAL.md).
Its companion contract is
[`DATA_PREPARATION_CNN_V5_1_CONTRACT.json`](docs/v5.1/DATA_PREPARATION_CNN_V5_1_CONTRACT.json).
The O0 request was recorded on 27 November 2026. SHA-256:

- Plan: `6a27c99f960e4e0520709216cf204e5890eca6e30ebbb438d7934f300af9f5a1`
- Contract: `6911c92b34696f351e4033b21af9ff74f0e6c49627b4987ebae87431d8fe0de4`

O0-A4 MUSAN group-coverage amendment SHA-256: `99eba1993b32ccddb6649ac6e73b4878b58131014807c4371aea8550189dba39`. O0-A5 authorizes one 105x B0 clip-cap attempt while retaining the 1 GiB cap.

Historical frozen identity: corpus-007/split-007 under O0-A4 remains P4-failed and ineligible for P5. The current frozen identity is corpus-007/split-008 under O0-A5 with P4 PASS; P5 is now eligible. Its selection retains all AE/DD/ITU files and 70 complete
MUSAN groups under the exact 1 GiB byte cap. O0-A3 corpus-006/split-006 and
all earlier attempts remain immutable history. O0-A4 cache identities
derived-build-v51-005 and cnn-v51-005 remain planned/unbuilt. MIC and Wang
remain OOD by default; optional Cochl and local recordings are absent.
See reports/v5.1/DATA_PREPARATION_CNN_V5_1_REPORT.md for execution evidence.

Implementation is isolated on branch `study-v5-1`, based on the V4 code
snapshot for reuse of unchanged numerical primitives. No V3 compatibility
mounts are connected. V4 artifacts and runtime remain out of scope.

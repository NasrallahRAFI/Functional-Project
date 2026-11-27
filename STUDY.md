# Study V5.1

Status: O0-A2 corpus-005/split-005 remains historical: P0-P4 passed and P5
failed. Under O0-A3, corpus-006/split-006 is frozen after P1-P3 and R-DET
passed; P4 failed B0 because the structural reference QP was primal
infeasible. P5-P7 for corpus-006 remain NOT_RUN. Preparation is incomplete.
Preparation phases A0 and P0–P7 are authorized. Model training, new-model
threshold selection, local recording, downloads, and held-out evaluation are
not authorized. V4 runtime selection remains active and unchanged.

The authoritative execution plan is
[`DATA_PREPARATION_CNN_V5_1_FINAL.md`](docs/v5.1/DATA_PREPARATION_CNN_V5_1_FINAL.md).
Its companion contract is
[`DATA_PREPARATION_CNN_V5_1_CONTRACT.json`](docs/v5.1/DATA_PREPARATION_CNN_V5_1_CONTRACT.json).
The O0 request was recorded on 27 November 2026. SHA-256:

- Plan: `20d2e6d1bd89be9cca7c2465db71bc53e0ed723ebdf96d3bcf061ba6fdd02d46`
- Contract: `76d912a8563a54786cb401c7a51104e1cc61be30d7eb0130a707aba81b731839`

Current frozen identities: corpus-006 and split-006 under O0-A3; do not use
P5 because P4 failed. The 1 GiB byte cap retains all AE/DD/ITU files and a
deterministic MUSAN subset. corpus-005/split-005 and all prior attempts remain
immutable history. New cache identities derived-build-v51-004 and cnn-v51-004
remain planned/unbuilt. MIC and Wang remain OOD by default; optional Cochl and
local recordings are absent.
See reports/v5.1/DATA_PREPARATION_CNN_V5_1_REPORT.md for execution evidence.

Implementation is isolated on branch `study-v5-1`, based on the V4 code
snapshot for reuse of unchanged numerical primitives. No V3 compatibility
mounts are connected. V4 artifacts and runtime remain out of scope.

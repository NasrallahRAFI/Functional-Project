# CNN V5.1 data preparation report

**Status:** preparation incomplete. The O0-A2 corpus-005/split-005 attempt has P0-P4 PASS and P5 FAIL from the recorded memory/ranking attempts. Under O0-A3, corpus-006/split-006 reached P1-P3/R-DET PASS; P4 failed B0 because the fixed structural-reference QP was primal infeasible. P5-P7 for corpus-006 are NOT_RUN. No V5.1 model training, threshold selection, runtime activation, or held-out test/OOD evaluation has been performed.

**Amended plan SHA-256:** `20d2e6d1bd89be9cca7c2465db71bc53e0ed723ebdf96d3bcf061ba6fdd02d46`  
**Amended contract SHA-256:** `76d912a8563a54786cb401c7a51104e1cc61be30d7eb0130a707aba81b731839`  
**P4 attempt:** `v51-20261127T141423Z-f0855385`  
**Latest P5 attempt:** `v51-20261127T163755Z-1708fce3`  
**Frozen membership:** `ea8d667f12a0432e5d78e4b20b9741e47b54b8c565fef2e024316a213b369133`

## O0-A3 1 GiB capped attempt

The owner authorized retaining AE, DD, and ITU in full and selecting a deterministic MUSAN subset under a combined 1 GiB raw-audio byte cap. Corpus-005/split-005 remains unchanged. The selection ranked complete MUSAN `group_key` groups deterministically within `source_class`, allocated the available byte cap proportional to each class's full-inventory bytes, and preserved aliases as indivisible groups.

| Measure | Result |
|---|---:|
| Total cap | 1,073,741,824 bytes |
| Full non-MUSAN payload | 785,180,400 bytes |
| MUSAN cap | 288,561,424 bytes |
| Selected MUSAN | 58 files/groups; 288,037,624 bytes |
| Combined payload | 1,073,218,024 bytes (523,800 bytes below cap) |

Selection manifest SHA-256: `5a2bb35a03a6896353fb6c08e1d632e53e7c50565e1b9f6f554f8711ff078a21`. New identities are corpus-006/split-006. P1 filtering, P2 mechanical identity, P3 S1-S8, and independent P3 membership reproduction passed. Frozen membership SHA-256: `908482c9aa85e416d7a53bf26090d003dea9dea2e15de6142988b740e65ae3e0`.

P4 stopped at B0: the structural reference QP returned `primal infeasible` under the unchanged O0-A2 source/stratum shares and caps. The descriptor checkpoint and failure record are preserved under `local/study-v5-1/attempts/v51-budget-1g-a3/`. No P5, training, or test/OOD work may proceed from this split. The 1 GiB cap therefore does not currently produce a P4-eligible preparation under the unchanged balancing contract; resolving that conflict requires a new owner-approved amendment and new identities, while preserving this failure.

## Phase results

| Phase | State | Evidence |
|---|---|---|
| A0 | PASS | Existing V4 ambient log summary; no calibration claim. |
| P0 | PASS | Approved plan/contract binding, isolated workspace, pinned V5.1 dependencies and implementation snapshot. |
| P1 | PASS | 19,301 source-file metadata rows; ITU 5,491 segments; MUSAN 2,016 audio files; 85,726 candidates. X1/X2 token scan found no matching MUSAN files. Missing MUSAN annotation associations are listed in the local audit; no content labels were inferred. |
| P2 | PASS | 85,726 candidate rows processed; 84,635 retained. Legacy AE/DD preservation and sample-identity gates passed. |
| P3 | PASS | S1-S8 passed; split frozen and reproduced byte-for-byte. Train reservoir contains 42,000 MUSAN rows. Membership and artifacts are registered as corpus-005/split-005. |
| P4 | PASS | Under O0-A1 tolerances 1e-11 and O0-A2 exact 80% reference floors, OSQP solved with max original-unit violation 9.2e-10. B0, B1, B2, B3/B4, B5 and R-LED passed. Negative ESS 2853.19 exceeds 2830.26; positive ESS 1567.31 exceeds 1334.28. |
| P5 | FAIL | Earlier attempts hit MemoryError or impractical serial ranking. Latest parallel attempt ended before gate completion; no P5 output artifacts were written, and the process exit reason is unknown. |
| P6-P7 | NOT_RUN | Await a passing P5 attempt. |

## Source and split caveats

The admitted sources are AE, DD, ITU-ARIS, and MUSAN. ITU remains a single-session source and its background labels remain weak. MUSAN's per-collection license and annotation files are preserved in its registered source snapshot. MIC and Wang remain OOD by default; Cochl and local recording remain absent. No test/OOD waveform was opened by the V5.1 preparation phases.

Attempt artifacts and descriptor checkpoint are under ignored `local/study-v5-1/attempts/`. Canonical inventory and mechanical ledger are under `storage/corpora/corpus-005/`; frozen split artifacts are under `storage/splits/split-005/`. The V4 runtime selection remains unchanged.

## Failure history and next execution

The original OSQP precision failure and O0-A1 B2 failure remain preserved under corpus-003/split-003 and corpus-004/split-004. O0-A2 passed P4 on corpus-005/split-005; no further effective-size decision is pending.

The append-only phase ledger in `local/study-v5-1/attempts/v51-20261127T141423Z-f0855385/phase_gates.jsonl` records P4 PASS at 14:19:41 UTC and P5 FAIL at 15:16:34 UTC on 27 November 2026. Later initialization attempts do not replace that failed execution record.

P5 pair selection now consumes candidate pairs lazily and ranks bounded chunks in parallel, retaining only local top-200 lists before a deterministic merge. Eligibility, random keys, pair count and A-chk acceptance limits are unchanged. The latest attempt ended without producing outputs or a gate result; its process exit reason is unknown. P5 remains FAIL until a new recorded attempt completes its checks. P6/P7 await that result.

Repair verification: five synthetic metadata regression tests pass. They compare selection against a full-sort reference, cover digest ties with Unicode/escaped IDs, enforce eligibility and short/empty behavior, and exercise 64,260 candidate pairs with peak traced selection allocation below 2 MiB. No preparation phase was rerun for this repair.

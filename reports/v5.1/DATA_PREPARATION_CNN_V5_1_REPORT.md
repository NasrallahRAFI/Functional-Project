# CNN V5.1 data preparation report

**Status:** preparation incomplete. The O0-A2 corpus-005/split-005 attempt has P0-P4 PASS and P5 FAIL from the recorded memory/ranking attempts. O0-A3 corpus-006/split-006 and O0-A4 corpus-007/split-007 both passed P1/P2/P3/R-DET and failed P4 B0 because the unchanged structural-reference QP was primal infeasible. P5-P7 remain NOT_RUN for both capped splits. No V5.1 model training, threshold selection, runtime activation, or held-out test/OOD evaluation has been performed.

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

## O0-A4 1 GiB group-coverage attempt

After O0-A3 failed B0 with insufficient N3 speech-group support, the owner authorized a new selection-order amendment while retaining the exact cap and all fixed O0-A2 gates. O0-A4 reserves the 20 smallest eligible speech groups and 10 smallest eligible music groups (hash tie-breaks), then fills the remaining allowance by a separate deterministic hash rank. Exact-byte aliases remain indivisible. This creates a short-recording selection bias; it does not change strata shares, descriptor margins, group caps, or solver rules.

| Measure | Result |
|---|---:|
| Total cap | 1,073,741,824 bytes |
| Full non-MUSAN payload | 785,180,400 bytes |
| MUSAN allowance | 288,561,424 bytes |
| Selected MUSAN | 70 groups/files; 288,561,124 bytes |
| Combined payload | 1,073,741,524 bytes (300 bytes below cap) |

Selection manifest SHA-256: `532fea22a5d1e8d9cf8f478f4c4821054b9aef73c77b0fce352d7f348d7a25fe`. The selected MUSAN group counts are 29 speech, 19 music, and 22 noise. The new frozen identity is corpus-007/split-007. P1, P2 (19,367 candidates; 18,276 retained), P3 S1-S8, and independent P3 reproduction passed. Both P3 runs produced membership SHA-256 `81abc1824fec44b9744ec25caaadc3a064602897e3f7d754b2526e768eed92fa`. The frozen split has 12,406 train, 2,443 validation, and 2,462 test rows; the MUSAN counts are 244 train, 23 validation, and 40 test.

P4 failed at B0 again: OSQP returned `primal infeasible` for the structural reference QP under unchanged shares and caps. Descriptor checkpoint SHA-256: `7cab4f6a1780b8be1b28a020adabc7222fca0a7d45e05dedf0317af0134f6f55`. Failure evidence is recorded in `local/study-v5-1/attempts/v51-budget-1g-a4/p4_failure.json`; the phase ledger and full descriptors remain alongside it. O0-A4 permits no further selection retry under corpus-007. P5-P7 remain NOT_RUN, and the Colab P5 adapter has been updated to O0-A4 and still requires a completed P4 with every gate PASS. A new balancing or cap change requires a separately authorized amendment and new identities.

### B0 feasibility diagnosis under the 1 GiB limit

The P4 traceback reaches the **negative** structural-reference fit. For corpus-007, the negative training counts are N1=1,054, N2=6,584, N3=92, N4=60, and N5=220. With the fixed 20× median reference clip cap, the `q0` median is `2.2782503e-5`, giving a per-clip cap of `0.0004556501`. At that cap, the maximum possible mass from N3/N4/N5 is only `0.04192 / 0.02734 / 0.10024`, below their fixed shares `0.22 / 0.08 / 0.15`. Their group-cap upper bounds are `0.345 / 0.225 / 0.330`, so group support is sufficient for those strata in this split; the clip cap prevents B0 from assigning the required mass. For this membership, a clip-cap multiple of at least 104.963× the median (about 105× instead of 20×) would be needed just to clear the per-stratum clip-count upper bounds. That alone would not prove the full QP passes.

A metadata-only integer upper-bound analysis used the complete P1 inventory and P2 ledger, grouping all 2,016 MUSAN files by the frozen `group_key` and enforcing the 288,561,424-byte MUSAN allowance. It includes zero-base-candidate groups and relaxes the deterministic rank assignment in P3, so it gives selection every possible advantage while still respecting the fixed 60% noise and 80% speech/music train-group fractions. Even under that optimistic bound, the maximum number of MUSAN base clips that can enter train is 4,945. The full AE/DD/ITU split contributes 962 N1, 6,584 N2, and 220 N5 negative train clips independently of MUSAN selection, putting the total negative train population at no more than 12,711. N2 therefore remains more than half of all negative training clips for every subset under this cap. Its `q0` value, `0.15 / 6,584 = 2.2782503e-5`, stays the median; the clip cap stays `0.0004556501`.

The same upper-bound analysis allows at most 180 N3 base clips in train under the entire MUSAN allowance. Even assigning each one the full clip cap gives N3 at most `0.08202` mass, below the required `0.22`. Therefore **no MUSAN subset selection under the 1 GiB cap can pass B0 while the current P2 candidate basis, P3 group fractions, O0-A2 shares, and 20× clip cap remain fixed**. The diagnostic is preserved at `local/study-v5-1/attempts/v51-budget-1g-a4/b0_cap_feasibility_diagnostic.json`; it is not a new corpus selection or an official P4 gate. Continuing requires either a larger audio cap or a separately authorized amendment to the fixed B0 constraints, with new identities.

## Historical O0-A2 phase results (corpus-005)

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

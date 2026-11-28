# CNN V5.1 data preparation report

**Status:** preparation incomplete. O0-A2 corpus-005/split-005 P0-P4 passed but its P5 attempts failed. O0-A3 corpus-006/split-006 and O0-A4 corpus-007/split-007 both failed P4 B0 under the original 20x clip cap. Under O0-A5, corpus-007/split-008 retains the exact 1 GiB membership and uses a 105x clip cap; its P4 passed R-LED, B0, B1, B2, B3/B4, and B5. The single O0-A6 P5 attempt on that unchanged split passed A-chk and B6. P6-P7 remain NOT_RUN. No V5.1 model training, threshold selection, runtime activation, or held-out evaluation has occurred.

**Current O0-A6 plan SHA-256:** `a9b0f30d7ea3c93a780fe97cbd1dc0a8a2d510439d61e75a0b11b28d540ddfbf`
**Current O0-A6 contract SHA-256:** `b4626fbff73437418e4a3185ffb1a6d2460cc23e6b1aa9bf5e868a77ac7568a4`
**Parent O0-A5 plan SHA-256:** `6a27c99f960e4e0520709216cf204e5890eca6e30ebbb438d7934f300af9f5a1`
**Parent O0-A5 contract SHA-256:** `6911c92b34696f351e4033b21af9ff74f0e6c49627b4987ebae87431d8fe0de4`
**Latest P4 attempt:** `v51-budget-1g-a5` (P4 PASS)
**Latest P5 attempt:** `p5-o0-a6-20261128T144500Z` (O0-A6, PASS; same split-008 membership)
**Current frozen membership:** `81abc1824fec44b9744ec25caaadc3a064602897e3f7d754b2526e768eed92fa` (same membership as split-007, re-bound under O0-A5)

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

P5 pair selection now consumes candidate pairs lazily and ranks bounded chunks in parallel, retaining only local top-200 lists before a deterministic merge. Eligibility, random keys, pair count and A-chk acceptance limits are unchanged. The latest attempt ended without producing outputs or a gate result; its process exit reason is unknown. That O0-A2 attempt remains historically failed. The later O0-A6 split-008 result is recorded below. P6/P7 await a passing P5, which O0-A6 has now supplied.

Repair verification: five synthetic metadata regression tests pass. They compare selection against a full-sort reference, cover digest ties with Unicode/escaped IDs, enforce eligibility and short/empty behavior, and exercise 64,260 candidate pairs with peak traced selection allocation below 2 MiB. No preparation phase was rerun for this repair.


## O0-A5: 105x B0 clip-cap attempt

O0-A5 retained the 1 GiB input and the unchanged O0-A4 membership. The new split-008 binding uses a 105x reference-median clip cap. The P1 inventory, P2 mechanical ledger, P3 manifests, independent P3 reproduction, and P4 descriptors are byte-identical to the prior attempt; their new freeze binds the amended plan/contract and records the A4 parent freeze.

P4 passed all gates: R-LED, B0, B1, B2, B3_B4, and B5. The negative reference clip cap is 0.002392162819 with reference ESS 1106.90; the positive reference clip cap is 0.011470037453 with reference ESS 1667.85. Membership SHA-256 remains `81abc1824fec44b9744ec25caaadc3a064602897e3f7d754b2526e768eed92fa` and combined inventoried audio remains 1,073,741,524 bytes (300 bytes below 1 GiB). The prior split-007 20x B0 failure remains immutable. At the time of O0-A5, P5 was eligible but NOT_RUN. O0-A6 P5 results are recorded below. P6-P7 and training/test/OOD evaluation remain NOT_RUN.


## O0-A6 P5 attempt on unchanged split-008

The owner authorized exactly two P5 changes and one new attempt: (1) choose the 200 A-chk pairs only after requiring positive 100–4000 Hz band power for both deterministic base/donor crops, retaining the 3 dB gate; and (2) replace B6 Monte Carlo shares with exact expected mass from the unchanged P4 weights, six bins, 5% floor and uniform A1 gain [-36,+6) dB. No upstream identity or artifact was rebuilt.

**Attempt:** `p5-o0-a6-20261128T144500Z`
**Frozen membership:** `81abc1824fec44b9744ec25caaadc3a064602897e3f7d754b2526e768eed92fa` (unchanged split-008)
**Contract:** `b4626fbff73437418e4a3185ffb1a6d2460cc23e6b1aa9bf5e868a77ac7568a4`
**Result:** P5 PASS; A-chk and B6 both PASS.

A-chk deterministically selected 200 pairs from 15,573,198 eligible unique-group pairs. Mean absolute dB error was **0.6177323376 dB**, below the unchanged 3 dB maximum. No zero-power pair entered the selected 200. The selected pair IDs, crop frames, band powers, and individual errors are recorded in `p5_summary.json`.

B6 used exact expected mass, averaging over each clip’s bound crop windows and integrating the uniform gain interval. The shares for bins `[-70,-60)`, `[-60,-50)`, `[-50,-40)`, `[-40,-30)`, `[-30,-20)`, and `[-20,-10]` dBFS were:

| Class | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Bin 6 | Out of range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.2950% | 18.2095% | 20.3391% | 15.9063% | 11.3589% | **5.0318%** | 16.8594% |
| 1 | 13.0006% | 20.0112% | 22.2244% | 16.2806% | 11.9319% | **5.0702%** | 11.4811% |

Every in-range class/bin mass meets the unchanged 5% floor. The previous O0-A5 attempt’s 100,000-draw shares remain informational only; they do not affect this result.

The output directory is `local/study-v5-1/attempts/v51-budget-1g-a5/p5-o0-a6-20261128T144500Z/`. `p5_summary.json` SHA-256 is `9142218ecc9f6eb21c4cf387a8d01aec06e406471a9dc655c35fc40415f5355a`; all recorded artifact hashes were rechecked against the files. Train/validation cache content hash is unchanged at `026ff2ecaa8a240383af8666f61bbfcd9ea7ef3d4e9a51c73e803d82259a43b0`. The run log is `local/study-v5-1/attempts/v51-budget-1g-a5/p5-o0-a6-20261128T144500Z.log`.

A prior launcher invocation stopped before entering P5 because the YAML still contained the removed `B6_draws` field. The config was aligned with O0-A6 and the preserved preflight log is `local/study-v5-1/attempts/v51-budget-1g-a5/p5-o0-a6-20261128T143000Z.log`; it produced no P5 artifacts and read no source data.

P6-P7 remain NOT_RUN. No model training, threshold selection, runtime activation, or test/OOD evaluation occurred.

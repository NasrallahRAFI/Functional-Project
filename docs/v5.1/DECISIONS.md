# V5.1 decisions

## O0 — execution authorization

**Date:** 2026-11-27
**Decision:** Approved by the owner's explicit request, “implement:
DATA_PREPARATION_CNN_V5_1_FINAL.md”. This authorizes the exact final plan and
companion contract, including the mechanical-pass relaxation and revised
effective-size policy.
**Scope:** A0 and P0–P7 only. No model training, new-model threshold sweep,
local recording, downloads, V4 changes, or test evaluation.
**Plan SHA-256:** `0d7e802b81c1a79f118645614bf535ac767fb0eed37e0a45a7aa82641ec7318f`
**Contract SHA-256:** `36d4001a34af881fbe86d052a9668d350b40ba9e8637a4ece396fada4f488720`

The exact approved documents are copied under `docs/v5.1/`. Later owner
decisions O1/O2/O3/O5 retain their plan defaults until recorded.


## O0-A1 — deterministic OSQP precision amendment

**Date:** 2026-11-27
**Decision:** Approved by the owner's explicit selection, “Amend O0: tighter OSQP tolerance”. This amendment responds to P4 attempt `v51-20261127T134425Z-92ab3fee`, where OSQP 1.1.3 returned `solved` with minimum weight `-1.6288860751502927e-9`, exceeding the fixed `1e-10` negativity tolerance while the maximum original-unit constraint violation remained under `1e-8`.
**Change:** OSQP `eps_abs` and `eps_rel` are tightened from `1e-9` to `1e-11`. Solver/version/backend, objective, margins, structural constraints, clip caps, nonnegativity check, original-unit violation limit, margin-drop order, and no-fallback rule are unchanged. No negative-value clipping range is widened.
**Amended plan SHA-256:** `bca93f9e1f98e82328f49b005bd6278c8f8318f37e2ecbf188f10d29ea56d3ce`
**Amended contract SHA-256:** `6911c92b34696f351e4033b21af9ff74f0e6c49627b4987ebae87431d8fe0de4`


## O0-A6 — P5 A-chk crop eligibility and exact B6 expected mass

**Date:** 2026-11-28
**Decision:** The owner authorized one amendment with two changes, followed by one new P5 attempt on the same frozen split-008. No upstream artifacts are rebuilt.
**Parent binding:** O0-A5 plan `6a27c99f960e4e0520709216cf204e5890eca6e30ebbb438d7934f300af9f5a1`; contract `6911c92b34696f351e4033b21af9ff74f0e6c49627b4987ebae87431d8fe0de4`; membership `81abc1824fec44b9744ec25caaadc3a064602897e3f7d754b2526e768eed92fa`.
**A-chk change:** Assign each retained train clip one deterministic crop using the named `a_chk` crop key over its ascending bound crop lookup. Admit only unique-group candidate pairs whose selected base and donor crops both have positive 100–4000 Hz band power, then take the 200 lowest existing deterministic `mix_check` ranks. Preserve the 3 dB limit and all other check behavior. Fewer than 200 eligible pairs fails; no post-selection replacements.
**B6 change:** Replace the 100,000-draw gate with exact expected bin mass. Integrate each crop RMS over the unchanged uniform A1 gain `[-36,+6)` dB, average over the clip's bound crop rows, weight by P4 train weights, and normalize within each class. The six fixed bins and 5% per-class floor remain unchanged. The prior attempt's 100,000-draw shares are retained as informational evidence only.
**Unchanged:** split-008 membership and all upstream manifests, P4 weights, 3 dB A-chk limit, B6 5% floor, six bins, gain range, augmentation, and every other P5/P6/P7 rule. No sampling/range/threshold retry is authorized. The O0-A5 frozen split remains byte-identical; the P5 artifact must record this O0-A6 parent binding.
**Prior Monte Carlo information only:** attempt `p5-windows-20261128T121057-2419f237`; class 0 shares `[0.12244109694862881, 0.18277327153341058, 0.20414059482425648, 0.15762070297412129, 0.11426805716492855, 0.04993433758207802]`, class 1 shares `[0.13119773210489014, 0.19991495393338057, 0.21545003543586108, 0.16274982282069456, 0.12439404677533664, 0.05267186392629341]`. These values do not participate in the O0-A6 B6 decision.

The prior corpus-003/split-003 and attempts remain preserved under their original hashes. Rebuild and register a new corpus/split/cache identity before connecting the amended preparation to `study-v5-1`.


## O0-A2 — exact 80% reference floor

**Date:** 2026-11-27
**Decision:** Approved by the owner's explicit selection, “Amend O0-A2: use the exact 80% floor”.
**Basis:** Under O0-A1, final negative ESS was 2,853.19. This exceeds exact 80% of the fixed negative reference (2,830.20) but not the ceiling-rounded floor (2,900). Positive ESS was 1,567.31 against a 1,334.28 exact 80% floor.
**Change:** B2 uses `min(n_ref, 0.8*n_ref)` with no round-up step. The 80% reference retention, reference QP, source/stratum shares, descriptor margins, caps, solver profile and post-solve rules remain unchanged.
**Amended plan SHA-256:** `20d2e6d1bd89be9cca7c2465db71bc53e0ed723ebdf96d3bcf061ba6fdd02d46`
**Amended contract SHA-256:** `6911c92b34696f351e4033b21af9ff74f0e6c49627b4987ebae87431d8fe0de4`

O0-A1 corpus-004/split-004 and its B2-failed P4 attempt remain immutable history. The O0-A2 execution uses newly reserved identities.


## O0-A3 — 1 GiB raw-audio budget with proportional MUSAN cap

**Date:** 2026-11-27
**Decision:** Approved by the owner's explicit instruction, “yes do it,” following the proposal to retain AE/DD/ITU in full and cap MUSAN to keep the combined inventoried audio payload at or below 1 GiB.
**Scope:** Applies to a new V5.1 attempt only. It does not edit corpus-005/split-005, their manifests, P4 evidence, or the O0-A2 plan/contract snapshot.
**Budget:** 1 GiB = 1,073,741,824 bytes. The current immutable P1 inventory gives 785,180,400 bytes for all non-MUSAN sources, leaving a maximum 288,561,424 bytes for MUSAN. Recompute this allowance from the new P1 inventory; stop if non-MUSAN byte identities differ from the approved basis.
**Selection:** Group MUSAN inventory rows by `group_key` so exact-byte aliases are indivisible. Allocate the MUSAN byte cap among metadata `source_class` values in proportion to their full-inventory byte totals; use floor quotas and assign the integer remainder to the final class in lexical order. Within each class, rank groups by the existing deterministic `cap` purpose and key `["musan", "budget_1g_v1", group_key]`. In rank order, include a complete group if it fits the remaining class quota; otherwise skip it and continue. Record every MUSAN group/file, byte count, class quota, rank digest, selection decision, inventory hash and selection-manifest hash. No waveform content or inferred labels may affect selection.
**Draft sizing evidence:** Applied metadata-only to the corpus-005 P1 inventory, this rule selected 58 MUSAN files/groups totalling 288,037,624 bytes. Combined with all non-MUSAN inventory files, the projected audio payload is 1,073,218,024 bytes, 523,800 bytes below the cap. This is a draft reproduction result, not a frozen membership or P1/P2/P3/P4 pass.
**New identity requirement:** Register corpus-006 and split-006 as `planned-unbuilt` before connecting selection output. Run P1 admission/register reconciliation, P2 mechanical preparation, P3 splitting/reproduction, and P4 sizing/weight gates under new attempt IDs and identities. A failed gate remains failed; no previously passing corpus-005 result transfers. P5 may run only after new P4 passes.


## O0-A4 — 1 GiB subset with minimum B0 group support

**Date:** 2026-11-27
**Decision:** Authorized as continuation of the owner's explicit instruction to proceed with the 1 GiB decision and update the study/Colab materials after O0-A3 P4 failed. O0-A3/corpus-006/split-006 remain immutable.
**Change:** Keep the exact 1 GiB combined byte cap, full AE/DD/ITU sources, and every O0-A2 sampling share/cap/gate. Revise only the MUSAN selection ordering for a new corpus-007 attempt: reserve 20 speech groups and 10 music groups by minimum byte size with deterministic hash tie-breaks, then fill remaining budget by deterministic cap rank. The core targets >=15 N3 and >=6 N4 train groups after the fixed MUSAN splits, to satisfy the unchanged 1.5% group cap.
**Selection keys:** `cap` / `["musan", "budget_1g_a4_v1_core", group_key]` for the minimum-byte core; `cap` / `["musan", "budget_1g_a4_v1_fill", group_key]` for deterministic fill.
**Limit:** Minimum-byte selection favors shorter source groups; this limitation is reported. P3/P4 must pass unchanged. Any failure remains immutable; no further selection retries under corpus-007.
**New identities:** corpus-007, split-007, derived-build-v51-005 and cnn-v51-005, all initially `planned-unbuilt`.


## O0-A5 — retain 1 GiB; amend the B0 clip cap for one new attempt

**Date:** 2026-11-28
**Decision:** The owner replied “yes do it” after the two alternatives were presented. Read together with the explicit requirement to keep AE/DD/ITU in full and cap total audio at 1 GiB, this authorizes the 1 GiB option with a B0 clip-cap amendment.
**Change:** Raise `clip_cap_reference_class_median_multiple` from 20 to 105 for a new split attempt only. 105 is the first whole-number value above the 104.963× necessary lower bound calculated for O0-A4 membership. It is not evidence that the complete B0 QP will pass. All other shares, group/source caps, margins, solver rules and gates remain fixed. No further cap increase or selection retry is authorized by O0-A5.
**Identity boundary:** Keep corpus-007 byte-identical; register split-008, derived-build-v51-006 and cnn-v51-006 before connecting any new outputs. The prior split-007 B0 failure remains immutable. P5 requires a passing new P4.
**Amended plan SHA-256:** `6a27c99f960e4e0520709216cf204e5890eca6e30ebbb438d7934f300af9f5a1`
**Amended contract SHA-256:** `6911c92b34696f351e4033b21af9ff74f0e6c49627b4987ebae87431d8fe0de4`

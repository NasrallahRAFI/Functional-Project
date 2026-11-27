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
**Amended contract SHA-256:** `c4227643919a3bf7ea1f1d8bab5a9191ae2a69470de59af6ccc56885144df827`

The prior corpus-003/split-003 and attempts remain preserved under their original hashes. Rebuild and register a new corpus/split/cache identity before connecting the amended preparation to `study-v5-1`.


## O0-A2 — exact 80% reference floor

**Date:** 2026-11-27  
**Decision:** Approved by the owner's explicit selection, “Amend O0-A2: use the exact 80% floor”.  
**Basis:** Under O0-A1, final negative ESS was 2,853.19. This exceeds exact 80% of the fixed negative reference (2,830.20) but not the ceiling-rounded floor (2,900). Positive ESS was 1,567.31 against a 1,334.28 exact 80% floor.  
**Change:** B2 uses `min(n_ref, 0.8*n_ref)` with no round-up step. The 80% reference retention, reference QP, source/stratum shares, descriptor margins, caps, solver profile and post-solve rules remain unchanged.  
**Amended plan SHA-256:** `20d2e6d1bd89be9cca7c2465db71bc53e0ed723ebdf96d3bcf061ba6fdd02d46`  
**Amended contract SHA-256:** `76d912a8563a54786cb401c7a51104e1cc61be30d7eb0130a707aba81b731839`

O0-A1 corpus-004/split-004 and its B2-failed P4 attempt remain immutable history. The O0-A2 execution uses newly reserved identities.


## O0-A3 — 1 GiB raw-audio budget with proportional MUSAN cap

**Date:** 2026-11-27  
**Decision:** Approved by the owner's explicit instruction, “yes do it,” following the proposal to retain AE/DD/ITU in full and cap MUSAN to keep the combined inventoried audio payload at or below 1 GiB.  
**Scope:** Applies to a new V5.1 attempt only. It does not edit corpus-005/split-005, their manifests, P4 evidence, or the O0-A2 plan/contract snapshot.  
**Budget:** 1 GiB = 1,073,741,824 bytes. The current immutable P1 inventory gives 785,180,400 bytes for all non-MUSAN sources, leaving a maximum 288,561,424 bytes for MUSAN. Recompute this allowance from the new P1 inventory; stop if non-MUSAN byte identities differ from the approved basis.  
**Selection:** Group MUSAN inventory rows by `group_key` so exact-byte aliases are indivisible. Allocate the MUSAN byte cap among metadata `source_class` values in proportion to their full-inventory byte totals; use floor quotas and assign the integer remainder to the final class in lexical order. Within each class, rank groups by the existing deterministic `cap` purpose and key `["musan", "budget_1g_v1", group_key]`. In rank order, include a complete group if it fits the remaining class quota; otherwise skip it and continue. Record every MUSAN group/file, byte count, class quota, rank digest, selection decision, inventory hash and selection-manifest hash. No waveform content or inferred labels may affect selection.  
**Draft sizing evidence:** Applied metadata-only to the corpus-005 P1 inventory, this rule selected 58 MUSAN files/groups totalling 288,037,624 bytes. Combined with all non-MUSAN inventory files, the projected audio payload is 1,073,218,024 bytes, 523,800 bytes below the cap. This is a draft reproduction result, not a frozen membership or P1/P2/P3/P4 pass.  
**New identity requirement:** Register corpus-006 and split-006 as `planned-unbuilt` before connecting selection output. Run P1 admission/register reconciliation, P2 mechanical preparation, P3 splitting/reproduction, and P4 sizing/weight gates under new attempt IDs and identities. A failed gate remains failed; no previously passing corpus-005 result transfers. P5 may run only after new P4 passes.

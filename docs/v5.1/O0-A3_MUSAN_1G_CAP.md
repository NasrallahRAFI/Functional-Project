# O0-A3 — 1 GiB raw-audio budget

**Owner authorization:** 27 November 2026, explicit instruction: “yes do it,” in response to retaining AE/DD/ITU in full while reducing MUSAN to keep total inventoried audio at or below 1 GiB.

This is a new-attempt amendment to the O0-A2 preparation contract. It does not modify the approved O0-A2 plan/contract copies, corpus-005, split-005, or their historical results.

## Byte budget

- Total maximum: **1,073,741,824 bytes** (1 GiB).
- Non-MUSAN sources remain complete. The current immutable inventory totals **785,180,400 bytes** for AE, DD, and ITU together.
- MUSAN allowance: at most **288,561,424 bytes**, recalculated from the new P1 inventory. A different non-MUSAN total is a stop requiring owner review.

## Deterministic selection rule

1. Use only P1 inventory fields. Never decode audio or infer labels for selection.
2. Group MUSAN inventory rows by `group_key`; exact-byte aliases remain together.
3. Partition groups by registered `source_class`. Allocate the MUSAN byte allowance proportional to each class's total bytes in the full P1 inventory. Floor each quota; give the remaining integer bytes to the lexically final class.
4. Within each class, rank groups by the established `cap` purpose with canonical key `["musan", "budget_1g_v1", group_key]`; order by complete SHA-1 digest, then canonical key.
5. Traverse that order. Include a whole group when its bytes fit the remaining class quota; otherwise skip it and continue. Never exceed any class quota or the total 1 GiB budget.
6. Record all admitted and omitted MUSAN files/groups, byte totals, class quotas, rank digests, selection reasons, full P1 inventory hash, selection-manifest hash, and the combined-byte total.

The metadata-only draft against the corpus-005 inventory selected 58 files/groups totaling 288,037,624 MUSAN bytes. Combined payload was 1,073,218,024 bytes, 523,800 bytes under the cap. The authoritative selection must be regenerated against the new P1 inventory and is not yet a split or gate pass.

## Rebuild boundary

Reserve corpus-006 and split-006 as `planned-unbuilt`. Build new P1/P2 artifacts from the complete AE/DD/ITU inputs and the selected MUSAN subset; then run new P3 deterministic split/reproduction and P4 gates. Do not edit or relabel corpus-005/split-005. Do not carry O0-A2 P4 results across the changed membership. P5 requires a passing P4 result for the new identities. Training, test/OOD scoring, and runtime activation remain outside this amendment.

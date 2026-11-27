# Project catalog

Layout established 26 November 2026 by owner instruction. The catalog checkout is repository/; active scientific development is in workspaces/study-v3/. V4 has its own branch/workspace, frozen two-source corpus/split, trained A0 model, and experimental raw-score monitor.

## Navigation

| Location, relative to the project root | Contents |
|---|---|
| repository/ | Versioned catalog, governance, source reports, registry and templates |
| workspaces/study-v3/ | Active V3 code, plan, decisions, tests and launchers |
| workspaces/study-v4/ | Separate V4 study workspace; corpus-002/split-002 prepared and frozen |
| workspaces/study-v5-1/ | Independent V5.1 preparation workspace; O0-A2 corpus-005 and O0-A3 corpus-006 are historical; latest O0-A4 corpus-007 P4 failed B0 |
| storage/sources/ | Exact imported source revisions, documentation and SHA-256 inventories |
| storage/corpora/ | Corpus membership, cleaning/group records and preparation evidence |
| storage/splits/ | Frozen split manifests and their identities |
| storage/caches/ | Existing prepared features and statistics, registered by identity |
| storage/derived/ | Frozen validation derivatives and future registered audio builds |
| runs/ | Existing experiments and their records, separated by study |
| evaluations/ | Held-out scoring records and locks |
| releases/ | Runnable package snapshots, including experimental status |
| reports/ | Generated reports tied to a study |
| local/ | Machine paths, environments, private recordings, diagnostics and temporary files |
| external/ | Preserved WSU/Wang reference files |
| archive/ | Original backups, historical documents and existing restore evidence |

## Study status

V1 is the historical multi-model study. V2 is the frozen CNN baseline. V3 is current development; its deployed A1 model is explicitly experimental and validation-ineligible. V4.0 has completed two-source data preparation and one D048 A0 CNN training run with seed 42. Salford is explicitly excluded. The corpus and frozen split are registered as corpus-002 and split-002; run cnn_a0_seed42 is frozen before test. No tuning, additional run, threshold selection or test scoring was performed. A later local V4 activation displays raw logits only on port 8766; threshold selection remains pending. Reports and evidence are recorded in reports/study-v4/.

V5.1 preparation is authorized by O0 and amendments O0-A1/A2 on 27 November 2026 on branch `study-v5-1`. O0-A2 corpus-005/split-005 and its P4 PASS remain immutable history; its P5 attempts failed. O0-A3 introduced the 1 GiB combined raw-audio cap, retaining AE/DD/ITU and selecting MUSAN under class-proportional byte quotas. Its corpus-006/split-006 passed P1/P2/P3/R-DET but failed P4 B0 because the fixed structural reference QP was primal infeasible. O0-A4 changed only the MUSAN selection order to reserve minimum-byte speech/music group coverage, retaining the cap and every fixed O0-A2 gate. New corpus-007/split-007 passed P1/P2/P3/R-DET but again failed P4 B0 with the structural reference QP primal infeasible. P5-P7 remain NOT_RUN; neither failed split is eligible for P5 or training. Further changes require a new owner-approved amendment and new identities. Prior corpus-003/004/005/006 histories remain immutable. No V5.1 training, runtime activation, or test/OOD evaluation occurred. Evidence is in `workspaces/study-v5-1/reports/v5.1/`.

Original scientific decisions remain in docs/history/original_context/PROJECT_CONTEXT_V8.md. Each study names its authoritative plan. The combined-source V4 scope supersedes the one-source restriction for V4 only. V5.1's 1 GiB raw-audio cap and O0-A4 MUSAN selection order are recorded amendments; neither changes the scoring protocol.

## Git and document authority

The V5.1 preparation authority is [DATA_PREPARATION_CNN_V5_1_FINAL.md](proposals/DATA_PREPARATION_CNN_V5_1_FINAL.md), with a [machine-readable contract](proposals/DATA_PREPARATION_CNN_V5_1_CONTRACT.json) and [metadata-only effective-size basis](proposals/DATA_PREPARATION_CNN_V5_1_EFFECTIVE_SIZE_BASIS.json). O0 is recorded in `workspaces/study-v5-1/docs/v5.1/DECISIONS.md`; the workspace holds exact copies with their hashes. Authoring/sizing checks remain distinct from preparation execution.

repository/ is the project-catalog branch checkout. workspaces/study-v3/ is the cnn-v3 checkout; study-v4 is separate. They share one Git object database, currently housed in Functional Project/.git because Windows prevented moving that open metadata directory. Do not delete or relocate that primary checkout manually. Use Git worktree operations and the migration records for a later relocation.

Project-wide catalog/governance documents are edited in repository/. V3 plans are edited in the V3 checkout. V4 plans are edited in the V4 checkout. Branch snapshots of documentation are history; they are not additional editable authorities. Cross-study registry records are edited in the catalog and synchronized explicitly when a study requires a code change.

The original Project documents tree is preserved under archive/legacy/project-documents-original. Its duplicate plans are historical copies. Generated HTML reports now live under reports/study-v1/original-model-report. Content and presentation were not redesigned during relocation.

## Compatibility

The old root Functional Project directory contains links to V3 and the shared Git metadata; it is a compatibility view. Project documents is a compatibility view of historical documents. V3's data/results/runtime/archive entries expose the registered storage through junctions. These views preserve frozen path-dependent scripts and evidence; they contain no second payload copy. Newly registered studies must declare their own outputs and dependencies rather than mounting the V3 result tree.

Use `python scripts/project_layout.py check` from a checkout for registry and mount checks. See docs/migration/RESTRUCTURING.md for the move map, preserved working state, integrity checks and known provenance gaps.

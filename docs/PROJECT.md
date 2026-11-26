# Project catalog

Layout established 26 November 2026 by owner instruction. The catalog checkout is repository/; active scientific development is in workspaces/study-v3/. V4 has its own branch/workspace and remains a scope scaffold with no built corpus or split.

## Navigation

| Location, relative to the project root | Contents |
|---|---|
| repository/ | Versioned catalog, governance, source reports, registry and templates |
| workspaces/study-v3/ | Active V3 code, plan, decisions, tests and launchers |
| workspaces/study-v4/ | Separate V4 study workspace; corpus-002/split-002 prepared and frozen |
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

V1 is the historical multi-model study. V2 is the frozen CNN baseline. V3 is current development; its deployed A1 model is explicitly experimental and validation-ineligible. V4.0 has completed two-source data preparation and one D048 A0 CNN training run with seed 42. Salford is explicitly excluded. The corpus and frozen split are registered as corpus-002 and split-002; run cnn_a0_seed42 is frozen before test. No tuning, additional run, threshold selection, test scoring, bundle or runtime activation was performed. Reports and evidence are recorded in reports/study-v4/.

Original scientific decisions remain in docs/history/original_context/PROJECT_CONTEXT_V8.md. Each study names its authoritative plan. The combined-source V4 scope supersedes the one-source restriction for V4 only; no numerical data budget or new scoring protocol is invented here.

## Git and document authority

repository/ is the project-catalog branch checkout. workspaces/study-v3/ is the cnn-v3 checkout; study-v4 is separate. They share one Git object database, currently housed in Functional Project/.git because Windows prevented moving that open metadata directory. Do not delete or relocate that primary checkout manually. Use Git worktree operations and the migration records for a later relocation.

Project-wide catalog/governance documents are edited in repository/. V3 plans are edited in the V3 checkout. V4 plans are edited in the V4 checkout. Branch snapshots of documentation are history; they are not additional editable authorities. Cross-study registry records are edited in the catalog and synchronized explicitly when a study requires a code change.

The original Project documents tree is preserved under archive/legacy/project-documents-original. Its duplicate plans are historical copies. Generated HTML reports now live under reports/study-v1/original-model-report. Content and presentation were not redesigned during relocation.

## Compatibility

The old root Functional Project directory contains links to V3 and the shared Git metadata; it is a compatibility view. Project documents is a compatibility view of historical documents. V3's data/results/runtime/archive entries expose the registered storage through junctions. These views preserve frozen path-dependent scripts and evidence; they contain no second payload copy. Newly registered studies must declare their own outputs and dependencies rather than mounting the V3 result tree.

Use `python scripts/project_layout.py check` from a checkout for registry and mount checks. See docs/migration/RESTRUCTURING.md for the move map, preserved working state, integrity checks and known provenance gaps.

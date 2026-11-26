# Restructuring verification — 26 November 2026

- Original inventory: 18,269 files. 18,264 retain their original byte hashes. Five existing code/test files have intentional, reviewed changes; originals are preserved in the local working-source snapshot.
- Imported sources: 24,981 files copied and checked byte-for-byte against the original downloads (23,410 Al-Emadi; 1,396 thesis dataset; 175 Salford).
- Active V3 unit suite: 110 passed, 25 subtests passed, 22 warnings; 127.12 seconds. Final run used a dedicated local temporary directory.
- Active A1 detector --check: ready. Model SHA-256 7847c73d905811bd53b164ebb923a20e95eccac41d32b478585c0aa658704fed; threshold 1.9109516143798828; experimental_activation=true; validation_eligible=false; test_access=false. Existing three failed guards remain recorded.
- Registry checks: all existing records, split/manifest identities, deployment selection, dependency references and 44 compatibility views passed. corpus-002/split-002 and V4 are explicitly planned/unbuilt.
- V4 scaffold: own branch and configuration namespace; no V3 data/results/runtime mounts or selected package. No V4 training or corpus construction was performed.
- Original archive payloads, weights, prepared arrays, configs and evaluation locks retain their bytes. No new full-project ZIP was created.
- Pre-existing edits to scripts/run_frontend_correction_baseline.py, untracked work, and results/model_report deletion state were preserved and excluded from this task's commit.

Initial checks found the migrated package's display path needed to remain relative to its compatibility view; this was corrected. Existing launcher/release tests were bound to V2 despite the A1 selection and now verify the selected known release and experimental state. An inaccessible pre-existing system pytest-current directory was bypassed with a dedicated local --basetemp; system temporary files were not removed.

Exact original inventory, working patches/source snapshot, copy manifests, move journal, integrity results and JUnit XML are local under local/migration/2026-11-26. Private diagnostic inputs and machine paths are not included in the publication.

The common Git database remains in Functional Project/.git because Windows held it open. Functional Project and Project documents are retained compatibility directories; deleting them as duplicates would break existing links or Git metadata. This is a documented platform-specific difference from the proposed top-level rename.

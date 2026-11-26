# Restructuring record — 26 November 2026

The owner-approved migration physically relocates the primary V3 checkout and scientific payloads, adds registered sources, and creates independent catalog/V4 worktrees. Frozen configs, arrays, labels, split memberships, checkpoints, thresholds, package manifests, original archives and evaluation records retain their bytes.

The original working state is preserved locally in local/migration/2026-11-26/: Git HEAD/branch, staged and unstaged binary patches, source-file snapshot, original status, original file inventory with hashes, a move journal and verification results. This record is private/local. A sanitized path map is versioned in this directory. Pre-existing model-report deletions and the modified run_frontend_correction_baseline.py were not adopted into the restructuring commit.

Legacy data paths are documented compatibility views of canonical storage. New versions must use their own outputs. Existing archives remain unchanged; restore folders remain under archive/restore_checks. This migration adds no new full-project ZIP.

Known provenance gap: the preserved external WSU/Wang reference folder has a .git pointer to an absent main repository at its former location. Its files are preserved, but Git history is not reconstructed. Some historical runs predate the new registry and have incomplete source/environment records; their registries point to existing evidence and do not assert reconstructed provenance.

The project-catalog checkout owns global registry/governance documents. V3 owns its scientific plan and implementation. V4 is a separate scope scaffold, without generated data or a selected release. The common Git object store remains physically in Functional Project/.git because Windows prevented moving that open metadata directory; other worktrees refer to it.

Verification outcomes and exact counts are recorded in VERIFICATION.md after migration checks complete.

Windows also held the Project documents root open. Its entries were moved individually; the old directory is now a compatibility view using directory junctions and a hard-linked historical context file. Functional Project likewise contains compatibility entries plus the common .git directory. These retained roots must not be deleted as redundant folders.

# Storage and identity

Git contains code, documents, small registry records and templates. Raw audio/video, checkpoints, prepared arrays, browser profiles, virtual environments, local machine paths and private diagnostic inputs stay outside Git.

Source files are imported once into storage/sources/<source>/<revision>/original. The original download layout is preserved, including the source's own binary/multiclass duplication. checksums/files.csv binds relative paths, byte counts and SHA-256. Downloads outside the project are retained. Original backup ZIPs remain separate recovery objects.

A corpus selects source files or sample intervals and records labels, exclusions and group identities. It may reference raw files without writing duplicate WAVs. A split binds membership and assignments. A derived build identifies any persisted transformation. Cache identities bind their input membership and actual processing contract; reuse requires matching identities. Retrospective registry entries preserve existing hash domains rather than replacing them with newly invented hashes.

Canonical payload directories are outside all Git worktrees. Directory junctions support existing V3 scripts. V1/V2 artifacts are baseline dependencies; never regenerate them from V3. Future studies get their own output roots. Do not use broad recursive copy, deletion or archiving commands through junctions.

Local path values live in local/paths.local.json. Each study has its own environment under local/environments. Existing Windows Python executables remain reachable through the preserved .venv/legacy absolute-path links; use python -m pip rather than relocated pip launchers. Rebuild environments from locked dependencies if those compatibility links are later removed.

Backups must include the Git history plus referenced source and artifact objects. A catalog and checksums do not replace payloads. A same-disk archive does not protect against disk failure. Existing restore folders are retained under archive/restore_checks; no restored payload was deleted during this migration.

After restoring shared payloads, preview compatibility links with `python scripts/setup_workspace.py --study study-v3`. Add `--apply` to create only missing links. Existing entries are never overwritten. The registered views and their targets are in registry/compatibility/study-v3.json.

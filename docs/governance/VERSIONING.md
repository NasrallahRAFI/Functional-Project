# Versioning

A study version identifies code, source membership, split, frontend, training protocol, evaluation and runtime policy. It is broader than network architecture.

Source revisions, corpora, splits, derived builds, caches, runs and releases have independent identifiers. V2 and V3 explicitly share corpus-001/split-001 and baseline cache dependencies. V4 has reserved identifiers corpus-002/split-002 with status planned; no empty manifest is presented as a completed dataset.

Record every execution in a new run directory. Record parent run, exact code commit or working-source snapshot, resolved configuration, environment, seed and dependency identities. Completed runs and releases are immutable; a corrected or resumed execution receives a new identifier with its parent recorded. Existing legacy runs keep their historical names and records; missing provenance is marked unknown rather than manufactured.

Changes to a study's waveform processing or browser runtime are committed to that study branch. Sharing a source dataset does not imply sharing mutable code. Reuse code through reviewed Git changes; do not junction one branch's src directory into another.

Release packages bind model, frontend, normalization, threshold and temporal decision policy. Experimental selection and failed validation guards remain part of release identity. Existing held-out evaluation locks and test-use history follow the source groups across studies; renaming or splitting again does not erase prior access.

Historical source archives may include dirty working files absent from Git commits. Existing V1/V2 archives and their checksums remain the authority for those captured states. No complete V1 source tag is claimed by this migration.

# Study V4

Status: data preparation planned; corpus and split unbuilt. Branch: study-v4. Parent code snapshot: c357afb (CNN V3 A1 activation).

The authoritative preparation plan is [DATA_PREPARATION_CNN_V4_0.md](docs/v4/DATA_PREPARATION_CNN_V4_0.md), imported from the owner's supplied revision dated 26 November 2026. The identifier remains CNN V4.0. [specification.md](docs/v4/specification.md) binds that plan to the current workspace layout and records implementation wording that remains to be clarified.

Included sources: al-emadi-r001 (Binary_Drone_Audio only) and thesis-r001 (Data/Audio only). Salford is explicitly excluded from V4.0. Its original source registration and stored files remain separate from V4 membership.

The planned data identities are corpus-002 and split-002. Both remain planned/unbuilt. No source inventory, segmentation, new manifest, split, frontend cache, statistics fit, training, scoring or runtime activation was performed by this document update.

The plan specifies binary pooling, Al-Emadi cleaning/grouping rules, 1-second dd segments, duplicate checks, source/flag audits, and the frozen group-aware train/validation/test split procedure. It retains the 16 kHz, 64-band, 7,936-sample / (1,64,28) frontend contract. New caches, statistics, models, thresholds and packages are subsequent work outside the preparation plan.

V4 uses its own outputs and has no V3 data/results/runtime mounts or selected package. Historical V2/V3 configurations remain under docs/history/inherited-config. Existing test-use history must accompany source material retained in V4; a new split is not evidence that those recordings were previously unseen.

Decision record: [V4-PLAN-001.md](docs/v4/decisions/V4-PLAN-001.md). Global source/corpus/split registry authority: repository/registry in the enclosing project folder.

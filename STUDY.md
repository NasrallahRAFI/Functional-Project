# Study V4

Status: V4.0 data preparation passed; corpus and split frozen. No V4 model, feature cache, normalization fit, threshold, bundle, or runtime selection exists. Branch: study-v4. Parent code snapshot: c357afb (CNN V3 A1 activation).

The authoritative preparation plan is [DATA_PREPARATION_CNN_V4_0.md](docs/v4/DATA_PREPARATION_CNN_V4_0.md), imported from the owner's supplied revision dated 26 November 2026. The identifier remains CNN V4.0. [specification.md](docs/v4/specification.md) binds the plan to the current workspace layout and records approved implementation definitions.

Included sources: al-emadi-r001 (Binary_Drone_Audio only) and thesis-r001 (Data/Audio only). Salford is explicitly excluded from V4.0. Its original source registration and stored files remain separate from V4 membership.

The prepared data identities are corpus-002 and split-002. See the preparation report and requirement audit for exact counts, gates, hashes, and limitations. The train/validation/test manifests are frozen. Source inventory, DD segmentation, corpus cleanup, and split creation are complete; no frontend cache, statistics fit, training, scoring, or runtime activation was performed.

The plan specifies binary pooling, Al-Emadi cleaning/grouping rules, 1-second dd segments, duplicate checks, source/flag audits, and the frozen group-aware train/validation/test split procedure. It retains the 16 kHz, 64-band, 7,936-sample / (1,64,28) frontend contract. New caches, statistics, models, thresholds and packages are subsequent work outside the preparation plan.

V4 uses its own outputs and has no V3 data/results/runtime mounts or selected package. Historical V2/V3 configurations remain under docs/history/inherited-config. Existing test-use history must accompany source material retained in V4; a new split is not evidence that those recordings were previously unseen.

Decision record: [V4-PLAN-001.md](docs/v4/decisions/V4-PLAN-001.md). Global source/corpus/split registry authority: repository/registry in the enclosing project folder.

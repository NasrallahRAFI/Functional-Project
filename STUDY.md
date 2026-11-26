# Study V4

Status: V4.0 corpus and split frozen; one A0 CNN trained with seed 42. Experimental local raw-score monitoring is active; threshold selection and test scoring remain pending. No additional model was trained. Branch: study-v4. Parent code snapshot: c357afb (CNN V3 A1 activation).

The authoritative preparation plan is [DATA_PREPARATION_CNN_V4_0.md](docs/v4/DATA_PREPARATION_CNN_V4_0.md), imported from the owner's supplied revision dated 26 November 2026. The identifier remains CNN V4.0. [specification.md](docs/v4/specification.md) binds the plan to the current workspace layout and records approved implementation definitions.

Included sources: al-emadi-r001 (Binary_Drone_Audio only) and thesis-r001 (Data/Audio only). Salford is explicitly excluded from V4.0. Its original source registration and stored files remain separate from V4 membership.

The prepared data identities are corpus-002 and split-002. See the [preparation report](reports/DATA_PREPARATION_CNN_V4_0_REPORT.md) and [requirement audit](reports/V4_PREPARATION_REQUIREMENT_AUDIT.md) for data counts, gates, hashes, and limitations, and the [single-model training report](reports/V4_CNN_TRAINING_REPORT.md) for the one seed-42 training run. The train/validation/test manifests are frozen. Canonical train/validation caches and train-only normalization statistics were produced. Test audio remains unaccessed; no threshold or test evaluation was produced. The later V4-LIVE-001 activation uses the checkpoint and fitted training statistics directly.

The plan specifies binary pooling, Al-Emadi cleaning/grouping rules, 1-second dd segments, duplicate checks, source/flag audits, and the frozen group-aware train/validation/test split procedure. It retains the 16 kHz, 64-band, 7,936-sample / (1,64,28) frontend contract. Cache construction, normalization fitting and the single model training followed the preparation plan. Threshold selection and packaging remain separate future stages.

V4 uses its own outputs and has no V3 data/results/runtime mounts or V3 package. The V4 monitor reads its registered V4 checkpoint and training normalization directly. Historical V2/V3 configurations remain under docs/history/inherited-config. Existing test-use history must accompany source material retained in V4; a new split is not evidence that those recordings were previously unseen.

Decision records: [V4-PLAN-001.md](docs/v4/decisions/V4-PLAN-001.md), [V4-IMPL-001.md](docs/v4/decisions/V4-IMPL-001.md), [V4-TRAIN-001.md](docs/v4/decisions/V4-TRAIN-001.md), [V4-LIVE-001.md](docs/v4/decisions/V4-LIVE-001.md), and [V4-LIVE-002.md](docs/v4/decisions/V4-LIVE-002.md). Global source/corpus/split/run registry authority: repository/registry in the enclosing project folder.

Live monitor: run `start_detector.bat` from this workspace, or open http://127.0.0.1:8766/live/index.html once the service is ready. It reports raw logits only, with threshold pending. See docs/v4/decisions/V4-LIVE-001.md.

Supplemental threshold analysis: [V4_MUSAN_NOISE_SUPPLEMENT.md](reports/V4_MUSAN_NOISE_SUPPLEMENT.md). MUSAN is external negative-only evidence, not a replacement for representative 24-hour live ambience. Threshold remains pending.

Live monitor: run `start_detector.bat` from this workspace, or open http://127.0.0.1:8766/live/index.html once the service is ready. It reports raw logits only, with threshold pending. See docs/v4/decisions/V4-LIVE-001.md.

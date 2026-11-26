# CNN V4.0 preparation specification

## Authority and status

The owner supplied DATA_PREPARATION_CNN_V4_0_UPDATED_FINAL.md on 26 November 2026 and requested a V4 plan update. Its contents are imported into [DATA_PREPARATION_CNN_V4_0.md](DATA_PREPARATION_CNN_V4_0.md). That document is authoritative for the numerical and scientific preparation rules. The plan has now been executed; results and gate evidence are in `reports/study-v4/DATA_PREPARATION_CNN_V4_0_REPORT.md` and `reports/study-v4/V4_PREPARATION_REQUIREMENT_AUDIT.md`.

This two-source revision supersedes the earlier three-source V4 scaffold. V4.0 includes Al-Emadi binary audio and Drone-detection-dataset audio. Salford is excluded because the supplied structure lacks independently verified segment-level drone/non-drone labels; the rejected energy/activity gate does not establish source identity. Salford is not a V4 source, audit input or manifest member. Its global source record remains available as historical/source inventory.

## Declared preparation contract

- Labels: Al-Emadi yes_drone and dd DRONE are class 1; Al-Emadi unknown plus dd HELICOPTER and BACKGROUND are class 0.
- Al-Emadi: retain the supplied V8 rejection, zero-deduplication, grouping, flagging and train-pinning rules. Its files remain referenced at their raw paths.
- dd: channel 0, soxr HQ resampling to 16 kHz, non-overlapping 16,000-sample segments, remainder discarded, float32 WAV, no level normalization or dither; one group per original file.
- Cleaning/audits: decoded float32 SHA-256 duplicate checks, cross-class duplicate STOP, the specified silence rule, descriptor/flag summaries and source-only shortcut audit. The weak_positive flag retains its supplied definition and keeps flagged positives.
- Splitting: fixed seed from the project configuration; class|source strata; two-stage StratifiedGroupKFold with 7 folds then 6; pinned groups stay in train. G-GRP, G-DUP, G-FRAC, G-POS and G-STR remain exactly as supplied. Failed gates or contradicted expected facts require the plan's STOP procedure.
- Frontend smoke check: training split only; 20 clips per included source; finite (1,64,28) windows. The frontend's numerical contract is retained.
- Preparation deliverables are complete and frozen. After preparation, only train/validation caches and train-eligible normalization were built, followed by one D048 A0 CNN training run using seed 42 and the fixed D049 recipe. No search, extra run, threshold selection, test access, bundle, or runtime activation was performed; details are recorded in `reports/V4_CNN_TRAINING_REPORT.md`.

## Paths in the current layout

All canonical paths below are relative to the enclosing ML Course Project folder. The corpus, audits, DD clips, split manifests, and reports have been generated at these destinations. The supplied plan's legacy relative paths are logical names; they must not resolve into a V1–V3 compatibility tree.

| Supplied plan path/reference | Current binding |
|---|---|
| Al-Emadi raw binary input | storage/sources/al-emadi/al-emadi-r001/original/Binary_Drone_Audio/ |
| dd audio input | storage/sources/drone-detection-thesis/thesis-r001/original/Data/Audio/ |
| data/interim/v4/clips/ | storage/derived/corpus-002/dd-clips/ |
| data/interim/v4/manifest_v4_full.csv | storage/corpora/corpus-002/manifest_v4_full.csv |
| data/interim/v4/audits/ | storage/corpora/corpus-002/audits/ |
| data/interim/v4/environment_log_v4.json | storage/corpora/corpus-002/environment_log_v4.json |
| data/interim/v4/source_licences_v4.md | storage/corpora/corpus-002/source_licences_v4.md |
| data/processed/splits/v4/ | storage/splits/split-002/ |
| reports/DATA_PREPARATION_CNN_V4_0_REPORT.md | reports/study-v4/DATA_PREPARATION_CNN_V4_0_REPORT.md |
| PROJECT_CONTEXT_V8.md | docs/history/original_context/PROJECT_CONTEXT_V8.md in the V4 checkout |
| Frontend correction plan V1.0 | archive/legacy/project-documents-original/Front end/correction plan V1.0.md |
| Historical config.yaml seed reference | docs/history/inherited-config/config.yaml, seed=42 and frontend.manifest_seed=42 |

The historical configuration remains a read-only reference. V4 now has its own active `config.yaml`, seed 42, and recorded output bindings; the environment log records resolved source paths and the execution environment. The original download path printed in the supplied plan remains source-document context; the registered revision paths identify the preserved project inputs.

## Approved implementation definitions

The supplied document is preserved verbatim rather than silently amended. The owner approved the following operational definitions before the successful execution; the implementation decision record is `decisions/V4-IMPL-001.md`:

1. The source-only audit uses the pooled surviving pre-split clips to determine each source's majority label. A tie remains a STOP condition.
2. `weak_positive` uses the absolute difference from the median BACKGROUND band level, with a threshold of 3 dB. Band power uses SciPy's one-sided periodogram with boxcar window, no detrending, inclusive 100–4,000 Hz bins, and amplitude-1 reference.
3. Al-Emadi's explicit zero-content representative and filename preference is applied before the general duplicate rule; the nine prescribed representatives are retained.
4. V4 has its own active `config.yaml` with seed 42. The archived historical config remains a read-only reference.

The supplied introduction refers to phase F audit statistics, while descriptor/shortcut audits are described in phase E. This cross-reference discrepancy has no effect on the implemented numerical rules; no phase or rule has been renumbered. P4-07 is absent from the supplied decision table and no replacement plan decision has been invented.

Prior Al-Emadi test-use history remains applicable to reused source groups. The future V4 report must distinguish a newly frozen split from previously unseen recordings, as required by the project versioning rules.

## Later local monitoring activation

The owner subsequently authorized activation of the frozen A0 seed-42 CNN for raw scores only. V4-LIVE-001 records this separate step. The V4 monitor uses port 8766; no threshold, extra training run or test evaluation was introduced. See reports/V4_LIVE_INTEGRATION_REPORT.md.

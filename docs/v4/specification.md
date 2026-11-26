# CNN V4.0 preparation specification

## Authority and status

The owner supplied DATA_PREPARATION_CNN_V4_0_UPDATED_FINAL.md on 26 November 2026 and requested a V4 plan update. Its contents are imported into [DATA_PREPARATION_CNN_V4_0.md](DATA_PREPARATION_CNN_V4_0.md). That document is authoritative for the numerical and scientific preparation rules; the integration notes here identify current paths and unresolved wording. This update records a plan. No preparation phase has been executed and no gate is marked passed.

This two-source revision supersedes the earlier three-source V4 scaffold. V4.0 includes Al-Emadi binary audio and Drone-detection-dataset audio. Salford is excluded because the supplied structure lacks independently verified segment-level drone/non-drone labels; the rejected energy/activity gate does not establish source identity. Salford is not a V4 source, audit input or manifest member. Its global source record remains available as historical/source inventory.

## Declared preparation contract

- Labels: Al-Emadi yes_drone and dd DRONE are class 1; Al-Emadi unknown plus dd HELICOPTER and BACKGROUND are class 0.
- Al-Emadi: retain the supplied V8 rejection, zero-deduplication, grouping, flagging and train-pinning rules. Its files remain referenced at their raw paths.
- dd: channel 0, soxr HQ resampling to 16 kHz, non-overlapping 16,000-sample segments, remainder discarded, float32 WAV, no level normalization or dither; one group per original file.
- Cleaning/audits: decoded float32 SHA-256 duplicate checks, cross-class duplicate STOP, the specified silence rule, descriptor/flag summaries and source-only shortcut audit. The weak_positive flag retains its supplied definition and keeps flagged positives.
- Splitting: fixed seed from the project configuration; class|source strata; two-stage StratifiedGroupKFold with 7 folds then 6; pinned groups stay in train. G-GRP, G-DUP, G-FRAC, G-POS and G-STR remain exactly as supplied. Failed gates or contradicted expected facts require the plan's STOP procedure.
- Frontend smoke check: training split only; 20 clips per included source; finite (1,64,28) windows. The frontend's numerical contract is retained.
- Deliverables: standardized dd clips, full manifest, three split manifests, audit files, environment/license records and a preparation report. Caches, fitted statistics, model training, thresholds, packages and final scoring are subsequent work.

## Paths in the current layout

All canonical paths below are relative to the enclosing ML Course Project folder. These are planned destinations, not existing generated datasets. The supplied plan's legacy relative paths are logical names; they must not resolve into a V1–V3 compatibility tree.

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

The historical configuration is a read-only reference, not an activated V4 configuration. This update creates no config.yaml and no output mounts. An implementation must establish a V4-owned active configuration, preserve the specified seed, bind these destinations explicitly, and log the resolved source paths. The original download path printed in the supplied plan remains source-document context; the registered revision paths above identify the preserved project inputs.

## Implementation wording to resolve before execution

The supplied document is preserved rather than silently amended. These are documentation observations, not measured gate failures:

1. Phase E calls the source-only rule the 'training-majority class' while requiring the audit before a split exists. Specify which pre-split population determines that majority before implementing this calculation.
2. 'Within 3 dB' for weak_positive does not define whether the comparison is an absolute difference or a one-sided bound. The 100–4,000 Hz band-RMS computation also needs an exact filter/estimator and reference convention. No method or inequality is selected by this update.
3. Phase D's general 'keep the first sorted clip id' duplicate rule may conflict with Al-Emadi's named-silence preference and preservation of nine length-distinct zero representatives. Establish precedence for those exceptions before implementing general deduplication.
4. The literal V4 config.yaml referenced for the seed does not yet exist. The archived reference has seed 42; activating the V4 configuration remains implementation work.

The intro refers to phase F audit statistics, while descriptor/shortcut audits are described in phase E. This is recorded as a cross-reference discrepancy; no phase or rule has been renumbered. P4-07 is absent from the supplied decision table and no replacement decision has been invented.

Prior Al-Emadi test-use history remains applicable to reused source groups. The future V4 report must distinguish a newly frozen split from previously unseen recordings, as required by the project versioning rules.

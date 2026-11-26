# Study V3

Status: active development. Git branch: cnn-v3. Parent: CNN V2.0.

Authoritative scientific plan: docs/CNN_V3_0_IMPLEMENTATION_PLAN.md.
Decisions: docs/CNN_V3_0_DECISION_LOG.md. Original global context: docs/history/original_context/PROJECT_CONTEXT_V8.md.

Data: corpus-001 and split-001. Registered sources: Al-Emadi binary membership only. V2 cache and frozen V2 release are explicit baseline dependencies. Imported thesis and Salford data are registered source material; they do not enter V3 training.

Current detector selection is the A1 seed-42 experimental package. Its selection records `validation_eligible=false`, its three failed guards, and `test_access=false`. Relocation does not promote this model or alter its threshold.

Canonical outputs, relative to the enclosing project folder:

- runs/study-v3/experiments/: existing named experiments.
- runs/study-v3/preparation/: calibration, environment records and review evidence.
- storage/derived/v3-validation-suite/: frozen validation conditions.
- storage/caches/v3-raw-train/: raw-loader cache evidence.
- releases/study-v3/A1-experimental/: deployed package snapshot.
- local/private_recordings/: private diagnostic inputs.

Use `python scripts/project_layout.py check` to verify the directory registry.
Use scripts/with_study.ps1 to run Python with the registered raw-data path.
Large output paths are supplied by documented directory junctions for compatibility with frozen scripts and records.

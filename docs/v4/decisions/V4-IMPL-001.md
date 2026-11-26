# V4-IMPL-001 — Approved audit definitions and implementation bindings

The owner approved the proposed definitions in this chat before implementation:

- The source-only audit predicts each source's majority class in the pooled surviving data before a split exists. This is a descriptive shortcut audit, not a trained-model or test evaluation.
- weak_positive is true for dd DRONE segments whose 100–4,000 Hz level differs from the BACKGROUND-segment median by an absolute amount no greater than 3 dB. The estimator is SciPy's one-sided periodogram, boxcar window, no detrending, density scaling; sum inclusive band bins times frequency-bin width and convert power to dB relative to amplitude 1.

The active V4 config.yaml copies the archived seed 42, frontend constants and cleaning thresholds. No statistics, weights or previous split assignment are activated. Outputs use the paths in specification.md. AE's explicit B1.3 named-silence preference takes precedence over the general duplicate rule, which says those AE cases are already handled by B1.3.

RMS descriptors reuse the existing -240 dB zero-energy sentinel. 99%-power rolloff uses the cumulative whole-clip periodogram. Crop/eligibility flags share the existing cache-index implementation exposed as features.power_quality_flags; existing silent/near-silent/clipped flags and splits.flag_columns supply the remaining definitions. Short rejected files have no valid crop: crop_all_zero and stats_eligible are recorded as not applicable, not fabricated values, and never enter retained manifests.

The preparation interpreter is reused from the existing pinned environment without installation or mutation. All imported project code comes from the V4 checkout through PYTHONPATH and all generated files go to V4 storage. Dependency versions, interpreter, source paths, config/plan/code hashes are logged by A1.

The frontend smoke test uses identity mean/std placeholders only to exercise the canonical (1,64,28) tensor interface. It does not fit, persist or reuse training normalization statistics. No mel cache, model, threshold or runtime package is produced.

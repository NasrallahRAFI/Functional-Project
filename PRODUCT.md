# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Assumption from the supplied brief: the primary reader is the project owner or reviewer who needs to understand every evaluated model and verify the final comparison without opening result files.

## Product Purpose

This is an offline research report for the drone audio classification project. It brings the frozen model results, tuning history, comparison evidence, and reproducibility details into five shared sections plus one dedicated page for each of the 14 frozen model entries.

## Positioning

The report exposes the full model selection trail and the locked test protocol, including uncertainty and known limitations, rather than presenting a leaderboard alone.

## Operating Context

The pages are generated locally from the project's CSV and JSON results. The report must work without a network connection when the complete output folder is opened or shared.

## Capabilities and Constraints

- No model is refit to build or view this report.
- Final test scores are read from the locked score file and use validation-frozen thresholds.
- The report covers 14 frozen model fits across 11 model families and includes tuning variants where a run ledger exists.
- Any conclusions are limited by validation reuse, the available drone types, and possible artefact confounding.

## Brand Commitments

The user requires this brand surface to follow the product lane “premium web presence for content creators and local businesses — Designs of Desire studio. Never generic SaaS or corporate template look.” This research report uses a project-specific editorial data treatment while keeping its subject and evidence clear; it must not invent commercial claims or customer proof.

## Evidence on Hand

- `results/tables/test_scores.csv` and `test_scored.lock`
- Model run ledgers under `results/tables/runs_*.csv`
- Model, CNN stage, operating-point, recall, and paired-comparison reports under `results/tables/`

## Product Principles

- Make the final evidence easy to inspect.
- Keep model selection and final test evaluation distinct.
- Show uncertainty alongside point estimates.
- Preserve reproducibility metadata and caveats.

## V4 local acoustic monitor

The owner uses the existing web monitor to capture microphone audio locally and inspect CNN V4.0 A0 seed-42 raw logits. One trained model has 23,585 parameters. Its threshold has not been selected, so the monitor issues no binary detection verdict and labels threshold selection pending. Test audio remains locked. The existing brand surface uses the Designs of Desire studio lane and the project's listed anti-references; this task changes model status and readout behavior.

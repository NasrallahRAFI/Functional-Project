# CNN V3.0 ONE_CHANNEL_seed42 report

Date: 26 November 2026
Run: `ONE_CHANNEL_seed42`
Status: **Completed; validation-ineligible; active locally as an experimental model at the owner's request.**

## Run and selection

The single seed-42 run used the registered fixed depth_4 CNN (60,641 parameters) and the combined A4 channel recipe. It completed 31 epochs and 3,689 optimizer steps, stopping under weighted-validation-loss patience 8. Epoch 23 was selected with weighted validation loss 0.3419212. The model was calibrated once at raw-logit threshold 1.7455270290, selected by maximum weighted pooled validation F1 with ties to the higher logit.

The candidate's weighted pooled validation F1 is 0.8849302 and weighted PR-AUC is 0.9294718. Frozen V2 scored 0.7092360 weighted pooled F1 on the same suite at its historical threshold. A1 scored 0.7302053 under the same V3 weighted-pool selection protocol. These validation comparisons are descriptive and remain subject to selection optimism; this single seed does not estimate training-seed variance.

## Frozen validation guards

| Guard | Result | Outcome |
|---|---:|---|
| Clean recall ≥159/185 | 158/185 | Fail by 1 positive |
| Clean false positives ≤5/1,325 | 7/1,325 | Fail by 2 false positives |
| Mild FAR ≤1% | 0.528% (7/1,325) | Pass |
| Moderate FAR ≤1% | 0.906% (12/1,325) | Pass |
| Severe FAR ≤1% | 0.604% (8/1,325) | Pass |
| Mean augmented recall gain ≥5 percentage points vs V2 | +56.76 points | Pass |
| No augmented severity >3 points below V2 | All three improve | Pass |

Overall validation eligibility is **false**. Clean validation recall is 0.8541 and clean F1 is 0.9029. Mild, moderate and severe recall are 0.8162, 0.8054 and 0.7838. The augmented recall gain is substantial on this reused simulation suite; it does not demonstrate performance on phones, speakers, rooms or new drone types.

The failed validation gate ended the proposal. The held-out test was not opened, no test score or final benchmark claim exists, and no additional fit or threshold search was run.

## Package and frontend

The verified package is `releases/study-v3/ONE_CHANNEL_seed42/`. It contains the model, V2-compatible frontend and training statistics, golden fixtures, model card, software verification, and per-member SHA-256 manifest. TorchScript model SHA-256: `f727446758512fc1fd7e2a1f6b81021503a878bf1bffb6560ff54bb634206a1c`. Checkpoint SHA-256: `615694d0c3af6b3ca2d410826e7719ce70cf38bb776e893b946c2e8862ba00c7`. Frontend identity: `dab196aee59c482e2938d90ea0c55d33eb5606f8aa04d11ff8d441fb6cd31678`.

At the owner's request, the exact candidate is mounted locally through `runtime/assets/CNN_V3.0_ONE_CHANNEL`. The live page and API label it experimental and validation-ineligible, name its two failed clean guards, and identify CNN V2.0 as the validated baseline. The 0.496-second cadence and three-window majority remain unchanged. No final-test acceptance claim is made.

The owner has explicitly frozen the runtime smoothing policy at `N = 3` (decision V3-011). The system thresholds each independent CNN window score and uses a strict majority of the latest three votes. Prior votes do not enter the CNN as model inputs. Keep this policy unchanged unless the owner explicitly revises the decision; three fresh contiguous windows take approximately 1.488 seconds at the current cadence.

## Verification

- Workspace registry/layout check: pass.
- A4 effective augmentation config and source hashes: match the training-only calibration records.
- Focused augmentation, raw/cache parity and test-access suite: 13 passed.
- Live detector, launcher and package rejection suite: 17 passed, including 16, 44.1 and 48 kHz inference parity checks.
- Browser lifecycle/capture suite: 5 passed.
- Package export: pass; all four validation confusion matrices reproduced, V2 frontend golden checks passed, and eager/TorchScript parity was within tolerance.
- `scripts/live_detector.py --check`: pass; API reports the ONE_CHANNEL model hash, threshold, experimental state and guard failures.
- Impeccable live detector scan: no findings. Desktop and 390 px mobile captures were inspected through Chrome DevTools Protocol; the mobile document has no horizontal overflow.

Screenshots and runtime-only artifacts stay under `local/runtime/study-v3/` and are not part of the source commit.

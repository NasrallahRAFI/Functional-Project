# Local drone detector integration plan

Date: 2026-11-25. Status: implementation specification; acceptance work below is pending.

Deliver a Windows application that starts when the owner double-clicks `Functional Project/start_detector.bat`. It must verify and load the selected CNN, start the local service, open the browser after readiness, and analyze microphone audio continuously after the owner grants browser permission and presses Start monitoring. Python setup may require internet on the first run; subsequent launches must work offline with the installed environment and local model assets.

This plan extends the completed audio frontend correction. The governing source is `Project documents/Front end/correction plan V1.0.md`, especially its final correction report. Archived documents provide history; they do not supersede that report. Existing G1–G15 evidence covers the frontend, including capture with a stub scorer. This task must establish the complete browser-to-CNN path separately.

Paths in coding instructions are relative to `D:/Work/ML Course Project/Functional Project` unless stated otherwise. Canonical version of this plan: `docs/LOCAL_DETECTOR_IMPLEMENTATION_PLAN.md`; an identical reading copy belongs in `Project documents/Front end/Local detector integration plan V1.0.md`.

## Model selection and frozen operating point

Use the registered main CNN, Stage C `depth_4`, seed 44. The earlier chat identified Stage D seed 43 as the best CNN too broadly: it is the designated best seed of the separate fixed-window, no-augmentation control experiment. The main model metadata and existing live service both select Stage C seed 44. Preserve that registered choice. Do not choose a deployment seed by comparing held-out test results or retune the threshold during wiring.

| Runtime property | Required value |
| --- | --- |
| Model ID | `30_cnn` |
| Stage and variant | `C`, `depth_4` |
| Designated seed | `44` |
| Checkpoint | `results/models/30/fit_325062e300481096b40e8f3dd34ebc129876cdb156fd06bee9352743367504ed_seed44.pt` |
| Checkpoint SHA-256 | `e949a91471d4a89370fdabaab6bf10063ab47cd767a0a549f50c2116f8195c06` |
| Selection metadata | `results/models/30/metadata.json` |
| Channels | `[16, 32, 64, 64]` |
| First kernel, pooling, head, dropout | `3`, `[2, 2]`, `global_average`, `0.3` |
| Trainable parameters | `60641` |
| Raw logit threshold | `2.7514936923980713` |
| Validation F1 and ROC AUC | `0.9322033898305084`, `0.992550739418664` |
| Per-window tensor | float32 `(1, 64, 28)` |
| Model batch tensor | float32 `(1, 1, 64, 28)` |
| Target audio | 16000 Hz, mono channel 0 |
| Input window and decision cadence | 7936 samples, 0.496 seconds, 28 mel frames |
| Decision policy | `threshold-majority-v1`, three thresholded votes, equality is positive |

Checkpoint existence, digest, and checkpoint metadata were inspected while preparing this plan. Its fields are `state_dict`, `lineage_hash`, `seed`, `stage`, `variant`, and `architecture`; `architecture.model.channels` contains the channel sequence. The checkpoint lineage is the historical V1 lineage below. Startup must verify these facts again.

The existing notebook 90 summary was already generated on 2026-11-24. Training completion does not mean final test evaluation is pending. This integration must not rerun training, notebook 90, test scoring, or test cache extraction. Existing aggregate evaluation reports may be linked as evidence; microphone acceptance does not establish field accuracy or sequence-level metrics.

## Current integration gaps

1. `start_detector.bat` opens the browser before `live_detector.py` has loaded the model or bound the port. It checks imports for only torch and sounddevice, misses required runtime imports, and has no readiness timeout or duplicate-instance handling.
2. `requirements.txt` contains research dependencies and omits torch. A newly created environment cannot be reproduced by the present launcher. The current environment contains `torch==2.15.0+cpu`; keep this working installation while establishing an installable runtime lock.
3. `scripts/live_detector.py` sends `transform()` output directly to the model. `transform()` returns `(1,64,28)`; the network's BatchNorm2d path requires the batched shape `(1,1,64,28)`. Add and test that dimension explicitly.
4. The service verifies a checkpoint digest and some V1 fields, but does not complete statistics shape/value checks, recompute the complete legacy normalization lineage, or require V2 native-rate golden verification before declaring readiness.
5. `live/app.js` queues audio in an unbounded JavaScript array. Slow inference or a stalled request can cause stale decisions and growing memory. An in-flight response can also update a stopped or newly restarted session.
6. The page declares MODEL READY, model details, and threshold in static HTML. It understates warm-up as half a second, although three votes require at least 1.488 seconds of contiguous target audio, plus capture, resampling, transport, and processing delay.
7. The service exposes the repository root through `SimpleHTTPRequestHandler`. Static serving should cover only the intended live assets and an explicit evidence route. The current evidence link targets a deleted `results/model_report` location; the report now exists under the sibling `Project documents/Project report` folder.
8. Browser capture, its HTTP continuity behavior, and real CNN latency still need acceptance. The recorded five-minute sounddevice session used a stub scorer and cannot close this gate.
9. Several live entry points and assets are currently untracked. Existing model-report deletions and a modification to `scripts/run_frontend_correction_baseline.py` belong to other work. Preserve them.

## Runtime architecture

```text
start_detector.bat
  -> scripts/launch_detector.py using the project's .venv
     -> verify environment and deployment assets
     -> scripts/live_detector.py bound to 127.0.0.1:8765
        -> verify model, stats, frontend identities and goldens
        -> warm the actual frontend and CNN
        -> publish ready health metadata
     -> open the browser after matching ready health

browser microphone -> bounded mono float32 chunks -> POST /api/audio
  -> session and sequence guard -> streaming resampler and fixed ring
  -> canonical mel_power -> canonical transform -> batched CNN -> raw logit
  -> threshold each logit -> strict majority of 3 -> browser result
```

Keep inference on CPU for this PC. Set a bounded torch thread count explicitly, initially four as used by the recorded runs, then measure actual latency. Keep audio processing off the browser capture callback and serialize inference for the one active microphone session. Audio must remain in memory; logs contain operational metadata and scores only.

Reuse canonical frontend code in `src/dronedet`. The initial HTTP service may continue importing tested stream helpers from `demo/run_demo.py`. Move helpers into a shared package module only if needed for clean dependencies, preserve compatibility imports, and run existing parity tests after a move. Do not create a second frontend implementation.

## Package 1 Establish baseline and runtime assets

**Files:** new `config/live_detector.json`, new `scripts/prepare_local_detector.py`, existing metadata and frontend assets, `.gitignore` for runtime outputs only.

1. Record HEAD, branch, remote, tracked and untracked changes before edits. Record the model card, checkpoint, statistics and frozen frontend digests. Protect trained weights, manifests, normalization values, and installed research caches from writes.
2. Add a versioned runtime manifest containing `schema_version`, `application_id`, model ID, variant, stage, seed, explicit complete inference architecture, input shape, score kind, threshold, relative asset paths and expected digests. Read the seed and threshold from the existing selection metadata in the preparation script; reject disagreement rather than silently picking a different file.
3. The manifest must pin `score_kind: raw_logit`, `policy_id: threshold-majority-v1`, `smoothing_windows: 3`, and `interval_s: 0.5`. It must include a digest of the selection metadata and the V1-to-V2 identity migration record used for compatibility. Preserve the original model card and checkpoint.
4. Materialize an independent local runtime asset folder at `runtime/assets/` using atomic directory publication. Copy only the selected checkpoint, approved statistics, frontend configuration, frozen metadata, selection metadata, identity mapping, and native-rate golden bundle. Include a complete file manifest with hashes. Treat `runtime/` as generated local output and exclude it from Git.
5. For V2 goldens, start from the verified files in `data/interim/frontend_bundle_v2_migration/`, verify them, then copy them to the stable runtime folder. The deployed app must not depend on a temporary/interim folder. Fail preparation if the verified source bundle is missing; provide the existing bundle writer as an explicit setup route using approved stats and existing native fixtures, followed by verification. Never derive a missing golden expectation from the currently running code and immediately call that proof of parity.
6. Use paths relative to project root in committed manifests. Reject absolute asset paths and path traversal. Resolve and check containment before loading or copying. Do not copy raw dataset audio, split payloads, full caches, all candidate fits, or the virtual environment into the asset package.
7. Make preparation idempotent: an exact verified package is reused; changed sources require an explicit rebuild command with before/after digest evidence. Ordinary startup verifies the stable package and never fits statistics or regenerates goldens.

Required frozen identities:

```text
V1 frontend parameters:
97544883461de11a1ee51cc17d7028431d6d0ea8f175effc8e63bc18595f28fc
V1 normalization lineage:
327b5e0c54632f3c7e40533d2d0b521faa876cbaf58b8dd8c5bfa57fe9f6b8f0
Original statistics file SHA-256:
545aa9cfdded3ea9a8aa4fd2053e1b6d55c80329a4cdada1f246571cf1a2e5df
V2 transform:
bb7a21a1e5fb317c43e1e214bc04af6e46e04d75053383d1c62ef6849c9466c7
V2 normalization lineage:
dab196aee59c482e2938d90ea0c55d33eb5606f8aa04d11ff8d441fb6cd31678
```

The bundle's own stats.npz may have a different archive digest from the original statistics file; compare their float32 mean/std values exactly and record each file's own digest. Never substitute one file digest for the other.

**Acceptance:** a complete verified package can be prepared without reading train/validation/test payloads. No protected source artifact changes. Missing or inconsistent inputs produce an actionable setup error.

## Package 2 Implement a verified CNN loader

**Files:** new `src/dronedet/inference.py`, update `scripts/live_detector.py` to call it.

Provide these interfaces, using a dataclass for the returned immutable provenance:

```python
load_detector(project_root: Path, manifest_path: Path) -> LoadedDetector
LoadedDetector.score_window(window: np.ndarray) -> float
LoadedDetector.health_metadata() -> dict
```

Loading order:

1. Validate the manifest schema, supported model/policy/score kinds, finite threshold, seed and architecture; verify required file existence and checksums before deserialization.
2. Load config through `dronedet.config.load_config`. Require the approved dimensions, transform parameters, `global_stats`, and no inference augmentation.
3. Load mean/std with `allow_pickle=False`. Require float32 vectors of shape `(64,)`, finite means/stds, and strictly positive stds. Compare to the verified bundle arrays exactly.
4. Use `legacy_params_hash` and `legacy_config_hash(config, stats)` for immutable V1 checks. Compare checkpoint, original model metadata, frozen frontend record, and recomputed lineage. Do not compare a V1 checkpoint field directly to a V2 hash.
5. Recompute `transform_hash` and `normalization_lineage_hash`; require the verified migration mapping. Compute `decision_hash(config, threshold=threshold)` separately, including the loaded threshold. Preserve the historical artifacts unchanged.
6. Run `dronedet.bundle.verify_bundle` against the V2 runtime bundle. Require both native 16 kHz and resampled 48 kHz coverage and normalized golden verification. Preserve existing 1e-3 dB and 1e-6 normalized tolerances.
7. Build `build_cnn((1,64,28), channels=(16,32,64,64), kernel_size=3, pool=(2,2), head="global_average", dropout=0.3)`. Verify `parameter_count(model)==60641`.
8. Deserialize with `torch.load(..., map_location="cpu", weights_only=True)`. Require stage C, seed 44, variant depth_4 and matching checkpoint architecture/lineage. Load `state_dict` with `strict=True`; call `eval()`.
9. Warm the real mel/transform and CNN path with deterministic synthetic audio before readiness. Also forward the verified normalized golden through the loaded model. Require finite output; repeated identical inputs must produce the same output within a documented float tolerance. Clear all live state after warm-up.

Implement the critical tensor handling explicitly:

```python
features = transform(mel_power(window, config), config, stats)
if features.shape != (1, config.n_mels, config.n_win):
    raise ValueError("Unexpected frontend tensor shape")
if features.dtype != np.float32 or not np.isfinite(features).all():
    raise ValueError("Invalid frontend tensor")
x = torch.from_numpy(np.ascontiguousarray(features)).unsqueeze(0)
with torch.inference_mode():
    output = model(x)
if output.shape != (1,) or not torch.isfinite(output).all():
    raise RuntimeError("Invalid CNN output")
return float(output.item())
```

Validate raw windows as finite float32 mono arrays of exactly 7936 target samples. Compare raw logits directly with the frozen threshold. Do not apply a sigmoid and retain the logit threshold, or display the score as a calibrated percentage.

**Acceptance:** the real saved CNN loads, verifies all dependencies, and scores both goldens and synthetic windows. Corrupt checkpoint, mismatched lineage, invalid std, wrong architecture, missing golden, and wrong shape all prevent ready status.

## Package 3 Make the local service explicit and bounded

**Files:** `scripts/live_detector.py`, shared inference module, existing stream helpers.

1. Add `--host`, `--port`, `--manifest`, `--instance-id`, `--check`, and `--no-browser` as appropriate to the launcher/service split. Restrict the supported host to loopback. `--check` verifies/warm-ups assets and exits without opening microphone hardware or starting a server.
2. Remove mutable module-level model initialization as the main design. Inject the loaded detector and session state into a server object/handler factory. Keep startup exceptions in a short human-readable console message plus a detailed local log.
3. Serve only `/`, explicit live assets, `/api/health`, `/api/audio`, and a scoped evidence route. Disable directory listings, repository-root file serving and traversal. Resolve `/evidence/` to the existing sibling report only when its entry point exists; otherwise return an explicit unavailable state and hide the broken link. No report generation at launch.
4. `/api/health` returns application ID, protocol version, startup instance ID, readiness, model ID/stage/variant/seed, checkpoint digest, threshold, score kind, frontend identities, policy, vote count, window samples, and actual cadence. Return readiness only after verification and warm-up. Avoid exposing local absolute paths in browser payloads.
5. Keep audio protocol as little-endian float32 binary. Require valid Content-Type, bounded Content-Length divisible by four, supported sample rate, valid session ID and nonnegative integer sequence. Require exactly the expected body length; reject truncated reads. Apply socket/request timeouts. Accept no more than two seconds per request and set a small global upper byte bound.
6. Bind requests to the local UI using an instance token obtained from health plus Host/Origin checks and no permissive CORS. Allow only the local served page's requests. This also prevents an unrelated process on port 8765 from being mistaken for this detector during launch.
7. Support one active capture session: claim it on start, release it on stop, expire it after inactivity, and return a clear busy response for another active tab. Add `/api/session/start` and `/api/session/stop` with explicit ownership tokens. Do not let two tabs alternate and repeatedly reset shared state.
8. Preserve serial processing within each session and reset the resampler, ring, pending samples and smoother on a known gap/rate/session change. Reject stale/duplicate chunks without rewinding `expected_sequence`; the existing `SessionSequenceGuard.observe()` alone does not enforce this replay rule. Advance a generation and require fresh contiguous audio before resuming.
9. Return statuses `buffering`, `warming_up`, `scored`, and `standby`; add generation, session ID, echoed sequence, recovery reasons, audio window sample offsets, and processing duration. If multiple windows result from one chunk, process every window in order and return the latest result plus an emitted count. Do not accidentally skip smoother updates for intermediate windows.
10. Quiet or near-silent windows clear votes and return standby, not a negative prediction. Clipped input carries a warning. Nonfinite model output terminates/rejects the current scoring session with a clear error and resets continuity.
11. Remove or bound `LiveWindowBuilder.emitted_starts` accumulation for indefinite operation while preserving the test helper's ability to inspect offsets. Keep only a counter/latest offset or enable full collection solely in bounded diagnostics.

**Acceptance:** the health contract reflects the actual loaded package, directory access is limited, requests and state remain bounded, one session has clear ownership, and gaps never produce a window spanning missing audio.

## Package 4 Make browser capture recover reliably

**Files:** `live/app.js`, `live/index.html`, optionally new `live/audio-capture-worklet.js`, minimal CSS changes only as required.

1. Apply the project's website build instructions before UI edits. Check `.codex/hooks.json`; install the Impeccable project hook only if absent. Run Impeccable init for this brand surface and use the existing live surface as context. Apply the required Designs of Desire lane and anti-references from AGENTS.md. Use emil-design-eng timing/easing guidance for any motion. Update the live surface's stale score-smoothing wording to three boolean votes.
2. On page load, fetch health and render actual model seed/threshold/provenance; show Checking model until confirmed. Start is enabled only when ready. Update the broken model evidence link using service capability metadata.
3. Prefer an AudioWorklet for capture to avoid the deprecated ScriptProcessor path and main-thread capture dependence. Give every message a source-frame counter. Keep the worklet bounded and free of networking/model work. Transfer fixed float32 blocks to the main thread; preserve the browser's actual AudioContext sample rate and explicitly average/select channels according to the mono capture contract.
4. Request mono audio with echo cancellation, noise suppression and auto gain disabled as now. Inspect actual track settings and report unsupported constraints as diagnostics; do not claim browser preprocessing was certainly disabled. Call `AudioContext.resume()` from the Start gesture and verify state.
5. Use a bounded sample queue with at most one request in flight and at most one unsent half-second chunk. Track sample counts rather than growing a JavaScript number array indefinitely. Set a finite fetch timeout. A full queue, source-frame gap, aborted request or device interruption invalidates continuity: discard queued audio, create a new generation/session as required by ownership protocol, and warm up from fresh samples.
6. Associate each request with a local session epoch and AbortController. On Stop, abort requests, disconnect nodes, stop tracks, close the context, release ownership and clear buffers. A response from an older epoch must never change the current UI or drain a new session's queue. Catch/finally handlers must check their epoch before changing requestBusy or scheduling another send.
7. Handle permission denied, no device, device removed/track ended, context suspended, server stopped, server restarting, busy session and timeout as distinct recoverable messages. Preserve the meaningful error instead of having `stop()` overwrite it with the normal privacy message.
8. Render buffering/recovery immediately and clear any previous positive/negative result. Render warming-up vote progress and standby separately. Show raw logit and the loaded threshold together. Describe the decision as three-window majority; display actual 0.496-second cadence in details.
9. Correct the first-result copy: three votes need at least 1.488 seconds of contiguous target audio, plus runtime delays. A quiet window resets votes, so the first decision can take longer. Avoid promising a first decision in half a second.
10. Use a bounded requestAnimationFrame waveform loop rather than redrawing for every incoming sample. Support reduced motion and keyboard operation. Do not record/upload audio or load external fonts/assets required for offline use.

**Acceptance:** real and simulated capture support start/stop/restart without stale results; stalled inference drops old audio visibly and recovers; offline UI loads; model details come from verified health.

## Package 5 Build deterministic runtime setup

**Files:** new `requirements-live.txt`, optional `requirements-live-lock.txt`, new `scripts/check_live_environment.py`, launcher integration.

1. Reuse `.venv/Scripts/python.exe` when it passes the supported Python 3.12 and dependency checks. Runtime imports include torch, NumPy, scipy, librosa, soxr, soundfile and yaml; resolve and pin required transitive dependencies such as numba/llvmlite from the working environment. Browser capture does not require sounddevice. Keep sounddevice in the diagnostic dependency set if the CLI hardware probe is retained.
2. Capture the installed working versions before any install. Observed direct versions: torch 2.15.0+cpu, numpy 2.5.5, scipy 1.18.1, librosa 1.0.1, soxr 1.1.0, soundfile 0.14.0, PyYAML 6.0.4; numba 0.68.1 and llvmlite 0.50.0 are also installed. These are installed facts, not proof that matching wheels are currently downloadable.
3. Establish a tested Windows/Python 3.12 installation route for the CPU torch wheel and all locked dependencies. If matching wheels are unavailable, preserve the working environment and stage a local wheel cache or document the exact unresolved installation step; do not silently replace the trained model's runtime with an arbitrary torch version.
4. Test new installation in an isolated environment before making it the batch launcher's default bootstrap route. Keep notebooks, plotting tools, and training dependencies outside the runtime lock unless imports actually require them.
5. Make the check script distinguish missing dependencies, wrong versions, binary import failures, missing assets and failed model verification. It must print actionable messages and return nonzero; no pip operations inside inference.
6. Routine launch must not upgrade packages. Bootstrap only when the environment is absent or clearly incomplete, then rerun dependency checks, bundle parity and real model preflight. Failed installs leave the existing working environment intact.

**Acceptance:** the current environment works unchanged; isolated setup is reproducible or the remaining prerequisite is reported exactly; a prepared machine runs without internet.

## Package 6 Finish the one click launcher

**Files:** `start_detector.bat`, new `scripts/launch_detector.py`.

Keep the batch file small. Use `setlocal`, `cd /d "%~dp0"`, quoted paths and explicit errorlevel handling. Never rely on shell activation or an assumed current directory.

Batch flow:

```text
locate project from %~dp0
find .venv/Scripts/python.exe
if absent, verify py -3.12 then bootstrap with the locked setup route
run the environment check; on error print remedy and pause
run scripts/launch_detector.py
capture exit code; pause on failure; return that same code
```

Python launcher flow:

1. Probe localhost:8765 with a short timeout. If health proves this application, expected model digest and compatible protocol are already ready, open that instance and exit successfully. If another service owns the port, explain the collision and stop. Never terminate an unknown process or open it as this app.
2. Prevent two simultaneous first launches with a Windows-compatible singleton mechanism, held during startup. Recheck health after acquiring it. Do not rely on a stale PID text file as the only ownership proof.
3. Create a fresh instance ID; spawn the service using `subprocess.Popen([sys.executable, absolute_script_path, ...], cwd=project_root)` without shell=True. Keep output visible in the launcher console and tee operational logs to a bounded rotating file.
4. Poll `/api/health` for the matching instance ID, model digest and ready status with a monotonic deadline, initially 90 seconds to accommodate cold JIT/frontend initialization. Stop early if the child exits. Show a concise Starting model message while waiting.
5. Call Python `webbrowser.open` only after matching readiness. If browser opening fails, show the usable local URL while retaining the running service.
6. Supervise the child in the foreground. On Ctrl+C terminate only the child started by this launcher, wait for clean shutdown and escalate to killing that child only after timeout. Closing the launcher console must not leave an unnoticed service; use a Windows Job Object with kill-on-job-close for the owned child or an equivalent verified parent-death mechanism.
7. Closing a browser tab releases capture; the service may stay running until the launcher is closed. State this behavior in the console. A second launcher attaching to an existing service must not take ownership of that service's shutdown.
8. Log failure context and point to the local log file; do not exit before the owner can read the error. Healthy repeated runs open the existing app without model reload or duplicate listeners.

Single click here means one double-click to launch the app. Browser microphone permission and the Start monitoring gesture remain necessary on first use and whenever the browser requires them.

**Acceptance:** double-click from Explorer works with spaces in the path, opens only after ready, handles repeat launch/port collision, shuts down its owned service, and retains useful errors.

## Package 7 Add focused automated verification

**Files:** existing `tests/unit/test_dsp_frontend.py`, `tests/unit/test_stream_parity.py`, and a focused `tests/unit/test_live_detector.py` within the existing test tree.

Implement tests that exercise integration boundaries:

- Real selected checkpoint: verified loader, batch dimension, finite scalar raw logit, eval mode, strict parameter loading, correct threshold and parameter count. Mark asset-dependent tests explicitly and require them in local release acceptance; a skipped real-model test cannot satisfy release.
- Loader rejection: byte-tampered checkpoint, wrong seed/stage, bad metadata digest, invalid mean/std, mismatched V1 lineage/V2 mapping and tampered native-rate golden. Use temporary copies, never mutate production artifacts.
- Score parity: same canonical synthetic/golden window through direct loaded-model inference and HTTP processing gives the same raw score at a documented tight tolerance. Test batched dimensions and batch-of-one output explicitly.
- Runtime policy: equality to threshold is positive, three thresholded boolean votes, quiet reset, discontinuity reset and fresh warm-up. Compare logits on both sides of the actual threshold; do not assume a 0.5 probability threshold.
- HTTP integration on an ephemeral loopback port: health identity, malformed/alignment/truncated payload, session ownership, stale sequence, wrong token/origin, stopped session, static traversal and unavailable evidence route.
- Stream regression: retain exact 16 kHz and <=1e-3 dB 48 kHz parity; ensure post-gap windows are contiguous. Simulate transport loss and backpressure independently of sounddevice callback tests.
- Launcher process tests: child exits during loading, readiness timeout, wrong service on port, matching already-running app, duplicate launch and owned-child cleanup. Mock browser opening and use short test deadlines.
- Browser checks: deterministic fake microphone stream/request delays establish bounded queue, timeout reset, stale-response suppression and Stop/Start behavior. Real microphone permission/capture is verified separately below.

Suggested commands after implementation, from the project directory:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/unit -q
.\.venv\Scripts\python.exe -m compileall -q src/dronedet/inference.py scripts/live_detector.py scripts/launch_detector.py scripts/prepare_local_detector.py scripts/check_live_environment.py
node --check live/app.js
.\.venv\Scripts\python.exe scripts/live_detector.py --check
```

Run equivalent syntax checks for any added worklet. Use the project's installed lint tool for changed Python files. Bind all acceptance reports to the tested commit, model digest, config and asset manifest digests. None of these checks may retrain or rescore test data.

## Package 8 Verify the complete detector on this PC

1. Verify launch from Explorer with the existing environment, then with internet disconnected. Test launching twice and a deliberately occupied port. Test startup failure using a temporary manifest with a missing or tampered asset; a failure must not open a misleading ready page.
2. Use chrome-devtools-mcp to inspect console/network errors and capture the final desktop and mobile layouts. Run Impeccable audit and resolve flagged issues before closeout. If that browser tool is unavailable, record the missing screenshot gate and establish an actual available browser route; do not invent screenshots or mark the requested gate passed.
3. Run at least five minutes of real browser microphone capture through the loaded Stage C seed 44 model. Measure the actual browser/device sample rate, captured/dropped chunks, recovery count, completed windows, request latency and processing mean/p95/max. Report startup warm-up separately.
4. Exercise permission denied, Stop/Start, unplugged or disabled input, tab suspension, delayed/failed HTTP request, server stop/restart and an intentionally slow inference path. Recover without scoring a mixed pre-gap/post-gap window or showing an old decision as current.
5. Use synthetic signals and explicitly selected training audio to prove the browser pipeline reaches the real CNN. Ambient quiet must show standby, and ordinary room sound may be recorded as an operational observation. A synthetic tone is not required to classify as a drone. Do not promise a positive result for arbitrary test sounds or reuse held-out clips for threshold adjustment.
6. Measure model processing against the actual 496 ms cadence. Release target on this PC: p95 full frontend-plus-CNN processing below 496 ms, bounded queue and no sustained backlog during ordinary capture. Record all drops; induced-stall tests are evaluated for correct reset/recovery rather than zero drops. If throughput fails, profile initialization/threading and resolve it before claiming continuous operation.
7. Track process memory over the session and assert bounded sample/vote/offset histories. Verify logs contain no microphone samples and audio is never persisted.
8. Check shutdown from Ctrl+C and console close, then relaunch immediately. Confirm the owned server is gone and port 8765 is free. Confirm closing only a browser tab stops microphone capture without corrupting the service for a later session.

Store acceptance data under ignored `runtime/logs/` or `data/interim/`; put a concise sanitized closeout report in `docs/`. A complete release needs both automated acceptance and physical browser/CNN evidence. Frontend G15 remains historical frontend evidence, with its original scope preserved.

## Implementation order and commit boundaries

Implement Packages 1–2 first so the selected model is independently verifiable. Then Package 3 defines the HTTP/session contract, followed by Package 4 browser integration. Packages 5–6 deliver setup and launch behavior. Add the boundary tests alongside each package; run Packages 7–8 for release acceptance. Complete the runtime preparation on this PC so double-click actually works after the implementation task.

Use task-scoped commits for verified model loading/assets preparation, service/session integration, browser capture, launcher/setup, and final acceptance documentation as appropriate. Preserve unrelated deletions and modifications. Review currently untracked live files individually before adding them; include only the files required by this task and their design metadata. Never use `git add -A` in this dirty workspace.

After required checks pass, follow the user's GitHub preference: commit the task files on the current task branch and push to the existing `origin` at `https://github.com/NasrallahRAFI/Functional-Project.git`. Do not force-push. Keep the selected weights, runtime assets, caches, local logs, build outputs and private audio out of Git. Explain that a fresh clone needs the separately retained local model package; publishing code alone does not transfer ignored model weights.

## Completion checklist

- [ ] Registered main CNN Stage C depth_4 seed 44, original weights and threshold are used.
- [ ] Complete runtime assets verify, including statistics, V1/V2 mapping and native-rate goldens.
- [ ] Real inference uses `(1,1,64,28)` and returns finite raw logits.
- [ ] Majority, quiet, buffering and recovery states follow the accepted frontend contract.
- [ ] Browser buffering, request concurrency, histories and logs remain bounded.
- [ ] Live model identity and readiness are populated from verified health.
- [ ] Runtime dependencies have a verified setup route; prepared launches work offline.
- [ ] Batch launch waits for readiness, handles duplicate launch and cleans up its service.
- [ ] Focused regression and real-model integration tests pass.
- [ ] Desktop/mobile screenshots and design audit are complete.
- [ ] Five-minute browser-to-real-CNN session and recovery cases pass with measured latency.
- [ ] Closeout reports actual results, limitations, commit and push status.

## Source references

- `Project documents/Front end/correction plan V1.0.md`: current accepted frontend decisions and final G1–G15 scope.
- `Project documents/Front end/Archive/FRONTEND_CURRENT_CONTRACT.md`: current numerical, identity and runtime contracts.
- `Project documents/Front end/Archive/records/frontend_identity_migration_v2.json`: verified identity-domain mapping.
- `Project documents/Front end/Archive/records/frontend_legacy_caller_inventory_2026-11-25.md`: canonical and compatibility caller boundaries.
- `results/models/30/metadata.json` and selected checkpoint: registered main model selection, threshold and architecture.
- `results/models/30_cnn_control/model_card.json`, `results/tables/run_report_30_local_stage_d.json`: separate control CNN scope.
- `results/tables/run_report_notebook90_local.json`: existing aggregate final evaluation; read-only historical evidence.
- `results/tables/cnn_frontend_frozen.json`, `config.yaml`, `src/dronedet/config.py`, `features.py`, `bundle.py`, and `models/cnn.py`: executable numerical and loading contracts.
- `demo/run_demo.py`, `scripts/live_detector.py`, `start_detector.bat`, and `live/`: current capture and launch implementation.

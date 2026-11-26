# CNN V2.0 frontend verification

Verified on 2026-11-26. The local monitor serves the selected CNN V2.0 depth_4 model, seed 42, with 60,641 parameters.

## Runtime contract

- TorchScript SHA-256: `0e97e3194e42f8fa3a2541e37dcff9fe33cdc1246d5a260ab65c6d157af6f4fb`.
- Raw-logit threshold: `2.823960542678833` (shown as `2.8240` in the monitor).
- Input: mono audio resampled to 16 kHz, transformed using the bundled 64-band mel configuration and training normalization statistics. Each model input has shape `(1, 1, 64, 28)`.
- Window: 7,936 samples. Decision cadence: 0.496 seconds after hop alignment. Final decisions require a majority of three thresholded windows; quiet audio is standby.
- Versioned package: `runtime/assets/CNN_V2.0`. Runtime files are local artifacts and are excluded from Git.

## Corrections from this review

The backend already loaded the correct V2 model and statistics. The browser's identity check previously accepted a matching name, seed, and architecture without checking the actual artifact or decision settings. Both browser readiness and microphone startup now require the exact artifact hash, raw-logit threshold, protocol, window, cadence, and smoothing policy. Static responses use `Cache-Control: no-store` to prevent subsequent monitor loads from reusing cached V1 files. The loader also rejects an inconsistent model version in the selection manifest.

## Evidence

- Package staging verification passed for all 15 manifest members, including both native-rate frontend goldens.
- The focused Python suite passed 23 tests, including three sample-rate subtests. After adding the cache-header assertion, the final detector and launcher run passed all 10 tests and three sample-rate subtests.
- Audio sent through the real HTTP handler at 16 kHz, 44.1 kHz, and 48 kHz matched direct V2 inference on independently resampled windows within `0.0001` raw logit. The test also checked the frozen threshold and majority decisions.
- Five browser lifecycle and audio worklet tests passed, including rejection of wrong model hashes, thresholds, protocols, score spaces, and smoothing settings before microphone access.
- Chrome DevTools MCP inspection at desktop 1440 px and mobile 390 px showed the V2 ready state, the correct threshold and cadence, no horizontal overflow, and no console messages. Screenshots and browser observations are saved locally under `results/cnn_v2/frontend_verification/`.
- The Impeccable detector returned no findings for the monitor HTML and script. Visual review confirmed that model labels and controls remain readable at both sizes.

The frozen evaluation used a benchmark reused from V1, not a new unseen test. The four available microphone recordings are real environment checks, but their intended sound labels were not supplied, so they cannot establish a labeled real-world miss rate. No model was retrained or selected using these checks.

## Saved microphone recordings

Four saved 10-second FLOAT WAV recordings were replayed through the V2 `LiveWindowBuilder`, frontend, normalization, TorchScript model, frozen threshold, and three-vote smoother. All were 48 kHz mono and about -45.2 to -47.1 dBFS RMS. The 80 analyzed windows scored between -8.256 and -0.905 raw logit; none crossed the frozen threshold (2.82396), so the live decision was consistently no drone. The screenshot's -1.968 raw score lies in this replayed score range.

In the locked benchmark, the fifth percentile of drone scores was 0.099. Both alternative validation thresholds, 1.894 at the 1% false-alarm operating point and −0.221 at the 5% operating point, exceed the highest recorded score of −0.905. All 80 windows would therefore still receive negative raw votes at either threshold. The files lack confirmed drone-event labels. They were replayed locally.

## Start and reproduce

On the prepared machine, run `start_detector.bat`. If the runtime package is absent but the verified export exists locally, run `.venv\Scripts\python.exe scripts\prepare_local_detector.py` first. This copies the export from `results/cnn_v2/export/CNN_V2.0` to the versioned runtime directory and verifies it. A fresh source checkout needs that locally supplied package or a reproduced export; the weights are not downloaded by the launcher.

Run the focused automated checks from the project root:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m pytest tests/unit/test_live_detector.py tests/unit/test_live_launcher.py tests/unit/test_cnn_v2_protocol.py -q
node --test tests/browser/*.test.mjs
.venv\Scripts\python.exe scripts/prepare_local_detector.py
```

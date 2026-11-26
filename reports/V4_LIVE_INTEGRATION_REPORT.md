# V4 live score integration and server boundary verification

Date: 26 November 2026.

## Result

The trained CNN V4.0 A0 seed-42 model is connected to the V4 browser monitor.
The owner explicitly selected live raw scores with threshold pending. The service
runs on http://127.0.0.1:8766 and accepts the corresponding localhost origin.
Start it using the V4 workspace's start_detector.bat. V3 remains separate.

Checkpoint SHA-256:
f4372da59644395c174514944aa6387901ed483f64c137fd1ae4e31c83092049.

Training normalization archive SHA-256:
4dc6820ca53a7965ae1e0264e4bc69e6197bd1f91ce0299e287ddeb71a26d6a4.

The loader verifies the checkpoint against the frozen training run and registry,
checks config/provenance and normalization identity, loads the original state dict
into the three-block 23,585-parameter architecture, and switches it to evaluation
mode. It warms the model on a synthetic zero window. The raw score API returns
threshold=null and drone=null and does not accumulate decision votes.

## Server boundary fixes

- Host and Origin checks consistently support localhost and 127.0.0.1 with the
  actual service port and matching browser origin.
- The CLI no longer advertises IPv6 binding for an IPv4 HTTP server.
- Socket reads have a five-second timeout, including audio and session uploads.
- Invalid sample rates and oversized duration metadata are rejected before
  reading request bodies.
- JavaScript responses explicitly use text/javascript.
- JSON serialization rejects non-finite values. V4 raw model outputs are checked
  before being returned; V3's existing finite-vote check remains effective.

These boundary fixes also passed the V3 service/launcher tests.

## Verification

- V4 service, HTTP boundary, launcher and activation checks: 23 tests passed,
  with three browser-rate subtests also passed.
- V3 HTTP boundary/service/launcher checks: 17 tests passed, three subtests passed.
- V4 browser capture/lifecycle suite: seven tests passed, including simulated
  sustained capture, gap signaling, stale-result suppression and contract rejection.
- Synthetic signals at 16, 44.1 and 48 kHz matched independently resampled training
  frontend scores within 1e-4 through the live V4 path.
- A headless Chrome session used Chrome's simulated microphone. At the sampled
  checkpoint it had completed ten HTTP inference requests with zero dropped frames,
  zero resets, and no browser exceptions. Stop released capture.
- Desktop (1440x1050) and mobile (390x844) captures were inspected. Neither
  viewport had page-level horizontal overflow.
- Impeccable source and rendered scans returned no primary findings at both sizes.

Chrome DevTools MCP was not available in this session. Screenshots and browser
interaction used a local headless Chrome CDP connection instead. This was simulated
capture, not physical microphone acceptance or a drone-detection performance test.
Screenshots and the detailed simulated trace remain under ignored local/screenshots/
v4-live in the enclosing project folder.

## UI audit

Implementation integrity passes for the scoped activation: model identity and raw
score semantics are explicit; threshold pending is visible; no probability,
validation-pass, final-test or binary-detection claim is introduced.

| Dimension | Score / 4 | Observation |
|---|---:|---|
| Accessibility | 3 | Existing semantic controls, focus styles and live result region retained; no independent assistive-technology session |
| Performance | 3 | Bounded worklet pool and asynchronous HTTP; short simulated capture passed, physical sustained operation unverified |
| Theming | 3 | Established tokens retained; existing canvas colors remain explicit |
| Responsive layout | 4 | Inspected desktop/mobile states without horizontal overflow |
| Integrity | 4 | Visible identity, pending threshold and score units agree with verified runtime |
| Total | 17 / 20 | Good; limits above remain explicit |

No task-blocking UI finding was observed. The audit does not establish physical
microphone quality, real acoustic detection accuracy or longer-duration throughput.

## Data and scope

No raw audio, frozen manifests, test audio, weights or fitted statistics were
changed. No training, threshold search, validation-set model selection or held-out
scoring was performed. The existing checkpoint and normalization stay in local
project storage; Git publication contains code, documentation and non-sensitive
identity metadata only.

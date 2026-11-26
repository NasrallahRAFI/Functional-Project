const $ = (id) => document.getElementById(id);
const canvas = $('waveform');
const ctx = canvas.getContext('2d');
const CHUNK_SECONDS = 0.5;
const REQUEST_TIMEOUT_MS = 4000;
const CAPTURE_VERSION = 'pool-v2';
function isSelectedModel(m) {
  const common = m?.status === 'ready' && m.application_id === 'local-drone-detector'
    && m.protocol_version === 2 && m.variant === 'depth_4' && m.seed === 42
    && m.score_kind === 'raw_logit' && m.policy_id === 'threshold-majority-v1'
    && m.smoothing_windows === 3 && m.window_samples === 7936 && m.cadence_seconds === 0.496;
  const v2 = m?.model_id === 'CNN_V2.0' && m.model_version === '2.0'
    && m.checkpoint_sha256 === '0e97e3194e42f8fa3a2541e37dcff9fe33cdc1246d5a260ab65c6d157af6f4fb'
    && m.threshold === 2.823960542678833 && m.validation_eligible === true
    && m.experimental_activation === false;
  const v3 = m?.model_id === 'CNN_V3.0' && m.model_version === '3.0' && m.stage === 'A1'
    && m.checkpoint_sha256 === '7847c73d905811bd53b164ebb923a20e95eccac41d32b478585c0aa658704fed'
    && m.threshold === 1.9109516143798828 && m.validation_eligible === false
    && m.experimental_activation === true && m.test_access === false
    && Array.isArray(m.validation_guard_failures)
    && m.validation_guard_failures.includes('clean_recall_at_least_159_of_185')
    && m.validation_guard_failures.includes('clean_false_positives_at_most_5_of_1325')
    && m.validation_guard_failures.includes('each_augmented_far_at_most_1pct');
  return common && (v2 || v3);
}
function updateModelUI(m) {
  state.selectedModel = {
    model_id: m.model_id, model_version: m.model_version, stage: m.stage,
    variant: m.variant, seed: m.seed, checkpoint_sha256: m.checkpoint_sha256,
    threshold: m.threshold, validation_eligible: m.validation_eligible,
    experimental_activation: m.experimental_activation,
    validation_guard_failures: m.validation_guard_failures || [], test_access: m.test_access ?? false,
  };
  const experimental = m.model_id === 'CNN_V3.0';
  const name = experimental ? 'CNN V3.0 · A1 · depth 4' : 'CNN V2.0 · depth 4';
  $('systemText').textContent = experimental ? `CNN V3.0 A1 EXPERIMENTAL · SEED ${m.seed}` : `CNN V2.0 READY · SEED ${m.seed}`;
  $('activeModelLede').textContent = experimental
    ? 'This local session is running the experimental CNN V3.0 A1 model. It failed its registered validation guards.'
    : 'A continuous microphone feed, read by the project’s validation-selected CNN V2.0 model.';
  $('modelStampStatus').textContent = experimental ? 'EXPERIMENTAL · VALIDATION GUARDS FAILED' : 'VALIDATED MODEL';
  $('modelStampStatus').classList.toggle('experimental-status', experimental);
  $('activeModelName').textContent = name;
  $('activeModelDetails').textContent = `Seed ${m.seed} · 60,641 parameters`;
  $('modelEvidenceActive').textContent = `LIVE MODEL · ${name} · seed ${m.seed}`;
  $('modelEvidenceStatus').textContent = experimental
    ? 'Activated experimentally by request despite failed validation: clean recall 157/185; clean false positives 25/1,325; moderate/severe FAR 2.49%/5.21%. Test data remains locked. CNN V2.0 remains available as the validated baseline.'
    : 'CNN V2.0 is the selected validation baseline. The CNN V3.0 A1 candidate is available only as an experimental, validation-ineligible model.';
  $('modelEvidenceStatus').classList.toggle('experimental-note', experimental);
  $('thresholdValue').textContent = Number(m.threshold).toFixed(4);
  $('cadenceValue').textContent = `${m.cadence_seconds} sec · 3-window majority`;
}
const state = {
  active: false, epoch: 0, stream: null, context: null, source: null,
  processor: null, samples: new Float32Array(0), sampleRate: 0,
  requestBusy: false, controller: null, sessionId: null, ownerToken: null, nextSequence: 0,
  expectedSourceFrame: null, gapPending: false, lastAudioBlockAt: 0, captureWatchdog: null,
  points: new Float32Array(240), pointAt: 0,
  revision: 0, captureBlocks: 0, captureDrops: 0, resets: 0, completedRequests: 0, lastDrawAt: 0,
  recentResets: [],
  recording: null, savingRecording: false,
  trace: [], captureSettings: null, sessionStartedAt: 0, selectedModel: null,
};

function drawWave() {
  const box = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(box.width * dpr) || canvas.height !== Math.round(box.height * dpr)) {
    canvas.width = Math.round(box.width * dpr); canvas.height = Math.round(box.height * dpr);
  }
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h); ctx.strokeStyle = '#31433a'; ctx.lineWidth = Math.max(1, dpr);
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  ctx.strokeStyle = state.active ? '#b9e476' : '#64756b'; ctx.lineWidth = 1.7 * dpr;
  ctx.beginPath();
  for (let i = 0; i < state.points.length; i++) {
    const index = (state.pointAt + i) % state.points.length;
    const x = i / (state.points.length - 1) * w, y = h / 2 - state.points[index] * h * 0.42;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
function setMessage(text, error = false) {
  $('message').textContent = text;
  if (error) $('message').dataset.error = 'true'; else delete $('message').dataset.error;
}
function showError(title, note) {
  $('result').dataset.state = 'error'; $('resultOverline').textContent = 'INPUT NEEDS ATTENTION';
  $('resultText').textContent = title; $('resultNote').textContent = note;
}
function render(data, requestMs) {
  $('latency').textContent = `LAST READ · ${Math.round(requestMs)} ms`;
  if (data.recovered) {
    $('result').dataset.state = 'idle'; $('resultOverline').textContent = 'AUDIO RESET';
    $('resultText').textContent = 'Rebuilding recent window';
    $('resultNote').textContent = 'A capture gap was detected. Waiting for fresh contiguous audio.';
  }
  if (data.status === 'standby') {
    $('result').dataset.state = 'idle'; $('resultOverline').textContent = data.reason === 'quiet' ? 'LOW INPUT LEVEL' : 'LISTENING';
    $('resultText').textContent = 'Standing by'; $('resultNote').textContent = 'Quiet input is not counted as a negative prediction.';
    $('scoreValue').textContent = '—';
  } else if (data.status === 'warming_up' || data.status === 'buffering') {
    $('result').dataset.state = 'idle'; $('resultOverline').textContent = 'WARMING UP';
    $('resultText').textContent = 'Checking recent windows';
    $('resultNote').textContent = data.status === 'buffering' ? 'Building a fresh contiguous audio window.' : `Decision votes ${data.votes_received}/3`;
    $('scoreValue').textContent = Number.isFinite(data.raw_score) ? Number(data.raw_score).toFixed(3) : '—';
  } else if (data.status === 'scored') {
    $('result').dataset.state = data.drone ? 'drone' : 'clear';
    $('resultOverline').textContent = data.drone ? 'PATTERN ABOVE THRESHOLD' : 'NO DRONE PATTERN';
    $('resultText').textContent = data.drone ? 'Drone detected' : 'No drone detected';
    $('resultNote').textContent = `Raw CNN score ${Number(data.score).toFixed(3)} · threshold ${Number(data.threshold).toFixed(3)} · three-window majority`;
    $('scoreValue').textContent = Number(data.score).toFixed(3);
  }
}
function stop({ preserveMessage = false } = {}) {
  cancelRecording('Recording cancelled because monitoring stopped. Nothing was saved.');
  const sessionId = state.sessionId, ownerToken = state.ownerToken;
  state.epoch++; state.active = false; state.sessionId = null;
  state.expectedSourceFrame = null; state.gapPending = false;
  state.ownerToken = null;
  if (sessionId && ownerToken) fetch('/api/session/stop', { method: 'POST', keepalive: true,
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sessionId, owner_token: ownerToken }) }).catch(() => {});
  state.controller?.abort(); state.controller = null; state.requestBusy = false;
  clearTimeout(state.captureWatchdog); state.captureWatchdog = null;
  if (state.processor?.port) {
    state.processor.port.postMessage({ type: 'stop' });
    state.processor.port.onmessage = null;
  }
  state.processor?.disconnect(); state.processor = null;
  state.source?.disconnect(); state.source = null;
  state.stream?.getTracks().forEach((track) => track.stop()); state.stream = null;
  if (state.context) state.context.close(); state.context = null; state.samples = new Float32Array(0);
  $('micDot').classList.remove('on'); $('micLabel').textContent = 'MICROPHONE OFF';
  $('latency').textContent = 'MONITOR PAUSED'; $('sampleRate').textContent = '— Hz';
  $('startButton').disabled = false; $('startButton').querySelector('span:last-child').textContent = 'Start monitoring';
  $('stopButton').disabled = true;
  updateRecordButton();
  if (!preserveMessage) setMessage('Audio is processed locally. Saving a clip requires Record 10 seconds.');
  drawWave();
}
function resetCapture(reason) {
  cancelRecording('Recording cancelled after an audio gap. Nothing was saved; try again when capture is stable.');
  state.revision++; state.resets++;
  state.recentResets.push(performance.now());
  if (state.recentResets.length > 8) state.recentResets.shift();
  state.samples = new Float32Array(0);
  state.expectedSourceFrame = null;
  state.gapPending = true;
  $('result').dataset.state = 'idle'; $('resultOverline').textContent = 'AUDIO RESET';
  $('resultText').textContent = 'Rebuilding recent window'; $('resultNote').textContent = reason;
}
function checkCaptureHealth(epoch) {
  if (!state.active || epoch !== state.epoch) return;
  if (state.recentResets.length === 8 && performance.now() - state.recentResets[0] < 10000) {
    stop({ preserveMessage: true });
    showError('Audio capture is unstable', 'Repeated audio gaps prevented a decision. Select a working microphone and restart monitoring.');
    setMessage('Capture stopped after repeated gaps. No reliable decision is available.', true);
    return;
  }
  if (performance.now() - state.lastAudioBlockAt > 2500) {
    stop({ preserveMessage: true });
    showError('Microphone stream stopped', 'No fresh audio blocks are arriving. Check the selected input and press Start monitoring again.');
    setMessage('Microphone capture stopped and was reset.', true);
    return;
  }
  state.captureWatchdog = setTimeout(() => checkCaptureHealth(epoch), 1000);
}
async function releaseClaim(sessionId, ownerToken) {
  if (!sessionId || !ownerToken) return;
  try {
    await fetch('/api/session/stop', { method: 'POST', keepalive: true,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, owner_token: ownerToken }) });
  } catch (_) { /* The service also expires abandoned ownership. */ }
}
function updateRecordButton() {
  $('recordButton').disabled = !state.active || state.savingRecording;
  $('recordButton').textContent = state.recording ? 'Cancel recording' : state.savingRecording ? 'Saving…' : 'Record 10 seconds';
  $('recordButton').setAttribute?.('aria-pressed', String(Boolean(state.recording)));
}
function cancelRecording(note) {
  if (!state.recording) return;
  state.recording = null;
  $('recordingStatus').textContent = note;
  updateRecordButton();
}
function toggleRecording() {
  if (state.recording) { cancelRecording('Recording cancelled. Nothing was saved.'); return; }
  if (!state.active || state.savingRecording) return;
  state.recording = { samples: new Float32Array(state.sampleRate * 10), at: 0,
    sampleRate: state.sampleRate, sessionId: state.sessionId, ownerToken: state.ownerToken, seconds: -1 };
  $('recordingStatus').textContent = 'Recording 0 / 10 sec — play the drone sound now. This clip will be saved locally.';
  updateRecordButton();
}
function collectRecording(block) {
  const recording = state.recording;
  if (!recording) return;
  const count = Math.min(block.length, recording.samples.length - recording.at);
  recording.samples.set(block.subarray(0, count), recording.at); recording.at += count;
  const seconds = Math.floor(recording.at / recording.sampleRate);
  if (seconds !== recording.seconds) {
    recording.seconds = seconds;
    $('recordingStatus').textContent = `Recording ${seconds} / 10 sec — keep the drone sound playing.`;
  }
  if (recording.at === recording.samples.length) {
    state.recording = null;
    saveRecording(recording);
  }
}
async function saveRecording(recording) {
  state.savingRecording = true; updateRecordButton();
  $('recordingStatus').textContent = 'Saving the 10-second clip on this computer…';
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch('/api/diagnostic-recording', { method: 'POST', signal: controller.signal,
      headers: { 'Content-Type': 'application/octet-stream', 'X-Sample-Rate': String(recording.sampleRate),
        'X-Session-ID': recording.sessionId, 'X-Session-Token': recording.ownerToken }, body: recording.samples.buffer });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Recording could not be saved.');
    $('recordingStatus').textContent = `Saved ${result.filename} · ${result.duration_seconds.toFixed(1)} sec. Ready to inspect.`;
  } catch (error) {
    $('recordingStatus').textContent = `Save could not be confirmed: ${error.message || 'connection interrupted'}. Keep monitoring active and try again.`;
  } finally {
    clearTimeout(timeout); state.savingRecording = false; updateRecordButton();
  }
}
function appendAudio(block) {
  collectRecording(block);
  for (let i = 0; i < block.length; i++) {
    state.pointAt = (state.pointAt + 1) % state.points.length; state.points[state.pointAt] = block[i];
  }
  let merged;
  if (state.samples.length + block.length > Math.ceil(state.sampleRate * 1.1)) {
    resetCapture('Audio fell behind inference. Discarding stale samples and warming up again.');
    merged = block.slice();
  } else {
    merged = new Float32Array(state.samples.length + block.length);
    merged.set(state.samples); merged.set(block, state.samples.length);
  }
  state.samples = merged;
  const sum = block.reduce((total, x) => total + x * x, 0);
  $('levelReadout').textContent = `${(20 * Math.log10(Math.max(Math.sqrt(sum / Math.max(block.length, 1)), 1e-7))).toFixed(0)} dB`;
  if (performance.now() - state.lastDrawAt >= 80) {
    state.lastDrawAt = performance.now(); drawWave();
  }
  if (!state.requestBusy && state.samples.length >= Math.round(state.sampleRate * CHUNK_SECONDS)) sendAudio();
}
async function sendAudio() {
  if (!state.active || state.requestBusy) return;
  const epoch = state.epoch, revision = state.revision, count = Math.round(state.sampleRate * CHUNK_SECONDS);
  if (state.samples.length < count) return;
  const chunk = state.samples.slice(0, count); state.samples = state.samples.slice(count);
  if (state.gapPending) { state.nextSequence++; state.gapPending = false; }
  const sequence = state.nextSequence++, controller = new AbortController();
  state.controller = controller; state.requestBusy = true;
  const timeout = setTimeout(() => controller.abort(new Error('Detector request timed out.')), REQUEST_TIMEOUT_MS);
  try {
    const started = performance.now();
    const response = await fetch('/api/audio', { method: 'POST', signal: controller.signal, cache: 'no-store',
      headers: { 'Content-Type': 'application/octet-stream', 'X-Sample-Rate': String(state.sampleRate),
        'X-Session-ID': state.sessionId, 'X-Session-Token': state.ownerToken,
        'X-Chunk-Sequence': String(sequence) }, body: chunk.buffer });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Detector returned ${response.status}`);
    const completedAt = performance.now();
    const windows = data.window_trace || [data];
    windows.forEach((result, index) => state.trace.push({
      elapsed_seconds: Number(((completedAt - state.sessionStartedAt) / 1000).toFixed(3)),
      sequence, window_in_chunk: index, source_sample_rate: state.sampleRate,
      chunk_seconds: CHUNK_SECONDS,
      request_ms: Number((completedAt - started).toFixed(1)),
      status: result.status, reason: result.reason || null,
      window_rms_dbfs: result.window_rms_dbfs ?? null,
      raw_logit: result.raw_score ?? null, threshold: result.threshold ?? null,
      raw_vote: result.raw_vote ?? null, votes: result.votes ?? [],
      votes_required: result.votes_required ?? 3, drone: result.drone ?? null,
      generation: result.generation ?? data.generation ?? null,
      recovered: Boolean(result.recovered || data.recovered),
      recovery_reasons: result.recovery_reasons || data.recovery_reasons || [],
      dropped_frames: state.captureDrops,
    }));
    if (windows.length) $('traceButton').disabled = false;
    if (epoch === state.epoch && state.active) {
      state.completedRequests++;
      if (revision === state.revision) render(data, performance.now() - started);
    }
  } catch (error) {
    if (epoch === state.epoch && state.active) {
      resetCapture(error.name === 'AbortError' ? 'Inference timed out. Waiting for a fresh audio window.' : 'Connection lost. Restart the local detector and try again.');
      showError('Connection interrupted', error.message || 'Restart the local detector and try again.');
      setMessage('The local model did not respond. Capture has been reset.', true);
    }
  } finally {
    clearTimeout(timeout);
    if (epoch === state.epoch) {
      state.requestBusy = false; state.controller = null;
      if (state.active && state.samples.length >= Math.round(state.sampleRate * CHUNK_SECONDS)) sendAudio();
    }
  }
}
async function start() {
  $('startButton').disabled = true; setMessage('Requesting microphone access…'); const epoch = ++state.epoch;
  try {
    const health = await fetch('/api/health', { cache: 'no-store' });
    if (epoch !== state.epoch) return;
    if (!health.ok) throw new Error('The local CNN service is not ready. Relaunch with start_detector.bat.');
    const model = await health.json();
    if (!isSelectedModel(model)) throw new Error('The running detector does not match the verified local CNN package. Relaunch with start_detector.bat.');
    updateModelUI(model);
    const device = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false });
    if (epoch !== state.epoch) { device.getTracks().forEach((track) => track.stop()); return; }
    state.stream = device;
    state.trace = [];
    state.sessionStartedAt = performance.now();
    state.captureSettings = device.getAudioTracks()[0]?.getSettings?.() || {};
    $('traceButton').disabled = true;
    const sessionId = crypto.randomUUID();
    state.sessionId = sessionId; state.nextSequence = 0;
    state.expectedSourceFrame = null; state.gapPending = false;
    state.recentResets = [];
    const claim = await fetch('/api/session/start', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }) });
    const claimData = await claim.json();
    if (epoch !== state.epoch) {
      await releaseClaim(sessionId, claimData.owner_token);
      device.getTracks().forEach((track) => track.stop());
      return;
    }
    if (!claim.ok) throw new Error(claimData.error || 'Another monitor tab owns the microphone session.');
    state.ownerToken = claimData.owner_token;
    state.context = new AudioContext(); await state.context.resume();
    if (epoch !== state.epoch) return;
    if (state.context.state !== 'running') throw new Error('The audio engine did not start. Press Start monitoring again.');
    await state.context.audioWorklet.addModule(`/live/audio-capture-worklet.js?v=${CAPTURE_VERSION}`);
    if (epoch !== state.epoch) return;
    const track = device.getAudioTracks()[0];
    if (!track || track.readyState !== 'live') throw new Error('No active microphone input is available. Check the selected device and try again.');
    track.addEventListener('ended', () => {
      if (epoch !== state.epoch || !state.active) return;
      stop({ preserveMessage: true });
      showError('Microphone disconnected', 'Reconnect or re-enable the input, then start monitoring again.');
      setMessage('Microphone input ended. No more audio is being analyzed.', true);
    }, { once: true });
    state.context.addEventListener('statechange', () => {
      if (epoch !== state.epoch || !state.active || state.context?.state === 'running') return;
      stop({ preserveMessage: true });
      showError('Audio capture paused', 'The browser audio engine was interrupted. Start monitoring again to rebuild a fresh window.');
      setMessage('Audio capture was interrupted and has been reset.', true);
    });
    state.sampleRate = state.context.sampleRate; state.source = state.context.createMediaStreamSource(device);
    state.processor = new AudioWorkletNode(state.context, 'mono-capture', {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
      channelCount: 1, channelCountMode: 'explicit',
    });
    const processor = state.processor;
    processor.port.onmessage = (event) => {
      const block = event.data;
      if (block?.type !== 'audio-block') return;
      const samples = block.samples instanceof Float32Array ? block.samples : new Float32Array(block.samples);
      const recycle = () => processor.port.postMessage({ type: 'recycle', buffer: samples.buffer }, [samples.buffer]);
      if (!state.active || epoch !== state.epoch) { recycle(); return; }
      state.lastAudioBlockAt = performance.now();
      state.captureBlocks++;
      const ageSeconds = state.context.currentTime - (block.sourceFrame + samples.length) / state.sampleRate;
      if (ageSeconds > 0.5) {
        state.captureDrops += samples.length;
        if (!state.gapPending) resetCapture('Discarding delayed audio. Waiting for fresh microphone input.');
        recycle(); return;
      }
      if (block.droppedFrames > 0 || (state.expectedSourceFrame !== null && block.sourceFrame !== state.expectedSourceFrame)) {
        state.captureDrops += block.droppedFrames || Math.max(0, block.sourceFrame - state.expectedSourceFrame);
        resetCapture('The mic stream skipped audio. Rebuilding the three-vote window needs about 1.5 sec of uninterrupted input. If this keeps repeating, check the mic and keep this tab active.');
      }
      state.expectedSourceFrame = block.sourceFrame + samples.length;
      try { appendAudio(samples); } finally { recycle(); }
    };
    state.processor.addEventListener?.('processorerror', () => {
      if (epoch !== state.epoch || !state.active) return;
      stop({ preserveMessage: true });
      showError('Microphone processing stopped', 'The browser audio worklet failed. Press Start monitoring to create a fresh capture session.');
      setMessage('Browser microphone processing stopped and was reset.', true);
    }, { once: true });
    state.source.connect(state.processor); state.processor.connect(state.context.destination); state.active = true;
    updateRecordButton();
    state.lastAudioBlockAt = performance.now();
    state.captureWatchdog = setTimeout(() => checkCaptureHealth(epoch), 2500);
    $('micDot').classList.add('on'); $('micLabel').textContent = 'MICROPHONE LIVE'; $('latency').textContent = 'BUILDING FIRST DECISION';
    $('sampleRate').textContent = `${state.sampleRate.toLocaleString()} Hz`; $('result').dataset.state = 'idle';
    $('resultOverline').textContent = 'CAPTURING AUDIO'; $('resultText').textContent = 'Listening';
    $('resultNote').textContent = 'Three votes need at least 1.488 seconds of clear audio; quiet input takes longer.';
    $('startButton').querySelector('span:last-child').textContent = 'Monitoring'; $('stopButton').disabled = false;
    setMessage('Listening continuously. Audio is saved only when you press Record 10 seconds.');
  } catch (error) {
    if (epoch === state.epoch) { stop({ preserveMessage: true }); showError('Microphone unavailable', error.message || 'Check microphone permissions and try again.'); setMessage(error.message || 'Microphone access is required.', true); }
  }
}
fetch('/api/health', { cache: 'no-store' }).then((r) => r.ok ? r.json() : Promise.reject()).then((m) => {
  if (isSelectedModel(m)) {
    updateModelUI(m);
  } else throw new Error('CNN selection mismatch');
}).catch(() => { $('systemText').textContent = 'MODEL SERVICE UNAVAILABLE'; setMessage('Relaunch with start_detector.bat to load the local model.', true); });
window.addEventListener('resize', drawWave);
window.addEventListener('pagehide', () => stop({ preserveMessage: true }));
$('startButton').addEventListener('click', start);
$('stopButton').addEventListener('click', () => { stop(); $('resultOverline').textContent = 'MONITOR STOPPED'; $('resultText').textContent = 'Standing by'; $('resultNote').textContent = 'Start monitoring to analyze nearby sound.'; $('latency').textContent = 'WAITING FOR AUDIO'; });
$('recordButton').addEventListener('click', toggleRecording);
drawWave();
// Bounded numerical diagnostics only; microphone samples are never exposed here.
window.detectorDiagnostics = () => ({ captureVersion: CAPTURE_VERSION, sampleRate: state.sampleRate,
  captureBlocks: state.captureBlocks, droppedFrames: state.captureDrops, resets: state.resets,
  completedRequests: state.completedRequests, queuedSamples: state.samples.length, active: state.active,
  captureSettings: state.captureSettings, trace: state.trace.slice() });

function downloadTrace() {
  const selected = state.selectedModel || {};
  const report = {
    application: 'Local drone detector',
    model: selected,
    captured_at: new Date().toISOString(), sample_rate: state.sampleRate,
    capture_settings: state.captureSettings,
    capture_summary: { blocks: state.captureBlocks, dropped_frames: state.captureDrops,
      resets: state.resets, completed_requests: state.completedRequests },
    windows: state.trace,
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url;
  link.download = `${(selected.model_id || 'cnn').toLowerCase()}-session-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
$('traceButton').addEventListener('click', downloadTrace);

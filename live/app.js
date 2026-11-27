const $ = (id) => document.getElementById(id);
const canvas = $('waveform');
const ctx = canvas.getContext('2d');
const melCanvas = $('melSpectrum');
const melCtx = melCanvas.getContext('2d', { alpha: false });
const CHUNK_SECONDS = 0.5;
const REQUEST_TIMEOUT_MS = 4000;
const CAPTURE_VERSION = 'pool-v3';
const MEL_FFT_SIZE = 2048;
const MEL_HOP_SIZE = 1024;
const MEL_BANDS = 64;
const MEL_HISTORY_SECONDS = 5;
const MEL_FMIN = 50;
const MEL_FMAX = 7500;
const V4_CHECKPOINT = 'f4372da59644395c174514944aa6387901ed483f64c137fd1ae4e31c83092049';
const PROVISIONAL_THRESHOLD = -3.0;
const PROVISIONAL_VOTE_COUNT = 3;
function isSelectedModel(m) {
  return m?.status === 'ready' && m.application_id === 'local-drone-detector'
    && m.protocol_version === 2 && m.model_id === 'CNN_V4.0' && m.model_version === '4.0'
    && m.stage === 'A0' && m.variant === 'depth_3' && m.seed === 42
    && m.checkpoint_sha256 === V4_CHECKPOINT && m.parameter_count === 23585
    && m.score_kind === 'raw_logit' && m.policy_id === 'raw-score-only-v1'
    && m.threshold === null && m.threshold_selected === false
    && m.smoothing_windows === 0
    && m.window_samples === 7936 && m.cadence_seconds === 0.496
    && m.experimental_activation === true && m.test_access === false;
}
function updateModelUI(m) {
  state.selectedModel = m;
  $('systemText').textContent = 'CNN V4.0 · PROVISIONAL';
  $('activeModelLede').textContent = 'Live drone/no-drone labels use provisional logit −3.00 and a three-window majority, based on one short ambient recording.';
  $('modelStampStatus').textContent = 'PROVISIONAL · AMBIENT-ONLY';
  $('modelStampStatus').classList.add('experimental-status');
  $('activeModelName').textContent = 'CNN V4.0 · A0';
  $('activeModelDetails').textContent = 'Seed 42 · 23,585 parameters';
  $('modelEvidenceActive').textContent = 'LIVE MODEL · V4.0 A0 · seed 42';
  $('modelEvidenceStatus').textContent = 'Provisional client-side decision at logit −3.00. Ambient sample: 9.4 minutes; false-alarm rate is not established. Validation negatives crossed this threshold frequently. Raw logits are not probabilities; test remains locked.';
  $('thresholdValue').textContent = '−3.00 · provisional';
  $('cadenceValue').textContent = `${m.cadence_seconds} sec · 2 of 3 windows`;
}
const state = {
  active: false, epoch: 0, stream: null, context: null, source: null,
  processor: null, samples: new Float32Array(0), sampleRate: 0,
  requestBusy: false, controller: null, sessionId: null, ownerToken: null, nextSequence: 0,
  expectedSourceFrame: null, gapPending: false, lastAudioBlockAt: 0, captureWatchdog: null,
  points: new Float32Array(240), pointAt: 0, capturedAudio: null, mel: null,
  revision: 0, captureBlocks: 0, captureDrops: 0, resets: 0, completedRequests: 0, lastDrawAt: 0,
  recentResets: [],
  recording: null, savingRecording: false,
  trace: [], captureSettings: null, sessionStartedAt: 0, selectedModel: null,
  samplesSourceFrame: null, ambientLogActive: false, ambientLogWindows: 0,
  provisionalVotes: [],
};

function resetProvisionalVotes() {
  state.provisionalVotes = [];
}

function consumeProvisionalDecision(data) {
  if (data.recovered) resetProvisionalVotes();
  const windows = Array.isArray(data.window_trace) && data.window_trace.length
    ? data.window_trace : [data];
  let decision = null;
  for (const window of windows) {
    if (window.recovered) resetProvisionalVotes();
    if (window.status !== 'score_only' || !Number.isFinite(window.raw_score)) {
      resetProvisionalVotes();
      decision = null;
      continue;
    }
    state.provisionalVotes.push(window.raw_score >= PROVISIONAL_THRESHOLD);
    state.provisionalVotes = state.provisionalVotes.slice(-PROVISIONAL_VOTE_COUNT);
    if (state.provisionalVotes.length < PROVISIONAL_VOTE_COUNT) {
      decision = { ready: false, votes: state.provisionalVotes.length };
      continue;
    }
    const drone = state.provisionalVotes.filter(Boolean).length >= 2;
    decision = { ready: true, drone, votes: state.provisionalVotes.length };
  }
  return decision;
}

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
  if (state.capturedAudio?.length) {
    const samples = state.capturedAudio, bins = state.points.length;
    for (let i = 0; i < bins; i++) {
      const start = Math.floor(i * samples.length / bins);
      const end = Math.max(start + 1, Math.floor((i + 1) * samples.length / bins));
      let low = 1, high = -1;
      for (let j = start; j < end; j++) {
        low = Math.min(low, samples[j]); high = Math.max(high, samples[j]);
      }
      const x = i / (bins - 1) * w;
      ctx.moveTo(x, h / 2 - high * h * 0.42);
      ctx.lineTo(x, h / 2 - low * h * 0.42);
    }
  } else {
    for (let i = 0; i < state.points.length; i++) {
      const index = (state.pointAt + i) % state.points.length;
      const x = i / (state.points.length - 1) * w, y = h / 2 - state.points[index] * h * 0.42;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}
function showCapturedAudio(audio = null) {
  state.capturedAudio = audio;
  const captured = Boolean(audio?.length);
  $('waveformMode').textContent = captured ? 'FULL CLIP · 10 SEC' : 'LIVE INPUT';
  $('waveform').setAttribute('aria-label', captured
    ? 'Full 10-second captured microphone audio waveform' : 'Live microphone capture waveform');
  $('levelReadout').textContent = captured ? '10.0 sec' : '— dB';
  drawWave();
}
function hzToHtkMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function htkMelToHz(mel) { return 700 * (10 ** (mel / 2595) - 1); }
function createMelState(sampleRate) {
  const halfBins = MEL_FFT_SIZE / 2 + 1;
  const highHz = Math.min(MEL_FMAX, sampleRate * 0.49);
  const lowMel = hzToHtkMel(MEL_FMIN), highMel = hzToHtkMel(highHz);
  const edges = Array.from({ length: MEL_BANDS + 2 }, (_, i) =>
    htkMelToHz(lowMel + (highMel - lowMel) * i / (MEL_BANDS + 1)));
  const weights = new Float32Array(MEL_BANDS * halfBins);
  for (let band = 0; band < MEL_BANDS; band++) {
    const left = edges[band], center = edges[band + 1], right = edges[band + 2];
    for (let bin = 0; bin < halfBins; bin++) {
      const hz = bin * sampleRate / MEL_FFT_SIZE;
      const triangle = hz < center ? (hz - left) / (center - left) : (right - hz) / (right - center);
      if (triangle > 0) weights[band * halfBins + bin] = triangle * 2 / (right - left);
    }
  }
  return {
    sampleRate, halfBins, weights, hann: Float32Array.from({ length: MEL_FFT_SIZE }, (_, i) =>
      0.5 - 0.5 * Math.cos(2 * Math.PI * i / (MEL_FFT_SIZE - 1))),
    frame: new Float32Array(MEL_FFT_SIZE), writeAt: 0, samplesSeen: 0, sinceFrame: 0,
    real: new Float32Array(MEL_FFT_SIZE), imag: new Float32Array(MEL_FFT_SIZE),
    powers: new Float64Array(MEL_BANDS), historyFrames: Math.ceil(sampleRate * MEL_HISTORY_SECONDS / MEL_HOP_SIZE),
    history: new Float32Array(Math.ceil(sampleRate * MEL_HISTORY_SECONDS / MEL_HOP_SIZE) * MEL_BANDS),
    historyAt: 0, historyCount: 0,
  };
}
function melFrame(mel) {
  const { frame, writeAt, hann, real, imag } = mel;
  for (let i = 0; i < MEL_FFT_SIZE; i++) {
    real[i] = frame[(writeAt + i) % MEL_FFT_SIZE] * hann[i]; imag[i] = 0;
  }
  for (let i = 1, j = 0; i < MEL_FFT_SIZE; i++) {
    let bit = MEL_FFT_SIZE >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  for (let size = 2; size <= MEL_FFT_SIZE; size <<= 1) {
    const half = size >> 1, angle = -2 * Math.PI / size;
    for (let start = 0; start < MEL_FFT_SIZE; start += size) {
      for (let offset = 0; offset < half; offset++) {
        const wr = Math.cos(angle * offset), wi = Math.sin(angle * offset);
        const even = start + offset, odd = even + half;
        const tr = real[odd] * wr - imag[odd] * wi;
        const ti = real[odd] * wi + imag[odd] * wr;
        real[odd] = real[even] - tr; imag[odd] = imag[even] - ti;
        real[even] += tr; imag[even] += ti;
      }
    }
  }
  let peakDb = -Infinity;
  for (let band = 0; band < MEL_BANDS; band++) {
    let power = 0, offset = band * mel.halfBins;
    for (let bin = 0; bin < mel.halfBins; bin++) {
      const magnitude = real[bin] * real[bin] + imag[bin] * imag[bin];
      power += mel.weights[offset + bin] * magnitude;
    }
    mel.powers[band] = power;
    peakDb = Math.max(peakDb, 10 * Math.log10(Math.max(power, 1e-12)));
  }
  const historyOffset = mel.historyAt * MEL_BANDS;
  for (let band = 0; band < MEL_BANDS; band++) {
    const db = 10 * Math.log10(Math.max(mel.powers[band], 1e-12));
    mel.history[historyOffset + band] = peakDb < -90 ? 0 : Math.max(0, Math.min(1, (db - peakDb + 48) / 48));
  }
  mel.historyAt = (mel.historyAt + 1) % mel.historyFrames;
  mel.historyCount = Math.min(mel.historyCount + 1, mel.historyFrames);
}
function drawMelSpectrum() {
  const mel = state.mel;
  if (!mel) return;
  if (melCanvas.width !== mel.historyFrames || melCanvas.height !== MEL_BANDS) {
    melCanvas.width = mel.historyFrames; melCanvas.height = MEL_BANDS;
  }
  const image = melCtx.createImageData(mel.historyFrames, MEL_BANDS);
  const firstEmpty = mel.historyFrames - mel.historyCount;
  for (let x = firstEmpty; x < mel.historyFrames; x++) {
    const sourceFrame = (mel.historyAt - mel.historyCount + x - firstEmpty + mel.historyFrames) % mel.historyFrames;
    for (let band = 0; band < MEL_BANDS; band++) {
      const intensity = mel.history[sourceFrame * MEL_BANDS + band];
      const row = MEL_BANDS - 1 - band, pixel = (row * mel.historyFrames + x) * 4;
      image.data[pixel] = Math.round(12 + 173 * intensity);
      image.data[pixel + 1] = Math.round(28 + 206 * intensity);
      image.data[pixel + 2] = Math.round(25 + 93 * intensity);
      image.data[pixel + 3] = 255;
    }
  }
  melCtx.putImageData(image, 0, 0);
  drawMelGrid();
}
function drawMelGrid() {
  melCtx.save();
  melCtx.strokeStyle = 'rgba(168, 183, 174, 0.16)'; melCtx.lineWidth = 0.5;
  melCtx.beginPath();
  for (let x = 0.5; x < melCanvas.width; x += melCanvas.width / 5) {
    melCtx.moveTo(x, 0); melCtx.lineTo(x, melCanvas.height);
  }
  for (let y = 0.5; y < melCanvas.height; y += melCanvas.height / 4) {
    melCtx.moveTo(0, y); melCtx.lineTo(melCanvas.width, y);
  }
  melCtx.stroke(); melCtx.restore();
}
function resetMelSpectrum() {
  state.mel = null;
  melCanvas.width = 240; melCanvas.height = MEL_BANDS;
  melCtx.fillStyle = '#101a18'; melCtx.fillRect(0, 0, melCanvas.width, melCanvas.height);
  drawMelGrid();
  $('melStatus').textContent = 'WAITING FOR MIC';
  melCanvas.setAttribute('aria-label', 'Live microphone Mel spectrogram; relative energy across 64 bands over the last 5 seconds. Visualization only, separate from detector input.');
}
function processMelAudio(block) {
  if (!state.mel || state.mel.sampleRate !== state.sampleRate) {
    state.mel = createMelState(state.sampleRate);
    $('melStatus').textContent = 'UP TO 5 SEC · 64 BANDS';
  }
  const mel = state.mel;
  let updated = false;
  for (let i = 0; i < block.length; i++) {
    mel.frame[mel.writeAt] = block[i];
    mel.writeAt = (mel.writeAt + 1) % MEL_FFT_SIZE;
    mel.samplesSeen++;
    mel.sinceFrame++;
    if (mel.samplesSeen >= MEL_FFT_SIZE && mel.sinceFrame >= MEL_HOP_SIZE) {
      mel.sinceFrame = 0; melFrame(mel); updated = true;
    }
  }
  if (updated) drawMelSpectrum();
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
  const provisionalDecision = consumeProvisionalDecision(data);
  if (data.recovered) {
    $('result').dataset.state = 'idle'; $('resultOverline').textContent = 'AUDIO RESET';
    $('resultText').textContent = 'Rebuilding recent windows';
    $('resultNote').textContent = 'A capture gap was detected. The provisional vote was reset; fresh audio is needed.';
  }
  if (data.status === 'standby') {
    $('result').dataset.state = 'idle'; $('resultOverline').textContent = data.reason === 'quiet' ? 'LOW INPUT LEVEL' : 'LISTENING';
    $('resultText').textContent = 'Standing by'; $('resultNote').textContent = 'No label during quiet input; it is not counted as a no-drone vote.';
    $('scoreValue').textContent = '—';
  } else if (data.status === 'score_only') {
    const score = Number(data.raw_score);
    $('scoreValue').textContent = score.toFixed(3);
    if (!provisionalDecision?.ready) {
      $('result').dataset.state = 'idle'; $('resultOverline').textContent = 'PROVISIONAL · WARMING UP';
      $('resultText').textContent = 'Checking recent windows';
      $('resultNote').textContent = `${provisionalDecision?.votes || 0}/3 windows · logit threshold −3.00 · labels are provisional`;
    } else {
      $('result').dataset.state = provisionalDecision.drone ? 'drone' : 'clear';
      $('resultOverline').textContent = provisionalDecision.drone ? 'PROVISIONAL DRONE PATTERN' : 'PROVISIONAL NO-DRONE PATTERN';
      $('resultText').textContent = provisionalDecision.drone ? 'Drone detected · provisional' : 'No drone detected · provisional';
      $('resultNote').textContent = `Raw logit ${score.toFixed(3)} · threshold −3.00 · 2 of 3 windows · not validated for field use`;
    }
  } else if (data.status === 'warming_up' || data.status === 'buffering') {
    $('result').dataset.state = 'idle'; $('resultOverline').textContent = 'WARMING UP';
    resetProvisionalVotes();
    $('resultText').textContent = 'Checking recent windows';
    $('resultNote').textContent = data.status === 'buffering' ? 'Building a fresh contiguous audio window.' : 'Rebuilding fresh audio before the provisional vote.';
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
  const hadAmbientLog = state.ambientLogActive;
  state.ambientLogActive = false;
  resetProvisionalVotes();
  state.epoch++; state.active = false; state.sessionId = null;
  if (state.mel?.historyCount) {
    $('melStatus').textContent = 'PAUSED · UP TO 5 SEC';
    melCanvas.setAttribute('aria-label', 'Paused live microphone Mel spectrogram; relative energy across 64 bands over up to the last 5 seconds. Visualization only, separate from detector input.');
  }
  state.expectedSourceFrame = null; state.samplesSourceFrame = null; state.gapPending = false;
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
  updateAmbientControls();
  $('ambientNoDroneConfirm').checked = false;
  if (hadAmbientLog) $('ambientLogStatus').textContent = `Ambient score log closed locally with the monitor session · ${state.ambientLogWindows.toLocaleString()} window records.`;
  if (!preserveMessage) setMessage('Audio is processed locally. Saving a clip requires Record 10 seconds.');
  drawWave();
}
function resetCapture(reason) {
  cancelRecording('Recording cancelled after an audio gap. Nothing was saved; try again when capture is stable.');
  state.revision++; state.resets++;
  resetProvisionalVotes();
  state.recentResets.push(performance.now());
  if (state.recentResets.length > 8) state.recentResets.shift();
  state.samples = new Float32Array(0);
  state.samplesSourceFrame = null;
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
  showCapturedAudio();
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
    showCapturedAudio(recording.samples);
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
function appendAudio(block, sourceFrame) {
  collectRecording(block);
  processMelAudio(block);
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
  if (state.samplesSourceFrame === null) state.samplesSourceFrame = sourceFrame;
  state.samples = merged;
  const sum = block.reduce((total, x) => total + x * x, 0);
  if (!state.capturedAudio) $('levelReadout').textContent = `${(20 * Math.log10(Math.max(Math.sqrt(sum / Math.max(block.length, 1)), 1e-7))).toFixed(0)} dB`;
  if (!state.capturedAudio && performance.now() - state.lastDrawAt >= 80) {
    state.lastDrawAt = performance.now(); drawWave();
  }
  if (!state.requestBusy && state.samples.length >= Math.round(state.sampleRate * CHUNK_SECONDS)) sendAudio();
}
async function sendAudio() {
  if (!state.active || state.requestBusy) return;
  const epoch = state.epoch, revision = state.revision, count = Math.round(state.sampleRate * CHUNK_SECONDS);
  if (state.samples.length < count) return;
  const chunk = state.samples.slice(0, count), sourceFrame = state.samplesSourceFrame;
  state.samples = state.samples.slice(count);
  if (state.samplesSourceFrame !== null) state.samplesSourceFrame += count;
  if (state.gapPending) { state.nextSequence++; state.gapPending = false; }
  const sequence = state.nextSequence++, controller = new AbortController();
  state.controller = controller; state.requestBusy = true;
  const timeout = setTimeout(() => controller.abort(new Error('Detector request timed out.')), REQUEST_TIMEOUT_MS);
  try {
    const started = performance.now();
    const response = await fetch('/api/audio', { method: 'POST', signal: controller.signal, cache: 'no-store',
      headers: { 'Content-Type': 'application/octet-stream', 'X-Sample-Rate': String(state.sampleRate),
        'X-Session-ID': state.sessionId, 'X-Session-Token': state.ownerToken,
        'X-Chunk-Sequence': String(sequence), 'X-Capture-Drops': String(state.captureDrops),
        ...(sourceFrame === null ? {} : { 'X-Source-Frame': String(sourceFrame) }) }, body: chunk.buffer });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Detector returned ${response.status}`);
    if (data.score_log_error && state.ambientLogActive) {
      state.ambientLogActive = false;
      $('ambientLogStatus').textContent = data.score_log_error;
      updateAmbientControls();
    }
    const completedAt = performance.now();
    const windows = data.window_trace || [data];
    if (!state.ambientLogActive) windows.forEach((result, index) => state.trace.push({
      elapsed_seconds: Number(((completedAt - state.sessionStartedAt) / 1000).toFixed(3)),
      sequence, window_in_chunk: index, source_sample_rate: state.sampleRate,
      chunk_seconds: CHUNK_SECONDS,
      request_ms: Number((completedAt - started).toFixed(1)),
      status: result.status, reason: result.reason || null,
      window_rms_dbfs: result.window_rms_dbfs ?? null,
      raw_logit: result.raw_score ?? null, threshold: result.threshold ?? null,
      raw_vote: result.raw_vote ?? null, votes: result.votes ?? [],
      votes_required: result.votes_required ?? 0, drone: result.drone ?? null,
      generation: result.generation ?? data.generation ?? null,
      recovered: Boolean(result.recovered || data.recovered),
      recovery_reasons: result.recovery_reasons || data.recovery_reasons || [],
      dropped_frames: state.captureDrops,
    }));
    if (state.ambientLogActive) state.ambientLogWindows += windows.filter(
      (result) => Number.isFinite(result.window_rms_dbfs)).length;
    if (windows.length && !state.ambientLogActive) $('traceButton').disabled = false;
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
  resetMelSpectrum();
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
    resetProvisionalVotes();
    state.sessionStartedAt = performance.now();
    state.captureSettings = device.getAudioTracks()[0]?.getSettings?.() || {};
    $('traceButton').disabled = true;
    const sessionId = crypto.randomUUID();
    state.sessionId = sessionId; state.nextSequence = 0;
    state.expectedSourceFrame = null; state.gapPending = false;
    state.samplesSourceFrame = null;
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
        const staleEndFrame = block.sourceFrame + samples.length;
        const lostFrames = state.expectedSourceFrame === null
          ? (block.droppedFrames || 0) + samples.length
          : Math.max(block.droppedFrames || 0, staleEndFrame - state.expectedSourceFrame);
        state.captureDrops += lostFrames;
        state.expectedSourceFrame = staleEndFrame;
        if (!state.gapPending) resetCapture('Discarding delayed audio. Waiting for fresh microphone input.');
        recycle(); return;
      }
      if (block.droppedFrames > 0 || (state.expectedSourceFrame !== null && block.sourceFrame !== state.expectedSourceFrame)) {
        state.captureDrops += block.droppedFrames || Math.max(0, block.sourceFrame - state.expectedSourceFrame);
        resetCapture('The mic stream skipped audio. Rebuilding a fresh raw-score window needs about 0.5 sec of uninterrupted input. If this keeps repeating, check the mic and keep this tab active.');
      }
      state.expectedSourceFrame = block.sourceFrame + samples.length;
      try { appendAudio(samples, block.sourceFrame); } finally { recycle(); }
    };
    state.processor.addEventListener?.('processorerror', () => {
      if (epoch !== state.epoch || !state.active) return;
      stop({ preserveMessage: true });
      showError('Microphone processing stopped', 'The browser audio worklet failed. Press Start monitoring to create a fresh capture session.');
      setMessage('Browser microphone processing stopped and was reset.', true);
    }, { once: true });
    state.source.connect(state.processor); state.processor.connect(state.context.destination); state.active = true;
    updateRecordButton();
    updateAmbientControls();
    state.lastAudioBlockAt = performance.now();
    state.captureWatchdog = setTimeout(() => checkCaptureHealth(epoch), 2500);
    $('micDot').classList.add('on'); $('micLabel').textContent = 'MICROPHONE LIVE'; $('latency').textContent = 'BUILDING FIRST DECISION';
    $('sampleRate').textContent = `${state.sampleRate.toLocaleString()} Hz`; $('result').dataset.state = 'idle';
    $('resultOverline').textContent = 'CAPTURING AUDIO'; $('resultText').textContent = 'Listening';
    $('resultNote').textContent = 'Preparing the first 0.496-second window. Quiet input is shown as standby.';
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

function updateAmbientControls() {
  const logging = state.active && state.ambientLogActive;
  $('ambientLogButton').disabled = !state.active || logging || !$('ambientNoDroneConfirm').checked;
  $('stopAmbientLogButton').disabled = !logging;
  $('ambientNoDroneConfirm').disabled = !state.active || logging;
}
async function toggleAmbientLog() {
  if (!state.active || !state.sessionId || !state.ownerToken) return;
  if (state.ambientLogActive) {
    $('stopAmbientLogButton').disabled = true;
    try {
      const response = await fetch('/api/score-log/stop', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId, owner_token: state.ownerToken }) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not stop the local score log.');
      state.ambientLogActive = false;
      $('ambientLogStatus').textContent = result.status === 'saved'
        ? `Saved ${result.window_records.toLocaleString()} score records locally · ${result.filename}. No audio was saved.`
        : 'No active ambient log was found. Check that a complete session file was saved locally.';
    } catch (error) {
      state.ambientLogActive = false;
      $('ambientLogStatus').textContent = `Could not confirm log closure: ${error.message || 'connection interrupted'}. Stop monitoring and check whether the local log has a complete end record.`;
    } finally { updateAmbientControls(); }
    return;
  }
  if (!$('ambientNoDroneConfirm').checked) return;
  $('ambientLogButton').disabled = true;
  try {
    const response = await fetch('/api/score-log/start', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, owner_token: state.ownerToken,
        confirmed_ambient_only: true, threshold_selected: false,
        environment_label: $('environmentLabel').value.trim() }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not start the local score log.');
    state.ambientLogActive = true; state.ambientLogWindows = 0;
    state.trace = []; $('traceButton').disabled = true;
    $('ambientLogStatus').textContent = `Logging timestamped scores locally · ${result.filename}. No audio is saved. Keep this tab open for the collection period.`;
  } catch (error) {
    $('ambientLogStatus').textContent = `Ambient log did not start: ${error.message || 'local service unavailable'}. Monitoring continues without logging.`;
  } finally { updateAmbientControls(); }
}
$('ambientLogButton').addEventListener('click', toggleAmbientLog);
$('stopAmbientLogButton').addEventListener('click', toggleAmbientLog);
$('ambientNoDroneConfirm').addEventListener('change', updateAmbientControls);
drawWave();
resetMelSpectrum();
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

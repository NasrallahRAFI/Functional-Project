import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const selection = JSON.parse(fs.readFileSync(path.join(root, 'config/live_detector.json'), 'utf8'));
const readyHealth = () => ({ status: 'ready', application_id: 'local-drone-detector',
  protocol_version: 2, model_version: selection.model_version, seed: selection.seed,
  model_id: selection.model_id, variant: selection.variant, threshold: selection.threshold,
  stage: selection.stage, validation_eligible: selection.validation_eligible,
  experimental_activation: selection.experimental_activation,
  validation_guard_failures: selection.validation_guard_failures,
  test_access: selection.test_access, run_id: selection.run_id,
  validation_metrics: selection.validation_metrics || null,
  final_test_passed: selection.final_test_passed || false,
  checkpoint_sha256: selection.model_sha256, score_kind: selection.score_kind,
  policy_id: selection.policy_id, smoothing_windows: selection.smoothing_windows,
  window_samples: 7936, cadence_seconds: 0.496 });

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

test('page hide during session claim releases late ownership and microphone tracks', async () => {
  const elements = new Map();
  const makeElement = (id) => ({
    id, textContent: '', dataset: {}, disabled: false, listeners: {},
    addEventListener(name, fn) { this.listeners[name] = fn; },
    classList: { add() {}, remove() {}, toggle() {} },
    querySelector: () => ({ textContent: '' }),
    getBoundingClientRect: () => ({ width: 320, height: 80 }),
    getContext: () => ({ clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {} }),
  });
  const document = { getElementById(id) { if (!elements.has(id)) elements.set(id, makeElement(id)); return elements.get(id); } };
  const windowListeners = new Map();
  const window = { devicePixelRatio: 1, addEventListener(name, fn) { windowListeners.set(name, fn); } };
  const claim = deferred();
  const calls = [];
  let trackStopped = false;
  const response = (data, ok = true) => ({ ok, status: ok ? 200 : 409, json: async () => data });
  const fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (url === '/api/health') return response(readyHealth());
    if (url === '/api/session/start') return claim.promise;
    if (url === '/api/session/stop') return response({ status: 'stopped' });
    throw new Error(`Unexpected fetch ${url}`);
  };
  class FakeAudioContext {
    constructor() { this.state = 'running'; this.sampleRate = 44100; this.destination = {}; }
    async resume() {}
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createScriptProcessor() { return { connect() {}, disconnect() {}, onaudioprocess: null }; }
    async close() { this.state = 'closed'; }
    addEventListener() {}
  }
  const context = {
    document, window, fetch, console, URL, AbortController,
    navigator: { mediaDevices: { getUserMedia: async () => ({
      getAudioTracks: () => [{ readyState: 'live', addEventListener() {} }],
      getTracks: () => [{ stop() { trackStopped = true; } }],
    }) } },
    AudioContext: FakeAudioContext,
    crypto: { randomUUID: () => 'test-session' },
    performance: { now: () => 0 },
    setTimeout, clearTimeout,
    Float32Array, Math, Number, Error, Promise,
  };
  vm.runInNewContext(fs.readFileSync(path.join(root, 'live/app.js'), 'utf8'), context);
  await new Promise((resolve) => setImmediate(resolve));

  const start = document.getElementById('startButton').listeners.click;
  assert.equal(typeof start, 'function');
  const startPromise = start();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(calls.some((call) => call.url === '/api/session/start'), true);

  windowListeners.get('pagehide')();
  claim.resolve(response({ owner_token: 'owner-token' }));
  await startPromise;
  await new Promise((resolve) => setImmediate(resolve));

  const release = calls.find((call) => call.url === '/api/session/stop');
  assert.ok(release, 'late session ownership is released');
  assert.equal(JSON.parse(release.options.body).owner_token, 'owner-token');
  assert.equal(trackStopped, true, 'microphone track stops on page hide');
});

test('a score arriving after Stop cannot overwrite the stopped monitor state', async () => {
  const elements = new Map();
  let workletNode;
  let watchdog;
  const makeElement = (id) => ({
    id, textContent: '', dataset: {}, disabled: false, listeners: {},
    addEventListener(name, fn) { this.listeners[name] = fn; },
    classList: { add() {}, remove() {}, toggle() {} }, querySelector: () => ({ textContent: '' }),
    getBoundingClientRect: () => ({ width: 320, height: 80 }),
    getContext: () => ({ clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {} }),
  });
  const document = { getElementById(id) { if (!elements.has(id)) elements.set(id, makeElement(id)); return elements.get(id); } };
  const window = { devicePixelRatio: 1, addEventListener() {} };
  const audio = deferred();
  const calls = [];
  const audioSequences = [];
  const recordings = [];
  const response = (data) => ({ ok: true, status: 200, json: async () => data });
  const fetch = async (url, options = {}) => {
    calls.push(url);
    if (url === '/api/health') return response(readyHealth());
    if (url === '/api/session/start') return response({ owner_token: 'owner' });
    if (url === '/api/diagnostic-recording') {
      recordings.push(options.body);
      return response({ filename: 'synthetic.wav', duration_seconds: 10 });
    }
    if (url === '/api/audio') {
      audioSequences.push(options.headers['X-Chunk-Sequence']);
      return { ok: true, status: 200, json: () => audio.promise };
    }
    if (url === '/api/session/stop') return response({ status: 'stopped' });
    throw new Error(`Unexpected fetch ${url}`);
  };
  class FakeAudioContext {
    constructor() { this.state = 'running'; this.sampleRate = 44100; this.destination = {}; this.audioWorklet = { addModule: async () => {} }; }
    async resume() {}
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    async close() { this.state = 'closed'; }
    addEventListener() {}
  }
  class FakeAudioWorkletNode {
    constructor() { this.port = { onmessage: null, postMessage() {} }; workletNode = this; }
    connect() {}
    disconnect() {}
  }
  const context = {
    document, window, fetch, console, AbortController,
    navigator: { mediaDevices: { getUserMedia: async () => ({
      getAudioTracks: () => [{ readyState: 'live', addEventListener() {} }],
      getTracks: () => [{ stop() {} }],
    }) } },
    AudioContext: FakeAudioContext, AudioWorkletNode: FakeAudioWorkletNode,
    crypto: { randomUUID: () => 'session' },
    performance: { now: () => 0 },
    setTimeout(callback, delay) { if (delay === 2500) watchdog = callback; return setTimeout(callback, delay); },
    clearTimeout,
    Float32Array, Math, Number, Error, Promise,
  };
  vm.runInNewContext(fs.readFileSync(path.join(root, 'live/app.js'), 'utf8'), context);
  await new Promise((resolve) => setImmediate(resolve));
  await document.getElementById('startButton').listeners.click();
  assert.ok(workletNode);

  const samples = new Float32Array(2048);
  for (let i = 0; i < 5; i++) {
    workletNode.port.onmessage({ data: { type: 'audio-block', sourceFrame: i * samples.length,
      samples, droppedFrames: 0 } });
  }
  workletNode.port.onmessage({ data: { type: 'audio-block', sourceFrame: 6 * samples.length,
    samples, droppedFrames: samples.length } });
  assert.equal(document.getElementById('resultText').textContent, 'Rebuilding recent window',
    'a single skipped source block resets the buffer before a mixed window can be sent');
  for (let i = 7; i < 17; i++) {
    workletNode.port.onmessage({ data: { type: 'audio-block', sourceFrame: i * samples.length,
      samples, droppedFrames: 0 } });
  }
  assert.equal(calls.filter((url) => url === '/api/audio').length, 1, 'only one request is in flight');
  assert.deepEqual(audioSequences, ['1'], 'the capture gap is signaled as a missing sequence to reset server audio history');
  document.getElementById('stopButton').listeners.click();
  audio.resolve({ status: 'scored', drone: true, score: 9.2, threshold: 2.8240 });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(document.getElementById('resultOverline').textContent, 'MONITOR STOPPED');
  assert.equal(document.getElementById('resultText').textContent, 'Standing by');

  await document.getElementById('startButton').listeners.click();
  for (let i = 0; i < 9; i++) {
    workletNode.port.onmessage({ data: { type: 'audio-block', sourceFrame: i * 4096,
      samples, droppedFrames: i === 0 ? 0 : 2048 } });
  }
  watchdog();
  assert.equal(document.getElementById('resultText').textContent, 'Audio capture is unstable',
    'persistent gaps stop capture with an actionable error instead of rebuilding indefinitely');
  assert.equal(document.getElementById('stopButton').disabled, true);

  await document.getElementById('startButton').listeners.click();
  document.getElementById('recordButton').listeners.click();
  for (let i = 0; i < 216; i++) {
    workletNode.port.onmessage({ data: { type: 'audio-block', sourceFrame: i * 2048, samples, droppedFrames: 0 } });
    if (i % 10 === 0) await new Promise((resolve) => setImmediate(resolve));
  }
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(recordings.length, 1);
  assert.equal(recordings[0].byteLength, 44100 * 10 * 4);
  assert.match(document.getElementById('recordingStatus').textContent, /Saved synthetic.wav/);
  document.getElementById('recordButton').listeners.click();
  document.getElementById('stopButton').listeners.click();
  assert.equal(recordings.length, 1, 'stopping cancels a partial recording without uploading it');
});


test('the browser rejects a wrong artifact, threshold, or decision contract before microphone access', async () => {
  for (const change of [{checkpoint_sha256: '0'.repeat(64)}, {threshold: 0},
      {protocol_version: 1}, {score_kind: 'probability'}, {smoothing_windows: 1}]) {
    const elements = new Map();
    const document = {getElementById(id) {
      if (!elements.has(id)) elements.set(id, {textContent: '', dataset: {},
        addEventListener(name, fn) {this[name] = fn;}, classList: {add() {}, remove() {}, toggle() {}},
        querySelector: () => ({}), getBoundingClientRect: () => ({width: 320, height: 80}),
        getContext: () => ({clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}})});
      return elements.get(id);
    }};
    let microphoneRequests = 0;
    vm.runInNewContext(fs.readFileSync(path.join(root, 'live/app.js'), 'utf8'), {
      document, window: {addEventListener() {}}, Float32Array, AbortController,
      fetch: async () => ({ok: true, json: async () => ({...readyHealth(), ...change})}),
      navigator: {mediaDevices: {getUserMedia() {microphoneRequests++; throw new Error('unexpected mic');}}},
      setTimeout, clearTimeout,
    });
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(elements.get('systemText').textContent, 'MODEL SERVICE UNAVAILABLE');
    await elements.get('startButton').click();
    assert.equal(microphoneRequests, 0);
    assert.match(elements.get('message').textContent, /does not match the verified local CNN package/);
  }
});

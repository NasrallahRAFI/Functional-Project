import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function captureHarness() {
  let Processor;
  const messages = [];
  class AudioWorkletProcessor {
    constructor() {
      this.port = { onmessage: null, postMessage(data, transfer) {
        messages.push(structuredClone(data, { transfer }));
      } };
    }
  }
  const sandbox = {
    AudioWorkletProcessor,
    currentFrame: 0,
    registerProcessor(name, definition) { assert.equal(name, 'mono-capture'); Processor = definition; },
    Float32Array,
    ArrayBuffer,
  };
  vm.runInNewContext(fs.readFileSync(path.join(root, 'live/audio-capture-worklet.js'), 'utf8'), sandbox);
  const processor = new Processor();
  const render = (startFrame, frames = 128, missing = false) => {
    sandbox.currentFrame = startFrame;
    const output = new Float32Array(frames).fill(1);
    const left = new Float32Array(frames).fill(0.2), right = new Float32Array(frames).fill(0.4);
    processor.process([missing ? [] : [left, right]], [[output]]);
    assert.equal(output.every((value) => value === 0), true, 'capture node produces silence to the speakers');
  };
  const recycle = (message) => {
    const data = structuredClone({ type: 'recycle', buffer: message.samples.buffer },
      { transfer: [message.samples.buffer] });
    processor.port.onmessage({ data });
  };
  return { messages, processor, render, recycle };
}

test('five minutes of capture tolerate delayed buffer recycling without gaps at 48 kHz', () => {
  const { messages, render, recycle } = captureHarness();
  const pending = [];
  let at = 0, nextExpected = 0;
  for (let tick = 0; tick < 300 * 48000 / 128; tick++) {
    // Main-thread response varies from 16 to 64 ms, much longer than one render tick.
    while (pending.length && pending[0].due <= tick) recycle(pending.shift().message);
    render(tick * 128);
    while (at < messages.length) {
      const message = messages[at++];
      assert.equal(message.droppedFrames, 0);
      assert.equal(message.sourceFrame, nextExpected);
      nextExpected += 2048;
      assert.ok(Math.abs(message.samples[0] - 0.3) < 1e-6);
      pending.push({ due: tick + 6 + (at % 19), message });
    }
    // Simulate delivery scheduling independently of capture ticks.
    pending.sort((a, b) => a.due - b.due);
  }
  assert.ok(at > 7000);
});

test('missing input counts the partial block plus the actual missing quantum size', () => {
  const { messages, processor, render } = captureHarness();
  for (let frame = 0; frame < 1024; frame += 128) render(frame);
  render(1024, 256, true);
  for (let frame = 1280; frame < 3328; frame += 128) render(frame);

  assert.equal(messages.length, 1);
  assert.equal(messages[0].sourceFrame, 1280);
  assert.equal(messages[0].droppedFrames, 1280);
  assert.equal(processor.droppedFrames, 0);
});

test('pool exhaustion inside a large render quantum drops its remainder safely', () => {
  const { messages, render, recycle } = captureHarness();
  let sourceFrame = 0;
  for (let tick = 0; tick < 35; tick++) {
    render(sourceFrame, 480);
    sourceFrame += 480;
  }

  assert.equal(messages.length, 8);
  recycle(messages[0]);
  while (messages.length < 9) {
    render(sourceFrame, 480);
    sourceFrame += 480;
  }
  assert.equal(messages[8].sourceFrame, 16800);
  assert.equal(messages[8].droppedFrames, 416);
});

test('exhausted pool bounds queued blocks and reports a real gap before recovery', () => {
  const { messages, render, recycle } = captureHarness();

  for (let block = 0; block < 16; block++) render(block * 128);
  assert.equal(messages.length, 1);
  assert.equal(messages[0].sourceFrame, 0);
  assert.equal(messages[0].droppedFrames, 0);
  assert.ok(Math.abs(messages[0].samples[0] - 0.3) < 1e-6, 'input channels are averaged to mono');

  for (let block = 16; block < 160; block++) render(block * 128);
  assert.equal(messages.length, 8, 'the pool permits at most eight outstanding blocks');
  recycle(messages[0]);
  for (let block = 160; block < 176; block++) render(block * 128);
  assert.equal(messages.length, 9);
  assert.equal(messages[8].sourceFrame, 20480);
  assert.equal(messages[8].droppedFrames, 4096);
});

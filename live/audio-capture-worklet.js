const CAPTURE_BLOCK_FRAMES = 2048;
const CAPTURE_POOL_SIZE = 8;

class MonoCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.freeBlocks = Array.from({ length: CAPTURE_POOL_SIZE }, () => new Float32Array(CAPTURE_BLOCK_FRAMES));
    this.block = this.freeBlocks.pop();
    this.writeAt = 0;
    this.blockStartFrame = null;
    this.droppedFrames = 0;
    this.stopped = false;
    this.port.onmessage = ({ data }) => {
      if (data?.type === 'recycle' && data.buffer instanceof ArrayBuffer &&
          data.buffer.byteLength === CAPTURE_BLOCK_FRAMES * 4 && this.freeBlocks.length < CAPTURE_POOL_SIZE) {
        this.freeBlocks.push(new Float32Array(data.buffer));
      }
      if (data?.type === 'stop') this.stopped = true;
    };
  }

  process(inputs, outputs) {
    for (const channel of outputs[0] || []) channel.fill(0);
    if (this.stopped) return false;

    const channels = inputs[0] || [];
    if (!channels.length || !channels[0].length) {
      this.droppedFrames += 128;
      this.writeAt = 0;
      this.blockStartFrame = null;
      return true;
    }

    const frames = channels[0].length;
    if (!this.block) this.block = this.freeBlocks.pop();
    if (!this.block) {
      this.droppedFrames += frames;
      this.writeAt = 0;
      this.blockStartFrame = null;
      return true;
    }

    for (let frame = 0; frame < frames; frame++) {
      if (this.writeAt === 0) this.blockStartFrame = currentFrame + frame;
      let mixed = 0;
      for (const channel of channels) mixed += channel[frame] || 0;
      this.block[this.writeAt++] = mixed / channels.length;

      if (this.writeAt === CAPTURE_BLOCK_FRAMES) {
        const samples = this.block;
        this.block = this.freeBlocks.pop();
        this.writeAt = 0;
        this.port.postMessage({
          type: 'audio-block',
          sourceFrame: this.blockStartFrame,
          samples,
          droppedFrames: this.droppedFrames,
        }, [samples.buffer]);
        this.droppedFrames = 0;
        this.blockStartFrame = null;
      }
    }
    return true;
  }
}

registerProcessor('mono-capture', MonoCaptureProcessor);

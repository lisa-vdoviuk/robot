class TalkCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = false;
    this.reset();
    this.port.onmessage = ({data}) => {
      if (data === 'start') { this.reset(); this.active = true; }
      if (data === 'stop') {
        this.active = false;
        if (this.count) this.port.postMessage(this.block.slice(0, this.count));
        this.port.postMessage('ended');
        this.reset();
      }
    };
  }
  reset() {
    this.phase = 0;
    this.sum = 0;
    this.weight = 0;
    this.block = new Float32Array(3200);
    this.count = 0;
  }
  process(inputs) {
    const input = inputs[0][0];
    if (!this.active || !input) return true;
    const ratio = sampleRate / 16000;
    // Continuous weighted averaging; retain fractional phase across callbacks.
    for (const value of input) {
      let remaining = 1;
      while (remaining > 1e-8) {
        const take = Math.min(remaining, ratio - this.phase);
        this.sum += value * take;
        this.weight += take;
        this.phase += take;
        remaining -= take;
        if (this.phase >= ratio - 1e-8) {
          this.block[this.count++] = this.sum / this.weight;
          this.phase = this.sum = this.weight = 0;
          if (this.count === 3200) {
            this.port.postMessage(this.block, [this.block.buffer]);
            this.block = new Float32Array(3200);
            this.count = 0;
          }
        }
      }
    }
    return true;
  }
}
registerProcessor('talk-capture', TalkCapture);

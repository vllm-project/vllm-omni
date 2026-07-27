class FullDuplexPcmPlayback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.playedFrames = 0;
    this.underrunFrames = 0;
    this.drain = null;
    this.started = false;
    this.activeResponseId = null;
    this.initialBufferFrames = Math.round(sampleRate * 0.2);
    this.bufferWaitFrames = this.initialBufferFrames;
    this.rebuffering = false;
    this.fadeFrames = Math.max(1, Math.round(sampleRate * 0.005));
    this.fadeInFrames = 0;
    this.port.onmessage = (event) => this.handleMessage(event.data || {});
  }

  handleMessage(message) {
    if (message.type === 'audio' && message.pcm) {
      const wasEmpty = this.queue.length === 0;
      if (!this.started && !this.activeResponseId) {
        this.activeResponseId = message.responseId || null;
      }
      if (!this.started && Number.isFinite(message.initialBufferMs)) {
        this.initialBufferFrames = Math.max(0, Math.round((sampleRate * message.initialBufferMs) / 1000));
      }
      this.queue.push(message.pcm);
      if (!this.started && wasEmpty && !this.rebuffering) {
        this.bufferWaitFrames = this.initialBufferFrames;
      }
    } else if (message.type === 'drain') {
      this.drain = { responseId: message.responseId || null };
      if (!this.started && this.bufferedFrames() > 0) {
        this.bufferWaitFrames = 0;
        this.startPlayback();
      }
      this.notifyIfDrained();
    } else if (message.type === 'clear') {
      this.queue = [];
      this.offset = 0;
      this.playedFrames = 0;
      this.underrunFrames = 0;
      this.drain = null;
      this.started = false;
      this.activeResponseId = null;
      this.bufferWaitFrames = this.initialBufferFrames;
      this.rebuffering = false;
      this.fadeInFrames = 0;
    }
  }

  bufferedFrames() {
    return this.queue.reduce((total, pcm, index) => (
      total + pcm.length - (index === 0 ? this.offset : 0)
    ), 0);
  }

  startPlayback() {
    if (this.started) return;
    this.started = true;
    this.fadeInFrames = this.fadeFrames;
    if (this.playedFrames === 0) {
      this.port.postMessage({ type: 'playback-started', responseId: this.activeResponseId });
    }
  }

  notifyIfDrained() {
    if (!this.drain || this.queue.length > 0) return;
    this.port.postMessage({
      type: 'playback-drained',
      responseId: this.drain.responseId,
      playedMs: Math.round((this.playedFrames * 1000) / sampleRate),
      underrunMs: Math.round((this.underrunFrames * 1000) / sampleRate),
    });
    this.playedFrames = 0;
    this.underrunFrames = 0;
    this.drain = null;
    this.started = false;
    this.activeResponseId = null;
    this.bufferWaitFrames = this.initialBufferFrames;
    this.rebuffering = false;
    this.fadeInFrames = 0;
  }

  reportUnderrun() {
    this.port.postMessage({
      type: 'playback-underrun',
      responseId: this.activeResponseId,
      underrunMs: Math.round((this.underrunFrames * 1000) / sampleRate),
    });
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    if (!this.started) {
      if (this.rebuffering && !this.drain) {
        this.underrunFrames += output.length;
        this.reportUnderrun();
      }
      if (this.queue.length > 0 && this.bufferWaitFrames > 0) {
        this.bufferWaitFrames = Math.max(0, this.bufferWaitFrames - output.length);
        return true;
      }
      if (this.queue.length > 0) {
        this.startPlayback();
        this.rebuffering = false;
      }
    }
    if (!this.started) {
      this.notifyIfDrained();
      return true;
    }
    let target = 0;
    while (target < output.length && this.queue.length > 0) {
      const pcm = this.queue[0];
      const count = Math.min(output.length - target, pcm.length - this.offset);
      for (let index = 0; index < count; index += 1) {
        let sample = pcm[this.offset + index] / 32768;
        if (this.fadeInFrames > 0) {
          const elapsed = this.fadeFrames - this.fadeInFrames;
          sample *= elapsed / this.fadeFrames;
          this.fadeInFrames -= 1;
        }
        output[target + index] = sample;
      }
      target += count;
      this.offset += count;
      this.playedFrames += count;
      if (this.offset >= pcm.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (target < output.length && !this.drain) {
      const fadeCount = Math.min(target, this.fadeFrames);
      for (let index = 0; index < fadeCount; index += 1) {
        output[target - fadeCount + index] *= (fadeCount - index - 1) / fadeCount;
      }
      this.underrunFrames += output.length - target;
      this.reportUnderrun();
      this.started = false;
      this.rebuffering = true;
      this.bufferWaitFrames = this.initialBufferFrames;
      this.fadeInFrames = 0;
    }
    this.notifyIfDrained();
    return true;
  }
}

registerProcessor('fullduplex-pcm-playback', FullDuplexPcmPlayback);

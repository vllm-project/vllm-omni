class FullDuplexPcmPlayback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.pendingMessages = [];
    this.offset = 0;
    this.playedFrames = 0;
    this.underrunFrames = 0;
    this.drain = null;
    this.terminalFadeFrames = 0;
    this.started = false;
    this.activeResponseId = null;
    this.initialBufferFrames = Math.round(sampleRate * 0.2);
    this.bufferWaitFrames = this.initialBufferFrames;
    this.rebuffering = false;
    this.fadeFrames = Math.max(1, Math.round(sampleRate * 0.005));
    this.fadeInFrames = 0;
    this.playbackSpeed = 1;
    this.stretch = null;
    this.audioChunks = 0;
    this.holdUntilSecondChunk = false;
    this.port.onmessage = (event) => this.handleMessage(event.data || {});
  }

  handleMessage(message) {
    // Keep each response's samples and drain marker together. The next
    // response can finish generating before this one finishes playing.
    if (message.type !== 'clear' && this.activeResponseId
        && message.responseId && message.responseId !== this.activeResponseId) {
      this.pendingMessages.push(message);
      return;
    }
    if (message.type === 'audio' && message.pcm) {
      const wasEmpty = this.queue.length === 0;
      if (!this.started && !this.activeResponseId) {
        this.activeResponseId = message.responseId || null;
      }
      if (!this.started && Number.isFinite(message.initialBufferMs)) {
        this.initialBufferFrames = Math.max(0, Math.round((sampleRate * message.initialBufferMs) / 1000));
      }
      if (!this.started && Number.isFinite(message.playbackSpeed) && message.playbackSpeed > 0) {
        this.playbackSpeed = message.playbackSpeed;
      }
      if (!this.started && message.holdUntilSecondChunk === true) {
        this.holdUntilSecondChunk = true;
      }
      this.audioChunks += 1;
      this.queue.push(message.pcm);
      if (!this.started && wasEmpty && !this.rebuffering) {
        this.bufferWaitFrames = this.initialBufferFrames;
      }
    } else if (message.type === 'drain') {
      this.drain = { responseId: message.responseId || null };
      this.terminalFadeFrames = Math.min(this.fadeFrames, this.bufferedFrames());
      if (!this.started && this.bufferedFrames() > 0) {
        this.bufferWaitFrames = 0;
        this.startPlayback();
      }
      this.notifyIfDrained();
    } else if (message.type === 'clear') {
      if (this.activeResponseId) {
        this.port.postMessage({
          type: 'playback-stopped',
          responseId: this.activeResponseId,
          playedMs: Math.round((this.playedFrames * 1000) / sampleRate),
        });
      }
      this.queue = [];
      this.pendingMessages = [];
      this.offset = 0;
      this.playedFrames = 0;
      this.underrunFrames = 0;
      this.drain = null;
      this.terminalFadeFrames = 0;
      this.started = false;
      this.activeResponseId = null;
      this.bufferWaitFrames = this.initialBufferFrames;
      this.rebuffering = false;
      this.fadeInFrames = 0;
      this.stretch = null;
      this.audioChunks = 0;
      this.holdUntilSecondChunk = false;
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
    if (this.stretch && this.stretch.fifoLen > 0) return;
    this.port.postMessage({
      type: 'playback-drained',
      responseId: this.drain.responseId,
      playedMs: Math.round((this.playedFrames * 1000) / sampleRate),
      underrunMs: Math.round((this.underrunFrames * 1000) / sampleRate),
    });
    this.playedFrames = 0;
    this.underrunFrames = 0;
    this.drain = null;
    this.terminalFadeFrames = 0;
    this.started = false;
    this.activeResponseId = null;
    this.bufferWaitFrames = this.initialBufferFrames;
    this.rebuffering = false;
    this.fadeInFrames = 0;
    this.stretch = null;
    this.audioChunks = 0;
    this.holdUntilSecondChunk = false;
    const pending = this.pendingMessages;
    this.pendingMessages = [];
    for (const message of pending) this.handleMessage(message);
  }

  ensureStretch() {
    if (this.stretch) return this.stretch;
    const synthHop = Math.round(sampleRate * 0.01);
    const win = synthHop * 2;
    const speed = this.playbackSpeed;
    const window = new Float32Array(win);
    for (let index = 0; index < win; index += 1) {
      window[index] = 0.5 * (1 - Math.cos((2 * Math.PI * index) / win));
    }
    this.stretch = {
      win,
      synthHop,
      anaHop: Math.max(1, Math.round(synthHop * speed)),
      search: Math.round(sampleRate * 0.012),
      window,
      prev: new Float32Array(synthHop),
      hasPrev: false,
      cursor: 0,
      inputBase: 0,
      fifo: new Float32Array(win * 2),
      fifoLen: 0,
      done: false,
    };
    return this.stretch;
  }

  // Absolute index 0 is the first sample queued for this response.
  inputSample(absIndex) {
    const stretch = this.stretch;
    let rel = absIndex - stretch.inputBase;
    for (let index = 0; index < this.queue.length; index += 1) {
      const pcm = this.queue[index];
      const start = index === 0 ? this.offset : 0;
      const len = pcm.length - start;
      if (rel < len) return pcm[start + rel];
      rel -= len;
    }
    return 0;
  }

  inputEnd() {
    return this.stretch.inputBase + this.bufferedFrames();
  }

  discardBefore(absIndex) {
    const stretch = this.stretch;
    while (this.queue.length > 0 && stretch.inputBase < absIndex) {
      const pcm = this.queue[0];
      const chunkEnd = stretch.inputBase + (pcm.length - this.offset);
      if (chunkEnd > absIndex) {
        const drop = absIndex - stretch.inputBase;
        this.offset += drop;
        stretch.inputBase = absIndex;
        return;
      }
      stretch.inputBase = chunkEnd;
      this.queue.shift();
      this.offset = 0;
    }
  }

  pushStretch(sample) {
    const stretch = this.stretch;
    if (stretch.fifoLen >= stretch.fifo.length) {
      const grown = new Float32Array(stretch.fifo.length * 2);
      grown.set(stretch.fifo);
      stretch.fifo = grown;
    }
    stretch.fifo[stretch.fifoLen] = sample;
    stretch.fifoLen += 1;
  }

  // One WSOLA hop: overlap-add a grain locked to the previous period, then
  // advance through the input by anaHop. Hann windows at 50% overlap sum to 1,
  // so the pitch stays put while the utterance gets shorter.
  synthesizeHop(forceTail) {
    const stretch = this.stretch;
    if (stretch.done) return;
    const end = this.inputEnd();
    const need = stretch.cursor + stretch.win + (stretch.hasPrev ? stretch.search : 0);
    if (!forceTail && end < need) return;
    const remain = end - stretch.cursor;
    if (remain <= 0) {
      if (stretch.hasPrev) {
        for (let index = 0; index < stretch.synthHop; index += 1) this.pushStretch(stretch.prev[index]);
      }
      stretch.done = true;
      this.discardBefore(end);
      return;
    }
    const nominal = stretch.cursor;
    let best = nominal;
    if (stretch.hasPrev && !forceTail) {
      const lo = Math.max(stretch.inputBase, nominal - stretch.search);
      const hi = Math.min(nominal + stretch.search, end - stretch.win);
      let bestScore = -Infinity;
      for (let pos = lo; pos <= hi; pos += 1) {
        let score = 0;
        for (let index = 0; index < stretch.synthHop; index += 2) {
          score += stretch.prev[index] * this.inputSample(pos + index);
        }
        if (score > bestScore) {
          bestScore = score;
          best = pos;
        }
      }
    }
    const grain = Math.min(stretch.win, end - best);
    const emit = forceTail ? Math.max(grain, stretch.hasPrev ? stretch.synthHop : 0) : Math.min(stretch.synthHop, grain);
    for (let index = 0; index < emit; index += 1) {
      const raw = index < grain ? this.inputSample(best + index) / 32768 : 0;
      if (stretch.hasPrev && index < stretch.synthHop) {
        this.pushStretch(stretch.prev[index] + raw * stretch.window[index]);
      } else {
        this.pushStretch(raw);
      }
    }
    if (!forceTail && grain >= stretch.win) {
      for (let index = 0; index < stretch.synthHop; index += 1) {
        stretch.prev[index] = (this.inputSample(best + stretch.synthHop + index) / 32768)
          * stretch.window[stretch.synthHop + index];
      }
      stretch.hasPrev = true;
      // The search only picks a periodic join. The input cursor still advances
      // by a fixed analysis hop, or a backward match would stretch instead of shrink.
      stretch.cursor = nominal + stretch.anaHop;
      this.discardBefore(stretch.cursor - stretch.search);
      return;
    }
    stretch.done = true;
    this.discardBefore(end);
  }

  renderStretched(output) {
    const stretch = this.ensureStretch();
    while (stretch.fifoLen < output.length && !stretch.done) {
      const before = stretch.fifoLen;
      const end = this.inputEnd();
      const need = stretch.cursor + stretch.win + (stretch.hasPrev ? stretch.search : 0);
      this.synthesizeHop(Boolean(this.drain) && end < need);
      if (stretch.fifoLen === before) break;
    }
    const count = Math.min(output.length, stretch.fifoLen);
    for (let index = 0; index < count; index += 1) {
      let sample = stretch.fifo[index];
      if (this.fadeInFrames > 0) {
        sample *= (this.fadeFrames - this.fadeInFrames) / this.fadeFrames;
        this.fadeInFrames -= 1;
      }
      output[index] = sample;
    }
    if (count > 0) {
      stretch.fifo.copyWithin(0, count, stretch.fifoLen);
      stretch.fifoLen -= count;
      this.playedFrames += Math.round(count * this.playbackSpeed);
    }
    if (stretch.done && stretch.fifoLen === 0) this.discardBefore(this.inputEnd());
    return count;
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
        // The first codec packet is only 0.64s. Sped-up playback runs it out before the
        // next packet arrives and the cut is a click. Hold the start until a
        // second packet is queued. A finished utterance (drain) still starts
        // on the first packet; a mid-stream rebuffer does not wait again.
        if (
          this.holdUntilSecondChunk
          && !this.rebuffering
          && !this.drain
          && this.audioChunks < 2
        ) {
          return true;
        }
        this.startPlayback();
        this.rebuffering = false;
      }
    }
    if (!this.started) {
      this.notifyIfDrained();
      return true;
    }
    let target = 0;
    if (this.playbackSpeed > 1.01) {
      target = this.renderStretched(output);
    }
    while (this.playbackSpeed <= 1.01 && target < output.length && this.queue.length > 0) {
      const pcm = this.queue[0];
      const count = Math.min(output.length - target, pcm.length - this.offset);
      const remainingBeforeChunk = this.bufferedFrames();
      for (let index = 0; index < count; index += 1) {
        let sample = pcm[this.offset + index] / 32768;
        if (this.fadeInFrames > 0) {
          const elapsed = this.fadeFrames - this.fadeInFrames;
          sample *= elapsed / this.fadeFrames;
          this.fadeInFrames -= 1;
        }
        const remainingFrames = remainingBeforeChunk - index;
        if (
          this.drain
          && this.terminalFadeFrames > 0
          && remainingFrames <= this.terminalFadeFrames
        ) {
          sample *= this.terminalFadeFrames === 1
            ? 0
            : (remainingFrames - 1) / (this.terminalFadeFrames - 1);
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

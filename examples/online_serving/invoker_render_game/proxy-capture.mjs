// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

export class ProxyCapture {
  constructor(config) {
    this.config = config;
    this.frameBytes = config.width * config.height * 3;
    this.credit = config.credit_frames;
    this.acknowledged = 0;
    this.sent = 0;
    this.displayed = 0;
    this.sequence = 0;
    this.pending = [];
    this.finishing = false;
    this.ended = false;
  }

  acknowledge(message) {
    if (!Number.isSafeInteger(message.accepted_frames) || message.accepted_frames < this.acknowledged ||
        message.accepted_frames > this.sent || !Number.isSafeInteger(message.credit_frames) ||
        message.credit_frames < 0 || message.credit_frames > this.config.credit_frames) {
      throw new Error("Invalid input credit returned by server");
    }
    this.acknowledged = message.accepted_frames;
    this.credit = message.credit_frames;
  }

  played(mediaTime) {
    if (Number.isFinite(mediaTime) && mediaTime >= 0) {
      const position = Math.floor(mediaTime * this.config.fps + 0.001) + 1;
      this.displayed = Math.max(this.displayed, Math.min(position, this.sent));
    }
  }

  get canCapture() {
    const inFlight = this.sent + this.pending.length - this.displayed;
    const unacknowledged = this.sent - this.acknowledged + this.pending.length;
    return !this.ended && inFlight < this.config.credit_frames && unacknowledged < this.credit;
  }

  capture(rgb, time) {
    if (!this.canCapture) throw new Error("Proxy capture reached the input or playback credit limit");
    if (!(rgb instanceof Uint8Array) || rgb.length !== this.frameBytes || !Number.isFinite(time)) {
      throw new Error("Invalid proxy frame size or capture timestamp");
    }
    this.pending.push({rgb, time});
    return this.packet();
  }

  finish() {
    this.finishing = true;
    return this.packet();
  }

  packet() {
    const {packet_frames: packetFrames, tail_frames: tailFrames} = this.config;
    const tail = this.finishing && this.sent >= 2 * packetFrames && this.pending.length <= tailFrames;
    const count = tail ? tailFrames : packetFrames;
    if (this.ended || this.pending.length !== count) return null;
    const pixels = new Uint8Array(count * this.frameBytes);
    const captureMs = [];
    for (const [index, frame] of this.pending.entries()) {
      pixels.set(frame.rgb, index * this.frameBytes);
      captureMs.push(frame.time);
    }
    const header = {
      type: "session.proxy.append", input_seq: this.sequence, frame_start: this.sent,
      num_frames: count, eos: tail, capture_ms: captureMs,
    };
    this.sent += count;
    this.sequence += 1;
    this.pending = [];
    this.ended = tail;
    return {header, pixels};
  }
}

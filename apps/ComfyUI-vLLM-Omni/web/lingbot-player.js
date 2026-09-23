// Adapted from vLLM-Omni's Apache-2.0 streaming_video_generation/video-stream-view.js.
// Copyright contributors to the vLLM-Omni project.
export function cameraEvent(keys, eventId, { fps = 16, speed = 1 } = {}) {
  if (!Number.isFinite(fps) || fps <= 0 || !Number.isFinite(speed) || speed < 0) {
    throw new Error("Camera FPS must be positive and speed must be non-negative.");
  }
  // The current camera API integrates once per latent frame. LingBot
  // expands each steady-state latent into four output frames.
  const controlFps = fps / 4;
  const step = 0.2 * speed / controlFps;
  const dx = Number(keys.has("d")) - Number(keys.has("a"));
  const dz = Number(keys.has("w")) - Number(keys.has("s"));
  const length = Math.max(1, Math.hypot(dx, dz));
  const pitch = 16 * speed / controlFps * (Number(keys.has("i")) - Number(keys.has("k"))) * Math.PI / 360;
  const yaw = 24 * speed / controlFps * (Number(keys.has("l")) - Number(keys.has("j"))) * Math.PI / 360;
  return {
    type: "session.interaction",
    interaction: {
      event_id: eventId,
      event: { multi_modal_data: { camera: {
        mode: "velocity",
        data: {
          translation: [step * dx / length, 0, step * dz / length],
          rotation: [Math.cos(yaw) * Math.sin(pitch), Math.sin(yaw) * Math.cos(pitch),
            -Math.sin(yaw) * Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)],
        },
      } } },
    },
  };
}

export class VideoSession {
  constructor(video, report) {
    this.video = video;
    this.report = report;
    this.ws = null;
    this.source = null;
    this.buffer = null;
    this.url = null;
    this.queue = [];
    this.queuedBytes = 0;
    this.done = false;
    this.stopping = false;
    this.ready = false;
    this.stopTimer = null;
    this.startTimer = null;
    this.pingTimer = null;
    this.playbackTimer = null;
    this.lastPlaybackPosition = -1;
    this.lastProgress = 0;
    video.addEventListener("error", () => {
      if (this.source) this.fail(video.error?.message || "Video loading failed; check the page's media security policy.");
    });
    video.addEventListener("ended", () => {
      if (this.source && this.done && !this.stopping) this.report("Playback complete.");
    });
  }

  start(config) {
    this.dispose();
    if (location.protocol === "https:" && !config.url.startsWith("wss://")) {
      throw new Error("An HTTPS ComfyUI page requires a wss:// Omni endpoint.");
    }
    const mime = ['video/mp4; codecs="avc1.42E01E"', 'video/mp4; codecs="avc1.4D401F"',
      'video/mp4; codecs="avc1.64001F"'].find(m => window.MediaSource?.isTypeSupported(m));
    if (!mime) throw new Error("This browser does not support H.264 Media Source Extensions.");
    this.done = false;
    this.stopping = false;
    this.ready = false;
    this.lastProgress = Date.now();
    const source = this.source = new MediaSource();
    this.url = URL.createObjectURL(source);
    this.video.src = this.url;
    this.video.play().catch(error => {
      if (this.source === source && error.name === "NotAllowedError") this.report("Click the video play button to allow playback.");
    });
    this.startTimer = setTimeout(() => this.fail("Timed out opening the video connection."), 15000);
    source.addEventListener("sourceopen", () => {
      if (this.source !== source) return;
      try {
        this.buffer = source.addSourceBuffer(mime);
        this.buffer.addEventListener("updateend", () => {
          if (this.source === source) this.pump();
        });
        this.buffer.addEventListener("error", () => {
          if (this.source === source) this.fail("The browser could not decode the video stream.");
        });
        const ws = this.ws = new WebSocket(config.url);
        ws.binaryType = "arraybuffer";
        ws.onopen = () => {
          if (this.ws !== ws) return;
          clearTimeout(this.startTimer);
          this.send(config.keyboard
            ? { ...config.payload, streaming_buffer_seconds: config.payload.streaming_buffer_seconds ?? 1.25 }
            : config.payload);
          this.pingTimer = setInterval(() => this.send({ type: "session.ping" }), 20000);
          if (config.keyboard) this.playbackTimer = setInterval(() => this.sendPlayback(), 100);
          this.report("Generating the first chunk…");
        };
        ws.onmessage = ({ data }) => {
          if (this.ws !== ws) return;
          this.lastProgress = Date.now();
          if (typeof data === "string") {
            try { this.control(JSON.parse(data)); }
            catch (error) { this.fail(error.message); }
          } else if (!this.done && !this.stopping) {
            this.queue.push(data);
            this.queuedBytes += data.byteLength;
            // Guard bytes that have not yet reached MSE, independently of playback feedback.
            if (this.queuedBytes > 32 * 1024 * 1024) this.fail("Video backlog exceeded 32 MiB; restart with fewer frames.");
            else this.pump();
          }
        };
        ws.onerror = () => { if (this.ws === ws) this.fail("WebSocket failed; check the Omni address and server log."); };
        ws.onclose = () => {
          if (this.ws !== ws) return;
          this.ready = false;
          if (!this.done) this.fail(this.stopping ? "Stopped (connection closed)." : "Connection lost; start a new session.");
        };
      } catch (error) { this.fail(error.message); }
    }, { once: true });
  }

  control(message) {
    if (message.type === "error") {
      this.fail(`${message.message || "Server error"}. Keyboard mode requires Omni camera interaction support.`);
    } else if (message.type === "video.chunk_metadata") {
      if (message.kind === "media") {
        this.ready = !this.stopping;
        const applied = message.started_event_ids || [];
        this.report(`Chunk ${message.generation_chunk_index ?? "?"} received` +
          (applied.length ? `; applied: ${applied.join(", ")}` : ""));
      }
    } else if (message.type === "session.interaction.queued") {
      this.report(`Queued: ${message.event_id}`);
    } else if (message.type === "session.done") {
      clearTimeout(this.stopTimer);
      clearInterval(this.pingTimer);
      clearInterval(this.playbackTimer);
      this.done = true;
      this.ready = false;
      this.report(message.stopped ? "Stopped." : "Generation complete; playing remaining video.");
      this.pump();
      this.ws.close();
    }
  }

  sendPlayback() {
    const position = this.video.currentTime;
    if (!this.ready || this.done || this.stopping || !Number.isFinite(position) || position < 0 || position <= this.lastPlaybackPosition) return;
    if (this.send({ type: "session.playback", position_seconds: position })) this.lastPlaybackPosition = position;
  }

  send(message) {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(message));
    return true;
  }

  pump() {
    if (!this.buffer || this.buffer.updating || this.source?.readyState !== "open") return;
    try {
      // Let MSE manage coded-frame eviction. Arbitrary time-based removal
      // can discard the only keyframe and all later dependent frames.
      if (this.queue.length) {
        const data = this.queue.shift();
        this.queuedBytes -= data.byteLength;
        this.buffer.appendBuffer(data);
      } else if (this.done) this.source.endOfStream();
    } catch (error) { this.fail(error.message); }
  }

  stop() {
    this.ready = false;
    if (this.stopping || this.done) return;
    this.stopping = true;
    if (this.send({ type: "session.stop" })) {
      this.report("Stopping…");
      this.stopTimer = setTimeout(() => this.fail("Stop timed out; connection closed."), 5000);
    } else this.fail("Stopped.");
  }

  fail(message) {
    this.dispose();
    this.done = true;
    this.report(message);
  }

  dispose() {
    clearTimeout(this.startTimer);
    clearTimeout(this.stopTimer);
    clearInterval(this.pingTimer);
    clearInterval(this.playbackTimer);
    this.lastPlaybackPosition = -1;
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      if (ws.readyState === WebSocket.OPEN && !this.done) ws.send(JSON.stringify({ type: "session.stop" }));
      ws.close();
    }
    this.ready = false;
    this.source = null;
    this.buffer = null;
    this.queue = [];
    this.queuedBytes = 0;
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = null;
  }
}

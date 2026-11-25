/**
 * This file and the corresponding HTML file are needed to display a streamable video view in the Gradio demo.
 * Gradio's built-in video player does not support modern M4S streaming format.
 */
(function () {
  const playerId = "vllm-streaming-video-player";
  const statusId = "vllm-streaming-video-status";
  const finishAudioId = "vllm-streaming-video-finish-audio";
  const cancelId = "vllm-streaming-video-cancel";
  const logId = "vllm-streaming-video-log";
  let ws = null;
  let mediaSource = null;
  let sourceBuffer = null;
  let queue = [];
  let chunkCount = 0;
  let totalBytes = 0;
  let done = false;
  let objectUrl = null;
  let promptUpdateTimers = [];
  let capture = null;
  const pcmRate = 24000;

  const workletSource = `
    class StreamingVideoCapture extends AudioWorkletProcessor {
      constructor() {
        super();
        this.pcm = new Uint8Array(sampleRate * 2);
        this.view = new DataView(this.pcm.buffer);
        this.offset = 0;
        this.finished = false;
        this.port.onmessage = ({data}) => {
          if (data !== "finish" || this.finished) return;
          this.finished = true;
          this.flush();
          this.port.postMessage("done");
        };
      }

      flush() {
        if (!this.offset) return;
        const packet = this.pcm.slice(0, this.offset);
        this.port.postMessage(packet.buffer, [packet.buffer]);
        this.offset = 0;
      }

      process(inputs) {
        if (this.finished) return false;
        for (const sample of inputs[0]?.[0] || []) {
          const scaled = Math.max(-1, Math.min(1, sample));
          this.view.setInt16(this.offset, scaled * (scaled < 0 ? 32768 : 32767), true);
          this.offset += 2;
          if (this.offset === this.pcm.length) this.flush();
        }
        return true;
      }
    }
    registerProcessor("streaming-video-capture", StreamingVideoCapture);
  `;

  function closeCapture() {
    const current = capture;
    capture = null;
    if (current) {
      current.media?.getTracks().forEach(track => track.stop());
      if (current.context) void current.context.close().catch(() => {});
    }
    const finish = el(finishAudioId);
    if (finish) finish.disabled = true;
  }

  async function openCapture(socket) {
    const current = {};
    capture = current;
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
        throw new Error("Microphone capture requires localhost or HTTPS and Web Audio support.");
      }
      current.media = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
      if (capture !== current || socket !== ws) {
        current.media.getTracks().forEach(track => track.stop());
        return;
      }
      current.context = new AudioContext({ sampleRate: pcmRate });
      if (current.context.sampleRate !== pcmRate || !current.context.audioWorklet) {
        throw new Error("This browser cannot capture 24 kHz PCM with AudioWorklet.");
      }
      const moduleUrl = URL.createObjectURL(new Blob([workletSource], { type: "text/javascript" }));
      try {
        await current.context.audioWorklet.addModule(moduleUrl);
      } finally {
        URL.revokeObjectURL(moduleUrl);
      }
      if (capture !== current || socket !== ws) return;
      current.node = new AudioWorkletNode(current.context, "streaming-video-capture");
      current.node.port.onmessage = ({ data }) => {
        if (capture !== current || socket !== ws || socket.readyState !== WebSocket.OPEN) return;
        if (data === "done") {
          socket.send(JSON.stringify({ type: "input_audio_buffer.done" }));
          closeCapture();
          setStatus("Finishing video...");
          return;
        }
        const bytes = new Uint8Array(data);
        let binary = "";
        for (let offset = 0; offset < bytes.length; offset += 8192) {
          binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
        }
        socket.send(JSON.stringify({ type: "input_audio_buffer.append", audio: btoa(binary) }));
      };
      const source = current.context.createMediaStreamSource(current.media);
      const silent = current.context.createGain();
      silent.gain.value = 0;
      source.connect(current.node).connect(silent).connect(current.context.destination);
      await current.context.resume();
      if (capture === current && socket === ws) {
        const finish = el(finishAudioId);
        if (finish) finish.disabled = false;
        setStatus("Recording...");
      }
    } catch (err) {
      if (capture !== current || socket !== ws) return;
      stopCurrent();
      done = true;
      setStatus("Microphone error");
      setStartButton(false, "Restart");
      log("Microphone error: " + err.message);
    }
  }

  function el(id) {
    return document.getElementById(id);
  }

  function setStatus(text) {
    const node = el(statusId);
    if (node) node.textContent = text;
  }

  function log(line) {
    const node = el(logId);
    if (!node) return;
    node.textContent += "\n" + line;
    node.scrollTop = node.scrollHeight;
  }

  function setStartButton(disabled, label) {
    const btn = document.querySelector("#streaming-video-start button") || document.getElementById("streaming-video-start");
    if (!btn) return;
    btn.disabled = disabled;
    btn.textContent = label;
  }

  function chooseMime(withAudio) {
    const audioCodec = withAudio ? ", mp4a.40.2" : "";
    const candidates = ["avc1.42E01E", "avc1.4D401F", "avc1.64001F"]
      .map(codec => `video/mp4; codecs="${codec}${audioCodec}"`);
    candidates.push("video/mp4");
    for (const mime of candidates) {
      if (window.MediaSource && MediaSource.isTypeSupported(mime)) return mime;
    }
    return null;
  }

  function pump() {
    if (!sourceBuffer || sourceBuffer.updating || queue.length === 0) return;
    try {
      sourceBuffer.appendBuffer(queue.shift());
    } catch (err) {
      done = true;
      setStatus("SourceBuffer error");
      setStartButton(false, "Restart");
      log("SourceBuffer append error: " + err.message);
      finishStream();
    }
  }

  function finishStream() {
    if (!mediaSource || mediaSource.readyState !== "open") return;
    if (sourceBuffer && sourceBuffer.updating) return;
    if (queue.length > 0) return;
    try {
      mediaSource.endOfStream();
    } catch (_) { }
  }

  function clearPromptUpdateTimers() {
    for (const timer of promptUpdateTimers) {
      clearTimeout(timer);
    }
    promptUpdateTimers = [];
  }

  function schedulePromptUpdates(updates) {
    clearPromptUpdateTimers();
    if (!Array.isArray(updates) || updates.length === 0) return;

    const videoStartedAt = performance.now();
    for (const update of updates) {
      const atSeconds = Number(update.at);
      if (!Number.isFinite(atSeconds) || atSeconds < 0) continue;

      const delayMs = Math.max(0, atSeconds * 1000 - (performance.now() - videoStartedAt));
      const transitionChunks = Number.isFinite(Number(update.transition_chunks))
        ? Number(update.transition_chunks)
        : 3;

      const timer = setTimeout(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const payload = {
          type: "session.interaction",
          interaction: {
            event_id: "scheduled-" + atSeconds.toFixed(3),
            event: { prompt: update.prompt },
            transition_chunks: transitionChunks,
          },
        };
        try {
          ws.send(JSON.stringify(payload));
          log("Sent session.interaction at t=" + atSeconds.toFixed(2) + "s: " + JSON.stringify(payload));
        } catch (err) {
          log("Failed to send session.interaction: " + err.message);
        }
      }, delayMs);
      promptUpdateTimers.push(timer);
    }
  }

  function stopCurrent() {
    clearPromptUpdateTimers();
    closeCapture();
    const cancel = el(cancelId);
    if (cancel) cancel.hidden = true;
    const finish = el(finishAudioId);
    if (finish) finish.hidden = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ type: "session.stop" })); } catch (_) { }
    }
    if (ws && ws.readyState < WebSocket.CLOSING) ws.close();
    ws = null;
    const video = el(playerId);
    if (video && objectUrl) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }
    sourceBuffer = null;
    mediaSource = null;
    queue = [];
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }

  window.vllmStreamingVideoStart = function (configJson) {
    stopCurrent();
    chunkCount = 0;
    totalBytes = 0;
    done = false;
    const logNode = el(logId);
    if (logNode) logNode.textContent = "Starting...";
    setStatus("Starting...");

    let config;
    try {
      config = JSON.parse(configJson);
    } catch (err) {
      setStatus("Invalid request");
      setStartButton(false, "Restart");
      log("Invalid request config: " + err.message);
      return configJson;
    }

    const liveAudio = config.payload.extra_params?.live_audio === true;
    const mime = chooseMime(liveAudio || Boolean(config.payload.audio_reference));
    if (!mime) {
      setStatus("MSE unsupported");
      setStartButton(false, "Restart");
      log("This browser does not support MP4 Media Source Extensions playback.");
      return configJson;
    }

    const video = el(playerId);
    if (!video) return configJson;
    mediaSource = new MediaSource();
    objectUrl = URL.createObjectURL(mediaSource);
    video.src = objectUrl;

    const activeMediaSource = mediaSource;
    const finish = el(finishAudioId);
    if (finish) finish.hidden = !liveAudio;
    const cancel = el(cancelId);
    if (cancel) cancel.hidden = false;

    mediaSource.addEventListener("sourceopen", () => {
      if (mediaSource !== activeMediaSource) return;
      try {
        sourceBuffer = mediaSource.addSourceBuffer(mime);
        sourceBuffer.mode = "segments";
        sourceBuffer.addEventListener("updateend", () => {
          pump();
          if (done) finishStream();
        });
      } catch (err) {
        setStatus("SourceBuffer error");
        setStartButton(false, "Restart");
        log("SourceBuffer error: " + err.message);
        return;
      }

      const socket = new WebSocket(config.url);
      ws = socket;
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        if (ws !== socket) return;
        setStatus("Streaming...");
        setStartButton(true, "Streaming...");
        log("Connected: " + config.url);
        ws.send(JSON.stringify(config.payload));
        log("Sent session.start fields: " + Object.keys(config.payload).join(", "));
      };
      ws.onmessage = (event) => {
        if (ws !== socket) return;
        if (typeof event.data === "string") {
          const msg = JSON.parse(event.data);
          if (msg.type === "video.start") {
            log("Video session started: request_id=" + (msg.request_id || "") + " format=" + (msg.format || ""));
            if (liveAudio) void openCapture(socket);
            else schedulePromptUpdates(config.prompt_updates);
          } else if (msg.type === "video.chunk_metadata") {
            log("Chunk metadata: " + JSON.stringify(msg));
          } else if (msg.type === "session.interaction.queued") {
            log("Interaction queued by server: event_id=" + (msg.event_id || ""));
          } else if (msg.type === "session.done") {
            closeCapture();
            if (cancel) cancel.hidden = true;
            done = true;
            setStatus("Done");
            setStartButton(false, "Restart");
            log("Session complete: " + JSON.stringify(msg));
            finishStream();
          } else if (msg.type === "error") {
            closeCapture();
            if (cancel) cancel.hidden = true;
            done = true;
            setStatus("Error");
            setStartButton(false, "Restart");
            log("ERROR: " + (msg.message || JSON.stringify(msg)));
            finishStream();
          } else {
            log("Control message: " + JSON.stringify(msg));
          }
          return;
        }

        const data = event.data;
        queue.push(data);
        chunkCount += 1;
        totalBytes += data.byteLength;
        log(`[chunk ${String(chunkCount).padStart(3, "0")}] bytes=${data.byteLength} total_bytes=${totalBytes}`);
        pump();
      };
      ws.onerror = () => {
        if (ws !== socket) return;
        closeCapture();
        if (cancel) cancel.hidden = true;
        done = true;
        setStatus("Error");
        setStartButton(false, "Restart");
        log("ERROR: WebSocket error");
        finishStream();
      };
      ws.onclose = () => {
        if (ws !== socket) return;
        closeCapture();
        if (cancel) cancel.hidden = true;
        if (!done) {
          done = true;
          setStatus("Closed");
          setStartButton(false, "Restart");
          log("WebSocket closed before session.done");
          finishStream();
        }
      };
    }, { once: true });

    return configJson;
  };

  el(finishAudioId)?.addEventListener("click", () => {
    if (!capture?.node) return;
    el(finishAudioId).disabled = true;
    capture.node.port.postMessage("finish");
  });
  el(cancelId)?.addEventListener("click", () => {
    stopCurrent();
    done = true;
    setStatus("Cancelled");
    setStartButton(false, "Restart");
  });

  console.log("video-stream-view.html loaded");
})();

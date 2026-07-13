/**
 * This file and the corresponding HTML file are needed to display a streamable video view in the Gradio demo.
 * Gradio's built-in video player does not support modern M4S streaming format.
 */
(function () {
  const playerId = "vllm-streaming-video-player";
  const statusId = "vllm-streaming-video-status";
  const logId = "vllm-streaming-video-log";
  let ws = null;
  let mediaSource = null;
  let sourceBuffer = null;
  let queue = [];
  let chunkCount = 0;
  let totalBytes = 0;
  let done = false;
  let objectUrl = null;

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

  function chooseMime() {
    const candidates = [
      'video/mp4; codecs="avc1.42E01E"',
      'video/mp4; codecs="avc1.4D401F"',
      'video/mp4; codecs="avc1.64001F"',
      "video/mp4"
    ];
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

  function stopCurrent() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ type: "session.stop" })); } catch (_) { }
      try { ws.close(); } catch (_) { }
    }
    ws = null;
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

    const mime = chooseMime();
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

    mediaSource.addEventListener("sourceopen", () => {
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

      ws = new WebSocket(config.url);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        setStatus("Streaming...");
        setStartButton(true, "Streaming...");
        log("Connected: " + config.url);
        ws.send(JSON.stringify(config.payload));
        log("Sent session.start: " + JSON.stringify(config.payload));
      };
      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          const msg = JSON.parse(event.data);
          if (msg.type === "video.start") {
            log("Video session started: request_id=" + (msg.request_id || "") + " format=" + (msg.format || ""));
          } else if (msg.type === "session.done") {
            done = true;
            setStatus("Done");
            setStartButton(false, "Restart");
            log("Session complete: " + JSON.stringify(msg));
            finishStream();
          } else if (msg.type === "error") {
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
        done = true;
        setStatus("Error");
        setStartButton(false, "Restart");
        log("ERROR: WebSocket error");
        finishStream();
      };
      ws.onclose = () => {
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

  console.log("video-stream-view.html loaded");
})();

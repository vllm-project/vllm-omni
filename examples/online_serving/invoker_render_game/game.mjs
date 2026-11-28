// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import {ProxyCapture} from "./proxy-capture.mjs";

const el = id => document.getElementById(id);
const proxyCanvas = el("proxy");
const video = el("vllm-streaming-video-player");
const renderWaiting = el("render-waiting");
const renderFormat = el("render-format");
const expectedFormat = renderFormat.textContent;
const sentFrames = el("sent-frames");
const playedFrames = el("played-frames");
let game;
let capture;
let socket;
let previousTick;
let nextCapture = 0;
let inSession = false;

function status(message, state = "idle") {
  el("status").textContent = message;
  el("session-status").dataset.state = state;
}

function close(message, state = "error") {
  inSession = false;
  if (!capture?.ended) capture = undefined;
  socket?.close();
  socket = undefined;
  game?.setControlsSuspended(false);
  el("start").disabled = !game;
  el("finish").disabled = true;
  el("cancel").disabled = true;
  if (!renderWaiting.hidden) renderWaiting.textContent = "Generated video will play here once a session starts";
  status(message, state);
}

function send(packet) {
  if (!packet) return;
  socket.send(JSON.stringify(packet.header));
  socket.send(packet.pixels);
  sentFrames.textContent = `${capture.sent} frames`;
  if (packet.header.eos) status("Draining generated video", "active");
}

function control(message, connection) {
  try {
    if (message.type === "video.start") {
      if (!message.proxy) throw new Error("Start the server with the Invoker-Render streaming config");
      const proxy = message.proxy;
      if (proxy.width !== game.width || proxy.height !== game.height) {
        throw new Error("This racing demo requires a 448×256 proxy canvas");
      }
      socket = connection;
      capture = new ProxyCapture(proxy);
      game.prepareRun();
      game.setControlsSuspended(false);
      nextCapture = performance.now();
      el("finish").disabled = false;
      el("cancel").disabled = false;
      const bounds = proxyCanvas.getBoundingClientRect();
      proxyCanvas.focus({preventScroll: bounds.top >= 0 && bounds.bottom <= window.innerHeight});
      status("Capturing proxy frames", "active");
    } else if (message.type === "session.proxy.credit") {
      capture?.acknowledge(message);
    } else if (message.type === "session.done") {
      close(message.stopped ? "Cancelled" : "Completed", "idle");
    } else if (message.type === "error") {
      close(message.message);
    }
  } catch (error) {
    close(error.message);
  }
}

function tick(now) {
  requestAnimationFrame(tick);
  const delta = Math.min((now - (previousTick ?? now)) / 1000, 0.1);
  previousTick = now;
  if (!game || (inSession && (!capture || !capture.canCapture))) return;
  try {
    game.render(delta);
    if (!inSession || !capture || now < nextCapture) return;
    const interval = 1000 / capture.config.fps;
    nextCapture += (Math.floor((now - nextCapture) / interval) + 1) * interval;
    send(capture.capture(game.capture(), performance.now()));
  } catch (error) {
    close(error.message);
  }
}

function fileUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

el("inputs").addEventListener("submit", async event => {
  event.preventDefault();
  if (!game || inSession) return;
  inSession = true;
  capture = undefined;
  el("start").disabled = true;
  sentFrames.textContent = playedFrames.textContent = "0 frames";
  renderFormat.textContent = expectedFormat;
  renderWaiting.textContent = "Waiting for first frame";
  renderWaiting.hidden = false;
  status("Connecting to render server", "active");
  try {
    const image = await fileUrl(el("reference").files[0]);
    const url = new URL(el("endpoint").value);
    if (url.protocol !== "ws:" && url.protocol !== "wss:") throw new Error("Enter a valid ws:// or wss:// URL");
    window.vllmStreamingVideoStart(JSON.stringify({
      url: url.href,
      retain_seconds: 8,
      payload: {
        type: "session.start", prompt: el("prompt").value.trim(), seed: 42,
        image_reference: {image_url: image}, format: "m4s",
      },
    }), {
      open: connection => {
        socket = connection;
        el("cancel").disabled = false;
      },
      control,
      error: message => { if (inSession) close(message); },
    });
  } catch (error) {
    close(error.message);
  }
});

el("finish").addEventListener("click", () => {
  el("finish").disabled = true;
  if (capture) {
    status("Capturing tail frames", "active");
    send(capture.finish());
  }
});
el("cancel").addEventListener("click", () => {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type: "session.stop"}));
  el("cancel").disabled = true;
});

function recordPlayback(mediaTime) {
  if (!capture) return;
  const before = capture.displayed;
  capture.played(mediaTime);
  if (capture.displayed !== before) playedFrames.textContent = `${capture.displayed} frames`;
}

video.addEventListener("loadeddata", () => {
  if (!capture) return;
  renderFormat.textContent = `${video.videoWidth} × ${video.videoHeight} · ${capture.config.fps} FPS`;
  renderWaiting.hidden = true;
});

if (video.requestVideoFrameCallback) {
  const onFrame = (_, metadata) => {
    recordPlayback(metadata.mediaTime);
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
} else {
  video.addEventListener("timeupdate", () => recordPlayback(video.currentTime));
}

el("start").disabled = true;
try {
  const {createGame} = await import("./codeworldmodel/game.js");
  game = createGame(proxyCanvas, {width: 448, height: 256, scene: "racing", canReset: () => !inSession});
  el("start").disabled = false;
  status("Select a reference image matching the opening camera view to start");
  requestAnimationFrame(tick);
} catch (error) {
  close(`Failed to load racing scene: ${error.message}`);
}

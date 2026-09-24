import { app } from "../../scripts/app.js";
import { VideoSession, cameraEvent } from "./lingbot-player.js";

let closePanel = null;

function openPanel(node) {
  closePanel?.();
  const dialog = document.createElement("dialog");
  dialog.setAttribute("aria-label", "Omni LingBot World");
  dialog.style.cssText = "width:min(960px,92vw);max-height:92vh;overflow:auto;background:#171b24;color:#eee;border:1px solid #555;border-radius:12px;padding:20px";
  // Static markup only; prompts and server messages are assigned with textContent.
  dialog.innerHTML = `
    <h2>Omni LingBot World</h2>
    <p data-info></p>
    <video controls autoplay muted playsinline style="width:100%;max-height:55vh;background:black" aria-label="Generated world"></video>
    <div data-controls tabindex="0" style="padding:12px;border:1px solid #777;border-radius:6px;margin:10px 0;outline-offset:3px" role="group" aria-label="Keyboard camera controls">
      Click here to control: W/A/S/D move · I/J/K/L look. Release keys to stop moving.
    </div>
    <label>Camera speed <input data-speed type="range" min="0.1" max="2" step="0.1" value="0.5" aria-label="Camera speed">
      <output data-speed-value>0.5×</output></label>
    <p data-keys>Keys: none</p>
    <p data-status role="status" aria-live="polite">Run the workflow to prepare the image and settings.</p>
    <p data-buffer>Buffered: 0.00 s</p>
    <div style="display:flex;gap:10px">
      <button data-start>Start / Restart</button><button data-stop>Stop</button><button data-close>Close</button>
    </div>`;
  document.body.append(dialog);
  const status = dialog.querySelector("[data-status]");
  const video = dialog.querySelector("video");
  const controls = dialog.querySelector("[data-controls]");
  const start = dialog.querySelector("[data-start]");
  const stop = dialog.querySelector("[data-stop]");
  const speed = dialog.querySelector("[data-speed]");
  const keys = new Set();
  let config = null;
  let eventIndex = 0;
  const session = new VideoSession(video, message => { status.textContent = message; });
  const listeners = new AbortController();
  const options = { signal: listeners.signal };
  const showKeys = () => { dialog.querySelector("[data-keys]").textContent = `Keys: ${[...keys].join(" ").toUpperCase() || "none"}`; };
  const sendKeys = () => {
    if (config?.keyboard && session.ready) session.send(cameraEvent(keys, `camera-${++eventIndex}`, { fps: config.payload.fps, speed: Number(speed.value) }));
    showKeys();
  };
  speed.addEventListener("input", () => {
    dialog.querySelector("[data-speed-value]").textContent = `${speed.value}×`;
    if (keys.size) sendKeys();
  }, options);
  const release = () => {
    if (!keys.size) return;
    keys.clear();
    sendKeys();
  };
  controls.addEventListener("keydown", event => {
    const key = event.key.toLowerCase();
    if (!"wasdijkl".includes(key) || key.length !== 1 || event.ctrlKey || event.metaKey || event.altKey) return;
    event.preventDefault();
    event.stopPropagation();
    if (!config?.keyboard || !session.ready || keys.has(key)) return;
    keys.add(key);
    sendKeys();
  }, options);
  window.addEventListener("keyup", event => {
    if (keys.delete(event.key.toLowerCase())) sendKeys();
  }, options);
  controls.addEventListener("blur", release, options);
  window.addEventListener("blur", release, options);
  document.addEventListener("visibilitychange", () => { if (document.hidden) release(); }, options);
  start.addEventListener("click", () => {
    const prepared = app.nodeOutputs[String(node.id)]?.omni_lingbot?.[0];
    if (!prepared) return;
    release();
    config = { ...prepared };
    eventIndex = 0;
    status.textContent = "Connecting…";
    try {
      session.start(config);
      controls.focus();
    } catch (error) { session.fail(error.message); }
  }, options);
  stop.addEventListener("click", () => { release(); session.stop(); }, options);
  const close = () => {
    release();
    session.dispose();
    clearInterval(timer);
    listeners.abort();
    dialog.remove();
    if (closePanel === close) closePanel = null;
  };
  closePanel = close;
  dialog.querySelector("[data-close]").addEventListener("click", close, options);
  dialog.addEventListener("cancel", event => { event.preventDefault(); close(); }, options);
  window.addEventListener("pagehide", close, options);
  const update = () => {
    if (!node.graph || node.graph.getNodeById(node.id) !== node) { close(); return; }
    const prepared = app.nodeOutputs[String(node.id)]?.omni_lingbot?.[0];
    start.disabled = !prepared || (session.ws && !session.done);
    stop.disabled = !session.ws || session.done || session.stopping;
    dialog.querySelector("[data-info]").textContent = prepared
      ? `${prepared.payload.width}×${prepared.payload.height} · ${prepared.payload.fps} FPS · ${prepared.payload.num_frames} frames · ${prepared.keyboard ? "Keyboard (requires camera API)" : "Preset trajectory"}`
      : "Run the workflow first. No model runs inside ComfyUI.";
    const ahead = video.buffered.length ? Math.max(0, video.buffered.end(video.buffered.length - 1) - video.currentTime) : 0;
    dialog.querySelector("[data-buffer]").textContent = `Buffered: ${ahead.toFixed(2)} s` +
      (ahead > 2 ? " — controls will be delayed by video already generated." : "");
    if (session.ws && !session.done && Date.now() - session.lastProgress > 90000) session.fail("No video progress for 90 seconds; restart the session.");
    if (!session.ready && keys.size) { keys.clear(); showKeys(); }
  };
  // Also detects node removal without replacing ComfyUI's node lifecycle methods.
  const timer = setInterval(update, 250);
  dialog.showModal();
  update();
}

app.registerExtension({
  name: "Omni.LingBotWorld",
  nodeCreated(node) {
    if (!["VLLMOmniLingBotWorld", "VLLMOmniLingBotRealtimeJSON"].includes(node.comfyClass)) return;
    node.addWidget("button", "Open interaction panel", null, () => openPanel(node), { serialize: false });
  },
  beforeConfigureGraph() {
    closePanel?.();
  },
});

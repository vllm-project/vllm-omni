// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

/**
 * @file Interaction panel for `VLLMOmniLingBotWorld` and
 * `VLLMOmniLingBotRealtimeJSON`.
 *
 * The Python node prepares a finite session and publishes
 * `app.nodeOutputs[String(node.id)].omni_lingbot[0]` as
 * `{url, payload, keyboard}`. This panel opens from the node's "Open
 * interaction panel" button and owns the live WebSocket through
 * `VideoSession` in `lingbot-player.js`.
 *
 * The layout borrows the visual language of ComfyUI-Reactor's realtime window
 * (Apache-2.0): a full-screen backdrop, a centred dark card, a 16:9 preview
 * with a spinner and an error overlay, lit physical keycaps, a prompt
 * composer, a status line and a tabular-nums stats line. The code, markup and
 * `lingbot-live-*` class names are ours; nothing was copied from that file.
 *
 * The pure helpers below (`hudCaps`, `statusLine`, `statsText`, `saveName`,
 * `eventId`) touch no DOM, so the tests import this module from a `data:` URL.
 * For that reason the module has no static imports: `lingbot-player.js` is
 * loaded with a dynamic `import()` only when a panel opens, and registration
 * is guarded on `window`/`document` exactly like `lingbot-timeline.js`.
 */

/** Injected style sheet id, so opening the panel twice never adds it twice. */
const STYLE_ID = "lingbot-live-style";
/** How many words of the prompt become the download's file name. */
const MAX_NAME_WORDS = 4;
/** A session with no bytes or metadata for this long is declared stalled. */
const STALL_MS = 90000;
/** Longer than `VideoSession.stop()`'s own 5 s timeout, so Save downloads the settled take. */
const SAVE_STOP_TIMEOUT_MS = 6000;
/** Physical key codes, so the legend matches the caps whatever the keyboard layout. */
const CODE_TO_KEY = Object.freeze({
  KeyW: "w", KeyA: "a", KeyS: "s", KeyD: "d",
  KeyI: "i", KeyJ: "j", KeyK: "k", KeyL: "l",
});
/** [key, grid column, grid row] — W above A/S/D. */
const MOVE_LAYOUT = [["w", 2, 1], ["a", 1, 2], ["s", 2, 2], ["d", 3, 2]];
/** I/J/K/L are the look keys, drawn as an arrow pad with the letter in the corner. */
const LOOK_LAYOUT = [["i", 2, 1, "↑"], ["j", 1, 2, "←"], ["k", 2, 2, "↓"], ["l", 3, 2, "→"]];

/** One short sentence per main state; errors carry their own message. */
const STATUS = Object.freeze({
  idle: "Run the workflow to prepare the image and settings.",
  ready: "Ready. Press Start to open the session.",
  connecting: "Connecting…",
  waiting: "Generating the first chunk…",
  playing: "Playing",
  paused: "Paused",
  stopping: "Stopping…",
  stopped: "Stopped",
  complete: "Generation complete; playing remaining video.",
  saving: "Preparing the download…",
  error: "Something went wrong.",
});

/**
 * The two 3-column keycap pads, with the held keys already lit. Pure so the
 * test can check that a key set lights exactly the matching caps.
 *
 * @param {Set<string>|string[]} [held]
 * @returns {{name: string, caps: {key: string, label: string, glyph: string|null,
 *   column: number, row: number, on: boolean}[]}[]}
 */
export function hudCaps(held = new Set()) {
  const has = (key) => (typeof held?.has === "function" ? held.has(key)
    : Array.isArray(held) ? held.includes(key) : false);
  const pad = (name, layout) => ({
    name,
    caps: layout.map(([key, column, row, glyph = null]) => ({
      key, label: key.toUpperCase(), glyph, column, row, on: has(key),
    })),
  });
  return [pad("move", MOVE_LAYOUT), pad("look", LOOK_LAYOUT)];
}

/**
 * The status line as text plus whether it should read as an error (red).
 *
 * @param {string} state one of the STATUS keys
 * @param {string} [detail] a server/player sentence that replaces the state text
 * @returns {{text: string, error: boolean}}
 */
export function statusLine(state, detail = "") {
  if (state === "error") {
    return { text: detail || STATUS.error, error: true };
  }
  return { text: detail || STATUS[state] || "", error: false };
}

/** The status line's text alone. */
export function statusText(state, detail = "") {
  return statusLine(state, detail).text;
}

/**
 * Reactor-style stats line: what the session is doing, joined by ` · `.
 *
 * @param {object} stats
 * @returns {string}
 */
export function statsText({
  width = null,
  height = null,
  fps = null,
  chunkIndex = null,
  generatedSeconds = null,
  playedSeconds = null,
  bufferAhead = null,
  sessionId = null,
} = {}) {
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const parts = [];
  if (finite(width) && finite(height)) parts.push(`${Math.round(width)}×${Math.round(height)}`);
  if (finite(fps)) parts.push(`${fps} fps`);
  if (finite(chunkIndex)) parts.push(`chunk ${chunkIndex}`);
  if (finite(generatedSeconds) && generatedSeconds > 0) parts.push(`${generatedSeconds.toFixed(1)} s generated`);
  if (finite(playedSeconds) && playedSeconds > 0) parts.push(`${playedSeconds.toFixed(1)} s played`);
  if (finite(bufferAhead)) parts.push(`buffer ${bufferAhead.toFixed(2)} s`);
  if (sessionId) parts.push(`session ${String(sessionId).slice(0, 8)}`);
  return parts.join(" · ");
}

/**
 * The download name: the prompt's first few words plus a UTC `YYYYMMDD-HHMMSS`
 * stamp, e.g. `lingbot-a-quiet-forest-path-20261021-113925.mp4`.
 *
 * @param {string} prompt
 * @param {number} [now] epoch milliseconds, injectable for tests
 * @returns {string}
 */
export function saveName(prompt, now = Date.now()) {
  const words = String(prompt ?? "")
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]+/gu, " ")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, MAX_NAME_WORDS);
  const slug = words.join("-").replace(/-+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48) || "world";
  const stamp = new Date(now).toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "-");
  return `lingbot-${slug}-${stamp}.mp4`;
}

/**
 * A unique interaction id. The zero-padded sequence keeps ids of one kind in
 * increasing string order, which is what the server records for playback.
 *
 * @param {string} kind `camera` or `prompt`
 * @param {number} sequence a 1-based counter
 * @returns {string}
 */
export function eventId(kind, sequence) {
  const n = Number.isFinite(sequence) ? Math.max(0, Math.floor(sequence)) : 0;
  const name = String(kind || "event").toLowerCase().replace(/[^a-z0-9-]+/g, "-") || "event";
  return `lingbot-${name}-${String(n).padStart(8, "0")}`;
}

const STYLE = `
.lingbot-live-backdrop { position: fixed; inset: 0; z-index: 1000; display: flex; align-items: center; justify-content: center; background: rgb(0 0 0 / 0.55); }
.lingbot-live { box-sizing: border-box; width: min(920px, 94vw); max-height: 92vh; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; padding: 12px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--comfy-menu-bg); color: var(--input-text); font: 12px sans-serif; }
.lingbot-live-title { font-size: 14px; font-weight: bold; }
.lingbot-live-preview { position: relative; align-self: center; width: 100%; background: #000; border: 1px solid var(--border-color); border-radius: 4px; overflow: hidden; outline: none; }
.lingbot-live-preview.drive { cursor: pointer; }
.lingbot-live-preview:focus { outline: 2px solid var(--p-primary-color, #3b82f6); outline-offset: 2px; cursor: default; }
.lingbot-live-stage { position: relative; width: 100%; aspect-ratio: 16 / 9; background: #000; }
.lingbot-live-output { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; background: #000; }
.lingbot-live-loading { position: absolute; inset: 0; z-index: 1; display: flex; flex-direction: column; gap: 12px; align-items: center; justify-content: center; padding: 16px; background: rgb(0 0 0 / 0.6); color: #fff; font-size: 13px; text-align: center; pointer-events: none; }
.lingbot-live-loading[hidden] { display: none; }
.lingbot-live-loading.error { background: rgb(0 0 0 / 0.8); color: #ff8a8a; pointer-events: auto; user-select: text; }
.lingbot-live-spinner { width: 28px; height: 28px; border: 3px solid rgb(255 255 255 / 0.25); border-top-color: #fff; border-radius: 50%; animation: lingbot-live-spin 0.9s linear infinite; }
.lingbot-live-loading.error .lingbot-live-spinner { display: none; }
@keyframes lingbot-live-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .lingbot-live-spinner { animation-duration: 3s; } }
.lingbot-live-hint { position: absolute; left: 50%; bottom: 10px; transform: translateX(-50%); padding: 3px 10px; border-radius: 3px; background: rgb(0 0 0 / 0.65); color: #fff; pointer-events: none; white-space: nowrap; }
.lingbot-live-hint[hidden] { display: none; }
.lingbot-live-preview:focus .lingbot-live-hint { display: none; }
.lingbot-live-status { min-height: 15px; opacity: 0.8; }
.lingbot-live-status:empty { display: none; }
.lingbot-live-status.error { color: #e05252; opacity: 1; }
.lingbot-live-keys { display: flex; gap: 24px; justify-content: center; }
.lingbot-live-pad-group { display: flex; flex-direction: column; gap: 4px; align-items: center; }
.lingbot-live-pad-caption { font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.55; }
.lingbot-live-pad { display: grid; grid-template-columns: repeat(3, 30px); gap: 4px; }
.lingbot-live-key { position: relative; display: flex; align-items: center; justify-content: center; height: 30px; border: 1px solid var(--border-color); border-radius: 4px; background: var(--comfy-input-bg); opacity: 0.6; transition: background 80ms, opacity 80ms; }
.lingbot-live-key { grid-row: 2; }
.lingbot-live-key.up { grid-row: 1; }
.lingbot-live-key.on { opacity: 1; background: var(--p-primary-color, #3b82f6); border-color: var(--p-primary-color, #3b82f6); color: #fff; }
.lingbot-live-key-glyph { font-size: 14px; line-height: 1; }
.lingbot-live-key-label { position: absolute; right: 3px; bottom: 2px; font-size: 8px; line-height: 1; opacity: 0.75; }
.lingbot-live-stats { opacity: 0.6; font-variant-numeric: tabular-nums; }
.lingbot-live-stats:empty { display: none; }
.lingbot-live-composer { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: start; }
.lingbot-live-actions { display: flex; flex-direction: column; gap: 6px; }
.lingbot-live-speed { display: flex; align-items: center; gap: 6px; opacity: 0.8; }
.lingbot-live-speed input { width: 110px; accent-color: var(--p-primary-color, #3b82f6); }
.lingbot-live textarea, .lingbot-live button { background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 3px; font: inherit; }
.lingbot-live textarea { min-height: 52px; padding: 6px 8px; resize: vertical; }
.lingbot-live button { padding: 4px 12px; cursor: pointer; }
.lingbot-live button:disabled { cursor: default; opacity: 0.5; }
.lingbot-live-row { display: flex; gap: 8px; align-items: center; }
.lingbot-live-row.end { justify-content: flex-end; }
@media (max-width: 560px) { .lingbot-live-composer { grid-template-columns: 1fr; } }
`;

/** Build a DOM node from a tag, attributes and children. No `innerHTML`. */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "className") node.className = value;
    else if (name === "textContent") node.textContent = value;
    else if (name === "dataset") Object.assign(node.dataset, value);
    else if (name === "style") node.style.cssText = value;
    else if (name === "value") node.value = value;
    else node.setAttribute(name, value === true ? "" : value);
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child);
  }
  return node;
}

/** Inject the panel style sheet once, while a ComfyUI page is present. */
function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = STYLE;
  document.head.append(style);
}

/** `Blob` + `<a download>`: the browser's normal download flow, no server round trip. */
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = el("a", { href: url, download: filename });
  anchor.style.display = "none";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  // The click has started the download; the URL can go shortly after.
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function formatBytes(bytes) {
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** The open panel's close function, so a second open or a graph reload can close it. */
let closePanel = null;
/** Guards the `await import()` so a double click cannot build two panels. */
let openingPanel = false;

/**
 * Load the player and build the panel. Kept async only for the dynamic import;
 * the panel body below is synchronous.
 */
async function openPanel(node, app) {
  if (openingPanel) return;
  openingPanel = true;
  closePanel?.();
  try {
    const { VideoSession, cameraEvent } = await import("./lingbot-player.js");
    buildPanel(node, app, VideoSession, cameraEvent);
  } finally {
    openingPanel = false;
  }
}

function buildPanel(node, app, VideoSession, cameraEvent) {
  ensureStyle();

  // ---- DOM: backdrop, dark card, 16:9 preview, legend, composer, actions, stats ----
  const output = el("video", {
    className: "lingbot-live-output",
    autoplay: true,
    muted: true,
    playsInline: true,
    controls: true,
    "aria-label": "Generated world",
  });
  output.muted = true;
  const stage = el("div", { className: "lingbot-live-stage" });
  const spinner = el("div", { className: "lingbot-live-spinner", "aria-hidden": "true" });
  const loadingText = el("div", { className: "lingbot-live-loading-text" });
  const loading = el("div", { className: "lingbot-live-loading" }, spinner, loadingText);
  const hint = el("div", { className: "lingbot-live-hint", textContent: "Click the video to drive", hidden: true });
  const preview = el("div", {
    className: "lingbot-live-preview",
    tabindex: 0,
    role: "group",
    "aria-label": "Generated world. Click the video, then use the W A S D and I J K L keys.",
  }, stage, output, loading, hint);

  const status = el("div", { className: "lingbot-live-status", role: "status", "aria-live": "polite" });

  const capNodes = new Map();
  const buildPad = (pad) => {
    const grid = el("div", { className: "lingbot-live-pad", role: "group", "aria-label": `${pad.name} keys` });
    for (const cap of pad.caps) {
      const keycap = el("div", {
        className: `lingbot-live-key${cap.row === 1 ? " up" : ""}`,
        "aria-hidden": "true",
      });
      keycap.style.gridColumn = String(cap.column);
      if (cap.glyph) {
        keycap.append(
          el("span", { className: "lingbot-live-key-glyph", textContent: cap.glyph }),
          el("span", { className: "lingbot-live-key-label", textContent: cap.label }),
        );
      } else {
        keycap.textContent = cap.label;
      }
      capNodes.set(cap.key, keycap);
      grid.append(keycap);
    }
    return el("div", { className: "lingbot-live-pad-group" },
      el("div", { className: "lingbot-live-pad-caption", textContent: pad.name }), grid);
  };
  const legend = el("div", { className: "lingbot-live-keys" }, ...hudCaps().map(buildPad));

  const prompt = el("textarea", {
    className: "lingbot-live-prompt",
    rows: 2,
    placeholder: "Describe what happens next…",
    "aria-label": "Session prompt",
  });
  const apply = el("button", { className: "lingbot-live-apply", textContent: "Apply", disabled: true });
  const speed = el("input", {
    type: "range",
    min: 0.1,
    max: 2,
    step: 0.1,
    value: 0.5,
    "aria-label": "Camera speed",
  });
  const speedValue = el("output", { className: "lingbot-live-speed-value", textContent: "0.5×" });
  const actions = el("div", { className: "lingbot-live-actions" },
    apply,
    el("label", { className: "lingbot-live-speed" }, el("span", { textContent: "Speed" }), speed, speedValue));
  const composer = el("div", { className: "lingbot-live-composer" }, prompt, actions);

  const start = el("button", { className: "lingbot-live-start", textContent: "Start" });
  const save = el("button", {
    className: "lingbot-live-save",
    textContent: "Save",
    title: "Stop the session and download the received video",
    disabled: true,
  });
  const stop = el("button", { className: "lingbot-live-stop", textContent: "Stop", title: "Stop and close the panel" });
  const buttons = el("div", { className: "lingbot-live-row end" }, start, save, stop);
  const stats = el("div", { className: "lingbot-live-stats", "aria-label": "Session stats" });

  const title = el("div", { className: "lingbot-live-title", textContent: "Omni LingBot World" });
  const card = el("div", {
    className: "lingbot-live",
    role: "dialog",
    "aria-modal": "true",
    "aria-label": "Omni LingBot World",
  }, title, preview, status, legend, composer, buttons, stats);
  const backdrop = el("div", { className: "lingbot-live-backdrop" }, card);
  document.body.append(backdrop);

  // ---- State ----
  let prepared = app.nodeOutputs[String(node.id)]?.omni_lingbot?.[0] ?? null;
  let config = null;          // the prepared record copied when Start was pressed
  let phase = prepared ? "ready" : "idle";
  let detail = "";
  let eventIndex = 0;         // one counter for camera and prompt events, reset per session
  const held = new Set();
  let promptEdited = false;
  let saving = false;
  let saved = false;
  let closed = false;

  const listeners = new AbortController();
  const options = { signal: listeners.signal };

  if (typeof prepared?.payload?.prompt === "string") prompt.value = prepared.payload.prompt;

  const session = new VideoSession(output, (text) => onReport(text));

  // ---- Rendering ----
  function paintCaps() {
    for (const [key, node_] of capNodes) node_.classList.toggle("on", held.has(key));
  }

  function renderStatus() {
    const line = statusLine(phase, detail);
    status.textContent = line.text;
    status.classList.toggle("error", line.error);
    const overlaid = phase === "connecting" || phase === "waiting" || phase === "error";
    loading.hidden = !overlaid;
    loading.classList.toggle("error", line.error);
    loadingText.textContent = line.error ? line.text
      : phase === "connecting" ? "Connecting…"
      : phase === "waiting" ? "Generating the first chunk…"
      : "";
  }

  function statsSnapshot() {
    const payload = config?.payload ?? prepared?.payload ?? null;
    const buffered = output.buffered;
    const ahead = buffered?.length
      ? Math.max(0, buffered.end(buffered.length - 1) - output.currentTime)
      : null;
    const fps = payload?.fps;
    return {
      width: payload?.width ?? null,
      height: payload?.height ?? null,
      fps: Number.isFinite(fps) ? fps : null,
      chunkIndex: Number.isInteger(session.chunkIndex) ? session.chunkIndex : null,
      generatedSeconds: Number.isFinite(fps) && fps > 0 && session.framesReceived
        ? session.framesReceived / fps
        : null,
      playedSeconds: Number.isFinite(output.currentTime) ? output.currentTime : null,
      bufferAhead: ahead,
      sessionId: session.requestId ?? null,
    };
  }

  function updateControls() {
    const active = Boolean(session.ws && !session.done);
    start.disabled = !prepared || saving || active;
    apply.disabled = saving || saved || !session.ready || session.done || session.stopping
      || !prompt.value.trim();
    save.disabled = saving || saved || session.recordedBytes <= 0;
    stop.disabled = saving;
    const keyboard = Boolean(config?.keyboard ?? prepared?.keyboard);
    hint.hidden = !keyboard;
    preview.classList.toggle("drive", keyboard);
  }

  function render() {
    renderStatus();
    paintCaps();
    stats.textContent = statsText(statsSnapshot());
    updateControls();
  }

  function setPhase(next, note = "") {
    phase = next;
    detail = note;
    render();
  }

  function setDetail(note) {
    detail = note;
    render();
  }

  function onReport(text) {
    if (closed || saving) return;
    if (text === "Generating the first chunk…") return setPhase("waiting");
    if (text.startsWith("Chunk ")) return setPhase(session.done ? "complete" : "playing");
    if (text.startsWith("Queued:")) return setDetail("Prompt queued; the next chunk applies it.");
    if (text === "Stopping…") return setPhase("stopping");
    if (text.startsWith("Stopped")) return setPhase("stopped");
    if (text.startsWith("Generation complete") || text === "Playback complete.") return setPhase("complete");
    if (text.startsWith("Click the video play button")) return setDetail(text);
    return setPhase("error", text);
  }

  // ---- Keyboard and camera ----
  function sendCamera() {
    if (!config?.keyboard || !session.ready || session.done || session.stopping) return false;
    return session.send(cameraEvent(held, eventId("camera", ++eventIndex), {
      fps: config.payload.fps,
      speed: Number(speed.value),
    }));
  }

  function releaseKeys() {
    if (!held.size) return;
    held.clear();
    paintCaps();
    sendCamera();
  }

  function onKeyDown(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopImmediatePropagation();
      close();
      return;
    }
    // Text fields keep the keys they need; the overlay owns every other key so
    // ComfyUI's hotkeys cannot fire behind it (the reference does the same).
    if (event.target?.closest?.("textarea, input, select")) return;
    event.stopImmediatePropagation();
    if (document.activeElement !== preview) return;
    const key = CODE_TO_KEY[event.code];
    if (!key || event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
    event.preventDefault();
    if (!config?.keyboard || !session.ready || session.done || session.stopping || held.has(key)) return;
    held.add(key);
    paintCaps();
    sendCamera();
  }

  function onKeyUp(event) {
    const key = CODE_TO_KEY[event.code];
    if (!key || !held.delete(key)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    paintCaps();
    sendCamera();
  }

  // ---- Composer ----
  function applyPrompt() {
    const text = prompt.value.trim();
    if (!text || !session.ready || session.done || session.stopping) return;
    const sent = session.send({
      type: "session.interaction",
      interaction: { event_id: eventId("prompt", ++eventIndex), event: { prompt: text } },
    });
    if (sent) setDetail("Prompt queued; the next chunk applies it.");
  }

  // ---- Session ----
  function startSession() {
    const next = app.nodeOutputs[String(node.id)]?.omni_lingbot?.[0];
    if (!next) {
      setPhase("error", "Run the workflow first; it prepares the image and settings.");
      return;
    }
    releaseKeys();
    prepared = next;
    config = { ...next };
    eventIndex = 0;
    saved = false;
    if (!promptEdited && typeof config.payload?.prompt === "string") prompt.value = config.payload.prompt;
    if (Number.isFinite(config.payload?.width) && Number.isFinite(config.payload?.height)) {
      stage.style.aspectRatio = `${config.payload.width} / ${config.payload.height}`;
    }
    setPhase("connecting");
    try {
      session.start(config);
      if (config.keyboard) preview.focus();
    } catch (error) {
      setPhase("error", error?.message || String(error));
    }
    render();
  }

  function waitForStop() {
    if (!session.ws || session.done) return Promise.resolve();
    return new Promise((resolve) => {
      const startedAt = Date.now();
      const check = setInterval(() => {
        if (closed || session.done || !session.ws || Date.now() - startedAt >= SAVE_STOP_TIMEOUT_MS) {
          clearInterval(check);
          resolve();
        }
      }, 100);
    });
  }

  async function saveTake() {
    if (saving || closed) return;
    releaseKeys();
    saving = true;
    setPhase("saving");
    const source = config?.payload?.prompt || prompt.value;
    if (session.ws && !session.done && !session.stopping) session.stop();
    await waitForStop();
    if (closed) return;
    saving = false;
    const blob = typeof session.recordingBlob === "function" ? session.recordingBlob() : null;
    if (!blob || blob.size === 0) {
      setPhase("error", "No video bytes were received to save.");
      return;
    }
    downloadBlob(blob, saveName(source, Date.now()));
    saved = true;
    setPhase("stopped", `Saved ${formatBytes(blob.size)} to your downloads.`);
  }

  function close() {
    if (closed) return;
    closed = true;
    releaseKeys();
    // Same teardown as before: an open socket is told `session.stop` and closed.
    session.dispose();
    clearInterval(timer);
    listeners.abort();
    backdrop.remove();
    if (closePanel === close) closePanel = null;
  }

  // ---- Wiring ----
  preview.addEventListener("click", () => preview.focus(), options);
  preview.addEventListener("blur", releaseKeys, options);
  window.addEventListener("keydown", onKeyDown, { capture: true, signal: listeners.signal });
  window.addEventListener("keyup", onKeyUp, { capture: true, signal: listeners.signal });
  window.addEventListener("blur", releaseKeys, options);
  document.addEventListener("visibilitychange", () => { if (document.hidden) releaseKeys(); }, options);

  speed.addEventListener("input", () => {
    speedValue.textContent = `${Number(speed.value).toFixed(1)}×`;
    if (held.size) sendCamera();
  }, options);

  prompt.addEventListener("input", () => { promptEdited = true; updateControls(); }, options);
  apply.addEventListener("click", applyPrompt, options);
  start.addEventListener("click", startSession, options);
  save.addEventListener("click", () => { saveTake().catch((error) => setPhase("error", error?.message || String(error))); }, options);
  stop.addEventListener("click", close, options);
  backdrop.addEventListener("pointerdown", (event) => { if (event.target === backdrop) close(); }, options);
  output.addEventListener("play", () => {
    if (!session.ready || session.done || session.stopping || phase === "error" || saving) return;
    setPhase("playing");
  }, options);
  output.addEventListener("pause", () => {
    if (saving || session.done || session.stopping || !session.ready || phase !== "playing") return;
    setPhase("paused");
  }, options);
  window.addEventListener("pagehide", close, options);

  // The node output appears only after a run, so poll it; the same tick keeps
  // the buttons, stats line and stall check current without new listeners.
  const timer = setInterval(() => {
    if (closed) return;
    if (!node.graph || node.graph.getNodeById(node.id) !== node) { close(); return; }
    prepared = app.nodeOutputs[String(node.id)]?.omni_lingbot?.[0] ?? null;
    if (prepared?.payload?.prompt && !promptEdited && !prompt.value) prompt.value = prepared.payload.prompt;
    if (!prepared && !session.ws && phase === "ready") setPhase("idle");
    else if (prepared && phase === "idle") setPhase("ready");
    if (session.ws && !session.done && Date.now() - session.lastProgress > STALL_MS) {
      session.fail("No video progress for 90 seconds; restart the session.");
    }
    if (!session.ready && held.size) { held.clear(); paintCaps(); }
    render();
  }, 250);

  closePanel = close;
  render();
}

/** Install the extension on a ComfyUI `app`; kept separate so tests never run it. */
export function register(app) {
  if (!app || typeof app.registerExtension !== "function") return;
  app.registerExtension({
    name: "Omni.LingBotWorld",
    setup() { ensureStyle(); },
    nodeCreated(node) {
      if (!["VLLMOmniLingBotWorld", "VLLMOmniLingBotRealtimeJSON"].includes(node?.comfyClass)) return;
      node.addWidget("button", "Open interaction panel", null, () => {
        openPanel(node, app).catch((error) => {
          console.warn("vLLM-Omni: could not open the LingBot interaction panel.", error);
        });
      }, { serialize: false });
    },
    beforeConfigureGraph() {
      closePanel?.();
    },
  });
}

// ComfyUI imports every file in `WEB_DIRECTORY` itself, so the panel has to
// register here; the guard keeps the module importable in Node, where neither
// ComfyUI's `app` nor a DOM exists.
if (typeof window !== "undefined" && typeof document !== "undefined") {
  const comfy = globalThis.app;
  if (comfy?.registerExtension) register(comfy);
  else {
    import("../../scripts/app.js")
      .then((module) => register(module.app))
      .catch((error) => console.warn("vLLM-Omni: could not register the LingBot panel.", error));
  }
}

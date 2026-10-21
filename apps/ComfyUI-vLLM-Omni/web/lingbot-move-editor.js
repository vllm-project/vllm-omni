// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

/**
 * @file In-node camera timeline editor for `VLLMOmniLingBotShot`.
 *
 * The shot node declares a required `timeline` input of type
 * `LINGBOT_TIMELINE`, whose value is `{"moves": [...]}` with each move
 * `{lane, value, start_frame, frames}` on the shot's own frame clock. This
 * module implements that input as a DOM widget: a frame/second ruler, the
 * `9 + 12k` chunk grid, one 22 px lane per camera lane, and the moves drawn as
 * draggable blocks.
 *
 * A shot's pixel length comes from its own `chunks` widget and its place in the
 * chain: a shot without a `sequence` link opens the sequence and lasts
 * `9 + 12 * (chunks - 1)` frames, while every later shot lasts `12 * chunks`.
 * The Python shot reads this widget's value as its camera moves, so what the
 * editor draws is exactly what renders.
 *
 * The visual language follows ComfyUI-Reactor's timeline editor
 * (Apache-2.0): the same ruler/lane/block/panel anatomy and pointer-drag
 * gestures. The markup, code and `lingbot-tl-*` class names are ours.
 *
 * `layoutMoves`, `moveProblems` and `snapFrame` are pure — they touch neither
 * the DOM nor `window` — so the Node tests can import this file through a
 * `data:` URL. The extension registers itself under the usual
 * `window`/`document` guard, exactly like `lingbot-timeline.js`.
 */

/** The server's key rate; only labels the ruler's seconds. */
const FPS = 16;

/** A ruler keeps at most this many 1-2-5 ticks. */
const MAX_TICKS = 8;

/** Chunk grid lines are thinned to stay at least this far apart on screen. */
const MIN_CHUNK_GAP_PX = 6;

/** A guarded bound so a corrupt payload cannot loop forever. */
const MAX_CHUNKS = 20000;

/** The injected style sheet's id, so a reload does not stack sheets. */
const STYLE_ID = "lingbot-tl-style";

/**
 * Matches `LingBotGeometry`'s defaults. `temporalScale` is one latent frame in
 * pixel frames, and so the shortest move the compiler keeps (4).
 */
const DEFAULT_GEOMETRY = Object.freeze({
  firstChunkFrames: 9,
  chunkFrames: 12,
  latentFramesPerChunk: 3,
  temporalScale: 4,
});

/** Camera lanes, in drawing order, with their direction values and action keys. */
const LANE_SPECS = Object.freeze([
  Object.freeze({ lane: "move", values: Object.freeze(["forward", "back"]), keys: Object.freeze({ forward: "w", back: "s" }) }),
  Object.freeze({ lane: "strafe", values: Object.freeze(["left", "right"]), keys: Object.freeze({ left: "a", right: "d" }) }),
  Object.freeze({ lane: "look_v", values: Object.freeze(["up", "down"]), keys: Object.freeze({ up: "i", down: "k" }) }),
  Object.freeze({ lane: "look_h", values: Object.freeze(["left", "right"]), keys: Object.freeze({ left: "j", right: "l" }) }),
]);

const LANE_BY_NAME = new Map(LANE_SPECS.map((spec, row) => [spec.lane, { ...spec, row }]));

/* ------------------------------------------------------------------------- */
/* Pure layout                                                               */
/* ------------------------------------------------------------------------- */

const isFiniteNumber = (value) => typeof value === "number" && Number.isFinite(value);
const positiveInt = (value) => (isFiniteNumber(value) && value > 0 ? Math.floor(value) : 0);
const asList = (value) => (Array.isArray(value) ? value : []);

/** Accept both the Python payload's snake_case keys and camelCase ones. */
function normalizeGeometry(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const pick = (snake, camel, fallback) =>
    positiveInt(source[snake]) || positiveInt(source[camel]) || fallback;
  return {
    firstChunkFrames: pick("first_chunk_frames", "firstChunkFrames", DEFAULT_GEOMETRY.firstChunkFrames),
    chunkFrames: pick("chunk_frames", "chunkFrames", DEFAULT_GEOMETRY.chunkFrames),
    latentFramesPerChunk: pick(
      "latent_frames_per_chunk",
      "latentFramesPerChunk",
      DEFAULT_GEOMETRY.latentFramesPerChunk,
    ),
    temporalScale: pick("temporal_scale", "temporalScale", DEFAULT_GEOMETRY.temporalScale),
  };
}

/** A shot's length in whole pixel frames; a missing or negative length is 0. */
function shotLength(raw) {
  return isFiniteNumber(raw) ? Math.max(0, Math.floor(raw)) : 0;
}

/** The smallest 1-2-5 step at least `target`, so a ruler needs at most `MAX_TICKS` labels. */
function niceStep(target) {
  if (!(target > 0)) return null;
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const normalized = target / magnitude;
  const multiple = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiple * magnitude;
}

function formatSeconds(frame) {
  const seconds = frame / FPS;
  const text = Math.abs(seconds - Math.round(seconds)) < 1e-9
    ? seconds.toFixed(1)
    : String(Number(seconds.toFixed(3)));
  return `${text}s`;
}

/** Ruler ticks: frame number and the same frame in seconds at 16 FPS. */
function buildTicks(length) {
  const step = niceStep(length / MAX_TICKS);
  if (!step) return { step: null, ticks: [] };
  const count = Math.min(Math.floor(length / step + 1e-9) + 1, MAX_TICKS);
  const ticks = [];
  for (let index = 0; index < count; index += 1) {
    const frame = index * step;
    ticks.push({ frame, seconds: frame / FPS, label: `${frame}f · ${formatSeconds(frame)}` });
  }
  return { step, ticks };
}

/**
 * Chunk boundaries relative to this shot. The first chunk is short (9 frames)
 * and every later one 12, the `9 + 12k` grid; a later shot passes a geometry
 * whose `firstChunkFrames` is already 12 because it begins on a chunk edge.
 */
function buildChunks(geometry, length) {
  const chunks = [];
  if (length <= 0) return chunks;
  let start = 0;
  for (let index = 0; start < length && index < MAX_CHUNKS; index += 1) {
    const frames = index === 0 ? geometry.firstChunkFrames : geometry.chunkFrames;
    const end = Math.min(start + frames, length);
    chunks.push({ index, start, end, frames: end - start });
    start += frames;
  }
  return chunks;
}

/**
 * Round `frame` to the nearest latent boundary (a multiple of
 * `temporalScale`, 4 pixel frames) and keep it inside `[0, limit]`. `limit`
 * itself is legal, so a move may end exactly on a shot's last frame even when
 * that frame is not on the latent grid (the opening shot is `9 + 12k`).
 */
export function snapFrame(frame, geometry, limit = Infinity) {
  const config = normalizeGeometry(geometry);
  const value = isFiniteNumber(frame) ? frame : 0;
  const max = isFiniteNumber(limit) ? Math.max(0, limit) : Infinity;
  // The shot's last frame is a legal end even when it is not on the latent grid.
  if (value >= max) return max;
  if (value <= 0) return 0;
  const snapped = Math.round(value / config.temporalScale) * config.temporalScale;
  return Math.max(0, Math.min(max, snapped));
}

/**
 * One entry per move in `moves`, with the raw and visible spans and the row it
 * is drawn on. Lanes are always the four known lanes, in camera order, plus a
 * row for every unknown lane so a bad move stays selectable and deletable.
 */
function normalizeMoves(moves, geometry, length) {
  const entries = [];
  const lanes = LANE_SPECS.map((spec) => ({
    lane: spec.lane,
    row: LANE_BY_NAME.get(spec.lane).row,
    known: true,
    values: [...spec.values],
    keys: { ...spec.keys },
    moves: [],
  }));
  const extras = new Map();

  asList(moves).forEach((raw, index) => {
    const lane = typeof raw?.lane === "string" ? raw.lane : "";
    const value = typeof raw?.value === "string" ? raw.value : "";
    const spec = LANE_BY_NAME.get(lane) ?? null;
    const start = isFiniteNumber(raw?.start_frame) ? Math.max(0, Math.round(raw.start_frame)) : 0;
    const requestedFrames = isFiniteNumber(raw?.frames) ? Math.max(0, Math.round(raw.frames)) : 0;
    const requestedEnd = start + requestedFrames;
    const end = Math.min(requestedEnd, length);
    const frames = Math.max(0, end - start);
    const clipped = requestedEnd > length && start < length;

    let row;
    if (spec) {
      row = spec.row;
    } else {
      if (!extras.has(lane)) {
        const extra = {
          lane,
          row: lanes.length,
          known: false,
          values: [],
          keys: {},
          moves: [],
        };
        extras.set(lane, extra);
        lanes.push(extra);
      }
      row = extras.get(lane).row;
    }

    entries.push({
      index,
      lane,
      value,
      key: spec ? spec.keys[value] ?? null : null,
      row,
      laneKnown: Boolean(spec),
      valueKnown: Boolean(spec && Object.prototype.hasOwnProperty.call(spec.keys, value)),
      start,
      end,
      frames,
      requestedStart: start,
      requestedFrames,
      clipped,
      startPercent: length > 0 ? (100 * start) / length : 0,
      endPercent: length > 0 ? (100 * end) / length : 0,
      widthPercent: length > 0 ? Math.max(0, (100 * frames) / length) : 0,
      label: `${lane || "?"} ${value || "?"} · ${frames}f`,
      geometry,
    });
    lanes[row].moves.push(index);
  });

  return { entries, lanes };
}

/**
 * Everything the editor draws, in shot-relative frames.
 *
 * @returns {{
 *   fps: number,
 *   frames: number,
 *   seconds: number,
 *   geometry: object,
 *   lanes: {lane: string, row: number, known: boolean, values: string[],
 *     keys: Record<string, string>, moves: number[]}[],
 *   moves: {index: number, lane: string, value: string, key: string|null,
 *     row: number, start: number, end: number, frames: number,
 *     startPercent: number, endPercent: number, widthPercent: number,
 *     label: string, clipped: boolean, laneKnown: boolean,
 *     valueKnown: boolean, requestedStart: number, requestedFrames: number}[],
 *   chunks: {index: number, start: number, end: number, frames: number}[],
 *   ticks: {frame: number, seconds: number, label: string}[],
 *   tickStep: number|null,
 * }}
 */
export function layoutMoves(moves, geometry, shotFrames) {
  const config = normalizeGeometry(geometry);
  const length = shotLength(shotFrames);
  const { entries, lanes } = normalizeMoves(moves, config, length);
  const chunks = buildChunks(config, length);
  const { step, ticks } = buildTicks(length);
  return {
    fps: FPS,
    frames: length,
    seconds: length / FPS,
    geometry: config,
    lanes,
    moves: entries,
    chunks,
    ticks,
    tickStep: step,
  };
}

/**
 * Why the shot would refuse each move, keyed by the move's index.
 *
 * Overlaps, unknown lanes/directions, a start past the shot's end and a move
 * shorter than one latent frame are errors (`invalid`); an end clipped by the
 * shot's end is only a warning, mirroring `lingbot_moves.py`'s cut.
 *
 * @returns {Map<number, {index: number, errors: string[], warnings: string[],
 *   invalid: boolean, messages: string[]}>}
 */
export function moveProblems(moves, geometry, shotFrames) {
  const config = normalizeGeometry(geometry);
  const length = shotLength(shotFrames);
  const { entries } = normalizeMoves(moves, config, length);
  const found = new Map();

  const add = (index, kind, message) => {
    let problem = found.get(index);
    if (!problem) {
      problem = { index, errors: [], warnings: [], invalid: false, messages: [] };
      found.set(index, problem);
    }
    problem[kind].push(message);
  };

  for (const move of entries) {
    if (!move.laneKnown) {
      add(move.index, "errors", `unknown camera lane "${move.lane}"`);
    } else if (!move.valueKnown) {
      add(move.index, "errors", `unknown ${move.lane} direction "${move.value}"`);
    }
    if (move.start >= length) {
      add(move.index, "errors", `starts at frame ${move.start}, past the shot's end at ${length}`);
    } else if (move.frames < config.temporalScale) {
      add(
        move.index,
        "errors",
        `shorter than one latent frame (${config.temporalScale} pixel frames)`,
      );
    }
    if (move.clipped) {
      add(move.index, "warnings", `runs past the shot's end at ${length} frames; cut there`);
    }
  }

  // Two moves on one lane may not hold the camera at once, as compile_sequence refuses.
  for (let left = 0; left < entries.length; left += 1) {
    for (let right = left + 1; right < entries.length; right += 1) {
      const a = entries[left];
      const b = entries[right];
      if (a.lane !== b.lane || a.frames <= 0 || b.frames <= 0) continue;
      if (a.start < b.end && b.start < a.end) {
        add(a.index, "errors", `overlaps another ${a.lane} move`);
        add(b.index, "errors", `overlaps another ${b.lane} move`);
      }
    }
  }

  for (const problem of found.values()) {
    problem.invalid = problem.errors.length > 0;
    problem.messages = [...problem.errors, ...problem.warnings];
  }
  return found;
}

/* ------------------------------------------------------------------------- */
/* Styles                                                                    */
/* ------------------------------------------------------------------------- */

const STYLE = `
.lingbot-tl { display: flex; flex-direction: column; gap: 6px; width: 100%; box-sizing: border-box; font: 12px sans-serif; color: var(--input-text); }
.lingbot-tl > * { flex-shrink: 0; }
.lingbot-tl [hidden] { display: none; }
.lingbot-tl-status { font-size: 11px; opacity: 0.65; font-variant-numeric: tabular-nums; }
.lingbot-tl-plot { display: flex; flex-direction: column; }
.lingbot-tl-ruler { position: relative; height: 18px; border-bottom: 1px solid var(--border-color); overflow: hidden; }
.lingbot-tl-tick { position: absolute; top: 0; bottom: 0; padding-left: 3px; border-left: 1px dotted var(--border-color); font-size: 10px; line-height: 16px; white-space: nowrap; opacity: 0.8; pointer-events: none; }
.lingbot-tl-lanes { display: flex; flex-direction: column; gap: 2px; margin-top: 3px; }
.lingbot-tl-lane { position: relative; height: 22px; box-sizing: border-box; background: var(--comfy-input-bg); border-radius: 4px; overflow: hidden; cursor: copy; }
.lingbot-tl-lane.unknown { background: color-mix(in srgb, #e05252 10%, var(--comfy-input-bg)); cursor: default; }
.lingbot-tl-lane-name { position: absolute; left: 4px; top: 4px; z-index: 1; font-size: 10px; opacity: 0.55; pointer-events: none; }
.lingbot-tl-grid { position: absolute; top: 0; bottom: 0; z-index: 0; border-left: 1px dotted var(--border-color); opacity: 0.5; pointer-events: none; }
.lingbot-tl-move { position: absolute; top: 2px; bottom: 2px; z-index: 2; box-sizing: border-box; min-width: 3px; padding: 0 6px; border: 1px solid var(--border-color); border-radius: 3px; background: color-mix(in srgb, var(--p-primary-color, #3b82f6) 30%, var(--comfy-menu-bg)); font-size: 11px; line-height: 16px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: grab; user-select: none; }
.lingbot-tl-move.selected { outline: 2px solid var(--p-primary-color, #3b82f6); outline-offset: -1px; }
.lingbot-tl-move.invalid { border-color: #e05252; background: color-mix(in srgb, #e05252 25%, var(--comfy-menu-bg)); }
.lingbot-tl-move.clipped { border-style: dashed; }
.lingbot-tl-resize { position: absolute; top: 0; right: 0; bottom: 0; width: 7px; cursor: ew-resize; background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--input-text) 22%, transparent)); }
.lingbot-tl-note { font-size: 11px; opacity: 0.75; }
.lingbot-tl-panel { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 8px; align-items: center; }
.lingbot-tl-field { display: flex; align-items: center; gap: 5px; font-size: 11px; min-width: 0; }
.lingbot-tl-field > span { min-width: 54px; opacity: 0.75; }
.lingbot-tl-panel select, .lingbot-tl-panel input, .lingbot-tl-panel button { min-width: 0; padding: 1px 3px; border: 1px solid var(--border-color); border-radius: 3px; background: var(--comfy-input-bg); color: var(--input-text); font: inherit; font-size: 11px; }
.lingbot-tl-panel button { padding: 2px 8px; cursor: pointer; }
.lingbot-tl-row { display: flex; align-items: center; gap: 6px; }
.lingbot-tl-problem { grid-column: 1 / -1; color: #e05252; }
.lingbot-tl-warning { grid-column: 1 / -1; color: #c98a2e; }
`;

/* ------------------------------------------------------------------------- */
/* DOM helpers                                                               */
/* ------------------------------------------------------------------------- */

/** Build a DOM node from a tag, properties and children. No `innerHTML`. */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "className") node.className = value;
    else if (name === "textContent") node.textContent = value;
    else if (name === "style") node.style.cssText = value;
    else if (name === "value") node.value = value;
    else if (name === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(name, value === true ? "" : value);
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child);
  }
  return node;
}

let styleInjected = false;

/** Add the editor's style sheet once; safe to call when no document exists. */
function ensureStyle() {
  if (typeof document === "undefined") return;
  if (styleInjected || document.getElementById(STYLE_ID)) {
    styleInjected = true;
    return;
  }
  styleInjected = true;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = STYLE;
  document.head.append(style);
}

/** Whether an optional input is connected. Absent sockets count as unconnected. */
function inputLinked(node, name) {
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  const slot = inputs.findIndex((input) => input?.name === name);
  if (slot < 0) return false;
  if (typeof node.isInputConnected === "function") return node.isInputConnected(slot);
  return inputs[slot].link != null;
}

function widgetValue(node, name) {
  return node?.widgets?.find((widget) => widget.name === name)?.value;
}

/** An editor's natural height: its visible rows plus the 6 px gaps between them. */
function contentHeight(root) {
  if (!root || root.hidden) return 0;
  if (typeof root.getClientRects === "function" && !root.getClientRects().length) return null;
  const rows = [...root.children].filter((child) => !child.hidden);
  return rows.reduce((height, child) => height + (child.offsetHeight || 0), 0) + 6 * Math.max(0, rows.length - 1);
}

/** Size a node to its content, leaving a size the user dragged alone. */
function fitNode(node) {
  try {
    const size = node.computeSize?.();
    if (Array.isArray(size) && Number.isFinite(size[1]) && Math.abs(size[1] - node.size[1]) > 1) {
      node.setSize([node.size[0], size[1]]);
    }
  } catch {
    // A node that is already detached has no size to fit.
  }
}

/* ------------------------------------------------------------------------- */
/* Keyboard deletion                                                         */
/* ------------------------------------------------------------------------- */

/** The editor clicked last; the canvas takes focus, so Delete is caught on the window. */
let deleteTarget = null;
let deleteKeyInstalled = false;

function installDeleteKey() {
  if (deleteKeyInstalled || typeof window === "undefined") return;
  deleteKeyInstalled = true;
  window.addEventListener("pointerdown", (event) => {
    if (deleteTarget && !deleteTarget.root.contains(event.target)) deleteTarget = null;
  }, true);
  window.addEventListener("keydown", (event) => {
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    const target = deleteTarget;
    if (!target || !target.root.isConnected) return;
    if (event.target?.closest?.("input, textarea, select")) return;
    if (target.remove() === false) return;
    event.stopImmediatePropagation();
    event.preventDefault();
  }, true);
}

/* ------------------------------------------------------------------------- */
/* Editor widget                                                             */
/* ------------------------------------------------------------------------- */

let warnedMissingAddDOMWidget = false;

/** A no-op stand-in for ComfyUI versions that cannot host DOM widgets. */
function createFallbackWidget(inputName) {
  const value = { moves: [] };
  return {
    name: inputName,
    type: "LINGBOT_TIMELINE",
    value,
    options: { hideInPanel: true },
    getValue: () => ({ moves: value.moves.map((move) => ({ ...move })) }),
    setValue: () => {},
    getMinHeight: () => 0,
    onDraw: () => {},
  };
}

/**
 * Build the timeline editor for one shot node.
 *
 * @param {object} node the ComfyUI node
 * @param {string} inputName the `LINGBOT_TIMELINE` input's name (`timeline`)
 * @returns {object} the DOM widget, or a no-op widget on older ComfyUI
 */
function createMoveEditor(node, inputName) {
  ensureStyle();
  if (typeof node?.addDOMWidget !== "function") {
    if (!warnedMissingAddDOMWidget) {
      warnedMissingAddDOMWidget = true;
      console.warn(
        "vLLM-Omni: this ComfyUI has no addDOMWidget; the LingBot camera timeline editor is read-only.",
      );
    }
    return createFallbackWidget(inputName);
  }

  let value = { moves: [] };
  let selected = null;
  let signature = "";
  let height = 160;
  let laidOut = false;
  let currentState = null;

  const root = el("div", { className: "lingbot-tl" });
  const status = el("div", { className: "lingbot-tl-status" });
  const ruler = el("div", { className: "lingbot-tl-ruler", "aria-hidden": "true" });
  const lanesEl = el("div", { className: "lingbot-tl-lanes", role: "group", "aria-label": "Camera move lanes" });
  const plot = el("div", { className: "lingbot-tl-plot" }, ruler, lanesEl);
  const panel = el("div", { hidden: true });
  const note = el("div", { className: "lingbot-tl-note" });
  root.append(status, plot, panel, note);

  installDeleteKey();
  root.addEventListener("pointerdown", () => {
    deleteTarget = { root, remove: removeSelected };
  }, true);

  /** The shot length, its chunk geometry, and the linked sockets, read from the node. */
  const state = () => {
    const chunkCount = Math.max(1, Math.floor(Number(widgetValue(node, "chunks")) || 1));
    const later = inputLinked(node, "sequence");
    const geometry = later
      ? { ...DEFAULT_GEOMETRY, firstChunkFrames: DEFAULT_GEOMETRY.chunkFrames }
      : { ...DEFAULT_GEOMETRY };
    return {
      chunkCount,
      later,
      frames: later
        ? DEFAULT_GEOMETRY.chunkFrames * chunkCount
        : DEFAULT_GEOMETRY.firstChunkFrames + DEFAULT_GEOMETRY.chunkFrames * (chunkCount - 1),
      geometry,
      width: lanesEl.clientWidth || 0,
      prompt: String(widgetValue(node, "prompt") ?? ""),
    };
  };

  const changed = () => {
    signature = "";
    render();
    node.graph?.setDirtyCanvas?.(true, true);
  };

  function removeSelected() {
    if (selected === null || selected < 0 || selected >= value.moves.length) return false;
    value.moves.splice(selected, 1);
    selected = null;
    changed();
    return true;
  }

  /** Window-level pointer drag, since each redraw replaces the element the drag began on. */
  const drag = (onMove) => {
    const onPointerMove = (event) => {
      onMove(event);
      changed();
    };
    const finish = () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
  };

  const frameAt = (clientX, frames) => {
    const rect = plot.getBoundingClientRect();
    if (!rect.width) return 0;
    return ((clientX - rect.left) / rect.width) * frames;
  };

  const pct = (frame, frames) => (frames > 0 ? `${(100 * frame) / frames}%` : "0%");

  function drawRuler(layout) {
    ruler.replaceChildren();
    const ticks = layout.ticks;
    const width = currentState.width || 400;
    // Thin labels (not the 1-2-5 step) when the node is too narrow to read them all.
    const every = ticks.length > 1 && width / ticks.length < 62
      ? Math.ceil(62 / (width / ticks.length))
      : 1;
    ticks.forEach((tick, index) => {
      if (index % every !== 0) return;
      ruler.append(el("div", {
        className: "lingbot-tl-tick",
        style: `left:${pct(tick.frame, layout.frames)}`,
        textContent: tick.label,
      }));
    });
  }

  function gridLines(layout, width) {
    const chunkFrames = layout.geometry.chunkFrames;
    const effectiveWidth = Math.max(width || 400, 1);
    const every = Math.max(
      1,
      Math.ceil((MIN_CHUNK_GAP_PX * layout.frames) / (effectiveWidth * chunkFrames)),
    );
    return layout.chunks
      .filter((chunk) => chunk.index > 0 && chunk.index % every === 0)
      .map((chunk) => el("div", {
        className: "lingbot-tl-grid",
        style: `left:${pct(chunk.start, layout.frames)}`,
      }));
  }

  function moveTitle(move, problem) {
    const lines = [
      `${move.lane || "(no lane)"} ${move.value || "(no direction)"}, frames ${move.start} to ${move.end}`,
    ];
    for (const message of problem?.messages ?? []) lines.push(message);
    return lines.join("\n");
  }

  function drawLanes(layout, found) {
    lanesEl.replaceChildren();
    for (const lane of layout.lanes) {
      const row = el("div", {
        className: `lingbot-tl-lane${lane.known ? "" : " unknown"}`,
        dataset: { lane: lane.lane },
        title: lane.known ? `Click to add a ${lane.lane} move` : `Unknown camera lane "${lane.lane}"`,
      });
      row.append(el("span", {
        className: "lingbot-tl-lane-name",
        textContent: lane.known ? lane.lane : `${lane.lane || "(no lane)"} (unknown)`,
      }));
      row.append(...gridLines(layout, currentState.width));

      if (lane.known) {
        row.addEventListener("pointerdown", (event) => {
          if (event.button !== 0) return;
          const spec = LANE_BY_NAME.get(lane.lane);
          const start = snapFrame(frameAt(event.clientX, layout.frames), layout.geometry, layout.frames);
          if (start >= layout.frames) return;
          // One chunk by default, cut at the shot's end; no move shorter than one latent frame.
          const end = Math.min(layout.frames, start + layout.geometry.chunkFrames);
          if (end - start < layout.geometry.temporalScale) return;
          event.preventDefault();
          value.moves.push({
            lane: lane.lane,
            value: spec.values[0],
            start_frame: start,
            frames: end - start,
          });
          selected = value.moves.length - 1;
          changed();
        });
      }

      for (const index of lane.moves) {
        const move = layout.moves[index];
        const problem = found.get(index);
        const block = el("div", {
          className: [
            "lingbot-tl-move",
            index === selected && "selected",
            problem?.invalid && "invalid",
            move.clipped && "clipped",
          ].filter(Boolean).join(" "),
          title: moveTitle(move, problem),
          style: `left:${move.startPercent}%;width:${move.widthPercent}%`,
        });
        block.append(el("span", { className: "lingbot-tl-move-label", textContent: move.label }));

        block.addEventListener("pointerdown", (event) => {
          if (event.button !== 0) return;
          event.stopPropagation();
          event.preventDefault();
          selected = index;
          changed();
          const origin = move.start;
          const maxStart = Math.max(0, layout.frames - move.frames);
          const grabbedAt = frameAt(event.clientX, layout.frames);
          const raw = value.moves[index];
          drag((dragEvent) => {
            const delta = frameAt(dragEvent.clientX, layout.frames) - grabbedAt;
            raw.start_frame = Math.min(
              maxStart,
              snapFrame(origin + delta, layout.geometry, layout.frames),
            );
          });
        });

        const handle = el("div", {
          className: "lingbot-tl-resize",
          title: "Drag to set how long this move lasts",
        });
        handle.addEventListener("pointerdown", (event) => {
          if (event.button !== 0) return;
          event.stopPropagation();
          event.preventDefault();
          selected = index;
          changed();
          const raw = value.moves[index];
          const start = Number(raw.start_frame) || 0;
          drag((dragEvent) => {
            const end = Math.max(
              Math.min(layout.frames, start + layout.geometry.temporalScale),
              snapFrame(frameAt(dragEvent.clientX, layout.frames), layout.geometry, layout.frames),
            );
            raw.frames = Math.max(0, end - start);
          });
        });
        block.append(handle);
        row.append(block);
      }
      lanesEl.append(row);
    }
  }

  function drawPanel(layout, found) {
    panel.replaceChildren();
    panel.hidden = selected === null;
    note.hidden = selected !== null;
    if (selected === null) return;

    const move = layout.moves[selected];
    const raw = value.moves[selected];
    if (!move || !raw) return;
    const problem = found.get(selected);

    const laneSelect = el("select", { "aria-label": "Camera lane" });
    const laneNames = LANE_SPECS.map((spec) => spec.lane);
    if (!laneNames.includes(move.lane)) laneNames.push(move.lane);
    for (const laneName of laneNames) {
      laneSelect.append(el("option", { value: laneName, textContent: laneName || "(none)" }));
    }
    laneSelect.value = move.lane;
    laneSelect.addEventListener("change", () => {
      const spec = LANE_BY_NAME.get(laneSelect.value);
      raw.lane = laneSelect.value;
      if (spec && !spec.values.includes(raw.value)) raw.value = spec.values[0];
      changed();
    });

    const directionSelect = el("select", { "aria-label": "Camera direction" });
    const spec = LANE_BY_NAME.get(move.lane);
    const directions = spec ? [...spec.values] : [];
    if (!directions.includes(move.value)) directions.push(move.value);
    for (const direction of directions) {
      directionSelect.append(el("option", {
        value: direction,
        textContent: direction || "(none)",
      }));
    }
    directionSelect.value = move.value;
    directionSelect.addEventListener("change", () => {
      raw.value = directionSelect.value;
      changed();
    });

    const startInput = el("input", {
      type: "number",
      min: "0",
      max: String(layout.frames),
      step: String(layout.geometry.temporalScale),
      value: String(move.start),
      "aria-label": "Start frame",
    });
    startInput.addEventListener("change", () => {
      const currentFrames = Math.max(Number(raw.frames) || 0, layout.geometry.temporalScale);
      const maxStart = Math.max(0, layout.frames - currentFrames);
      raw.start_frame = Math.min(
        maxStart,
        snapFrame(Number(startInput.value) || 0, layout.geometry, layout.frames),
      );
      changed();
    });

    const framesInput = el("input", {
      type: "number",
      min: "1",
      max: String(Math.max(1, layout.frames - move.start)),
      step: String(layout.geometry.temporalScale),
      value: String(move.frames),
      "aria-label": "Frames",
    });
    framesInput.addEventListener("change", () => {
      // Like a resize, the end snaps to the latent grid and keeps one latent frame.
      const start = Number(raw.start_frame) || 0;
      const minEnd = Math.min(layout.frames, start + layout.geometry.temporalScale);
      const end = Math.max(
        minEnd,
        snapFrame(start + (Math.round(Number(framesInput.value) || 0)), layout.geometry, layout.frames),
      );
      raw.frames = Math.max(0, end - start);
      changed();
    });

    const removeButton = el("button", {
      type: "button",
      textContent: "Remove move",
      title: "Remove this camera move (Delete)",
    });
    removeButton.addEventListener("click", () => removeSelected());

    panel.append(
      el("label", { className: "lingbot-tl-field" },
        el("span", { textContent: "lane" }), laneSelect),
      el("label", { className: "lingbot-tl-field" },
        el("span", { textContent: "direction" }), directionSelect),
      el("label", { className: "lingbot-tl-field" },
        el("span", { textContent: "start frame" }), startInput),
      el("label", { className: "lingbot-tl-field" },
        el("span", { textContent: "frames" }), framesInput),
      el("div", { className: "lingbot-tl-row" }, removeButton),
    );
    if (problem?.errors.length) {
      panel.append(el("div", { className: "lingbot-tl-problem", textContent: problem.errors.join("; ") }));
    }
    if (problem?.warnings.length) {
      panel.append(el("div", { className: "lingbot-tl-warning", textContent: problem.warnings.join("; ") }));
    }
  }

  function render() {
    const shot = state();
    currentState = shot;
    const layout = layoutMoves(value.moves, shot.geometry, shot.frames);
    const found = moveProblems(value.moves, shot.geometry, shot.frames);
    if (selected !== null && (selected < 0 || selected >= value.moves.length)) selected = null;

    status.textContent = `${shot.chunkCount} chunk${shot.chunkCount === 1 ? "" : "s"} · ` +
      `${layout.frames} frames · ${(layout.frames / FPS).toFixed(1)} s at ${FPS} FPS · ` +
      `${shot.later ? "later shot" : "opening shot"}`;
    note.textContent = "Click a lane to add a camera move…";

    drawRuler(layout);
    drawLanes(layout, found);
    drawPanel(layout, found);
  }

  // Declared before addDOMWidget so getMinHeight can see it even if ComfyUI
  // asks for the minimum size during construction.
  let widget;
  widget = node.addDOMWidget(inputName, "LINGBOT_TIMELINE", root, {
    getValue: () => ({ moves: value.moves.map((move) => ({ ...move })) }),
    setValue: (next) => {
      value = {
        moves: asList(next?.moves)
          .filter((move) => move && typeof move === "object")
          .map((move) => ({ ...move })),
      };
      selected = null;
      changed();
    },
    getMinHeight: () => height + 2 * (widget?.margin ?? 0),
    // The Parameters panel draws a widget by setting its width to the panel's and never restores it.
    hideInPanel: true,
    onDraw: () => {
      const shot = state();
      const nextSignature = JSON.stringify([
        shot.chunkCount,
        shot.later,
        shot.frames,
        shot.prompt,
        shot.width,
      ]);
      if (nextSignature !== signature) {
        signature = nextSignature;
        render();
      }
      // ComfyUI caches a DOM widget's height during the first layout, so a node
      // that measured tall before its styles settled would keep that height
      // forever. Fit both ways once the content settles instead.
      const measured = contentHeight(root);
      if (measured !== null && measured !== height) {
        height = measured;
        const refit = () => fitNode(node);
        if (typeof requestAnimationFrame === "function") requestAnimationFrame(refit);
        else refit();
        laidOut = true;
      }
    },
  });

  render();
  // The first draw can happen before the stylesheet and the fonts settle, and
  // ComfyUI caches a DOM widget's height from that first layout, so fit once
  // more after the next frame to remove any stale slack.
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      height = contentHeight(root) ?? height;
      const desired = height + 2 * (widget?.margin ?? 0);
      if (typeof widget?.computedHeight === "number" && widget.computedHeight !== desired) {
        widget.computedHeight = desired;
      }
      fitNode(node);
    }));
  }
  return widget;
}

/* ------------------------------------------------------------------------- */
/* Registration                                                              */
/* ------------------------------------------------------------------------- */

/** Install the editor on a ComfyUI `app`; kept separate so tests never build a DOM. */
export function register(app) {
  if (!app || typeof app.registerExtension !== "function") return;
  // ComfyUI can evaluate this module under two URLs (the extension scan and a
  // relative import), so the registration must be idempotent.
  if (globalThis.__vllmOmniLingBotEditor === true) return;
  globalThis.__vllmOmniLingBotEditor = true;
  app.registerExtension({
    name: "vllm.lingbot_move_editor",
    setup() {
      ensureStyle();
    },
    getCustomWidgets() {
      return {
        LINGBOT_TIMELINE: (node, inputName) => ({ widget: createMoveEditor(node, inputName) }),
      };
    },
  });
}

// ComfyUI imports every file in `WEB_DIRECTORY` itself, so the editor has to
// register here; the guard keeps the module importable in Node, where neither
// ComfyUI's `app` nor a DOM exists.
if (typeof window !== "undefined" && typeof document !== "undefined") {
  const comfy = globalThis.app;
  if (comfy?.registerExtension) register(comfy);
  else {
    import("../../scripts/app.js")
      .then((module) => register(module.app))
      .catch((error) => console.warn("vLLM-Omni: could not register the LingBot move editor.", error));
  }
}

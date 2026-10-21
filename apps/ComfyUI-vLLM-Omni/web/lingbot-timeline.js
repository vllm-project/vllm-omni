// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

/**
 * @file In-node timeline for `VLLMOmniLingBotVisualizer` ("LingBot World Visualizer").
 *
 * The Python node renders nothing; it returns
 * `{"ui": {"omni_lingbot_sequence": [payload]}}`. This module draws the plan
 * *inside* the node, like ComfyUI-Reactor's timeline viewer does for its
 * segments: a frame ruler, one block per shot, the prompt switch on every shot
 * boundary, one row per camera lane with its move bars, and a footer with the
 * totals and warnings.
 *
 * The plan is read live from the graph, not from a completed run: starting at
 * the Visualizer's `sequence` input it walks upstream through
 * `VLLMOmniLingBotShot` nodes (each links to the previous shot through its own
 * `sequence` input) and collects each shot's `prompt`, `chunks` and
 * `timeline.moves`. Editing any of those widgets, or resizing the node,
 * redraws. Only when the graph cannot be walked (the sequence comes from a
 * node this file does not know) does it fall back to the last run's payload in
 * `app.nodeOutputs[String(node.id)].omni_lingbot_sequence[0]`.
 *
 * The visual language and interaction model follow ComfyUI-Reactor's
 * timeline (Apache-2.0): a ruler over a shot track, thin chunk lines, lane
 * rows under it, a click-to-select note and a playhead driven by ComfyUI's
 * `progress` event. The code, markup and `lingbot-tl-*` class names are ours.
 *
 * `layoutSequence` is pure — no DOM, no `window` — so the geometry is tested in
 * Node. The playhead listens through ComfyUI's `api`, so `api` is imported at
 * the top; `register` runs only under a `window`/`document` guard, which keeps
 * this module importable where there is neither.
 */

import { api } from "../../scripts/api.js";

/* ------------------------------------------------------------------------- */
/* Facts shared with the Python nodes                                        */
/* ------------------------------------------------------------------------- */

/** Chunk 0 holds this many pixel frames (`LingBotGeometry.first_chunk_frames`). */
const FIRST_CHUNK_FRAMES = 9;
/** Every chunk after the first holds this many pixel frames. */
const CHUNK_FRAMES = 12;
/** Pixel frames per latent frame; a shorter move is widened by the renderer. */
const TEMPORAL_SCALE = 4;
/** The ruler keeps at most about this many labels. */
const MAX_TICKS = 8;
/** A chain longer than this is not a shot chain this node can draw. */
const MAX_SHOTS = 512;
/** A description cannot ask the layout to iterate more chunks than this. */
const MAX_CHUNKS = 20000;
/** Chunk grid lines are thinned to stay at least this far apart on screen. */
const MIN_CHUNK_GAP_PX = 6;
/** Lane rows are drawn in this order, whatever order the moves arrive in. */
const LANE_ORDER = ["move", "strafe", "look_v", "look_h"];
/** The style sheet is injected once under this id. */
const STYLE_ID = "lingbot-tl-sequence-style";

/* ------------------------------------------------------------------------- */
/* Layout (pure)                                                             */
/* ------------------------------------------------------------------------- */

const isFiniteNumber = (value) => typeof value === "number" && Number.isFinite(value);
const positiveInt = (value) => (isFiniteNumber(value) && value > 0 ? Math.floor(value) : 0);
const asText = (value) => (typeof value === "string" ? value : typeof value === "number" ? String(value) : "");
const asList = (value) => (Array.isArray(value) ? value : []);

/** Frame offset inside a shot, never negative. */
const asFrameOffset = (value) => (isFiniteNumber(value) && value > 0 ? Math.floor(value) : 0);

/**
 * First pixel frame of `chunk`. Chunk 0 is the short opening chunk
 * (`firstChunkFrames`, 9), every later one adds `chunkFrames` (12).
 */
function chunkStartFrame(chunk, firstChunkFrames, chunkFrames) {
  if (chunk <= 0) return 0;
  return firstChunkFrames + (chunk - 1) * chunkFrames;
}

/** Pixel frames a shot of `chunks` chunks holds, as `LingBotGeometry.shot_frames`. */
function framesForChunks(index, chunks, firstChunkFrames, chunkFrames) {
  if (chunks <= 0) return 0;
  return index === 0 ? firstChunkFrames + (chunks - 1) * chunkFrames : chunks * chunkFrames;
}

/** Chunks a shot of `frames` pixel frames covers, as `LingBotGeometry.shot_chunks`. */
function chunksForFrames(index, frames, firstChunkFrames, chunkFrames) {
  if (frames <= 0) return 0;
  if (index === 0) {
    if (frames <= firstChunkFrames) return 1;
    return 1 + Math.round((frames - firstChunkFrames) / chunkFrames);
  }
  return Math.max(1, Math.round(frames / chunkFrames));
}

/** One move from a description or a payload, or null when it cannot be read. */
function readMove(raw) {
  if (!raw || typeof raw !== "object") return null;
  const frames = positiveInt(raw.frames ?? raw.length);
  if (frames <= 0) return null;
  return {
    lane: asText(raw.lane) || "move",
    value: asText(raw.value),
    start: asFrameOffset(raw.start_frame ?? raw.startFrame),
    frames,
  };
}

const overrideWarning = (index) =>
  `Shot ${index}: its camera moves come from the connected Moves node; run the workflow to draw them.`;
const cutWarning = (index, lane, frames) =>
  `Shot ${index}: the ${lane} move is cut to ${frames} frames at the shot's end.`;
const widenWarning = (index, lane, temporalScale) =>
  `Shot ${index}: the ${lane} move is widened to one latent frame (${temporalScale} pixel frames).`;
const lateWarning = (index, lane, start, shotFrames) =>
  `Shot ${index}: the ${lane} move starts at frame ${start}, after the shot's ${shotFrames} frames.`;

/** The smallest 1-2-5 step that is at least `target`. */
function niceStep(target) {
  if (!(target > 0)) return null;
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const normalized = target / magnitude;
  const multiple = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiple * magnitude;
}

function formatNumber(value) {
  const rounded = Math.abs(value) < 1e-9 ? 0 : Number(value.toFixed(3));
  return String(rounded);
}

/** Second labels from `fps`; frame numbers only when the description has no rate. */
function formatTick(frame, fps) {
  return fps ? `${formatNumber(frame / fps)}s` : `${formatNumber(frame)}f`;
}

function buildTicks(totalFrames, fps) {
  const step = niceStep(totalFrames / MAX_TICKS);
  if (!step || totalFrames <= 0) return { step: null, ticks: [] };
  const ticks = [];
  for (let index = 0; index * step <= totalFrames + 1e-9; index += 1) {
    const frame = Math.round(index * step * 1e6) / 1e6;
    ticks.push({ frame, label: formatTick(frame, fps) });
    if (ticks.length > MAX_TICKS + 1) break;
  }
  return { step, ticks };
}

/**
 * Everything needed to draw a sequence, in absolute frame numbers.
 *
 * Input, a plain description the viewer builds from the graph or from the
 * node's last `ui` payload; every field is optional and a missing one falls
 * back to the Python nodes' defaults:
 *
 *   `{shots: [{prompt, frames, moves}], firstChunkFrames, chunkFrames, fps}`
 *
 * A shot is `frames` pixel frames long and its `moves` are on the shot's own
 * clock (`{lane, value, start_frame, frames}`), exactly as the Shot node's
 * `timeline` widget stores them. `warnings` may carry the payload's own
 * messages; they are appended after the ones the layout finds.
 *
 * Output:
 * - `shots[i]`: `{index, prompt, start, end, frames, startChunk, chunks, problems}`,
 *   where `start`/`end` span the shot in sequence frames and `startChunk` its
 *   first chunk on the 9 + 12k grid.
 * - `bars`: `{shot, lane, value, start, end, frames, row}`, absolute frames,
 *   clipped to the shot's end.
 * - `lanes`: one `{lane, row, bars}` row per lane that carries moves.
 * - `chunks`: `{index, start, end, frames}` for the whole chunk grid.
 * - `switches`: `{shot, chunk, frame, prompt}` for every prompt switch.
 * - `ticks`/`tickStep`: the ruler frames and the 1-2-5 spacing between them.
 * - `totalFrames`, `totalChunks`, `seconds`, `warnings`, `empty`.
 *
 * Pure: no DOM and no `window`, and it accepts a missing or partial
 * description.
 */
export function layoutSequence(description) {
  const source = description && typeof description === "object" ? description : {};
  const firstChunkFrames = positiveInt(source.firstChunkFrames) || FIRST_CHUNK_FRAMES;
  const chunkFrames = positiveInt(source.chunkFrames) || CHUNK_FRAMES;
  const temporalScale = positiveInt(source.temporalScale) || TEMPORAL_SCALE;
  const fps = isFiniteNumber(source.fps) && source.fps > 0 ? source.fps : null;

  const shots = [];
  const bars = [];
  const laneBars = new Map();
  const derived = [];
  let frameCursor = 0;
  let chunkCursor = 0;

  for (const raw of asList(source.shots).slice(0, MAX_SHOTS)) {
    if (!raw || typeof raw !== "object") continue;
    const index = shots.length;
    const chunks = positiveInt(raw.chunks);
    let frames = positiveInt(raw.frames);
    if (!frames) frames = framesForChunks(index, chunks, firstChunkFrames, chunkFrames);
    const shotChunks = chunks || chunksForFrames(index, frames, firstChunkFrames, chunkFrames);
    const start = frameCursor;
    const end = start + frames;
    const problems = [];
    const note = (message) => {
      derived.push({ shot: index, message });
      if (!problems.includes(message)) problems.push(message);
    };
    if (raw.movesOverridden) note(overrideWarning(index));

    for (const rawMove of asList(raw.moves)) {
      const move = readMove(rawMove);
      if (!move) continue;
      if (move.start >= frames) {
        note(lateWarning(index, move.lane, move.start, frames));
        continue;
      }
      const barStart = start + move.start;
      const barEnd = Math.min(end, barStart + move.frames);
      const clipped = barEnd - barStart;
      if (clipped <= 0) continue;
      if (clipped < move.frames) note(cutWarning(index, move.lane, clipped));
      if (clipped < temporalScale) note(widenWarning(index, move.lane, temporalScale));
      const bar = {
        shot: index,
        lane: move.lane,
        value: move.value,
        start: barStart,
        end: barEnd,
        frames: clipped,
        row: 0,
      };
      bars.push(bar);
      if (!laneBars.has(move.lane)) laneBars.set(move.lane, []);
      laneBars.get(move.lane).push(bar);
    }

    shots.push({
      index,
      prompt: asText(raw.prompt),
      start,
      end,
      frames,
      startChunk: chunkCursor,
      chunks: shotChunks,
      problems,
    });
    frameCursor = end;
    chunkCursor += shotChunks;
  }

  const shotSpan = shots.length ? shots[shots.length - 1].end : 0;
  const totalFrames = Math.max(positiveInt(source.totalFrames), shotSpan);
  let totalChunks = positiveInt(source.totalChunks) || chunkCursor;
  if (!totalChunks && totalFrames > 0) {
    totalChunks = chunksForFrames(0, totalFrames, firstChunkFrames, chunkFrames);
  }
  totalChunks = Math.min(totalChunks, MAX_CHUNKS);

  const rank = (lane) => {
    const index = LANE_ORDER.indexOf(lane);
    return index === -1 ? LANE_ORDER.length : index;
  };
  const lanes = [...laneBars.entries()]
    .map(([lane, entries]) => ({
      lane,
      bars: entries.slice().sort((left, right) => left.start - right.start),
    }))
    .sort((left, right) => rank(left.lane) - rank(right.lane) ||
      (left.lane < right.lane ? -1 : left.lane > right.lane ? 1 : 0))
    .map((row, index) => {
      row.row = index;
      for (const bar of row.bars) bar.row = index;
      return row;
    });

  const chunks = [];
  for (let index = 0; index < totalChunks; index += 1) {
    const start = chunkStartFrame(index, firstChunkFrames, chunkFrames);
    const frames = index === 0 ? firstChunkFrames : chunkFrames;
    chunks.push({ index, start, end: start + frames, frames });
  }

  const switches = shots
    .filter((shot) => shot.index > 0)
    .map((shot) => ({
      shot: shot.index,
      chunk: shot.startChunk,
      frame: shot.start,
      prompt: shot.prompt,
    }));

  // A payload may carry warnings of its own; keep them after the ones the
  // layout found, without repeating a message the layout already derived.
  const warnings = [];
  const seen = new Set();
  for (const entry of derived) {
    if (seen.has(entry.message)) continue;
    seen.add(entry.message);
    warnings.push(entry.message);
  }
  for (const message of asList(source.warnings)) {
    if (typeof message !== "string" || seen.has(message)) continue;
    seen.add(message);
    warnings.push(message);
    const match = /^Shot (\d+):/.exec(message);
    const shot = match ? shots[Number(match[1])] : null;
    if (shot && !shot.problems.includes(message)) shot.problems.push(message);
  }

  const { step, ticks } = buildTicks(totalFrames, fps);

  return {
    fps,
    firstChunkFrames,
    chunkFrames,
    temporalScale,
    totalFrames,
    totalChunks,
    seconds: fps ? totalFrames / fps : null,
    shots,
    bars,
    lanes,
    chunks,
    switches,
    ticks,
    tickStep: step,
    warnings,
    empty: shots.length === 0,
  };
}

/* ------------------------------------------------------------------------- */
/* Reading the plan live from the graph                                      */
/* ------------------------------------------------------------------------- */

function widgetValue(node, name) {
  return node?.widgets?.find((widget) => widget?.name === name)?.value;
}

function linkedNode(node, slot) {
  const graph = node?.graph;
  if (typeof node?.getInputNode === "function") {
    const source = node.getInputNode(slot);
    if (source) return source;
  }
  const link = node?.inputs?.[slot]?.link;
  const data = typeof link === "object" && link
    ? link
    : graph?.links?.[link] ?? (typeof graph?.links?.get === "function" ? graph.links.get(link) : null);
  const origin = data?.origin_id;
  return origin == null ? null : graph?.getNodeById?.(origin) ?? null;
}

/** The node feeding an input, or null when the socket is unconnected. */
function upstream(node, inputName) {
  const slot = node?.inputs?.findIndex((input) => input?.name === inputName) ?? -1;
  if (slot < 0) return null;
  if (node.inputs[slot].link == null) return null;
  return linkedNode(node, slot);
}

/**
 * The moves a Shot node's `timeline` widget holds. The camera timeline editor
 * keeps its state behind `getValue`, so read that before the plain widget
 * value; a JSON string is accepted too.
 */
function timelineMoves(shotNode) {
  const widget = shotNode?.widgets?.find((entry) => entry?.name === "timeline");
  if (!widget) return [];
  const readers = [
    () => widget.options?.getValue?.(),
    () => widget.getValue?.(),
    () => widget.value,
  ];
  for (const read of readers) {
    let value;
    try {
      value = read();
    } catch {
      continue;
    }
    if (typeof value === "string") {
      try {
        value = JSON.parse(value);
      } catch {
        continue;
      }
    }
    if (value && typeof value === "object" && Array.isArray(value.moves)) return value.moves;
  }
  return [];
}

/**
 * The shots feeding `node`'s `sequence` input, first to last, as the layout
 * description's `shots`; null when the chain cannot be walked. Frames follow
 * `LingBotGeometry.shot_frames`: the first shot (the one with no upstream
 * `sequence`) opens the session and holds `9 + 12 * (chunks - 1)` frames;
 * every later shot is a run of full chunks, `12 * chunks`.
 */
function readShotChain(node) {
  const terminal = upstream(node, "sequence");
  if (!terminal) return null;
  const found = [];
  for (let current = terminal; current; current = upstream(current, "sequence")) {
    if (current.comfyClass !== "VLLMOmniLingBotShot") return null;
    found.push(current);
    if (found.length > MAX_SHOTS) return null;
  }
  found.reverse();
  return found.map((shotNode, index) => {
    const chunks = Math.max(1, Math.floor(Number(widgetValue(shotNode, "chunks")) || 0));
    return {
      prompt: asText(widgetValue(shotNode, "prompt")),
      frames: index === 0 ? FIRST_CHUNK_FRAMES + (chunks - 1) * CHUNK_FRAMES : CHUNK_FRAMES * chunks,
      moves: timelineMoves(shotNode),
      movesOverridden: Boolean(shotNode.inputs?.find?.((input) => input?.name === "moves")?.link),
    };
  });
}

/** The last run's `ui` payload as a layout description, or null when unusable. */
function descriptionFromPayload(payload, fps) {
  if (!payload || typeof payload !== "object") return null;
  const geometry = payload.geometry && typeof payload.geometry === "object" ? payload.geometry : {};
  const shots = asList(payload.shots)
    .map((shot) => ({
      prompt: asText(shot?.prompt),
      frames: positiveInt(shot?.frames),
      moves: asList(shot?.moves),
    }))
    .filter((shot) => shot.frames > 0);
  if (!shots.length) return null;
  const payloadFps = isFiniteNumber(payload.fps) && payload.fps > 0 ? payload.fps : null;
  return {
    shots,
    firstChunkFrames: positiveInt(geometry.first_chunk_frames),
    chunkFrames: positiveInt(geometry.chunk_frames),
    temporalScale: positiveInt(geometry.temporal_scale),
    fps: fps ?? payloadFps,
    warnings: asList(payload.warnings).filter((message) => typeof message === "string"),
  };
}

/* ------------------------------------------------------------------------- */
/* Style                                                                     */
/* ------------------------------------------------------------------------- */

const STYLE = `
.lingbot-tl.lingbot-tl-sequence { display: flex; flex-direction: column; gap: 6px; width: 100%;
  box-sizing: border-box; font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--input-text, #232a36); }
.lingbot-tl-sequence [hidden] { display: none !important; }
.lingbot-tl-sequence .lingbot-tl-ruler { position: relative; height: 16px; flex: none; overflow: hidden;
  border-bottom: 1px solid var(--border-color, #c3c9d6); }
.lingbot-tl-sequence .lingbot-tl-tick { position: absolute; top: 0; height: 100%; box-sizing: border-box;
  line-height: 16px; border-left: 1px solid var(--border-color, #c3c9d6); padding-left: 2px;
  font-size: 10px; opacity: 0.75; white-space: nowrap; pointer-events: none; }
.lingbot-tl-sequence .lingbot-tl-track { position: relative; height: 34px; flex: none; overflow: hidden;
  background: var(--comfy-input-bg, #eef2f8); border-radius: 4px; }
.lingbot-tl-sequence .lingbot-tl-grid { position: absolute; top: 0; bottom: 0; opacity: 0.5;
  pointer-events: none; border-left: 1px dotted var(--border-color, #c3c9d6); }
.lingbot-tl-sequence .lingbot-tl-shot { position: absolute; top: 3px; bottom: 3px; box-sizing: border-box;
  display: flex; align-items: center; gap: 4px; padding: 2px 5px; border-radius: 3px; overflow: hidden;
  cursor: pointer; user-select: none; font-size: 11px;
  background: var(--comfy-menu-bg, #e7ecf5); border: 1px solid var(--border-color, #93a1b8); }
.lingbot-tl-sequence .lingbot-tl-shot-alt {
  background: color-mix(in srgb, var(--comfy-menu-bg, #e7ecf5) 90%, var(--input-text, #232a36) 10%); }
.lingbot-tl-sequence .lingbot-tl-shot.selected {
  outline: 2px solid var(--p-primary-color, #4a9eff); outline-offset: -2px; z-index: 1; }
.lingbot-tl-sequence .lingbot-tl-shot-prompt {
  flex: 1 1 auto; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.lingbot-tl-sequence .lingbot-tl-length { flex: none; opacity: 0.65; font-variant-numeric: tabular-nums; }
.lingbot-tl-sequence .lingbot-tl-switch { position: absolute; top: 0; bottom: 0; width: 0;
  pointer-events: none; z-index: 1;
  border-left: 1px dashed color-mix(in srgb, var(--p-primary-color, #4a9eff) 65%, var(--input-text, #232a36) 35%); }
.lingbot-tl-sequence .lingbot-tl-playhead { position: absolute; top: 0; bottom: 0; width: 2px;
  margin-left: -1px; background: var(--p-primary-color, #4a9eff); pointer-events: none; z-index: 2; }
.lingbot-tl-sequence .lingbot-tl-lanes { display: flex; flex-direction: column; gap: 4px; flex: none; }
.lingbot-tl-sequence .lingbot-tl-lane { position: relative; height: 22px; overflow: hidden; cursor: default;
  background: var(--comfy-input-bg, #eef2f8); border-radius: 4px; }
.lingbot-tl-sequence .lingbot-tl-lane-name { position: absolute; left: 4px; top: 4px; z-index: 1;
  font-size: 10px; opacity: 0.55; pointer-events: none; }
.lingbot-tl-sequence .lingbot-tl-move { position: absolute; top: 2px; bottom: 2px; box-sizing: border-box;
  padding: 1px 5px; border-radius: 3px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
  font-size: 11px; line-height: 16px; cursor: default;
  background: color-mix(in srgb, var(--p-primary-color, #4a9eff) 28%, var(--comfy-menu-bg, #ffffff));
  border: 1px solid var(--border-color, #93a1b8); }
.lingbot-tl-sequence .lingbot-tl-empty { font-size: 11px; opacity: 0.65; }
.lingbot-tl-sequence .lingbot-tl-note { flex: none; font-size: 11px; opacity: 0.8;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lingbot-tl-sequence .lingbot-tl-problem { color: #e05252; }
.lingbot-tl-sequence .lingbot-tl-footer { display: flex; flex-direction: column; gap: 2px;
  flex: none; font-size: 11px; opacity: 0.8; }
.lingbot-tl-sequence .lingbot-tl-totals { font-variant-numeric: tabular-nums; }
.lingbot-tl-sequence .lingbot-tl-warnings { margin: 0; padding-left: 16px; color: #c26a2b; }
`;

function ensureStyle() {
  if (typeof document === "undefined" || document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = STYLE;
  document.head.append(style);
}

/* ------------------------------------------------------------------------- */
/* The in-node viewer                                                        */
/* ------------------------------------------------------------------------- */

/** A DOM element from a tag, attributes and children; no HTML strings anywhere. */
function el(tag, attributes = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null) continue;
    if (key === "style") element.style.cssText = value;
    else if (key === "className") element.className = value;
    else if (key === "textContent") element.textContent = value;
    else if (key === "dataset") Object.assign(element.dataset, value);
    else element.setAttribute(key, value === true ? "" : String(value));
  }
  element.append(...children.filter((child) => child !== null && child !== undefined && child !== false));
  return element;
}

const ROW_GAP = 6;

/**
 * The viewer's natural height: its visible rows and the gaps between them, or
 * null until it is laid out (an element that is not rendered measures 0).
 */
function contentHeight(root) {
  if (root.hidden) return 0;
  if (!root.getClientRects().length) return null;
  const rows = [...root.children].filter((child) => !child.hidden);
  return rows.reduce((height, child) => height + child.offsetHeight, 0) +
    ROW_GAP * Math.max(0, rows.length - 1);
}

/** Chunk grid lines thin to every `every`th chunk so they stay at least 6 px apart. */
function chunkGridEvery(width, layout) {
  const perChunk = (Math.max(1, width) * layout.chunkFrames) / Math.max(1, layout.totalFrames);
  return Math.max(1, Math.ceil(MIN_CHUNK_GAP_PX / perChunk));
}

/**
 * Build the DOM widget that draws `node`'s shot chain inside the node.
 *
 * The widget is `serialize: false`: it saves nothing, since the graph already
 * holds the plan. It grows the node once to fit its content and then leaves
 * the size alone, so a height the user dragged stays theirs.
 */
function createTimelineWidget(node, app) {
  ensureStyle();

  let selected = null;
  let signature = "";
  let height = 0;
  let laidOut = false;
  let widget = null;

  const root = el("div", { className: "lingbot-tl lingbot-tl-sequence" });
  const ruler = el("div", { className: "lingbot-tl-ruler" });
  const track = el("div", { className: "lingbot-tl-track" });
  const lanes = el("div", { className: "lingbot-tl-lanes" });
  const note = el("div", { className: "lingbot-tl-note" });
  const totals = el("div", { className: "lingbot-tl-totals" });
  const warningsList = el("ul", { className: "lingbot-tl-warnings" });
  const footer = el("div", { className: "lingbot-tl-footer" }, totals, warningsList);
  // The playhead lives in the track and is re-attached on every redraw.
  const playhead = el("div", { className: "lingbot-tl-playhead", hidden: true });
  root.append(ruler, track, lanes, note, footer);

  /** What the viewer draws from; any change here triggers a redraw. */
  const readState = () => {
    const fpsValue = Number(widgetValue(node, "fps"));
    const nodeFps = Number.isFinite(fpsValue) && fpsValue > 0 ? fpsValue : null;
    const live = readShotChain(node);
    const payload = app?.nodeOutputs?.[String(node.id)]?.omni_lingbot_sequence?.[0] ?? null;
    const description = live
      ? { shots: live, firstChunkFrames: FIRST_CHUNK_FRAMES, chunkFrames: CHUNK_FRAMES, fps: nodeFps }
      : descriptionFromPayload(payload, nodeFps);
    return {
      description,
      source: live ? "graph" : description ? "output" : null,
      linked: upstream(node, "sequence") !== null,
      width: track.clientWidth || 0,
    };
  };

  const stateSignature = (state) =>
    JSON.stringify([state.description, state.source, state.width, selected]);

  /** Put one line in the note, with the problems of the selected shot after it. */
  const renderNote = (shot) => {
    if (!shot) {
      note.textContent = "Click a shot block to read its prompt.";
      note.title = "";
      return;
    }
    const line = `Shot ${shot.index + 1} · ${shot.frames}f · ` +
      `${shot.chunks} chunk${shot.chunks === 1 ? "" : "s"} · frames ${shot.start}–${shot.end} — ` +
      `${shot.prompt || "(no prompt)"}`;
    const parts = [el("span", { textContent: line })];
    if (shot.problems.length) {
      parts.push(el("span", {
        className: "lingbot-tl-problem",
        textContent: ` · ${shot.problems.join("; ")}`,
      }));
    }
    note.replaceChildren(...parts);
    note.title = note.textContent;
  };

  function render(state) {
    const current = state ?? readState();
    const shots = current.description?.shots;
    if (selected !== null && (!Array.isArray(shots) || selected >= shots.length)) selected = null;
    const layout = current.description ? layoutSequence(current.description) : null;
    const drawable = !!layout && layout.totalFrames > 0;

    if (!drawable) {
      ruler.hidden = true;
      track.hidden = true;
      lanes.hidden = true;
      footer.hidden = true;
      note.hidden = false;
      note.textContent = current.source === null
        ? current.linked
          ? "This timeline can't read the sequence; connect VLLMOmniLingBotShot nodes to draw it."
          : "Connect a shot sequence to draw it."
        : "This sequence has no frames to draw.";
      note.title = note.textContent;
      return;
    }

    ruler.hidden = false;
    track.hidden = false;
    lanes.hidden = false;
    footer.hidden = false;
    const width = current.width || root.clientWidth || 360;
    const pct = (frame) => `${(100 * frame) / Math.max(1, layout.totalFrames)}%`;

    ruler.replaceChildren(...layout.ticks.map((tick) =>
      el("div", { className: "lingbot-tl-tick", textContent: tick.label, style: `left:${pct(tick.frame)}` })));

    const children = [playhead];
    const every = chunkGridEvery(width, layout);
    for (const chunk of layout.chunks) {
      if (!chunk.index || chunk.index % every) continue;
      children.push(el("div", { className: "lingbot-tl-grid", style: `left:${pct(chunk.start)}` }));
    }
    for (const shot of layout.shots) {
      const block = el("div", {
        className: `lingbot-tl-shot${shot.index % 2 ? " lingbot-tl-shot-alt" : ""}` +
          `${shot.index === selected ? " selected" : ""}`,
        title: `Shot ${shot.index + 1}: ${shot.frames} frames, ${shot.chunks} chunk` +
          `${shot.chunks === 1 ? "" : "s"}, frames ${shot.start}–${shot.end}\n${shot.prompt || "(no prompt)"}`,
        dataset: { shot: String(shot.index) },
        style: `left:${pct(shot.start)};width:${pct(shot.frames)}`,
      },
      el("span", { className: "lingbot-tl-shot-prompt", textContent: shot.prompt || "(no prompt)" }),
      el("span", { className: "lingbot-tl-length", textContent: `${shot.frames}f` }));
      children.push(block);
      if (shot.index > 0) {
        children.push(el("div", {
          className: "lingbot-tl-switch",
          title: `Prompt switch at chunk ${shot.startChunk}, frame ${shot.start}: ${shot.prompt}`,
          style: `left:${pct(shot.start)}`,
        }));
      }
    }
    track.replaceChildren(...children);

    if (!layout.lanes.length) {
      lanes.replaceChildren(el("div", { className: "lingbot-tl-empty", textContent: "No camera moves in this sequence." }));
    } else {
      lanes.replaceChildren(...layout.lanes.map((row) => {
        const lane = el("div", { className: "lingbot-tl-lane", title: `${row.lane} lane` },
          el("span", { className: "lingbot-tl-lane-name", textContent: row.lane }));
        for (const bar of row.bars) {
          lane.append(el("div", {
            className: "lingbot-tl-move",
            title: `${row.lane} ${bar.value} · ${bar.frames}f, shot ${bar.shot + 1}, frames ${bar.start}–${bar.end}`,
            style: `left:${pct(bar.start)};width:${pct(bar.frames)}`,
            textContent: `${bar.lane} ${bar.value} · ${bar.frames}f`,
          }));
        }
        return lane;
      }));
    }

    renderNote(selected !== null ? layout.shots[selected] : null);

    const parts = [`${layout.totalFrames} frames`];
    if (layout.seconds !== null) parts.push(`${layout.seconds.toFixed(2)} s at ${layout.fps} fps`);
    parts.push(`${layout.totalChunks} chunk${layout.totalChunks === 1 ? "" : "s"}`);
    parts.push(`${layout.shots.length} shot${layout.shots.length === 1 ? "" : "s"}`);
    if (current.source === "output") parts.push("from the last run");
    totals.textContent = parts.join(" · ");
    warningsList.replaceChildren(...layout.warnings.map((message) => el("li", { textContent: message })));
    warningsList.hidden = !layout.warnings.length;
  }

  const draw = () => {
    const state = readState();
    const next = stateSignature(state);
    if (next !== signature) {
      signature = next;
      render(state);
    }
    // Fit the node to its content once; after that a height the user dragged is
    // theirs, and only the widget's own minimum tracks the measured height.
    const measured = contentHeight(root);
    if (measured !== null && measured !== height) {
      height = measured;
      if (!laidOut) {
        laidOut = true;
        node.expandToFitContent?.();
      }
    }
  };

  widget = node.addDOMWidget("timeline", "LINGBOT_SEQUENCE_VIEW", root, {
    serialize: false,
    getMinHeight: () => (height ? height + 2 * (widget?.margin ?? 10) : 0),
    // The Parameters panel draws a widget by setting its width to the panel's
    // and never restores it.
    hideInPanel: true,
    onDraw: () => draw(),
  });

  // Clicking a shot selects it; clicking anywhere else clears the selection.
  root.addEventListener("pointerdown", (event) => {
    const block = event.target?.closest?.(".lingbot-tl-shot") ?? null;
    const index = block ? Number(block.dataset.shot) : null;
    const next = Number.isInteger(index) ? index : null;
    if (next === selected) return;
    selected = next;
    signature = "";
    draw();
  });

  // A `VLLMOmniLingBotRender` of this same chain reports frames rendered over
  // frames planned; the planned length is this ruler's length, so the ratio
  // places the playhead.
  const rendersThis = (id) => {
    const render = node.graph?.getNodeById?.(id) ?? null;
    if (render?.comfyClass !== "VLLMOmniLingBotRender") return false;
    const chain = upstream(node, "sequence");
    return !!chain && upstream(render, "sequence") === chain;
  };
  const onProgress = ({ detail }) => {
    if (!detail || !rendersThis(detail.node)) return;
    const max = Number(detail.max);
    if (!(max > 0)) return;
    const fraction = Math.min(1, Math.max(0, Number(detail.value) / max));
    playhead.hidden = false;
    playhead.style.left = `${fraction * 100}%`;
  };
  const onStop = () => {
    playhead.hidden = true;
  };
  api.addEventListener("progress", onProgress);
  api.addEventListener("executing", onStop);
  api.addEventListener("execution_interrupted", onStop);
  const removed = widget.onRemove;
  widget.onRemove = function () {
    api.removeEventListener("progress", onProgress);
    api.removeEventListener("executing", onStop);
    api.removeEventListener("execution_interrupted", onStop);
    removed?.call(this);
  };

  render();
  return widget;
}

/* ------------------------------------------------------------------------- */
/* Registration                                                              */
/* ------------------------------------------------------------------------- */

/** Install the extension on a ComfyUI `app`; kept separate so tests never run it. */
export function register(app) {
  if (!app || typeof app.registerExtension !== "function") return;
  app.registerExtension({
    name: "Omni.LingBotTimeline",
    setup() {
      ensureStyle();
    },
    nodeCreated(node) {
      if (node?.comfyClass !== "VLLMOmniLingBotVisualizer") return;
      // An older ComfyUI without DOM widgets simply gets no in-node viewer.
      if (typeof node.addDOMWidget !== "function") return;
      if (node.widgets?.some((entry) => entry?.type === "LINGBOT_SEQUENCE_VIEW")) return;
      createTimelineWidget(node, app);
    },
  });
}

// ComfyUI imports every file in `WEB_DIRECTORY` itself, so the viewer has to
// register here; the guard keeps the module importable in Node, where neither
// ComfyUI's `app` nor a DOM exists.
if (typeof window !== "undefined" && typeof document !== "undefined") {
  const comfy = globalThis.app;
  if (comfy?.registerExtension) register(comfy);
  else {
    import("../../scripts/app.js")
      .then((module) => register(module.app))
      .catch((error) => console.warn("vLLM-Omni: could not register the LingBot timeline.", error));
  }
}

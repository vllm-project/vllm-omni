// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

/**
 * @file Timeline for `VLLMOmniLingBotVisualizer` ("LingBot World Visualizer").
 *
 * The Python node returns `{"ui": {"omni_lingbot_sequence": [payload]}}`, so
 * after a run the browser reads
 * `app.nodeOutputs[String(node.id)].omni_lingbot_sequence[0]`. This module
 * draws that payload: a frame ruler, the `9 + 12k` chunk grid, one block per
 * shot with the prompt switch at its start, and one row per camera lane that
 * holds moves.
 *
 * `layoutSequence` is pure — no DOM, no `window` — so the geometry can be
 * tested in Node. The panel itself is a `<dialog>` and is only built by
 * `openTimeline`. `register(app)` installs the extension; the guarded call at
 * the bottom is what ComfyUI's automatic import of `WEB_DIRECTORY` files runs,
 * and it does nothing in Node.
 */

/** Lane rows are drawn in this order, whatever order the payload lists them. */
const LANE_ORDER = ["move", "strafe", "look_v", "look_h"];

/** Tick spacing is chosen for at most this many labels across the timeline. */
const MAX_TICKS = 10;

/** A payload cannot ask the layout to iterate more chunks than this. */
const MAX_CHUNKS = 20000;

/** Grid lines are thinned to stay at least this far apart on screen. */
const MIN_CHUNK_GAP_PX = 6;

/** Matches `LingBotGeometry`'s defaults, used only when a field is missing. */
const DEFAULT_GEOMETRY = {
  firstChunkFrames: 9,
  chunkFrames: 12,
  latentFramesPerChunk: 3,
  temporalScale: 4,
};

const FONT = "12px system-ui, -apple-system, 'Segoe UI', sans-serif";
const FONT_STRONG = "600 12px system-ui, -apple-system, 'Segoe UI', sans-serif";
const FONT_SMALL = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";

/**
 * A light, theme-independent palette. The canvas paints its own background, so
 * the drawing reads the same on ComfyUI's dark and light themes.
 */
const COLORS = {
  background: "#f7f8fb",
  text: "#232a36",
  muted: "#5b6270",
  ruler: "#c3c9d6",
  grid: "#d5dae4",
  shotFill: ["#e7ecf5", "#dde5f1"],
  shotStroke: "#93a1b8",
  switchLine: "#a4581e",
  row: "#eef2f8",
  barText: "#ffffff",
  laneFallback: "#6b7280",
  lanes: {
    move: "#4d6a91",
    strafe: "#4d7a68",
    look_v: "#75628a",
    look_h: "#8a6a4a",
  },
};

const isFiniteNumber = (value) => typeof value === "number" && Number.isFinite(value);
const positiveInt = (value) => (isFiniteNumber(value) && value > 0 ? Math.floor(value) : 0);
const asText = (value) => (typeof value === "string" ? value : "");
const asList = (value) => (Array.isArray(value) ? value : []);

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

/**
 * First pixel frame of `chunk`. Chunk 0 is the short opening chunk
 * (`first_chunk_frames`, 9), every later one adds `chunk_frames` (12).
 */
function chunkStartFrame(geometry, chunk) {
  if (chunk <= 0) return 0;
  return geometry.firstChunkFrames + (chunk - 1) * geometry.chunkFrames;
}

/** Smallest legal chunk count for `frames`, used only when `total_chunks` is absent. */
function chunksForFrames(geometry, frames) {
  if (frames <= 0) return 0;
  if (frames <= geometry.firstChunkFrames) return 1;
  return 1 + Math.ceil((frames - geometry.firstChunkFrames) / geometry.chunkFrames);
}

/**
 * The smallest 1-2-5 step that is at least `target`, so a ruler never needs
 * more than `MAX_TICKS + 1` labelled frames.
 */
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

/** Second labels from `fps`; frame numbers only when the payload has no rate. */
function formatTick(frame, fps) {
  return fps ? `${formatNumber(frame / fps)}s` : `${formatNumber(frame)}f`;
}

function buildTicks(totalFrames, fps) {
  const step = niceStep(totalFrames / MAX_TICKS);
  if (!step) return { step: null, ticks: [] };
  const count = Math.min(Math.floor(totalFrames / step) + 1, MAX_TICKS + 2);
  const ticks = [];
  for (let index = 0; index < count; index += 1) {
    const frame = Math.round(index * step * 1e6) / 1e6;
    ticks.push({ frame, label: formatTick(frame, fps) });
  }
  return { step, ticks };
}

/**
 * Everything needed to draw a sequence, in absolute frame numbers.
 *
 * - `shots[i]`: `{index, prompt, start, end, frames, startChunk, chunkEnd, chunks}`,
 *   where `start`/`end` are the block's x span in frames and `startChunk`/
 *   `chunkEnd` its span on the chunk grid.
 * - `bars`: `{shot, lane, value, start, end, frames, label, row}`, absolute
 *   frames (`shot.start + move.start_frame`), clipped to the shot's end.
 * - `lanes`: one row per lane that carries moves, in `LANE_ORDER`.
 * - `ticks`/`tickStep`: ruler frames and the 1-2-5 spacing between them.
 * - `chunks`: `{index, start, end, frames}` for the whole chunk grid.
 * - `boundaries`: `{chunk, frame, prompt}` for each prompt switch.
 * - `warnings`: the payload's warnings, verbatim.
 *
 * Pure: no DOM and no `window`, and it accepts a missing or partial payload.
 */
export function layoutSequence(payload) {
  const source = payload && typeof payload === "object" ? payload : {};
  const geometry = normalizeGeometry(source.geometry);
  const fps = isFiniteNumber(source.fps) && source.fps > 0 ? source.fps : null;

  const shots = [];
  const bars = [];
  const lanes = new Map();
  let frameCursor = 0;
  let chunkCursor = 0;

  for (const raw of asList(source.shots)) {
    if (!raw || typeof raw !== "object") continue;
    const frames = positiveInt(raw.frames);
    const start = isFiniteNumber(raw.start_frame) ? Math.max(0, Math.floor(raw.start_frame)) : frameCursor;
    const end = start + frames;
    const startChunk = isFiniteNumber(raw.start_chunk)
      ? Math.max(0, Math.floor(raw.start_chunk))
      : chunkCursor;
    const chunks = positiveInt(raw.chunks) ||
      (isFiniteNumber(raw.end_chunk) ? Math.max(0, Math.floor(raw.end_chunk) - startChunk) : 0);
    const index = shots.length;
    shots.push({
      index,
      prompt: asText(raw.prompt),
      start,
      end,
      frames,
      startChunk,
      chunkEnd: startChunk + chunks,
      chunks,
    });

    for (const move of asList(raw.moves)) {
      if (!move || typeof move !== "object") continue;
      const lane = asText(move.lane) || "move";
      const value = asText(move.value);
      const length = positiveInt(move.frames);
      const offset = isFiniteNumber(move.start_frame) ? Math.max(0, Math.floor(move.start_frame)) : 0;
      const barStart = Math.min(end, start + offset);
      const barEnd = Math.min(end, barStart + length);
      if (barEnd <= barStart) continue;
      const bar = {
        shot: index,
        lane,
        value,
        start: barStart,
        end: barEnd,
        frames: barEnd - barStart,
        label: `${lane} ${value}`.trim(),
        row: 0,
      };
      bars.push(bar);
      if (!lanes.has(lane)) lanes.set(lane, { lane, bars: [] });
      lanes.get(lane).bars.push(bar);
    }
    frameCursor = end;
    chunkCursor = startChunk + chunks;
  }

  const shotSpan = shots.length ? shots[shots.length - 1].end : 0;
  const totalFrames = Math.max(positiveInt(source.total_frames), shotSpan);
  let totalChunks = positiveInt(source.total_chunks) || chunkCursor;
  if (!totalChunks && totalFrames > 0) totalChunks = chunksForFrames(geometry, totalFrames);
  totalChunks = Math.min(totalChunks, MAX_CHUNKS);

  const rank = (lane) => {
    const index = LANE_ORDER.indexOf(lane);
    return index === -1 ? LANE_ORDER.length : index;
  };
  const laneRows = [...lanes.values()].sort((left, right) => rank(left.lane) - rank(right.lane));
  laneRows.forEach((row, index) => {
    row.row = index;
    for (const bar of row.bars) bar.row = index;
  });

  const chunks = [];
  for (let index = 0; index < totalChunks; index += 1) {
    const start = chunkStartFrame(geometry, index);
    const frames = index === 0 ? geometry.firstChunkFrames : geometry.chunkFrames;
    chunks.push({ index, start, end: start + frames, frames });
  }

  const boundaries = [];
  for (const entry of asList(source.prompt_switches)) {
    if (!Array.isArray(entry) || !isFiniteNumber(entry[0])) continue;
    const chunk = Math.max(0, Math.floor(entry[0]));
    boundaries.push({ chunk, frame: chunkStartFrame(geometry, chunk), prompt: asText(entry[1]) });
  }
  boundaries.sort((left, right) => left.frame - right.frame);

  const { step, ticks } = buildTicks(totalFrames, fps);

  return {
    fps,
    geometry,
    totalFrames,
    totalChunks,
    seconds: fps ? totalFrames / fps : null,
    shots,
    lanes: laneRows,
    bars,
    chunks,
    boundaries,
    ticks,
    tickStep: step,
    warnings: asList(source.warnings).filter((warning) => typeof warning === "string"),
    empty: shots.length === 0,
  };
}

/* ------------------------------------------------------------------------- */
/* Drawing                                                                   */
/* ------------------------------------------------------------------------- */

function pathRoundedRect(ctx, x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

/** Truncate with an ellipsis to what fits in `maxWidth`, measured in the ctx font. */
function fitText(ctx, value, maxWidth) {
  if (maxWidth <= 0) return "";
  if (ctx.measureText(value).width <= maxWidth) return value;
  let low = 0;
  let high = value.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (ctx.measureText(`${value.slice(0, middle)}…`).width <= maxWidth) low = middle;
    else high = middle - 1;
  }
  return low > 0 ? `${value.slice(0, low)}…` : "";
}

/** Vertical bands of the canvas: ruler, shot blocks, then one row per lane. */
function timelineGeometry(width, layout) {
  const pad = 16;
  const gutter = 96;
  const rulerHeight = 30;
  const shotTop = pad + rulerHeight + 16;
  const shotHeight = 48;
  const laneTop = shotTop + shotHeight + 20;
  const rowHeight = 28;
  const rowGap = 6;
  const plotBottom = laneTop + layout.lanes.length * (rowHeight + rowGap);
  return {
    pad,
    gutter,
    rulerHeight,
    shotTop,
    shotHeight,
    laneTop,
    rowHeight,
    rowGap,
    plotLeft: pad + gutter,
    plotRight: Math.max(pad + gutter + 60, width - pad),
    plotBottom,
    height: Math.max(200, plotBottom + pad),
  };
}

function describe(layout) {
  const parts = [
    `${layout.totalFrames} frames`,
    `${layout.totalChunks} chunks`,
    `${layout.shots.length} shots`,
  ];
  if (layout.fps) parts.push(`${layout.seconds.toFixed(2)} seconds at ${layout.fps} fps`);
  if (layout.lanes.length) parts.push(`camera lanes: ${layout.lanes.map((row) => row.lane).join(", ")}`);
  if (layout.warnings.length) parts.push(`${layout.warnings.length} warnings`);
  return parts.join(", ");
}

function drawRuler(ctx, layout, geometry, toX) {
  const baseline = geometry.pad + geometry.rulerHeight;
  ctx.strokeStyle = COLORS.ruler;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(geometry.plotLeft, baseline + 0.5);
  ctx.lineTo(geometry.plotRight, baseline + 0.5);
  ctx.stroke();
  ctx.font = FONT;
  ctx.fillStyle = COLORS.muted;
  for (const tick of layout.ticks) {
    const x = Math.round(toX(tick.frame));
    if (x < geometry.plotLeft - 1 || x > geometry.plotRight + 1) continue;
    const half = ctx.measureText(tick.label).width / 2;
    ctx.textAlign = x - half < geometry.plotLeft ? "left" : x + half > geometry.plotRight ? "right" : "center";
    ctx.fillText(tick.label, x, baseline - 8);
    ctx.beginPath();
    ctx.moveTo(x + 0.5, baseline);
    ctx.lineTo(x + 0.5, baseline + 4);
    ctx.stroke();
  }
}

function drawChunkGrid(ctx, layout, geometry, toX) {
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  let previous = Number.NEGATIVE_INFINITY;
  for (const chunk of layout.chunks) {
    const x = Math.round(toX(chunk.start)) + 0.5;
    // Thin dense grids out; the layout keeps every chunk, the canvas does not.
    if (x - previous < MIN_CHUNK_GAP_PX) continue;
    previous = x;
    ctx.moveTo(x, geometry.shotTop - 6);
    ctx.lineTo(x, geometry.plotBottom);
  }
  const right = Math.round(geometry.plotRight) + 0.5;
  ctx.moveTo(right, geometry.shotTop - 6);
  ctx.lineTo(right, geometry.plotBottom);
  ctx.stroke();
}

function drawShots(ctx, layout, geometry, toX) {
  layout.shots.forEach((shot, index) => {
    const left = toX(shot.start);
    const width = Math.max(2, toX(shot.end) - left);
    const top = geometry.shotTop;
    ctx.save();
    pathRoundedRect(ctx, left, top, width, geometry.shotHeight, 5);
    ctx.fillStyle = COLORS.shotFill[index % COLORS.shotFill.length];
    ctx.fill();
    ctx.strokeStyle = COLORS.shotStroke;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.clip();
    const inner = width - 16;
    ctx.textAlign = "left";
    if (inner > 8) {
      ctx.fillStyle = COLORS.text;
      ctx.font = FONT_STRONG;
      ctx.fillText(
        `${shot.chunks} chunk${shot.chunks === 1 ? "" : "s"} · ${shot.frames} frames`,
        left + 8,
        top + 16,
      );
      ctx.fillStyle = COLORS.muted;
      ctx.font = FONT;
      ctx.fillText(fitText(ctx, shot.prompt || "(no prompt)", inner), left + 8, top + 34);
    }
    ctx.restore();
  });
}

function drawBoundaries(ctx, layout, geometry, toX) {
  const baseline = geometry.pad + geometry.rulerHeight;
  for (const boundary of layout.boundaries) {
    const x = Math.round(toX(boundary.frame)) + 0.5;
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = COLORS.switchLine;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, baseline + 2);
    ctx.lineTo(x, geometry.plotBottom);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = COLORS.switchLine;
    ctx.beginPath();
    ctx.moveTo(x - 4, baseline + 1);
    ctx.lineTo(x + 4, baseline + 1);
    ctx.lineTo(x, baseline + 9);
    ctx.closePath();
    ctx.fill();
  }
}

function drawBar(ctx, bar, top, geometry, toX) {
  const left = Math.max(geometry.plotLeft, toX(bar.start));
  const right = Math.min(geometry.plotRight, toX(bar.end));
  const width = Math.max(3, right - left);
  const height = geometry.rowHeight - 8;
  const middle = top + 4 + height / 2;
  pathRoundedRect(ctx, left, top + 4, width, height, 3);
  ctx.fillStyle = COLORS.lanes[bar.lane] ?? COLORS.laneFallback;
  ctx.fill();
  ctx.font = FONT_SMALL;
  ctx.textAlign = "left";
  const full = `${bar.lane} ${bar.value} · ${bar.frames}f`;
  const compact = `${bar.value} ${bar.frames}f`;
  if (ctx.measureText(full).width + 10 <= width) {
    ctx.fillStyle = COLORS.barText;
    ctx.fillText(full, left + 6, middle);
  } else if (ctx.measureText(compact).width + 10 <= width) {
    ctx.fillStyle = COLORS.barText;
    ctx.fillText(compact, left + 6, middle);
  } else {
    ctx.fillStyle = COLORS.muted;
    if (right + 6 + ctx.measureText(full).width <= geometry.plotRight) {
      ctx.fillText(full, right + 6, middle);
    } else {
      ctx.textAlign = "right";
      ctx.fillText(full, left - 6, middle);
    }
  }
}

function drawLaneRows(ctx, layout, geometry, toX) {
  if (!layout.lanes.length) {
    ctx.font = FONT;
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "left";
    ctx.fillText("No camera moves in this sequence.", geometry.plotLeft, geometry.laneTop + 8);
    return;
  }
  layout.lanes.forEach((row, index) => {
    const top = geometry.laneTop + index * (geometry.rowHeight + geometry.rowGap);
    pathRoundedRect(ctx, geometry.plotLeft, top, geometry.plotRight - geometry.plotLeft, geometry.rowHeight, 4);
    ctx.fillStyle = COLORS.row;
    ctx.fill();
    ctx.font = FONT;
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "right";
    ctx.fillText(row.lane, geometry.plotLeft - 10, top + geometry.rowHeight / 2);
    for (const bar of row.bars) drawBar(ctx, bar, top, geometry, toX);
  });
}

/**
 * One transparent hit area per shot block, so the full prompt of a truncated
 * label is reachable as a native `title` tooltip (the canvas cannot carry one).
 */
function buildShotOverlay(overlay, layout, geometry, toX) {
  overlay.replaceChildren();
  for (const shot of layout.shots) {
    const left = toX(shot.start);
    const hit = document.createElement("div");
    hit.style.cssText = `position:absolute;left:${left}px;top:${geometry.shotTop}px;` +
      `width:${Math.max(2, toX(shot.end) - left)}px;height:${geometry.shotHeight}px;` +
      "pointer-events:auto;border-radius:5px";
    hit.title = shot.prompt || "Shot without a prompt";
    hit.setAttribute(
      "aria-label",
      `Shot ${shot.index + 1}: ${shot.chunks} chunks, ${shot.frames} frames from frame ` +
        `${shot.start}. ${shot.prompt}`,
    );
    overlay.append(hit);
  }
}

/** Size the canvas for its dialog, then draw the whole timeline on it. */
function drawTimeline(canvas, overlay, layout) {
  const width = Math.max(320, Math.round(canvas.clientWidth || 720));
  const geometry = timelineGeometry(width, layout);
  const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
  canvas.style.height = `${geometry.height}px`;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(geometry.height * ratio);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.fillStyle = COLORS.background;
  ctx.fillRect(0, 0, width, geometry.height);
  ctx.textBaseline = "middle";
  ctx.font = FONT;
  canvas.setAttribute("aria-label", describe(layout));
  if (!layout.totalFrames) {
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "center";
    ctx.fillText("This sequence has no frames to draw.", width / 2, geometry.height / 2);
    overlay.replaceChildren();
    return;
  }
  const toX = (frame) =>
    geometry.plotLeft + (frame / layout.totalFrames) * (geometry.plotRight - geometry.plotLeft);
  drawRuler(ctx, layout, geometry, toX);
  drawChunkGrid(ctx, layout, geometry, toX);
  drawShots(ctx, layout, geometry, toX);
  drawBoundaries(ctx, layout, geometry, toX);
  drawLaneRows(ctx, layout, geometry, toX);
  buildShotOverlay(overlay, layout, geometry, toX);
}

/* ------------------------------------------------------------------------- */
/* Panel                                                                     */
/* ------------------------------------------------------------------------- */

let closeTimeline = null;

function openTimeline(node, comfy) {
  closeTimeline?.();
  const dialog = document.createElement("dialog");
  dialog.setAttribute("aria-label", "LingBot World timeline");
  dialog.style.cssText =
    "width:min(1120px,94vw);max-height:92vh;overflow:auto;background:#f4f6fa;color:#1f2430;" +
    "border:1px solid #b9c0cd;border-radius:12px;padding:18px;" +
    "font:14px/1.45 system-ui,-apple-system,'Segoe UI',sans-serif";
  // Static markup only: prompts, totals and warnings are set with textContent.
  dialog.innerHTML = `
    <h2 style="margin:0 0 4px;font-size:17px">LingBot World timeline</h2>
    <p style="margin:0 0 12px;color:#4a5464;font-size:13px">
      Shots in playback order, the chunk grid, and the camera lane each move holds.</p>
    <p data-empty style="margin:0;padding:18px;border:1px dashed #aeb7c6;border-radius:8px;color:#3d4655">
      This node has not run yet, so there is no sequence to draw. Queue the workflow and reopen this panel;
      it shows only what the node returned.</p>
    <div data-body style="display:none;position:relative">
      <canvas data-canvas role="img" style="display:block;width:100%;border:1px solid #c7cedb;border-radius:8px">
      </canvas>
      <div data-overlay style="position:absolute;inset:0;pointer-events:none"></div>
    </div>
    <footer style="margin-top:12px;padding-top:10px;border-top:1px solid #d5dae4;font-size:13px">
      <p data-totals style="margin:0 0 6px;color:#2b3442"></p>
      <p data-warnings-title style="margin:0 0 4px;color:#4a5464;display:none">Warnings</p>
      <ul data-warnings style="margin:0;padding-left:18px;color:#7a3d17"></ul>
    </footer>
    <div style="display:flex;justify-content:flex-end;margin-top:12px">
      <button data-close style="background:#e3e8f0;border:1px solid #aab3c2;border-radius:6px;
        padding:6px 14px;color:#1f2430;cursor:pointer">Close</button>
    </div>`;
  document.body.append(dialog);

  const body = dialog.querySelector("[data-body]");
  const empty = dialog.querySelector("[data-empty]");
  const canvas = dialog.querySelector("[data-canvas]");
  const overlay = dialog.querySelector("[data-overlay]");
  const totals = dialog.querySelector("[data-totals]");
  const warningsTitle = dialog.querySelector("[data-warnings-title]");
  const warningsList = dialog.querySelector("[data-warnings]");
  const listeners = new AbortController();
  const options = { signal: listeners.signal };
  let layout = null;
  let payload = null;
  let measuredWidth = 0;
  let timer = null;

  const currentPayload = () => comfy?.nodeOutputs?.[String(node.id)]?.omni_lingbot_sequence?.[0] ?? null;

  const showEmpty = () => {
    body.style.display = "none";
    empty.style.display = "";
    totals.textContent = "";
    warningsTitle.style.display = "none";
    warningsList.replaceChildren();
  };

  const updateFooter = (value) => {
    const parts = [`${value.totalFrames} frames`];
    if (value.seconds !== null) parts.push(`${value.seconds.toFixed(2)} s at ${value.fps} fps`);
    parts.push(`${value.totalChunks} chunk${value.totalChunks === 1 ? "" : "s"}`);
    parts.push(`${value.shots.length} shot${value.shots.length === 1 ? "" : "s"}`);
    totals.textContent = parts.join(" · ");
    warningsTitle.style.display = value.warnings.length ? "" : "none";
    warningsList.replaceChildren(
      ...value.warnings.map((message) => {
        const item = document.createElement("li");
        item.textContent = message;
        return item;
      }),
    );
  };

  const draw = () => {
    if (!layout) {
      showEmpty();
      return;
    }
    body.style.display = "";
    empty.style.display = "none";
    measuredWidth = canvas.clientWidth || body.clientWidth || 0;
    drawTimeline(canvas, overlay, layout);
    updateFooter(layout);
  };

  const refresh = () => {
    if (!node.graph || node.graph.getNodeById?.(node.id) !== node) {
      close();
      return;
    }
    const next = currentPayload();
    if (next === payload) return;
    payload = next;
    layout = next ? layoutSequence(next) : null;
    if (layout) draw();
    else showEmpty();
  };

  const observer = typeof ResizeObserver === "function"
    ? new ResizeObserver(() => {
        const width = canvas.clientWidth;
        if (width && Math.abs(width - measuredWidth) > 0.5) draw();
      })
    : null;
  observer?.observe(body);
  window.addEventListener("resize", draw, options);

  const close = () => {
    clearInterval(timer);
    observer?.disconnect();
    listeners.abort();
    dialog.remove();
    if (closeTimeline === close) closeTimeline = null;
  };
  closeTimeline = close;

  dialog.querySelector("[data-close]").addEventListener("click", close, options);
  dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); }, options);
  window.addEventListener("pagehide", close, options);
  dialog.showModal();
  showEmpty();
  // The node may not have run when the panel opens, so poll until it has.
  timer = setInterval(refresh, 400);
  refresh();
}

/** Install the extension on a ComfyUI `app`; kept separate so tests never run it. */
export function register(app) {
  if (!app || typeof app.registerExtension !== "function") return;
  app.registerExtension({
    name: "Omni.LingBotTimeline",
    nodeCreated(node) {
      if (node?.comfyClass !== "VLLMOmniLingBotVisualizer") return;
      node.addWidget("button", "Open timeline", null, () => openTimeline(node, app), { serialize: false });
    },
    beforeConfigureGraph() {
      closeTimeline?.();
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
      .catch((error) => console.warn("vLLM-Omni: could not register the LingBot timeline.", error));
  }
}

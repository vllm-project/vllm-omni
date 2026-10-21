// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";

// Load the browser ES module the same way lingbot_player.test.mjs does. The
// module imports ComfyUI's `api` for the playhead, and a `data:` URL cannot
// resolve a relative specifier, so this test swaps that one import for a no-op
// stand-in before loading it; everything else is the file the browser runs.
// Registration is guarded on `window`/`document`, so the import also proves
// that layoutSequence touches no DOM.
const API_IMPORT = 'import { api } from "../../scripts/api.js";';
const API_STUB = "const api = { addEventListener() {}, removeEventListener() {} };";
const raw = (await readFile(
  new URL("../../../../apps/ComfyUI-vLLM-Omni/web/lingbot-timeline.js", import.meta.url),
)).toString();
const stubbed = raw.replace(API_IMPORT, API_STUB);
if (stubbed === raw) {
  throw new Error("lingbot-timeline.js no longer imports ComfyUI's api module; update the test stub.");
}
const { layoutSequence, register } = await import(
  `data:text/javascript;base64,${Buffer.from(stubbed).toString("base64")}`
);

const PROMPTS = [
  "A quiet forest path in daylight.",
  "The path opens onto a windswept ridge.",
  "Dusk falls over the ridge.",
];

// 45 + 84 + 60 = 189 frames = 9 + 12 * 15 frames = 16 chunks.
// Shot 1 holds chunks 0-3, shot 2 chunks 4-10, shot 3 chunks 11-15.
const SHOTS = [
  {
    prompt: PROMPTS[0],
    frames: 45,
    moves: [{ lane: "move", value: "forward", start_frame: 0, frames: 45 }],
  },
  {
    prompt: PROMPTS[1],
    frames: 84,
    moves: [
      // Starts at shot 2's own frame 0 and holds the first two chunks.
      { lane: "move", value: "forward", start_frame: 0, frames: 24 },
      // Starts 24 frames into shot 2, not at its start.
      { lane: "look_h", value: "right", start_frame: 24, frames: 60 },
    ],
  },
  {
    prompt: PROMPTS[2],
    frames: 60,
    moves: [{ lane: "look_v", value: "up", start_frame: 12, frames: 36 }],
  },
];

function fixture(overrides = {}) {
  return {
    shots: SHOTS,
    firstChunkFrames: 9,
    chunkFrames: 12,
    fps: 16,
    ...overrides,
  };
}

test("shot blocks span the sequence on both the frame and chunk axes", () => {
  const layout = layoutSequence(fixture());
  assert.deepEqual(
    layout.shots.map((shot) => [shot.index, shot.start, shot.end, shot.startChunk, shot.chunks]),
    [
      [0, 0, 45, 0, 4],
      [1, 45, 129, 4, 7],
      [2, 129, 189, 11, 5],
    ],
  );
  assert.equal(layout.totalFrames, 189);
  assert.equal(layout.totalChunks, 16);
  assert.equal(layout.seconds, 189 / 16);
  assert.deepEqual(layout.shots.map((shot) => shot.prompt), PROMPTS);
  assert.equal(layout.fps, 16);
  assert.equal(layout.firstChunkFrames, 9);
  assert.equal(layout.chunkFrames, 12);
  assert.equal(layout.empty, false);
  // Each prompt switch opens a shot after the first one.
  assert.deepEqual(
    layout.switches.map((entry) => [entry.shot, entry.chunk, entry.frame]),
    [
      [1, 4, 45],
      [2, 11, 129],
    ],
  );
  assert.deepEqual(layout.switches.map((entry) => entry.prompt), [PROMPTS[1], PROMPTS[2]]);
});

test("move bars are absolute, mid-shot moves included, and clipped to their shot", () => {
  const layout = layoutSequence(fixture());
  // Row order follows LANE_ORDER, which is the order the Python LANES dict uses.
  assert.deepEqual(layout.lanes.map((row) => row.lane), ["move", "look_v", "look_h"]);
  assert.deepEqual(layout.lanes.map((row) => row.row), [0, 1, 2]);
  assert.deepEqual(
    layout.bars.map((bar) => [bar.lane, bar.value, bar.start, bar.end, bar.frames]),
    [
      ["move", "forward", 0, 45, 45],
      ["move", "forward", 45, 69, 24],
      ["look_h", "right", 69, 129, 60],
      ["look_v", "up", 141, 177, 36],
    ],
  );
  // Each bar knows the row it is drawn on.
  assert.deepEqual(layout.bars.map((bar) => bar.row), [0, 0, 2, 1]);
  assert.deepEqual(layout.lanes.map((row) => row.bars.length), [2, 1, 1]);
  // The look_h move starts 24 frames into shot 2, not at its start.
  assert.equal(layout.bars[2].shot, 1);
  assert.equal(layout.bars[2].start, layout.shots[1].start + 24);
  assert.ok(layout.bars[2].start > layout.shots[1].start);

  const clipped = layoutSequence({
    shots: [{ prompt: "Short shot.", frames: 45, moves: [
      { lane: "move", value: "back", start_frame: 30, frames: 60 },
    ] }],
    firstChunkFrames: 9,
    chunkFrames: 12,
  });
  assert.deepEqual(clipped.bars.map((bar) => [bar.start, bar.end, bar.frames]), [[30, 45, 15]]);
  assert.equal(clipped.bars[0].end, clipped.shots[0].end);
});

test("ruler ticks use one 1-2-5 step and stay near eight labels", () => {
  const layout = layoutSequence(fixture());
  assert.ok(layout.ticks.length >= 3 && layout.ticks.length <= 8, `${layout.ticks.length} ticks`);
  assert.equal(layout.ticks[0].frame, 0);
  assert.ok(layout.ticks.at(-1).frame <= layout.totalFrames);
  const deltas = layout.ticks.slice(1).map((tick, index) => tick.frame - layout.ticks[index].frame);
  assert.ok(deltas.length > 0);
  for (const delta of deltas) {
    assert.ok(Math.abs(delta - layout.tickStep) < 1e-9, `tick step ${delta} vs ${layout.tickStep}`);
  }
  const magnitude = 10 ** Math.floor(Math.log10(layout.tickStep));
  const multiple = layout.tickStep / magnitude;
  assert.ok([1, 2, 5].some((candidate) => Math.abs(multiple - candidate) < 1e-9), `${layout.tickStep}`);
  // Second labels come from fps: every label is its frame over 16, to 3 places.
  for (const tick of layout.ticks) {
    assert.equal(tick.label, `${Number((tick.frame / 16).toFixed(3))}s`);
  }
});

test("chunk grid is the 9 + 12k sequence", () => {
  const layout = layoutSequence(fixture());
  assert.equal(layout.chunks.length, 16);
  assert.deepEqual(layout.chunks[0], { index: 0, start: 0, end: 9, frames: 9 });
  assert.deepEqual(layout.chunks[1], { index: 1, start: 9, end: 21, frames: 12 });
  assert.equal(layout.chunks.at(-1).end, layout.totalFrames);
  for (const chunk of layout.chunks) {
    assert.equal(chunk.end - chunk.start, chunk.frames);
    assert.equal(chunk.frames, chunk.index === 0 ? 9 : 12);
    assert.equal(chunk.start, chunk.index === 0 ? 0 : 9 + (chunk.index - 1) * 12);
  }
});

test("one shot without moves still lays out, and an empty description is harmless", () => {
  // Only the fields the viewer always has: geometry defaults to 9 + 12k.
  const single = layoutSequence({ shots: [{ prompt: "One short shot.", frames: 9 }], fps: null });
  assert.deepEqual(single.lanes, []);
  assert.deepEqual(single.bars, []);
  assert.deepEqual(single.chunks, [{ index: 0, start: 0, end: 9, frames: 9 }]);
  assert.equal(single.totalChunks, 1);
  assert.equal(single.seconds, null);
  assert.deepEqual(single.switches, []);
  assert.equal(single.shots[0].start, 0);
  assert.equal(single.shots[0].end, 9);
  assert.equal(single.shots[0].chunks, 1);
  assert.ok(single.ticks.length > 0);
  assert.ok(single.ticks.every((tick) => tick.label.endsWith("f")), "frames-only ruler without fps");
  assert.equal(single.ticks[0].label, "0f");

  const empty = layoutSequence(undefined);
  assert.deepEqual(empty.shots, []);
  assert.deepEqual(empty.bars, []);
  assert.deepEqual(empty.lanes, []);
  assert.deepEqual(empty.ticks, []);
  assert.deepEqual(empty.chunks, []);
  assert.deepEqual(empty.switches, []);
  assert.equal(empty.totalFrames, 0);
  assert.equal(empty.totalChunks, 0);
  assert.equal(empty.seconds, null);
  assert.equal(empty.empty, true);
});

test("moves are reported when a shot cuts them short or they miss a latent frame", () => {
  const cut = "Shot 0: the strafe move is cut to 15 frames at the shot's end.";
  const widened = "Shot 1: the look_h move is widened to one latent frame (4 pixel frames).";
  const warned = layoutSequence({
    shots: [
      { prompt: "A move that runs past the shot.", frames: 45, moves: [
        { lane: "strafe", value: "left", start_frame: 30, frames: 60 },
      ] },
      { prompt: "A move shorter than one latent frame.", frames: 12, moves: [
        { lane: "look_h", value: "right", start_frame: 0, frames: 2 },
      ] },
    ],
  });
  assert.deepEqual(warned.warnings, [cut, widened]);
  assert.deepEqual(warned.shots[0].problems, [cut]);
  assert.deepEqual(warned.shots[1].problems, [widened]);
  // The bar shows the length the move is actually cut to, not the widened one.
  assert.deepEqual(warned.bars.map((bar) => [bar.start, bar.end, bar.frames]), [
    [30, 45, 15],
    [45, 47, 2],
  ]);
});

test("missing fields fall back to the 9 + 12k defaults instead of throwing", () => {
  const sparse = layoutSequence({ shots: [{ prompt: "No frames." }, { chunks: 2 }] });
  // A shot without frames or chunks holds nothing and takes no space.
  assert.equal(sparse.shots[0].frames, 0);
  assert.equal(sparse.shots[0].chunks, 0);
  // A shot after the first is whole 12-frame chunks, so 2 chunks is 24 frames.
  assert.equal(sparse.shots[1].chunks, 2);
  assert.equal(sparse.shots[1].frames, 24);
  assert.equal(sparse.shots[1].start, 0);
  assert.equal(sparse.totalFrames, 24);
  assert.equal(sparse.totalChunks, 2);
  assert.equal(sparse.empty, false);
});

test("the module still exports layoutSequence and a guarded register", () => {
  assert.equal(typeof layoutSequence, "function");
  assert.equal(typeof register, "function");
  const installed = [];
  register({ registerExtension: (extension) => installed.push(extension) });
  assert.equal(installed.length, 1);
  assert.equal(installed[0].name, "vllm.lingbot_timeline");
  assert.equal(typeof installed[0].nodeCreated, "function");
});

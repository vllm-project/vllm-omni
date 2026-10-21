// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";

// Load the browser ES module the same way lingbot_player.test.mjs does. The
// module's extension registration is guarded on `window`/`document`, so this
// import also proves that layoutSequence touches no DOM.
const source = await readFile(
  new URL("../../../../apps/ComfyUI-vLLM-Omni/web/lingbot-timeline.js", import.meta.url),
);
const { layoutSequence, register } = await import(
  `data:text/javascript;base64,${source.toString("base64")}`
);

const GEOMETRY = {
  first_chunk_frames: 9,
  chunk_frames: 12,
  latent_frames_per_chunk: 3,
  temporal_scale: 4,
};

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
    start_frame: 0,
    start_chunk: 0,
    chunks: 4,
    moves: [{ lane: "move", value: "forward", start_frame: 0, frames: 45 }],
  },
  {
    prompt: PROMPTS[1],
    frames: 84,
    start_frame: 45,
    start_chunk: 4,
    chunks: 7,
    moves: [
      { lane: "move", value: "forward", start_frame: 0, frames: 24 },
      { lane: "look_h", value: "right", start_frame: 24, frames: 60 },
    ],
  },
  {
    prompt: PROMPTS[2],
    frames: 60,
    start_frame: 129,
    start_chunk: 11,
    chunks: 5,
    moves: [{ lane: "look_v", value: "up", start_frame: 12, frames: 36 }],
  },
];

function fixture(overrides = {}) {
  return {
    fps: 16,
    geometry: GEOMETRY,
    total_frames: 189,
    total_chunks: 16,
    shots: SHOTS,
    prompt_switches: [
      [4, PROMPTS[1]],
      [11, PROMPTS[2]],
    ],
    camera_script: [
      [["w"], ["w"], ["w"]],
      [["w"], ["w"], ["w"]],
    ],
    warnings: ["Shot 2: the look_h move is cut to 24 frames at the shot's end."],
    ...overrides,
  };
}

test("shot blocks span the sequence on both the frame and chunk axes", () => {
  const layout = layoutSequence(fixture());
  assert.deepEqual(
    layout.shots.map((shot) => [shot.start, shot.end, shot.startChunk, shot.chunkEnd, shot.chunks]),
    [
      [0, 45, 0, 4, 4],
      [45, 129, 4, 11, 7],
      [129, 189, 11, 16, 5],
    ],
  );
  assert.equal(layout.totalFrames, 189);
  assert.equal(layout.totalChunks, 16);
  assert.equal(layout.seconds, 189 / 16);
  assert.deepEqual(layout.shots.map((shot) => shot.prompt), PROMPTS);
  assert.equal(layout.empty, false);
});

test("move bars are absolute, mid-shot moves included, and clipped to their shot", () => {
  const layout = layoutSequence(fixture());
  // Row order follows LANE_ORDER, which is the order the Python LANES dict uses.
  assert.deepEqual(layout.lanes.map((row) => row.lane), ["move", "look_v", "look_h"]);
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
  assert.equal(layout.bars[2].start, layout.shots[1].start + 24);
  assert.ok(layout.bars[2].start > layout.shots[1].start);
  assert.equal(layout.bars[2].label, "look_h right");

  const clipped = layoutSequence(
    fixture({
      shots: [{ ...SHOTS[0], moves: [{ lane: "move", value: "back", start_frame: 30, frames: 60 }] }],
      prompt_switches: [],
    }),
  );
  assert.deepEqual(
    clipped.bars.map((bar) => [bar.start, bar.end, bar.frames]),
    [[30, 45, 15]],
  );
  assert.equal(clipped.bars[0].end, clipped.shots[0].end);
});

test("ruler ticks use one 1-2-5 step and stay near ten labels", () => {
  const layout = layoutSequence(fixture());
  assert.ok(layout.ticks.length >= 3 && layout.ticks.length <= 10, `${layout.ticks.length} ticks`);
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
  // Second labels come from fps; frame 20 is 1.25 s at 16 fps.
  assert.equal(layout.ticks[1].frame, 20);
  assert.equal(layout.ticks[1].label, "1.25s");
});

test("chunk grid is the 9 + 12k sequence and prompt switches sit on its lines", () => {
  const layout = layoutSequence(fixture());
  assert.equal(layout.chunks.length, 16);
  assert.deepEqual(layout.chunks[0], { index: 0, start: 0, end: 9, frames: 9 });
  assert.deepEqual(layout.chunks[1], { index: 1, start: 9, end: 21, frames: 12 });
  assert.equal(layout.chunks.at(-1).end, layout.totalFrames);
  for (const chunk of layout.chunks) {
    assert.equal(chunk.end - chunk.start, chunk.frames);
    assert.equal(chunk.frames, chunk.index === 0 ? 9 : 12);
  }
  assert.deepEqual(layout.boundaries.map((boundary) => boundary.frame), [45, 129]);
  assert.deepEqual(layout.boundaries.map((boundary) => boundary.chunk), [4, 11]);
  assert.deepEqual(layout.boundaries.map((boundary) => boundary.prompt), [PROMPTS[1], PROMPTS[2]]);
});

test("one shot without moves still lays out, and an empty payload is harmless", () => {
  const single = layoutSequence({
    fps: null,
    geometry: GEOMETRY,
    total_frames: 9,
    total_chunks: 1,
    shots: [{ prompt: "One short shot.", frames: 9, start_frame: 0, start_chunk: 0, chunks: 1, moves: [] }],
    prompt_switches: [],
    warnings: [],
  });
  assert.deepEqual(single.lanes, []);
  assert.deepEqual(single.bars, []);
  assert.deepEqual(single.chunks, [{ index: 0, start: 0, end: 9, frames: 9 }]);
  assert.equal(single.seconds, null);
  assert.equal(single.shots[0].start, 0);
  assert.equal(single.shots[0].end, 9);
  assert.ok(single.ticks.length > 0);
  assert.ok(single.ticks.every((tick) => tick.label.endsWith("f")), "frames-only ruler without fps");
  assert.equal(single.ticks[0].label, "0f");

  const empty = layoutSequence(undefined);
  assert.deepEqual(empty.shots, []);
  assert.deepEqual(empty.bars, []);
  assert.deepEqual(empty.ticks, []);
  assert.deepEqual(empty.chunks, []);
  assert.equal(empty.totalFrames, 0);
  assert.equal(empty.empty, true);
});

test("warnings are carried through verbatim and register() is exported", () => {
  const layout = layoutSequence(fixture());
  assert.deepEqual(layout.warnings, fixture().warnings);
  assert.equal(typeof register, "function");

  const calls = [];
  const installed = [];
  const app = {
    nodeOutputs: {},
    registerExtension(extension) {
      installed.push(extension);
    },
  };
  register(app);
  assert.equal(installed.length, 1);
  const node = {
    comfyClass: "VLLMOmniLingBotVisualizer",
    addWidget(...args) {
      calls.push(args);
    },
  };
  installed[0].nodeCreated(node);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].slice(0, 2), ["button", "Open timeline"]);
  assert.deepEqual(calls[0][4], { serialize: false });
  // Only the visualizer node gets the button, and openTimeline is never called here.
  installed[0].nodeCreated({ comfyClass: "VLLMOmniLingBotShot", addWidget: () => calls.push(["unexpected"]) });
  assert.equal(calls.length, 1);
});

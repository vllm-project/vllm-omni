// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";

// Load the browser ES module the same way lingbot_timeline.test.mjs does. The
// module registers itself only under a `window`/`document` guard, so this
// import also proves that layoutMoves, moveProblems and snapFrame touch no DOM.
const source = await readFile(
  new URL("../../../../apps/ComfyUI-vLLM-Omni/web/lingbot-move-editor.js", import.meta.url),
);
const { layoutMoves, moveProblems, snapFrame, register } = await import(
  `data:text/javascript;base64,${source.toString("base64")}`
);

const GEOMETRY = {
  first_chunk_frames: 9,
  chunk_frames: 12,
  latent_frames_per_chunk: 3,
  temporal_scale: 4,
};

test("three moves on three lanes get their lane order, row, key and label", () => {
  const moves = [
    { lane: "move", value: "forward", start_frame: 0, frames: 12 },
    { lane: "strafe", value: "right", start_frame: 12, frames: 12 },
    { lane: "look_h", value: "left", start_frame: 24, frames: 12 },
  ];
  const layout = layoutMoves(moves, GEOMETRY, 60);

  // Every lane is drawn, in camera order, even the one without a move.
  assert.deepEqual(layout.lanes.map((lane) => lane.lane), ["move", "strafe", "look_v", "look_h"]);
  assert.deepEqual(layout.lanes.map((lane) => lane.moves), [[0], [1], [], [2]]);
  assert.deepEqual(
    layout.moves.map((move) => [move.lane, move.value, move.row, move.key]),
    [
      ["move", "forward", 0, "w"],
      ["strafe", "right", 1, "d"],
      ["look_h", "left", 3, "j"],
    ],
  );
  assert.deepEqual(
    layout.moves.map((move) => [move.start, move.end, move.frames]),
    [[0, 12, 12], [12, 24, 12], [24, 36, 12]],
  );
  assert.deepEqual(
    layout.moves.map((move) => [move.startPercent, move.endPercent, move.widthPercent]),
    [[0, 20, 20], [20, 40, 20], [40, 60, 20]],
  );
  assert.deepEqual(
    layout.moves.map((move) => move.label),
    ["move forward · 12f", "strafe right · 12f", "look_h left · 12f"],
  );
  assert.equal(layout.frames, 60);
  assert.equal(layout.seconds, 60 / 16);
  assert.equal(moveProblems(moves, GEOMETRY, 60).size, 0);
});

test("every lane direction maps to the server's action key", () => {
  const moves = [
    { lane: "move", value: "forward", start_frame: 0, frames: 4 },
    { lane: "move", value: "back", start_frame: 4, frames: 4 },
    { lane: "strafe", value: "left", start_frame: 8, frames: 4 },
    { lane: "strafe", value: "right", start_frame: 12, frames: 4 },
    { lane: "look_v", value: "up", start_frame: 16, frames: 4 },
    { lane: "look_v", value: "down", start_frame: 20, frames: 4 },
    { lane: "look_h", value: "left", start_frame: 24, frames: 4 },
    { lane: "look_h", value: "right", start_frame: 28, frames: 4 },
  ];
  const layout = layoutMoves(moves, GEOMETRY, 32);
  assert.deepEqual(layout.moves.map((move) => move.key), ["w", "s", "a", "d", "i", "k", "j", "l"]);
  assert.deepEqual(layout.moves.map((move) => move.row), [0, 0, 1, 1, 2, 2, 3, 3]);
  assert.equal(moveProblems(moves, GEOMETRY, 32).size, 0);
});

test("a move starting mid-shot keeps its offset in frames and percent", () => {
  const moves = [{ lane: "look_v", value: "up", start_frame: 24, frames: 12 }];
  const layout = layoutMoves(moves, GEOMETRY, 48);
  const move = layout.moves[0];
  assert.equal(move.start, 24);
  assert.equal(move.end, 36);
  assert.equal(move.frames, 12);
  assert.equal(move.startPercent, 50);
  assert.equal(move.endPercent, 75);
  assert.equal(move.widthPercent, 25);
  assert.equal(move.row, 2);
  assert.equal(move.key, "i");
  assert.equal(move.label, "look_v up · 12f");
  assert.equal(moveProblems(moves, GEOMETRY, 48).size, 0);
});

test("two moves on one lane that overlap are invalid on both sides", () => {
  const moves = [
    { lane: "move", value: "forward", start_frame: 0, frames: 12 },
    { lane: "move", value: "back", start_frame: 8, frames: 12 },
  ];
  const layout = layoutMoves(moves, GEOMETRY, 45);
  assert.deepEqual(layout.lanes[0].moves, [0, 1]);

  const problems = moveProblems(moves, GEOMETRY, 45);
  assert.equal(problems.size, 2);
  for (const problem of problems.values()) {
    assert.equal(problem.invalid, true);
    assert.match(problem.errors.join(" "), /overlaps/);
    assert.equal(problem.warnings.length, 0);
  }
});

test("a move shorter than one latent frame is an error", () => {
  const moves = [{ lane: "move", value: "forward", start_frame: 8, frames: 3 }];
  const problems = moveProblems(moves, GEOMETRY, 45);
  const problem = problems.get(0);
  assert.ok(problem);
  assert.equal(problem.invalid, true);
  assert.match(problem.errors.join(" "), /shorter than one latent frame \(4 pixel frames\)/);
  assert.equal(layoutMoves(moves, GEOMETRY, 45).moves[0].frames, 3);
});

test("a move clipped by the shot's end is a warning, not an error", () => {
  const moves = [{ lane: "move", value: "back", start_frame: 36, frames: 24 }];
  const layout = layoutMoves(moves, GEOMETRY, 45);
  const move = layout.moves[0];
  assert.deepEqual([move.start, move.end, move.frames], [36, 45, 9]);
  assert.equal(move.clipped, true);
  // The label reports the frames that actually play, not the requested ones.
  assert.equal(move.label, "move back · 9f");

  const problem = moveProblems(moves, GEOMETRY, 45).get(0);
  assert.ok(problem);
  assert.equal(problem.invalid, false);
  assert.equal(problem.errors.length, 0);
  assert.match(problem.warnings.join(" "), /past the shot's end at 45 frames; cut there/);
});

test("an unknown lane gets its own row and an unknown direction is reported", () => {
  const moves = [
    { lane: "pan", value: "left", start_frame: 0, frames: 12 },
    { lane: "move", value: "left", start_frame: 12, frames: 12 },
  ];
  const layout = layoutMoves(moves, GEOMETRY, 45);

  // The four known lanes stay first; the unknown lane is appended so its move
  // is still visible, selectable and deletable.
  assert.deepEqual(
    layout.lanes.map((lane) => lane.lane),
    ["move", "strafe", "look_v", "look_h", "pan"],
  );
  assert.equal(layout.lanes.at(-1).known, false);
  assert.equal(layout.moves[0].row, 4);
  assert.equal(layout.moves[0].key, null);
  assert.equal(layout.moves[1].key, null);

  const problems = moveProblems(moves, GEOMETRY, 45);
  assert.match(problems.get(0).errors.join(" "), /unknown camera lane "pan"/);
  assert.match(problems.get(1).errors.join(" "), /unknown move direction "left"/);
  assert.equal(problems.get(0).invalid, true);
  assert.equal(problems.get(1).invalid, true);
});

test("a start frame off the latent grid snaps to a multiple of 4, never past the end", () => {
  assert.equal(snapFrame(13, GEOMETRY, 45), 12);
  assert.equal(snapFrame(14, GEOMETRY, 45), 16);
  assert.equal(snapFrame(16, GEOMETRY, 45), 16);
  // The shot's last frame is a legal end even when it is off the latent grid.
  assert.equal(snapFrame(45, GEOMETRY, 45), 45);
  assert.equal(snapFrame(60, GEOMETRY, 45), 45);
  assert.equal(snapFrame(-3, GEOMETRY, 45), 0);

  // The editor stores the snapped frame, so the layout draws it on the grid.
  const snapped = snapFrame(13, GEOMETRY, 45);
  const layout = layoutMoves(
    [{ lane: "move", value: "forward", start_frame: snapped, frames: 12 }],
    GEOMETRY,
    45,
  );
  assert.equal(layout.moves[0].start, 12);
  assert.equal(layout.moves[0].start % GEOMETRY.temporal_scale, 0);
});

test("ruler ticks use one 1-2-5 step and the chunk grid is 9 + 12k", () => {
  const layout = layoutMoves([], GEOMETRY, 45);
  assert.equal(layout.tickStep, 10);
  assert.ok(layout.ticks.length > 1 && layout.ticks.length <= 8, `${layout.ticks.length} ticks`);
  assert.deepEqual(
    layout.ticks.map((tick) => tick.frame),
    [0, 10, 20, 30, 40],
  );
  for (const tick of layout.ticks) {
    assert.match(tick.label, /^\d+f · /);
    assert.match(tick.label, /s$/);
    assert.equal(tick.seconds, tick.frame / 16);
  }
  const magnitude = 10 ** Math.floor(Math.log10(layout.tickStep));
  const multiple = layout.tickStep / magnitude;
  assert.ok([1, 2, 5].some((candidate) => Math.abs(multiple - candidate) < 1e-9), `${layout.tickStep}`);

  assert.deepEqual(
    layout.chunks.map((chunk) => [chunk.start, chunk.end, chunk.frames]),
    [[0, 9, 9], [9, 21, 12], [21, 33, 12], [33, 45, 12]],
  );
  assert.equal(layout.chunks.at(-1).end, layout.frames);
});

test("an empty timeline still lays out four empty lanes", () => {
  const layout = layoutMoves(undefined, GEOMETRY, 21);
  assert.deepEqual(layout.lanes.map((lane) => lane.lane), ["move", "strafe", "look_v", "look_h"]);
  assert.ok(layout.lanes.every((lane) => lane.moves.length === 0));
  assert.deepEqual(layout.moves, []);
  assert.equal(moveProblems(undefined, GEOMETRY, 21).size, 0);
});

test("register() installs the LINGBOT_TIMELINE custom widget and never needs a DOM", () => {
  const installed = [];
  const app = {
    registerExtension(extension) {
      installed.push(extension);
    },
  };
  register(app);
  assert.equal(installed.length, 1);
  assert.equal(installed[0].name, "vllm.lingbot_move_editor");
  const factories = installed[0].getCustomWidgets();
  assert.deepEqual(Object.keys(factories), ["LINGBOT_TIMELINE"]);
  assert.equal(typeof factories.LINGBOT_TIMELINE, "function");
  // setup() only injects CSS, and is a no-op while no document exists.
  assert.doesNotThrow(() => installed[0].setup());
  // Missing or invalid apps are ignored rather than throwing.
  assert.doesNotThrow(() => register(undefined));
  assert.doesNotThrow(() => register({}));
});

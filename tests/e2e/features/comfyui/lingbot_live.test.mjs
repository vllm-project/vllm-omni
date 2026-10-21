// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";

// Load the browser ES module the same way lingbot_player.test.mjs does. The
// panel has no static imports and guards its `window`/`document`
// registration, so this import also proves the pure helpers touch no DOM.
const source = await readFile(
  new URL("../../../../apps/ComfyUI-vLLM-Omni/web/lingbot.js", import.meta.url),
);
const {
  eventId,
  hudCaps,
  saveName,
  statsText,
  statusLine,
  statusText,
} = await import(`data:text/javascript;base64,${source.toString("base64")}`);

const NOW = Date.UTC(2026, 9, 21, 11, 39, 25);

test("a held key set lights exactly the matching keycaps", () => {
  const pads = hudCaps(new Set(["w", "l"]));
  assert.deepEqual(pads.map((pad) => pad.name), ["move", "look"]);

  const lit = Object.fromEntries(
    pads.flatMap((pad) => pad.caps.map((cap) => [cap.key, cap.on])),
  );
  assert.deepEqual(lit, {
    w: true, a: false, s: false, d: false,
    i: false, j: false, k: false, l: true,
  });

  // The look pad reads as an arrow pad, with I/J/K/L as corner labels.
  assert.deepEqual(pads[1].caps.map((cap) => cap.glyph), ["↑", "←", "↓", "→"]);
  assert.deepEqual(pads[1].caps.map((cap) => cap.label), ["I", "J", "K", "L"]);
  // W sits above A/S/D, the arrows mirror their keys.
  assert.deepEqual(
    pads[0].caps.map((cap) => [cap.key, cap.column, cap.row]),
    [["w", 2, 1], ["a", 1, 2], ["s", 2, 2], ["d", 3, 2]],
  );
});

test("releasing the keys clears the HUD", () => {
  for (const pad of hudCaps()) {
    assert.ok(pad.caps.every((cap) => cap.on === false));
  }
});

test("the status line names the main session states", () => {
  assert.equal(statusText("waiting"), "Generating the first chunk…");
  assert.equal(statusText("playing"), "Playing");
  assert.equal(statusText("paused"), "Paused");
  assert.equal(statusText("stopped"), "Stopped");

  const failed = statusLine("error", "Connection lost; start a new session.");
  assert.equal(failed.text, "Connection lost; start a new session.");
  assert.equal(failed.error, true);
  assert.equal(statusLine("playing").error, false);
});

test("Save names the file from the prompt's first words and a timestamp", () => {
  assert.equal(
    saveName("A quiet forest path in daylight.", NOW),
    "lingbot-a-quiet-forest-path-20261021-113925.mp4",
  );
  assert.equal(
    saveName("The path opens onto a windswept ridge.", NOW),
    "lingbot-the-path-opens-onto-20261021-113925.mp4",
  );
  assert.equal(saveName("   ", NOW), "lingbot-world-20261021-113925.mp4");
});

test("interaction ids are unique and monotonically increasing", () => {
  const ids = [1, 2, 3, 9, 10, 11].map((sequence) => eventId("camera", sequence));
  assert.deepEqual(ids, [
    "lingbot-camera-00000001",
    "lingbot-camera-00000002",
    "lingbot-camera-00000003",
    "lingbot-camera-00000009",
    "lingbot-camera-00000010",
    "lingbot-camera-00000011",
  ]);
  for (let i = 1; i < ids.length; i++) {
    assert.ok(ids[i - 1] < ids[i], `${ids[i - 1]} sorts before ${ids[i]}`);
  }
  assert.notEqual(eventId("camera", 1), eventId("prompt", 1));
});

test("the stats line reads like Reactor's and skips unknown fields", () => {
  assert.equal(
    statsText({
      width: 832,
      height: 480,
      fps: 16,
      chunkIndex: 3,
      generatedSeconds: 4.8,
      playedSeconds: 1.2,
      bufferAhead: 1.25,
      sessionId: "0123456789abcdef",
    }),
    "832×480 · 16 fps · chunk 3 · 4.8 s generated · 1.2 s played · buffer 1.25 s · session 01234567",
  );
  assert.equal(statsText({}), "");
  assert.equal(statsText({ chunkIndex: 0 }), "chunk 0");
});

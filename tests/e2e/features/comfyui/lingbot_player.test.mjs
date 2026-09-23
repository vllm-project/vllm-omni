// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";

// Load the browser ES module without imposing a package.json module type on the app.
const source = await readFile(new URL("../../../../apps/ComfyUI-vLLM-Omni/web/lingbot-player.js", import.meta.url));
const { cameraEvent, VideoSession } = await import(`data:text/javascript;base64,${source.toString("base64")}`);

test("held keys, release, and cancelling opposite directions", () => {
  const camera = keys => cameraEvent(new Set(keys), "test").interaction.event.multi_modal_data.camera;
  assert.deepEqual(camera("w").data.translation, [0, 0, 0.05]);
  assert.deepEqual(camera("wasd").data.translation, [0, 0, 0]);
  assert.deepEqual(camera("").data, { translation: [0, 0, 0], rotation: [0, 0, -0, 1] });
  assert.equal(camera("il").mode, "velocity");
  assert.ok(Math.abs(Math.hypot(...camera("il").data.rotation) - 1) < 1e-12);
  assert.ok(camera("i").data.rotation[0] > 0);
  assert.ok(camera("l").data.rotation[1] > 0);
  assert.equal(cameraEvent(new Set(), "release-2").interaction.event_id, "release-2");
});

test("keepalive covers model initialization and stops with the session", async (t) => {
  const original = Object.fromEntries(["window", "location", "MediaSource", "WebSocket"].map(k => [k, globalThis[k]]));
  const sent = [];
  let ping;
  let cancelled = false;
  class Source extends EventTarget {
    static isTypeSupported() { return true; }
    readyState = "open";
    constructor() {
      super();
      queueMicrotask(() => this.dispatchEvent(new Event("sourceopen")));
    }
    addSourceBuffer() { return Object.assign(new EventTarget(), { updating: false, buffered: { length: 0 } }); }
    endOfStream() { this.readyState = "ended"; }
  }
  class Socket {
    static OPEN = 1;
    readyState = 1;
    constructor() { queueMicrotask(() => this.onopen()); }
    send(data) { sent.push(JSON.parse(data)); }
    close() { this.readyState = 3; }
  }
  Object.assign(globalThis, { MediaSource: Source, WebSocket: Socket, window: { MediaSource: Source }, location: { protocol: "http:" } });
  t.mock.method(URL, "createObjectURL", () => "blob:test");
  t.mock.method(URL, "revokeObjectURL", () => {});
  t.mock.method(globalThis, "setInterval", (callback, delay) => { assert.equal(delay, 20000); ping = callback; return 42; });
  t.mock.method(globalThis, "clearInterval", id => { if (id === 42) cancelled = true; });
  const video = Object.assign(new EventTarget(), { currentTime: 0, pause() {}, load() {}, removeAttribute() {}, play: async () => {} });
  const session = new VideoSession(video, () => {});
  try {
    session.start({ url: "ws://localhost/video", payload: { type: "session.start" } });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(session.ready, false); // No media yet: initialization still needs keepalive.
    ping();
    assert.deepEqual(sent.map(event => event.type), ["session.start", "session.ping"]);
    session.control({ type: "session.done", stopped: false });
    assert.equal(cancelled, true);
  } finally {
    session.dispose();
    for (const [key, value] of Object.entries(original)) {
      if (value === undefined) delete globalThis[key];
      else globalThis[key] = value;
    }
  }
});

test("playback credit follows the video clock and stops on pause, stop, and completion", () => {
  const video = Object.assign(new EventTarget(), { currentTime: 0 });
  const session = new VideoSession(video, () => {});
  const messages = [];
  session.send = message => { messages.push(message); return true; };
  session.sendPlayback();
  assert.equal(messages.length, 0);
  session.ready = true;
  session.sendPlayback();
  video.currentTime = 0.25;
  session.sendPlayback();
  session.sendPlayback(); // Paused video: no new credit.
  video.currentTime = 0.1;
  session.sendPlayback(); // A backwards seek cannot grant more credit.
  assert.deepEqual(messages.map(m => m.position_seconds), [0, 0.25]);
  assert.ok(messages.every(m => m.type === "session.playback"));
  video.currentTime = 1;
  session.stopping = true;
  session.sendPlayback();
  session.stopping = false;
  session.done = true;
  session.sendPlayback();
  assert.equal(messages.length, 2);
});

test("camera velocity is FPS-independent and bounded on diagonals", () => {
  for (const fps of [8, 16, 24, 32]) {
    const camera = (keys, speed = 1) => cameraEvent(new Set(keys), "timed", { fps, speed }).interaction.event.multi_modal_data.camera.data;
    assert.ok(Math.abs(camera("w").translation[2] * fps / 4 - 0.2) < 1e-12);
    assert.ok(Math.abs(Math.hypot(...camera("wd").translation) * fps / 4 - 0.2) < 1e-12);
    assert.ok(Math.abs(camera("w", 0.5).translation[2] * fps / 4 - 0.1) < 1e-12);
    const yawPerFrame = 2 * Math.atan2(camera("l").rotation[1], camera("l").rotation[3]);
    assert.ok(Math.abs(yawPerFrame * fps / 4 - 24 * Math.PI / 180) < 1e-12);
    assert.deepEqual(camera("").translation, [0, 0, 0]);
  }
});

test("camera rates reject invalid time scales", () => {
  for (const options of [{ fps: 0 }, { fps: NaN }, { speed: -1 }, { speed: Infinity }]) {
    assert.throws(() => cameraEvent(new Set(["w"]), "invalid", options));
  }
});

test("appending a live fragment never discards its earlier reference frames", async () => {
  const video = Object.assign(new EventTarget(), { currentTime: 6 });
  const appended = [];
  let removed = false;
  let ended = false;
  const reports = [];
  const session = new VideoSession(video, message => reports.push(message));
  session.source = { readyState: "open", endOfStream() { ended = true; } };
  session.buffer = {
    updating: false,
    buffered: { length: 1, start: () => 0 },
    remove() { removed = true; },
    appendBuffer(data) { appended.push(data); },
  };
  const fragment = new Uint8Array([1, 2, 3]).buffer;
  session.queue = [fragment];
  session.queuedBytes = fragment.byteLength;
  session.pump();
  assert.equal(removed, false);
  assert.deepEqual(appended, [fragment]);
  assert.equal(session.queuedBytes, 0);
  session.done = true;
  session.pump();
  assert.equal(ended, true);
  video.dispatchEvent(new Event("ended"));
  assert.equal(reports.at(-1), "Playback complete.");
});

test("startup buffering starts once and lets complete short clips play", () => {
  let end = 0.6, played = 0;
  const video = Object.assign(new EventTarget(), {
    currentTime: 0, buffered: { length: 1, end: () => end },
    play: () => { played++; return Promise.resolve(); },
  });
  const session = new VideoSession(video, () => {});
  session.playWhenBuffered();
  assert.equal(played, 0);
  end = 1.6;
  session.playWhenBuffered();
  session.playWhenBuffered();
  assert.equal(played, 1);
  session.playbackStarted = false;
  session.done = true;
  end = 0.6;
  session.playWhenBuffered();
  assert.equal(played, 2);
});

test("manual playback and pause during startup are not overridden by later buffering", () => {
  let end = 0.6, played = 0;
  const video = Object.assign(new EventTarget(), {
    currentTime: 0, buffered: { length: 1, end: () => end },
    play() { played++; this.dispatchEvent(new Event("play")); return Promise.resolve(); },
  });
  const session = new VideoSession(video, () => {});
  session.source = { readyState: "open" };
  session.playWhenBuffered();
  video.play(); // Native controls can start before our startup threshold.
  video.dispatchEvent(new Event("pause"));
  end = 1.6;
  session.playWhenBuffered();
  assert.equal(played, 1);
});

test("stopping during startup does not start the buffered remainder", () => {
  let played = 0;
  const video = Object.assign(new EventTarget(), {
    currentTime: 0, buffered: { length: 1, end: () => 1.6 },
    play: () => { played++; return Promise.resolve(); },
  });
  const session = new VideoSession(video, () => {});
  session.stopping = true;
  session.done = true;
  session.playWhenBuffered();
  assert.equal(played, 0);
});

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import assert from "node:assert/strict";
import test from "node:test";

import {ProxyCapture} from "../../examples/online_serving/invoker_render_game/proxy-capture.mjs";

const profile = {width: 2, height: 1, fps: 24, packet_frames: 17, tail_frames: 5, credit_frames: 56};

test("sends two regular packets before the first window and five real samples at EOS", () => {
  const capture = new ProxyCapture(profile);
  const packets = [];
  capture.finish();
  for (let index = 0; index < 39; index++) {
    const packet = capture.capture(new Uint8Array(6).fill(index), index * 1000 / 24);
    if (packet) packets.push(packet);
  }
  assert.deepEqual(packets.map(({header}) => [header.input_seq, header.frame_start, header.num_frames, header.eos]), [
    [0, 0, 17, false], [1, 17, 17, false], [2, 34, 5, true],
  ]);
  assert.deepEqual(Array.from(packets[2].pixels.filter((_, index) => index % 6 === 0)), [34, 35, 36, 37, 38]);
  assert.equal(capture.canCapture, false);
});

test("pauses when awaiting playback reaches the credit limit and resumes as playback advances", () => {
  const capture = new ProxyCapture(profile);
  for (let index = 0; index < 56; index++) capture.capture(new Uint8Array(6), index);
  assert.equal(capture.sent, 51);
  assert.equal(capture.canCapture, false);
  capture.acknowledge({accepted_frames: 51, credit_frames: 39});
  assert.equal(capture.canCapture, false);
  capture.played(25 / 24);
  assert.equal(capture.displayed, 26);
  assert.equal(capture.canCapture, true);
  assert.throws(() => capture.acknowledge({accepted_frames: 10, credit_frames: 56}), /Invalid input credit/);
});

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_worklet_completion_requires_clean_end():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for browser protocol tests")
    demo = Path(__file__).resolve().parents[2] / "examples/online_serving/text_to_speech/fish_speech/gradio_demo.py"
    module = ast.parse(demo.read_text())
    worklet = next(
        ast.literal_eval(stmt.value)
        for stmt in module.body
        if isinstance(stmt, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "WORKLET_JS" for target in stmt.targets)
    )
    harness = (
        """
const assert = require('node:assert/strict');
let Processor;
global.AudioWorkletProcessor = class {
    constructor() { this.port = {postMessage: m => messages.push(m)}; }
};
global.registerProcessor = (_, processor) => { Processor = processor; };
const messages = [];
"""
        + worklet
        + """
const p = new Processor();
const send = data => p.port.onmessage({data});
const render = () => p.process([], [[new Float32Array(128)]]);
send({type:'clear', token:1});
send({type:'pcm', token:1, data:new Int16Array(2)});
render();
assert.equal(messages.some(m => m.type === 'ended'), false);
// Clean EOF after an underrun must still report completion.
send({type:'end', token:1});
render(); render();
assert.equal(messages.filter(m => m.type === 'ended').length, 1);
messages.length = 0;
send({type:'clear', token:2});
send({type:'pcm', token:2, data:new Int16Array(256)});
send({type:'end', token:2});
render();
assert.equal(messages.some(m => m.type === 'ended'), false);
render(); render();
assert.equal(messages.filter(m => m.type === 'ended').length, 1);
messages.length = 0;
// Stop/error clear and stale EOF cannot complete the current generation.
send({type:'clear', token:3});
send({type:'end', token:2});
render();
assert.equal(messages.some(m => m.type === 'ended'), false);
"""
    )
    subprocess.run([node, "-e", harness], check=True, capture_output=True, text=True, timeout=30)

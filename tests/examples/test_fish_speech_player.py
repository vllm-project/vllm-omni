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


def test_player_initialization_and_terminal_states():
    import json

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for browser protocol tests")
    demo = Path(__file__).resolve().parents[2] / "examples/online_serving/text_to_speech/fish_speech/gradio_demo.py"
    module = ast.parse(demo.read_text())
    statements = [
        stmt
        for stmt in module.body
        if (isinstance(stmt, ast.FunctionDef) and stmt.name == "_build_player_js")
        or (
            isinstance(stmt, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "WORKLET_JS" for target in stmt.targets)
        )
    ]
    namespace = {"json": json}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(demo), "exec"), namespace)
    script = namespace["_build_player_js"](44100).replace("<script>", "").replace("</script>", "")
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const script = process.argv[1];
function setup() {
    let resolve, reject, modules = 0, closes = 0, requests = [], messages = [];
    const elements = {'tts-status': {textContent:''}, 'tts-final': {style:{}, innerHTML:''}};
    const sandbox = {
        window: {}, document: {getElementById: id => elements[id] || null},
        console: {error() {}}, performance, Blob, URL, AbortController,
        Uint8Array, Int16Array, ArrayBuffer, DataView, setTimeout,
        AudioContext: class {
            constructor() {
                this.state = 'running';
                this.audioWorklet = {addModule: () => {
                    modules++;
                    return new Promise((a,b) => {resolve=a;reject=b});
                }};
            }
            async close() { closes++; }
        },
        AudioWorkletNode: class {
            constructor() {this.port = {postMessage: m => messages.push(m)};}
            connect() {}
            disconnect() {}
        },
        fetch: async (url, options) => {
            requests.push(JSON.parse(options.body));
            return new Response(new Uint8Array([1,0,2,0]));
        }
    };
    vm.createContext(sandbox); vm.runInContext(script, sandbox);
    return {sandbox, elements, requests, messages, resolve: () => resolve(), reject: () => reject(new Error('init failed')), modules: () => modules, closes: () => closes};
}
(async () => {
    const s = setup();
    const first = s.sandbox.window.ttsGenerate({_req_id:'old'});
    const second = s.sandbox.window.ttsGenerate({_req_id:'new'});
    assert.equal(s.modules(),1);
    s.resolve(); await Promise.all([first,second]);
    assert.deepEqual(s.requests,[{_req_id:'new'}]);
    assert.equal(s.messages.filter(m => m.type === 'end').length,1);

    const retry = setup();
    const failed = retry.sandbox.window.ttsGenerate({_req_id:'failed'});
    retry.reject(); await failed;
    assert.equal(retry.closes(),1);
    assert.match(retry.elements['tts-status'].textContent,/Audio init error/);
    const recovered = retry.sandbox.window.ttsGenerate({_req_id:'retry'});
    assert.equal(retry.modules(),2);
    retry.resolve(); await recovered;
    assert.deepEqual(retry.requests,[{_req_id:'retry'}]);

    const stopped = setup();
    const pending = stopped.sandbox.window.ttsGenerate({_req_id:'stop'});
    stopped.sandbox.window.ttsStop(); stopped.resolve(); await pending;
    assert.equal(stopped.requests.length,0);
    assert.equal(stopped.elements['tts-status'].textContent,'Stopped');

    s.sandbox.fetch = async () => new Response(new ReadableStream({
        start(c) {c.enqueue(new Uint8Array([1,0]));setTimeout(() => c.error(new Error('stream failed')),10);}
    }));
    const ends = s.messages.filter(m => m.type === 'end').length;
    await s.sandbox.window.ttsGenerate({_req_id:'error'});
    assert.match(s.elements['tts-status'].textContent,/Error: stream failed/);
    assert.equal(s.elements['tts-final'].style.display,'none');
    assert.equal(s.messages.filter(m => m.type === 'end').length,ends);
})().catch(e => {console.error(e);process.exitCode=1;});
"""
    subprocess.run([node, "-e", harness, script], check=True, capture_output=True, text=True, timeout=30)

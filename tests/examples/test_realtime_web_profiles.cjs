const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../../examples/online_serving/realtime_web/app/static');
const context = vm.createContext({ URL });
for (const name of ['common', 'minicpm_native', 'qwen3_turn']) {
  vm.runInContext(fs.readFileSync(path.join(root, `profiles/${name}.js`), 'utf8'), context);
}
const profiles = context.OmniRealtimeProfiles;
const plain = (value) => JSON.parse(JSON.stringify(value));
const config = { model: 'Qwen/Qwen3-Omni-30B-A3B-Instruct', realtimePath: 'v1/realtime' };

test('MiniCPM keeps native flags, reference voice, camera, acknowledgements and no commits', () => {
  const native = profiles['minicpm-native']();
  const url = new URL(native.url({ ...config, model: 'MiniCPM' }, 'https://localhost/ui/'));
  assert.equal(url.protocol, 'wss:');
  assert.equal(url.searchParams.get('duplex'), '1');
  assert.equal(url.searchParams.get('native_duplex'), '1');
  const [update] = native.initialMessages({ refAudio: 'data:audio/wav;base64,AA==' }, 'Prompt');
  assert.equal(update.session.ref_audio, 'data:audio/wav;base64,AA==');
  assert.equal(update.session.extra_body.native_duplex, true);
  assert.equal(update.session.instructions, 'Prompt');
  assert.deepEqual(plain(native.commitMessages()), []);
  assert.equal(native.append('PCM', 'JPEG').video_frames[0], 'JPEG');
  assert.equal(native.ack('r1', 200).type, 'playback.ack');
  assert.equal(native.halfDuplex, false);
});

test('STT uses the shipped commit sequence and audio payload, including each new turn', () => {
  const stt = profiles['qwen3-turn'](config);
  for (let turn = 0; turn < 2; turn++) {
    assert.deepEqual(plain(stt.initialMessages(config, 'ignored')), [
      { type: 'session.update', model: config.model },
      { type: 'input_audio_buffer.commit', final: false },
    ]);
    assert.deepEqual(plain(stt.commitMessages()), [{ type: 'input_audio_buffer.commit', final: true }]);
  }
  assert.equal(stt.reconnectEachTurn, true);
  assert.equal(stt.mapEvent({ type: 'response.output_audio.delta', audio: 'PCM' }).event.delta, 'PCM');
  assert.equal(stt.mapEvent({ type: 'response.output_audio.done' }).kind, 'done');
  assert.equal(stt.mapEvent({ type: 'transcription.delta', delta: 'hello' }).role, 'assistant');
});

test('Qwen enables sampled camera frames only for VAD and never sends playback ACK', () => {
  for (const adapter of ['stt', 'vad']) {
    const p = profiles['qwen3-turn']({ ...config, adapter });
    const url = new URL(p.url({ ...config, realtimePath: 'wss://backend/v1/realtime?native_duplex=1&minicpmo45_native_duplex=1' }, 'http://localhost/'));
    assert.equal(url.protocol, 'wss:');
    assert.equal(url.searchParams.get('duplex'), adapter === 'vad' ? '1' : '0');
    assert.equal(url.searchParams.has('native_duplex'), false);
    assert.equal(url.searchParams.has('minicpmo45_native_duplex'), false);
    assert.deepEqual(plain(p.append('PCM', 'JPEG').video_frames || []), adapter === 'vad' ? ['JPEG'] : []);
    assert.equal(p.ack('r', 100), null);
    assert.equal(p.camera, adapter === 'vad');
    assert.equal(p.mapEvent({ type: 'response.listen' }).kind, 'ignore');
  }
});

test('VAD uses nested format and explicit non-interrupting endpoint detection', () => {
  const vad = profiles['qwen3-turn']({ ...config, adapter: 'vad' });
  const [update, historyUpdate] = vad.initialMessages(config, 'help');
  assert.deepEqual(plain(update.session.audio.input), {
    format: { type: 'audio/pcm', rate: 16000 },
    turn_detection: { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300,
      silence_duration_ms: 500, create_response: true, interrupt_response: false },
  });
  assert.deepEqual(plain(historyUpdate), {
    type: 'session.update', session: { playback_commit_policy: 'commit_all_on_done' },
  });
  assert.equal(update.session.extra_body, undefined);
  assert.deepEqual(plain(vad.commitMessages()), []);
  assert.equal(vad.mapEvent({ type: 'response.audio.done' }).kind, 'drain');
  assert.equal(vad.mapEvent({ type: 'response.done' }).kind, 'done');
  assert.equal(vad.mapEvent({ type: 'response.output_text.delta', delta: 'hello' }).text, 'hello');
  assert.equal(vad.mapEvent({ type: 'error', error: { code: 'input_backpressure' } }).kind, 'backpressure');
  assert.equal(vad.mapEvent({ type: 'input_audio_buffer.cleared' }).kind, 'backpressure');
  assert.match(vad.mapEvent({ type: 'error', code: 'unsupported', error: 'Unavailable' }).message, /session_mode: duplex/);
  assert.match(vad.mapEvent({ type: 'error', code: 'server_vad_initialization_failed' }).message, /Silero/);
});

function shell(profileName, adapter = 'stt', options = {}) {
  const elements = new Map();
  class Element {
    constructor() { this.style = {}; this.listeners = {}; this.children = []; this.value = ''; this.textContent = ''; this.classList = { add() {}, remove() {}, toggle() {} }; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    replaceChildren() { this.children = []; }
    remove() {}
  }
  const document = {
    getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
    createElement() { return new Element(); },
  };
  const timers = new Map();
  let timerId = 0;
  const sockets = [];
  const nodes = [];
  class Socket {
    static OPEN = 1;
    constructor() {
      this.readyState = 1; this.sent = []; sockets.push(this);
      queueMicrotask(() => {
        this.onopen?.();
        if (!options.silent) this.onmessage?.({ data: JSON.stringify(options.error || { type: adapter === 'stt' && profileName === 'qwen3-turn' ? 'session.created' : 'session.updated' }) });
      });
    }
    send(value) { this.sent.push(JSON.parse(value)); }
    close() { this.readyState = 3; this.onclose?.({ code: 1000 }); }
  }
  class AudioContext {
    constructor(options) { this.sampleRate = options.sampleRate; this.audioWorklet = { addModule: async () => {} }; this.destination = {}; }
    createMediaStreamSource() { return { connect() {} }; }
    createGain() { return { gain: {}, connect() {} }; }
    async resume() {}
    async close() {}
  }
  class AudioWorkletNode {
    constructor(_context, name) { this.name = name; this.sent = []; nodes.push(this); this.port = { postMessage: (message) => this.sent.push(message) }; }
    connect(target) { return target; }
  }
  const ctx = vm.createContext({
    URL, document, AudioContext, AudioWorkletNode, WebSocket: Socket,
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) } },
    location: { href: 'https://localhost/' },
    OMNI_REALTIME_CONFIG: { ...config, profile: profileName, adapter },
    addEventListener() {},
    setTimeout(fn, ms) { const id = ++timerId; timers.set(id, { fn, ms }); return id; },
    clearTimeout(id) { timers.delete(id); },
    setInterval() { return ++timerId; }, clearInterval() {},
    btoa: (value) => Buffer.from(value, 'binary').toString('base64'),
    atob: (value) => Buffer.from(value, 'base64').toString('binary'),
  });
  ctx.window = ctx;
  for (const name of ['common', 'minicpm_native', 'qwen3_turn']) {
    vm.runInContext(fs.readFileSync(path.join(root, `profiles/${name}.js`), 'utf8'), ctx);
  }
  // Expose closure controls only in the test VM; production has no test API.
  const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8').replace(/\}\)\(\);\s*$/, `
    globalThis.testUI = { startSession, stopSession, handleEvent, playbackDrained,
      microphoneUploadEnabled, flushCapture,
      capture() { pendingCapture.push(new Int16Array([100, 200])); },
      state() { return { running, connectionReady, assistantActive }; }
    };
  })();`);
  vm.runInContext(source, ctx);
  return { ui: ctx.testUI, elements, sockets, nodes,
    timeout() { for (const timer of timers.values()) if (timer.ms === 15000) timer.fn(); },
    async echo() {
      for (const [id, timer] of timers) if (timer.ms === 300) { timers.delete(id); await timer.fn(); }
    },
  };
}

test('shared shell STT sends final commit and starts a fresh second turn only after playback', async () => {
  const app = shell('qwen3-turn');
  await app.ui.startSession();
  assert.equal(app.ui.microphoneUploadEnabled(), true);
  app.ui.capture(); app.ui.flushCapture();
  app.elements.get('sendTurnButton').listeners.click();
  assert.equal(app.sockets[0].sent.at(-1).final, true);
  assert.equal(app.ui.microphoneUploadEnabled(), false);
  await app.ui.handleEvent({ type: 'response.output_audio.delta', audio: 'AAAAAA==' });
  await app.ui.handleEvent({ type: 'response.output_audio.done' });
  await app.echo();
  assert.equal(app.sockets.length, 1, 'wait for actual speaker drain');
  app.ui.playbackDrained({ responseId: 'turn-1', playedMs: 1 });
  await app.echo();
  assert.equal(app.sockets.length, 2);
  assert.equal(app.sockets[1].sent[1].final, false);
  assert.equal(app.ui.microphoneUploadEnabled(), true);
  assert.equal(app.elements.get('cameraButton').hidden, true);
  assert.equal(app.sockets.flatMap(s => s.sent).some(e => e.type === 'playback.ack'), false);
  await app.ui.stopSession({ terminal: false });
});

test('VAD gates input on backpressure and waits for response.done after audio drain', async () => {
  const app = shell('qwen3-turn', 'vad');
  await app.ui.startSession();
  await app.ui.handleEvent({ type: 'error', code: 'input_backpressure' });
  assert.equal(app.ui.microphoneUploadEnabled(), false);
  await app.ui.handleEvent({ type: 'response.created', response: { id: 'r1' } });
  await app.ui.handleEvent({ type: 'response.audio.delta', delta: 'AAAAAA==', response_id: 'r1' });
  await app.ui.handleEvent({ type: 'response.audio.done', response_id: 'r1' });
  app.ui.playbackDrained({ responseId: 'r1', playedMs: 1 });
  await app.echo();
  assert.equal(app.ui.microphoneUploadEnabled(), false, 'audio.done is not terminal');
  await app.ui.handleEvent({ type: 'response.done', response: { id: 'r1' } });
  await app.echo();
  assert.equal(app.ui.microphoneUploadEnabled(), true);
  assert.equal(app.sockets.length, 1, 'VAD keeps its session');
  assert.equal(app.sockets[0].sent.some(e => e.type === 'input_audio_buffer.commit'), false);
  await app.ui.stopSession({ terminal: false });
});

test('MiniCPM retains microphone upload while speaking and acknowledges speaker drain', async () => {
  const app = shell('minicpm-native');
  await app.ui.startSession();
  await app.ui.handleEvent({ type: 'response.speak', response_id: 'native-r1' });
  assert.equal(app.ui.microphoneUploadEnabled(), true);
  await app.ui.handleEvent({ type: 'response.output_audio.delta', delta: 'AAAAAA==', response_id: 'native-r1' });
  app.ui.playbackDrained({ responseId: 'native-r1', playedMs: 100 });
  assert.equal(app.sockets[0].sent.at(-1).type, 'playback.ack');
  assert.equal(app.sockets[0].sent.some(e => e.type === 'input_audio_buffer.commit'), false);
  await app.ui.stopSession({ terminal: false });
  await app.echo();
  assert.equal(app.ui.state().running, false);
});

test('VAD handshake errors remain visible after resource cleanup', async () => {
  const app = shell('qwen3-turn', 'vad', { error: { type: 'error', code: 'unsupported', error: 'Realtime API is not available' } });
  await app.ui.startSession();
  assert.equal(app.ui.state().running, false);
  assert.equal(app.elements.get('connectionState').textContent, 'Error');
  assert.match(app.elements.get('runtimeDetail').textContent, /session_mode: duplex/);
  assert.equal(app.sockets[0].readyState, 3);
});

test('missing session acknowledgement fails instead of leaving Start disabled forever', async () => {
  const app = shell('qwen3-turn', 'vad', { silent: true });
  const starting = app.ui.startSession();
  for (let i = 0; i < 20; i++) await Promise.resolve();
  app.timeout();
  await starting;
  assert.equal(app.ui.state().running, false);
  assert.equal(app.elements.get('callButton').disabled, false);
  assert.match(app.elements.get('runtimeDetail').textContent, /handshake timed out/);
});

test('Qwen does not duplicate text and audio-transcript channels', async () => {
  const app = shell('qwen3-turn', 'vad');
  await app.ui.startSession();
  await app.ui.handleEvent({ type: 'response.audio_transcript.delta', delta: '' });
  await app.ui.handleEvent({ type: 'response.output_text.delta', delta: 'Hello' });
  await app.ui.handleEvent({ type: 'response.audio_transcript.delta', delta: 'Hello' });
  await app.ui.handleEvent({ type: 'response.output_text.done', text: 'Hello' });
  await app.ui.handleEvent({ type: 'response.done' });
  const turns = app.elements.get('conversation').children;
  assert.equal(turns.length, 1);
  assert.equal(turns[0].children[1].textContent, 'Hello');
  await app.ui.stopSession({ terminal: false });
});

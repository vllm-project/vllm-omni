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
  assert.equal(url.searchParams.has('native_duplex'), false);
  const [update] = native.initialMessages({ refAudio: 'data:audio/wav;base64,AA==' }, 'Prompt');
  assert.equal(update.session.ref_audio, 'data:audio/wav;base64,AA==');
  assert.equal(update.session.extra_body.native_duplex, undefined);
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

test('Qwen enables camera and playback ACK only for the duplex VAD profile', () => {
  for (const adapter of ['stt', 'vad']) {
    const p = profiles['qwen3-turn']({ ...config, adapter });
    const url = new URL(p.url({ ...config, realtimePath: 'wss://backend/v1/realtime?native_duplex=1&minicpmo45_native_duplex=1' }, 'http://localhost/'));
    assert.equal(url.protocol, 'wss:');
    assert.equal(url.searchParams.get('duplex'), adapter === 'vad' ? '1' : '0');
    assert.equal(url.searchParams.has('native_duplex'), false);
    assert.equal(url.searchParams.has('minicpmo45_native_duplex'), false);
    assert.equal(p.append('PCM', 'JPEG').video_frames, undefined);
    assert.equal(plain(p.imageMessages('JPEG')).length, adapter === 'vad' ? 1 : 0);
    assert.equal(p.ack('r', 100)?.type || null, adapter === 'vad' ? 'playback.ack' : null);
    assert.equal(p.camera, adapter === 'vad');
    assert.equal(p.mapEvent({ type: 'response.listen' }).kind, 'ignore');
  }
});

test('VAD uses nested format and interruptible endpoint detection', () => {
  const vad = profiles['qwen3-turn']({ ...config, adapter: 'vad' });
  const [update] = vad.initialMessages(config, 'help');
  assert.deepEqual(plain(update.session.audio.input), {
    format: { type: 'audio/pcm', rate: 24000 },
    turn_detection: { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300,
      silence_duration_ms: 500, create_response: true, interrupt_response: true },
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
    decodeAudioData(bytes) { return options.decodeAudioData(bytes); }
    async resume() {}
    async close() {}
  }
  class AudioWorkletNode {
    constructor(_context, name) {
      this.name = name; this.sent = []; nodes.push(this);
      this.port = { postMessage: (message) => { this.sent.push(message); this.player?.handleMessage(message); } };
      if (options.realPlayback && name === 'fullduplex-pcm-playback') {
        let Playback;
        const node = this;
        const worker = vm.createContext({
          sampleRate: 24000,
          AudioWorkletProcessor: class {
            constructor() { this.port = { postMessage: message => queueMicrotask(() => node.port.onmessage?.({ data: message })) }; }
          },
          registerProcessor: (_name, cls) => { Playback = cls; },
        });
        vm.runInContext(fs.readFileSync(path.join(root, 'playback_worklet.js'), 'utf8'), worker);
        this.player = new Playback();
      }
    }
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
      microphoneUploadEnabled, flushCapture, awaitQueue: () => audioChain,
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
    async turnWatchdog() {
      for (const [id, timer] of timers) if (timer.ms === 120000) { timers.delete(id); await timer.fn(); }
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

test('VAD keeps uploading while generating and waits for response.done after audio drain', async () => {
  const app = shell('qwen3-turn', 'vad');
  await app.ui.startSession();
  await app.ui.handleEvent({ type: 'error', code: 'input_backpressure' });
  assert.equal(app.ui.microphoneUploadEnabled(), true);
  await app.ui.handleEvent({ type: 'response.created', response: { id: 'r1' } });
  await app.ui.handleEvent({ type: 'response.audio.delta', delta: 'AAAAAA==', response_id: 'r1' });
  await app.ui.handleEvent({ type: 'response.audio.done', response_id: 'r1' });
  app.ui.playbackDrained({ responseId: 'r1', playedMs: 1 });
  await app.echo();
  assert.equal(app.ui.microphoneUploadEnabled(), true, 'continuous capture remains enabled');
  assert.equal(app.ui.state().assistantActive, true, 'audio drain must wait for response.done');
  assert.equal(app.sockets[0].sent.at(-1).type, 'playback.ack');
  await app.ui.handleEvent({ type: 'response.done', response: { id: 'r1' } });
  await app.echo();
  assert.equal(app.ui.microphoneUploadEnabled(), true);
  assert.equal(app.sockets.length, 1, 'VAD keeps its session');
  assert.equal(app.sockets[0].sent.some(e => e.type === 'input_audio_buffer.commit'), false);
  await app.ui.stopSession({ terminal: false });
});

test('a rejected operation is reported without hanging up the call', async () => {
  const app = shell('qwen3-turn', 'vad');
  await app.ui.startSession();
  await app.ui.handleEvent({ type: 'error', code: 'invalid_image', error: 'input_image requires a JPEG or PNG base64 data URL' });
  assert.equal(app.ui.state().running, true, 'one refused image must not end the session');
  assert.equal(app.ui.state().connectionReady, true);
  assert.match(app.elements.get('runtimeDetail').textContent, /JPEG or PNG/);
  await app.ui.stopSession({ terminal: false });
});

test('a session-level error still tears the call down', async () => {
  const app = shell('qwen3-turn', 'vad');
  await app.ui.startSession();
  await app.ui.handleEvent({ type: 'error', code: 'unknown_session', error: 'Unknown or closed duplex session' });
  assert.equal(app.ui.state().running, false);
  assert.equal(app.elements.get('connectionState').textContent, 'Error');
});

test('an interrupted response disarms the turn watchdog it left armed', async () => {
  const app = shell('qwen3-turn', 'vad');
  await app.ui.startSession();
  await app.ui.handleEvent({ type: 'response.created', response: { id: 'r1' } });
  await app.ui.handleEvent({ type: 'error', code: 'input_backpressure' });
  await app.ui.handleEvent({ type: 'response.done', response: { id: 'r1', status: 'cancelled' } });
  await app.turnWatchdog();
  assert.equal(app.ui.state().running, true, 'barge-in must not time the session out two minutes later');
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

const flushTasks = async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); };

test('closing socket accepts only session.closed and completes without the fallback timer', async () => {
  const app = shell('minicpm-native');
  await app.ui.startSession();
  const old = app.sockets[0];
  let stopped = false;
  const stopping = app.ui.stopSession().then(() => { stopped = true; });
  assert.equal(old.sent.at(-1).type, 'session.close');
  old.onmessage({ data: JSON.stringify({ type: 'session.updated' }) });
  await flushTasks();
  assert.equal(stopped, false);
  old.onmessage({ data: JSON.stringify({ type: 'session.closed' }) });
  await flushTasks();
  assert.equal(stopped, true, 'must finish without firing the timeout');
  await stopping;
  await app.ui.startSession();
  old.onmessage({ data: JSON.stringify({ type: 'session.closed' }) });
  old.onmessage({ data: '{broken' });
  await flushTasks();
  assert.equal(app.ui.state().running, true, 'stale socket cannot affect a new session');
  await app.ui.stopSession({ terminal: false });
});

test('malformed JSON before readiness rejects the handshake and cleans up', async () => {
  const app = shell('qwen3-turn', 'vad', { silent: true });
  const starting = app.ui.startSession();
  await flushTasks();
  app.sockets[0].onmessage({ data: '{broken' });
  await starting;
  assert.equal(app.ui.state().running, false);
  assert.equal(app.sockets[0].readyState, 3);
  assert.equal(app.elements.get('connectionState').textContent, 'Error');
  assert.match(app.elements.get('runtimeDetail').textContent, /Invalid server event/);
});

test('malformed JSON after readiness fails the session and releases resources', async () => {
  const app = shell('qwen3-turn', 'vad');
  await app.ui.startSession();
  app.sockets[0].onmessage({ data: '{broken' });
  await flushTasks();
  assert.equal(app.ui.state().running, false);
  assert.equal(app.ui.microphoneUploadEnabled(), false);
  assert.equal(app.sockets[0].readyState, 3);
  assert.equal(app.elements.get('connectionState').textContent, 'Error');
  assert.match(app.elements.get('runtimeDetail').textContent, /Invalid server event/);
});


test('Qwen VAD interruption clears playback while continuing microphone upload', async () => {
  const app = shell('qwen3-turn', 'vad');
  await app.ui.startSession();
  await app.ui.handleEvent({ type: 'response.created', response: { id: 'r1' } });
  await app.ui.handleEvent({ type: 'response.audio.delta', delta: 'AAAAAA==', response_id: 'r1' });
  await app.ui.handleEvent({ type: 'output_audio_buffer.cleared', response_id: 'r1' });
  assert.equal(app.ui.state().assistantActive, false);
  assert.equal(app.ui.microphoneUploadEnabled(), true);
  assert.ok(app.nodes.some(node => node.sent.some(message => message.type === 'clear')));
  await app.ui.stopSession({ terminal: false });
});


test('Qwen runtime errors do not suggest changing a working deployment', () => {
  const p = profiles['qwen3-turn']({ adapter: 'vad' });
  const message = 'playback.ack arrived after a later user input was committed.';
  assert.equal(p.mapEvent({ type: 'error', code: 'playback_ack_too_late', error: message }).message, message);
});

test('interrupted playback reports its cursor before clear resets it', () => {
  let Playback;
  const messages = [];
  const ctx = vm.createContext({
    sampleRate: 24000,
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: m => messages.push(m) }; } },
    registerProcessor: (_name, cls) => { Playback = cls; },
  });
  vm.runInContext(fs.readFileSync(path.join(root, 'playback_worklet.js'), 'utf8'), ctx);
  const player = new Playback();
  player.handleMessage({ type: 'audio', responseId: 'old', pcm: new Int16Array(24000), initialBufferMs: 0 });
  player.process([], [[new Float32Array(2400)]]);
  player.handleMessage({ type: 'clear' });
  const report = messages.find(m => m.type === 'playback-stopped');
  assert.equal(report.responseId, 'old');
  assert.equal(report.playedMs, 100);
  assert.equal(player.playedFrames, 0);
  assert.equal(player.bufferedFrames(), 0);
  player.handleMessage({ type: 'clear' });
  assert.equal(messages.filter(m => m.type === 'playback-stopped').length, 1);
});

test('late interruption cursor acknowledges old response without finishing the new response', async () => {
  const app = shell('qwen3-turn', 'vad');
  await app.ui.startSession();
  const player = app.nodes.find(n => n.name === 'fullduplex-pcm-playback');
  await app.ui.handleEvent({ type: 'response.created', response: { id: 'old' } });
  await app.ui.handleEvent({ type: 'output_audio_buffer.cleared', response_id: 'old' });
  await app.ui.handleEvent({ type: 'response.created', response: { id: 'new' } });
  player.port.onmessage({ data: { type: 'playback-stopped', responseId: 'old', playedMs: 100 } });
  const ack = app.sockets[0].sent.find(m => m.type === 'playback.ack');
  assert.equal(ack.response_id, 'old');
  assert.equal(ack.played_ms, 100);
  assert.equal(app.ui.state().assistantActive, true);
});


test('Qwen camera frames ride the OpenAI image interface, not the audio append', () => {
  const p = profiles['qwen3-turn']({ ...config, adapter: 'vad' });
  assert.deepEqual(plain(p.append('PCM', 'JPEG')), { type: 'input_audio_buffer.append', audio: 'PCM' });

  const [created] = plain(p.imageMessages('JPEG'));
  assert.equal(created.type, 'conversation.item.create');
  assert.equal(created.item.role, 'user');
  assert.deepEqual(created.item.content, [
    { type: 'input_image', image_url: 'data:image/jpeg;base64,JPEG' },
  ]);
});

test('the camera retires its oldest image instead of exhausting the session budget', () => {
  const p = profiles['qwen3-turn']({ ...config, adapter: 'vad' });
  const ids = [];
  for (let i = 0; i < 8; i++) {
    const messages = plain(p.imageMessages('JPEG'));
    assert.equal(messages.length, 1, 'nothing to retire while under the budget');
    ids.push(messages[0].item.id);
  }
  // The engine counts what it already stores, so the delete has to lead.
  const ninth = plain(p.imageMessages('JPEG'));
  assert.deepEqual(ninth[0], { type: 'conversation.item.delete', item_id: ids[0] });
  assert.equal(ninth[1].type, 'conversation.item.create');
  assert.equal(new Set(ids).size, 8, 'item ids must be distinct to be deletable');

  // A new call reopens the budget; stale ids from the old one must not linger.
  assert.equal(plain(p.initialMessages(config, 'x')).length > 0, true);
  assert.equal(plain(p.imageMessages('JPEG')).length, 1);
});

// Exercise the real WebSocket dispatch queue together with the actual playback
// processor: sending a clear message alone does not prove the speaker is silent.
function receive(app, event) {
  app.sockets[0].onmessage({ data: JSON.stringify(event) });
}

function audioChunk(responseId, extra = {}) {
  const pcm = new Int16Array(24000).fill(12000);
  return { type: 'response.output_audio.delta', response_id: responseId,
    delta: Buffer.from(pcm.buffer).toString('base64'), format: 'pcm16', sample_rate_hz: 24000, ...extra };
}

function renderPlayback(player) {
  const output = new Float32Array(128);
  player.process([], [[output]]);
  return output;
}

async function playingQwen(options = {}) {
  const app = shell('qwen3-turn', 'vad', { realPlayback: true, ...options });
  await app.ui.startSession();
  receive(app, { type: 'response.created', response: { id: 'old' } });
  receive(app, audioChunk('old'));
  await app.ui.awaitQueue();
  app.player = app.nodes.find(node => node.name === 'fullduplex-pcm-playback').player;
  // Advance through the initial 400 ms playback buffer.
  for (let index = 0; index < 75; index++) renderPlayback(app.player);
  assert.ok(renderPlayback(app.player).some(sample => sample !== 0));
  return app;
}

function assertSilent(player) {
  assert.equal(player.bufferedFrames(), 0, 'interrupted audio must leave the playback queue');
  assert.ok(renderPlayback(player).every(sample => sample === 0), 'speaker must output silence');
}

for (const terminal of [false, true]) {
  test(`Qwen speech interrupts actual playback after generation completed=${terminal}`, async () => {
    const app = await playingQwen();
    if (terminal) {
      receive(app, { type: 'response.output_audio.done', response_id: 'old' });
      receive(app, { type: 'response.done', response: { id: 'old', status: 'completed' } });
      await app.ui.awaitQueue();
    }
    receive(app, { type: 'input_audio_buffer.speech_started', item_id: 'next', audio_start_ms: 1000 });
    await app.ui.awaitQueue();
    assertSilent(app.player);
    assert.equal(app.ui.microphoneUploadEnabled(), true);
    const ack = app.sockets[0].sent.find(event => event.type === 'playback.ack');
    assert.equal(ack.response_id, 'old');
    assert.equal(ack.played_ms, 5, 'ack only the 128 frames actually played, not all queued audio');
    await app.ui.stopSession({ terminal: false });
  });
}

test('Qwen cancellation preempts pending decode and does not delay the next response', async () => {
  let releaseDecode;
  const app = await playingQwen({ decodeAudioData: () => new Promise(resolve => { releaseDecode = resolve; }) });
  receive(app, audioChunk('old', { format: 'wav' }));
  await flushTasks();
  assert.equal(typeof releaseDecode, 'function');
  receive(app, { type: 'response.done', response: { id: 'old', status: 'cancelled' } });
  await flushTasks();
  assertSilent(app.player);
  receive(app, { type: 'response.created', response: { id: 'new' } });
  receive(app, audioChunk('new'));
  await flushTasks();
  assert.equal(app.player.activeResponseId, 'new');
  assert.equal(app.player.bufferedFrames(), 24000);
  releaseDecode({ sampleRate: 24000, getChannelData: () => new Float32Array(24000).fill(0.4) });
  await flushTasks();
  assert.equal(app.player.bufferedFrames(), 24000, 'old decoded audio must not enter the new response');
  await app.ui.stopSession({ terminal: false });
});

test('Qwen interruption rejects queued and late audio for the cancelled response', async () => {
  const app = await playingQwen();
  receive(app, audioChunk('old'));
  receive(app, { type: 'response.done', response: { id: 'old', status: 'cancelled' } });
  receive(app, audioChunk('old'));
  await app.ui.awaitQueue();
  assertSilent(app.player);
  await app.ui.stopSession({ terminal: false });
});

test('late cancellation of an old Qwen response does not clear a newer response', async () => {
  const app = await playingQwen();
  receive(app, { type: 'response.done', response: { id: 'old', status: 'cancelled' } });
  await app.ui.awaitQueue();
  receive(app, { type: 'response.created', response: { id: 'new' } });
  receive(app, audioChunk('new'));
  await app.ui.awaitQueue();
  receive(app, { type: 'output_audio_buffer.cleared', response_id: 'old' });
  receive(app, { type: 'response.done', response: { id: 'old', status: 'cancelled' } });
  await app.ui.awaitQueue();
  assert.equal(app.player.activeResponseId, 'new');
  assert.equal(app.player.bufferedFrames(), 24000);
  assert.equal(app.ui.state().assistantActive, true);
  await app.ui.stopSession({ terminal: false });
});

test('Qwen speech interruption follows accepted turn detection; MiniCPM mapping stays unchanged', () => {
  const qwen = profiles['qwen3-turn']({ adapter: 'vad' });
  const speech = { type: 'input_audio_buffer.speech_started', item_id: 'next' };
  assert.equal(qwen.mapEvent(speech).kind, 'interrupt');
  qwen.mapEvent({ type: 'session.updated', session: { audio: { input: { turn_detection: {
    type: 'server_vad', interrupt_response: false,
  } } } } });
  assert.equal(qwen.mapEvent(speech).kind, 'ignore');
  qwen.mapEvent({ type: 'session.updated', session: { audio: { input: { turn_detection: null } } } });
  assert.equal(qwen.mapEvent(speech).kind, 'ignore');
  qwen.mapEvent({ type: 'session.updated', session: { audio: { input: { turn_detection: {
    type: 'server_vad', interrupt_response: true,
  } } } } });
  assert.equal(qwen.mapEvent(speech).kind, 'interrupt');
  assert.equal(profiles['qwen3-turn']({ adapter: 'stt' }).mapEvent(speech).kind, 'ignore');
  assert.equal(profiles['minicpm-native']().mapEvent(speech).kind, 'ignore');
});

test('late old-response controls cannot invalidate a newer pending decode', async () => {
  let releaseDecode;
  const app = await playingQwen({ decodeAudioData: () => new Promise(resolve => { releaseDecode = resolve; }) });
  receive(app, { type: 'input_audio_buffer.speech_started', item_id: 'next', audio_start_ms: 1000 });
  await app.ui.awaitQueue();
  receive(app, { type: 'response.created', response: { id: 'new' } });
  receive(app, audioChunk('new', { format: 'wav' }));
  await flushTasks();
  assert.equal(typeof releaseDecode, 'function');
  receive(app, { type: 'output_audio_buffer.cleared', response_id: 'old' });
  receive(app, { type: 'response.done', response: { id: 'old', status: 'cancelled' } });
  releaseDecode({ sampleRate: 24000, getChannelData: () => new Float32Array(24000).fill(0.4) });
  await app.ui.awaitQueue();
  assert.equal(app.player.activeResponseId, 'new');
  assert.equal(app.player.bufferedFrames(), 24000);
  assert.equal(app.ui.state().assistantActive, true);
  await app.ui.stopSession({ terminal: false });
});

test('normal completion still waits for decoding before draining playback', async () => {
  let releaseDecode;
  const app = await playingQwen({ decodeAudioData: () => new Promise(resolve => { releaseDecode = resolve; }) });
  receive(app, audioChunk('old', { format: 'wav' }));
  await flushTasks();
  receive(app, { type: 'response.output_audio.done', response_id: 'old' });
  receive(app, { type: 'response.done', response: { id: 'old', status: 'completed' } });
  await flushTasks();
  assert.equal(app.player.drain, null);
  releaseDecode({ sampleRate: 24000, getChannelData: () => new Float32Array(24000).fill(0.4) });
  await app.ui.awaitQueue();
  assert.equal(app.player.bufferedFrames(), 47872);
  assert.equal(app.player.drain.responseId, 'old');
  await app.ui.stopSession({ terminal: false });
});

test('interrupting an old decode preserves a newer response already waiting in the event queue', async () => {
  let releaseDecode;
  const app = await playingQwen({ decodeAudioData: () => new Promise(resolve => { releaseDecode = resolve; }) });
  receive(app, audioChunk('old', { format: 'wav' }));
  await flushTasks();
  receive(app, { type: 'response.created', response: { id: 'new' } });
  receive(app, audioChunk('new'));
  receive(app, { type: 'output_audio_buffer.cleared', response_id: 'old' });
  await flushTasks();
  assert.equal(app.player.activeResponseId, 'new');
  assert.equal(app.player.bufferedFrames(), 24000);
  releaseDecode({ sampleRate: 24000, getChannelData: () => new Float32Array(24000).fill(0.4) });
  await flushTasks();
  assert.equal(app.player.bufferedFrames(), 24000);
  await app.ui.stopSession({ terminal: false });
});

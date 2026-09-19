(() => {
  'use strict';

  const config = window.OMNI_REALTIME_CONFIG || window.FULL_DUPLEX_CONFIG || {};
  const profile = window.OmniRealtimeProfiles[config.profile || 'minicpm-native'](config);
  const callButton = document.getElementById('callButton');
  const sendTurnButton = document.getElementById('sendTurnButton');
  const muteButton = document.getElementById('muteButton');
  const cameraButton = document.getElementById('cameraButton');
  const cameraPreview = document.getElementById('cameraPreview');
  const promptPreset = document.getElementById('promptPreset');
  const systemPromptInput = document.getElementById('systemPrompt');
  const connectionState = document.getElementById('connectionState');
  const modelState = document.getElementById('modelState');
  const playbackState = document.getElementById('playbackState');
  const sessionTimer = document.getElementById('sessionTimer');
  const meterFill = document.getElementById('meterFill');
  const conversation = document.getElementById('conversation');
  const emptyConversation = document.getElementById('emptyConversation');
  const eventLog = document.getElementById('eventLog');
  const eventCount = document.getElementById('eventCount');
  const runtimeDetail = document.getElementById('runtimeDetail');
  const clearLogButton = document.getElementById('clearLogButton');

  const INPUT_RATE = 16000;
  const OUTPUT_RATE = 24000;
  const SEND_INTERVAL_MS = 200;
  const ECHO_GUARD_MS = 300;
  const INITIAL_PLAYBACK_BUFFER_MS = 400;
  const SESSION_CLOSE_TIMEOUT_MS = 1000;

  const PROMPT_PRESETS = profile.presets;
  document.title = profile.title;
  document.getElementById('pageTitle').textContent = profile.title;
  document.getElementById('pageEyebrow').textContent = profile.eyebrow;
  document.getElementById('profileDescription').textContent = profile.description;
  document.getElementById('policyLabel').textContent = profile.policy;
  cameraButton.hidden = !profile.camera;
  sendTurnButton.hidden = !profile.clientCommit;
  promptPreset.replaceChildren();
  for (const name of [...Object.keys(PROMPT_PRESETS), 'custom']) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name.replaceAll('_', ' ');
    promptPreset.appendChild(option);
  }
  systemPromptInput.value = Object.values(PROMPT_PRESETS)[0];
  document.getElementById('promptControls').hidden = !profile.instructions;

  let socket = null;
  let closingSocket = null;
  let mediaStream = null;
  let captureContext = null;
  let captureNode = null;
  let playbackContext = null;
  let playbackNode = null;
  let sendTimer = null;
  let clockTimer = null;
  let startedAt = 0;
  let running = false;
  let muted = false;
  let assistantActive = false;
  let captureRate = INPUT_RATE;
  let cameraStream = null;
  let cameraTimer = null;
  let cameraPendingFrame = null;
  const cameraCanvas = document.createElement('canvas');
  let playbackRate = OUTPUT_RATE;
  let pendingCapture = [];
  let currentResponseId = null;
  let responseHasAudio = false;
  let logCount = 0;
  let liveUserTurn = null;
  let liveAssistantTurn = null;
  let sessionCloseResolver = null;
  let responseComplete = false;
  let playbackComplete = true;
  let turnSubmitted = false;
  let echoTimer = null;
  let sessionGeneration = 0;
  let audioChain = Promise.resolve();
  let assistantTextChannel = null;
  let connectionReady = false;
  let turnCounter = 0;
  let stopping = null;
  let turnTimeout = null;

  function markBusy() {
    assistantActive = true;
    if (profile.halfDuplex) pendingCapture = [];
  }

  function armTurnTimeout() {
    clearTimeout(turnTimeout);
    turnTimeout = window.setTimeout(() => {
      failSession('Timed out waiting for the model response. Check the backend event log and restart the session.');
    }, 120000);
  }

  async function failSession(message) {
    appendLog(message, true);
    await stopSession({ terminal: false });
    setConnection('Error', 'error');
    runtimeDetail.textContent = message;
  }

  function finishResponseIfReady() {
    if (!responseComplete || !playbackComplete || echoTimer !== null) return;
    const generation = sessionGeneration;
    echoTimer = window.setTimeout(async () => {
      echoTimer = null;
      if (!running || generation !== sessionGeneration) return;
      if (profile.reconnectEachTurn) {
        connectionReady = false;
        const previous = socket;
        socket = null;
        if (previous) { previous.onclose = null; previous.onmessage = null; previous.close(); }
        try { await openSocket(); }
        catch (error) { await failSession(error.message); return; }
        if (!running || generation !== sessionGeneration) return;
      }
      assistantActive = false;
      responseComplete = false;
      turnSubmitted = false;
      currentResponseId = null;
      responseHasAudio = false;
      assistantTextChannel = null;
      if (profile.halfDuplex) pendingCapture = [];
      sendTurnButton.disabled = !profile.clientCommit;
      setModel(profile.waiting);
    }, ECHO_GUARD_MS);
  }

  function staticAssetUrl(path) {
    const version = String(config.appVersion || '').trim();
    return version ? `${path}?v=${encodeURIComponent(version)}` : path;
  }

  if (promptPreset && systemPromptInput) {
    promptPreset.addEventListener('change', () => {
      const preset = PROMPT_PRESETS[promptPreset.value];
      if (preset !== undefined) systemPromptInput.value = preset;
    });
    systemPromptInput.addEventListener('input', () => {
      promptPreset.value = 'custom';
    });
  }

  function realtimeUrl() {
    return profile.url(config, window.location.href);
  }

  function setConnection(label, kind) {
    connectionState.textContent = label;
    connectionState.className = `status status-${kind}`;
  }

  function setModel(label) {
    modelState.textContent = label;
  }

  function setPlayback(label) {
    playbackState.textContent = label;
  }

  function compactEvent(event) {
    const fields = [];
    const payload = event.event || event;
    const responseId = responseIdOf(event);
    if (responseId) fields.push(`response=${responseId}`);
    if (event.item_id) fields.push(`item=${event.item_id}`);
    if (event.code) fields.push(`code=${event.code}`);
    if (event.response && event.response.status) fields.push(`status=${event.response.status}`);
    if (payload.played_ms !== undefined) fields.push(`played=${payload.played_ms}ms`);
    if (payload.committed_ms !== undefined) fields.push(`committed=${payload.committed_ms}ms`);
    if (payload.history_committed !== undefined) fields.push(`history=${payload.history_committed}`);
    return fields.join(' ');
  }

  function appendLog(message, error = false) {
    const time = new Date().toLocaleTimeString([], { hour12: false });
    const line = document.createElement('span');
    if (error) line.className = 'log-error';
    line.textContent = `${time}  ${message}\n`;
    eventLog.appendChild(line);
    eventLog.scrollTop = eventLog.scrollHeight;
    logCount += 1;
    eventCount.textContent = `${logCount} ${logCount === 1 ? 'event' : 'events'}`;
  }

  function appendEventLog(event) {
    const detail = compactEvent(event);
    appendLog(`${event.type || 'unknown'}${detail ? `  ${detail}` : ''}`, event.type === 'error');
  }

  function responseIdOf(event) {
    return event.response_id || (event.response && event.response.id) || null;
  }

  function ensureTurn(role) {
    const existing = role === 'user' ? liveUserTurn : liveAssistantTurn;
    if (existing) return existing;
    if (emptyConversation) emptyConversation.remove();
    const row = document.createElement('div');
    row.className = `turn turn-${role} turn-live`;
    const label = document.createElement('div');
    label.className = 'turn-role';
    label.textContent = role === 'user' ? 'You' : 'Assistant';
    const text = document.createElement('div');
    text.className = 'turn-text';
    row.append(label, text);
    conversation.appendChild(row);
    conversation.scrollTop = conversation.scrollHeight;
    const turn = { row, text, value: '' };
    if (role === 'user') liveUserTurn = turn;
    else liveAssistantTurn = turn;
    return turn;
  }

  function addTranscript(role, delta) {
    if (!delta) return;
    const turn = ensureTurn(role);
    turn.value += delta;
    turn.text.textContent = turn.value;
    conversation.scrollTop = conversation.scrollHeight;
  }

  function finishTranscript(role, finalText = '') {
    const turn = role === 'user' ? liveUserTurn : liveAssistantTurn;
    if (!turn && !finalText) return;
    const current = turn || ensureTurn(role);
    if (finalText) {
      current.value = finalText;
      current.text.textContent = finalText;
    }
    current.row.classList.remove('turn-live');
    if (role === 'user') liveUserTurn = null;
    else liveAssistantTurn = null;
  }

  function bytesToBase64(bytes) {
    let binary = '';
    const chunkSize = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
    }
    return btoa(binary);
  }

  function int16ToBase64(pcm) {
    return bytesToBase64(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength));
  }

  function base64ToBytes(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return bytes;
  }

  function resampleInt16(input, sourceRate, targetRate) {
    if (sourceRate === targetRate) return input;
    const ratio = sourceRate / targetRate;
    const output = new Int16Array(Math.floor(input.length / ratio));
    for (let index = 0; index < output.length; index += 1) {
      const start = Math.floor(index * ratio);
      const end = Math.max(start + 1, Math.min(input.length, Math.floor((index + 1) * ratio)));
      let sum = 0;
      for (let source = start; source < end; source += 1) sum += input[source];
      output[index] = sum / (end - start);
    }
    return output;
  }

  async function decodeAudioDelta(event) {
    const encoded = event.delta || (event.response && event.response.audio);
    if (!encoded) return null;
    const bytes = base64ToBytes(encoded);
    const format = String(event.format || event.audio_format || 'pcm16').toLowerCase();
    const sourceRate = Number(event.sample_rate_hz || event.sample_rate || OUTPUT_RATE);
    if (format.includes('f32')) {
      const floats = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
      const pcm = new Int16Array(floats.length);
      for (let index = 0; index < floats.length; index += 1) {
        const sample = Math.max(-1, Math.min(1, floats[index]));
        pcm[index] = sample < 0 ? sample * 32768 : sample * 32767;
      }
      return { pcm, sourceRate };
    }
    if (format.includes('wav')) {
      const decoded = await playbackContext.decodeAudioData(bytes.buffer.slice(0));
      const channel = decoded.getChannelData(0);
      const pcm = new Int16Array(channel.length);
      for (let index = 0; index < channel.length; index += 1) {
        const sample = Math.max(-1, Math.min(1, channel[index]));
        pcm[index] = sample < 0 ? sample * 32768 : sample * 32767;
      }
      return { pcm, sourceRate: decoded.sampleRate };
    }
    return {
      pcm: new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2)),
      sourceRate,
    };
  }

  function updateMeter(pcm) {
    let peak = 0;
    for (let index = 0; index < pcm.length; index += 8) peak = Math.max(peak, Math.abs(pcm[index]));
    meterFill.style.width = `${Math.min(100, (peak / 32768) * 150).toFixed(0)}%`;
  }

  function microphoneUploadEnabled() {
    return running && connectionReady && !muted && (!profile.halfDuplex || !assistantActive);
  }

  function flushCapture() {
    if (!socket || socket.readyState !== WebSocket.OPEN || pendingCapture.length === 0) return;
    if (!microphoneUploadEnabled()) {
      pendingCapture = [];
      return;
    }
    const length = pendingCapture.reduce((total, chunk) => total + chunk.length, 0);
    const merged = new Int16Array(length);
    let offset = 0;
    for (const chunk of pendingCapture) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    pendingCapture = [];
    const pcm = resampleInt16(merged, captureRate, profile.inputSampleRate || INPUT_RATE);
    const appendEvent = profile.append(int16ToBase64(pcm), cameraPendingFrame);
    cameraPendingFrame = null;
    socket.send(JSON.stringify(appendEvent));
  }

  function beginAssistant(responseId) {
    currentResponseId = responseId || currentResponseId;
    responseHasAudio = false;
    assistantActive = true;
    responseComplete = false;
    playbackComplete = true;
    if (profile.halfDuplex) pendingCapture = [];
    setModel(profile.halfDuplex ? 'Thinking' : 'Speaking');
  }

  function feedPlayback(decoded, responseId) {
    if (!decoded || !decoded.pcm || decoded.pcm.length === 0 || !playbackNode) return;
    const pcm = resampleInt16(decoded.pcm, decoded.sourceRate, playbackRate);
    responseHasAudio = true;
    playbackComplete = false;
    assistantActive = true;
    setPlayback('Buffering');
    playbackNode.port.postMessage({
      type: 'audio',
      pcm,
      responseId: responseId || currentResponseId,
      initialBufferMs: INITIAL_PLAYBACK_BUFFER_MS,
    }, [pcm.buffer]);
  }

  function requestPlaybackDrain(responseId) {
    if (!responseHasAudio) { playbackComplete = true; finishResponseIfReady(); return; }
    if (!playbackNode) return;
    playbackNode.port.postMessage({ type: 'drain', responseId: responseId || currentResponseId });
  }

  function sendPlaybackAck(responseId, playedMs) {
    if (!responseId || !socket || socket.readyState !== WebSocket.OPEN || playedMs <= 0) {
      if (!responseId && playedMs > 0) appendLog('playback ack skipped: missing response id', true);
      return;
    }
    const ack = profile.ack(responseId, playedMs);
    if (ack) socket.send(JSON.stringify(ack));
  }

  function playbackDrained(message) {
    const responseId = message.responseId || currentResponseId;
    // A response that finishes playing after the next one has started still
    // owes its ack: that ack is what commits its playback into history. Only
    // the UI state belongs to the response currently on screen.
    if (profile.playbackAck) sendPlaybackAck(responseId, Number(message.playedMs) || 0);
    if (message.underrunMs > 0) appendLog(`playback underrun ${message.underrunMs} ms`);
    if (currentResponseId && responseId !== currentResponseId) return;
    setPlayback('Idle');
    playbackComplete = true;
    if (!profile.halfDuplex && !profile.waitForResponseDone) responseComplete = true;
    finishResponseIfReady();
  }

  async function handleAudioEvent(action) {
    markBusy();
    currentResponseId = action.responseId || currentResponseId || `turn-${turnCounter}`;
    setModel('Speaking');
    const generation = sessionGeneration;
    const decoded = await decodeAudioDelta(action.event);
    if (generation === sessionGeneration) feedPlayback(decoded, currentResponseId);
  }

  function handleTranscriptEvent(action) {
    if (action.role === 'assistant') {
      // Qwen may emit both text and audio-transcript representations.
      // Display one channel per response instead of duplicating the answer.
      if (profile.deduplicateTranscript && assistantTextChannel && action.channel !== assistantTextChannel) return;
      if (action.text) assistantTextChannel = action.channel || assistantTextChannel;
    }
    if (action.kind === 'text') addTranscript(action.role, action.text);
    else if (profile.deduplicateTranscript && action.role === 'assistant') {
      if (action.text) {
        const turn = ensureTurn('assistant');
        turn.value = action.text;
        turn.text.textContent = action.text;
      }
    } else finishTranscript(action.role, action.text);
  }

  async function handleEvent(event) {
    appendEventLog(event);
    const action = profile.mapEvent(event);
    const responseId = action.responseId;
    switch (action.kind) {
      case 'connected':
        setConnection('Connected', 'online');
        if (!assistantActive) setModel(profile.waiting);
        break;
      case 'interrupt':
        sessionGeneration += 1;
        // A cancelled response never reaches 'done', so nothing else would
        // disarm the turn watchdog it left armed.
        clearTimeout(turnTimeout);
        if (playbackNode) playbackNode.port.postMessage({ type: 'clear' });
        responseComplete = true;
        playbackComplete = true;
        responseHasAudio = false;
        assistantActive = false;
        currentResponseId = null;
        assistantTextChannel = null;
        finishTranscript('assistant');
        setPlayback('Idle');
        setModel(profile.waiting);
        break;
      case 'listen':
        assistantActive = false;
        setModel(profile.waiting);
        break;
      case 'begin':
        if (profile.halfDuplex) armTurnTimeout();
        if (echoTimer !== null) { clearTimeout(echoTimer); echoTimer = null; }
        beginAssistant(responseId);
        break;
      case 'audio':
        await handleAudioEvent(action);
        break;
      case 'drain':
        requestPlaybackDrain(responseId);
        break;
      case 'text': case 'text-final':
        handleTranscriptEvent(action);
        break;
      case 'done':
        clearTimeout(turnTimeout);
        responseComplete = true;
        finishTranscript('assistant');
        requestPlaybackDrain(responseId);
        finishResponseIfReady();
        break;
      case 'backpressure':
        armTurnTimeout();
        markBusy();
        setModel('Thinking / Speaking');
        runtimeDetail.textContent = action.message;
        break;
      case 'ack':
        runtimeDetail.textContent = `Playback committed ${action.committedMs} ms`;
        break;
      case 'closed':
        if (sessionCloseResolver) sessionCloseResolver();
        sessionCloseResolver = null;
        break;
      case 'error':
        if (action.fatal) { await failSession(action.message); break; }
        // A rejected frame or a refused update costs one operation, not the
        // call: report it and keep listening.
        appendLog(action.message, true);
        runtimeDetail.textContent = action.message;
        break;
      default: break;
    }
  }

  async function openPlayback() {
    playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: OUTPUT_RATE });
    playbackRate = playbackContext.sampleRate;
    await playbackContext.audioWorklet.addModule(staticAssetUrl('static/playback_worklet.js'));
    playbackNode = new AudioWorkletNode(playbackContext, 'fullduplex-pcm-playback');
    const currentPlaybackNode = playbackNode;
    playbackNode.port.onmessage = (message) => {
      if (playbackNode !== currentPlaybackNode) return;
      if (message.data.type === 'playback-started') setPlayback('Playing');
      else if (message.data.type === 'playback-stopped') {
        // The old response may report after the next one has started. ACK its
        // own cursor without changing the current response's UI state.
        if (profile.playbackAck) sendPlaybackAck(message.data.responseId, Number(message.data.playedMs) || 0);
      }
      else if (message.data.type === 'playback-drained') playbackDrained(message.data);
      else if (message.data.type === 'playback-underrun') {
        runtimeDetail.textContent = `Playback underrun ${message.data.underrunMs || 0} ms`;
      }
    };
    playbackNode.connect(playbackContext.destination);
    await playbackContext.resume();
  }

  async function openCapture() {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: { ideal: INPUT_RATE },
      },
    });
    try {
      captureContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: INPUT_RATE });
    } catch (_error) {
      captureContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    captureRate = captureContext.sampleRate;
    await captureContext.audioWorklet.addModule(staticAssetUrl('static/pcm_worklet.js'));
    const source = captureContext.createMediaStreamSource(mediaStream);
    captureNode = new AudioWorkletNode(captureContext, 'fullduplex-pcm-capture');
    captureNode.port.onmessage = (message) => {
      const pcm = new Int16Array(message.data);
      updateMeter(pcm);
      if (microphoneUploadEnabled()) pendingCapture.push(pcm);
    };
    const silentSink = captureContext.createGain();
    silentSink.gain.value = 0;
    source.connect(captureNode);
    captureNode.connect(silentSink).connect(captureContext.destination);
    await captureContext.resume();
  }

  function openSocket() {
    return new Promise((resolve, reject) => {
      const url = realtimeUrl();
      const current = new WebSocket(url);
      socket = current;
      connectionReady = false;
      turnCounter += 1;
      let settled = false;
      const timer = window.setTimeout(() => {
        if (settled) return;
        settled = true;
        current.onclose = null;
        current.close();
        reject(new Error(`Session handshake timed out. ${profile.connectionHint}`));
      }, 15000);
      const rejectOnce = (message) => {
        clearTimeout(timer);
        if (!settled) { settled = true; reject(new Error(message)); }
      };
      current.onopen = () => {
        const instructions = systemPromptInput.value.trim();
        for (const event of profile.initialMessages(config, instructions)) current.send(JSON.stringify(event));
        appendLog(`websocket open  ${url}`);
      };
      current.onmessage = (message) => {
        if (typeof message.data !== 'string' || (socket !== current && closingSocket !== current)) return;
        let event;
        try { event = JSON.parse(message.data); }
        catch (error) {
          if (socket !== current) return;
          const detail = `Invalid server event: ${error.message}`;
          if (!settled) rejectOnce(detail);
          else void failSession(detail);
          return;
        }
        // Shutdown has detached the active socket. Only its close acknowledgement
        // may pass, directly: the event queue can itself be awaiting cleanup.
        if (closingSocket === current) {
          if (event?.type === 'session.closed' && sessionCloseResolver) sessionCloseResolver();
          return;
        }
        const action = profile.mapEvent(event);
        if (action.kind === 'error' && !settled) {
          rejectOnce(action.message);
          return;
        }
        if (event.type === profile.readyEvent && !settled) {
          clearTimeout(timer);
          settled = true;
          connectionReady = true;
          runtimeDetail.textContent = `${captureRate} Hz capture / ${playbackRate} Hz playback`;
          resolve();
        }
        // Serialize decoding with terminal events: a drain must never overtake
        // an asynchronously decoded WAV chunk.
        audioChain = audioChain.then(() => {
          if (socket === current || action.kind === 'closed') return handleEvent(event);
        }).catch((error) => failSession(`Server event failed: ${error.message}`));
      };
      current.onerror = () => rejectOnce(`WebSocket connection failed. ${profile.connectionHint}`);
      current.onclose = (event) => {
        clearTimeout(timer);
        connectionReady = false;
        appendLog(`websocket closed  code=${event.code}`);
        if (!settled) rejectOnce(`Connection closed before session ready. ${profile.connectionHint}`);
        else if (running && socket === current) failSession(`Backend disconnected (${event.code}). ${profile.connectionHint}`);
      };
    });
  }

  function formatElapsed(seconds) {
    const minutes = Math.floor(seconds / 60);
    return `${String(minutes).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  }

  function startClock() {
    startedAt = Date.now();
    clockTimer = window.setInterval(() => {
      sessionTimer.textContent = formatElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
  }

  async function startSession() {
    if (running) return;
    callButton.disabled = true;
    setConnection('Connecting', 'connecting');
    runtimeDetail.textContent = 'Requesting microphone access';
    try {
      sessionGeneration += 1;
      audioChain = Promise.resolve();
      await openPlayback();
      await openCapture();
      await openSocket();
      running = true;
      muted = false;
      assistantActive = false;
      sendTimer = window.setInterval(flushCapture, profile.sendIntervalMs || SEND_INTERVAL_MS);
      startClock();
      callButton.textContent = 'End session';
      callButton.classList.add('is-active');
      muteButton.disabled = false;
      cameraButton.disabled = !profile.camera;
      sendTurnButton.disabled = !profile.clientCommit;
      setConnection('Connected', 'online');
      setModel(profile.waiting);
      appendLog('session started');
    } catch (error) {
      appendLog(`start failed: ${error.message || error}`, true);
      await stopSession({ terminal: false });
      setConnection('Error', 'error');
      runtimeDetail.textContent = String(error.message || error);
    } finally {
      callButton.disabled = false;
    }
  }

  async function startCamera() {
    if (!profile.camera || cameraStream) return;
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraPreview.srcObject = cameraStream;
    cameraPreview.style.display = '';
    await cameraPreview.play().catch(() => {});
    // Official omni-duplex cadence: one JPEG (quality 0.7) per ~1 s chunk,
    // no client-side resize (the server normalizes at scale_resolution=448).
    const captureCameraFrame = () => {
      if (!cameraStream || cameraPreview.videoWidth === 0) return;
      const scale = profile.cameraMaxDimension
        ? Math.min(1, profile.cameraMaxDimension / Math.max(cameraPreview.videoWidth, cameraPreview.videoHeight)) : 1;
      cameraCanvas.width = Math.max(1, Math.round(cameraPreview.videoWidth * scale));
      cameraCanvas.height = Math.max(1, Math.round(cameraPreview.videoHeight * scale));
      cameraCanvas.getContext('2d').drawImage(cameraPreview, 0, 0, cameraCanvas.width, cameraCanvas.height);
      const frame = cameraCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
      if (profile.imageMessages) {
        if (connectionReady && socket?.readyState === WebSocket.OPEN) {
          for (const event of profile.imageMessages(frame)) socket.send(JSON.stringify(event));
        }
      } else cameraPendingFrame = frame;
    };
    // Do not make the first spoken turn race a one-second timer.
    captureCameraFrame();
    cameraTimer = window.setInterval(captureCameraFrame, 1000);
    cameraButton.textContent = 'Camera off';
    cameraButton.classList.add('is-active');
    appendLog('camera on (1 fps omni frames)');
  }

  function stopCamera() {
    if (cameraTimer !== null) clearInterval(cameraTimer);
    cameraTimer = null;
    if (cameraStream) {
      for (const track of cameraStream.getTracks()) track.stop();
    }
    cameraStream = null;
    cameraPendingFrame = null;
    cameraPreview.srcObject = null;
    cameraPreview.style.display = 'none';
    cameraButton.textContent = 'Camera';
    cameraButton.classList.remove('is-active');
  }

  cameraButton.addEventListener('click', () => {
    if (cameraStream) {
      stopCamera();
      appendLog('camera off');
      return;
    }
    startCamera().catch((error) => appendLog(`camera failed: ${error.message || error}`, true));
  });

  function waitForSessionClosed(targetSocket, timeoutMs) {
    if (!targetSocket || targetSocket.readyState !== WebSocket.OPEN) return Promise.resolve();
    return new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        if (sessionCloseResolver === finish) sessionCloseResolver = null;
        resolve();
      };
      sessionCloseResolver = finish;
      const timer = window.setTimeout(finish, timeoutMs);
    });
  }

  function stopSession(options = {}) {
    if (stopping) return stopping;
    stopping = cleanupSession(options).finally(() => { stopping = null; });
    return stopping;
  }

  async function cleanupSession({ terminal = true } = {}) {
    sessionGeneration += 1;
    connectionReady = false;
    clearTimeout(echoTimer);
    clearTimeout(turnTimeout);
    echoTimer = null;
    responseComplete = false;
    playbackComplete = true;
    turnSubmitted = false;
    assistantTextChannel = null;
    sendTurnButton.disabled = true;
    running = false;
    assistantActive = false;
    pendingCapture = [];
    if (sendTimer !== null) clearInterval(sendTimer);
    if (clockTimer !== null) clearInterval(clockTimer);
    sendTimer = null;
    clockTimer = null;
    if (socket) {
      closingSocket = socket;
      socket = null;
      closingSocket.onclose = null;
      if (terminal && profile.closeSession && closingSocket.readyState === WebSocket.OPEN) {
        const closed = waitForSessionClosed(closingSocket, SESSION_CLOSE_TIMEOUT_MS);
        closingSocket.send(JSON.stringify({ type: 'session.close' }));
        await closed;
      }
      closingSocket.close(1000, 'client stop');
      closingSocket = null;
    }
    if (playbackNode) playbackNode.port.postMessage({ type: 'clear' });
    if (mediaStream) {
      for (const track of mediaStream.getTracks()) track.stop();
    }
    mediaStream = null;
    stopCamera();
    cameraButton.disabled = true;
    if (captureContext) await captureContext.close().catch(() => {});
    if (playbackContext) await playbackContext.close().catch(() => {});
    captureContext = null;
    captureNode = null;
    playbackContext = null;
    playbackNode = null;
    currentResponseId = null;
    responseHasAudio = false;
    finishTranscript('user');
    finishTranscript('assistant');
    meterFill.style.width = '0%';
    sessionTimer.textContent = '00:00';
    callButton.textContent = 'Start session';
    callButton.classList.remove('is-active');
    muteButton.textContent = 'Mute';
    muteButton.classList.remove('is-active');
    muteButton.disabled = true;
    setConnection('Offline', 'offline');
    setModel('Idle');
    setPlayback('Idle');
    if (runtimeDetail.textContent.startsWith('Playback committed')) return;
    if (!runtimeDetail.textContent.startsWith('start failed')) runtimeDetail.textContent = 'No active connection';
  }

  function toggleMute() {
    if (!running) return;
    muted = !muted;
    pendingCapture = [];
    muteButton.textContent = muted ? 'Unmute' : 'Mute';
    muteButton.classList.toggle('is-active', muted);
    appendLog(muted ? 'microphone muted' : 'microphone unmuted');
  }

  callButton.addEventListener('click', () => {
    if (running) stopSession();
    else startSession();
  });
  sendTurnButton.addEventListener('click', () => {
    if (!running || !connectionReady || !profile.clientCommit || turnSubmitted) return;
    flushCapture();
    for (const event of profile.commitMessages()) socket.send(JSON.stringify(event));
    turnSubmitted = true;
    responseComplete = false;
    markBusy();
    armTurnTimeout();
    sendTurnButton.disabled = true;
    setModel('Thinking');
    appendLog('turn submitted');
  });
  muteButton.addEventListener('click', toggleMute);
  clearLogButton.addEventListener('click', () => {
    eventLog.textContent = '';
    logCount = 0;
    eventCount.textContent = '0 events';
  });
  window.addEventListener('beforeunload', () => { stopSession({ terminal: false }); });
})();

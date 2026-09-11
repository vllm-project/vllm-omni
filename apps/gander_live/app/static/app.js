(() => {
  'use strict';

  const config = window.FULL_DUPLEX_CONFIG || {};
  const callButton = document.getElementById('callButton');
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

  // Default prompts mirroring the official MiniCPM-o-Demo presets
  // (assets/presets/{omni,audio_duplex}/*.yaml).
  const PROMPT_PRESETS = {
    omni: 'Streaming Omni Conversation.',
    chinese_call: '扮演一个具有以上声音特征的助手。请认真、高质量地回复用户的问题。'
      + '请用高自然度的方式和用户聊天。你处于双工模式，可以一边听、一边说。'
      + '你是由面壁智能开发的人工智能助手：面壁小钢炮。',
    english_call: 'Replicate the tone and style from the input audio. Your task is to be '
      + 'a helpful assistant using this voice pattern. Please answer the user\'s questions '
      + 'seriously and in a high quality. Please chat with the user in a high naturalness '
      + 'style. You are in duplex mode, where you can listen and speak at the same time.',
  };

  let micEnergy = 0;
  let speakerEnergy = 0;
  let playbackEpoch = 0;
  const cancelledResponses = new Set();
  const playingResponses = new Set();
  let socket = null;
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
    const url = new URL(config.realtimePath, window.location.href);
    url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    url.searchParams.set('duplex', '1');
    url.searchParams.set('model', config.model || 'openbmb/MiniCPM-o-4_5');
    url.searchParams.set('native_duplex', '1');
    url.searchParams.set('autostart', '0');
    return url.toString();
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
    micEnergy = muted ? 0 : Math.min(1, peak / 10000);
    meterFill.style.width = `${Math.min(100, (peak / 32768) * 150).toFixed(0)}%`;
  }

  function microphoneUploadEnabled() {
    return running;
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
    const pcm = resampleInt16(merged, captureRate, INPUT_RATE);
    const appendEvent = {
      type: 'input_audio_buffer.append',
      audio: int16ToBase64(pcm),
      format: 'pcm16',
      sample_rate_hz: INPUT_RATE,
    };
    // Omni duplex: ~1 fps camera frame rides the audio append (official
    // MiniCPM-o-Demo contract: one base64 JPEG per ~1 s chunk).
    if (cameraPendingFrame) {
      appendEvent.video_frames = [cameraPendingFrame];
      cameraPendingFrame = null;
    }
    socket.send(JSON.stringify(appendEvent));
  }

  function beginAssistant(responseId) {
    currentResponseId = responseId || currentResponseId;
    responseHasAudio = false;
    assistantActive = true;
    setModel('正在回复');
  }

  function feedPlayback(decoded, responseId) {
    if (!decoded || !decoded.pcm || decoded.pcm.length === 0 || !playbackNode) return;
    const pcm = resampleInt16(decoded.pcm, decoded.sourceRate, playbackRate);
    responseHasAudio = true;
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
    if (!playbackNode) return;
    playbackNode.port.postMessage({ type: 'drain', responseId: responseId || currentResponseId });
  }

  function sendPlaybackAck(responseId, playedMs) {
    if (!responseId || !socket || socket.readyState !== WebSocket.OPEN || playedMs <= 0) {
      if (!responseId && playedMs > 0) appendLog('playback ack skipped: missing response id', true);
      return;
    }
    socket.send(JSON.stringify({
      type: 'playback.ack',
      response_id: responseId,
      item_id: `item_${responseId}`,
      played_ms: playedMs,
      committed_ms: playedMs,
    }));
  }

  function playbackDrained(message) {
    const responseId = message.responseId || currentResponseId;
    if (cancelledResponses.has(responseId)) return;
    sendPlaybackAck(responseId, Number(message.playedMs) || 0);
    playingResponses.delete(responseId);
    setPlayback(playingResponses.size ? '正在播放' : '待机');
    if (message.underrunMs > 0) {
      appendLog(`playback underrun ${message.underrunMs} ms`);
    }
    window.setTimeout(() => {
      if (currentResponseId && currentResponseId !== responseId) return;
      assistantActive = false;
      currentResponseId = null;
      responseHasAudio = false;
      if (running) setModel('正在聆听');
    }, ECHO_GUARD_MS);
  }

  function handleEvent(event) {
    appendEventLog(event);
    const native = event.event || event;
    if (event.type.startsWith('local.tool.')) {
      const panel = document.getElementById('toolEvents');
      panel.querySelector('.empty-state')?.remove();
      const card = document.createElement('div'); card.className = 'tool-card';
      const heading = document.createElement('strong');
      heading.textContent = event.type.endsWith('started') ? '模型调用 → 本地执行中' : '结果已发送 → 等待模型回复';
      const body = document.createElement('pre'); body.textContent = JSON.stringify(event.call || event.result, null, 2);
      card.append(heading, body); panel.append(card);
    }
    if (native.type === 'input.context.snapshot' || native.type === 'input.context.replaced') {
      document.getElementById('contextInfo').textContent = JSON.stringify(native, null, 2);
    }
    if (event.type === 'output_audio_buffer.cleared' || (event.type === 'response.done' && event.response?.status === 'cancelled')) {
      const id = responseIdOf(event);
      if (id) {
        cancelledResponses.add(id); playingResponses.delete(id);
        playbackNode?.port.postMessage({type:'cancel',responseId:id});
      } else {
        playbackEpoch++; playingResponses.clear();
        playbackNode?.port.postMessage({type:'clear'});
      }
      assistantActive = false; responseHasAudio = false;
      if (liveAssistantTurn) liveAssistantTurn.row.querySelector('.turn-role').textContent = '助手 · 已打断';
      finishTranscript('assistant'); setPlayback('已停止旧播放');
    }
    const responseId = responseIdOf(event);
    switch (event.type) {
      case 'session.created':
      case 'session.updated':
        setConnection('已连接', 'online');
        setModel('正在聆听');
        break;
      case 'response.listen':
        assistantActive = false;
        setModel('正在聆听');
        break;
      case 'response.created':
      case 'response.speak':
        beginAssistant(responseId);
        break;
      case 'response.audio.delta':
        currentResponseId = responseId || currentResponseId;
        assistantActive = true;
        setModel('正在回复');
        const decodeEpoch = playbackEpoch;
        decodeAudioDelta(event)
          .then((decoded) => { if (decodeEpoch === playbackEpoch && !cancelledResponses.has(responseId)) feedPlayback(decoded, responseId); })
          .catch((error) => appendLog(`audio decode failed: ${error.message || error}`, true));
        break;
      case 'response.audio.done':
        requestPlaybackDrain(responseId);
        break;
      case 'response.audio_transcript.delta':
        if (cancelledResponses.has(responseId)) break;
        addTranscript('assistant', event.delta || '');
        break;
      case 'response.audio_transcript.done':
        if (cancelledResponses.has(responseId)) break;
        finishTranscript('assistant', event.transcript || '');
        break;
      case 'conversation.item.input_audio_transcription.delta':
        addTranscript('user', event.delta || '');
        break;
      case 'conversation.item.input_audio_transcription.completed':
        finishTranscript('user', event.transcript || '');
        break;
      case 'response.done':
        finishTranscript('assistant');
        if (!responseHasAudio && event.response?.status !== 'cancelled') requestPlaybackDrain(responseId);
        break;
      case 'playback.acknowledged':
        {
          const acknowledgement = event.event || event;
          runtimeDetail.textContent = `Playback committed ${acknowledgement.committed_ms || 0} ms`;
        }
        break;
      case 'session.closed':
        if (sessionCloseResolver) sessionCloseResolver();
        sessionCloseResolver = null;
        break;
      case 'error':
        setConnection('Error', 'error');
        runtimeDetail.textContent = JSON.stringify(event.error || event.code || event);
        break;
      default:
        break;
    }
  }

  async function openPlayback() {
    playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: OUTPUT_RATE });
    playbackRate = playbackContext.sampleRate;
    await playbackContext.audioWorklet.addModule(staticAssetUrl('static/playback_worklet.js'));
    playbackNode = new AudioWorkletNode(playbackContext, 'fullduplex-pcm-playback');
    playbackNode.port.onmessage = (message) => {
      if (message.data.type === 'visual-level') { speakerEnergy = message.data.level; return; }
      if (message.data.type === 'playback-started') { playingResponses.add(message.data.responseId); setPlayback('正在播放'); }
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
      if (microphoneUploadEnabled()) pendingCapture.push(muted ? new Int16Array(pcm.length) : pcm);
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
      socket = new WebSocket(url);
      let settled = false;
      socket.onopen = () => {
        settled = true;
        const extraBody = {
          auto_response: true,
          native_duplex: true,
        };
        const session = {
          modalities: ['audio', 'text'],
          voice: 'default',
          turn_detection: null,
          extra_body: extraBody,
        };
        // Reference voice for TTS cloning, provided by the server via
        // --ref-audio (mirrors the official demo's default ref audio).
        if (config.refAudio) session.ref_audio = config.refAudio;
        if (document.getElementById('toolsEnabled').checked) {
          extraBody.realtime_tools = config.tools;
          extraBody.gander_task_slate = '当前没有后台任务。';
        }
        const instructions = systemPromptInput ? systemPromptInput.value.trim() : '';
        if (instructions) session.instructions = instructions;
        socket.send(JSON.stringify({ type: 'session.update', session }));
        runtimeDetail.textContent = `${captureRate} Hz capture / ${playbackRate} Hz playback`;
        appendLog(`websocket open  ${url}`);
        resolve();
      };
      socket.onmessage = (message) => {
        if (typeof message.data !== 'string') return;
        try {
          handleEvent(JSON.parse(message.data));
        } catch (error) {
          appendLog(`invalid server event: ${error.message || error}`, true);
        }
      };
      socket.onerror = () => {
        if (!settled) {
          settled = true;
          reject(new Error(`WebSocket connection failed: ${url}`));
        }
      };
      socket.onclose = (event) => {
        appendLog(`websocket closed  code=${event.code}`);
        if (running) {
          setConnection('Disconnected', 'error');
          stopSession();
        }
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
    setConnection('连接中', 'connecting');
    runtimeDetail.textContent = 'Requesting microphone access';
    try {
      await openPlayback();
      await openCapture();
      await openSocket();
      cancelledResponses.clear(); playbackEpoch++;
      document.getElementById('toolsEnabled').disabled = true;
      running = true;
      muted = false;
      assistantActive = false;
      sendTimer = window.setInterval(flushCapture, SEND_INTERVAL_MS);
      startClock();
      callButton.textContent = '结束对话';
      callButton.classList.add('is-active');
      muteButton.disabled = false;
      cameraButton.disabled = false;
      setConnection('已连接', 'online');
      setModel('正在聆听');
      appendLog('session started');
    } catch (error) {
      appendLog(`start failed: ${error.message || error}`, true);
      setConnection('Error', 'error');
      runtimeDetail.textContent = String(error.message || error);
      await stopSession();
    } finally {
      callButton.disabled = false;
    }
  }

  async function startCamera() {
    if (cameraStream) return;
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraPreview.srcObject = cameraStream;
    cameraPreview.style.display = '';
    await cameraPreview.play().catch(() => {});
    // Official omni-duplex cadence: one JPEG (quality 0.7) per ~1 s chunk,
    // no client-side resize (the server normalizes at scale_resolution=448).
    cameraTimer = window.setInterval(() => {
      if (!cameraStream || cameraPreview.videoWidth === 0) return;
      cameraCanvas.width = cameraPreview.videoWidth;
      cameraCanvas.height = cameraPreview.videoHeight;
      cameraCanvas.getContext('2d').drawImage(cameraPreview, 0, 0);
      cameraPendingFrame = cameraCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    }, 1000);
    cameraButton.textContent = '关闭摄像头';
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
        if (sessionCloseResolver === finish) sessionCloseResolver = null;
        resolve();
      };
      sessionCloseResolver = finish;
      window.setTimeout(finish, timeoutMs);
    });
  }

  async function stopSession({ terminal = true } = {}) {
    playbackEpoch++;
    document.getElementById('toolsEnabled').disabled = false;
    running = false;
    assistantActive = false;
    pendingCapture = [];
    if (sendTimer !== null) clearInterval(sendTimer);
    if (clockTimer !== null) clearInterval(clockTimer);
    sendTimer = null;
    clockTimer = null;
    if (socket) {
      const closingSocket = socket;
      socket = null;
      closingSocket.onclose = null;
      if (terminal && closingSocket.readyState === WebSocket.OPEN) {
        const closed = waitForSessionClosed(closingSocket, SESSION_CLOSE_TIMEOUT_MS);
        closingSocket.send(JSON.stringify({ type: 'session.close' }));
        await closed;
      }
      closingSocket.close(1000, 'client stop');
    }
    playingResponses.clear();
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
    meterFill.style.width = '0%';
    sessionTimer.textContent = '00:00';
    callButton.textContent = '开始对话';
    callButton.classList.remove('is-active');
    muteButton.textContent = '静音';
    muteButton.classList.remove('is-active');
    muteButton.disabled = true;
    setConnection('未连接', 'offline');
    setModel('待机');
    setPlayback('待机');
    if (runtimeDetail.textContent.startsWith('Playback committed')) return;
    if (!runtimeDetail.textContent.startsWith('start failed')) runtimeDetail.textContent = 'No active connection';
  }

  function toggleMute() {
    if (!running) return;
    muted = !muted;
    pendingCapture = [];
    muteButton.textContent = muted ? '取消静音' : '静音';
    muteButton.classList.toggle('is-active', muted);
    appendLog(muted ? 'microphone muted' : 'microphone unmuted');
  }

  document.getElementById('inspectContext').addEventListener('click', () => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type:'input.context.get'}));
  });
  fetch('/api/pickup').then(r => r.json()).then(r => { document.getElementById('pickupRecord').textContent = r.answer; });

  callButton.addEventListener('click', () => {
    if (running) stopSession();
    else startSession();
  });
  muteButton.addEventListener('click', toggleMute);
  clearLogButton.addEventListener('click', () => {
    eventLog.textContent = '';
    logCount = 0;
    eventCount.textContent = '0 events';
  });

  const visual = document.getElementById('liveVisual');
  const visualContext = visual.getContext('2d');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let previousFrame = 0;
  function drawVisual(timestamp) {
    requestAnimationFrame(drawVisual);
    if (document.hidden || timestamp - previousFrame < (reducedMotion ? 200 : 32)) return;
    previousFrame = timestamp;
    const width = visual.clientWidth, height = visual.clientHeight;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    if (visual.width !== Math.round(width * ratio) || visual.height !== Math.round(height * ratio)) {
      visual.width = Math.round(width * ratio); visual.height = Math.round(height * ratio);
    }
    const ctx = visualContext; ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
    const t = reducedMotion ? 0 : timestamp / 1000;
    const energy = running ? Math.max(micEnergy,speakerEnergy) : 0;
    micEnergy *= .91; speakerEnergy *= .94;
    const radius = Math.min(height*.29,width*.22), x=width/2, y=height*.46;
    const glow=ctx.createRadialGradient(x,y,0,x,y,radius*1.7);
    glow.addColorStop(0,`rgba(65,190,180,${.08+energy*.12})`);glow.addColorStop(.5,'rgba(143,83,224,.09)');glow.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle=glow;ctx.fillRect(0,0,width,height);
    for(let ring=0;ring<5;ring++) {
      ctx.beginPath();
      for(let i=0;i<=240;i++) {
        const angle=i/240*Math.PI*2;
        const ripple=Math.sin(angle*3+t*(.6+ring*.07)+ring)*(.04+energy*.12)+Math.cos(angle*5-t*.8+ring)*.025;
        const r=radius*(1+ring*.095+ripple);
        const px=x+Math.cos(angle)*r,py=y+Math.sin(angle)*r*.84;
        if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);
      }
      ctx.strokeStyle=ring%2 ? `rgba(174,126,255,${.26+energy*.3})` : `rgba(85,245,215,${.45+energy*.3})`;
      ctx.lineWidth=ring===0 ? 1.6 : .7;ctx.stroke();
    }
    for(let i=0;i<46;i++) {
      const phase=i*.137+t*.04, r=radius*(1.3+(i%7)*.08);
      ctx.fillStyle=i%3?'#62e7d170':'#c39cff80';ctx.fillRect(x+Math.cos(phase*7)*r,y+Math.sin(phase*7)*r*.72,1.2,1.2);
    }
    ctx.fillStyle=running?'#86fbe3':'#8798b5';ctx.font='10px ui-monospace, monospace';ctx.textAlign='center';
    ctx.fillText(!running?'G / 01':playingResponses.size?'TRANSMITTING':muted?'MIC MUTED':'LISTENING',x,y+4);
    document.getElementById('visualLabel').textContent=!running?'等待连接 · READY TO LISTEN':playingResponses.size?'正在回应 · VOICE ACTIVE':muted?'麦克风静音 · STILL CONNECTED':'正在聆听 · SPEAK FREELY';
  }
  requestAnimationFrame(drawVisual);

  window.addEventListener('beforeunload', () => { stopSession({ terminal: false }); });
})();

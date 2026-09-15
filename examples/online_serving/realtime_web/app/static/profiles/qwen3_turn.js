(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles;
  profiles['qwen3-turn'] = (config) => {
    const vad = config.adapter === 'vad';
    const hint = 'This Qwen deploy needs session_mode: duplex and Server VAD support; use --stt or enable Server VAD.';
    return {
      title: 'Qwen3-Omni Voice', eyebrow: 'Turn-based realtime call',
      policy: vad ? 'Server VAD • one response at a time' : 'Send turn • one response at a time',
      description: vad
        ? 'Pause to send your turn. Microphone upload pauses while the model answers. Requires session_mode: duplex and a Silero artifact. Optional camera frames accompany each spoken turn. No barge-in.'
        : 'Speak, then press Send turn. Microphone upload pauses while the model answers. Each turn uses a fresh connection; conversation history is not carried between turns. No barge-in or camera input.',
      waiting: 'Waiting for you', camera: vad, cameraMaxDimension: 448, playbackAck: false, clientCommit: !vad,
      halfDuplex: true, closeSession: vad, reconnectEachTurn: !vad,
      readyEvent: vad ? 'session.updated' : 'session.created', instructions: vad, sendIntervalMs: 200,
      presets: { assistant: 'You are a helpful assistant. Answer clearly and concisely.' },
      url(config, location) {
        const url = profiles.url(config, location);
        for (const key of ['native_duplex', 'minicpmo45_native_duplex', 'autostart']) url.searchParams.delete(key);
        // Explicit opt-out matches the local file client, including servers
        // that auto-select the duplex endpoint when the query is omitted.
        url.searchParams.set('duplex', vad ? '1' : '0');
        url.searchParams.set('model', config.model || 'Qwen/Qwen3-Omni-30B-A3B-Instruct');
        return url.toString();
      },
      initialMessages(config, instructions) {
        if (!vad) return [
          { type: 'session.update', model: config.model },
          { type: 'input_audio_buffer.commit', final: false },
        ];
        // This half-duplex client sends no playback ACKs. Retain completed
        // assistant replies so subsequent turns receive both sides of history.
        const session = { model: config.model, audio: { input: {
          format: { type: 'audio/pcm', rate: 16000 },
          turn_detection: { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300,
            silence_duration_ms: 500, create_response: true, interrupt_response: false },
        } } };
        if (instructions) session.instructions = instructions;
        // The server resets the policy to ack_only during session creation.
        // Apply it in a subsequent update, before any audio is uploaded.
        return [
          { type: 'session.update', session },
          { type: 'session.update', session: { playback_commit_policy: 'commit_all_on_done' } },
        ];
      },
      append(audio, frame) {
        const event = { type: 'input_audio_buffer.append', audio, format: 'pcm16', sample_rate_hz: 16000 };
        if (vad && frame) event.video_frames = [frame];
        return event;
      },
      commitMessages: () => vad ? [] : [{ type: 'input_audio_buffer.commit', final: true }],
      ack: () => null,
      mapEvent(event) {
        if (event.type === 'response.output_text.delta' || event.type === 'transcription.delta') {
          // On the shipped STT route transcription.* is model-generated text,
          // not a separate ASR transcript of the user. See realtime_connection.py.
          return { kind: 'text', role: 'assistant', channel: 'text', text: event.delta || '' };
        }
        if (event.type === 'response.output_text.done' || event.type === 'transcription.done') {
          return { kind: 'text-final', role: 'assistant', channel: 'text', text: event.text || '' };
        }
        if (event.type === 'input_audio_buffer.committed') return { kind: 'begin' };
        if (event.type === 'input_audio_buffer.cleared') return { kind: 'backpressure', message: 'Input cleared. Model is answering; please wait.' };
        if (!vad && event.type === 'response.output_audio.done') return { kind: 'done' };
        const action = profiles.event(event);
        if (action.kind === 'error') {
          if (action.code === 'input_backpressure') return { ...action, kind: 'backpressure', message: 'Model is answering. Microphone upload paused; please repeat dropped speech after the response.' };
          if (action.code === 'server_vad_initialization_failed') action.message = 'Server VAD could not load Silero. Configure the server artifact or restart this UI with --stt.';
          else if (vad) action.message += ` ${hint}`;
        }
        return action;
      },
      connectionHint: vad ? hint : 'Check the Qwen /v1/realtime STT endpoint and model name.',
    };
  };
})(globalThis);

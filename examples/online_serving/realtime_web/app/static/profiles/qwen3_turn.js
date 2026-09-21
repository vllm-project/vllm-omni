(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles;
  profiles['qwen3-turn'] = (config) => {
    const vad = config.adapter === 'vad';
    let interruptOnSpeech = vad;
    let imageSeq = 0;
    let imageItems = [];
    // The engine refuses a ninth image (or 4 MiB of them) for the whole
    // session, so the camera track retires its own oldest frame instead.
    const MAX_IMAGE_ITEMS = 8;
    const hint = 'This Qwen deploy needs session_mode: duplex and Server VAD support; use --stt or enable Server VAD.';
    return {
      title: 'Qwen3-Omni Voice', eyebrow: vad ? 'Engine-owned duplex call' : 'Turn-based realtime call',
      policy: vad ? 'Server VAD • interruptible responses' : 'Send turn • one response at a time',
      description: vad
        ? 'Pause to send your turn; speak again to interrupt. Microphone upload continues during playback. Requires the Qwen duplex plugin and Silero. Optional camera frames accompany each spoken turn.'
        : 'Speak, then press Send turn. Microphone upload pauses while the model answers. Each turn uses a fresh connection; conversation history is not carried between turns. No barge-in or camera input.',
      waiting: 'Waiting for you', camera: vad, cameraMaxDimension: 448, playbackAck: vad, clientCommit: !vad,
      waitForResponseDone: true, halfDuplex: !vad, deduplicateTranscript: true, closeSession: vad, reconnectEachTurn: !vad,
      inputSampleRate: vad ? 24000 : 16000,
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
        imageItems = [];
        interruptOnSpeech = vad;
        if (!vad) return [
          { type: 'session.update', model: config.model },
          { type: 'input_audio_buffer.commit', final: false },
        ];
        const session = { model: config.model, overlap_policy: 'barge_in_on_speech', audio: { input: {
          format: { type: 'audio/pcm', rate: 24000 },
          turn_detection: { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300,
            silence_duration_ms: 500, create_response: true, interrupt_response: true },
        } } };
        if (instructions) session.instructions = instructions;
        return [{ type: 'session.update', session }];
      },
      append(audio) {
        return vad ? { type: 'input_audio_buffer.append', audio }
          : { type: 'input_audio_buffer.append', audio, format: 'pcm16', sample_rate_hz: 16000 };
      },
      // Qwen is a turn model: a frame is a picture added to the conversation,
      // not a track interleaved into the audio. That is exactly the OpenAI
      // Realtime image interface, so the camera sends conversation items
      // rather than an extension field on the append.
      imageMessages(frame) {
        if (!vad) return [];
        const messages = [];
        if (imageItems.length >= MAX_IMAGE_ITEMS) {
          // Retire before creating: the engine counts what is already stored.
          messages.push({ type: 'conversation.item.delete', item_id: imageItems.shift() });
        }
        const id = `camera_${++imageSeq}`;
        imageItems.push(id);
        messages.push({ type: 'conversation.item.create', item: {
          id, type: 'message', role: 'user',
          content: [{ type: 'input_image', image_url: `data:image/jpeg;base64,${frame}` }],
        } });
        return messages;
      },
      commitMessages: () => vad ? [] : [{ type: 'input_audio_buffer.commit', final: true }],
      ack: (responseId, playedMs) => vad ? { type: 'playback.ack', response_id: responseId,
        item_id: `item_${responseId}`, played_ms: playedMs, committed_ms: playedMs } : null,
      mapEvent(event) {
        if (event.type === 'session.created' || event.type === 'session.updated') {
          const input = event.session?.audio?.input;
          const turnDetection = input && 'turn_detection' in input
            ? input.turn_detection : event.session?.turn_detection;
          if (turnDetection !== undefined) {
            interruptOnSpeech = vad && turnDetection?.type === 'server_vad'
              && turnDetection.interrupt_response !== false;
          }
        }
        // Generation can finish long before the speaker drains its queue. In
        // that case the server has no active response left to cancel for us.
        if (interruptOnSpeech && event.type === 'input_audio_buffer.speech_started') {
          return { kind: 'interrupt' };
        }
        if (event.type === 'playback.acknowledged') return { kind: 'ack', committedMs: (event.event || event).committed_ms || 0 };
        if (event.type === 'response.output_text.delta' || event.type === 'transcription.delta') {
          // On the shipped STT route transcription.* is model-generated text,
          // not a separate ASR transcript of the user. See realtime_connection.py.
          return { kind: 'text', role: 'assistant', channel: 'text', text: event.delta || '' };
        }
        if (event.type === 'response.output_text.done' || event.type === 'transcription.done') {
          return { kind: 'text-final', role: 'assistant', channel: 'text', text: event.text || '' };
        }
        if (vad && (event.type === 'output_audio_buffer.cleared' ||
            (event.type === 'response.done' && event.response?.status === 'cancelled'))) {
          return { kind: 'interrupt', responseId: event.response_id || event.response?.id || null };
        }
        if (event.type === 'input_audio_buffer.committed') return { kind: 'begin' };
        if (event.type === 'input_audio_buffer.cleared') return { kind: 'backpressure', message: 'Input cleared. Model is answering; please wait.' };
        if (!vad && event.type === 'response.output_audio.done') return { kind: 'done' };
        const action = profiles.event(event);
        if (action.kind === 'error') {
          if (action.code === 'input_backpressure') return { ...action, kind: 'backpressure', message: 'Input buffer is full; please repeat dropped speech after the response.' };
          if (action.code === 'server_vad_initialization_failed') action.message = 'Server VAD could not load Silero. Configure the server artifact or restart this UI with --stt.';
          else if (vad && action.code === 'unsupported') action.message += ` ${hint}`;
        }
        return action;
      },
      connectionHint: vad ? hint : 'Check the Qwen /v1/realtime STT endpoint and model name.',
    };
  };
})(globalThis);

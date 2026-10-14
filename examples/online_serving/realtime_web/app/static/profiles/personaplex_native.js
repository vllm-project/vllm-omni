(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles;
  profiles['personaplex-native'] = () => {
    let pending = new Int16Array(0);
    // The model keeps one response ID across speech and silence. This is a
    // display heuristic, not a model turn boundary. Use output audio time so
    // network stalls alone cannot split an answer.
    const CAPTION_PAUSE_MS = 2000;
    let audioMs = 0, lastTextMs = null, captionNumber = 0;
    let textChars = 0, captionStart = 0;
    function resetCaptions() {
      audioMs = 0; lastTextMs = null; captionNumber = 0;
      textChars = 0; captionStart = 0;
    }
    return {
      title: 'PersonaPlex', eyebrow: 'Native full-duplex voice · English',
      policy: 'Model-controlled conversation',
      description: 'Speak in English. Assistant captions appear below; user speech is not transcribed. The microphone stays active while the model speaks.',
      waiting: 'Listening', camera: false, playbackAck: false, clientCommit: false,
      halfDuplex: false, closeSession: true, reconnectEachTurn: false, silenceWhenMuted: true,
      readyEvent: 'session.updated', instructions: true, inputSampleRate: 24000, sendIntervalMs: 80,
      presets: { assistant: 'You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.' },
      url(config, location) {
        const url = profiles.url(config, location);
        url.searchParams.set('duplex', '1');
        url.searchParams.set('autostart', '0');
        url.searchParams.set('model', config.model || 'nvidia/personaplex-7b-v1');
        return url.toString();
      },
      initialMessages(config, instructions) {
        pending = new Int16Array(0);
        resetCaptions();
        return [{ type: 'session.update', session: {
          model: config.model || 'nvidia/personaplex-7b-v1', modalities: ['audio', 'text'],
          input_audio_format: 'pcm_f32le', output_audio_format: 'pcm16', sample_rate_hz: 24000,
          voice: 'NATF2.pt', instructions, turn_detection: null, extra_body: { auto_response: true },
        } }];
      },
      appendMessages(pcm) {
        const joined = new Int16Array(pending.length + pcm.length);
        joined.set(pending); joined.set(pcm, pending.length);
        const events = [];
        let offset = 0;
        while (offset + 1920 <= joined.length) {
          const raw = new ArrayBuffer(1920 * 4);
          const view = new DataView(raw);
          for (let i = 0; i < 1920; i += 1) view.setFloat32(i * 4, joined[offset + i] / 32768, true);
          events.push({ type: 'input_audio_buffer.append',
            audio: btoa(String.fromCharCode(...new Uint8Array(raw))), format: 'pcm_f32le', sample_rate_hz: 24000 });
          offset += 1920;
        }
        pending = joined.slice(offset);
        return events;
      },
      commitMessages: () => [],
      mapEvent(event) {
        const action = profiles.event(event);
        if (action.kind === 'begin') resetCaptions();
        if (action.kind === 'audio') {
          const encoded = action.event.delta || '';
          const padding = encoded.endsWith('==') ? 2 : encoded.endsWith('=') ? 1 : 0;
          const samples = (encoded.length * 3 / 4 - padding) / 2;
          audioMs += samples * 1000 / (event.sample_rate_hz || 24000);
        }
        if (action.role === 'assistant' && action.kind === 'text' && action.text) {
          action.newTranscript = lastTextMs === null || audioMs - lastTextMs >= CAPTION_PAUSE_MS;
          if (action.newTranscript) {
            captionNumber += 1;
            captionStart = textChars;
          }
          lastTextMs = audioMs;
          textChars += action.text.length;
        }
        if (action.role === 'assistant' && captionNumber) {
          // Only the caption key changes; audio keeps its original response ID.
          action.transcriptId = `${action.responseId}:caption-${captionNumber}`;
          if (action.kind === 'text-final') action.text = action.text.slice(captionStart);
        }
        return action;
      },
      connectionHint: 'Check the PersonaPlex native duplex backend.',
    };
  };
})(globalThis);

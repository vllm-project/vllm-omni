(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles;

  function assistantItemText(item) {
    if (!item || item.role !== 'assistant' || !Array.isArray(item.content)) return '';
    const parts = [];
    for (const part of item.content) {
      if (!part) continue;
      if (typeof part.transcript === 'string' && part.transcript.trim()) parts.push(part.transcript.trim());
      else if (typeof part.text === 'string' && part.text.trim()) parts.push(part.text.trim());
    }
    return parts.join('');
  }

  profiles['aura-ptt'] = () => ({
    title: 'AURA Realtime',
    eyebrow: 'Push-to-talk duplex',
    policy: 'Hold to talk • release to commit',
    description:
      'Hold to talk, release to send. Camera stays on whether you speak or not. '
    + 'Holding the button stops local playback.',
    waiting: 'Ready',
    camera: true,
    // Display only. Shared CSS puts .camera-preview-large on its own row.
    // Other profiles keep the 96×72 preview in index.html.
    cameraPreviewLarge: true,
    playbackAck: true,
    clientCommit: false,
    pushToTalk: true,
    stickyCamera: true,
    // OmniInteract / Native AURA clock: 2 fps. Other profiles omit this and stay at 1 s.
    cameraIntervalMs: 500,
    visionFollowWhileSpeaking: true,
    // Pitch-preserving time-stretch. 1 keeps the shared worklet's 1:1 copy.
    playbackSpeed: 1.3,
    // First codec packet is 8 frames (0.64s). At 1.3x that runs out before the
    // next packet and the worklet cuts. Wait for a second packet unless the
    // turn already finished inside the first one (drain starts immediately).
    holdPlaybackUntilSecondChunk: true,
    // Grounding boxes are 0–1000 on the frame the model saw. The page draws
    // them; Talker never receives the tokens.
    presentAssistantText(text) {
      const raw = typeof text === 'string' ? text : '';
      const boxes = [];
      const withoutSpans = raw.replace(/<\|box_start\|>([\s\S]*?)<\|box_end\|>/g, (_, body) => {
        const nums = String(body).replace(/[()]/g, '').match(/-?\d+(?:\.\d+)?/g);
        if (!nums || nums.length !== 4) return '';
        const values = nums.map(Number);
        if (values.some(value => !Number.isFinite(value))) return '';
        let [x1, y1, x2, y2] = values.map(value => Math.max(0, Math.min(1000, value)));
        if (x2 < x1) [x1, x2] = [x2, x1];
        if (y2 < y1) [y1, y2] = [y2, y1];
        if (x2 - x1 < 1 || y2 - y1 < 1) return '';
        boxes.push([x1, y1, x2, y2]);
        return '';
      });
      const open = withoutSpans.indexOf('<|box');
      return {
        displayText: (open === -1 ? withoutSpans : withoutSpans.slice(0, open)).trim(),
        boxes,
      };
    },
    deduplicateTranscript: true,
    halfDuplex: false,
    closeSession: true,
    reconnectEachTurn: false,
    readyEvent: 'session.updated',
    instructions: true,
    sendIntervalMs: 200,
    presets: {
      omni: 'You are a real-time multimodal AI companion in a live video conversation with the user. You are receiving a live video stream where the final frame is the present moment. Decide whether a response is needed based on the user\'s message or the visual context. If a response is needed, output `<|respond|>` followed immediately by the response content. If no response is needed, output exactly `<|silent|>` to signify silence.',
    },
    url(config, location) {
      const url = profiles.url(config, location);
      url.searchParams.set('duplex', '1');
      url.searchParams.set('model', config.model || 'aurateam/AURA');
      return url.toString();
    },
    initialMessages(config, instructions) {
      const session = {
        modalities: ['audio', 'text'],
        extra_body: { auto_response: true },
      };
      if (instructions) session.instructions = instructions;
      return [{ type: 'session.update', session }];
    },
    append(audio, frame, opts) {
      const isSpeech = !opts || opts.isSpeech !== false;
      const event = {
        type: 'input_audio_buffer.append',
        audio,
        format: 'pcm16',
        sample_rate_hz: 16000,
        is_speech: isSpeech,
      };
      const frames = Array.isArray(frame) ? frame.filter(Boolean) : (frame ? [frame] : []);
      if (frames.length) event.video_frames = frames;
      return event;
    },
    commitMessages: () => [{ type: 'input_audio_buffer.commit', create_response: true }],
    ack(responseId, playedMs) {
      return {
        type: 'playback.ack',
        response_id: responseId,
        item_id: `item_${responseId}`,
        played_ms: playedMs,
        committed_ms: playedMs,
      };
    },
    mapEvent(event) {
      if (event.type === 'response.listen') return { kind: 'listen' };
      // A call-only response: the UI server runs the tool and sends the
      // result + response.create, so this done must not reopen vision-follow.
      const output = event.type === 'response.done' && Array.isArray(event.response?.output) ? event.response.output : [];
      const call = output.find(item => item && item.type === 'function_call');
      if (call) return { ...profiles.event(event), toolCall: call.name || 'tool' };
      if (event.type === 'playback.acknowledged') {
        return { kind: 'ack', committedMs: (event.event || event).committed_ms || 0 };
      }
      const itemText = assistantItemText(event.item);
      if (itemText && (event.type === 'conversation.item.done' || event.type === 'conversation.item.added')) {
        return {
          kind: 'text-final',
          role: 'assistant',
          channel: 'audio',
          text: itemText,
          responseId: event.response_id || event.item.id || null,
        };
      }
      if (event.type === 'response.output_text.delta' && event.delta) {
        return {
          kind: 'text',
          role: 'assistant',
          channel: 'text',
          text: event.delta,
          responseId: event.response_id || event.response?.id || null,
        };
      }
      if (event.type === 'response.output_text.done' && (event.text || event.transcript)) {
        return {
          kind: 'text-final',
          role: 'assistant',
          channel: 'text',
          text: event.text || event.transcript,
          responseId: event.response_id || event.response?.id || null,
        };
      }
      return profiles.event(event);
    },
    connectionHint: 'Check the AURA duplex serve (run_duplex_smoke_serve.sh) and --ws-backend port.',
  });
})(globalThis);

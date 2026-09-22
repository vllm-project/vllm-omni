(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles;
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
    deduplicateTranscript: true,
    halfDuplex: false,
    closeSession: true,
    reconnectEachTurn: false,
    readyEvent: 'session.updated',
    instructions: true,
    sendIntervalMs: 200,
    presets: {
      omni: 'You are receiving a live video stream where the final frame is the present moment. Respond only when a response is needed based on the user\'s message or the visual context. Otherwise, output `<|silent|>` to signify silence.',
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
      if (event.type === 'playback.acknowledged') {
        return { kind: 'ack', committedMs: (event.event || event).committed_ms || 0 };
      }
      return profiles.event(event);
    },
    connectionHint: 'Check the AURA duplex serve (run_duplex_smoke_serve.sh) and --ws-backend port.',
  });
})(globalThis);

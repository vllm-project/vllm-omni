(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles;
  profiles['aura-ptt'] = () => ({
    title: 'AURA Realtime',
    eyebrow: 'Push-to-talk duplex',
    policy: 'Hold to talk • release to commit',
    description:
      'Hold to stream speech (is_speech=true) with camera frames. Release to commit. '
      + 'While the assistant speaks, camera frames go as vision-follow (is_speech=false). '
      + 'AURA has no client VAD.',
    waiting: 'Ready',
    camera: true,
    playbackAck: true,
    clientCommit: false,
    pushToTalk: true,
    stickyCamera: true,
    visionFollowWhileSpeaking: true,
    halfDuplex: false,
    closeSession: true,
    reconnectEachTurn: false,
    readyEvent: 'session.updated',
    instructions: true,
    sendIntervalMs: 200,
    presets: {
      omni: 'You are AURA, a helpful multimodal assistant. Answer clearly in the user\'s language.',
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
      if (frame) event.video_frames = [frame];
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

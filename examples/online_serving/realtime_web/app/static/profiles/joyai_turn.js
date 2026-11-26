(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles;
  profiles['joyai-turn'] = (config = {}) => {
    const speech = Boolean(config.speechInput);
    const vad = speech && config.turnMode === 'vad';
    return ({
    title: 'JoyAI Realtime',
    eyebrow: 'Camera observations',
    policy: 'Latest frame • one response at a time',
    description: speech
      ? (vad ? 'Speak and pause to send a turn with the latest camera frame. Requires the ASR duplex deployment and server VAD.'
        : 'Hold to talk, then release to send. Camera observations continue once per second while idle. Requires the ASR duplex deployment.')
      : 'Set an instruction, start a session and enable Camera. The latest frame is submitted once per second while idle. You can also press Send observation.',
    waiting: 'Watching',
    camera: true,
    cameraPreviewLarge: true,
    cameraMaxDimension: 448,
    stickyCamera: true,
    videoOnly: !speech,
    pushToTalk: speech && !vad,
    cameraObservations: !vad,
    playbackAck: true,
    clientCommit: !vad,
    waitForResponseDone: true,
    halfDuplex: !speech,
    deduplicateTranscript: true,
    closeSession: true,
    readyEvent: 'session.updated',
    instructions: true,
    sendIntervalMs: speech ? 200 : 1000,
    presets: {
      assistant: 'Watch the camera and briefly describe important changes. Respond only when useful; otherwise remain silent.',
    },
    url(config, location) {
      const url = profiles.url(config, location);
      url.searchParams.set('duplex', '1');
      url.searchParams.set('model', config.model || 'jdopensource/JoyAI-VL-Interaction-Preview');
      return url.toString();
    },
    initialMessages(config, instructions) {
      const session = {
        modalities: ['audio', 'text'],
        instructions,
        extra_body: { auto_response: false },
      };
      if (speech) session.audio = { input: { format: { type: 'audio/pcm', rate: 16000 },
        turn_detection: vad ? { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300,
          silence_duration_ms: 500, create_response: true, interrupt_response: true } : null } };
      return [{ type: 'session.update', session }];
    },
    append(audio, frame) {
      const event = { type: 'input_audio_buffer.append' };
      if (speech && audio) Object.assign(event, { audio, format: 'pcm16', sample_rate_hz: 16000 });
      if (frame) event.video_frames = [frame];
      return event;
    },
    commitMessages: () => [{ type: 'input_audio_buffer.commit', final: true, create_response: true }],
    ack(responseId, playedMs) {
      return { type: 'playback.ack', response_id: responseId,
        item_id: `item_${responseId}`, played_ms: playedMs, committed_ms: playedMs };
    },
    mapEvent(event) {
      if (event.type === 'input_audio_buffer.committed' && event.event?.no_response === true) {
        return { kind: 'commit-settled', eventId: event.event.realtime_event_id };
      }
      // A reply can finish generating while its audio is still queued locally.
      if (vad && event.type === 'input_audio_buffer.speech_started') return { kind: 'interrupt' };
      if (vad && (event.type === 'output_audio_buffer.cleared' ||
          (event.type === 'response.done' && event.response?.status === 'cancelled'))) {
        return { kind: 'interrupt', responseId: event.response_id || event.response?.id || null };
      }
      if (event.type === 'playback.acknowledged') return { kind: 'ack', committedMs: (event.event || event).committed_ms || 0 };
      if (event.type === 'response.output_text.delta') {
        return { kind: 'text', role: 'assistant', channel: 'text', text: event.delta || '' };
      }
      if (event.type === 'response.output_text.done') {
        return { kind: 'text-final', role: 'assistant', channel: 'text', text: event.text || '' };
      }
      return profiles.event(event);
    },
    connectionHint: 'Start the matching JoyAI duplex deployment (ASR for speech input) and check --ws-backend.',
  });
  };
})(globalThis);

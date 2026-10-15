(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles;
  profiles['joyai-turn'] = () => ({
    title: 'JoyAI Realtime',
    eyebrow: 'Camera observations',
    policy: 'Latest frame • one response at a time',
    description: 'Set an instruction, start a session and enable Camera. The latest frame is submitted once per second while idle. You can also press Send observation. Replies play through your speakers; microphone input is not enabled.',
    waiting: 'Watching',
    camera: true,
    cameraPreviewLarge: true,
    cameraMaxDimension: 448,
    stickyCamera: true,
    videoOnly: true,
    playbackAck: true,
    clientCommit: true,
    waitForResponseDone: true,
    halfDuplex: true,
    deduplicateTranscript: true,
    closeSession: true,
    readyEvent: 'session.updated',
    instructions: true,
    sendIntervalMs: 1000,
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
      return [{ type: 'session.update', session: {
        modalities: ['audio', 'text'],
        instructions,
        extra_body: { auto_response: false },
      } }];
    },
    append(_audio, frame) {
      return { type: 'input_audio_buffer.append', video_frames: [frame] };
    },
    commitMessages: () => [{ type: 'input_audio_buffer.commit', final: true, create_response: true }],
    ack(responseId, playedMs) {
      return { type: 'playback.ack', response_id: responseId,
        item_id: `item_${responseId}`, played_ms: playedMs, committed_ms: playedMs };
    },
    mapEvent(event) {
      if (event.type === 'playback.acknowledged') return { kind: 'ack', committedMs: (event.event || event).committed_ms || 0 };
      if (event.type === 'response.output_text.delta') {
        return { kind: 'text', role: 'assistant', channel: 'text', text: event.delta || '' };
      }
      if (event.type === 'response.output_text.done') {
        return { kind: 'text-final', role: 'assistant', channel: 'text', text: event.text || '' };
      }
      return profiles.event(event);
    },
    connectionHint: 'Start the JoyAI duplex deployment and check --ws-backend.',
  });
})(globalThis);

(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles;
  profiles['minicpm-native'] = () => ({
    title: 'MiniCPM-o 4.5 Voice', eyebrow: 'Experimental full-duplex runtime',
    policy: 'Model-controlled listen / speak',
    description: 'Speak naturally. The model can listen and speak at the same time. Camera frames are sent with your audio.',
    waiting: 'Listening', camera: true, playbackAck: true, clientCommit: false,
    halfDuplex: false, closeSession: true, reconnectEachTurn: false,
    readyEvent: 'session.updated', instructions: true, sendIntervalMs: 200,
    presets: {
      omni: 'Streaming Omni Conversation.',
      chinese_call: '扮演一个具有以上声音特征的助手。请认真、高质量地回复用户的问题。'
        + '请用高自然度的方式和用户聊天。你处于双工模式，可以一边听、一边说。'
        + '你是由面壁智能开发的人工智能助手：面壁小钢炮。',
      english_call: 'Replicate the tone and style from the input audio. Your task is to be '
        + 'a helpful assistant using this voice pattern. Please answer the user\'s questions '
        + 'seriously and in a high quality. Please chat with the user in a high naturalness '
        + 'style. You are in duplex mode, where you can listen and speak at the same time.',
    },
    url(config, location) {
      const url = profiles.url(config, location);
      url.searchParams.set('duplex', '1');
      url.searchParams.set('model', config.model || 'openbmb/MiniCPM-o-4_5');
      url.searchParams.set('native_duplex', '1');
      url.searchParams.set('autostart', '0');
      return url.toString();
    },
    initialMessages(config, instructions) {
      const session = { modalities: ['audio', 'text'], voice: 'default',
        extra_body: { auto_response: true, native_duplex: true } };
      if (config.refAudio) session.ref_audio = config.refAudio;
      if (instructions) session.instructions = instructions;
      return [{ type: 'session.update', session }];
    },
    append(audio, frame) {
      const event = { type: 'input_audio_buffer.append', audio, format: 'pcm16', sample_rate_hz: 16000 };
      if (frame) event.video_frames = [frame];
      return event;
    },
    commitMessages: () => [],
    ack(responseId, playedMs) {
      return { type: 'playback.ack', response_id: responseId, item_id: `item_${responseId}`,
        played_ms: playedMs, committed_ms: playedMs };
    },
    mapEvent(event) {
      if (event.type === 'response.listen') return { kind: 'listen' };
      if (event.type === 'response.speak') return { kind: 'begin', responseId: event.response_id || event.response?.id };
      if (event.type === 'playback.acknowledged') {
        return { kind: 'ack', committedMs: (event.event || event).committed_ms || 0 };
      }
      return profiles.event(event);
    },
    connectionHint: 'Check the MiniCPM duplex deployment and reference voice configuration.',
  });
})(globalThis);

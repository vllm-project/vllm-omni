/* Shared wire helpers. Profiles translate server events into shell actions. */
(function (root) {
  'use strict';
  const profiles = root.OmniRealtimeProfiles = root.OmniRealtimeProfiles || {};
  profiles.url = (config, location) => {
    const url = new URL(config.realtimePath || 'v1/realtime', location);
    if (url.protocol === 'http:') url.protocol = 'ws:';
    if (url.protocol === 'https:') url.protocol = 'wss:';
    return url;
  };
  /* Errors that leave the session unusable, so the shell has to tear it down.
     Everything else costs one operation --- a rejected image, a refused
     session.update, a late playback ack --- and the call carries on.
     'runtime_append_failed' is deliberately absent: the engine reports it and
     keeps the session, unlike 'runtime_append_task_failed' which kills it. */
  profiles.fatalErrorCodes = new Set([
    'unknown_session',
    'session_closed',
    'resource_exhausted',
    'unsupported',
    'unsupported_turn_detection',
    'server_vad_unavailable',
    'runtime_append_task_failed',
    'internal_error',
  ]);
  profiles.event = (event) => {
    const responseId = event.response_id || event.response?.id || null;
    const action = (kind, extra = {}) => ({ kind, responseId, ...extra });
    switch (event.type) {
      case 'session.created': case 'session.updated': return action('connected');
      case 'response.created': return action('begin');
      case 'response.audio.delta': case 'response.output_audio.delta':
        return action('audio', { event: { ...event, delta: event.delta || event.audio || event.response?.audio } });
      case 'response.audio.done': case 'response.output_audio.done': return action('drain');
      case 'response.audio_transcript.delta': case 'response.output_audio_transcript.delta':
        return action('text', { role: 'assistant', channel: 'audio', text: event.delta || '' });
      case 'response.audio_transcript.done': case 'response.output_audio_transcript.done':
        return action('text-final', { role: 'assistant', channel: 'audio', text: event.transcript || '' });
      case 'conversation.item.input_audio_transcription.delta':
        return action('text', { role: 'user', text: event.delta || '' });
      case 'conversation.item.input_audio_transcription.completed':
        return action('text-final', { role: 'user', text: event.transcript || '' });
      case 'response.done': return action('done');
      case 'session.closed': return action('closed');
      case 'error': {
        const code = event.code || event.error?.code;
        return action('error', {
          code,
          fatal: profiles.fatalErrorCodes.has(code),
          message: typeof event.error === 'string' ? event.error : event.error?.message || event.message || code || 'Server error',
        });
      }
      default: return action('ignore');
    }
  };
})(globalThis);

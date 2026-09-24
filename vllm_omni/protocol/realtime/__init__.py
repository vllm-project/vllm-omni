# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The OpenAI Realtime wire codec, shared by every vLLM-Omni Realtime surface.

What this is
------------
The model-agnostic half of ``WS /v1/realtime``: how a client event is parsed,
which audio formats and session fields are accepted, what a conversation item
means, how audio is decoded and resampled, and the shape of the error envelope
(the *codes* inside it are a consumer's own --- ours are in
``vllm_omni.protocol.duplex.errors``). It is a library of pure functions and value objects.

What it is not
--------------
It owns no session. It does not decide when a response starts, what a model is
prompted with, whether a turn is over, or where committed audio goes. Those are
runtime decisions, and after PR #7413 a duplex session's runtime decisions all
belong to ``vllm_omni.engine.duplex`` --- the session runner, the typed
``DuplexCommand`` / ``DuplexEvent`` boundary and the ``DuplexModelPlugin`` seam.
This package sits *under* that: the duplex engine binds the codec to its own
command and event vocabulary in
``vllm_omni.engine.duplex.realtime_commands`` and
``vllm_omni.engine.duplex.realtime_events``.

Why it is separate (RFC #6592, P0a)
-----------------------------------
So that a second Realtime surface --- a Qwen3-Omni GA profile, say --- can reuse
the parsing, validation, format negotiation and error envelope without adopting
the duplex session control plane (lease manager, commit policy, model channel),
and without a second copy of the codec drifting away from this one. A consumer
states what it can serve through
:class:`~vllm_omni.protocol.realtime.capabilities.RealtimeProtocolCapabilities`
and keeps its own session state.

Known duplex-shaped defaults
----------------------------
This package is model-agnostic in its *dependencies*, not yet in every
*default*. Two constants still carry the assumptions of its first consumer and
a second one will have to parameterize them:

* :func:`~vllm_omni.protocol.realtime.audio.convert_input_audio_with_rate`
  resamples to 16 kHz, which is MiniCPM-o's feature-extractor rate. A GA client
  sending 24 kHz is converted down rather than served at its own rate.
* :func:`~vllm_omni.protocol.realtime.items.validate_realtime_video_frames`
  encodes the MiniCPM-o duplex camera contract (at most two frames per append,
  ``max_slice_nums`` not selectable on the wire).

Both are behaviour this extraction deliberately did **not** change (RFC #6592
P0a is behaviour-preserving); making them per-consumer is P0b/P1 work.

Dependency rule
---------------
Nothing here may import ``vllm_omni.engine``, ``vllm_omni.entrypoints``,
``vllm_omni.model_executor``, ``vllm_omni.worker`` or ``vllm_omni.clients``;
``tests/protocol/realtime/test_protocol_import_boundary.py`` asserts it.
"""

from vllm_omni.protocol.realtime.audio import (
    MAX_INPUT_SAMPLE_RATE_HZ,
    MIN_INPUT_SAMPLE_RATE_HZ,
    convert_input_audio_with_rate,
    convert_output_audio,
    encode_float32_mono_wav_base64,
    resample_pcm16_mono,
    validate_input_sample_rate_hz,
    wav_payload_to_pcm16,
)
from vllm_omni.protocol.realtime.audio_input import (
    REALTIME_INPUT_HINT_KEYS,
    RealtimeAudioAppend,
    copy_realtime_input_hints,
    decode_audio_append,
    input_explicitly_non_speech,
    input_looks_like_speech,
)
from vllm_omni.protocol.realtime.capabilities import (
    RealtimeProtocolCapabilities,
    RealtimeSessionRejection,
    validate_session_payload,
)
from vllm_omni.protocol.realtime.commands import RealtimeCommand
from vllm_omni.protocol.realtime.errors import RealtimeProtocolError
from vllm_omni.protocol.realtime.events import (
    RealtimeEvent,
    ResponseScopedEvent,
    new_event_id,
    wire_value,
)
from vllm_omni.protocol.realtime.formats import (
    REALTIME_INPUT_AUDIO_FORMATS,
    REALTIME_OUTPUT_AUDIO_FORMATS,
    is_supported_realtime_input_format,
    parse_realtime_audio_format,
    realtime_audio_format_object,
    realtime_output_format,
    validate_conversation_item_audio_formats,
    validate_realtime_response_audio_formats,
    validate_realtime_session_audio_formats,
)
from vllm_omni.protocol.realtime.items import (
    input_transcript_from_item,
    normalize_conversation_item,
    text_chars_for_audio_ms_from_marks,
    truncate_realtime_item_content,
    validate_realtime_item_truncate,
    validate_realtime_video_frames,
)
from vllm_omni.protocol.realtime.session import (
    RealtimeInputDefaults,
    apply_realtime_session_defaults,
    input_audio_transcription_config,
    json_safe_realtime_payload,
    realtime_max_output_tokens,
    realtime_overlap_fields,
)

__all__ = [
    "MAX_INPUT_SAMPLE_RATE_HZ",
    "MIN_INPUT_SAMPLE_RATE_HZ",
    "REALTIME_INPUT_AUDIO_FORMATS",
    "REALTIME_INPUT_HINT_KEYS",
    "REALTIME_OUTPUT_AUDIO_FORMATS",
    "RealtimeAudioAppend",
    "RealtimeCommand",
    "RealtimeEvent",
    "RealtimeInputDefaults",
    "RealtimeProtocolCapabilities",
    "RealtimeProtocolError",
    "RealtimeSessionRejection",
    "ResponseScopedEvent",
    "apply_realtime_session_defaults",
    "convert_input_audio_with_rate",
    "convert_output_audio",
    "copy_realtime_input_hints",
    "decode_audio_append",
    "encode_float32_mono_wav_base64",
    "input_audio_transcription_config",
    "input_explicitly_non_speech",
    "input_looks_like_speech",
    "input_transcript_from_item",
    "is_supported_realtime_input_format",
    "json_safe_realtime_payload",
    "new_event_id",
    "normalize_conversation_item",
    "parse_realtime_audio_format",
    "realtime_audio_format_object",
    "realtime_max_output_tokens",
    "realtime_output_format",
    "realtime_overlap_fields",
    "resample_pcm16_mono",
    "text_chars_for_audio_ms_from_marks",
    "truncate_realtime_item_content",
    "validate_conversation_item_audio_formats",
    "validate_input_sample_rate_hz",
    "validate_realtime_item_truncate",
    "validate_realtime_response_audio_formats",
    "validate_realtime_session_audio_formats",
    "validate_realtime_video_frames",
    "validate_session_payload",
    "wav_payload_to_pcm16",
    "wire_value",
]

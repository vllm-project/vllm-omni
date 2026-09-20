# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""vLLM-Omni's Realtime error-code vocabulary.

OpenAI's Realtime API standardises the ``error.type`` classes
(``invalid_request_error`` / ``server_error`` / ``rate_limit_error``) but not
the ``code`` values inside them. Ours are listed here --- Tier 3 in
``docs/serving/realtime_duplex_api.md``, "the full error-code vocabulary" ---
so ``protocol/realtime`` stays the OpenAI surface and nothing more.
"""

from __future__ import annotations

from vllm_omni.protocol.realtime.errors import RealtimeProtocolError

__all__ = ["REALTIME_ERROR_TYPES_BY_CODE", "RealtimeProtocolError", "realtime_error_type"]


#: OpenAI Realtime ``error.type`` for each internal error code.
REALTIME_ERROR_TYPES_BY_CODE: dict[str, str] = {
    "bad_event": "invalid_request_error",
    "bad_audio": "invalid_request_error",
    "config_timeout": "invalid_request_error",
    "invalid_json": "invalid_request_error",
    "event_too_large": "invalid_request_error",
    "unknown_event": "invalid_request_error",
    "internal_error": "server_error",
    "runtime_append_failed": "server_error",
    "context_operation_failed": "server_error",
    "context_output_failed": "server_error",
    "context_not_initialized": "invalid_request_error",
    "invalid_context_input": "invalid_request_error",
    "stale_context_epoch": "invalid_request_error",
    "stale_context_version": "invalid_request_error",
    "stale_task_slate": "invalid_request_error",
    "unknown_function_call": "invalid_request_error",
    "duplicate_function_output": "invalid_request_error",
    "context_event_conflict": "invalid_request_error",
    "gander_context_limit": "invalid_request_error",
    "gander_tool_parse_error": "invalid_request_error",
    "gander_tool_schema_error": "invalid_request_error",
    "invalid_tools": "invalid_request_error",
    "context_input_unsupported": "invalid_request_error",
    "context_replacement_unsupported": "invalid_request_error",
    "runtime_input_failed": "server_error",
    "runtime_append_task_failed": "server_error",
    "runtime_signal_failed": "server_error",
    "runtime_abort_failed": "server_error",
    "runtime_data_plane_stream_failed": "server_error",
    "runtime_data_plane_text_without_audio": "server_error",
    "resource_exhausted": "rate_limit_error",
    "session_exists": "invalid_request_error",
    "session_closed": "invalid_request_error",
    "unknown_session": "invalid_request_error",
    "invalid_duplex_runtime_config": "invalid_request_error",
    "instructions_update_unsupported": "invalid_request_error",
    "persona_update_unsupported": "invalid_request_error",
    "voice_update_unsupported": "invalid_request_error",
    "unsupported_nemotron_duplex_mode": "invalid_request_error",
    "unsupported_native_response_options": "invalid_request_error",
    "runtime_touch_failed": "server_error",
    "engine_error": "server_error",
    "input_backpressure": "rate_limit_error",
    "response_already_active": "invalid_request_error",
    "response_not_active": "invalid_request_error",
    "response_create_without_input": "invalid_request_error",
    "text_only_turn_unsupported": "invalid_request_error",
    "commit_aborted": "server_error",
    "input_audio_buffer_empty": "invalid_request_error",
    "missing_item_id": "invalid_request_error",
    "item_not_found": "invalid_request_error",
    "playback_item_mismatch": "invalid_request_error",
    "playback_item_not_found": "invalid_request_error",
    "playback_ack_too_late": "invalid_request_error",
    "unsupported_audio_format": "invalid_request_error",
    "unsupported_turn_detection": "invalid_request_error",
    "unsupported_ref_audio_path": "invalid_request_error",
    "ref_audio_required": "invalid_request_error",
    "model_update_unsupported": "invalid_request_error",
    "voice_update_after_audio_unsupported": "invalid_request_error",
    "ref_audio_update_unsupported": "invalid_request_error",
    "native_text_append_unsupported": "invalid_request_error",
    "invalid_video_frames": "invalid_request_error",
    "invalid_function_call_output": "invalid_request_error",
    "server_vad_unavailable": "server_error",
}


def realtime_error_type(code: str) -> str:
    """The OpenAI ``error.type`` bucket for an internal error code."""
    return REALTIME_ERROR_TYPES_BY_CODE.get(code, "invalid_request_error")

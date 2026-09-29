# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from vllm_omni.engine.duplex.config import DuplexCapabilities


def nemotron_voicechat_capabilities(*, max_sessions: int = 1) -> DuplexCapabilities:
    """The Nemotron VoiceChat native-duplex preset of the generic capability payload.

    One 80 ms acoustic frame is appended to one resumable Stage-0 request, so
    the scheduler owns the frame clock (``requires_model_runner_kv``) and the
    model decides by itself when to speak (``supports_model_native_turn_policy``).
    Barge-in and mid-incarnation context edits are not supported: the fused
    prompt/audio KV cannot be revised in place.
    """
    supports_multi_session = max_sessions > 1
    return DuplexCapabilities(
        supports_model_native_turn_policy=True,
        supports_external_turn_signal=False,
        supports_client_commit=True,
        supports_barge_in=False,
        supports_playback_ack=True,
        supports_input_append=True,
        supports_replace_latest_chunk=False,
        supports_reencode_context=False,
        supports_rollback_to_checkpoint=False,
        supports_turn_commit_only=False,
        supports_kv_lease=False,
        supports_core_kv_lease=False,
        supports_model_internal_state=True,
        supports_stage_resumption=True,
        supports_scheduler_native_append=False,
        supports_core_resumable_request=True,
        supports_stage_connector_handoff=True,
        supports_independent_io_streams=True,
        supports_realtime_endpoint=True,
        supports_multi_session=supports_multi_session,
        supports_multi_session_same_replica=False,
        supports_session_lease=True,
        supports_session_resume=True,
        session_admission_mode="engine_managed",
        supports_audio_truncate=False,
        requires_model_runner_kv=True,
        requires_native_stage_role=True,
        adapter_patterns=["scheduler_data_plane"],
        signal_sources=["model_native", "client_event"],
        stage_handoff_transport="scheduler_data_plane",
        chunk_period_ms=80,
        target_barge_in_latency_ms=None,
    )


__all__ = ["nemotron_voicechat_capabilities"]

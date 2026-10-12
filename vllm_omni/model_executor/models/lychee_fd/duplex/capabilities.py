# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unified Duplex capabilities for the native Lychee session implementation."""

from vllm_omni.engine.duplex.config import DuplexCapabilities

LYCHEE_CHUNK_PERIOD_MS = 400


def lychee_native_capabilities(*, max_sessions: int = 1) -> DuplexCapabilities:
    """Describe the native session API and the configured admission surface.

    Multi-session capacity remains deployment-owned. C1 stays the default;
    the explicit C2/C4 profiles configure both stage schedulers and AR KV.
    """
    supports_multi_session = max_sessions > 1
    return DuplexCapabilities(
        supports_model_native_turn_policy=True,
        supports_external_turn_signal=True,
        supports_client_commit=True,
        supports_barge_in=True,
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
        supports_concurrent_turn_requests=False,
        required_input_modalities=frozenset({"audio"}),
        optional_input_modalities=frozenset(),
        supports_stage_connector_handoff=False,
        supports_independent_io_streams=False,
        supports_realtime_endpoint=True,
        supports_multi_session=supports_multi_session,
        supports_multi_session_same_replica=supports_multi_session,
        supports_session_lease=True,
        supports_session_resume=True,
        session_admission_mode="engine_managed",
        supports_audio_truncate=False,
        supports_image_input=False,
        supports_text_only_turn=False,
        supports_chat_completions=False,
        requires_model_runner_kv=True,
        requires_native_stage_role=True,
        adapter_patterns=["scheduler_data_plane"],
        signal_sources=["model_native", "client_event"],
        stage_handoff_transport=None,
        chunk_period_ms=LYCHEE_CHUNK_PERIOD_MS,
        target_barge_in_latency_ms=None,
    )


__all__ = ["LYCHEE_CHUNK_PERIOD_MS", "lychee_native_capabilities"]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from vllm_omni.engine.duplex.config import DuplexCapabilities


def qwen3_omni_duplex_capabilities(*, max_sessions: int = 1) -> DuplexCapabilities:
    """Qwen3-Omni turn-commit duplex capabilities."""
    supports_multi_session = max_sessions > 1
    return DuplexCapabilities(
        supports_model_native_turn_policy=False,
        supports_barge_in=False,
        supports_input_append=True,
        supports_replace_latest_chunk=False,
        supports_reencode_context=False,
        supports_turn_commit_only=True,
        supports_kv_lease=False,
        supports_core_kv_lease=False,
        supports_model_internal_state=False,
        supports_stage_resumption=False,
        supports_scheduler_native_append=False,
        supports_core_resumable_request=False,
        supports_stage_connector_handoff=False,
        supports_independent_io_streams=False,
        supports_realtime_endpoint=True,
        supports_multi_session=supports_multi_session,
        supports_multi_session_same_replica=supports_multi_session,
        supports_session_lease=True,
        supports_session_resume=False,
        session_admission_mode="engine_managed",
        supports_audio_truncate=False,
        supports_chat_completions=True,
        requires_model_runner_kv=False,
        requires_native_stage_role=False,
        adapter_patterns=["turn_commit"],
        signal_sources=["client_event", "server_policy"],
        chunk_period_ms=1000,
        target_barge_in_latency_ms=None,
    )


__all__ = ["qwen3_omni_duplex_capabilities"]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from vllm_omni.engine.duplex.config import DuplexCapabilities


def aura_duplex_capabilities(*, max_sessions: int = 1) -> DuplexCapabilities:
    """AURA turn-commit duplex capabilities (not MiniCPM resident append)."""
    supports_multi_session = max_sessions > 1
    return DuplexCapabilities(
        supports_model_native_turn_policy=False,
        supports_barge_in=True,
        supports_input_append=True,
        supports_replace_latest_chunk=False,
        supports_reencode_context=False,
        supports_turn_commit_only=True,
        supports_kv_lease=False,
        supports_core_kv_lease=False,
        supports_model_internal_state=True,
        supports_stage_resumption=False,
        supports_scheduler_native_append=False,
        supports_core_resumable_request=False,
        supports_overlapped_commit=True,
        # Video required, audio optional (vision-only / silent+frames legal).
        required_input_modalities=frozenset({"video"}),
        optional_input_modalities=frozenset({"audio"}),
        supports_stage_connector_handoff=False,
        supports_independent_io_streams=True,
        supports_realtime_endpoint=True,
        supports_multi_session=supports_multi_session,
        supports_multi_session_same_replica=supports_multi_session,
        supports_session_lease=True,
        supports_session_resume=True,
        session_admission_mode="engine_managed",
        supports_audio_truncate=False,
        requires_model_runner_kv=False,
        requires_native_stage_role=False,
        adapter_patterns=["turn_commit"],
        signal_sources=["client_event", "server_policy"],
        stage_handoff_transport="stage_connector",
        chunk_period_ms=1000,
        target_barge_in_latency_ms=None,
    )


__all__ = ["aura_duplex_capabilities"]

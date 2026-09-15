# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from vllm_omni.entrypoints.duplex.protocol import DuplexCapabilities


def minicpmo45_native_capabilities(*, max_sessions: int = 1) -> DuplexCapabilities:
    """The MiniCPM-o 4.5 native-duplex preset of the generic capability payload."""
    supports_multi_session = max_sessions > 1
    return DuplexCapabilities(
        supports_model_native_turn_policy=True,
        supports_barge_in=True,
        supports_input_append=True,
        supports_replace_latest_chunk=False,
        supports_reencode_context=False,
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
        supports_multi_session_same_replica=supports_multi_session,
        supports_session_lease=True,
        supports_session_resume=True,
        session_admission_mode="engine_managed",
        supports_audio_truncate=True,
        requires_model_runner_kv=True,
        requires_native_stage_role=True,
        implementation_level="model_native_duplex",
        adapter_patterns=["scheduler_data_plane"],
        input_modes=["append_audio_chunk"],
        signal_sources=["model_native", "client_event", "server_policy"],
        stage_handoff_transport="scheduler_data_plane",
        chunk_period_ms=1000,
        # Barge-in latency depends on client chunking and lacks hardware E2E
        # measurement, so do not advertise an invented target.
        target_barge_in_latency_ms=None,
    )


__all__ = ["minicpmo45_native_capabilities"]

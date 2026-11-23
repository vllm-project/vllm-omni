# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capabilities and codec frame constants for Magenta RealTime 2 full-duplex streaming."""

from __future__ import annotations

from vllm_omni.engine.duplex.config import DuplexCapabilities

SAMPLE_RATE = 48000
CHANNELS = 2
FRAME_RATE = 25.0
FRAME_SIZE = 1920  # 40 ms @ 48 kHz
CHUNK_PERIOD_MS = 40


def magenta_rt2_capabilities(*, max_sessions: int = 1) -> DuplexCapabilities:
    """Advertise Magenta RealTime 2's 25 Hz lockstep full-duplex streaming contract."""
    supports_multi_session = max_sessions > 1
    return DuplexCapabilities(
        supports_model_native_turn_policy=True,
        supports_external_turn_signal=False,
        supports_client_commit=False,
        supports_barge_in=False,
        supports_playback_ack=True,
        supports_input_append=True,
        supports_replace_latest_chunk=False,
        supports_reencode_context=False,
        supports_rollback_to_checkpoint=False,
        supports_turn_commit_only=False,
        supports_model_internal_state=True,
        supports_stage_resumption=True,
        supports_core_resumable_request=True,
        supports_stage_connector_handoff=True,
        supports_independent_io_streams=True,
        supports_realtime_endpoint=True,
        supports_multi_session=supports_multi_session,
        supports_multi_session_same_replica=supports_multi_session,
        supports_session_lease=True,
        supports_session_resume=False,
        session_admission_mode="engine_managed",
        supports_audio_truncate=False,
        supports_chat_completions=False,
        requires_model_runner_kv=False,
        requires_native_stage_role=True,
        adapter_patterns=["scheduler_data_plane"],
        signal_sources=["model_native", "client_event"],
        stage_handoff_transport="scheduler_data_plane",
        chunk_period_ms=CHUNK_PERIOD_MS,
        target_barge_in_latency_ms=None,
    )


__all__ = [
    "CHANNELS",
    "CHUNK_PERIOD_MS",
    "FRAME_RATE",
    "FRAME_SIZE",
    "SAMPLE_RATE",
    "magenta_rt2_capabilities",
]

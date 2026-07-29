# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-Omni duplex constants and server-owned configuration keys.

Model-specific names live here so the generic scheduler/orchestrator path
never learns about Qwen3-Omni. This mirrors
``vllm_omni/experimental/fullduplex/minicpmo45/policy.py``.
"""

from __future__ import annotations


class Qwen3OmniDuplexPolicy:
    """Model-specific constants for the Qwen3-Omni duplex path."""

    #: Thinker audio input rate. Qwen3-Omni's feature extractor reports this
    #: via ``processor.feature_extractor.sampling_rate``; the serving layer
    #: cannot import the processor, so the value is pinned here and MUST be
    #: cross-checked against the checkpoint at startup.
    SAMPLE_RATE_HZ = 16000

    #: Input chunk cadence for a duplex append, in milliseconds.
    #:
    #: NOTE this is deliberately far shorter than the 5000 ms segment used by
    #: the half-duplex ``/v1/realtime`` path
    #: (``Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio``). That
    #: 5 s segment is the dominant term in barge-in latency. 1000 ms matches
    #: the cadence the duplex framework has actually been exercised at (for
    #: MiniCPM-o 4.5) and is a starting point, NOT a validated value for
    #: Qwen3-Omni. Shortening the chunk changes model input statistics and
    #: needs a quality evaluation before it can be called correct.
    CHUNK_PERIOD_MS = 1000

    #: Samples consumed per emitted chunk.
    CHUNK_SAMPLES = SAMPLE_RATE_HZ * CHUNK_PERIOD_MS // 1000

    #: Audio samples represented by one thinker embedding slot.
    #:
    #: !! UNVERIFIED — MUST be confirmed against the checkpoint before this
    #: path can work. It determines how many scheduler token slots each
    #: append reserves; if it is wrong the reservation and the produced
    #: embedding count disagree and the worker will truncate or pad silently.
    #: Derive it from the audio tower's mel hop length, conv stride, and any
    #: pooling ratio rather than trusting this default.
    SAMPLES_PER_AUDIO_TOKEN = 1600

    #: Wire format accepted from the Realtime client. The serving layer
    #: normalizes to this before anything reaches the worker.
    PCM_FORMAT = "pcm_f32le"

    #: Server-owned runtime-config keys. A client that sets any of these in
    #: ``extra_body`` is rejected rather than silently overridden.
    PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
        {
            "duplex_stage_sampling_params",
            "duplex_stage_max_tokens",
            "duplex_scheduler_token_id",
            "duplex_scheduler_token_budget",
            "duplex_chunk_period_ms",
            "duplex_samples_per_audio_token",
        }
    )

    #: Client opt-in flag on ``extra_body``.
    ENABLE_FLAG = "qwen3_omni_native_duplex"

    @classmethod
    def tokens_per_chunk(cls, *, samples_per_token: int | None = None) -> int:
        """Scheduler token slots to reserve for one appended audio chunk."""
        per_token = samples_per_token or cls.SAMPLES_PER_AUDIO_TOKEN
        if per_token <= 0:
            raise ValueError("samples_per_audio_token must be positive")
        return max(1, -(-cls.CHUNK_SAMPLES // per_token))

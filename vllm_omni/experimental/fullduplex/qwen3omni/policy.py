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
    #: 1000 ms is not arbitrary: the thinker's audio tower splits its conv
    #: input into ``n_window * 2 == 100`` mel-frame chunks
    #: (``qwen3_omni_moe_thinker.py``, ``Qwen3OmniMoeAudioEncoder.forward``),
    #: and at ``hop_length=160`` / 16 kHz that is exactly 16000 samples =
    #: 1.0 s. A 1 s duplex chunk therefore lands on the model's own conv
    #: chunk boundary, so per-chunk encoding introduces no convolutional
    #: boundary error. (The tower's ATTENTION still spans up to
    #: ``n_window_infer // (n_window * 2) == 8`` such chunks, so streaming
    #: one chunk at a time remains an approximation at the attention level,
    #: just not at the conv level.) Changing this value away from a multiple
    #: of 1 s reintroduces conv boundary error.
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

    #: Mel-spectrogram hop length. ``WhisperFeatureExtractor`` with
    #: ``hop_length=160`` at 16 kHz => one mel frame per 10 ms. Verified
    #: against Qwen3-Omni-30B-A3B-Instruct ``preprocessor_config.json``.
    MEL_HOP_LENGTH = 160

    #: Wire format accepted from the Realtime client. The serving layer
    #: normalizes to this before anything reaches the worker.
    PCM_FORMAT = "pcm_f32le"

    #: ``<|audio_pad|>``. Occupies the prompt positions that stage 0 overwrites
    #: with audio embeddings, so the token ids stay meaningful rather than
    #: being filler. Matches ``qwen3_omni.py:AUDIO_PAD_TOKEN_ID``.
    AUDIO_PAD_TOKEN_ID = 151675

    #: Conversation scaffolding.
    #:
    #: Without this the thinker receives bare audio embeddings with no
    #: instruction to reply and emits EOS immediately. These mirror the
    #: framing the half-duplex path already uses in
    #: ``Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio``, so
    #: duplex and half-duplex present the model with the same shape.
    #:
    #: Emitted once per session, ahead of the first user turn.
    SESSION_PREFIX_TEMPLATE = "<|im_start|>system\n{instructions}<|im_end|>\n"
    #: Opens each user turn.
    TURN_PREFIX = "<|im_start|>user\n"
    #: Closes the user turn and hands the floor to the assistant.
    TURN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

    #: Runtime-config keys carrying the pre-tokenized scaffolding. Tokenizing
    #: happens once in the serving layer, where the tokenizer lives; the
    #: worker only embeds the ids. This keeps the engine's slot reservation
    #: and the worker's produced-embedding count derived from one source.
    SESSION_PREFIX_IDS_KEY = "duplex_session_prefix_token_ids"
    TURN_PREFIX_IDS_KEY = "duplex_turn_prefix_token_ids"
    TURN_SUFFIX_IDS_KEY = "duplex_turn_suffix_token_ids"

    #: Server-owned runtime-config keys. A client that sets any of these in
    #: ``extra_body`` is rejected rather than silently overridden.
    PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
        {
            "duplex_stage_sampling_params",
            "duplex_stage_max_tokens",
            "duplex_scheduler_token_id",
            "duplex_scheduler_token_budget",
            "duplex_chunk_period_ms",
            "duplex_session_prefix_token_ids",
            "duplex_turn_prefix_token_ids",
            "duplex_turn_suffix_token_ids",
        }
    )

    #: Client opt-in flag on ``extra_body``.
    ENABLE_FLAG = "qwen3_omni_native_duplex"

    #: Marks an audio payload produced by a client commit rather than a
    #: mid-turn append.
    #:
    #: The framework's append path always passes ``final=False``
    #: (``session_runner.py:1184,1434``) because MiniCPM decides listen/speak
    #: natively and never needs the turn closed for it. Qwen3-Omni does: the
    #: assistant generation prompt is what produces a reply. The commit is
    #: therefore signalled on the payload, which is model-owned data and
    #: survives ``_merge_native_audio_payloads`` (it does ``dict(second)``).
    TURN_FINAL_KEY = "duplex_turn_final"

    @staticmethod
    def audio_tokens_for_mel_frames(mel_frames: int) -> int:
        """Thinker audio tokens produced by ``mel_frames`` mel frames.

        Mirrors ``_get_feat_extract_output_lengths`` in vLLM's
        ``qwen3_omni_moe_thinker.py`` exactly -- the conv stack's own length
        arithmetic. Kept as an integer reimplementation rather than an import
        so the serving layer does not pull in torch, but it MUST stay in sync
        with that function; ``test_contracts.py`` pins the shared cases.

        Note this is not a linear samples-per-token ratio: the ``// 100 * 13``
        term makes it 13 tokens per whole second plus a sub-second remainder.
        A linear approximation is wrong by ~30% (10 vs 13 tokens/s) and would
        under-reserve scheduler slots, which the model runner absorbs
        silently by truncating embeddings.
        """
        if mel_frames <= 0:
            return 0
        leave = mel_frames % 100
        feat_lengths = (leave - 1) // 2 + 1
        return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (mel_frames // 100) * 13

    @classmethod
    def audio_tokens_for_samples(cls, num_samples: int) -> int:
        """Thinker audio tokens produced by ``num_samples`` PCM samples."""
        return cls.audio_tokens_for_mel_frames(num_samples // cls.MEL_HOP_LENGTH)

    @classmethod
    def tokens_per_chunk(cls) -> int:
        """Scheduler token slots to reserve for one whole appended chunk."""
        return max(1, cls.audio_tokens_for_samples(cls.CHUNK_SAMPLES))

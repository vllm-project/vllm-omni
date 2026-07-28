from __future__ import annotations

from typing import Any


class MiniCPMO45DuplexPolicy:
    """MiniCPM-o 4.5 native duplex model policy.

    Keep model-specific token names, input modes, and handoff type strings out
    of the generic scheduler/orchestrator path. These names are part of the
    MiniCPM-o 4.5 remote-code contract, not a general vLLM-Omni duplex schema.
    """

    # Audio framing contract shared by serving, orchestrator, and worker.
    # MiniCPM-o consumes 1 s units at 16 kHz and pools audio to one embedding
    # per 100 ms, so a unit contributes exactly 10 audio embeddings plus the
    # <unit> open (and a </unit> closure for every unit after the first).
    # Scheduler token budgets must match the worker-built embeddings exactly:
    # surplus slots become pad embeddings inside the KV and measurably corrupt
    # the model's listen/speak behavior.
    SAMPLE_RATE_HZ = 16000
    CHUNK_SAMPLES = 16000
    SAMPLES_PER_AUDIO_TOKEN = 1600
    # Vision framing contract (omni duplex). Official streaming_prefill feeds
    # each frame as <image> + 64 resampler embeddings + </image> inside the
    # unit, ahead of the unit's audio embeddings (max_slice_nums=1 in
    # streaming, so exactly one 64-token block per frame).
    VISION_EMBEDS_PER_FRAME = 64
    VISION_TOKENS_PER_FRAME = VISION_EMBEDS_PER_FRAME + 2  # <image> + embeds + </image>
    DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK = 20
    DEFAULT_MAX_SPEAK_CHARS_PER_CHUNK = 28
    DEFAULT_MIN_NEW_SPEAK_TOKENS_BEFORE_CHUNK_BOUNDARY = 8
    REPETITION_HISTORY_SIZE = 512

    @classmethod
    def audio_token_count(cls, sample_count: int) -> int:
        """Audio embedding count for a clip of ``sample_count`` samples.

        Matches the whole-clip encoder math (hop 160 mel frames -> CNN stride 2
        -> avg-pool 5) for any multiple of ``SAMPLES_PER_AUDIO_TOKEN``; serving
        normalizes clips to that boundary before this is used for budgets.
        """
        return max(0, int(sample_count) // cls.SAMPLES_PER_AUDIO_TOKEN)

    @staticmethod
    def session_context_texts(instructions: object, has_ref_audio: bool) -> tuple[str, str]:
        """System-context prefix/suffix, matching MiniCPMODuplex.prepare()."""
        system_prompt = (
            instructions if isinstance(instructions, str) and instructions else "Streaming Omni Conversation."
        )
        prefix = f"<|im_start|>system\n{system_prompt}"
        suffix = "<|im_end|>"
        if has_ref_audio:
            prefix += "\n<|audio_start|>"
            suffix = "<|audio_end|>" + suffix
        return prefix, suffix

    SPECIAL_TOKEN_FIELDS: dict[str, str] = {
        "unit_token_id": "<unit>",
        "unit_end_token_id": "</unit>",
        "listen_token_id": "<|listen|>",
        "speak_token_id": "<|speak|>",
        "tts_bos_token_id": "<|tts_bos|>",
        "tts_eos_token_id": "<|tts_eos|>",
        "tts_pad_token_id": "<|tts_pad|>",
        "chunk_eos_token_id": "<|chunk_eos|>",
        "chunk_tts_eos_token_id": "<|chunk_tts_eos|>",
        "turn_eos_token_id": "<|turn_eos|>",
    }
    OPTIONAL_TOKEN_FIELDS: dict[str, str] = {
        "audio_placeholder_token_id": "<|audio|>",
        "image_start_token_id": "<image>",
        "image_end_token_id": "</image>",
    }

    @classmethod
    def token_ids_from_tokenizer(cls, tokenizer: Any) -> dict[str, int]:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)

        def token_id(token: str) -> int:
            if not callable(convert):
                value = None
            else:
                value = convert(token)
                if isinstance(value, list):
                    value = value[0] if len(value) == 1 else None
            unk_token_id = getattr(tokenizer, "unk_token_id", None)
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                candidate = -1
            if candidate >= 0 and candidate != unk_token_id:
                return candidate

            encode = getattr(tokenizer, "encode", None)
            if callable(encode):
                try:
                    ids = list(encode(token, add_special_tokens=False))
                except TypeError:
                    ids = list(encode(token))
                if len(ids) == 1:
                    try:
                        candidate = int(ids[0])
                    except (TypeError, ValueError):
                        candidate = -1
                    if candidate >= 0 and candidate != unk_token_id:
                        return candidate
            return -1

        resolved = {field: token_id(token) for field, token in cls.SPECIAL_TOKEN_FIELDS.items()}
        resolved.update({field: token_id(token) for field, token in cls.OPTIONAL_TOKEN_FIELDS.items()})
        if resolved.get("listen_token_id", -1) < 0:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            try:
                resolved["listen_token_id"] = int(eos_id)
            except (TypeError, ValueError):
                pass
        return resolved

    @staticmethod
    def native_forbidden_token_ids(
        token_ids: dict[str, int],
        *,
        bad_token_ids: list[int] | tuple[int, ...] = (),
    ) -> list[int]:
        return [
            token_ids.get("tts_pad_token_id", -1),
            *list(bad_token_ids),
            token_ids.get("chunk_eos_token_id", -1),
        ]

    @staticmethod
    def native_special_token_ids(
        token_ids: dict[str, int],
        *,
        tokenizer_special_ids: list[int] | tuple[int, ...] = (),
    ) -> set[int]:
        special = set(tokenizer_special_ids or [])
        special.update(value for value in token_ids.values() if isinstance(value, int) and value >= 0)
        return special

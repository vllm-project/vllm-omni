import math
from typing import Any, Literal

import numpy as np
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

_MAX_EMBEDDING_DIM = 8192


def _normalize_ref_audio_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("'ref_audio' list entries must be strings")
            items.append(item)
        if not items:
            raise ValueError("'ref_audio' list cannot be empty")
        return items
    raise TypeError("'ref_audio' must be a string or list of strings")


def _normalize_speaker_embedding_value(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError("'speaker_embedding' must be a list of numbers or list of embedding vectors")
    if not value:
        return []

    first = value[0]
    if isinstance(first, (list, tuple)):
        embeddings = []
        for item in value:
            if not isinstance(item, (list, tuple)):
                raise TypeError("'speaker_embedding' must not mix flat and nested values")
            embeddings.append([float(x) for x in item])
        return embeddings

    return [float(x) for x in value]


class OpenAICreateSpeechRequest(BaseModel):
    input: str
    model: str | None = None
    # Accept both "voice" (OpenAI convention) and "speaker" (model/internal
    # convention) as input keys.  Intentionally global — all TTS backends
    # (Qwen3-TTS, Voxtral, Fish Speech) use this field for the speaker name.
    voice: str | None = Field(
        default=None,
        validation_alias=AliasChoices("voice", "speaker"),
        description="Speaker/voice to use. For Qwen3-TTS: vivian, ryan, aiden, etc.",
    )
    instructions: str | None = Field(
        default=None,
        description="Instructions for voice style/emotion (maps to 'instruct' for Qwen3-TTS)",
    )
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
    )
    stream_format: Literal["sse", "audio"] | None = Field(
        default=None,
        description=(
            "Streaming output format. 'audio' streams raw pcm/wav bytes; "
            "'sse' streams OpenAI speech.audio.* SSE events. If omitted, stream=true "
            "selects SSE and stream=false remains non-streaming."
        ),
    )
    stream: bool = Field(
        default=False,
        description=(
            "Streaming switch; defaults to OpenAI speech.audio.* SSE events. "
            "Set stream_format='audio' to opt into raw pcm/wav byte streaming. "
            "Requires response_format='pcm' or 'wav'. Speed adjustment is not supported when streaming."
        ),
    )

    # Qwen3-TTS specific parameters
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = Field(
        default=None,
        description="TTS task type: CustomVoice, VoiceDesign, or Base (voice clone)",
    )
    language: str | None = Field(
        default=None,
        description="Language code (e.g., 'Chinese', 'English', 'Auto')",
    )
    ref_audio: str | list[str] | None = Field(
        default=None,
        description="Reference audio for voice cloning (Base task). URL, base64, or file URI.",
    )
    ref_text: str | None = Field(
        default=None,
        description="Transcript of reference audio for voice cloning (Base task)",
    )
    ref_audio_2: str | None = Field(
        default=None,
        description="Second reference audio for two-speaker dialogue (MOSS-TTSD). "
        "URL, base64, or file URI. Ignored by single-speaker models.",
    )
    ambient_sound: str | None = Field(
        default=None,
        description="Sound description for ambient/effect synthesis (MOSS-SoundEffect). "
        "Natural language, e.g. 'ocean waves crashing on a rocky beach'.",
    )
    duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Target audio duration in seconds (MOSS-SoundEffect). Converted to ~12.5 frames/s internally.",
    )
    x_vector_only_mode: bool | None = Field(
        default=None,
        description="Use speaker embedding only without in-context learning (Base task)",
    )
    speaker_embedding: list[float] | list[list[float]] | None = Field(
        default=None,
        max_length=_MAX_EMBEDDING_DIM,
        description="Pre-computed speaker embedding vector (1024-dim for 0.6B, "
        "2048-dim for 1.7B). Skips speaker encoder extraction from ref_audio. "
        "Implies x_vector_only_mode=True. Mutually exclusive with ref_audio.",
    )
    max_new_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**63 - 1,
        description="Random seed for reproducible generation. When set, ensures "
        "deterministic output for the same input text and seed value.",
    )
    initial_codec_chunk_frames: int | None = Field(
        default=None,
        ge=0,
        description="Per-request initial chunk size override. If null, computed dynamically based on server load.",
    )
    non_streaming_mode: bool | None = Field(
        default=None,
        description=(
            "Qwen3-TTS prompt construction mode override. This does not "
            "control HTTP response streaming or async-chunk pipelining. "
            "When null, use model defaults: Base=False, CustomVoice/VoiceDesign=True."
        ),
    )
    extra_params: dict[str, Any] | None = Field(
        default=None,
        description=("Optional model-specific parameters passed directly to the model's extra_args."),
    )
    word_timestamps: bool = Field(
        default=False,
        description=(
            "When true, the server runs a shared forced aligner alongside the streamed "
            "audio and emits per-chunk word timestamps. Requires the server to be "
            "launched with --forced-aligner pointing at an aligner model. No effect "
            "when streaming is off."
        ),
    )

    @field_validator("stream_format")
    @classmethod
    def validate_stream_format(cls, v: str) -> str:
        return v

    @field_validator("ref_audio", mode="before")
    @classmethod
    def normalize_ref_audio(cls, v):
        return _normalize_ref_audio_value(v)

    @field_validator("speaker_embedding")
    @classmethod
    def validate_speaker_embedding(
        cls, v: list[float] | list[list[float]] | None
    ) -> list[float] | list[list[float]] | None:
        v = _normalize_speaker_embedding_value(v)
        if v is None:
            return None
        if not v:
            return []
        if isinstance(v[0], list):
            for item in v:
                if not item:
                    raise ValueError("'speaker_embedding' nested vectors must be non-empty")
                if not all(math.isfinite(x) for x in item):
                    raise ValueError("'speaker_embedding' values must be finite (no NaN or Inf)")
            return v
        if not all(math.isfinite(x) for x in v):
            raise ValueError("'speaker_embedding' values must be finite (no NaN or Inf)")
        return v

    @model_validator(mode="before")
    @classmethod
    def normalize_references_alias(cls, data):
        """Map the BosonAI Higgs Audio v3 cookbook ``references`` array onto ``ref_audio`` / ``ref_text``.

        Upstream Higgs Audio v3 examples and the boson.ai cookbook send voice
        clones as::

            {"references": [{"audio_path": str, "text": str}]}

        vllm-omni's existing convention is ``ref_audio`` + ``ref_text``.
        Without this alias pydantic silently drops the unknown ``references``
        field and the request falls through to zero-shot synthesis with no
        error, which is hard to debug from the client side. The normalizer
        translates a single reference and rejects multi-reference payloads
        (vllm-omni does not yet support multi-shot voice clone) or conflicts
        with the explicit ``ref_audio`` / ``ref_text`` fields.
        """
        if not isinstance(data, dict):
            return data
        refs = data.get("references")
        if refs is None:
            return data
        if not isinstance(refs, list) or len(refs) == 0:
            raise ValueError("'references' must be a non-empty array of {audio_path, text} objects")
        if len(refs) > 1:
            raise ValueError("'references' only supports a single reference; multi-shot voice clone is not supported")
        first = refs[0]
        if not isinstance(first, dict):
            raise ValueError("'references[0]' must be an object with 'audio_path' and optional 'text'")
        audio_path = first.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(
                "'references[0].audio_path' is required and must be a string (URL, data: URI, or file path)"
            )
        existing_ref_audio = data.get("ref_audio")
        if existing_ref_audio and existing_ref_audio != audio_path:
            raise ValueError("'references' and 'ref_audio' are mutually exclusive")
        data["ref_audio"] = audio_path
        text = first.get("text")
        if isinstance(text, str) and text.strip():
            existing_ref_text = data.get("ref_text")
            if existing_ref_text and existing_ref_text != text:
                raise ValueError("'references[0].text' and 'ref_text' conflict; supply only one")
            data["ref_text"] = text
        data.pop("references", None)
        return data

    @model_validator(mode="before")
    @classmethod
    def reject_higgs_audio_v2_unsupported_aliases(cls, data):
        """Reject unsupported rich-input aliases for higgs_audio_v2 BEFORE pydantic strips them.

        OpenAICreateSpeechRequest is a permissive schema (it accepts any extra
        field that the OpenAI Speech API didn't promise to ban) but several
        of those aliases pull a request out of the v1 higgs_audio_v2 scope.
        Catching them at parse time lets the API return a deterministic 4xx
        with a model-specific error message instead of silently dropping the
        field and proceeding with a degraded request.
        """
        # Keys that are silently dropped by pydantic when posted to /v1/audio/speech
        # (the schema doesn't declare them) but which a model-specific
        # validator MUST be able to reject. Kept inline to avoid clashing
        # with pydantic's ModelPrivateAttr discovery for class-level
        # underscore-prefixed attributes.
        higgs_audio_v2_reserved_keys = (
            "messages",  # ChatML rich content (out of scope for v1)
            "reference_audio",  # voice-cloning alias 1
            "voice_prompt",  # voice-cloning alias 2
            "speaker_audio",  # voice-cloning alias 3
            "speakers",  # multi-speaker dialogue
        )
        if not isinstance(data, dict):
            return data
        model_id = data.get("model")
        if not isinstance(model_id, str):
            return data
        # Match the "higgs_audio_v2" model_type label, the HF architecture id
        # in pipeline_registry.hf_architectures, and the hyphenated HF repo id
        # (e.g. "bosonai/higgs-audio-v2-generation-3B-base") by normalizing
        # both underscores and hyphens out of the id before substring matching.
        normalized = model_id.lower().replace("-", "").replace("_", "")
        if "higgsaudiov2" not in normalized:
            return data
        offending = sorted(k for k in higgs_audio_v2_reserved_keys if k in data)
        if offending:
            raise ValueError(
                "higgs_audio_v2 v1 does not support these rich-input fields: "
                f"{offending}. Supply plain text via the 'input' field instead."
            )
        return data

    @model_validator(mode="after")
    def validate_embedding_constraints(self) -> "OpenAICreateSpeechRequest":
        if self.speaker_embedding is not None:
            if self.ref_audio is not None and not isinstance(self.ref_audio, list):
                raise ValueError("'speaker_embedding' and 'ref_audio' are mutually exclusive")
        return self

    def is_raw_audio_stream(self) -> bool:
        return self.stream_format == "audio"

    def is_sse_stream(self) -> bool:
        return (self.stream or self.stream_format == "sse") and not self.is_raw_audio_stream()

    def is_streaming(self) -> bool:
        return self.is_raw_audio_stream() or self.is_sse_stream()

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "OpenAICreateSpeechRequest":
        if self.is_streaming():
            if self.response_format not in ("pcm", "wav"):
                raise ValueError(
                    "Streaming (stream=true, stream_format='audio', or stream_format='sse') "
                    "requires response_format='pcm' or 'wav'. "
                    f"Got response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError("Speed adjustment is not supported when streaming. Set speed=1.0 or omit it.")
        return self


class OpenAICreateAudioGenerateRequest(BaseModel):
    """Request model for audio generation via diffusion models (e.g. Stable Audio)."""

    input: str = Field(
        description="Text prompt describing the audio to generate",
    )
    model: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
    )
    stream_format: Literal["sse", "audio"] | None = "audio"
    audio_length: float | None = Field(
        default=None,
        description="Audio length in seconds",
    )
    audio_start: float | None = Field(
        default=0.0,
        description="Audio start time in seconds",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="Negative prompt for classifier-free guidance",
    )
    guidance_scale: float | None = Field(
        default=None,
        description="Guidance scale for diffusion models",
    )
    num_inference_steps: int | None = Field(
        default=None,
        description="Number of inference steps",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )

    @field_validator("stream_format")
    @classmethod
    def validate_stream_format(cls, v: str) -> str:
        if v == "sse":
            raise ValueError("'sse' is not a supported stream_format yet. Please use 'audio'.")
        return v


class CreateAudio(BaseModel):
    audio_tensor: np.ndarray
    sample_rate: int = 24000
    response_format: str = "wav"
    speed: float = 1.0
    base64_encode: bool = True

    class Config:
        arbitrary_types_allowed = True


class AudioResponse(BaseModel):
    audio_data: bytes | str
    media_type: str


# --- Batch Speech Models ---


class SpeechBatchItem(BaseModel):
    """Per-item input for batch speech. Only `input` is required;
    all other fields override the batch-level defaults when set."""

    input: str
    voice: str | None = Field(default=None, validation_alias=AliasChoices("voice", "speaker"))
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] | None = None
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)
    non_streaming_mode: bool | None = None


class BatchSpeechRequest(BaseModel):
    """Top-level request for batch speech generation.
    Fields here act as shared defaults; per-item overrides win."""

    model: str | None = None
    items: list[SpeechBatchItem] = Field(..., min_length=1)
    voice: str | None = Field(default=None, validation_alias=AliasChoices("voice", "speaker"))
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = None
    initial_codec_chunk_frames: int | None = Field(default=None, ge=0)
    non_streaming_mode: bool | None = None


class SpeechInputTokenDetails(BaseModel):
    """Per-modality breakdown of the speech request's *input* tokens.

    The aggregate ``input_tokens`` on :class:`SpeechTokenUsage` is the sum of
    these. We surface the split (rather than one opaque number) for the same
    reason OpenAI's realtime/chat usage does: text and audio inputs are billed
    and reasoned about differently, and folding them together is misleading.

    Fields:
        text_tokens: Tokens of the text to synthesize. This is ``input`` plus
            ``instructions`` (style/emotion prompt), because both are tokenized
            into the model prefill. This is the number that should scale with
            how much text the caller asked to speak.
        audio_tokens: Reference-audio codec frames used as voice-cloning
            conditioning. NON-ZERO only when in-context voice cloning is
            actually active (Qwen3-TTS ``task_type='Base'`` ICL). It is 0 for
            CustomVoice/VoiceDesign and for x-vector-only cloning, because those
            paths put no reference codec frames into the prefill. See issue
            #4646: this is the value that previously leaked into ``prompt_tokens``
            and made Base usage look independent of the input text.
    """

    text_tokens: int = 0
    audio_tokens: int = 0


class SpeechTokenUsage(BaseModel):
    """Token usage for a speech (TTS) request.

    Field naming follows OpenAI's documented ``speech.audio.done`` event
    (``input_tokens``/``output_tokens``/``total_tokens``), NOT chat's
    ``prompt_tokens``/``completion_tokens``.

    input_tokens  = text_tokens + audio_tokens   (see SpeechInputTokenDetails)
    output_tokens = generated codec/audio tokens (stage-0 decode steps)
    total_tokens  = input_tokens + output_tokens

    IMPORTANT: ``input_tokens`` is computed from the *semantic* request inputs
    (tokenized text + reference-audio frames), NOT from ``len(prompt_token_ids)``.
    For staged TTS models that engine prompt is a ``[1] * prefill_len``
    placeholder whose length mirrors the model prefill, so it is not a faithful
    input-token count (issue #4646).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_details: SpeechInputTokenDetails = Field(default_factory=SpeechInputTokenDetails)


class SpeechBatchItemResult(BaseModel):
    index: int
    status: Literal["success", "error"]
    audio_data: str | None = None
    media_type: str | None = None
    error: str | None = None
    # Per-item token usage (input text + reference-audio conditioning, and
    # generated audio tokens). None when the item errored before generation.
    usage: SpeechTokenUsage | None = None


class BatchSpeechResponse(BaseModel):
    id: str
    results: list[SpeechBatchItemResult]
    total: int
    succeeded: int
    failed: int


class StreamingSpeechSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message for streaming TTS."""

    model: str | None = None
    voice: str | None = Field(default=None, validation_alias=AliasChoices("voice", "speaker"))
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] | None = None
    language: str | None = None
    instructions: str | None = None
    response_format: Literal["wav", "pcm", "flac", "mp3", "aac", "opus"] = "wav"
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    max_new_tokens: int | None = Field(default=None, ge=1)
    initial_codec_chunk_frames: int | None = Field(
        default=None,
        ge=0,
        description="Initial chunk size for reduced TTFA. Overrides stage config for this session.",
    )
    non_streaming_mode: bool | None = Field(
        default=None,
        description=(
            "Qwen3-TTS prompt construction mode override. This does not "
            "control WebSocket audio streaming or async-chunk pipelining. "
            "When null, use model defaults: Base=False, CustomVoice/VoiceDesign=True."
        ),
    )
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    speaker_embedding: list[float] | None = Field(
        default=None,
        max_length=_MAX_EMBEDDING_DIM,
        description="Pre-computed speaker embedding vector. Mutually exclusive with ref_audio.",
    )
    stream_audio: bool = Field(
        default=False,
        description=(
            "If true, send raw PCM audio chunks progressively over WebSocket. "
            "Requires response_format='pcm'. Speed adjustment is not supported when streaming."
        ),
    )
    word_timestamps: bool = Field(
        default=False,
        description=(
            "When true, audio chunks are wrapped in JSON 'audio.chunk' frames carrying "
            "base64-encoded PCM plus aligned word timestamps. Requires the server to be "
            "launched with --forced-aligner. When false, audio is sent as raw binary "
            "frames (existing behavior)."
        ),
    )

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "StreamingSpeechSessionConfig":
        if self.stream_audio:
            if self.response_format != "pcm":
                raise ValueError(
                    "WebSocket streaming audio (stream_audio=true) requires response_format='pcm'. "
                    f"Got response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError("Speed adjustment is not supported when stream_audio=true. Set speed=1.0 or omit it.")
        return self

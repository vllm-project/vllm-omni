from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class OpenAICreateSpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: str | None = Field(
        default=None,
        description="Voice to use. For OpenAI: alloy, echo, etc. For Qwen3-TTS: Vivian, Ryan, etc.",
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
    stream_format: Literal["sse", "audio"] | None = "audio"
    stream: bool = Field(
        default=False,
        description=(
            "If true, stream raw PCM audio chunks as they are decoded. "
            "Requires response_format='pcm'. Speed adjustment is not supported when streaming."
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
    ref_audio: str | None = Field(
        default=None,
        description="Reference audio for voice cloning (Base task). URL, base64, or file path.",
    )
    ref_text: str | None = Field(
        default=None,
        description="Transcript of reference audio for voice cloning (Base task)",
    )
    x_vector_only_mode: bool | None = Field(
        default=None,
        description="Use speaker embedding only without in-context learning (Base task)",
    )
    max_new_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate",
    )
    initial_codec_chunk_frames: int | None = Field(
        default=None,
        ge=0,
        description="Per-request initial chunk size override. If null, computed dynamically based on server load.",
    )

    @field_validator("stream_format")
    @classmethod
    def validate_stream_format(cls, v: str) -> str:
        if v == "sse":
            raise ValueError("'sse' is not a supported stream_format yet. Please use 'audio'.")
        return v

    @model_validator(mode="after")
    def validate_streaming_constraints(self) -> "OpenAICreateSpeechRequest":
        if self.stream:
            if self.response_format != "pcm":
                raise ValueError(
                    "Streaming (stream=true) requires response_format='pcm'. "
                    f"Got response_format='{self.response_format}'."
                )
            if self.speed is None:
                self.speed = 1.0
            elif self.speed != 1.0:
                raise ValueError(
                    "Speed adjustment is not supported when streaming (stream=true). Set speed=1.0 or omit it."
                )
        return self


class CreateAudio(BaseModel):
    audio_tensor: np.ndarray
    sample_rate: int = 24000
    response_format: str = "wav"
    speed: float = 1.0
    stream_format: Literal["sse", "audio"] | None = "audio"
    base64_encode: bool = True

    class Config:
        arbitrary_types_allowed = True


class AudioResponse(BaseModel):
    audio_data: bytes | str
    media_type: str


class StreamingSpeechSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message for streaming TTS."""

    model: str | None = None
    voice: str | None = None
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
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    stream_audio: bool = Field(
        default=False,
        description=(
            "If true, send raw PCM audio chunks progressively over WebSocket. "
            "Requires response_format='pcm'. Speed adjustment is not supported when streaming."
        ),
    )
    split_granularity: Literal["sentence", "clause"] = Field(
        default="sentence",
        description=(
            "Text splitting granularity: 'sentence' splits on .!?。！？, "
            "'clause' also splits on CJK commas ， and semicolons ；."
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

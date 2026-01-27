from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator


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
    stream_format: Literal["sse", "audio"] | None = "audio"
    base64_encode: bool = True

    class Config:
        arbitrary_types_allowed = True


class AudioResponse(BaseModel):
    audio_data: bytes | str
    media_type: str

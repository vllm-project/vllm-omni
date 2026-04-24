from collections.abc import Callable
from enum import Enum, auto
from typing import (
    Any,
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
)

AudioFormat: TypeAlias = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class AutoregressionSamplingParams(dict):
    pass


class DiffusionSamplingParams(dict):
    pass


class QwenTTSModelSpecificParams(dict):
    pass


class WanModelSpecificParams(dict):
    pass


class ModelMode(Enum):
    IMAGE_GENERATION = auto()
    VIDEO_GENERATION = auto()
    AUDIO_GENERATION = auto()
    UNDERSTANDING = auto()


class Modality(Enum):
    TEXT = auto()  # maybe not useful. Prompt is always required
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()


class ModelModeSpec(TypedDict):
    mode: ModelMode
    input_modalities: list[Modality]


PayloadPreprocessor: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]


class Spec(TypedDict):
    stages: list[Literal["diffusion", "autoregression"]]
    modes: list[ModelModeSpec]
    payload_preprocessor: NotRequired[PayloadPreprocessor]

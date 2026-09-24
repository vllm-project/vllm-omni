# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Literal, TypeAlias, get_args

from typing_extensions import NotRequired, TypedDict

AudioFormat: TypeAlias = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class AutoregressionSamplingParams(dict):
    pass


class DiffusionSamplingParams(dict):
    pass


class QwenTTSModelSpecificParams(dict):
    pass


class WanModelSpecificParams(dict):
    pass


class MiniMaxH3ModelSpecificParams(dict):
    pass


MiniMaxH3ControlType: TypeAlias = Literal["canny", "depth", "hed", "mlsd", "pose", "inpaint"]
MINIMAX_H3_CONTROL_TYPES = get_args(MiniMaxH3ControlType)


class MiniMaxH3Control(TypedDict):
    control_type: MiniMaxH3ControlType
    control_context_scale: float
    control_video: NotRequired[Any]
    source_video: NotRequired[Any]
    mask: NotRequired[Any]
    mask_video: NotRequired[Any]


MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_TOTAL_REFERENCES = 12


class VideoReferences(dict):
    pass


class LatentMaskEditing(dict):
    pass


class FastH3Deployment(dict):
    """Descriptor for a server that fused FastH3 at startup."""


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
ParamsBuilder: TypeAlias = Callable[..., dict[str, Any]]


class Spec(TypedDict):
    stages: list[Literal["diffusion", "autoregression"]]
    modes: list[ModelModeSpec]
    payload_preprocessor: NotRequired[PayloadPreprocessor]
    params_builder: NotRequired[ParamsBuilder]

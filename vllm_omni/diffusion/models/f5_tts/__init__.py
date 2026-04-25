# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""F5-TTS model support for vLLM-Omni."""

from vllm_omni.diffusion.models.f5_tts.audio_utils import (
    MelSpec,
    MelSpecConfig,
    load_vocoder,
    process_audio,
    run_vocoder,
)
from vllm_omni.diffusion.models.f5_tts.f5_tts_transformer import (
    F5TTSDiTModel,
)
from vllm_omni.diffusion.models.f5_tts.pipeline_f5_tts import (
    F5TTSPipeline,
    get_f5_tts_post_process_func,
)
from vllm_omni.diffusion.models.f5_tts.text_utils import (
    Tokenizer,
    estimate_duration,
    load_tokenizer,
    pad_and_batch,
    process_text,
    quantize,
)

__all__ = [
    # Pipeline
    "F5TTSPipeline",
    "get_f5_tts_post_process_func",
    # Transformer
    "F5TTSDiTModel",
    # Audio utilities
    "MelSpec",
    "MelSpecConfig",
    "load_vocoder",
    "process_audio",
    "run_vocoder",
    # Text utilities
    "Tokenizer",
    "estimate_duration",
    "load_tokenizer",
    "pad_and_batch",
    "process_text",
    "quantize",
]

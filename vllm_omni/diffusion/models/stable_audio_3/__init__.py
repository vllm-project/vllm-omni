# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio 3 model support for vLLM-Omni.

Issue: vllm-project/vllm-omni#3787
Upstream: github.com/Stability-AI/stable-audio-3 (MIT)

Initial scope: text-to-audio. Editing + inpainting deferred.
"""

from vllm_omni.diffusion.models.stable_audio_3.conditioners import (
    MultiConditioner,
    NumberConditioner,
    T5GemmaConditioner,
)
from vllm_omni.diffusion.models.stable_audio_3.diffusion_wrapper import (
    ConditionedDiffusionModelWrapper,
    DiTWrapper,
)
from vllm_omni.diffusion.models.stable_audio_3.pipeline_stable_audio_3 import (
    StableAudio3Pipeline,
    get_stable_audio_3_post_process_func,
)
from vllm_omni.diffusion.models.stable_audio_3.same_autoencoder import (
    AudioAutoencoder,
    SAMEDecoder,
    SAMEEncoder,
    SoftNormBottleneck,
)
from vllm_omni.diffusion.models.stable_audio_3.stable_audio_3_transformer import (
    DiffusionTransformer,
    StableAudio3DiTModel,
)

__all__ = [
    "AudioAutoencoder",
    "ConditionedDiffusionModelWrapper",
    "DiTWrapper",
    "DiffusionTransformer",
    "MultiConditioner",
    "NumberConditioner",
    "SAMEDecoder",
    "SAMEEncoder",
    "SoftNormBottleneck",
    "StableAudio3DiTModel",
    "StableAudio3Pipeline",
    "T5GemmaConditioner",
    "get_stable_audio_3_post_process_func",
]

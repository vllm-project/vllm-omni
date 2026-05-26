# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable Audio 3 (medium) model support for vLLM-Omni.

Issue: https://github.com/vllm-project/vllm-omni/issues/3787
Reference impl: https://github.com/Stability-AI/stable-audio-3 (MIT)

Initial scope: text-to-audio. audio-to-audio editing and inpainting are
out of scope for v1.
"""

from vllm_omni.diffusion.models.stable_audio_3.pipeline_stable_audio_3 import (
    StableAudio3Pipeline,
    get_stable_audio_3_post_process_func,
)
from vllm_omni.diffusion.models.stable_audio_3.same_autoencoder import (
    SAMEAutoencoder,
)
from vllm_omni.diffusion.models.stable_audio_3.stable_audio_3_transformer import (
    StableAudio3DiTModel,
)

__all__ = [
    "SAMEAutoencoder",
    "StableAudio3DiTModel",
    "StableAudio3Pipeline",
    "get_stable_audio_3_post_process_func",
]

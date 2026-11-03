# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native Torch implementations of the EasyMagpie spectral codec."""

from vllm_omni.model_executor.models.easymagpie.codec.config import EasyMagpieCodecConfig
from vllm_omni.model_executor.models.easymagpie.codec.encoder import (
    EasyMagpieCodecEncoder,
    EasyMagpieCodecEncoderOutput,
)
from vllm_omni.model_executor.models.easymagpie.codec.encoder_config import EasyMagpieCodecEncoderConfig
from vllm_omni.model_executor.models.easymagpie.codec.reference_speaker_encoder import EasyMagpieReferenceSpeakerEncoder

__all__ = [
    "EasyMagpieCodecConfig",
    "EasyMagpieCodecEncoder",
    "EasyMagpieCodecEncoderConfig",
    "EasyMagpieCodecEncoderOutput",
    "EasyMagpieReferenceSpeakerEncoder",
]

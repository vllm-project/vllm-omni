# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Index-AniSora model support for vLLM-Omni.

Supports two architectures:
- V1.0 (5B): CogVideoX-based - Uses AniSoraI2VCogVideoXPipeline
- V2/V3 (14B): Wan2.1-based - Uses AniSoraV2I2VPipeline

V1.0 Models:
- Disty0/Index-anisora-5B-diffusers

V2/V3 Models (hybrid loading with Wan2.1):
- aardsoul-music/Wan2.1-Anisora-14B
- ikusa/anisorav2
- IndexTeam/Index-anisora (V3.1, V3.2, anymask)
"""

from .pipeline_anisora_i2v_cogvideox import (
    AniSoraI2VCogVideoXPipeline,
)
from .pipeline_anisora_v2_i2v import (
    AniSoraV2I2VPipeline,
)

__all__ = [
    "AniSoraI2VCogVideoXPipeline",
    "AniSoraV2I2VPipeline",
]

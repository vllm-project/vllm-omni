# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SkyReels V2 DiT backbone.

Diffusers implements SkyReels V2 as a Wan-family 3D transformer
(`SkyReelsV2Transformer3DModel` in `transformer_skyreels_v2.py`). For Omni T2V
we reuse the already-ported Wan DiT (`WanTransformer3DModel`), which matches the
checkpoint layout used by `Skywork/SkyReels-V2-T2V-*-Diffusers`.

Extra Diffusers-only SkyReels features (diffusion-forcing causal masks,
`inject_sample_info` FPS embeddings) are out of scope for this T2V port.
"""

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

# Public alias so callers / docs can refer to the SkyReels name.
SkyReelsV2Transformer3DModel = WanTransformer3DModel

__all__ = ["SkyReelsV2Transformer3DModel"]

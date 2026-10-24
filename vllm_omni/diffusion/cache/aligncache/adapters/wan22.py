# SPDX-License-Identifier: Apache-2.0
"""Native Wan2.2 T2V-A14B, I2V-A14B and TI2V-5B admission; other Wan2.2 tasks stay rejected."""

from .wan21 import validate_scheduler, validate_transformer

# Wan2.2 I2V concatenates a 20-channel VAE image condition; it has no CLIP image branch.
I2V_A14B = {"in_channels": 36}
TI2V_5B = {"in_channels": 48, "out_channels": 48, "num_attention_heads": 24, "ffn_dim": 14336}


def validate_pipeline(pipeline, profile="wan22_t2v_reference_v1"):
    from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import Wan22I2VPipeline
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

    if profile == "wan22_ti2v_reference_v1":
        if type(pipeline) is not Wan22Pipeline or not pipeline.expand_timesteps or getattr(pipeline, "is_dmd", False):
            raise ValueError("align_cache: Wan2.2 TI2V-5B requires the native expand-timesteps WanPipeline")
        if getattr(pipeline, "has_transformer_2", False) or pipeline.transformer_2 is not None:
            raise ValueError("align_cache: Wan2.2 TI2V-5B requires one native transformer")
        models, large, structure = (pipeline.transformer,), False, TI2V_5B
    else:
        if profile == "wan22_i2v_reference_v1":
            native, structure = Wan22I2VPipeline, I2V_A14B
            if getattr(pipeline, "image_encoder", None) is not None:
                raise ValueError("align_cache: Wan2.2 I2V-A14B must not use a CLIP image encoder")
        else:
            native, structure = Wan22Pipeline, None
        if type(pipeline) is not native or pipeline.expand_timesteps or getattr(pipeline, "is_dmd", False):
            raise ValueError(f"align_cache: {profile} requires its native A14B pipeline, not TI2V or subclasses")
        if pipeline.transformer is pipeline.transformer_2:
            raise ValueError("align_cache: Wan2.2 components must have distinct identities")
        models, large = (pipeline.transformer, pipeline.transformer_2), True
    for model in models:
        if type(model) is not WanTransformer3DModel:
            raise ValueError("align_cache: Wan2.2 requires native WanTransformer3DModel components")
        validate_transformer(model, large=large, overrides=structure)
    if type(pipeline.scheduler) is not FlowUniPCMultistepScheduler:
        raise ValueError("align_cache: Wan2.2 requires FlowUniPCMultistepScheduler")
    validate_scheduler(pipeline.scheduler)

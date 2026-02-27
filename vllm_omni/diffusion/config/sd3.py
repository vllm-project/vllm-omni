# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.config.base import BaseDiTConfig


class SD3Transformer2DModelConfig(BaseDiTConfig):
    """
    A pretrainedConfig for SD3 DiT models.

    Ref for default values: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_sd3.py#L120
    """  # noqa: E501

    # Expected _class_name in Diffusers
    _class_name = "SD3Transformer2DModel"

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: str | None = None,
        **kwargs,
    ):
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.caption_projection_dim = caption_projection_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.out_channels = out_channels
        self.pos_embed_max_size = pos_embed_max_size
        self.dual_attention_layers = dual_attention_layers
        self.qk_norm: str | None = qk_norm

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.config.base import BaseDiTConfig


class LongCatImageTransformer2DModelConfig(BaseDiTConfig):
    """
    A pretrainedConfig for LongCat DiT models.

    Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_longcat_image.py#L413

    However, the default values are actually not used for any LongCat model
    and seemed to be copied from Flux; as such, we currently use the LongCat
    Image model for ref for default values for num_layers & num_single_layers:
    https://huggingface.co/meituan-longcat/LongCat-Image/blob/main/transformer/config.json
    """  # noqa: E501

    # Expected _class_name in Diffusers
    _class_name = "LongCatImageTransformer2DModel"

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 10,
        num_single_layers: int = 20,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 3584,
        axes_dims_rope: list[int] = [16, 56, 56],
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.axes_dims_rope = axes_dims_rope

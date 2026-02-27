# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.config.base import BaseDiTConfig


class StableAudioDiTModelConfig(BaseDiTConfig):
    """
    A pretrainedConfig for Stable Audio DiT models.

    Ref for default values: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/stable_audio_transformer.py#L212
    """  # noqa: E501

    # Expected _class_name in Diffusers
    _class_name = "StableAudioDiTModel"

    def __init__(
        self,
        sample_size: int = 1024,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_key_value_attention_heads: int = 12,
        out_channels: int = 64,
        cross_attention_dim: int = 768,
        time_proj_dim: int = 256,
        global_states_input_dim: int = 1536,
        cross_attention_input_dim: int = 768,
        **kwargs,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_attention_heads = num_key_value_attention_heads
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim
        self.time_proj_dim = time_proj_dim
        self.global_states_input_dim = global_states_input_dim
        self.cross_attention_input_dim = cross_attention_input_dim

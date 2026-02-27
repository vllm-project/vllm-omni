# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.config.base import BaseDiTConfig


class WanTransformer3DModelConfig(BaseDiTConfig):
    """
    A pretrainedConfig for Wan DIT models.

    Ref for default values: https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/transformers/transformer_wan.py#L565
    """  # noqa: E501

    # Expected _class_name in Diffusers
    _class_name = "WanTransformer3DModel"

    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.image_dim = image_dim
        self.added_kv_proj_dim = added_kv_proj_dim
        self.rope_max_seq_len = rope_max_seq_len
        self.pos_embed_seq_len = pos_embed_seq_len

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Iterable
from typing import Any, Optional, Union, List


import torch
import torch.nn as nn
from vllm.logger import init_logger
from diffusers.models.embeddings import Timesteps, TimestepEmbedding,FluxPosEmbed
from diffusers.models.transformers.transformer_flux import \
    FluxTransformerBlock, FluxSingleTransformerBlock, \
    AdaLayerNormContinuous, Transformer2DModelOutput

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm_omni.diffusion.attention.layer import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

class TimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        return timesteps_emb


class FluxAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        pre_only: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads if out_dim is None else out_dim // head_dim
        self.head_dim = head_dim
        self.inner_dim = out_dim if out_dim is not None else head_dim * num_heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.pre_only = pre_only

        # Fused QKV projection using vLLM's optimized layer
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=self.num_heads,
            bias=bias,
            disable_tp=True,
        )

        if added_kv_proj_dim is not None:
            self.to_added_qkv = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=head_dim,
                total_num_heads=self.num_heads,
                bias=added_proj_bias,
                disable_tp=True,
            )
        
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(head_dim, eps=eps)
            self.norm_added_k = RMSNorm(head_dim, eps=eps)
        
        if not pre_only:
            self.to_out = nn.ModuleList([
                ReplicatedLinear(
                    self.inner_dim, 
                    out_dim or dim,
                    bias=out_bias
                    ), nn.Dropout(dropout)
            ])
        else:
            self.to_out = None
        
        if added_kv_proj_dim is not None:
            self.to_add_out = ReplicatedLinear(self.inner_dim, dim, bias=out_bias)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=head_dim,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        ip_hidden_states=None,
        ip_adapter_masks=None,
    ):
        qkv, _ = self.to_qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.unflatten(-1, (self.num_heads, -1))
        k = k.unflatten(-1, (self.num_heads, -1))
        v = v.unflatten(-1, (self.num_heads, -1))

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            add_qkv, _ = self.to_added_qkv(encoder_hidden_states)
            aq, ak, av = add_qkv.chunk(3, dim=-1)
            aq = self.norm_added_q(aq)
            ak = self.norm_added_k(ak)

            aq = aq.unflatten(-1, (self.num_heads, -1))
            ak = ak.unflatten(-1, (self.num_heads, -1))
            av = av.unflatten(-1, (self.num_heads, -1))

            q = torch.cat([aq, q], dim=1)
            k = torch.cat([ak, k], dim=1)
            v = torch.cat([av, v], dim=1)

        # RoPE
        if image_rotary_emb is not None:
            q, k = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1), apply_rotary_emb(
                k, image_rotary_emb, sequence_dim=1
            )

        attn_metadata = AttentionMetadata(attention_mask) if attention_mask is not None else None
        attn_out = self.attn(q, k, v, attn_metadata=attn_metadata)
        attn_out = attn_out.flatten(2, 3).to(hidden_states.dtype)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            ctx_len = encoder_hidden_states.shape[1]
            ctx_out, img_out = attn_out.split_with_sizes([ctx_len, attn_out.shape[1] - ctx_len], dim=1)
            img_out, _ = self.to_out[0](img_out)
            if len(self.to_out) > 1:
                img_out = self.to_out[1](img_out)
            ctx_out, _ = self.to_add_out(ctx_out)
            return img_out, ctx_out  # 兼容原 Flux 返回 (img, ctx)
        else:
            if self.to_out is None:
                return attn_out
            attn_out, _ = self.to_out[0](attn_out)
            if len(self.to_out) > 1:
                attn_out = self.to_out[1](attn_out)
            return attn_out


class LongCatImageTransformer2DModel(nn.Module):
    """
    The Transformer model introduced in Flux.
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 3584,
        axes_dims_rope: List[int] = [16, 56, 56],
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pooled_projection_dim = pooled_projection_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.time_embed = TimestepEmbeddings(embedding_dim=self.inner_dim)

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.initialize_weights()

        self.use_checkpoint = [True] * num_layers
        self.use_single_checkpoint = [True] * num_single_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The  forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = self.time_embed( timestep, hidden_states.dtype )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_checkpoint[index_block]:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_single_checkpoint[index_block]:
                encoder_hidden_states,hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.context_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)
            nn.init.constant_(block.norm1_context.linear.weight, 0)
            nn.init.constant_(block.norm1_context.linear.bias, 0)

        for block in self.single_transformer_blocks:
            nn.init.constant_(block.norm.linear.weight, 0)
            nn.init.constant_(block.norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
        ]

        params_dict = dict(self.named_parameters())

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
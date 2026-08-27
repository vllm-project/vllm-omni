# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache
from math import prod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO replace this with vLLM implementation
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.quantization.component_config import safe_quant_config

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rotary_emb_torch

logger = init_logger(__name__)


def _normalize_qwen_image_weight_name(name: str) -> str:
    name = name.removeprefix("transformer.")
    if ".to_out.0." in name:
        name = name.replace(".to_out.0.", ".to_out.")
    return name


def _resolve_qwen_image_lookup_name(
    name: str,
    stacked_params_mapping: list[tuple[str, str, str]],
) -> tuple[str, str | None]:
    lookup_name = _normalize_qwen_image_weight_name(name)
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if weight_name not in lookup_name or param_name in lookup_name:
            continue
        return lookup_name.replace(weight_name, param_name), shard_id
    return lookup_name, None


class ImageRopePrepare(nn.Module):
    """Prepares image hidden_states and RoPE embeddings for sequence parallel.

    This module encapsulates the input linear projection and RoPE computation.
    Similar to Z-Image's UnifiedPrepare, this creates a module boundary where
    _sp_plan can shard outputs via split_output=True.

    The key insight is that hidden_states and vid_freqs must be sharded together
    to maintain dimension alignment for RoPE computation in attention layers.

    Note: Our _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism).
    """

    def __init__(self, img_in: nn.Module, pos_embed: nn.Module):
        super().__init__()
        self.img_in = img_in
        self.pos_embed = pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        img_shapes: list[tuple[int, int, int]],
        txt_seq_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare hidden_states and RoPE for SP.

        Args:
            hidden_states: [batch, img_seq_len, channels]
            img_shapes: List of (frame, height, width) tuples
            txt_seq_lens: List of text sequence lengths

        Returns:
            hidden_states: Processed hidden states [batch, img_seq_len, dim]
            vid_freqs: Image RoPE frequencies [img_seq_len, rope_dim]
            txt_freqs: Text RoPE frequencies [txt_seq_len, rope_dim]

        Note: _sp_plan will shard hidden_states and vid_freqs via split_output=True
              txt_freqs is kept replicated for dual-stream attention
        """
        # Apply input projection
        hidden_states = self.img_in(hidden_states)

        # Compute RoPE embeddings
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        vid_freqs, txt_freqs = image_rotary_emb

        return hidden_states, vid_freqs, txt_freqs


class ModulateIndexPrepare(nn.Module):
    """Prepares modulate_index for sequence parallel when zero_cond_t is enabled.

    This module encapsulates the creation of modulate_index tensor, which is used
    to select different conditioning parameters (shift/scale/gate) for different
    token positions in image editing tasks.

    Similar to Z-Image's UnifiedPrepare and ImageRopePrepare, this creates a module
    boundary where _sp_plan can shard the output via split_output=True.

    The modulate_index must be sharded along the sequence dimension to match the
    sharded hidden_states in SP mode.

    Note: Our _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism).
    """

    def __init__(self, zero_cond_t: bool = False):
        super().__init__()
        self.zero_cond_t = zero_cond_t

    def forward(
        self,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare timestep and modulate_index for SP.

        Args:
            timestep: Timestep tensor [batch]
            img_shapes: List of image shape tuples per batch item.
                Each item is a list of (frame, height, width) tuples.
                For edit models: [[source_shape], [target_shape1, target_shape2, ...]]

        Returns:
            timestep: Doubled timestep if zero_cond_t, else original [batch] or [2*batch]
            modulate_index: Token condition index [batch, seq_len] if zero_cond_t, else None
                - index=0: source image tokens (use normal timestep conditioning)
                - index=1: target image tokens (use zero timestep conditioning)

        Note: _sp_plan will shard modulate_index via split_output=True when SP is enabled.
              The modulate_index sequence dimension must match hidden_states after sharding.
        """
        if self.zero_cond_t:
            # Double the timestep: [timestep, timestep * 0]
            # This creates two sets of conditioning parameters in AdaLayerNorm
            timestep = torch.cat([timestep, timestep * 0], dim=0)

            # Create modulate_index to select conditioning per token position
            # - First image (sample[0]): source image, use index=0 (normal timestep)
            # - Remaining images (sample[1:]): target images, use index=1 (zero timestep)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
            return timestep, modulate_index

        return timestep, None


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim,
        use_additional_t_cond: bool = False,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        # Time embedding MLP is kept full precision (quant_config=None) —
        # small layers that feed per-block modulation; precision-sensitive
        # (see #2728).
        self.timestep_embedder.linear_1 = ReplicatedLinear(
            256,
            embedding_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="timestep_embedder.linear_1",
        )
        self.timestep_embedder.linear_2 = ReplicatedLinear(
            embedding_dim,
            embedding_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="timestep_embedder.linear_2",
        )
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return conditioning


class QwenEmbedLayer3DRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx != layer_num:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            else:
                ### For the condition image, we set the layer index to -1
                video_freq = self._compute_condition_freqs(frame, height, width)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @lru_cache(maxsize=16)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()

    @lru_cache(maxsize=16)
    def _compute_condition_freqs(self, frame, height, width):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000):
        """
        Args:
            index (`torch.Tensor`): [0, 1, 2, 3] 1D Tensor representing the position index of the token
            dim (`int`): Dimension for the rope parameters
            theta (`int`): Theta parameter for rope
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @lru_cache(maxsize=16)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class ColumnParallelApproxGELU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        approximate: str,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj" if prefix else "proj",
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        activation_fn: str = "gelu-approximate",
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert activation_fn == "gelu-approximate", "Only gelu-approximate is supported."

        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(
                dim,
                inner_dim,
                approximate="tanh",
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.net.0",
            ),
            nn.Identity(),  # placeholder for weight loading
            RowParallelLinear(
                inner_dim,
                dim_out,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.net.2",
            ),
        ]

        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class QwenImageCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,  # query_dim
        num_heads: int,
        head_dim: int,
        added_kv_proj_dim: int,
        window_size: tuple[int, int] = (-1, -1),
        out_bias: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
        pre_only: bool = False,
        context_pre_only: bool = False,
        out_dim: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )
        self.query_num_heads = self.to_qkv.num_heads
        self.kv_num_heads = self.to_qkv.num_kv_heads

        self.norm_q = nn.RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()

        self.inner_dim = out_dim if out_dim is not None else head_dim * self.total_num_heads

        assert context_pre_only is not None
        self.add_kv_proj = QKVParallelLinear(
            hidden_size=added_kv_proj_dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.add_kv_proj",
        )
        self.add_query_num_heads = self.add_kv_proj.num_heads
        self.add_kv_num_heads = self.add_kv_proj.num_kv_heads

        assert not context_pre_only
        self.to_add_out = RowParallelLinear(
            self.inner_dim,
            self.dim,
            bias=out_bias,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_add_out",
        )

        assert not pre_only
        self.to_out = RowParallelLinear(
            self.inner_dim,
            self.dim,
            bias=out_bias,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out.0",
        )

        self.norm_added_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=eps)

        self.attn = Attention(
            num_heads=self.query_num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.kv_num_heads,
        )
        self.rope = RotaryEmbedding(is_neox_style=False)

        try:
            config = get_forward_context().omni_diffusion_config
            self.parallel_config = config.parallel_config
        except Exception:
            self.parallel_config = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        vid_freqs: torch.Tensor,
        txt_freqs: torch.Tensor,
        hidden_states_mask: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_qkv, _ = self.to_qkv(hidden_states)
        q_size = self.query_num_heads * self.head_dim
        kv_size = self.kv_num_heads * self.head_dim
        img_query, img_key, img_value = img_qkv.split([q_size, kv_size, kv_size], dim=-1)

        txt_qkv, _ = self.add_kv_proj(encoder_hidden_states)
        add_q_size = self.add_query_num_heads * self.head_dim
        add_kv_size = self.add_kv_num_heads * self.head_dim
        txt_query, txt_key, txt_value = txt_qkv.split([add_q_size, add_kv_size, add_kv_size], dim=-1)

        img_query = img_query.unflatten(-1, (self.query_num_heads, self.head_dim))
        img_key = img_key.unflatten(-1, (self.kv_num_heads, self.head_dim))
        img_value = img_value.unflatten(-1, (self.kv_num_heads, self.head_dim))

        txt_query = txt_query.unflatten(-1, (self.add_query_num_heads, self.head_dim))
        txt_key = txt_key.unflatten(-1, (self.add_kv_num_heads, self.head_dim))
        txt_value = txt_value.unflatten(-1, (self.add_kv_num_heads, self.head_dim))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        img_cos = torch.real(vid_freqs).to(img_query.dtype)
        img_sin = torch.imag(vid_freqs).to(img_query.dtype)
        txt_cos = torch.real(txt_freqs).to(txt_query.dtype)
        txt_sin = torch.imag(txt_freqs).to(txt_query.dtype)

        img_query = self.rope(img_query, img_cos, img_sin)
        img_key = self.rope(img_key, img_cos, img_sin)
        txt_query = self.rope(txt_query, txt_cos, txt_sin)
        txt_key = self.rope(txt_key, txt_cos, txt_sin)

        seq_len_txt = encoder_hidden_states.shape[1]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        if (
            self.parallel_config is not None
            and self.parallel_config.sequence_parallel_size > 1
            and not get_forward_context().split_text_embed_in_sp
        ):
            attn_metadata = AttentionMetadata(
                joint_query=txt_query,
                joint_key=txt_key,
                joint_value=txt_value,
                joint_strategy="front",
            )
            if hidden_states_mask is not None:
                attn_metadata.attn_mask = hidden_states_mask
            if encoder_hidden_states_mask is not None:
                attn_metadata.joint_attn_mask = encoder_hidden_states_mask

            joint_hidden_states = self.attn(img_query, img_key, img_value, attn_metadata)
        else:
            attn_metadata = None
            if hidden_states_mask is not None or encoder_hidden_states_mask is not None:
                mask_list: list[torch.Tensor] = []
                if encoder_hidden_states_mask is not None:
                    mask_list.append(encoder_hidden_states_mask)
                else:
                    mask_list.append(
                        torch.ones(
                            encoder_hidden_states.shape[:2],
                            dtype=torch.bool,
                            device=encoder_hidden_states.device,
                        )
                    )
                if hidden_states_mask is not None:
                    mask_list.append(hidden_states_mask)
                else:
                    mask_list.append(
                        torch.ones(
                            hidden_states.shape[:2],
                            dtype=torch.bool,
                            device=hidden_states.device,
                        )
                    )
                joint_mask = torch.cat(mask_list, dim=1) if len(mask_list) > 1 else mask_list[0]
                attn_metadata = AttentionMetadata(attn_mask=joint_mask)

            joint_hidden_states = self.attn(joint_query, joint_key, joint_value, attn_metadata)

        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
        txt_attn_output = joint_hidden_states[:, :seq_len_txt, :]
        img_attn_output = joint_hidden_states[:, seq_len_txt:, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    def forward_mixfusion(
        self,
        image_chunks: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_freq_chunks: torch.Tensor,
        text_freqs: torch.Tensor,
        chunk_to_request: torch.Tensor,
        request_chunk_ranges: list[tuple[int, int]],
        encoder_hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen joint attention with chunked image tokens and request-level text.

        Text/prompt tokens are projected once per request. Image tokens stay in
        MixFusion chunks for token-wise projections, then each request is
        recovered to ``text + full image`` immediately before attention so the
        attention semantics match independent dense execution.
        """
        if int(chunk_to_request.numel()) != int(image_chunks.shape[0]):
            raise ValueError("chunk_to_request must have one entry per image chunk.")

        img_qkv, _ = self.to_qkv(image_chunks)
        q_size = self.query_num_heads * self.head_dim
        kv_size = self.kv_num_heads * self.head_dim
        img_query, img_key, img_value = img_qkv.split([q_size, kv_size, kv_size], dim=-1)

        txt_qkv, _ = self.add_kv_proj(encoder_hidden_states)
        add_q_size = self.add_query_num_heads * self.head_dim
        add_kv_size = self.add_kv_num_heads * self.head_dim
        txt_query, txt_key, txt_value = txt_qkv.split([add_q_size, add_kv_size, add_kv_size], dim=-1)

        img_query = img_query.unflatten(-1, (self.query_num_heads, self.head_dim))
        img_key = img_key.unflatten(-1, (self.kv_num_heads, self.head_dim))
        img_value = img_value.unflatten(-1, (self.kv_num_heads, self.head_dim))

        txt_query = txt_query.unflatten(-1, (self.add_query_num_heads, self.head_dim))
        txt_key = txt_key.unflatten(-1, (self.add_kv_num_heads, self.head_dim))
        txt_value = txt_value.unflatten(-1, (self.add_kv_num_heads, self.head_dim))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        img_cos = image_freq_chunks.real.to(img_query.dtype)
        img_sin = image_freq_chunks.imag.to(img_query.dtype)
        txt_cos = text_freqs.real.to(txt_query.dtype)
        txt_sin = text_freqs.imag.to(txt_query.dtype)

        # The fused RotaryEmbedding path drops 3D cos/sin to the first batch
        # item. MixFusion needs per-chunk/per-request positions, so use the
        # native implementation that preserves the leading dimension.
        img_query = apply_rotary_emb_torch(img_query, img_cos, img_sin, interleaved=self.rope.interleaved)
        img_key = apply_rotary_emb_torch(img_key, img_cos, img_sin, interleaved=self.rope.interleaved)
        txt_query = apply_rotary_emb_torch(txt_query, txt_cos, txt_sin, interleaved=self.rope.interleaved)
        txt_key = apply_rotary_emb_torch(txt_key, txt_cos, txt_sin, interleaved=self.rope.interleaved)

        seq_len_txt = encoder_hidden_states.shape[1]
        chunk_size = image_chunks.shape[1]
        num_requests = encoder_hidden_states.shape[0]
        text_outputs: list[torch.Tensor | None] = [None] * num_requests
        image_outputs: list[torch.Tensor | None] = [None] * image_chunks.shape[0]

        # Per-request real text length. Prompt padding is dropped from the flat
        # packed joint attention so padded text tokens never enter the kernel
        # (padded_tokens == 0), matching the mask-unpad path used by the dense
        # joint forward.
        if encoder_hidden_states_mask is not None:
            real_txt_lens = encoder_hidden_states_mask.sum(dim=1).tolist()
        else:
            real_txt_lens = [seq_len_txt] * num_requests

        flat_queries: list[torch.Tensor] = []
        flat_keys: list[torch.Tensor] = []
        flat_values: list[torch.Tensor] = []
        cu_seqlens = [0]
        max_seq_len = 0
        for req_idx, (chunk_start, chunk_end) in enumerate(request_chunk_ranges):
            txt_len = int(real_txt_lens[req_idx])
            image_seq_len = (chunk_end - chunk_start) * chunk_size
            req_img_query = img_query[chunk_start:chunk_end].reshape(-1, self.query_num_heads, self.head_dim)
            req_img_key = img_key[chunk_start:chunk_end].reshape(-1, self.kv_num_heads, self.head_dim)
            req_img_value = img_value[chunk_start:chunk_end].reshape(-1, self.kv_num_heads, self.head_dim)
            flat_queries.append(torch.cat([txt_query[req_idx, :txt_len], req_img_query], dim=0))
            flat_keys.append(torch.cat([txt_key[req_idx, :txt_len], req_img_key], dim=0))
            flat_values.append(torch.cat([txt_value[req_idx, :txt_len], req_img_value], dim=0))
            req_len = txt_len + image_seq_len
            cu_seqlens.append(cu_seqlens[-1] + req_len)
            max_seq_len = max(max_seq_len, req_len)

        joint_query = torch.cat(flat_queries, dim=0)
        joint_key = torch.cat(flat_keys, dim=0)
        joint_value = torch.cat(flat_values, dim=0)
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=joint_query.device)
        attn_metadata = AttentionMetadata(
            is_varlen=True,
            q_cu_seqlens=cu_seqlens_tensor,
            kv_cu_seqlens=cu_seqlens_tensor,
            max_q_len=max_seq_len,
            max_kv_len=max_seq_len,
            padded_tokens=0,
        )
        joint_hidden_states = self.attn(joint_query, joint_key, joint_value, attn_metadata)

        for req_idx, (chunk_start, chunk_end) in enumerate(request_chunk_ranges):
            txt_len = int(real_txt_lens[req_idx])
            req_out = joint_hidden_states[cu_seqlens[req_idx] : cu_seqlens[req_idx + 1]]
            text_outputs[req_idx] = req_out[:txt_len].unsqueeze(0)
            image_output = req_out[txt_len:].reshape(
                chunk_end - chunk_start,
                chunk_size,
                self.query_num_heads,
                self.head_dim,
            )
            for local_chunk_idx, chunk_idx in enumerate(range(chunk_start, chunk_end)):
                image_outputs[chunk_idx] = image_output[local_chunk_idx : local_chunk_idx + 1]

        # Re-pad per-request text outputs to the padded text length so the
        # residual add and downstream text projections stay shape-compatible.
        # Padded positions get zero output, identical to the mask-unpad path.
        padded_text_outputs: list[torch.Tensor] = []
        for req_idx in range(num_requests):
            out = text_outputs[req_idx]
            if out is None:
                raise ValueError(f"Missing text attention output for request {req_idx}.")
            if out.shape[1] < seq_len_txt:
                out = torch.cat(
                    [out, out.new_zeros((1, seq_len_txt - out.shape[1], out.shape[2], out.shape[3]))],
                    dim=1,
                )
            padded_text_outputs.append(out)

        txt_attn_output = torch.cat(padded_text_outputs, dim=0)
        img_attn_output = torch.cat([out for out in image_outputs if out is not None], dim=0)

        txt_attn_output = txt_attn_output.flatten(2, 3).to(txt_query.dtype)
        img_attn_output = img_attn_output.flatten(2, 3).to(img_query.dtype)

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        txt_mod_quant_config = safe_quant_config(quant_config)

        # Image processing modules.
        # The re-quantized W4A16 checkpoint keeps img_mod.1 in full precision.
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(
                dim,
                6 * dim,
                bias=True,
                return_bias=False,
                quant_config=None,
                prefix=f"{prefix}.img_mod.1",
            ),
        )
        self.img_norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImageCrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            added_kv_proj_dim=dim,
            context_pre_only=False,
            head_dim=attention_head_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.img_norm2 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.img_mlp",
        )

        # Text processing modules.
        # AutoRound keeps txt_mod.1 quantized inside transformer_blocks.
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(
                dim,
                6 * dim,
                bias=True,
                return_bias=False,
                quant_config=txt_mod_quant_config,
                prefix=f"{prefix}.txt_mod.1",
            ),
        )
        self.txt_norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.txt_mlp",
        )

        self.zero_cond_t = zero_cond_t

    def _modulate(self, mod_params, index=None):
        """Apply modulation to input tensor"""
        # shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return scale_result, shift_result, gate_result

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        joint_attention_kwargs: dict[str, Any] | None = None,
        modulate_index: list[int] | None = None,
        hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]

        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]

        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_scale1, img_shift1, img_gate1 = self._modulate(img_mod1, modulate_index)
        img_modulated = self.img_norm1(hidden_states, img_scale1, img_shift1)

        # Process text stream - norm1 + modulation
        txt_scale1, txt_shift1, txt_gate1 = self._modulate(txt_mod1)
        txt_modulated = self.txt_norm1(encoder_hidden_states, txt_scale1, txt_shift1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            vid_freqs=image_rotary_emb[0],
            txt_freqs=image_rotary_emb[1],
            hidden_states_mask=hidden_states_mask,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_scale2, img_shift2, img_gate2 = self._modulate(img_mod2, modulate_index)
        img_modulated2 = self.img_norm2(hidden_states, img_scale2, img_shift2)

        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_scale2, txt_shift2, txt_gate2 = self._modulate(txt_mod2)
        txt_modulated2 = self.txt_norm2(encoder_hidden_states, txt_scale2, txt_shift2)

        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_mixfusion(
        self,
        image_chunks: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_freq_chunks: torch.Tensor,
        text_freqs: torch.Tensor,
        chunk_to_request: torch.Tensor,
        request_chunk_ranges: list[tuple[int, int]],
        encoder_hidden_states_mask: torch.Tensor | None,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)

        img_mod_params = img_mod_params.index_select(0, chunk_to_request.to(img_mod_params.device))
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_scale1, img_shift1, img_gate1 = self._modulate(img_mod1)
        img_modulated = self.img_norm1(image_chunks, img_scale1, img_shift1)

        txt_scale1, txt_shift1, txt_gate1 = self._modulate(txt_mod1)
        txt_modulated = self.txt_norm1(encoder_hidden_states, txt_scale1, txt_shift1)

        img_attn_output, txt_attn_output = self.attn.forward_mixfusion(
            image_chunks=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_freq_chunks=image_freq_chunks,
            text_freqs=text_freqs,
            chunk_to_request=chunk_to_request,
            request_chunk_ranges=request_chunk_ranges,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )

        image_chunks = image_chunks + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_scale2, img_shift2, img_gate2 = self._modulate(img_mod2)
        img_modulated2 = self.img_norm2(image_chunks, img_scale2, img_shift2)
        image_chunks = image_chunks + img_gate2 * self.img_mlp(img_modulated2)

        txt_scale2, txt_shift2, txt_gate2 = self._modulate(txt_mod2)
        txt_modulated2 = self.txt_norm2(encoder_hidden_states, txt_scale2, txt_shift2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if image_chunks.dtype == torch.float16:
            image_chunks = image_chunks.clip(-65504, 65504)

        return encoder_hidden_states, image_chunks


# Note: inheriting from CachedTransformer only when we support caching
class QwenImageTransformer2DModel(CachedTransformer):
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    # the small and frequently-repeated block(s) of a model
    # -- typically a transformer layer
    # used for torch compile optimizations
    _repeated_blocks = ["QwenImageTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    _hsdp_shard_conditions = [is_transformer_block_module]

    # Sequence Parallelism plan (following diffusers' _cp_plan pattern)
    # Similar to Z-Image's UnifiedPrepare, we use ImageRopePrepare to create
    # a module boundary where _sp_plan can shard hidden_states and vid_freqs together.
    #
    # Key insight: hidden_states and vid_freqs MUST be sharded together to maintain
    # dimension alignment for RoPE computation in attention layers.
    #
    # auto_pad=True enables automatic padding when sequence length is not divisible
    # by SP world size. This creates an attention mask stored in ForwardContext
    # that attention layers can use to ignore padding positions.
    #
    # Note: _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism)
    _sp_plan = {
        # Shard ImageRopePrepare outputs (hidden_states and vid_freqs must be sharded together)
        "image_rope_prepare": {
            # hidden_states: auto_pad=True for variable sequence length support
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            # vid_freqs: auto_pad=True to match hidden_states padding
            1: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
            # txt_freqs (index 2) is NOT sharded - kept replicated for dual-stream attention
        },
        # Shard ModulateIndexPrepare output (modulate_index must be sharded to match hidden_states)
        # This is only active when zero_cond_t=True (image editing models)
        # Output index 1 is modulate_index [batch, seq_len], needs sharding along dim=1
        "modulate_index_prepare": {
            1: SequenceParallelInput(split_dim=1, expected_dims=2, split_output=True, auto_pad=True),
        },
        # Gather output at proj_out
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.parallel_config = od_config.parallel_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.guidance_embeds = guidance_embeds
        self.quant_config = quant_config

        if not use_layer3d_rope:
            self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        else:
            self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            use_additional_t_cond=use_additional_t_cond,
            quant_config=quant_config,
        )

        self.txt_norm = nn.RMSNorm(joint_attention_dim, eps=1e-6)

        # Entry projections (image/text) are kept full precision —
        # small sensitive layers at the network boundary (see #2728).
        self.img_in = ReplicatedLinear(
            in_channels,
            self.inner_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="img_in",
        )
        self.txt_in = ReplicatedLinear(
            joint_attention_dim,
            self.inner_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="txt_in",
        )

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                    quant_config=quant_config,
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        # Final modulation and output projection are kept full precision —
        # they produce the output latent and are precision-sensitive
        # (see #2728).
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.norm_out.linear = ReplicatedLinear(
            self.inner_dim,
            2 * self.inner_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="norm_out.linear",
        )
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix="proj_out",
        )

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t

        # ImageRopePrepare module for _sp_plan to shard hidden_states and vid_freqs together
        # This ensures RoPE dimensions align with hidden_states after sharding
        self.image_rope_prepare = ImageRopePrepare(self.img_in, self.pos_embed)

        # ModulateIndexPrepare module for _sp_plan to shard modulate_index
        # This ensures modulate_index dimensions align with hidden_states after sharding
        # Only active when zero_cond_t=True (image editing models)
        self.modulate_index_prepare = ModulateIndexPrepare(zero_cond_t=zero_cond_t)

    @staticmethod
    def _mixfusion_chunk_size(hidden_states: list[torch.Tensor]) -> int:
        chunk_size = int(hidden_states[0].shape[1])
        for sample in hidden_states[1:]:
            chunk_size = math.gcd(chunk_size, int(sample.shape[1]))
        return chunk_size

    def _forward_mixfusion(
        self,
        hidden_states: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None,
        timestep: torch.LongTensor,
        img_shapes: list[tuple[int, int, int]] | None,
        txt_seq_lens: list[int] | None,
        guidance: torch.Tensor | None,
        additional_t_cond=None,
    ) -> list[torch.Tensor]:
        if self.zero_cond_t:
            raise ValueError("Qwen MixFusion does not support zero_cond_t/editing path yet.")
        if self.parallel_config is not None and self.parallel_config.sequence_parallel_size > 1:
            raise ValueError("Qwen MixFusion requires sequence parallel disabled.")
        if not hidden_states:
            raise ValueError("Qwen MixFusion requires at least one hidden-state tensor.")
        if img_shapes is None or txt_seq_lens is None:
            raise ValueError("Qwen MixFusion requires img_shapes and txt_seq_lens.")

        chunk_size = self._mixfusion_chunk_size(hidden_states)
        image_chunks: list[torch.Tensor] = []
        image_freq_chunks: list[torch.Tensor] = []
        chunk_to_request: list[int] = []
        request_chunk_ranges: list[tuple[int, int]] = []
        seq_len_txt = int(encoder_hidden_states.shape[1])
        text_freqs: list[torch.Tensor] = []

        for req_idx, sample in enumerate(hidden_states):
            seq_len = int(sample.shape[1])
            if seq_len % chunk_size != 0:
                raise ValueError(f"Qwen MixFusion seq_len={seq_len} is not divisible by chunk_size={chunk_size}.")
            chunk_start = len(image_chunks)
            image_chunks.extend(sample.split(chunk_size, dim=1))
            chunk_end = len(image_chunks)
            request_chunk_ranges.append((chunk_start, chunk_end))
            chunk_to_request.extend([req_idx] * (chunk_end - chunk_start))

            req_vid_freqs, req_txt_freqs = self.pos_embed(img_shapes[req_idx], [seq_len_txt], device=sample.device)
            image_freq_chunks.extend(req_vid_freqs.split(chunk_size, dim=0))
            text_freqs.append(req_txt_freqs[:seq_len_txt])

        image_chunks_tensor = torch.cat(image_chunks, dim=0)
        image_chunks_tensor = self.img_in(image_chunks_tensor)
        image_freq_chunks_tensor = torch.stack(image_freq_chunks, dim=0)
        text_freqs_tensor = torch.stack(text_freqs, dim=0)
        chunk_to_request_tensor = torch.tensor(chunk_to_request, dtype=torch.long, device=image_chunks_tensor.device)

        timestep = timestep.to(device=image_chunks_tensor.device, dtype=image_chunks_tensor.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(image_chunks_tensor.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, image_chunks_tensor, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, image_chunks_tensor, additional_t_cond)
        )

        if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.all():
            encoder_hidden_states_mask = None

        for block in self.transformer_blocks:
            encoder_hidden_states, image_chunks_tensor = block.forward_mixfusion(
                image_chunks=image_chunks_tensor,
                encoder_hidden_states=encoder_hidden_states,
                image_freq_chunks=image_freq_chunks_tensor,
                text_freqs=text_freqs_tensor,
                chunk_to_request=chunk_to_request_tensor,
                request_chunk_ranges=request_chunk_ranges,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
            )

        chunk_temb = temb.index_select(0, chunk_to_request_tensor)
        image_chunks_tensor = self.norm_out(image_chunks_tensor, chunk_temb)
        image_chunks_tensor = self.proj_out(image_chunks_tensor)

        outputs: list[torch.Tensor] = []
        for chunk_start, chunk_end in request_chunk_ranges:
            outputs.append(image_chunks_tensor[chunk_start:chunk_end].reshape(1, -1, image_chunks_tensor.shape[-1]))
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: dict[str, Any] | None = None,
        additional_t_cond=None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput | list[torch.Tensor]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if isinstance(hidden_states, list):
            return self._forward_mixfusion(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                guidance=guidance,
                additional_t_cond=additional_t_cond,
            )

        # if attention_kwargs is not None:
        #     attention_kwargs = attention_kwargs.copy()
        #     lora_scale = attention_kwargs.pop("scale", 1.0)
        # else:
        #     lora_scale = 1.0

        # Set split_text_embed_in_sp = False for dual-stream attention
        # QwenImage uses *dual-stream* (text + image) and runs a *joint attention*.
        # Text embeddings must be replicated across SP ranks for correctness.
        if self.parallel_config.sequence_parallel_size > 1:
            get_forward_context().split_text_embed_in_sp = False

        # Prepare hidden_states and RoPE via ImageRopePrepare module
        # _sp_plan will shard hidden_states and vid_freqs together via split_output=True
        # txt_freqs is kept replicated for dual-stream attention
        hidden_states, vid_freqs, txt_freqs = self.image_rope_prepare(hidden_states, img_shapes, txt_seq_lens)
        image_rotary_emb = (vid_freqs, txt_freqs)

        # Ensure timestep tensor is on the same device and dtype as hidden_states
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # Prepare timestep and modulate_index via ModulateIndexPrepare module
        # _sp_plan will shard modulate_index via split_output=True (when zero_cond_t=True)
        # This ensures modulate_index sequence dimension matches sharded hidden_states
        timestep, modulate_index = self.modulate_index_prepare(timestep, img_shapes)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        # Check for SP auto_pad: create attention mask dynamically if padding was applied
        # In Ulysses mode, attention is computed on the FULL sequence (after All-to-All)
        hidden_states_mask = None  # default
        ctx = get_forward_context()
        if (
            self.parallel_config is not None
            and self.parallel_config.sequence_parallel_size > 1
            and self.parallel_config.mask_sp_padding
            and ctx.sp_original_seq_len is not None
            and ctx.sp_padding_size > 0
        ):
            # Create mask for the full (padded) sequence
            # valid positions = True, padding positions = False
            batch_size = hidden_states.shape[0]
            padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
            hidden_states_mask = torch.ones(
                batch_size,
                padded_seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            hidden_states_mask[:, ctx.sp_original_seq_len :] = False
            if hidden_states_mask.all():
                hidden_states_mask = None
        elif (
            self.parallel_config is not None
            and self.parallel_config.sequence_parallel_size > 1
            and not self.parallel_config.mask_sp_padding
            and ctx.sp_original_seq_len is not None
            and ctx.sp_padding_size > 0
        ):
            logger.warning_once(
                "SP auto-padding applied %d token(s) (seq_len=%d, ulysses_degree=%d). "
                "Padding tokens are not masked from attention (mask_sp_padding=False), "
                "which avoids the varlen attention path but may produce minor numerical differences. "
                "Set parallel_config.mask_sp_padding=True to restore strict masking.",
                ctx.sp_padding_size,
                ctx.sp_original_seq_len,
                self.parallel_config.sequence_parallel_size,
            )

        if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.all():
            encoder_hidden_states_mask = None

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
                hidden_states_mask=hidden_states_mask,
            )

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # Note: SP gather is handled automatically by _sp_plan's SequenceParallelGatherHook
        # on proj_out output. No manual all_gather needed here.

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # cross-attn
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]
        # Expose packed shard mappings for LoRA handling of fused projections.
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())

        # we need to load the buffers for beta and eps (XIELU)
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name.removeprefix("transformer.")
            lookup_name, shard_id = _resolve_qwen_image_lookup_name(
                original_name,
                stacked_params_mapping,
            )

            if lookup_name.endswith(".bias") and lookup_name not in params_dict:
                continue

            param = params_dict.get(lookup_name)
            if param is None:
                logger.debug("Skipping unexpected Qwen-Image transformer weight %s", original_name)
                continue

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params

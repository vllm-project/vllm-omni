# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Iterable
from math import prod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO replace this with vLLM implementation
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = init_logger(__name__)


class ImageRopePrepare(nn.Module):
    """Prepares image hidden_states and RoPE embeddings for sequence parallel.

    This module encapsulates the input linear projection and RoPE computation.
    Similar to Z-Image's UnifiedPrepare, this creates a module boundary where
    _sp_plan can shard outputs via split_output=True.

    The key insight is that hidden_states and vid_freqs must be sharded together
    to maintain dimension alignment for RoPE computation in attention layers.

    Note: Our _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism).
    """

    def __init__(self, img_in: nn.Linear, pos_embed: nn.Module):
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
    def __init__(self, embedding_dim, use_additional_t_cond=False):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
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

    @functools.cache
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

    @functools.cache
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
        self.rope_cache = {}

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
            rope_key = f"{idx}_{height}_{width}"

            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq = self.rope_cache[rope_key]
            else:
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

    @functools.cache
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
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str, bias: bool = True):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
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
    ) -> None:
        super().__init__()

        assert activation_fn == "gelu-approximate", "Only gelu-approximate is supported."

        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(dim, inner_dim, approximate="tanh", bias=bias),
            nn.Identity(),  # placeholder for weight loading
            RowParallelLinear(
                inner_dim,
                dim_out,
                input_is_parallel=True,
                return_bias=False,
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
        )
        self.query_num_heads = self.to_qkv.num_heads
        self.kv_num_heads = self.to_qkv.num_kv_heads

        self.norm_q = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(head_dim, eps=eps) if qk_norm else nn.Identity()

        self.inner_dim = out_dim if out_dim is not None else head_dim * self.total_num_heads

        assert context_pre_only is not None
        self.add_kv_proj = QKVParallelLinear(
            hidden_size=added_kv_proj_dim,
            head_size=head_dim,
            total_num_heads=num_heads,
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
        )

        assert not pre_only
        self.to_out = RowParallelLinear(
            self.inner_dim,
            self.dim,
            bias=out_bias,
            input_is_parallel=True,
            return_bias=False,
        )

        self.norm_added_q = RMSNorm(head_dim, eps=eps)
        self.norm_added_k = RMSNorm(head_dim, eps=eps)

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

        img_cos = vid_freqs.real.to(img_query.dtype)
        img_sin = vid_freqs.imag.to(img_query.dtype)
        txt_cos = txt_freqs.real.to(txt_query.dtype)
        txt_sin = txt_freqs.imag.to(txt_query.dtype)

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


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImageCrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            added_kv_proj_dim=dim,
            context_pre_only=False,
            head_dim=attention_head_dim,
        )
        self.img_norm2 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim)

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = AdaLayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim)

        self.zero_cond_t = zero_cond_t

    def _modulate(self, x, mod_params, index=None):
        """Apply modulation to input tensor"""
        # x: b l d, shift: b d, scale: b d, gate: b d
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

        return x * (1 + scale_result) + shift_result, gate_result

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
        img_modulated, img_gate1 = self.img_norm1(hidden_states, img_mod1, modulate_index)

        # Process text stream - norm1 + modulation
        txt_modulated, txt_gate1 = self.txt_norm1(encoder_hidden_states, txt_mod1)

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
        img_modulated2, img_gate2 = self.img_norm2(hidden_states, img_mod2, modulate_index)

        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_modulated2, txt_gate2 = self.txt_norm2(encoder_hidden_states, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


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
    _layerwise_offload_blocks_attr = "transformer_blocks"
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

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
    ):
        super().__init__()
        self.parallel_config = od_config.parallel_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.guidance_embeds = guidance_embeds

        if not use_layer3d_rope:
            self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        else:
            self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, use_additional_t_cond=use_additional_t_cond
        )

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t

        # ImageRopePrepare module for _sp_plan to shard hidden_states and vid_freqs together
        # This ensures RoPE dimensions align with hidden_states after sharding
        self.image_rope_prepare = ImageRopePrepare(self.img_in, self.pos_embed)

        # ModulateIndexPrepare module for _sp_plan to shard modulate_index
        # This ensures modulate_index dimensions align with hidden_states after sharding
        # Only active when zero_cond_t=True (image editing models)
        self.modulate_index_prepare = ModulateIndexPrepare(zero_cond_t=zero_cond_t)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: dict[str, Any] | None = None,
        additional_t_cond=None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
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
        if self.parallel_config is not None and self.parallel_config.sequence_parallel_size > 1:
            ctx = get_forward_context()
            if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
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

        # if mask is all true, set it to None
        if hidden_states_mask is not None and hidden_states_mask.all():
            hidden_states_mask = None
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

        params_dict = dict(self.named_parameters())

        # we need to load the buffers for beta and eps (XIELU)
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params

# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from cache_dit import ForwardPattern
from torch import nn
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
)
from transformers.utils import ModelOutput
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs.bagel import BagelConfig

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata as DiffusionAttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as DiffusionAttention
from vllm_omni.diffusion.cache.cache_dit_backend import BagelCachedAdapter, CacheDiTAdapterConfig
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.mot.mot_layernorm import MoTRMSNorm
from vllm_omni.diffusion.layers.mot.mot_qkv_parallel_linear import MoTQKVParallelLinear
from vllm_omni.diffusion.layers.mot.mot_row_parallel_linear import MoTRowParallelLinear
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.utils.kv_utils import left_pad_stack
from vllm_omni.model_executor.layers.timestep_embedding import timestep_embedding

logger = init_logger(__name__)


def patchify(imgs, p):
    """
    imgs: (N, 3, H, W) or (3, H, W)
    x: (N, L, patch_size**2 *3) or (L, patch_size**2 *3)
    """
    is_batch = imgs.ndim == 4
    if not is_batch:
        imgs = imgs.unsqueeze(0)

    # n: batch, c: channel, h: grid_h, p: patch_h, w: grid_w, q: patch_w
    x = imgs.reshape(imgs.shape[0], 3, imgs.shape[2] // p, p, imgs.shape[3] // p, p)
    # Permute to (n, grid_h, grid_w, c, patch_h, patch_w) to match Conv2d (c, h, w) flattening
    x = torch.einsum("nchpwq->nhwcpq", x)
    x = x.reshape(imgs.shape[0], -1, 3 * p**2)

    if not is_batch:
        x = x.squeeze(0)
    return x


class MLPconnector(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation="gelu_pytorch_tanh",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_dim, output_dim, bias=True, gather_output=False, quant_config=quant_config, prefix=f"{prefix}.fc1"
        )
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "gelu_pytorch_tanh":
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.ReLU()
        self.fc2 = RowParallelLinear(
            output_dim, output_dim, bias=True, input_is_parallel=True, quant_config=quant_config, prefix=f"{prefix}.fc2"
        )

    def forward(self, x):
        x_parallel, _ = self.fc1(x)
        x_parallel = self.act(x_parallel)
        return self.fc2(x_parallel)[0]


class BagelRotaryEmbedding(nn.Module):
    """Standalone rotary embedding that generates cos/sin from position ids.

    Replaces HuggingFace's Qwen2RotaryEmbedding while preserving full
    ``rope_scaling`` support.  When ``config.rope_scaling`` is set (e.g.
    linear, dynamic-NTK, YaRN, …), we delegate the ``inv_freq`` /
    ``attention_scaling`` computation to HF's ``ROPE_INIT_FUNCTIONS`` so
    that the frequency basis and scaling factor are identical to the
    original checkpoint.

    For Qwen2.5-VL-style multimodal RoPE (``rope_scaling.rope_type == "mrope"``)
    the ``inv_freq`` basis is the standard default-rope one; the difference
    is that position ids are 3-D ``(t, h, w)`` per token and the
    ``mrope_section`` describes how the head dimension is split across axes.
    This module accepts either 2-D scalar position ids ``(B, S)`` or 3-D
    multimodal position ids ``(B, 3, S)`` and dispatches accordingly so the
    same module works for both BAGEL (1-D rope) and Lance (Qwen2.5-VL mrope).
    This module has no learnable parameters.
    """

    def __init__(self, config):
        super().__init__()

        # transformers>=5.0 stores rope params under ``rope_parameters`` and
        # always populates ``rope_scaling`` (even for the default type), so we
        # infer ``rope_type`` from either and fall back to "default".
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = (
            rope_scaling.get("rope_type") or rope_scaling.get("type") or rope_parameters.get("rope_type") or "default"
        )
        # Cache mrope_section for the forward-time section-split.
        self._mrope_section: list[int] | None = None
        if rope_type == "mrope":
            section = rope_scaling.get("mrope_section") or rope_parameters.get("mrope_section")
            if section is None:
                raise ValueError("rope_scaling.rope_type == 'mrope' requires 'mrope_section'.")
            self._mrope_section = list(section)

        if rope_type in ("default", "mrope"):
            # mrope shares the default sinusoidal frequency basis; the
            # multimodal split happens at forward time, not at init.
            rope_theta = (
                getattr(config, "rope_theta", None)
                or rope_parameters.get("rope_theta")
                or rope_scaling.get("rope_theta")
                or 10000.0
            )
            dim = config.hidden_size // config.num_attention_heads
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            self.attention_scaling = 1.0
        else:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(config, device=None)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate cos/sin embeddings for given position ids.

        Args:
            x: Input tensor (only used for dtype inference).
            position_ids: Either 2-D scalar ``(batch_size, seq_len)`` for plain
                1-D RoPE, or 3-D multimodal ``(batch_size, 3, seq_len)`` for
                Qwen2.5-VL-style mRoPE.  The latter is auto-detected from
                ``position_ids.ndim``.

        Returns:
            cos, sin: Rotary embeddings, each of shape (batch_size, seq_len, dim).
        """
        if position_ids.ndim == 3 and self._mrope_section is not None:
            # multimodal path: position_ids is (B, 3, S) with rows = (t, h, w).
            # Compute per-axis frequencies, then assemble the per-section
            # rotary basis matching Qwen2-VL's ``apply_multimodal_rotary_pos_emb``.
            B = position_ids.shape[0]
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(B, 3, -1, 1)
            position_ids_expanded = position_ids[:, :, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
            # ``freqs`` is (B, 3, S, head_dim/2); double along the last axis
            # to get the full ``head_dim`` rotary basis.
            emb = torch.cat((freqs, freqs), dim=-1)  # (B, 3, S, head_dim)
            cos_per_axis = emb.cos() * self.attention_scaling
            sin_per_axis = emb.sin() * self.attention_scaling
            # ``mrope_section`` (e.g. [16, 24, 24] for Qwen2.5-VL) sums to
            # head_dim/2; doubled it sums to head_dim and cycles axis = i % 3.
            sec_full = self._mrope_section * 2
            cos_split = cos_per_axis.split(sec_full, dim=-1)
            sin_split = sin_per_axis.split(sec_full, dim=-1)
            cos = torch.cat([c[:, i % 3] for i, c in enumerate(cos_split)], dim=-1)
            sin = torch.cat([s[:, i % 3] for i, s in enumerate(sin_split)], dim=-1)
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BagelMLP(nn.Module):
    """FFN with Mixture-of-Tokens routing via MoT parallel linear layers.

    gate_proj + up_proj are fused into a single MoTMergedColumnParallelLinear.
    down_proj uses MoTRowParallelLinear.  Both layers hold text weights on self
    and vae weights on self.gen_exp, routing by text_indices / vae_indices.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.intermediate_size = intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported.")
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2MoTConfig(Qwen2Config):
    """Configuration for Qwen2MoT (Mixture of Tokens) model.

    This is fundamentally different from Qwen2, hence the distinct name.
    """

    model_type = "qwen2_mot"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        is_causal=True,
        _attn_implementation="eager",
        qk_norm=True,
        layer_module="Qwen2MoTDecoderLayer",
        freeze_und=False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            is_causal=is_causal,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
        self.qk_norm = qk_norm
        self.layer_module = layer_module


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}
        # Track kv_lens; we need this because we pack the forward passes
        # for CFG into a single forward call and the kv length may be different,
        # e.g., due to 0 kvs for text_cfg path and nonzero for others
        self.key_values_lens: list[int] | None = None

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0

    @classmethod
    def from_object(cls, obj) -> "NaiveCache":
        """Convert a duck-typed cache (e.g., SimpleNamespace from KV transfer)
        to NaiveCache; in the future, we should find a better way to handle this,
        e.g., a model agnostic abstraction for key cache transfer instead of having
        this cache live in bagel.

        NOTE: If a NaiveCache is provided, the object is just returned. Otherwise,
        we enumerate over the key/value cache values and map layer indices to the
        corresponding tensors.
        """
        if isinstance(obj, cls):
            return obj
        cache = cls(len(obj.key_cache))
        for i, (k, v) in enumerate(zip(obj.key_cache, obj.value_cache, strict=True)):
            cache.key_cache[i] = k
            cache.value_cache[i] = v
        return cache

    @staticmethod
    def merge(caches: Sequence["NaiveCache"]) -> "NaiveCache":
        """Merge per-branch NaiveCaches into one for batched attention;
        this lets us do the forward passes for CFG in one batched pass,
        although it's worth noting that this is currently only used
        for single request. We need this so that gen mode knows the
        respective kv lengths, and can split things back out as needed.
        """
        num_layers = caches[0].num_layers
        merged = NaiveCache(num_layers)
        lens = [c.seq_lens for c in caches]
        merged.key_values_lens = lens

        nonempty = [c for c in caches if c.key_cache[0] is not None]
        if not nonempty:
            return merged

        for layer in range(num_layers):
            merged.key_cache[layer] = torch.cat([c.key_cache[layer] for c in nonempty], dim=0)
            merged.value_cache[layer] = torch.cat([c.value_cache[layer] for c in nonempty], dim=0)

        return merged

    @staticmethod
    def split_with_zeros(
        tensor: torch.Tensor,
        lengths: Sequence[int],
    ) -> list[torch.Tensor | None]:
        """Split tensor by lengths, which may include 0 entries, e.g., for splitting cfg
        branches out, since text_cfg may have 0 kv length.

        0 lengths will be replaced with None in the returned list.
        """
        # Ensure that the lengths are all nonzero and sum to the first dim of our tensor
        if not all(isinstance(ln, int) and ln >= 0 for ln in lengths):
            raise ValueError("split lengths must be greater than or equal to zero")

        expected = sum(ln for ln in lengths if ln > 0)
        if tensor.shape[0] != expected:
            raise ValueError(f"tensor dim 0 ({tensor.shape[0]}) != sum of nonzero lengths ({expected})")

        result: list[torch.Tensor | None] = []
        offset = 0
        for ln in lengths:
            if ln > 0:
                result.append(tensor[offset : offset + ln])
                offset += ln
            else:
                result.append(None)
        return result


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: NaiveCache | None = None


class PackedAttentionMoT(nn.Module):
    """Packed attention with Mixture-of-Tokens routing for understanding/generation.

    Uses MoTQKVParallelLinear and MoTRowParallelLinear for tensor parallelism.
    Text and vae weights are held within the same MoT layer (text on self,
    vae on self.gen_exp).  Token routing is driven by text_indices / vae_indices.
    """

    def __init__(
        self,
        config,
        layer_idx: int | None = None,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.parallel_config = parallel_config

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = MoTQKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            vae_bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = MoTRowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = MoTRMSNorm(self.head_dim, head_norm=True, eps=config.rms_norm_eps)
        self.k_norm = MoTRMSNorm(self.head_dim, head_norm=True, eps=config.rms_norm_eps)

        self.rotary_op = RotaryEmbedding(is_neox_style=True)

        self.attn_causal = DiffusionAttention(
            num_heads=self.total_num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=True,
            num_kv_heads=self.total_num_kv_heads,
        )
        self.attn_noncausal = DiffusionAttention(
            num_heads=self.total_num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.total_num_kv_heads,
        )

    def _is_sp_active(self) -> bool:
        """Check if SP is active for this attention layer."""
        if not is_forward_context_available():
            return False
        return get_forward_context().sp_active

    def _forward_gen(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        past_key_values: NaiveCache | None,
        packed_vae_token_indexes: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        update_past_key_values: bool = False,
    ) -> tuple[torch.Tensor, NaiveCache | None]:
        """Forward pass for generation mode.

        This path does the following:

        1. Apply qkv projection to the text seq & vae seqs
        2. Reshape both to 3D & apply RMS norms
        3. Apply RoPE to text / VAE components independently
        4. Create the full K/V; Bagel currently manages its own KV cache
           (NaiveCache) independently since it is a diffusion model
        5. Apply non-causal attention, while taking sequence parallelism into account
        6. Apply output projections on the split parts
        7. Merge back into the packed format
        8. Update the NaiveCache.

        TODO (Alex): it would be best to remove packing from Bagel to simplify the code.
        Currently we shouldn't need it in the model, and it would be ideal to handle
        packing/batching etc in a more model agnostic way.
        """
        text_indices = packed_text_indexes
        vae_indices = packed_vae_token_indexes

        cache_k = cache_v = None
        packed_query_sequence = packed_query_sequence.to(torch.bfloat16)

        # MoT QKV projection routes text/vae tokens to the matching weights.
        qkv, _ = self.qkv_proj(packed_query_sequence, text_indices, vae_indices)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape to (tokens, heads, head_dim)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # MoT QK norms route text/vae tokens to weight/gen_weight internally.
        q = self.q_norm(q.to(torch.float32), text_indices, vae_indices)
        k = self.k_norm(k.to(torch.float32), text_indices, vae_indices)

        cos, sin = [x[..., : self.head_dim // 2] for x in packed_query_position_embeddings]
        q = self.rotary_op(q.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)
        k = self.rotary_op(k.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        text_q = q[text_indices]
        text_k = k[text_indices]
        text_v = v[text_indices]
        vae_q = q[vae_indices]
        vae_k = k[vae_indices]
        vae_v = v[vae_indices]

        num_branches = len(query_lens)
        text_per_branch = text_q.shape[0] // num_branches
        vae_per_branch = vae_q.shape[0] // num_branches

        # Build joint K/V: [kv_cache, text_markers] (replicated across SP ranks)
        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            cache_k = past_key_values.key_cache[self.layer_idx]
            cache_v = past_key_values.value_cache[self.layer_idx]
            ctx_k = torch.cat([cache_k, text_k], dim=0)
            ctx_v = torch.cat([cache_v, text_v], dim=0)
        else:
            ctx_k = text_k
            ctx_v = text_v

        # NOTE: we reshape to batched (1, S, H, D) for diffusion Attention
        # attn_out should be: (1, text_len + local_vae_len, H, D)
        if self._is_sp_active():
            # Joint mechanism keeps text+cache replicated across SP ranks
            attn_out = self.attn_noncausal(
                vae_q.unsqueeze(0),
                vae_k.unsqueeze(0),
                vae_v.unsqueeze(0),
                DiffusionAttentionMetadata(
                    joint_query=text_q.unsqueeze(0),
                    joint_key=ctx_k.unsqueeze(0),
                    joint_value=ctx_v.unsqueeze(0),
                    joint_strategy="front",
                ),
            )
        else:
            text_q_parts = text_q.split([text_per_branch] * num_branches)
            vae_q_parts = vae_q.split([vae_per_branch] * num_branches)
            text_k_parts = text_k.split([text_per_branch] * num_branches)
            vae_k_parts = vae_k.split([vae_per_branch] * num_branches)
            text_v_parts = text_v.split([text_per_branch] * num_branches)
            vae_v_parts = vae_v.split([vae_per_branch] * num_branches)

            # Query lengths should not be variable since we
            # just split above, so we just concat + stack to 4D
            q_4d = torch.stack([torch.cat([t, v]) for t, v in zip(text_q_parts, vae_q_parts)])

            if cache_k is not None and cache_v is not None:
                kv_lens = getattr(past_key_values, "key_values_lens", None)
                if kv_lens is None:
                    per_branch = cache_k.shape[0] // num_branches
                    kv_lens = [per_branch] * num_branches

                ck_per_branch = NaiveCache.split_with_zeros(cache_k, kv_lens)
                cv_per_branch = NaiveCache.split_with_zeros(cache_v, kv_lens)
                k_branches = [
                    torch.cat([t for t in (ck_per_branch[i], text_k_parts[i], vae_k_parts[i]) if t is not None])
                    for i in range(num_branches)
                ]
                v_branches = [
                    torch.cat([t for t in (cv_per_branch[i], text_v_parts[i], vae_v_parts[i]) if t is not None])
                    for i in range(num_branches)
                ]
                k_4d, mask = left_pad_stack(k_branches)
                v_4d, _ = left_pad_stack(v_branches)
                metadata = DiffusionAttentionMetadata(attn_mask=mask) if mask is not None else None
            else:
                k_4d = torch.stack([torch.cat([t, v]) for t, v in zip(text_k_parts, vae_k_parts)])
                v_4d = torch.stack([torch.cat([t, v]) for t, v in zip(text_v_parts, vae_v_parts)])
                metadata = None
            attn_out = self.attn_noncausal(q_4d, k_4d, v_4d, metadata)

        attn_out = attn_out.reshape(num_branches, -1, self.q_size)
        text_attn = attn_out[:, :text_per_branch].reshape(-1, self.q_size)
        vae_attn = attn_out[:, text_per_branch:].reshape(-1, self.q_size)
        text_len = text_attn.shape[0]

        # MoT output projection over local [text, vae] order.
        local_packed = torch.cat([text_attn, vae_attn], dim=0)
        local_text_idx = torch.arange(text_len, device=local_packed.device)
        local_vae_idx = torch.arange(text_len, text_len + vae_attn.shape[0], device=local_packed.device)
        local_out, _ = self.o_proj(local_packed, local_text_idx, local_vae_idx)
        text_out = local_out[:text_len]
        vae_out = local_out[text_len:]

        # Merge back into packed format
        total_len = packed_query_sequence.shape[0]
        full_output = text_out.new_zeros((total_len, self.hidden_size))
        full_output[packed_text_indexes] = text_out
        full_output[packed_vae_token_indexes] = vae_out

        if update_past_key_values:
            new_k = torch.cat([ctx_k, vae_k], dim=0)
            new_v = torch.cat([ctx_v, vae_v], dim=0)
            past_key_values.key_cache[self.layer_idx] = new_k
            past_key_values.value_cache[self.layer_idx] = new_v

        return full_output, past_key_values

    def _forward_und(
        self,
        packed_query_sequence: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        past_key_values: NaiveCache | None,
        is_causal: bool,
        update_past_key_values: bool = True,
    ) -> tuple[torch.Tensor, NaiveCache | None]:
        """Forward pass for understanding mode.

        This path does the following (not hard to read):

        1. Apply qkv projection to the text seq
        2. Reshape to 3D & apply RMS norms
        3. Apply RoPE
        4. Create the full K/V; Bagel currently manages its own KV cache
           (NaiveCache) independently since it is a diffusion model
        5. Apply attention based on causality kwarg
        6. Apply output projection
        7. Update the NaiveCache.
        """

        qkv, _ = self.qkv_proj(packed_query_sequence)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # Pre-merge code cast to float32 before q_norm/k_norm — bf16
        # RMSNorm accumulation drift compounds across segmented prefill
        # (x2t builds the cache via 3+ sequential forward_cache_update_*
        # calls), producing degenerate token repetition in generate_text.
        q = self.q_norm(q.to(torch.float32))
        k = self.k_norm(k.to(torch.float32))

        cos, sin = [x[..., : self.head_dim // 2] for x in packed_query_position_embeddings]
        q = self.rotary_op(q.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)
        k = self.rotary_op(k.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            cache_k = past_key_values.key_cache[self.layer_idx]
            cache_v = past_key_values.value_cache[self.layer_idx]
            full_k = torch.cat([cache_k, k], dim=0)
            full_v = torch.cat([cache_v, v], dim=0)
            cache_len = cache_k.shape[0]
        else:
            full_k = k
            full_v = v
            cache_len = 0

        if is_causal and cache_len > 0:
            # PyTorch SDPA's ``is_causal=True`` with ``Q != K`` uses
            # ``tril(diagonal=0)`` — top-left aligned, so for Q=1 against a
            # long cache only ``q_0 -> k_0`` is unmasked (degenerate
            # generate_text output).  Build the correct bottom-right
            # aligned mask manually: ``mask[i, j] = -inf if j > cache_len
            # + i`` so each new query attends to all of ``cache + self
            # up to i``.  Bypass ``DiffusionAttention`` entirely — its
            # ``_maybe_reshape_attn_mask`` helper only handles 2-D
            # ``(B, K)`` shape, not ``(Q, K)``, so the reshape path
            # silently drops the per-query mask for Q>1.
            Q_len = q.shape[0]
            K_len = full_k.shape[0]
            arange_q = torch.arange(Q_len, device=q.device).unsqueeze(1)
            arange_k = torch.arange(K_len, device=q.device).unsqueeze(0)
            mask = arange_k > (cache_len + arange_q)  # (Q, K) bool
            attn_bias = torch.zeros(Q_len, K_len, dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask, float("-inf"))
            # Permute to (B=1, H, S, D) for SDPA.
            q_4d = q.unsqueeze(0).permute(0, 2, 1, 3)
            k_4d = full_k.unsqueeze(0).permute(0, 2, 1, 3)
            v_4d = full_v.unsqueeze(0).permute(0, 2, 1, 3)
            attn_out_4d = torch.nn.functional.scaled_dot_product_attention(
                q_4d,
                k_4d,
                v_4d,
                attn_mask=attn_bias.unsqueeze(0).unsqueeze(0),  # (1, 1, Q, K)
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / (self.head_dim**0.5),
                enable_gqa=(self.num_heads != self.num_kv_heads),
            )
            attn_out = attn_out_4d.permute(0, 2, 1, 3)
        else:
            attn = self.attn_causal if is_causal else self.attn_noncausal
            attn_out = attn(
                q.unsqueeze(0),
                full_k.unsqueeze(0),
                full_v.unsqueeze(0),
            )

        attn_out = attn_out.squeeze(0).reshape(-1, self.q_size)
        attn_out, _ = self.o_proj(attn_out)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = full_k
            past_key_values.value_cache[self.layer_idx] = full_v

        return attn_out, past_key_values

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        if mode == "gen":
            if is_causal:
                raise ValueError("Generation model for Bagel requires non-causal attention")
            return self._forward_gen(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                past_key_values=past_key_values,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
                update_past_key_values=update_past_key_values,
            )

        return self._forward_und(
            packed_query_sequence=packed_query_sequence,
            packed_query_position_embeddings=packed_query_position_embeddings,
            past_key_values=past_key_values,
            is_causal=is_causal,
            update_past_key_values=update_past_key_values,
        )


class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int | None = None,
        attn_module: type[nn.Module] | None = PackedAttentionMoT,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = attn_module(
            config, layer_idx, parallel_config=parallel_config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )

        self.input_layernorm = MoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = BagelMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.mlp_moe_gen = BagelMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp_moe_gen",
        )

        self.post_attention_layernorm = MoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        packed_query_sequence: torch.Tensor | None = None,
        query_lens: torch.Tensor = None,
        packed_query_position_embeddings: torch.Tensor = None,
        past_key_values: NaiveCache | None = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        if packed_query_sequence is None:
            packed_query_sequence = hidden_states

        text_indices = packed_text_indexes if mode == "gen" else None
        vae_indices = packed_vae_token_indexes if mode == "gen" else None

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence, text_indices, vae_indices)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            past_key_values=past_key_values,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_normed = self.post_attention_layernorm(packed_query_sequence, text_indices, vae_indices).to(
                torch.bfloat16
            )
            packed_text_query_sequence = packed_normed[packed_text_indexes]
            packed_vae_query_sequence = packed_normed[packed_vae_token_indexes]
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2MoTModel(Qwen2PreTrainedModel):
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={
            "layers": ForwardPattern.Pattern_0,
        },
        cached_adapter_cls=BagelCachedAdapter,
    )

    _layerwise_offload_blocks_attrs = ["layers"]

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "layers" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        config,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = "Mo" in config.layer_module

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen2MoTDecoderLayer(
                    config,
                    layer_idx,
                    attn_module=PackedAttentionMoT,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = MoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BagelRotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        packed_query_sequence: torch.Tensor | None = None,
        query_lens: torch.Tensor | None = None,
        packed_query_position_ids: torch.Tensor | None = None,
        past_key_values: NaiveCache | None = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        packed_text_ids: torch.Tensor | None = None,
        return_embeddings_only: bool = False,
    ) -> BaseNavitOutputWithPast:
        if packed_query_sequence is None:
            if packed_text_ids is None:
                raise ValueError("Either packed_query_sequence or packed_text_ids must be provided.")
            packed_query_sequence = self.embed_tokens(packed_text_ids)

        if return_embeddings_only:
            return BaseNavitOutputWithPast(
                packed_query_sequence=packed_query_sequence,
                past_key_values=past_key_values,
            )

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == "gen":
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        for layer_idx, decoder_layer in enumerate(self.layers):
            # TODO (Alex): Remove encoder_hidden_states as a kwarg; currently we keep it
            # for compatibility with the current custom CacheDiT adapter, as we need to be
            # careful to not break the NaiveCache handling when switching from pattern
            # 0 -> 4.
            packed_query_sequence, past_key_values = decoder_layer(
                hidden_states=packed_query_sequence,
                encoder_hidden_states=None,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                past_key_values=past_key_values,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        text_indices = packed_text_indexes if self.use_moe and mode == "gen" else None
        vae_indices = packed_vae_token_indexes if self.use_moe and mode == "gen" else None
        packed_query_sequence = self.norm(packed_query_sequence, text_indices, vae_indices)

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen2MoTForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.model = Qwen2MoTModel(
            config, parallel_config=parallel_config, quant_config=quant_config, prefix=f"{prefix}.model"
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        packed_query_sequence: torch.Tensor | None = None,
        query_lens: torch.Tensor | None = None,
        packed_query_position_ids: torch.Tensor | None = None,
        past_key_values: NaiveCache | None = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        packed_text_ids: torch.Tensor | None = None,
        return_embeddings_only: bool = False,
    ) -> BaseNavitOutputWithPast:
        outputs = self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            past_key_values=past_key_values,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
            packed_text_ids=packed_text_ids,
            return_embeddings_only=return_embeddings_only,
        )

        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for MoT parallel layers.

        Stacked parameter remapping (checkpoint name → model parameter):
          - q/k/v_proj       → qkv_proj          (text, shard q/k/v)
          - q/k/v_proj_moe_gen → qkv_proj.gen_exp (gen,  shard q/k/v)

        Direct remapping (no shard dimension):
          - o_proj_moe_gen   → o_proj.gen_exp
          - {norm}_moe_gen.weight → {norm}.gen_weight  (all MoTRMSNorm layers)

        Text norm weights (input_layernorm.weight, q_norm.weight, etc.) and
        other names (embed_tokens, lm_head) pass through unchanged.
        """
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            # _moe_gen patterns MUST come first — `.q_proj` is a substring
            # of `.q_proj_moe_gen`, so the more specific pattern must match first.
            (".qkv_proj.gen_exp", ".q_proj_moe_gen", "q"),
            (".qkv_proj.gen_exp", ".k_proj_moe_gen", "k"),
            (".qkv_proj.gen_exp", ".v_proj_moe_gen", "v"),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".mlp_moe_gen.gate_up_proj", ".mlp_moe_gen.gate_proj", 0),
            (".mlp_moe_gen.gate_up_proj", ".mlp_moe_gen.up_proj", 1),
            (".mlp.gate_up_proj", ".mlp.gate_proj", 0),
            (".mlp.gate_up_proj", ".mlp.up_proj", 1),
        ]

        direct_remap = [
            (".o_proj_moe_gen.", ".o_proj.gen_exp."),
            # Norm _moe_gen.weight → {norm_name}.gen_weight
            (".input_layernorm_moe_gen.", ".input_layernorm.gen_"),
            (".post_attention_layernorm_moe_gen.", ".post_attention_layernorm.gen_"),
            (".q_norm_moe_gen.", ".q_norm.gen_"),
            (".k_norm_moe_gen.", ".k_norm.gen_"),
            (".norm_moe_gen.", ".norm.gen_"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def handle_weight(name, loaded_weight, shard_id=None):
            param = params_dict.get(name)
            if param is None:
                logger.warning_once("Skipping weight %r: no matching parameter found in model.", name)
                return
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is not None:
                weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        for name, loaded_weight in weights:
            # match direct remap
            handled = False
            for old_substr, new_substr in direct_remap:
                if old_substr in name:
                    name = name.replace(old_substr, new_substr)
                    handle_weight(name, loaded_weight)
                    handled = True
                    break

            if handled:
                continue

            # match stacked params mapping
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                handle_weight(name, loaded_weight, shard_id)
                handled = True
                break

            if handled:
                continue

            # no-name-match cases are handled here
            handle_weight(name, loaded_weight)

        return loaded_params


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(torch.zeros(max_num_patch_per_side**2, hidden_size), requires_grad=False)
        self._init_weights()

    def _init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids):
        return self.pos_embed[position_ids]


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


class Bagel(CFGParallelMixin, nn.Module):
    config_class = BagelConfig
    base_model_prefix = "bagel"

    # Flow-matching denoise schedule convention. Official BAGEL samples
    # ``num_timesteps`` points over [1, 0] and drops the terminal t=0, yielding
    # ``num_timesteps - 1`` Euler steps. Lance samples one extra point
    # (``num_timesteps + 1``) for ``num_timesteps`` steps; ``LanceBagel`` flips
    # this on. See https://github.com/vllm-project/vllm-omni/issues/4470.
    _denoise_schedule_extra_step: bool = False

    def __init__(
        self,
        language_model,
        vit_model,
        config: BagelConfig,
        parallel_config: DiffusionParallelConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads
        self.parallel_config = parallel_config

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(
                self.vit_hidden_size,
                self.hidden_size,
                config.connector_act,
                quant_config=quant_config,
                prefix=f"{prefix}.connector",
            )
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    @property
    def _sp_size(self) -> int:
        if self.parallel_config is None:
            return 1
        sp = self.parallel_config.sequence_parallel_size
        return sp if sp is not None and sp > 1 else 1

    def _split_vae_for_sp(
        self,
        x_t: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_vae_token_indexes: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_seqlens: torch.Tensor,
        packed_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split VAE tokens across SP ranks for the denoising loop.

        Returns adjusted (x_t, packed_vae_position_ids, packed_vae_token_indexes,
        packed_text_indexes, packed_seqlens, packed_position_ids) for the local rank.
        """
        sp_size = self._sp_size
        sp_rank = get_sequence_parallel_rank()
        num_vae = x_t.shape[0]
        assert num_vae % sp_size == 0, f"VAE token count {num_vae} not divisible by SP size {sp_size}"
        chunk = num_vae // sp_size
        start = sp_rank * chunk
        end = start + chunk

        local_x_t = x_t[start:end]
        local_vae_pos_ids = packed_vae_position_ids[start:end]

        # Rebuild local packed indices:
        # packed sequence = [start_of_image, local_vae_tokens..., end_of_image]
        # BAGEL always has exactly 2 text markers (start/end_of_image).
        num_text = packed_text_indexes.shape[0]
        assert num_text == 2, f"Expected exactly 2 text markers (start/end_of_image), got {num_text}"
        assert packed_seqlens.numel() == 1, (
            f"SP currently supports single-image batches only, got {packed_seqlens.numel()} sequences"
        )
        local_vae_len = chunk
        local_total = num_text + local_vae_len

        local_text_indexes = torch.tensor([0, local_vae_len + 1], device=packed_text_indexes.device)
        local_vae_indexes = torch.arange(1, 1 + local_vae_len, device=packed_vae_token_indexes.device)

        local_seqlens = torch.tensor([local_total], device=packed_seqlens.device, dtype=packed_seqlens.dtype)

        # Build local position IDs preserving global positions.
        # Text markers keep their original positions; VAE tokens get
        # the global positions for the local chunk.
        text_pos_ids = packed_position_ids[packed_text_indexes]
        vae_pos_ids_full = packed_position_ids[packed_vae_token_indexes]
        local_vae_pos = vae_pos_ids_full[start:end]
        local_position_ids = torch.zeros(
            local_total, device=packed_position_ids.device, dtype=packed_position_ids.dtype
        )
        local_position_ids[local_text_indexes] = text_pos_ids
        local_position_ids[local_vae_indexes] = local_vae_pos

        return local_x_t, local_vae_pos_ids, local_vae_indexes, local_text_indexes, local_seqlens, local_position_ids

    def _gather_vae_for_sp(self, local_v_t: torch.Tensor) -> torch.Tensor:
        """Gather VAE velocity outputs from all SP ranks."""
        sp_size = self._sp_size
        gathered = [torch.zeros_like(local_v_t) for _ in range(sp_size)]
        sp_group = get_sp_group()
        dist.all_gather(gathered, local_v_t.contiguous(), group=sp_group.device_group)
        return torch.cat(gathered, dim=0)

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()

        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            text_ids = tokenizer.encode(prompt, add_special_tokens=False)
            text_ids = [new_token_ids["bos_token_id"]] + text_ids + [new_token_ids["eos_token_id"]]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
        }

        return generation_input, newlens, new_rope

    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
    ):
        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward(
            packed_text_ids=packed_text_ids,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            past_key_values=past_key_values,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids = list(), list()

        _curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            packed_vae_position_ids.append(vae_position_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: list,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
    ):
        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, : h * p, : w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed

        # Mirror forward_cache_update_vit: build a full packed_sequence so MoE gen-mode
        # indexes (packed_text_indexes / packed_vae_token_indexes) match tensor length.
        packed_text_embedding = self.language_model.forward(
            packed_text_ids=packed_text_ids,
            return_embeddings_only=True,
        ).packed_query_sequence
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes,
            }

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            past_key_values=past_key_values,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids = list(), list()

        _curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(_curr)
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.vit_max_num_patch_per_side,
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(_curr)
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.forward(
            packed_text_ids=packed_text_ids,
            return_embeddings_only=True,
        ).packed_query_sequence
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            past_key_values=past_key_values,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_input(self, curr_kvlens, curr_rope, image_sizes, new_token_ids=None):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens = list(), list()

        query_curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(query_curr)
            query_curr += 1

            vae_position_ids = self.get_flattened_position_ids(
                H, W, self.latent_downsample, max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_position_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w

            packed_init_noises.append(torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size**2))
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_seqlens.append(num_image_tokens + 2)
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(query_curr)
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        # Construct Output
        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        return self.prepare_input(curr_kvlens, curr_rope, image_sizes, new_token_ids)

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids = list()

        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
        }

        return generation_input

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        """Prepare start tokens for autoregressive text generation.

        Ported from the original BAGEL ``Bagel.prepare_start_tokens``.
        """
        packed_start_tokens = list()
        packed_query_position_ids = list()

        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_start_tokens.append(new_token_ids["bos_token_id"])
            packed_query_position_ids.append(curr_position_id)
        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
        }
        return generation_input

    @torch.no_grad()
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int | None = None,
    ):
        """Autoregressive text generation (ported from original BAGEL).

        Decodes tokens one at a time, appending to ``past_key_values``
        until ``max_length`` is reached or ``end_token_id`` is generated.
        """
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)

            output = self.language_model(
                packed_text_ids=curr_tokens,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                past_key_values=past_key_values,
                update_past_key_values=True,
                is_causal=True,
                mode="und",
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id:
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        past_key_values: NaiveCache,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: tuple[float, float] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: torch.LongTensor | None = None,
        cfg_text_past_key_values: NaiveCache | None = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: torch.LongTensor | None = None,
        cfg_img_past_key_values: NaiveCache | None = None,
        return_trajectory_latents: bool = False,
        scheduler: object | None = None,
        scheduler_kwargs: dict | None = None,
        # Lance i2v: tokens to freeze at their initial (encoded-image)
        # value throughout the denoise loop.  Matches upstream's
        # ``mse_loss_indexes``-exclusion behaviour from PR #33: cond
        # positions are never updated and get ``timestep=0``.
        frame_condition_token_indexes: torch.LongTensor | None = None,
    ):
        x_t = packed_init_noises
        # Snapshot the pinned subtensor BEFORE the denoise loop touches
        # x_t; the cond positions in packed_init_noises hold the
        # VAE-encoded conditioning latent that must be preserved verbatim.
        pinned_x_t = None
        if frame_condition_token_indexes is not None:
            frame_condition_token_indexes = frame_condition_token_indexes.to(x_t.device).long()
            pinned_x_t = x_t[frame_condition_token_indexes].clone()

        # Build the flow-matching schedule. BAGEL drops the terminal t=0 for
        # ``num_timesteps - 1`` Euler steps; Lance keeps it for ``num_timesteps``.
        # ``_denoise_schedule_extra_step`` (overridden by ``LanceBagel``) selects which.
        num_sample_points = num_timesteps + 1 if self._denoise_schedule_extra_step else num_timesteps
        timesteps = torch.linspace(1, 0, num_sample_points, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        # Optional trajectory recording for RL rollout data collection
        trajectory_latents: list[torch.Tensor] | None = [] if return_trajectory_latents else None
        trajectory_timesteps: list[torch.Tensor] | None = [] if return_trajectory_latents else None
        trajectory_log_probs: list[torch.Tensor] | None = (
            [] if (return_trajectory_latents and scheduler is not None) else None
        )
        _sched_kw = scheduler_kwargs or {}

        use_cfg_text = cfg_text_scale > 1.0
        use_cfg_img = cfg_img_scale > 1.0

        # ── Detect CFG parallel mode ──
        cfg_parallel_ready = use_cfg_text and get_classifier_free_guidance_world_size() > 1

        if cfg_parallel_ready:
            return self._generate_image_parallel(
                x_t=x_t,
                timesteps=timesteps,
                dts=dts,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_seqlens=packed_seqlens,
                packed_position_ids=packed_position_ids,
                past_key_values=past_key_values,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_interval=cfg_interval,
                cfg_text_scale=cfg_text_scale,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_scale=cfg_img_scale,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_past_key_values=cfg_img_past_key_values,
                return_trajectory_latents=return_trajectory_latents,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
            )

        # ── SP + CFG: sequential single-branch forwards ──
        use_sp = self._sp_size > 1
        if use_sp and use_cfg_text:
            if return_trajectory_latents and len(timesteps) > 0:
                trajectory_latents.append(x_t.clone())
            for i, t in enumerate(timesteps.tolist()):  # host floats; a 0-d tensor t would sync each step
                timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                if frame_condition_token_indexes is not None:
                    # Cond positions stay at t=0 (clean signal).  Matches upstream
                    # PR #33 lance.py line 1605:
                    #     timestep[current_vae_mse_indexes_local_in_vae] = t
                    # (cond positions remain at the ``torch.zeros`` init value).
                    timestep[frame_condition_token_indexes] = 0.0
                in_cfg_window = t > cfg_interval[0] and t <= cfg_interval[1]
                cfg_text_scale_ = cfg_text_scale if in_cfg_window else 1.0
                cfg_img_scale_ = cfg_img_scale if in_cfg_window else 1.0

                common = dict(
                    x_t=x_t,
                    timestep=timestep,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_seqlens=packed_seqlens,
                )

                v_t = self.forward_single_branch(
                    **common,
                    packed_position_ids=packed_position_ids,
                    past_key_values=past_key_values,
                )

                if cfg_text_scale_ > 1.0:
                    cfg_text_v_t = self.forward_single_branch(
                        **common,
                        packed_position_ids=cfg_text_packed_position_ids,
                        past_key_values=cfg_text_past_key_values,
                    )
                    cfg_img_v_t = None
                    if cfg_img_scale_ > 1.0:
                        cfg_img_v_t = self.forward_single_branch(
                            **common,
                            packed_position_ids=cfg_img_packed_position_ids,
                            past_key_values=cfg_img_past_key_values,
                        )
                    v_t = self._combine_cfg(
                        v_t,
                        cfg_text_v_t,
                        cfg_img_v_t,
                        cfg_text_scale_,
                        cfg_img_scale_,
                        cfg_renorm_type,
                        cfg_renorm_min,
                    )

                if scheduler is not None:
                    out = scheduler.step(v_t.to(x_t.device), timesteps[i], x_t, dts[i], **_sched_kw)
                    x_t = out.prev_sample
                    if trajectory_log_probs is not None and out.log_prob is not None:
                        trajectory_log_probs.append(out.log_prob)
                else:
                    x_t = x_t - v_t.to(x_t.device) * dts[i]
                if return_trajectory_latents:
                    trajectory_latents.append(x_t.clone())
                    trajectory_timesteps.append(timesteps[i])

            unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
            return unpacked_latent, trajectory_latents, trajectory_timesteps, trajectory_log_probs

        # ── SP without CFG: direct single-branch loop ──
        if use_sp:
            if return_trajectory_latents and len(timesteps) > 0:
                trajectory_latents.append(x_t.clone())
            for i, t in enumerate(timesteps.tolist()):  # host floats; a 0-d tensor t would sync each step
                timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                if frame_condition_token_indexes is not None:
                    # Cond positions stay at t=0 (clean signal).  Matches upstream
                    # PR #33 lance.py line 1605:
                    #     timestep[current_vae_mse_indexes_local_in_vae] = t
                    # (cond positions remain at the ``torch.zeros`` init value).
                    timestep[frame_condition_token_indexes] = 0.0
                v_t = self.forward_single_branch(
                    x_t=x_t,
                    timestep=timestep,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_vae_position_ids=packed_vae_position_ids,
                    packed_text_ids=packed_text_ids,
                    packed_text_indexes=packed_text_indexes,
                    packed_position_ids=packed_position_ids,
                    packed_seqlens=packed_seqlens,
                    past_key_values=past_key_values,
                )
                if scheduler is not None:
                    out = scheduler.step(v_t.to(x_t.device), timesteps[i], x_t, dts[i], **_sched_kw)
                    x_t = out.prev_sample
                    out_log_prob = getattr(out, "log_prob", None)
                    if trajectory_log_probs is not None and out_log_prob is not None:
                        trajectory_log_probs.append(out_log_prob)
                else:
                    x_t = x_t - v_t.to(x_t.device) * dts[i]
                if return_trajectory_latents:
                    trajectory_latents.append(x_t.clone())
                    trajectory_timesteps.append(timesteps[i])

            unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
            return unpacked_latent, trajectory_latents, trajectory_timesteps, trajectory_log_probs

        # ── Sequential CFG mode (cfg_parallel_size=1, no SP) ──
        # Each CFG branch runs its own LLM forward; we just need the
        # per-branch packed_position_ids and past_key_values for
        # ``Bagel.forward`` to dispatch through.
        cfg_branch_pids: list[torch.Tensor] | None = None
        cfg_branch_caches: list[NaiveCache] | None = None

        if use_cfg_text:
            cfg_branch_pids = [packed_position_ids, cfg_text_packed_position_ids]
            cfg_branch_caches = [past_key_values, cfg_text_past_key_values]
            if use_cfg_img:
                cfg_branch_pids.append(cfg_img_packed_position_ids)
                cfg_branch_caches.append(cfg_img_past_key_values)

        if return_trajectory_latents and len(timesteps) > 0:
            trajectory_latents.append(x_t.clone())

        for i, t in enumerate(timesteps.tolist()):  # host floats; a 0-d tensor t would sync each step
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if frame_condition_token_indexes is not None:
                # Cond positions stay at t=0 (clean signal).  Matches upstream
                # PR #33 lance.py line 1605:
                #     timestep[current_vae_mse_indexes_local_in_vae] = t
                # (cond positions remain at the ``torch.zeros`` init value).
                timestep[frame_condition_token_indexes] = 0.0
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self.forward(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_seqlens=packed_seqlens,
                past_key_values=past_key_values,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_text_scale=cfg_text_scale_,
                cfg_img_scale=cfg_img_scale_,
                cfg_branch_pids=cfg_branch_pids,
                cfg_branch_caches=cfg_branch_caches,
            )

            if scheduler is not None:
                out = scheduler.step(v_t.to(x_t.device), timesteps[i], x_t, dts[i], **_sched_kw)
                x_t = out.prev_sample
                if trajectory_log_probs is not None and out.log_prob is not None:
                    trajectory_log_probs.append(out.log_prob)
            else:
                x_t = x_t - v_t.to(x_t.device) * dts[i]  # velocity pointing from data to noise
                if pinned_x_t is not None:
                    # i2v: restore cond positions to their encoded-image
                    # latent.  Matches upstream PR #33 lance.py line 1712:
                    #     x_t[mse_indexes] = x_t[mse_indexes] - v_t[mse_indexes] * dts[i]
                    # (cond positions are excluded from the update).
                    x_t[frame_condition_token_indexes] = pinned_x_t
            if return_trajectory_latents:
                trajectory_latents.append(x_t.clone())
                trajectory_timesteps.append(timesteps[i])

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent, trajectory_latents, trajectory_timesteps, trajectory_log_probs

    def _generate_image_parallel(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        dts: torch.Tensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        past_key_values: NaiveCache,
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        cfg_interval: tuple[float, float],
        cfg_text_scale: float,
        cfg_text_packed_position_ids: torch.LongTensor | None,
        cfg_text_past_key_values: NaiveCache | None,
        cfg_img_scale: float,
        cfg_img_packed_position_ids: torch.LongTensor | None,
        cfg_img_past_key_values: NaiveCache | None,
        return_trajectory_latents: bool = False,
        scheduler: object | None = None,
        scheduler_kwargs: dict | None = None,
        frame_condition_token_indexes: torch.LongTensor | None = None,
    ):
        """CFG parallel denoising loop: each rank computes one CFG branch.

        Rank 0: gen branch (full conditioning)
        Rank 1: text_cfg branch (unconditional text)
        Rank 2: img_cfg branch (no image condition), only when cfg_img_scale > 1.0
        """
        cfg_group = get_cfg_group()
        cfg_world_size = get_classifier_free_guidance_world_size()
        use_cfg_img = cfg_img_scale > 1.0

        # Validate cfg_parallel_size vs cfg_img_scale consistency
        if cfg_world_size == 3 and not use_cfg_img:
            raise ValueError(
                f"cfg_parallel_size=3 requires cfg_img_scale > 1.0, "
                f"but got cfg_img_scale={cfg_img_scale}. "
                f"Use cfg_parallel_size=2 for text-only CFG parallel(text2img), or set cfg_img_scale > 1.0."
            )
        if cfg_world_size == 2 and use_cfg_img:
            raise ValueError(
                f"Image CFG (cfg_img_scale={cfg_img_scale}) requires cfg_parallel_size=3, "
                f"but got cfg_parallel_size=2. "
                f"Use cfg_parallel_size=3 to enable image CFG in parallel mode."
            )

        # Ensure all ranks start with the same x_t (initial noise may differ
        # across ranks when no per-request seed is set).
        x_t = x_t.contiguous()
        cfg_group.broadcast(x_t, src=0)

        trajectory_latents: list[torch.Tensor] | None = [] if return_trajectory_latents else None
        trajectory_timesteps: list[torch.Tensor] | None = [] if return_trajectory_latents else None
        trajectory_log_probs: list[torch.Tensor] | None = (
            [] if (return_trajectory_latents and scheduler is not None) else None
        )
        _sched_kw = scheduler_kwargs or {}

        if return_trajectory_latents and len(timesteps) > 0:
            trajectory_latents.append(x_t.clone())

        for i, t in enumerate(timesteps.tolist()):  # host floats; a 0-d tensor t would sync each step
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if frame_condition_token_indexes is not None:
                # Cond positions stay at t=0 (clean signal).  Matches upstream
                # PR #33 lance.py line 1605:
                #     timestep[current_vae_mse_indexes_local_in_vae] = t
                # (cond positions remain at the ``torch.zeros`` init value).
                timestep[frame_condition_token_indexes] = 0.0
            use_cfg_this_step = t > cfg_interval[0] and t <= cfg_interval[1] and cfg_text_scale > 1.0

            # Per-branch kwargs. Branch 0 (gen, full conditioning) is also the
            # branch used by all ranks when do_true_cfg is False (outside the
            # CFG interval) — CFGParallelMixin only runs branches_kwargs[0] then,
            # mirroring the previous "all ranks compute gen inputs, no comm" path.
            common = dict(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_seqlens=packed_seqlens,
            )
            branches_kwargs = [
                dict(**common, packed_position_ids=packed_position_ids, past_key_values=past_key_values),
                dict(
                    **common, packed_position_ids=cfg_text_packed_position_ids, past_key_values=cfg_text_past_key_values
                ),
            ]
            if use_cfg_img:
                branches_kwargs.append(
                    dict(
                        **common,
                        packed_position_ids=cfg_img_packed_position_ids,
                        past_key_values=cfg_img_past_key_values,
                    )
                )

            # Each rank computes its assigned branch, then all_gather + combine
            # happen inside the mixin (identical result on every rank).
            v_t = self.predict_noise_with_multi_branch_cfg(
                do_true_cfg=use_cfg_this_step,
                true_cfg_scale={
                    "cfg_text_scale": cfg_text_scale,
                    "cfg_img_scale": cfg_img_scale,
                    "cfg_renorm_type": cfg_renorm_type,
                    "cfg_renorm_min": cfg_renorm_min,
                },
                branches_kwargs=branches_kwargs,
            )

            if scheduler is not None:
                out = scheduler.step(v_t.to(x_t.device), timesteps[i], x_t, dts[i], **_sched_kw)
                x_t = out.prev_sample
                if trajectory_log_probs is not None and out.log_prob is not None:
                    trajectory_log_probs.append(out.log_prob)
            else:
                x_t = x_t - v_t.to(x_t.device) * dts[i]
            if return_trajectory_latents:
                trajectory_latents.append(x_t.clone())
                trajectory_timesteps.append(timesteps[i])

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent, trajectory_latents, trajectory_timesteps, trajectory_log_probs

    @staticmethod
    def _combine_cfg(
        v_t: torch.Tensor,
        cfg_text_v_t: torch.Tensor,
        cfg_img_v_t: torch.Tensor | None,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_renorm_type: str,
        cfg_renorm_min: float,
    ) -> torch.Tensor:
        """Combine 3-branch CFG predictions with renormalization.

        Args:
            v_t: velocity from gen branch (full conditioning)
            cfg_text_v_t: velocity from text_cfg branch (unconditional text)
            cfg_img_v_t: velocity from img_cfg branch (no image), or None
            cfg_text_scale: text guidance scale
            cfg_img_scale: image guidance scale
            cfg_renorm_type: "text_channel", "global", or "channel"
            cfg_renorm_min: minimum renormalization scale
        """
        if cfg_renorm_type == "text_channel":
            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
            norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
            scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            v_t_text = v_t_text_ * scale
            if cfg_img_scale > 1.0 and cfg_img_v_t is not None:
                v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
            else:
                v_t = v_t_text
        else:
            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

            if cfg_img_scale > 1.0 and cfg_img_v_t is not None:
                v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
            else:
                v_t_ = v_t_text_

            # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
            if cfg_renorm_type == "global":
                norm_v_t = torch.norm(v_t)
                norm_v_t_ = torch.norm(v_t_)
            elif cfg_renorm_type == "channel":
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
            else:
                raise NotImplementedError(f"{cfg_renorm_type} is not supported")
            scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            v_t = v_t_ * scale

        return v_t

    # ── CFGParallelMixin hooks ──
    # Bagel mounts CFGParallelMixin (see class declaration) to reuse the shared
    # N-branch CFG dispatch/all_gather logic in predict_noise_with_multi_branch_cfg.
    # Only two hooks need model-specific behaviour:
    #   * predict_noise: one branch == one Bagel forward (per-branch KV cache).
    #   * combine_multi_branch_cfg_noise: Bagel's renorm-aware 3-branch combine.

    def predict_noise(self, **kwargs) -> torch.Tensor:
        """Single-branch velocity prediction for CFGParallelMixin.

        Each CFG branch differs only by ``packed_position_ids`` and
        ``past_key_values`` (carried in ``kwargs``); the heavy lifting is the
        per-branch ``forward_single_branch`` pass.
        """
        return self.forward_single_branch(**kwargs)

    def combine_multi_branch_cfg_noise(
        self,
        predictions: list[torch.Tensor],
        true_cfg_scale: dict[str, float],
        cfg_normalize: bool = False,
    ) -> torch.Tensor:
        """Combine gen/text/img branch velocities via Bagel's renorm CFG.

        ``predictions[0]`` is the gen branch, ``[1]`` the text-CFG branch, and
        ``[2]`` (when present) the image-CFG branch. ``cfg_normalize`` is unused
        because renormalization is folded into ``_combine_cfg`` itself.
        """
        cfg_img_v_t = predictions[2] if len(predictions) > 2 else None
        return self._combine_cfg(
            predictions[0],
            predictions[1],
            cfg_img_v_t,
            true_cfg_scale["cfg_text_scale"],
            true_cfg_scale["cfg_img_scale"],
            true_cfg_scale["cfg_renorm_type"],
            true_cfg_scale["cfg_renorm_min"],
        )

    def forward_single_branch(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        past_key_values: NaiveCache,
    ) -> torch.Tensor:
        """Run a single-branch forward pass (no CFG batching).

        Used by CFG parallel mode where each rank computes one branch.
        Returns the velocity v_t for the given branch.
        Supports Ulysses / Ring SP when parallel_config.sequence_parallel_size > 1.
        """
        use_sp = self._sp_size > 1

        if use_sp:
            # Split VAE tokens across SP ranks
            (
                local_x_t,
                local_vae_pos_ids,
                local_vae_indexes,
                local_text_indexes,
                local_seqlens,
                local_position_ids,
            ) = self._split_vae_for_sp(
                x_t,
                packed_vae_position_ids,
                packed_vae_token_indexes,
                packed_text_indexes,
                packed_seqlens,
                packed_position_ids,
            )

            packed_text_embedding = self.language_model.forward(
                packed_text_ids=packed_text_ids,
                return_embeddings_only=True,
            ).packed_query_sequence
            packed_sequence = packed_text_embedding.new_zeros((int(local_seqlens.sum()), self.hidden_size))
            packed_sequence[local_text_indexes] = packed_text_embedding

            # i2v relaxes this: per-token timestep (cond=0, noncond=t) is valid.
            packed_pos_embed = self.latent_pos_embed(local_vae_pos_ids)
            local_timestep = timestep[: local_x_t.shape[0]]
            packed_timestep_embeds = self.time_embedder(local_timestep)
            x_t_emb = self.vae2llm(local_x_t) + packed_timestep_embeds + packed_pos_embed
            if x_t_emb.dtype != packed_sequence.dtype:
                x_t_emb = x_t_emb.to(packed_sequence.dtype)
            packed_sequence[local_vae_indexes] = x_t_emb

            extra_inputs = {}
            if self.use_moe:
                extra_inputs["mode"] = "gen"
                extra_inputs["packed_vae_token_indexes"] = local_vae_indexes
                extra_inputs["packed_text_indexes"] = local_text_indexes

            output = self.language_model.forward(
                packed_query_sequence=packed_sequence,
                query_lens=local_seqlens,
                packed_query_position_ids=local_position_ids,
                past_key_values=past_key_values,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )

            local_v_t = self.llm2vae(output.packed_query_sequence)
            local_v_t = local_v_t[local_vae_indexes]
            return self._gather_vae_for_sp(local_v_t)

        # Original non-SP path
        packed_text_embedding = self.language_model.forward(
            packed_text_ids=packed_text_ids,
            return_embeddings_only=True,
        ).packed_query_sequence
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # i2v relaxes this: per-token timestep (cond=0, noncond=t) is valid.
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t_emb = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t_emb.dtype != packed_sequence.dtype:
            x_t_emb = x_t_emb.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t_emb

        extra_inputs = {}
        if self.use_moe:
            extra_inputs["mode"] = "gen"
            extra_inputs["packed_vae_token_indexes"] = packed_vae_token_indexes
            extra_inputs["packed_text_indexes"] = packed_text_indexes

        output = self.language_model.forward(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            past_key_values=past_key_values,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]
        return v_t

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        past_key_values: NaiveCache,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_text_scale: float = 1.0,
        cfg_img_scale: float = 1.0,
        cfg_branch_pids: list[torch.Tensor] | None = None,
        cfg_branch_caches: list[NaiveCache] | None = None,
    ):
        # Build query sequence (identical for all CFG branches)
        packed_text_embedding = self.language_model.forward(
            packed_text_ids=packed_text_ids,
            return_embeddings_only=True,
        ).packed_query_sequence
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # i2v relaxes this: per-token timestep (cond=0, noncond=t) is valid.
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs["mode"] = "gen"
            extra_inputs["packed_vae_token_indexes"] = packed_vae_token_indexes
            extra_inputs["packed_text_indexes"] = packed_text_indexes

        use_cfg = cfg_text_scale > 1.0
        cfg_text_v_t = None
        cfg_img_v_t = None

        if use_cfg and cfg_branch_pids is not None and cfg_branch_caches is not None:
            num_branches = len(cfg_branch_pids)
            seq_len = int(packed_seqlens.sum())

            batched_sequence = packed_sequence.repeat(num_branches, 1)
            batched_vae_indexes = torch.cat([packed_vae_token_indexes + i * seq_len for i in range(num_branches)])
            batched_position_ids = torch.cat(cfg_branch_pids)
            batched_seqlens = packed_seqlens.repeat(num_branches)
            merged_cache = NaiveCache.merge(cfg_branch_caches)

            if self.use_moe:
                batched_text_indices = torch.cat([packed_text_indexes + i * seq_len for i in range(num_branches)])
                extra_inputs["packed_vae_token_indexes"] = batched_vae_indexes
                extra_inputs["packed_text_indexes"] = batched_text_indices

            output = self.language_model.forward(
                packed_query_sequence=batched_sequence,
                query_lens=batched_seqlens,
                packed_query_position_ids=batched_position_ids,
                past_key_values=merged_cache,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )

            all_vae_v_t = self.llm2vae(output.packed_query_sequence)[batched_vae_indexes]
            vae_per_branch = packed_vae_token_indexes.shape[0]
            branch_v_ts = all_vae_v_t.split(vae_per_branch)
            v_t = branch_v_ts[0]
            cfg_text_v_t = branch_v_ts[1]
            cfg_img_v_t = branch_v_ts[2] if len(branch_v_ts) > 2 else None
        else:
            # Single forward (no CFG or outside cfg_interval).
            output = self.language_model.forward(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=packed_position_ids,
                past_key_values=past_key_values,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            v_t = self.llm2vae(output.packed_query_sequence)[packed_vae_token_indexes]

        # ── CFG combination ──
        if use_cfg:
            v_t = self._combine_cfg(
                v_t,
                cfg_text_v_t,
                cfg_img_v_t,
                cfg_text_scale,
                cfg_img_scale,
                cfg_renorm_type,
                cfg_renorm_min,
            )

        return v_t

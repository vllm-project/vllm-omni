# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
# Copyright 2026 The Qwen team, Alibaba Group.
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3TTS Talker model compatible with HuggingFace weights."""

import bisect
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import ignore_torch_compile, support_torch_compile
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
)

logger = init_logger(__name__)


def _unsupported_encode_ref_audio_batch(*_args: Any, **_kwargs: Any) -> list[torch.Tensor]:
    """Placeholder for ``Qwen3TTSPromptEmbedsBuilder.encode_ref_audio_batch``.

    The NV talker variant only supports ``task_type="CustomVoice"``, which
    never invokes the ref-audio codec encoder. We must still pass *some*
    callable because the builder requires the keyword arg; raising here
    surfaces accidental use of a Base/voice-clone code path explicitly
    instead of failing later with a confusing ``'NoneType' is not callable``.
    """
    raise NotImplementedError(
        "Qwen3TTSTalkerForConditionalGenerationNv only supports task_type='CustomVoice'; "
        "ref-audio codec encoding (used by 'Base'/voice-clone) is not available."
    )


# ── RoPE helpers for the native code predictor ──────────────────────


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (standard RoPE helper)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply standard 1-D rotary position embeddings to Q and K.

    Args:
        q, k: [batch, num_heads, seq_len, head_dim]
        cos, sin: [1, 1, seq_len, head_dim]  (broadcastable)
    """
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3TTSNativeRotaryEmbedding(nn.Module):
    """Simple 1-D rotary position embedding for the native code predictor.

    Matches the ``Qwen3TTSRotaryEmbedding`` in the original HF code, but
    simplified: no dynamic-rope, no MRoPE – just standard RoPE with a
    configurable ``rope_theta``.
    """

    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # Use nn.Parameter so vLLM natively handles device/dtype casting.
        # requires_grad=False because this is deterministic and not trained.
        # The weight-loader already skips "rotary_emb.inv_freq".
        self.inv_freq = nn.Parameter(inv_freq, requires_grad=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` tensors for positions ``[0 .. seq_len)``.

        Returns:
            cos: [1, 1, seq_len, head_dim]
            sin: [1, 1, seq_len, head_dim]
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # [seq_len] x [head_dim/2] → [seq_len, head_dim/2]
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin


def _gumbel_sample(logits: torch.Tensor) -> torch.Tensor:
    """Gumbel-max trick: equivalent to categorical sampling.

    Uses only uniform RNG + log + argmax — all CUDA-graph safe.
    Unlike ``torch.multinomial``, this degrades gracefully on degenerate
    inputs (all-zero probs / all-``-inf`` logits) instead of triggering
    a device-side assert that poisons the CUDA context.  Also ~2.5x
    faster than multinomial in graph replay benchmarks.
    """
    u = torch.empty_like(logits).uniform_(1e-20, 1.0 - 1e-20)
    return (logits - torch.log(-torch.log(u))).argmax(dim=-1)


def _multinomial_sample(logits: torch.Tensor) -> torch.Tensor:
    """Standard softmax + multinomial sampling.

    CUDA-graph capturable on PyTorch >= 2.8, but will crash with a
    device-side assert if any row has all-zero probabilities (e.g.
    during graph warmup with uninitialised buffers).
    """
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def _sample_from_logits(
    logits: torch.Tensor,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    previous_tokens: torch.Tensor | None = None,
    use_gumbel: bool = True,
) -> torch.Tensor:
    """Sample tokens from logits (CUDA-graph safe).

    All operations are legal inside ``torch.cuda.graph()`` capture on
    PyTorch >= 2.8 (``topk``, ``sort``, ``multinomial``, ``uniform_``,
    ``argmax``, ``gather``, ``scatter_``, ``masked_fill``).

    The only patterns that remain **unsafe** during capture are
    host-to-device copies such as ``torch.tensor(scalar, device=cuda)``
    and ``torch.full_like(t, val)`` for some values — use
    ``masked_fill`` or pre-allocated buffers instead.

    Args:
        use_gumbel: If ``True`` (default), use the Gumbel-max trick for
            the final categorical draw.  Gumbel-max is ~2.5x faster
            than ``multinomial`` and robust to degenerate warmup data.
            Set ``False`` to use ``softmax → multinomial`` instead.
    """
    if repetition_penalty != 1.0 and previous_tokens is not None:
        score = torch.gather(logits, -1, previous_tokens)
        score = torch.where(
            score < 0,
            score * repetition_penalty,
            score / repetition_penalty,
        )
        logits.scatter_(-1, previous_tokens, score)

    if not do_sample:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-6)

    # ── Top-k filtering ─────────────────────────────────────────────
    if top_k is not None and top_k > 0:
        vals, idxs = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)

        # ── Top-p (nucleus) within the top-k slice ──────────────────
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_vals, sort_idx = torch.sort(vals, dim=-1, descending=True)
            probs = torch.softmax(sorted_vals, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)
            remove = (cum_probs - probs) > top_p
            sorted_vals = sorted_vals.masked_fill(remove, -1e10)
            # Unsort back to topk order
            unsort_idx = sort_idx.argsort(dim=-1)
            vals = sorted_vals.gather(-1, unsort_idx)

        sampled_in_k = _gumbel_sample(vals) if use_gumbel else _multinomial_sample(vals)
        return idxs.gather(-1, sampled_in_k.unsqueeze(-1)).squeeze(-1)

    # ── Top-p only (no top-k) ───────────────────────────────────────
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        remove = (cum_probs - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, -1e10)

        sampled_sorted = _gumbel_sample(sorted_logits) if use_gumbel else _multinomial_sample(sorted_logits)
        return sorted_indices.gather(-1, sampled_sorted.unsqueeze(-1)).squeeze(-1)

    # ── No filtering — sample from full distribution ────────────────
    if use_gumbel:
        return _gumbel_sample(logits)
    return _multinomial_sample(logits)


class Qwen3TTSTalkerResizeMLP(nn.Module):
    """Resize MLP for text projection in Qwen3TTS Talker.

    Maps from text_hidden_size to hidden_size with an intermediate layer.
    """

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(
            input_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
        )
        self.linear_fc2 = RowParallelLinear(
            intermediate_size,
            output_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
        )
        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_fc1(x)
        x = self.act_fn(x)
        x, _ = self.linear_fc2(x)
        return x


class Qwen3TTSNativeAttention(nn.Module):
    """Native attention for Qwen3TTS using torch SDPA.

    Used for the code predictor which has deterministic shapes and doesn't
    benefit from KV caching. Can be captured in CUDA graphs.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim else hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=qkv_bias)

        # QK normalization
        self.q_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass using torch SDPA.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_embeddings: Optional (cos, sin) tuple from rotary
                embedding, each [1, 1, seq_len, head_dim].
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, seq, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to [batch, num_heads, seq, head_dim] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary position embeddings (standard 1-D RoPE)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads if using GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Apply scaled dot product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,  # Use causal if no mask provided
            scale=self.scaling,
        )

        # Reshape back to [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        output = self.o_proj(attn_output)
        return output


class Qwen3TTSNativeMLP(nn.Module):
    """Native MLP for Qwen3TTS Code Predictor using standard PyTorch layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3TTSCodePredictorDecoderLayer(nn.Module):
    """Native decoder layer for Qwen3TTS Code Predictor.

    Uses native PyTorch attention (SDPA) instead of vLLM attention.
    This is more efficient for the code predictor since:
    - Shapes are deterministic (fixed 15 steps)
    - No KV cache benefit
    - Can be captured in CUDA graphs
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3TTSNativeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
        )

        self.mlp = Qwen3TTSNativeMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
        )

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Self Attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# Keys whose values must stay as plain dicts (expected by downstream code)
_KEEP_AS_DICT_KEYS = {"rope_scaling"}


def _dict_to_namespace(d, _key: str | None = None):
    """Recursively convert a dict to SimpleNamespace for attribute access.

    Certain keys (e.g. ``rope_scaling``) are kept as plain dicts because
    downstream code (``get_rope``, ``"mrope_section" in rope_scaling``, etc.)
    expects dict-like objects.
    """
    if isinstance(d, dict):
        if _key in _KEEP_AS_DICT_KEYS:
            return d  # keep as plain dict
        return SimpleNamespace(**{k: _dict_to_namespace(v, _key=k) for k, v in d.items()})
    return d


def _get_talker_config(hf_config: PretrainedConfig):
    """Get the talker config from either full TTS config or talker config directly.

    If talker_config is stored as a plain dict (from Qwen3TTSConfig),
    convert it to a namespace so attribute access (config.hidden_size etc.) works.
    """
    if hasattr(hf_config, "talker_config"):
        tc = hf_config.talker_config
        if isinstance(tc, dict):
            return _dict_to_namespace(tc)
        return tc
    # Otherwise assume this is already the talker config
    return hf_config


class Qwen3TTSTalkerCodePredictorModel(nn.Module):
    """Native PyTorch code predictor model for Qwen3TTS Talker.

    Uses native attention (SDPA) instead of vLLM attention since:
    - Runs for fixed 15 steps per global time step
    - Shapes are deterministic
    - No benefit from KV caching
    - Can be captured in CUDA graphs for efficiency
    """

    def __init__(self, config: PretrainedConfig, embedding_dim: int) -> None:
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_code_groups = config.num_code_groups

        # Codec embeddings for groups 1 to N-1 (group 0 uses main model embedding)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, embedding_dim) for _ in range(config.num_code_groups - 1)]
        )

        # Decoder layers using native attention
        self.layers = nn.ModuleList(
            [Qwen3TTSCodePredictorDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Standard 1-D rotary position embeddings (matches HF code predictor)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = Qwen3TTSNativeRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=getattr(config, "rope_theta", 1_000_000.0),
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        """Get codec embedding layers for all groups."""
        return self.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs_embeds: [batch_size, seq_len, hidden_size]
            attention_mask: Optional causal mask

        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        hidden_states = inputs_embeds

        # Compute position embeddings shared across all decoder layers.
        # Positions are simply [0, 1, ..., seq_len-1] since we
        # recompute from scratch each call (no KV cache).
        seq_len = hidden_states.shape[1]
        position_embeddings = self.rotary_emb(seq_len, hidden_states.device, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return hidden_states


@support_torch_compile
class Qwen3TTSTalkerCodePredictor(nn.Module):
    """Code predictor for Qwen3TTS Talker — groups 1..N-1 only.

    Given the previous step's backbone hidden state and the group-0 token
    (sampled by vLLM), autoregressively predicts codec groups 1 through
    N-1 using a small native-attention transformer.

    Group-0 prediction (``codec_head``, ``suppress_mask``) is handled by
    the outer model's ``compute_logits()`` + vLLM sampler.

    Also owns ``codec_embedding`` (group-0 codebook), shared with the
    outer model for input-embedding lookups.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        talker_config = _get_talker_config(hf_config)
        config = talker_config.code_predictor_config
        if isinstance(config, dict):
            config = _dict_to_namespace(config)
        quant_config = vllm_config.quant_config

        self.config = config
        self.num_code_groups = config.num_code_groups
        self.hidden_size = config.hidden_size
        self.talker_hidden_size = talker_config.hidden_size

        # Group-0 codec embedding (shared with outer model)
        self.codec_embedding = VocabParallelEmbedding(
            talker_config.vocab_size,
            talker_config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.codec_embedding",
        )

        # Code-predictor transformer backbone
        self.model = Qwen3TTSTalkerCodePredictorModel(config, self.talker_hidden_size)

        # Projection from talker hidden size to code predictor hidden size
        if config.hidden_size != self.talker_hidden_size:
            self.small_to_mtp_projection = nn.Linear(self.talker_hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # LM heads for each code group (1 to N-1)
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        # Sampling parameters for the internal groups-1..N-1 loop,
        # read from code_predictor_config. Fallback defaults match the
        # original HF implementation's subtalker_* arguments.
        self.do_sample = getattr(config, "do_sample", True)
        self.temperature = getattr(config, "temperature", 0.9)
        self.top_k = getattr(config, "top_k", 50)
        self.top_p = getattr(config, "top_p", 1.0)
        self.repetition_penalty = getattr(config, "repetition_penalty", 1.0)
        self.use_gumbel = getattr(config, "use_gumbel", True)

        # ── Persistent scratch buffers ──────────────────────────────────
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        N = config.num_code_groups
        hidden = talker_config.hidden_size
        cp_hidden = config.hidden_size
        dtype = vllm_config.model_config.dtype
        self._max_cp_len = 1 + N  # prev_hidden ctx + group0 + groups 1..N-1

        self._cp_inputs_embeds = torch.zeros(max_num_tokens, self._max_cp_len, hidden, dtype=dtype)
        self._cp_hidden_states = torch.empty(max_num_tokens, self._max_cp_len, cp_hidden, dtype=dtype)
        # Only groups 1..N-1 (N-1 columns)
        self._cp_all_codecs = torch.empty(max_num_tokens, N - 1, dtype=torch.long)

    def get_group0_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up group-0 codec embeddings."""
        return self.codec_embedding(input_ids)

    def get_group_embeddings(self) -> nn.ModuleList:
        """Get codec embedding layers for groups 1..N-1."""
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the code predictor transformer."""
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        hidden_states = self.model(inputs_embeds)
        return hidden_states

    def _compute_inner_logits(
        self,
        hidden_states: torch.Tensor,
        generation_step: int,
    ) -> torch.Tensor:
        """Compute logits for a specific inner code group (1..N-1)."""
        if generation_step >= len(self.lm_head):
            raise ValueError(f"generation_step {generation_step} exceeds number of code groups {len(self.lm_head)}")
        return self.lm_head[generation_step](hidden_states)

    def generate_groups_1_15(
        self,
        prev_hidden: torch.Tensor,
        group0_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Generate codec groups 1..N-1 given previous hidden state and group0.

        Args:
            prev_hidden: [seq_len, hidden_size] backbone output from previous step
            group0_tokens: [seq_len] group-0 tokens (from vLLM sampling)

        Returns:
            codes_1_15: [seq_len, num_code_groups - 1]
        """
        seq_len = prev_hidden.shape[0]
        N = self.num_code_groups

        inputs_embeds = self._cp_inputs_embeds[:seq_len]  # Batch x Books x Dim
        all_codecs = self._cp_all_codecs[:seq_len]

        inputs_embeds.zero_()

        # Position 0: previous backbone hidden state
        inputs_embeds[:, 0, :] = prev_hidden

        # Position 1: group-0 codec embedding
        inputs_embeds[:, 1, :] = self.codec_embedding(group0_tokens)

        for step in range(N - 1):
            # some how it is more efficient to re-run same graph for
            # bx16xdim input instead of capturing 15 graphs for different
            # input lengths
            hidden_states = self(inputs_embeds)

            current_len = step + 2
            logits = self._compute_inner_logits(hidden_states[:, current_len - 1, :], step)

            if self.repetition_penalty != 1.0 and step > 0:
                current_context = all_codecs[:, :step]
            else:
                current_context = None

            next_token = _sample_from_logits(
                logits,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                previous_tokens=current_context,
                use_gumbel=self.use_gumbel,
            )
            all_codecs[:, step] = next_token

            next_embed = self.get_group_embeddings()[step](next_token)
            inputs_embeds[:, current_len, :] = next_embed

        return all_codecs


@ignore_torch_compile
@support_torch_compile
class Qwen3TTSTalkerForConditionalGenerationNv(nn.Module):
    """Qwen3TTS Talker for conditional generation.

    Per-step flow:

    1. **Code predictor** (conditional): given the previous step's backbone
       hidden state (``prev_hidden``, custom input) and the group-0 token
       (``input_ids``, sampled by vLLM at the previous step), predict codec
       groups 1..N-1.  Skipped when ``prev_hidden`` is all-zero (prefill).
    2. **Embedding**: text_projection(text_embed) + codec_embed(group0)
       + sum of groups-1..N-1 embeddings from the code predictor.
    3. **Backbone**: transformer with vLLM paged attention and KV cache.
    4. **Logits**: ``compute_logits()`` projects backbone output through
       ``codec_head`` and applies ``suppress_mask``.  vLLM's standard
       sampler then samples the next group-0 token.

    Custom I/O:
      Inputs:  ``text_ids`` (int64), ``prev_hidden`` (float, dim=hidden_size)
      Outputs: ``codes`` (int64, dim=N-1), ``hidden`` (float, dim=hidden_size)
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # Maps HuggingFace checkpoint names (raw, unconverted) to the vLLM
    # module layout used in this file. Applied to weights with the
    # ``talker.`` prefix; ``speaker_encoder.*`` and other top-level
    # checkpoint sections are filtered out (the NV variant doesn't use
    # them). Order matters: more-specific prefixes first so that e.g.
    # ``talker.model.codec_embedding.`` is rerouted before the generic
    # ``talker.model.layers.`` rule could match.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Group-0 codec embedding lives inside the code predictor.
            "talker.model.codec_embedding.": "code_predictor.codec_embedding.",
            # Text embedding lifted to the outer model (vLLM's Qwen3Model
            # owns ``embed_tokens`` for codec ids; text tokens use a
            # separate top-level table).
            "talker.model.text_embedding.": "text_embedding.",
            # Talker backbone (transformer + final norm) — uses vLLM's
            # ``Qwen3Model`` directly, matching layer/norm names 1:1.
            "talker.model.layers.": "model.layers.",
            "talker.model.norm.": "model.norm.",
            # Side modules.
            "talker.codec_head.": "codec_head.",
            "talker.text_projection.": "text_projection.",
            # Code predictor (groups 1..N-1, native attention).
            "talker.code_predictor.": "code_predictor.",
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        config = _get_talker_config(hf_config)
        quant_config = vllm_config.quant_config

        self.hf_config = hf_config
        self.config = config
        self.quant_config = quant_config
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        # Omni preprocess/postprocess hooks (consumed by OmniGPUModelRunner).
        self.has_preprocess = True
        self.has_postprocess = True
        # Required so the runner unpacks ``multimodal_outputs`` (audio_codes)
        # from the ``OmniOutput`` returned by :meth:`make_omni_output`.
        # Without this, ``extract_multimodal_outputs`` discards the codes.
        self.have_multimodal_outputs = True
        # Keep small per-step buffers GPU-resident (avoids CPU round-trips).
        self.gpu_resident_buffer_keys: set[str] = {
            "last_talker_hidden",
        }

        # Transformer backbone — vLLM's reusable Qwen3Model. The talker
        # has Qwen3-style decoder layers, so we delegate the entire
        # backbone (decoder layers, final norm, and a per-rank
        # ``embed_tokens`` table that we do not actually consume — every
        # forward goes through ``inputs_embeds``).
        with set_model_tag("talker"):
            self.model = Qwen3Model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        # Text-token embedding lives outside the backbone (Qwen3Model only
        # owns the codec-vocab ``embed_tokens``).
        self.text_embedding = VocabParallelEmbedding(
            config.text_vocab_size,
            config.text_hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "text_embedding"),
        )

        # Text projection MLP
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            input_size=config.text_hidden_size,
            intermediate_size=config.text_hidden_size,
            output_size=config.hidden_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "text_projection"),
        )

        # Compiled code predictor (groups 1..N-1 only)
        with set_model_tag("code_predictor"):
            self.code_predictor = Qwen3TTSTalkerCodePredictor(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code_predictor"),
            )

        # Group-0 prediction head + suppress mask (used by compute_logits
        # so vLLM's standard sampler can sample group-0).
        self.codec_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "codec_head"),
        )

        self.suppress_mask = nn.Parameter(
            torch.zeros(config.vocab_size, dtype=torch.bool),
            requires_grad=False,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # Persistent buffers — addresses must be stable across CUDA graph
        # replays.  The piecewise CUDAGraphWrapper does NOT copy inputs on
        # replay; it expects the same ``data_ptr()`` that was recorded during
        # capture.  Any tensor created transiently in ``forward()`` (like
        # ``text_embed + codec_embed``) would have a new address each call,
        # causing the replayed graph to read stale memory.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        self._out_codes = torch.zeros(max_num_tokens, self.code_predictor.num_code_groups, dtype=torch.long)
        self._combined_embeddings = torch.zeros(max_num_tokens, config.hidden_size, dtype=dtype)
        # Per-token slot for the previous-step backbone hidden state fed to the
        # code predictor. Written by ``preprocess`` at the request's offset,
        # read by ``forward`` at decode positions.
        self._prev_hidden_buffer = torch.zeros(max_num_tokens, config.hidden_size, dtype=dtype)
        # ``text_proj(text_emb(tts_pad_token_id))`` — request-independent
        # constant added on top of ``codec_emb(group0)`` at every decode
        # step. Declared here so the address is stable across CUDA graph
        # replays; actual value is populated from weights in ``load_weights``.
        self._tts_pad_text_embed = torch.zeros(1, config.hidden_size, dtype=dtype)

        # Shared prefill-prompt builder. Wraps embedding tables + tokenizers
        # so the prompt-layout logic stays in one place (also used by the AR
        # talker variant). NV currently only supports ``task_type="CustomVoice"``,
        # so the Base-only dependencies (``speaker_encoder``, ``speaker_cache``,
        # ``residual_code_embeddings`` for ICL, and ``encode_ref_audio_batch``
        # for ref-audio codec extraction) are unused. ``encode_ref_audio_batch``
        # is a required keyword arg on the builder, so we pass a stub that
        # raises a clear error if any unsupported Base/voice-clone path ever
        # reaches it. ``_tts_pad_text_embed`` is passed by reference; its
        # contents are populated in ``_init_runtime_buffers`` once weights are
        # loaded and the builder reads it lazily on each ``build_prompt_embeds``
        # call.
        self._prompt_builder = Qwen3TTSPromptEmbedsBuilder(
            config=hf_config,
            talker_config=config,
            model_path=self.model_path,
            text_embedding=self.text_embedding,
            text_projection=self.text_projection,
            codec_embed=self.code_predictor.get_group0_embeddings,
            residual_code_embeddings=lambda: self.code_predictor.get_group_embeddings(),
            speaker_encoder=None,
            tts_pad_embed=self._tts_pad_text_embed,
            encode_ref_audio_batch=_unsupported_encode_ref_audio_batch,
            speaker_cache=None,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get group-0 codec embeddings for input ids."""
        return self.code_predictor.get_group0_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def _get_decode_idxs(self):
        """
        helper function that returns indices of decoding tokens,
        that's where exactly the local transformer should be
        applied.

        Returns:
            decode_idx: indices of decoder requests, if None returned,
                        local transformer should be applied everywhere
            num_requests: number of decoding requests, before padding
        """
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        if attn_metadata is None:
            # when attention metadata is not provided (capturing, dummy run)
            # then we should apply the local transformer everywhere
            return None, 0

        if isinstance(attn_metadata, dict):
            any_layer_meta = next(iter(attn_metadata.values()))
        else:
            any_layer_meta = attn_metadata

        if any_layer_meta.max_query_len == 1:
            # all requests in the batch a decode-only,
            # apply local transformer everywhere
            return None, 0

        start_loc = any_layer_meta.query_start_loc
        tokens_per_req = start_loc[1:] - start_loc[:-1]
        is_decode = tokens_per_req == 1  # shape: (num_reqs,)
        decode_token_indices = start_loc[:-1][is_decode]

        num_requests = decode_token_indices.shape[0]
        padded_num_requests = num_requests
        if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            idx = bisect.bisect_left(sizes, num_requests)
            if idx < len(sizes):
                padded_num_requests = sizes[idx]
        if padded_num_requests != num_requests:
            decode_token_indices = torch.nn.functional.pad(
                decode_token_indices, (0, padded_num_requests - num_requests)
            )
        return decode_token_indices, num_requests

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Forward pass: code predictor -> embedding -> backbone.

        ``inputs_embeds`` is produced by :meth:`preprocess`:

        * **Prefill**: full prefill embedding sequence for the span.
        * **Decode**: zeros — the actual decode embedding
          (``codec_emb(group0) + sum(group_emb(group1..N-1)) +
          text_proj(text_emb(tts_pad))``) is assembled here on decode
          positions only.

        ``prev_hidden`` (backbone output of the previous step) is read from
        :attr:`_prev_hidden_buffer`, which is populated by :meth:`preprocess`
        at each request's token offset.

        Three regimes:

        * **Profile / dummy run** (``attn_metadata is None``): treat every
          token as a decode token so the code predictor and decode-side
          embedding assembly get captured in the compiled CUDA graph.
        * **Decode-only batch**: every token is a decode token — the
          compiled / CUDA-graphed path replays directly.
        * **Mixed prefill + decode**: only decode-token positions go through
          the code predictor (eager); the assembled decode embeddings are
          scattered back into the combined-embedding buffer at those
          positions, leaving prefill positions as the prefill embeds.

        Returns:
            Backbone ``hidden_states`` tensor. The codec groups 1..N-1
            produced inside this forward live in :attr:`_out_codes` and
            are exposed to the runner via :meth:`make_omni_output` (key
            ``"audio_codes"``).
        """
        num_tokens = input_ids.shape[0]
        combined_embeddings = self._combined_embeddings[:num_tokens]
        combined_embeddings.copy_(inputs_embeds)

        decode_idx, num_req = self._get_decode_idxs()
        group_embeddings = self.code_predictor.get_group_embeddings()
        if decode_idx is None:
            codes_1_15 = self.code_predictor.generate_groups_1_15(
                prev_hidden=self._prev_hidden_buffer[:num_tokens],
                group0_tokens=input_ids,
            )
            self._out_codes[: codes_1_15.shape[0], 1:] = codes_1_15
            # Assemble decode embedding in-place on top of the (zero)
            # ``inputs_embeds`` produced by ``preprocess``: group-0 codec
            # embedding + tts_pad text embedding + sum of groups 1..N-1.
            combined_embeddings.add_(self.code_predictor.codec_embedding(input_ids))
            combined_embeddings.add_(self._tts_pad_text_embed)
            for i in range(len(group_embeddings)):
                combined_embeddings.add_(group_embeddings[i](codes_1_15[:, i]))
        elif num_req > 0:
            # need to overwrite the batch descriptor since we are slicing the inputs
            ctx = get_forward_context()
            orig_batch_descriptor = ctx.batch_descriptor
            ctx.batch_descriptor = BatchDescriptor(
                # padded number of requests
                num_tokens=decode_idx.shape[0],
            )
            codes_1_15 = self.code_predictor.generate_groups_1_15(
                prev_hidden=self._prev_hidden_buffer[decode_idx],
                group0_tokens=input_ids[decode_idx],
            )
            # restore original batch descriptor
            ctx.batch_descriptor = orig_batch_descriptor
            valid_dec_idx = decode_idx[:num_req]
            self._out_codes[valid_dec_idx, 1:] = codes_1_15[:num_req]
            # Assemble decode embedding only at decode positions; prefill
            # positions keep the full prefill embedding produced by
            # ``preprocess``.
            decode_group0_ids = input_ids[valid_dec_idx]
            decode_embed = self.code_predictor.codec_embedding(decode_group0_ids) + self._tts_pad_text_embed
            for i in range(len(group_embeddings)):
                decode_embed = decode_embed + group_embeddings[i](codes_1_15[:num_req, i])
            combined_embeddings[valid_dec_idx] = decode_embed

        # Qwen3Model.forward(input_ids, positions, intermediate_tensors,
        # inputs_embeds): when ``inputs_embeds`` is provided, ``input_ids``
        # is ignored and the embedded sequence is fed directly into the
        # decoder layers.
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            combined_embeddings,
        )

        # save input ids to the output codes
        self._out_codes[: input_ids.shape[0], 0] = input_ids

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute group-0 logits for vLLM sampling.

        Projects backbone hidden states through ``codec_head``, applies the
        ``suppress_mask`` to block reserved token IDs, and returns logits
        of shape ``[batch, vocab_size]``.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        logits = self.logits_processor(self.codec_head, hidden_states)
        logits = logits.masked_fill(self.suppress_mask.bool(), float("-inf"))
        return logits

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **_: Any,
    ) -> OmniOutput:
        """Wrap backbone hidden states with the codec groups 1..N-1.

        The codes produced inside :meth:`forward` live in :attr:`_out_codes`;
        we slice the first ``num_tokens`` rows here and expose them under
        the conventional ``"audio_codes"`` multimodal key consumed by
        :class:`OmniGPUModelRunner`.

        ``last_talker_hidden`` (state needed by the *next* step's code
        predictor) is **not** part of the omni output — it is stashed into
        ``model_intermediate_buffer`` by :meth:`postprocess` and read back
        by :meth:`preprocess` on the next decode step.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        num_tokens = int(hidden.shape[0])
        audio_codes = self._out_codes[:num_tokens]
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"audio_codes": audio_codes},
        )

    # ------------------------------------------------------------------
    # Preprocess / postprocess (CustomVoice, non-streaming text only)
    # ------------------------------------------------------------------

    @staticmethod
    def _first_str(value: Any) -> str:
        """Return the first element of a list-wrapped scalar, or the scalar itself."""
        if isinstance(value, list):
            return str(value[0]) if value else ""
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _build_assistant_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        *,
        start: int = 0,
        end: int = 0,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build per-request ``(input_ids, inputs_embeds)`` for this step.

        CustomVoice + non-streaming only.

        Args:
            input_ids: This request's slice of the flat batch's token ids.
            input_embeds: Corresponding slice of the flat-batch
                ``inputs_embeds`` if the runner already populated one.
            start: This request's start position in the flat batch
                (``query_start_loc[req_index]``). Provided by the runner;
                used to index :attr:`_prev_hidden_buffer` at decode.
            end: This request's end position in the flat batch
                (``start + sched_tokens``). Provided by the runner; not
                used here directly (``input_ids.shape[0] == end - start``)
                but accepted for runner-contract symmetry.
            **info_dict: The request's ``additional_information`` plus
                runner-provided ``request_id`` and any state previously
                stashed by this method (e.g. ``talker_prompt_embeds``,
                ``talker_prefill_offset``, ``last_talker_hidden``).

        Prefill (``span_len > 1``):
            On the first prefill call, builds the full prompt embedding once
            (see :meth:`_build_prompt_embeds`) and stashes it under
            ``talker_prompt_embeds`` (CPU). On subsequent chunks, slices from
            that buffer using ``talker_prefill_offset``. ``input_ids`` are
            filled with ``codec_pad`` placeholders since the code predictor
            doesn't run during prefill.

        Decode (``span_len == 1``):
            Returns ``inputs_embeds`` of zeros — the actual decode
            embedding (``codec_emb(group0) + sum(group_emb(group1..N-1)) +
            text_proj(text_emb(tts_pad))``) is assembled inside
            :meth:`forward` at decode positions only. ``input_ids`` (the
            group-0 token sampled by vLLM) is passed through unchanged.
            The previous-step backbone hidden (``last_talker_hidden``,
            produced by :meth:`postprocess`) is written into
            :attr:`_prev_hidden_buffer` at ``start`` for the code predictor
            to read.
        """
        # Normalize: some runner paths still pass per-request state nested
        # under ``additional_information`` instead of flattened.
        nested = info_dict.get("additional_information")
        if isinstance(nested, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in nested.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            base = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, base, {}

        tc = self.config
        device = input_ids.device

        # ----- Prefill -------------------------------------------------
        if span_len > 1:
            text = self._first_str(info_dict.get("text"))
            if not text:
                raise ValueError("Qwen3-TTS NV talker.preprocess requires additional_information.text for prefill.")
            speaker = self._first_str(info_dict.get("speaker"))
            if not speaker:
                raise ValueError(
                    "Qwen3-TTS NV talker.preprocess requires additional_information.speaker (CustomVoice only)."
                )
            language = self._first_str(info_dict.get("language")) or "Auto"

            prompt_embeds_cpu = info_dict.get("talker_prompt_embeds")
            is_first = not isinstance(prompt_embeds_cpu, torch.Tensor) or prompt_embeds_cpu.ndim != 2
            if is_first:
                # The shared builder accesses ``info_dict[k][0]``, so always
                # pass list-wrapped scalars regardless of what the runner
                # provided (NV's ``_first_str`` already validated them above).
                builder_info = {
                    **info_dict,
                    "text": [text],
                    "speaker": [speaker],
                    "language": [language],
                }
                talker_prompt, _, _, _ = self._prompt_builder.build_prompt_embeds(
                    task_type="CustomVoice",
                    info_dict=builder_info,
                )
                prompt_embeds_cpu = talker_prompt.to(dtype=torch.bfloat16).detach().to("cpu").contiguous()
                offset = 0
                info_update: dict[str, Any] = {
                    "talker_prompt_embeds": prompt_embeds_cpu,
                    "talker_prefill_offset": 0,
                }
            else:
                offset = int(info_dict.get("talker_prefill_offset", 0) or 0)
                info_update = {}

            # Slice the span out of the stored prefill buffer; pad with the
            # last row if the scheduled chunk overshoots (shouldn't happen
            # when the placeholder length matches the true prefill length).
            s = max(0, min(offset, int(prompt_embeds_cpu.shape[0])))
            e = max(0, min(offset + span_len, int(prompt_embeds_cpu.shape[0])))
            take = prompt_embeds_cpu[s:e]
            if int(take.shape[0]) < span_len:
                pad_n = span_len - int(take.shape[0])
                if take.shape[0] > 0:
                    pad_rows = take[-1:].expand(pad_n, -1)
                else:
                    pad_rows = torch.zeros(
                        (pad_n, prompt_embeds_cpu.shape[-1]),
                        dtype=prompt_embeds_cpu.dtype,
                    )
                take = torch.cat([take, pad_rows], dim=0)
            prompt_embeds = take.to(device=device, dtype=torch.bfloat16)
            info_update["talker_prefill_offset"] = offset + span_len

            # input_ids for prefill: codec_pad placeholder (code predictor
            # is skipped for prefill positions, so the exact value doesn't
            # matter as long as it's a valid codec token).
            input_ids_out = torch.full_like(input_ids, int(tc.codec_pad_id))
            return input_ids_out, prompt_embeds, info_update

        # ----- Decode (span_len == 1) ---------------------------------
        # The decode embedding is assembled inside :meth:`forward` (where
        # we have visibility of decode-vs-prefill positions and the codes
        # produced by the code predictor). Here we just return zeros that
        # ``forward`` will accumulate the real embedding into.
        inputs_embeds_out = torch.zeros(
            (1, self.config.hidden_size),
            device=device,
            dtype=self._combined_embeddings.dtype,
        )

        # prev_hidden for the code predictor: ``last_talker_hidden`` stashed
        # by postprocess. When missing, we leave the slot untouched (first
        # decode step after prefill will have it available since postprocess
        # runs after every forward, including prefill).
        last_hidden = info_dict.get("last_talker_hidden")
        if isinstance(last_hidden, torch.Tensor) and last_hidden.numel() > 0:
            prev_h = last_hidden.to(device=device, dtype=self._prev_hidden_buffer.dtype).reshape(1, -1)
            self._prev_hidden_buffer[start : start + 1].copy_(prev_h)

        return input_ids, inputs_embeds_out, {}

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        """Stash the last backbone hidden as ``last_talker_hidden`` for the next step."""
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach()
        return {"last_talker_hidden": last}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights directly from a raw HuggingFace Qwen3-TTS checkpoint.

        No offline conversion is required: this method renames the
        ``talker.*`` weights to the vLLM module layout (see
        :attr:`hf_to_vllm_mapper`), drops unrelated checkpoint sections
        (``speaker_encoder.*``, etc.), and then computes the two
        derived buffers that the runtime needs:

        * :attr:`suppress_mask`: a boolean ``[vocab_size]`` mask that
          blocks the top-1024 reserved token IDs (except
          ``codec_eos_token_id``) when sampling group-0.
        * :attr:`_tts_pad_text_embed`: the projected text embedding of
          ``tts_pad_token_id`` — a request-independent constant added
          on top of ``codec_emb(group0)`` at every decode step.
        """
        # Filter to talker weights only (skip speaker_encoder.* etc).
        talker_weights = ((name, w) for name, w in weights if name.startswith("talker."))

        # ``suppress_mask`` is a Parameter we initialise ourselves below;
        # if a converted checkpoint happens to carry one, ignore it.
        loader = AutoWeightsLoader(self, skip_prefixes=["suppress_mask"])
        loaded = loader.load_weights(talker_weights, mapper=self.hf_to_vllm_mapper)

        self._init_runtime_buffers()

        # Mark the parameters we initialise without a checkpoint weight
        # as "loaded" so the strict-loading check in
        # ``DefaultModelLoader.load_weights`` doesn't flag them. These
        # are populated either in ``__init__`` (rotary inv_freq) or in
        # ``_init_runtime_buffers`` (suppress_mask).
        #
        # ``model.embed_tokens`` is created by ``Qwen3Model`` but is never
        # invoked in this model — every prefill / decode step feeds the
        # backbone via ``inputs_embeds`` assembled from
        # ``code_predictor.codec_embedding`` and ``text_embedding``.
        # Skip it from the strict-load check.
        loaded.add("suppress_mask")
        loaded.add("model.embed_tokens.weight")
        for name, _ in self.named_parameters():
            if name.endswith("rotary_emb.inv_freq"):
                loaded.add(name)

        logger.info(
            "Loaded %d weights for Qwen3TTSTalkerForConditionalGenerationNv",
            len(loaded),
        )
        return loaded

    @torch.no_grad()
    def _init_runtime_buffers(self) -> None:
        """Populate :attr:`suppress_mask` and :attr:`_tts_pad_text_embed`.

        Called from :meth:`load_weights` once the underlying parameters
        have been filled, so :meth:`text_projection` and
        :meth:`model.get_text_embeddings` can be evaluated to derive the
        constant ``tts_pad`` embedding used on every decode step.
        """
        tc = self.config
        hf = self.hf_config

        # Top-1024 token IDs are reserved/invalid; suppress them at
        # group-0 sampling time, except for ``codec_eos`` which must
        # remain reachable as an end-of-stream signal.
        vocab_size = int(tc.vocab_size)
        codec_eos = int(getattr(tc, "codec_eos_token_id", -1))
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=self.suppress_mask.device)
        suppress_start = vocab_size - 1024
        if suppress_start > 0:
            mask[suppress_start:] = True
            if suppress_start <= codec_eos < vocab_size:
                mask[codec_eos] = False
        self.suppress_mask.copy_(mask)

        # Precompute ``text_proj(text_emb(tts_pad_token_id))`` — added
        # to ``codec_emb(group0)`` at every decode step; depends only on
        # frozen weights so we evaluate it once here.
        device = next(self.parameters()).device
        pad_id = int(hf.tts_pad_token_id)
        pad_ids = torch.tensor([[pad_id]], device=device, dtype=torch.long)
        text_emb = self.text_embedding(pad_ids)
        pad_proj = self.text_projection(text_emb).reshape(1, -1)
        self._tts_pad_text_embed.copy_(
            pad_proj.to(
                device=self._tts_pad_text_embed.device,
                dtype=self._tts_pad_text_embed.dtype,
            )
        )

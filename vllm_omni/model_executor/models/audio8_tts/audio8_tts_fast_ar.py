# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Fast AR: the residual-codebook predictor.

Runs once per Slow AR step over ``num_codebooks + 1`` positions: position 0 is
the projected Slow AR hidden state (primes the KV cache, logits discarded),
position 1 is the sampled semantic code (not predicted, only fed in), positions
2.. are the previously sampled residual codes; only codes 1..num_codebooks-1
are sampled here.

Not paged: at most 10 tokens per call, SDPA over a per-call pre-allocated KV
cache. Matches the reference ``ArkttsModel._generate_codebooks``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .configuration_audio8_tts import Audio8TTSFastARConfig, Audio8TTSSlowARConfig
from .sampling import sample_scores

logger = init_logger(__name__)


class _FastARAttention(nn.Module):
    """GQA self-attention over the short codebook sequence.

    ``forward_one`` decodes a single position against a caller-owned KV cache;
    there is no batched-prefill path because the codebook loop is inherently
    sequential.
    """

    def __init__(self, config: Audio8TTSFastARConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self._use_gqa = self.num_kv_heads != self.num_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=config.attention_qkv_bias,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            # Audio8 TTS uses interleaved (GPT-J) RoPE, not NeoX.
            is_neox_style=False,
            rope_parameters={"rope_theta": config.rope_theta, "rope_type": "default"},
        )
        if config.attention_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward_one(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        bsz = int(hidden_states.shape[0])

        qkv, _ = self.qkv_proj(hidden_states.reshape(bsz, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.q_norm is not None:
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)

        q, k = self.rotary_emb(position_ids.reshape(-1), q, k)

        q = q.view(bsz, self.num_heads, self.head_dim).unsqueeze(2)
        k_cache[:bsz, :, cache_pos, :] = k.view(bsz, self.num_kv_heads, self.head_dim)
        v_cache[:bsz, :, cache_pos, :] = v.view(bsz, self.num_kv_heads, self.head_dim)

        attn_out = F.scaled_dot_product_attention(
            q,
            k_cache[:bsz, :, : cache_pos + 1, :],
            v_cache[:bsz, :, : cache_pos + 1, :],
            scale=self.scaling,
            is_causal=False,
            enable_gqa=self._use_gqa,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, -1)
        output, _ = self.o_proj(attn_out)
        return output


class _FastARMLP(nn.Module):
    """SiLU-gated MLP (``w1``/``w3`` gate-up, ``w2`` down)."""

    def __init__(self, config: Audio8TTSFastARConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x, _ = self.down_proj(F.silu(gate) * up)
        return x


class _FastARDecoderLayer(nn.Module):
    def __init__(self, config: Audio8TTSFastARConfig, *, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = _FastARAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = _FastARMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_one(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_pos: int,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn.forward_one(
            self.input_layernorm(hidden_states), position_ids, k_cache, v_cache, cache_pos
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Audio8TTSFastAR(nn.Module):
    """Residual-codebook predictor for Audio8 TTS.

    ``forward`` returns all ``num_codebooks`` codes for one frame, with code 0
    copied from the already-sampled semantic token.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Audio8TTSFastARConfig,
        slow_ar_config: Audio8TTSSlowARConfig,
        prefix: str = "fast_ar",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = config
        self.slow_ar_config = slow_ar_config

        self.layers = nn.ModuleList(
            [_FastARDecoderLayer(config, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)]
        )
        self.fast_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fast_output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.fast_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if slow_ar_config.hidden_size != config.hidden_size:
            self.fast_project_in = nn.Linear(slow_ar_config.hidden_size, config.hidden_size, bias=True)
        else:
            # 0.6b: dim == fast_dim == 896, and the checkpoint has no
            # fast_project_in tensor.
            self.fast_project_in = nn.Identity()

        self._num_codebooks = int(config.num_codebooks)
        self._fast_dim = int(config.hidden_size)
        self._k_cache: torch.Tensor | None = None
        self._v_cache: torch.Tensor | None = None
        self._pos_ids: torch.Tensor | None = None

    def _ensure_buffers(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        max_seq = self._num_codebooks + 1
        if (
            self._k_cache is not None
            and self._pos_ids is not None
            and self._k_cache.shape[1] >= bsz
            and self._k_cache.device == device
            and self._k_cache.dtype == dtype
            and self._pos_ids.shape[0] >= bsz
        ):
            return
        self._k_cache = torch.empty(
            self.config.num_hidden_layers,
            bsz,
            self.config.num_key_value_heads,
            max_seq,
            self.config.head_dim,
            dtype=dtype,
            device=device,
        )
        self._v_cache = torch.empty_like(self._k_cache)
        self._pos_ids = torch.arange(max_seq, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1).contiguous()

    def _forward_one(self, embed: torch.Tensor, position_ids: torch.Tensor, cache_pos: int) -> torch.Tensor:
        assert self._k_cache is not None and self._v_cache is not None
        hidden = embed
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer.forward_one(
                hidden,
                position_ids,
                self._k_cache[layer_idx],
                self._v_cache[layer_idx],
                cache_pos,
            )
        return hidden

    @torch.inference_mode()
    def forward(
        self,
        slow_ar_hidden: torch.Tensor,
        semantic_token_id: torch.Tensor,
        *,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Predict one frame of codec codes.

        ``slow_ar_hidden`` must already be post-final-norm when
        ``norm_fastlayer_input`` is set (it is for the released checkpoint) --
        vLLM's Qwen2Model returns exactly that. Returns ``[B, num_codebooks]``
        int64 codes; code 0 is the semantic code, codes 1.. come from this module.
        """
        bsz = int(slow_ar_hidden.shape[0])
        num_cb = self._num_codebooks
        device = slow_ar_hidden.device
        dtype = slow_ar_hidden.dtype

        semantic_begin = int(self.slow_ar_config.semantic_begin_id)
        codebook_size = int(self.slow_ar_config.codebook_size)
        # Non-semantic tokens (EOS) clamp to code 0; those frames are dropped
        # downstream because the request finishes on that token.
        semantic_code = (semantic_token_id.reshape(bsz) - semantic_begin).clamp(min=0, max=codebook_size - 1)

        codes = torch.empty(bsz, num_cb, dtype=torch.long, device=device)
        codes[:, 0] = semantic_code

        self._ensure_buffers(bsz, device, dtype)
        pos_ids = self._pos_ids
        assert pos_ids is not None

        # Position 0 primes the cache; its logits are unused.
        projected = self.fast_project_in(slow_ar_hidden.reshape(bsz, -1))
        self._forward_one(projected, pos_ids[:bsz, 0], 0)

        embed = self.fast_embeddings(semantic_code)
        for step in range(1, num_cb):
            hidden = self._forward_one(embed, pos_ids[:bsz, step], step)
            logits = self.fast_output(self.fast_norm(hidden)).float()
            next_ids = sample_scores(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                generator=generator,
            )
            codes[:, step] = next_ids
            if step < num_cb - 1:
                embed = self.fast_embeddings(next_ids)

        return codes

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load already-remapped Fast AR weights (see the Slow AR remapper)."""
        with set_current_vllm_config(self._vllm_config):
            params_dict = dict(self.named_parameters(remove_duplicate=False))
            loaded: set[str] = set()
            stacked_params_mapping = [
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]
            for name, loaded_weight in weights:
                handled = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    if mapped not in params_dict:
                        continue
                    param = params_dict[mapped]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    if weight_loader == default_weight_loader:
                        weight_loader(param, loaded_weight)
                    else:
                        weight_loader(param, loaded_weight, shard_id)
                    loaded.add(mapped)
                    handled = True
                    break
                if handled or name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)
            return loaded


__all__ = ["Audio8TTSFastAR"]

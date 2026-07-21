# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-native SenseNova-U1 AR model used by AR -> DiT separation."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    SenseNovaU1MLP,
    rotate_half,
)

TEXT_ROPE_THETA = 5000000.0
IMAGE_ROPE_THETA = 10000.0
MAX_TEXT_POSITION_EMBEDDINGS = 262144
MAX_IMAGE_POSITION_EMBEDDINGS = 10000


def _apply_rotary_flat(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[0].unsqueeze(1)
    sin = sin[0].unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def _clone_rope_config(
    config,
    *,
    head_dim: int,
    rope_theta: float,
    max_position_embeddings: int,
):
    cloned = config.__class__(**config.to_dict())
    cloned.head_dim = head_dim
    cloned.rope_theta = rope_theta
    cloned.max_position_embeddings = max_position_embeddings
    return cloned


def _normalize_language_weight_name(name: str) -> str | None:
    if name.startswith("language_model."):
        return name
    if name.startswith("model.language_model."):
        return name.removeprefix("model.")
    return None


class SenseNovaU1TextAttention(nn.Module):
    """SenseNova text attention with split 3D RoPE and vLLM paged KV cache."""

    def __init__(
        self,
        config,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        bias = getattr(config, "attention_bias", True)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.q_norm = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_hw = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)

        rope_theta = getattr(config, "rope_theta", TEXT_ROPE_THETA)
        rope_theta_hw = getattr(config, "rope_theta_hw", IMAGE_ROPE_THETA)
        max_position_embeddings = getattr(config, "max_position_embeddings", MAX_TEXT_POSITION_EMBEDDINGS)
        max_position_embeddings_hw = getattr(config, "max_position_embeddings_hw", MAX_IMAGE_POSITION_EMBEDDINGS)

        t_config = _clone_rope_config(
            config,
            head_dim=self.head_dim // 2,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config=t_config)

        hw_config = _clone_rope_config(
            config,
            head_dim=self.head_dim // 4,
            rope_theta=rope_theta_hw,
            max_position_embeddings=max_position_embeddings_hw,
        )
        self.rotary_emb_hw = Qwen3RotaryEmbedding(config=hw_config)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def _project_and_rope_text(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        num_tokens = hidden_states.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        q_t, q_hw = q.chunk(2, dim=-1)
        k_t, k_hw = k.chunk(2, dim=-1)
        q_t = self.q_norm(q_t)
        k_t = self.k_norm(k_t)
        q_hw = self.q_norm_hw(q_hw)
        k_hw = self.k_norm_hw(k_hw)

        q_h, q_w = q_hw.chunk(2, dim=-1)
        k_h, k_w = k_hw.chunk(2, dim=-1)

        t_positions = positions.unsqueeze(0)
        zero_positions = torch.zeros_like(positions).unsqueeze(0)
        cos_t, sin_t = self.rotary_emb(hidden_states, t_positions)
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, zero_positions)
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, zero_positions)

        q_t = _apply_rotary_flat(q_t, cos_t, sin_t)
        k_t = _apply_rotary_flat(k_t, cos_t, sin_t)
        q_h = _apply_rotary_flat(q_h, cos_h, sin_h)
        k_h = _apply_rotary_flat(k_h, cos_h, sin_h)
        q_w = _apply_rotary_flat(q_w, cos_w, sin_w)
        k_w = _apply_rotary_flat(k_w, cos_w, sin_w)

        q = torch.cat([q_t, q_h, q_w], dim=-1).reshape(num_tokens, self.q_size)
        k = torch.cat([k_t, k_h, k_w], dim=-1).reshape(num_tokens, self.kv_size)
        v = v.reshape(num_tokens, self.kv_size)
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self._project_and_rope_text(positions, hidden_states)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class SenseNovaU1TextDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = SenseNovaU1TextAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = SenseNovaU1MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SenseNovaU1TextModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        cache_config = vllm_config.cache_config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SenseNovaU1TextDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=vllm_config.quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"],
            config.hidden_size,
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states


class SenseNovaU1TextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = SenseNovaU1TextModel(vllm_config=vllm_config, config=config, prefix=f"{prefix}.model")

        if get_pp_group().is_last_rank:
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=f"{prefix}.lm_head",
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)


class OmniSenseNovaU1ForCausalLM(nn.Module, SupportsPP):
    """SenseNova-U1 text-prefix model that emits DiT-compatible paged KV."""

    packed_modules_mapping = SenseNovaU1TextForCausalLM.packed_modules_mapping

    _SKIP_LANGUAGE_MODEL_PARTS = (
        "_mot_gen",
        "vision_model.",
        "fm_modules.",
        "vae.",
        "t_embedder.",
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config or vllm_config.model_config.hf_config
        self.language_model = SenseNovaU1TextForCausalLM(
            vllm_config=vllm_config,
            config=config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.language_model.sample(logits, sampling_metadata)

    @property
    def sampler(self) -> Sampler:
        return self.language_model.sampler

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        )
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        skip_lm_head = getattr(self.language_model.config, "tie_word_embeddings", False)

        for name, loaded_weight in weights:
            mapped_name = _normalize_language_weight_name(name)
            if mapped_name is None:
                continue

            if any(part in mapped_name for part in self._SKIP_LANGUAGE_MODEL_PARTS):
                continue
            if skip_lm_head and mapped_name.startswith("language_model.lm_head."):
                continue
            if "rotary_emb.inv_freq" in mapped_name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped_name:
                    continue
                mapped_name = mapped_name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    break
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)

        return loaded_params

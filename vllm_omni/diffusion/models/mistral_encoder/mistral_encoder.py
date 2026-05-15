# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TP-aware Mistral model for use as a text encoder in diffusion pipelines.

Follows the same pattern as T5EncoderModel: uses vLLM's parallel linear layers
for tensor parallelism but simple scaled_dot_product_attention instead of
PagedAttention, so it can be used as a standalone encoder without VllmConfig.

The model supports autoregressive text generation via ``generate()``
using KV caching and the tied embedding weights as a language-model head.
This replaces the dependency on ``Mistral3ForConditionalGeneration`` for
caption upsampling.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)


class MistralRotaryEmbedding(nn.Module):
    """RoPE implementation for the encoder. Precomputes cos/sin tables."""

    def __init__(self, head_dim: int, max_position_embeddings: int, rope_theta: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        # position_ids: (batch, seq_len)
        inv_freq = self.inv_freq[None, :, None].float().to(position_ids.device)
        inv_freq = inv_freq.expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)  # (batch, seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _format_upsample_input(
    prompts: list[str],
    system_message: str,
    images: list | None = None,
) -> list[list[dict[str, Any]]]:
    cleaned_txt = [p.replace("[IMG]", "") for p in prompts]

    if images is None or len(images) == 0:
        return [
            [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]

    assert len(images) == len(prompts), "Number of images must match number of prompts"
    messages = [[{"role": "system", "content": [{"type": "text", "text": system_message}]}] for _ in cleaned_txt]
    for i, (el, batch_images) in enumerate(zip(messages, images)):
        if batch_images is not None:
            el.append({"role": "user", "content": [{"type": "image", "image": img} for img in batch_images]})
        el.append({"role": "user", "content": [{"type": "text", "text": cleaned_txt[i]}]})
    return messages


class MistralEncoderAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=self.scaling)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output, _ = self.o_proj(attn_output)
        return attn_output, new_kv


class MistralEncoderMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class MistralEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = MistralEncoderAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MistralEncoderMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states,
            cos,
            sin,
            attention_mask,
            past_key_value,
            use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv


class MistralEncoderOutput:
    """Simple output container matching HuggingFace's interface."""

    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values


class MistralEncoderModel(nn.Module):
    """
    TP-aware Mistral encoder for use as a text encoder in diffusion pipelines.

    Accepts a HuggingFace Mistral3Config (or its text_config). Uses vLLM
    parallel layers for TP but simple SDPA for attention (no PagedAttention).
    """

    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        self._processor = None
        self._system_message_t2i: str | None = None
        self._system_message_i2i: str | None = None
        # Handle Mistral3Config (has text_config) or plain MistralConfig
        if hasattr(config, "text_config"):
            text_config = config.text_config
        else:
            text_config = config
        self.config = text_config

        self.hidden_size = text_config.hidden_size
        self.num_heads = text_config.num_attention_heads
        self.num_kv_heads = getattr(text_config, "num_key_value_heads", text_config.num_attention_heads)
        self.head_dim = getattr(text_config, "head_dim", None) or (self.hidden_size // self.num_heads)
        self.intermediate_size = text_config.intermediate_size
        self.num_layers = text_config.num_hidden_layers
        self.rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-5)
        self.max_position_embeddings = getattr(text_config, "max_position_embeddings", 131072)
        self.rope_theta = getattr(text_config, "rope_theta", 1000000.0)
        self.vocab_size = text_config.vocab_size

        tp_size = get_tensor_model_parallel_world_size()
        logger.info(
            "MistralEncoderModel init: hidden_size=%d, num_heads=%d, "
            "num_kv_heads=%d, head_dim=%d, num_layers=%d, tp_size=%d",
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.num_layers,
            tp_size,
        )

        # Nest modules to match HF checkpoint hierarchy:
        #   language_model.model.embed_tokens
        #   language_model.model.layers.X...
        #   language_model.model.norm
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        m = self.language_model.model

        m.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)

        m.layers = nn.ModuleList(
            [
                MistralEncoderLayer(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    intermediate_size=self.intermediate_size,
                    rms_norm_eps=self.rms_norm_eps,
                    prefix=f"language_model.model.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        m.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        m.rotary_emb = MistralRotaryEmbedding(self.head_dim, self.max_position_embeddings, self.rope_theta)

        self.language_model.lm_head = ParallelLMHead(self.vocab_size, self.hidden_size, bias=False)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        **kwargs,
    ) -> MistralEncoderOutput:
        m = self.language_model.model
        hidden_states = m.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]

        # Determine position offset from cached KV length
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        total_len = past_len + seq_len

        # Compute position_ids from attention_mask so padded tokens get position 0
        # and real tokens get contiguous positions starting from 0.
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.clamp_(min=0)
            position_ids = position_ids[:, -seq_len:]
        else:
            position_ids = torch.arange(past_len, past_len + seq_len, device=hidden_states.device).unsqueeze(0)

        cos, sin = m.rotary_emb(position_ids, hidden_states.dtype)

        # Build causal attention mask combined with padding mask for SDPA.
        # Mistral is a decoder-only model, so hidden states are computed with
        # causal (autoregressive) attention even when used as an encoder.
        min_val = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.triu(
            torch.full((seq_len, total_len), min_val, device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=past_len + 1,
        )
        # (seq_len, total_len) -> (1, 1, seq_len, total_len)
        sdpa_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # Combine with padding mask: (batch, 1, 1, total_len)
            padding_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            padding_mask = (1.0 - padding_mask) * min_val
            sdpa_mask = sdpa_mask + padding_mask

        all_hidden_states = () if output_hidden_states else None
        new_key_values = [] if use_cache else None

        for i, layer in enumerate(m.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, layer_kv = layer(
                hidden_states,
                cos,
                sin,
                sdpa_mask,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_key_values.append(layer_kv)

        hidden_states = m.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return MistralEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            past_key_values=new_key_values,
        )

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute full-vocab logits from hidden states using the lm_head weight."""
        local_logits = F.linear(
            hidden_states,
            self.language_model.lm_head.weight,
        )
        if get_tensor_model_parallel_world_size() > 1:
            return tensor_model_parallel_all_gather(local_logits)
        return local_logits

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
    ) -> torch.Tensor:
        """Sample or greedily select the next token from logits. Returns (batch, 1)."""
        if do_sample:
            logits = logits / max(temperature, 1e-7)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return logits.argmax(dim=-1, keepdim=True)

    # TODO make common. Potentially with HF mixin
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 1.0,
        eos_token_id: int | list[int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive text generation with KV caching.

        Accepts the same keyword arguments as the HuggingFace
        ``GenerationMixin.generate`` interface used by the pipeline
        (``pixel_values`` etc. are accepted and ignored).

        Returns the full token sequence including the input prompt.
        """
        eos_token_id = eos_token_id or getattr(self.config, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_token_ids = {eos_token_id}
        elif isinstance(eos_token_id, list):
            eos_token_ids = set(eos_token_id)
        else:
            eos_token_ids = set()

        batch_size = input_ids.shape[0]
        device = input_ids.device
        generated = input_ids

        # Prefill ----------------------------------------------------------------
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = output.past_key_values

        logits = self._compute_logits(output.last_hidden_state[:, -1:, :])
        next_token = self._sample(logits.squeeze(1), do_sample, temperature)
        if get_tensor_model_parallel_world_size() > 1:
            torch.distributed.broadcast(next_token, src=0)
        generated = torch.cat([generated, next_token], dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_token_ids:
            eos_tensor = torch.tensor(list(eos_token_ids), device=device)
            finished = finished | torch.isin(next_token.squeeze(-1), eos_tensor)

        # Decode loop -------------------------------------------------------------
        for _ in range(max_new_tokens - 1):
            if finished.all():
                break

            output = self.forward(
                input_ids=next_token,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = output.past_key_values

            logits = self._compute_logits(output.last_hidden_state)
            next_token = self._sample(logits.squeeze(1), do_sample, temperature)
            if get_tensor_model_parallel_world_size() > 1:
                torch.distributed.broadcast(next_token, src=0)
            generated = torch.cat([generated, next_token], dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)],
                    dim=1,
                )

            if eos_token_ids:
                finished = finished | torch.isin(next_token.squeeze(-1), eos_tensor)

        return generated

    def set_processor(
        self,
        processor,
        system_message_t2i: str | None = None,
        system_message_i2i: str | None = None,
    ) -> None:
        self._processor = processor
        self._system_message_t2i = system_message_t2i
        self._system_message_i2i = system_message_i2i

    @torch.no_grad()
    def upsample_prompt(
        self,
        prompt: str | list[str],
        images: list | None = None,
        temperature: float = 0.15,
        device: torch.device | None = None,
        max_new_tokens: int = 512,
        max_length: int = 2048,
    ) -> list[str]:
        if self._processor is None:
            raise RuntimeError("upsample_prompt() requires a processor; call set_processor() first")

        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = device or self.device

        if images is None or len(images) == 0 or (len(images) > 0 and images[0] is None):
            system_message = self._system_message_t2i or ""
        else:
            system_message = self._system_message_i2i or ""

        messages_batch = _format_upsample_input(prompt, system_message, images)

        inputs = self._processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(device, self.dtype)

        generated_ids = self.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            use_cache=True,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[:, input_length:]

        return self._processor.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        params_dict.update(self.named_buffers())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip vision components (lm_head is needed — weights are not tied)
            if any(name.startswith(p) for p in ("vision_tower.", "multi_modal_projector.")):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if name not in params_dict:
                    logger.warning("Skipping weight %s", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        total_param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        logger.info(
            "MistralEncoderModel load_weights: loaded %d params, total param memory: %.2f GiB",
            len(loaded_params),
            total_param_bytes / (1024**3),
        )
        return loaded_params

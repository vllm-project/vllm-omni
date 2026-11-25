# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HiUni transformer: a shared vision-language backbone, a pixel diffusion head and HiGate.

Checkpoint keys (``ernie-research/HiUni``): ``model.visual.*``, ``model.language_model.*``,
``time_token.*``, ``time_embeddings.*``, ``diffusion_head_a.N.*``, ``diffusion_head_b.*``,
``pixel_proj.*`` and ``pixel_dit_gate.*``. The LM head is tied to ``embed_tokens``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3MLP

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

# One 16x16 RGB patch; the diffusion head predicts four of them (a merged 32x32 block) per token.
PATCH_DIM = 3 * 16 * 16
BLOCK_DIM = 4 * PATCH_DIM


def _prefix(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def merge_patches(raw: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Raster patches ``[4*h*w, d]`` -> merged 2x2 blocks ``[h*w, 4*d]`` (patch channel, block row, block column)."""
    dim = raw.shape[-1]
    return raw.reshape(height, 2, width, 2, dim).permute(0, 2, 4, 1, 3).reshape(-1, dim * 4)


def unmerge_patches(merged: torch.Tensor, height: int, width: int) -> torch.Tensor:
    dim = merged.shape[-1] // 4
    return merged.reshape(height, width, dim, 2, 2).permute(0, 3, 1, 4, 2).reshape(-1, dim)


def vision_patches(raw: torch.Tensor, height: int, width: int, patch_size: int = 16) -> torch.Tensor:
    """Raster RGB patches -> the ViT's block-major layout, duplicating the frame for temporal_patch_size=2."""
    patches = raw.reshape(height, 2, width, 2, 3, patch_size, patch_size).permute(0, 2, 1, 3, 4, 5, 6)
    patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1)
    return patches.reshape(4 * height * width, 6 * patch_size * patch_size)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() * (1 + scale.float()) + shift.float()).to(x.dtype)


class HiUniRotaryEmbedding(nn.Module):
    """Interleaved 3D M-RoPE. The backbone and the diffusion head each use their own head_dim."""

    def __init__(self, head_dim: int, theta: float, mrope_section: list[int]) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.mrope_section = list(mrope_section)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [3, B, S] (temporal, height, width). inv_freq is rebuilt in float32 on the CPU, like the
        # reference, on every call so that a bfloat16 default dtype or a module-wide `.to(dtype)` cannot round it.
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        inv_freq = (1.0 / (self.theta**exponent)).to(x.device)
        inv_freq = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq @ position_ids[:, :, None, :].float()).transpose(2, 3)
            freqs_t = freqs[0].clone()
            for dim, offset in ((1, 1), (2, 2)):
                idx = slice(offset, self.mrope_section[dim] * 3, 3)
                freqs_t[..., idx] = freqs[dim, ..., idx]
            emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class HiUniKVCache:
    """Per-layer key/value cache for autoregressive text decoding.

    Generated tokens are causal and never change the rows before them, so caching keys and values is exact.
    """

    def __init__(self) -> None:
        self.layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self.layers:
            past_key, past_value = self.layers[layer_idx]
            key, value = torch.cat((past_key, key), dim=1), torch.cat((past_value, value), dim=1)
        self.layers[layer_idx] = (key, value)
        return key, value


class HiUniAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        layer_idx: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0 or (num_kv_heads % tp_size != 0 and tp_size % num_kv_heads != 0):
            raise ValueError(
                f"heads={num_heads}/kv_heads={num_kv_heads} cannot be split over tensor_parallel={tp_size}"
            )
        self.head_dim = head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=_prefix(prefix, "qkv_proj"),
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=_prefix(prefix, "o_proj"),
        )
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.attn = Attention(
            num_heads=self.qkv_proj.num_heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=head_dim**-0.5,
            num_kv_heads=self.qkv_proj.num_kv_heads,
            prefix=_prefix(prefix, "attn"),
            role="self",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        kv_cache: HiUniKVCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.qkv_proj.num_heads * self.head_dim
        kv_size = self.qkv_proj.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
        query = self.q_norm(query.reshape(batch_size, seq_len, -1, self.head_dim))
        key = self.k_norm(key.reshape(batch_size, seq_len, -1, self.head_dim))
        value = value.reshape(batch_size, seq_len, -1, self.head_dim)
        cos, sin = (emb.unsqueeze(2) for emb in position_embeddings)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        if kv_cache is not None:
            key, value = kv_cache.update(self.layer_idx, key, value)
        # The layouts need a dense additive mask (bidirectional image blocks inside a causal sequence); backends
        # that cannot take one fall back to SDPA for this call.
        output = self.attn(query, key, value, AttentionMetadata(attn_mask=attention_mask))
        output, _ = self.o_proj(output.reshape(batch_size, seq_len, q_size))
        return output


class HiUniDecoderLayer(nn.Module):
    """Pre-norm block shared by the backbone (``head_dim=128``) and the diffusion head (``hidden/heads=80``)."""

    def __init__(
        self,
        text_config,
        head_dim: int,
        norm_eps: float,
        *,
        layer_idx: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = HiUniAttention(
            text_config.hidden_size,
            text_config.num_attention_heads,
            text_config.num_key_value_heads,
            head_dim,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=_prefix(prefix, "self_attn"),
        )
        self.mlp = Qwen3MLP(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_act=text_config.hidden_act,
            quant_config=quant_config,
            prefix=_prefix(prefix, "mlp"),
        )
        self.input_layernorm = RMSNorm(text_config.hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(text_config.hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        kv_cache: HiUniKVCache | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, attention_mask, kv_cache
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class HiUniTextModel(nn.Module):
    def __init__(
        self,
        text_config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            [
                HiUniDecoderLayer(
                    text_config,
                    text_config.head_dim,
                    text_config.rms_norm_eps,
                    layer_idx=index,
                    quant_config=quant_config,
                    prefix=_prefix(prefix, f"layers.{index}"),
                )
                for index in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        rope_scaling = getattr(text_config, "rope_scaling", None) or getattr(text_config, "rope_parameters", None) or {}
        self.rotary_emb = HiUniRotaryEmbedding(
            text_config.head_dim,
            _rope_theta(text_config),
            rope_scaling.get("mrope_section", [24, 20, 20]),
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kv_cache: HiUniKVCache | None = None,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask, kv_cache)
        return self.norm(hidden_states)


def _rope_theta(text_config) -> float:
    # transformers 5 moves rope_theta into rope_parameters.
    theta = getattr(text_config, "rope_theta", None)
    if theta is None:
        theta = (getattr(text_config, "rope_parameters", None) or {}).get("rope_theta", 5000000.0)
    return float(theta)


class HiUniTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Timesteps are on the training scale [0, 1000].
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=timesteps.device).float() / 128)
        phase = timesteps.float()[:, None] * frequencies[None]
        return self.mlp(torch.cat((phase.cos(), phase.sin()), dim=-1).to(self.mlp[0].weight.dtype))


class HiUniFinalLayer(nn.Module):
    """Time-modulated output projection of the diffusion head; the only place the head sees the timestep."""

    def __init__(self, hidden_size: int, out_channels: int = BLOCK_DIM) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(time_emb).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm_final(x), shift.to(x.dtype), scale.to(x.dtype)))


class HiUniGate(nn.Module):
    """HiGate: adds gated, normalized pixel features back to the backbone states of image tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_fc1 = nn.Linear(dim, dim)
        self.gate_fc2 = nn.Linear(dim, 1)
        self.layer_norm = nn.LayerNorm(dim)
        self.hi_info_norm = nn.LayerNorm(dim)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        gate = self.gate_fc2(F.silu(self.gate_fc1(low)))
        return self.layer_norm(low + gate * self.hi_info_norm(high))


class HiUniBackbone(nn.Module):
    """Holds ``model.visual`` and ``model.language_model`` under the checkpoint's names."""

    def __init__(
        self, config: Qwen3VLConfig, *, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None:
        super().__init__()
        vision_config = config.vision_config
        vision_config._attn_implementation = "sdpa"
        # The ViT is small and replicated on every rank; DeepStack mergers load but are unused, as in training.
        self.visual = Qwen3VLVisionModel._from_config(vision_config)
        self.language_model = HiUniTextModel(
            config.text_config, quant_config=quant_config, prefix=_prefix(prefix, "language_model")
        )


class HiUniTransformer(nn.Module):
    """Backbone, Pixel Diffusion Head and HiGate. Attribute names follow the checkpoint keys."""

    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        num_diffusion_layers: int = 10,
        gate_inject_layer: int = 3,
        image_token_id: int = 151655,
        time_token_id: int = 151661,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        text_config = config.text_config
        dim = text_config.hidden_size
        self.image_token_id = image_token_id
        self.time_token_id = time_token_id
        self.gate_inject_layer = gate_inject_layer
        self.vocab_size = text_config.vocab_size
        self.model = HiUniBackbone(config, quant_config=quant_config, prefix=_prefix(prefix, "model"))
        self.time_token = HiUniTimestepEmbedder(dim)
        self.time_embeddings = HiUniTimestepEmbedder(dim)
        head_dim = dim // text_config.num_attention_heads
        self.diffusion_head_a = nn.ModuleList(
            [
                HiUniDecoderLayer(
                    text_config,
                    head_dim,
                    1e-5,
                    quant_config=quant_config,
                    prefix=_prefix(prefix, f"diffusion_head_a.{index}"),
                )
                for index in range(num_diffusion_layers)
            ]
        )
        rope_scaling = getattr(text_config, "rope_scaling", None) or getattr(text_config, "rope_parameters", None) or {}
        self.head_rotary_emb = HiUniRotaryEmbedding(
            head_dim, _rope_theta(text_config), rope_scaling.get("mrope_section", [24, 20, 20])
        )
        self.diffusion_head_b = HiUniFinalLayer(dim, BLOCK_DIM)
        self.pixel_proj = nn.Linear(BLOCK_DIM, dim)
        self.pixel_dit_gate = HiUniGate(dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.pixel_proj.weight.dtype

    def visual(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Merged ViT features without DeepStack (transformers 4.x returns a tuple, 5.x an output object)."""
        output = self.model.visual(pixel_values.to(self.dtype), grid_thw=grid_thw)
        return output.pooler_output if hasattr(output, "pooler_output") else output[0]

    def text_logits(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_features: torch.Tensor | None = None,
        num_images: int = 0,
        kv_cache: HiUniKVCache | None = None,
    ) -> torch.Tensor:
        """Understanding and chat: clean images, time tokens embed t=0, logits of the last position.

        With ``kv_cache``, the first call is the prefill and later calls pass only the new tokens.
        """
        embeddings = self.model.language_model.embed_tokens(input_ids)
        if image_features is not None:
            embeddings = embeddings.clone()
            embeddings[input_ids.eq(self.image_token_id)] = image_features.to(embeddings.dtype)
            times = self.time_token(torch.zeros(num_images, device=input_ids.device))
            embeddings[input_ids.eq(self.time_token_id)] = times.to(embeddings.dtype)
        hidden = self.model.language_model(embeddings, position_ids, attention_mask, kv_cache)
        # The LM head is tied to the (vocab-parallel) input embedding.
        logits = F.linear(hidden[:, -1], self.model.language_model.embed_tokens.weight)
        if get_tensor_model_parallel_world_size() > 1:
            logits = tensor_model_parallel_all_gather(logits)
        return logits[..., : self.vocab_size]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        height: int,
        width: int,
        ref_features: torch.Tensor | None = None,
        ref_raster: torch.Tensor | None = None,
        ref_height: int | None = None,
        ref_width: int | None = None,
    ) -> torch.Tensor:
        """One CFG branch.

        ``latents`` holds raster patches ``[4*height*width, 768]`` where ``height``/``width`` count merged 32px
        blocks, and ``timestep`` is on the [0, 1000] scale. Returns the predicted clean patches (float32).
        """
        dtype = self.dtype
        pad = input_ids.eq(self.image_token_id)
        pad_index = pad[0].nonzero(as_tuple=True)[0]
        ref_mask = torch.zeros_like(pad)
        has_ref = ref_features is not None and pad_index.numel() == height * width + ref_height * ref_width
        if has_ref:
            ref_mask[0, pad_index[: ref_height * ref_width]] = True
        gen_mask = pad & ~ref_mask
        grid = torch.tensor([[1, height * 2, width * 2]], device=latents.device)
        gen_features = self.visual(vision_patches(latents.to(dtype), height, width), grid)
        embeddings = self.model.language_model.embed_tokens(input_ids).clone()
        if has_ref:
            embeddings[ref_mask] = ref_features.to(embeddings.dtype)
        embeddings[gen_mask] = gen_features.to(embeddings.dtype)
        embeddings[input_ids.eq(self.time_token_id)] = self.time_token(timestep).to(embeddings.dtype)
        # Generation runs without DeepStack injection, as in training.
        hidden = self.model.language_model(embeddings, position_ids, attention_mask)
        position_embeddings = self.head_rotary_emb(hidden, position_ids)
        gen_high = self.pixel_proj(merge_patches(latents.to(dtype), height, width))
        ref_high = self.pixel_proj(merge_patches(ref_raster.to(dtype), ref_height, ref_width)) if has_ref else None
        for index, layer in enumerate(self.diffusion_head_a):
            if index == self.gate_inject_layer:
                if ref_high is not None:
                    hidden[ref_mask] = self.pixel_dit_gate(hidden[ref_mask], ref_high).to(hidden.dtype)
                hidden[gen_mask] = self.pixel_dit_gate(hidden[gen_mask], gen_high).to(hidden.dtype)
            hidden = layer(hidden, position_embeddings, attention_mask)
        output = self.diffusion_head_b(hidden[gen_mask], self.time_embeddings(timestep))
        return unmerge_patches(output, height, width).float()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Fused qkv / gate_up only exist in the backbone and head blocks; ViT keys load into the HF vision tower as-is.
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name == "lm_head.weight":
                continue
            stacked = (
                None
                if name.startswith("model.visual.")
                else next((entry for entry in stacked_params_mapping if entry[1] in name), None)
            )
            if stacked is not None:
                param_name, weight_name, shard_id = stacked
                name = name.replace(weight_name, param_name)
                param = params[name]
                param.weight_loader(param, loaded_weight, shard_id)
            elif name in params:
                param = params[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                continue
            loaded_params.add(name)
        return loaded_params


__all__ = ["HiUniKVCache", "HiUniTransformer", "merge_patches", "unmerge_patches", "vision_patches"]

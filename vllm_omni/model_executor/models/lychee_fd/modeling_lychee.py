# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native single-GPU Lychee-FD model graph for vLLM 0.31 MRV2."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .audio_encoder import LycheeAudioAdaptor, LycheeAudioEncoder
from .configuration_lychee import LycheeFDConfig
from .released_attention import adapt_released_attention
from .side_head import LycheeSideHead
from .weight_loader import LycheeStreamingWeightTracker


class LycheeLogitsProcessor(LogitsProcessor):
    """Preserve the released dense vocabulary shape on the single-GPU graph."""

    def _apply_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            lm_head.tp_size == 1
            and isinstance(lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod))
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
        ):
            # Native loader storage stays padded. Projecting the unused rows
            # changes the CUDA GEMV reduction even when they are sliced later.
            bias = embedding_bias[: self.vocab_size] if embedding_bias is not None else None
            return nn.functional.linear(hidden_states, lm_head.weight[: self.vocab_size], bias)
        return super()._apply_head(lm_head, hidden_states, embedding_bias)


@dataclass(frozen=True, slots=True)
class LycheeBranchOutputs:
    """The three pre-sampling hidden-state streams produced by one forward."""

    text_hidden: torch.Tensor
    stoken_hidden: torch.Tensor
    control_hidden: torch.Tensor


def fuse_channel_embeddings(
    text_embeddings: torch.Tensor,
    *,
    stoken_embeddings: torch.Tensor | None = None,
    control_embeddings: torch.Tensor | None = None,
    audio_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse request-local channel inputs without any process-global state."""

    fused = text_embeddings
    for name, channel in (
        ("stoken", stoken_embeddings),
        ("control", control_embeddings),
        ("audio", audio_embeddings),
    ):
        if channel is None:
            continue
        if channel.shape != fused.shape:
            raise ValueError(
                f"{name} embeddings must match text embeddings: {tuple(channel.shape)} != {tuple(fused.shape)}"
            )
        fused = fused + channel.to(device=fused.device, dtype=fused.dtype)
    return fused


def merge_conditioning(stoken_hidden: torch.Tensor, sampled_text_embeddings: torch.Tensor) -> torch.Tensor:
    """Condition speech continuation on the exact same-step text sample."""

    if stoken_hidden.shape != sampled_text_embeddings.shape:
        raise ValueError(
            "Speech hidden and sampled-text embedding shapes must match: "
            f"{tuple(stoken_hidden.shape)} != {tuple(sampled_text_embeddings.shape)}"
        )
    return stoken_hidden + sampled_text_embeddings


# Torch 2.7 Reduce.cuh used ascending warp shuffle offsets. New Torch uses
# descending offsets, changing means at BF16 rounding boundaries. Preserve
# the released four-vector accumulator and reduction tree for this model.
@triton.jit
def _released_rms_mean_kernel(inputs, means, width: tl.constexpr, block_width: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, block_width)
    a0 = tl.full((block_width,), 0, tl.float32)
    a1 = tl.full((block_width,), 0, tl.float32)
    a2 = tl.full((block_width,), 0, tl.float32)
    a3 = tl.full((block_width,), 0, tl.float32)
    for step in tl.static_range(triton.cdiv(triton.cdiv(width, 4), block_width)):
        idx = (lanes + step * block_width) * 4
        x0 = tl.load(inputs + row * width + idx, mask=idx < width, other=0).to(tl.float32)
        x1 = tl.load(inputs + row * width + idx + 1, mask=idx + 1 < width, other=0).to(tl.float32)
        x2 = tl.load(inputs + row * width + idx + 2, mask=idx + 2 < width, other=0).to(tl.float32)
        x3 = tl.load(inputs + row * width + idx + 3, mask=idx + 3 < width, other=0).to(tl.float32)
        a0 = a0 + x0 * x0
        a1 = a1 + x1 * x1
        a2 = a2 + x2 * x2
        a3 = a3 + x3 * x3
    value = ((a0 + a1) + a2) + a3
    for shift in tl.static_range((block_width // 32).bit_length() - 1, 0, -1):
        offset = 16 << shift
        other = tl.gather(value, tl.minimum(lanes + offset, block_width - 1), 0)
        value = tl.where(lanes < offset, value + other, value)
    for power in tl.static_range(5):
        offset = 1 << power
        other = tl.gather(value, tl.minimum(lanes + offset, block_width - 1), 0)
        value = value + other
    reduced = tl.sum(tl.where(lanes == 0, value, 0), 0)
    tl.store(means + row, reduced * (1.0 / width))


def _released_rms_mean(inputs):
    inputs = inputs.contiguous()
    rows, width = inputs.shape
    if rows == 0:
        return torch.empty((0, 1), dtype=torch.float32, device=inputs.device)
    height = min(1 << (rows.bit_length() - 1), 16)
    block_width = min(1 << ((width // 4).bit_length() - 1), 512 // height)
    means = torch.empty((rows, 1), dtype=torch.float32, device=inputs.device)
    _released_rms_mean_kernel[(rows,)](inputs, means, width, block_width, enable_fp_fusion=False, num_warps=4)
    return means


class LycheeRMSNorm(nn.RMSNorm):
    """Released Torch 2.7 composition with the native residual tuple interface.

    Torch 2.7 used separate FP32 pow, mean, rsqrt and multiply operations.
    New Torch fused RMSNorm changes the rounding even on identical input.
    Keep that released composition, including BF16 residual addition and
    one final conversion after multiplying the checkpoint weight.
    """

    def _normalize(self, hidden: torch.Tensor) -> torch.Tensor:
        opmath = torch.float32 if hidden.dtype in (torch.float16, torch.bfloat16) else hidden.dtype
        upcast = hidden.to(opmath)
        dimensions = tuple(range(hidden.ndim - len(self.normalized_shape), hidden.ndim))
        eps = self.eps if self.eps is not None else torch.finfo(opmath).eps
        if hidden.is_cuda and hidden.dtype == torch.bfloat16 and hidden.ndim == 2 and hidden.shape[-1] == 3584:
            mean = _released_rms_mean(hidden)
        else:
            mean = upcast.pow(2).mean(dimensions, keepdim=True)
        inverse = mean.add_(eps).rsqrt()
        normalized = upcast * inverse
        if self.weight is not None:
            normalized = normalized * self.weight
        return normalized.to(hidden.dtype)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self._normalize(hidden)
        residual = hidden + residual
        return self._normalize(residual), residual


class LycheeGateUpLinear(MergedColumnParallelLinear):
    """Keep packed parameter loading, with the two released GEMM shapes.

    Concatenating gate/up output columns changes the CUDA BF16 reduction
    kernel selected for this checkpoint. Separate projections reproduce the
    old CUDA result on frozen inputs while preserving native weight routes.
    """

    def forward(self, hidden: torch.Tensor):
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            return super().forward(hidden)
        split = self.weight.shape[0] // 2
        bias = None if self.skip_bias_add else self.bias
        gate = torch.nn.functional.linear(hidden, self.weight[:split], None if bias is None else bias[:split])
        up = torch.nn.functional.linear(hidden, self.weight[split:], None if bias is None else bias[split:])
        return torch.cat((gate, up), dim=-1), self.bias if self.skip_bias_add else None


class LycheeSiluAndMul(nn.Module):
    """Preserve the released BF16 SiLU intermediate before multiplication."""

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class LycheeRotaryEmbedding(nn.Module):
    """Released RoPE rounds each multiply and add in the inference dtype.

    Current fused kernels can retain float32 intermediates. The old 0.9.2
    scalar_t CUDA formula materializes each BF16 product before its sum.
    Keep the existing cache and paged-attention Q/K layout, changing only
    these checkpoint-owned arithmetic operations.
    """

    def __init__(self, rotary: nn.Module) -> None:
        super().__init__()
        self.head_size = rotary.head_size
        self.rotary_dim = rotary.rotary_dim
        self.is_neox_style = rotary.is_neox_style
        self.register_buffer("cos_sin_cache", rotary.cos_sin_cache, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor | None = None):
        cache = self.cos_sin_cache.to(device=query.device, dtype=query.dtype)
        cos, sin = cache.index_select(0, positions.reshape(-1)).chunk(2, dim=-1)
        cos, sin = cos[:, None, :], sin[:, None, :]

        def rotate(states: torch.Tensor) -> torch.Tensor:
            hidden = states.reshape(positions.numel(), -1, self.head_size)
            rotated, tail = hidden[..., : self.rotary_dim], hidden[..., self.rotary_dim :]
            if self.is_neox_style:
                left, right = rotated.chunk(2, dim=-1)
                result = torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
            else:
                left, right = rotated[..., ::2], rotated[..., 1::2]
                result = torch.stack((left * cos - right * sin, right * cos + left * sin), dim=-1).flatten(-2)
            return torch.cat((result, tail), dim=-1).reshape_as(states)

        return rotate(query), None if key is None else rotate(key)


class _LycheeDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.input_layernorm = LycheeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LycheeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn.rotary_emb = LycheeRotaryEmbedding(self.self_attn.rotary_emb)
        self.self_attn.attn.impl = adapt_released_attention(self.self_attn.attn.impl)
        self.mlp.gate_up_proj = LycheeGateUpLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=kwargs.get("quant_config"),
            prefix=f"{kwargs.get('prefix', '')}.mlp.gate_up_proj",
        )
        self.mlp.act_fn = LycheeSiluAndMul()


class _LycheeDecoderBranch(nn.Module):
    def __init__(
        self,
        config,
        *,
        cache_config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _LycheeDecoderLayer(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{index}",
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = LycheeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden


class _LycheeMainDecoder(nn.Module):
    def __init__(
        self,
        config,
        *,
        cache_config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                _LycheeDecoderLayer(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{index}",
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = LycheeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LycheeFullDuplexForConditionalGeneration(nn.Module, SupportsQuant):
    """Released Lychee 28+4+4+4 graph with native paged-attention layers.

    The forward stops before sampling and returns text, speech-token and
    control hidden streams. MRV2 samples text/control first, then calls
    :meth:`continue_after_primary_sample` so merge is conditioned on the exact
    text tensor published for that tick.
    """

    is_text_generation_model = True
    is_pooling_model = False
    supported_tasks = {"generate"}
    have_multimodal_outputs = True
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def create_mrv2_model_state(self, vllm_config: VllmConfig, encoder_cache: Any, device: torch.device):
        from vllm_omni.worker_v2.model_states.lychee_model_state import LycheeModelState

        return LycheeModelState(vllm_config, self, encoder_cache, device)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise NotImplementedError("Lychee-FD currently supports single-GPU TP=1, PP=1 only")
        if getattr(parallel, "prefill_context_parallel_size", 1) != 1:
            raise NotImplementedError("Lychee-FD does not support context parallelism")

        raw_config = vllm_config.model_config.hf_config
        self.config = (
            raw_config if isinstance(raw_config, LycheeFDConfig) else LycheeFDConfig.from_dict(raw_config.to_dict())
        )
        self.contract = self.config.validate_lychee_contract()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config

        self.model = _LycheeMainDecoder(
            self.config.text_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.stoken_model = _LycheeDecoderBranch(
            self.config.stoken_layer_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "stoken_model"),
        )
        self.control_model = _LycheeDecoderBranch(
            self.config.control_layer_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "control_model"),
        )
        self.merge_model = _LycheeDecoderBranch(
            self.config.merge_layer_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "merge_model"),
        )
        self.encoder = LycheeAudioEncoder(self.config.audio_encoder_config)
        self.adapter = LycheeAudioAdaptor(self.config.audio_encoder_config)
        self.lm_head = ParallelLMHead(
            self.contract.layout.vocab_size,
            self.contract.layout.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LycheeLogitsProcessor(self.contract.layout.vocab_size)
        self.side_head = LycheeSideHead(self.config, self.contract.layout.vocab_size)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def encode_audio(
        self,
        audio_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, encoder_lengths = self.encoder(audio_features, feature_lengths)
        return self.adapter(hidden), self.adapter.output_lengths(encoder_lengths)

    @staticmethod
    def _effective_hidden(hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        return hidden if residual is None else hidden + residual

    def forward_branches(
        self,
        positions: torch.Tensor,
        fused_embeddings: torch.Tensor,
    ) -> LycheeBranchOutputs:
        hidden = fused_embeddings
        residual = None
        control_input = None
        stoken_input = None
        control_boundary = self.contract.layout.control_branch_index
        stoken_boundary = self.contract.layout.main_layers - self.contract.layout.stoken_layers

        if control_boundary == 0:
            control_input = hidden
        if stoken_boundary == 0:
            stoken_input = hidden
        for index, layer in enumerate(self.model.layers):
            hidden, residual = layer(positions, hidden, residual)
            boundary = index + 1
            if boundary == control_boundary:
                control_input = self._effective_hidden(hidden, residual)
            if boundary == stoken_boundary:
                stoken_input = self._effective_hidden(hidden, residual)

        if control_input is None or stoken_input is None:
            raise RuntimeError(f"Failed to capture Lychee branch inputs at H{control_boundary} and H{stoken_boundary}")
        text_hidden, _ = self.model.norm(hidden, residual)
        return LycheeBranchOutputs(
            text_hidden=text_hidden,
            stoken_hidden=self.stoken_model(positions, stoken_input),
            control_hidden=self.control_model(positions, control_input),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        stoken_input_ids: torch.Tensor | None = None,
        control_input_ids: torch.Tensor | None = None,
        stoken_input_mask: torch.Tensor | None = None,
        control_input_mask: torch.Tensor | None = None,
        audio_embeddings: torch.Tensor | None = None,
        **_: object,
    ) -> OmniOutput:
        if intermediate_tensors is not None:
            raise NotImplementedError("Lychee-FD does not support pipeline-parallel intermediate tensors")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            text_embeddings = self.embed_input_ids(input_ids)
        else:
            text_embeddings = inputs_embeds
        stoken_embeddings = None if stoken_input_ids is None else self.embed_input_ids(stoken_input_ids)
        control_embeddings = None if control_input_ids is None else self.embed_input_ids(control_input_ids)
        if stoken_embeddings is not None and stoken_input_mask is not None:
            stoken_embeddings = stoken_embeddings * stoken_input_mask[:, None].to(stoken_embeddings.dtype)
        if control_embeddings is not None and control_input_mask is not None:
            control_embeddings = control_embeddings * control_input_mask[:, None].to(control_embeddings.dtype)
        fused = fuse_channel_embeddings(
            text_embeddings,
            stoken_embeddings=stoken_embeddings,
            control_embeddings=control_embeddings,
            audio_embeddings=audio_embeddings,
        )
        outputs = self.forward_branches(positions, fused)
        return OmniOutput(
            text_hidden_states=outputs.text_hidden,
            multimodal_outputs={
                "lychee_stoken_hidden": outputs.stoken_hidden,
                "lychee_control_hidden": outputs.control_hidden,
            },
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_control_logits(self, control_hidden: torch.Tensor) -> torch.Tensor:
        return self.side_head.project("control", control_hidden, self.lm_head, self.logits_processor)

    def continue_after_primary_sample(
        self,
        *,
        positions: torch.Tensor,
        stoken_hidden: torch.Tensor,
        sampled_text_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampled_text_embeddings = self.embed_input_ids(sampled_text_token_ids.reshape(-1))
        merge_input = merge_conditioning(stoken_hidden, sampled_text_embeddings)
        speech_hidden = self.merge_model(positions, merge_input)
        return speech_hidden, self.side_head.project("speech", speech_hidden, self.lm_head, self.logits_processor)

    def release_worker_resources(self) -> None:
        """Release model-owned CUDA plans after output consumers are drained."""
        self.side_head.clear()
        self.adapter.conv.close()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters(remove_duplicate=False))
        tracker = LycheeStreamingWeightTracker(params)
        for source_name, loaded_weight in weights:
            route = tracker.route(source_name)
            if route is None:
                continue
            parameter = params[route.target_name]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            if route.shard_id is None:
                loader(parameter, loaded_weight)
            else:
                loader(parameter, loaded_weight, route.shard_id)
        loaded = tracker.finish()
        self.side_head.clear()
        return loaded


__all__ = [
    "LycheeBranchOutputs",
    "LycheeFullDuplexForConditionalGeneration",
    "fuse_channel_embeddings",
    "merge_conditioning",
]

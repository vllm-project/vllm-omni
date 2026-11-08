# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native vLLM Qwen backbone and HF-schema VibeVoice audio components."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers.initialization import no_init_weights
from transformers.models.vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerConv1dPaddingCache as PaddingCache,
)
from transformers.models.vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerEncoderModel,
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceAcousticTokenizerRMSNorm,
)
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from .diffusion import (
    VIBEVOICE_DEFAULT_GUIDANCE_SCALE,
    VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS,
    VibeVoiceDiffusionHead,
    create_dpm_scheduler,
)
from .negative_kv import VibeVoiceNegativeKV
from .processing_vibevoice import (
    VibeVoiceDummyInputsBuilder,
    VibeVoiceMultiModalProcessor,
    VibeVoiceProcessingInfo,
)
from .prompt import AUDIO_TOKEN
from .stateful import VibeVoiceStatefulInference

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors

    from vllm_omni.model_executor.models.model_local_kv import ModelLocalKVSpec


@dataclass(slots=True)
class VibeVoiceAudioTokenDecodeOutput:
    """One decoded audio token per request; caches are request-owned by the caller."""

    audio: torch.Tensor
    next_embeddings: torch.Tensor
    acoustic_caches: list[PaddingCache]
    semantic_caches: list[PaddingCache]


def _stack_padding_caches(caches: list[PaddingCache | None]) -> PaddingCache | None:
    """Stack per-request causal-conv caches (``None`` = new request) into one batch cache.

    Cache layers are row-independent ``(B, C, left_pad)`` tensors and a new
    request's lazily created layer is all zeros, so stacking is exact.
    """
    if len(caches) == 1:
        return caches[0]
    template = next((cache for cache in caches if cache is not None), None)
    if template is None:
        return None
    stacked = PaddingCache()
    for key, layer in template.layers.items():
        stacked.layers[key] = merged = copy.copy(layer)
        merged.cache = torch.cat(
            [torch.zeros_like(layer.cache) if cache is None else cache.layers[key].cache for cache in caches]
        )
    return stacked


def _split_padding_cache(cache: PaddingCache, batch_size: int) -> list[PaddingCache]:
    if batch_size == 1:
        return [cache]
    rows = [PaddingCache() for _ in range(batch_size)]
    for key, layer in cache.layers.items():
        for index, row in enumerate(rows):
            row.layers[key] = row_layer = copy.copy(layer)
            row_layer.cache = layer.cache[index : index + 1]
    return rows


class VibeVoiceMultiModalProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.act = VibeVoiceAcousticTokenizerRMSNorm(output_dim, eps=1e-6)
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(features)))


class VibeVoiceModel(nn.Module):
    """Checkpoint-aligned modules; request state belongs to the outer model.

    Only Qwen's own loader packs Q/K/V and gate/up projections. In particular,
    that mapper must not be applied to the diffusion head's unfused MLPs.
    No full Transformers TTS model or duplicate Qwen weights are constructed.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        super().__init__()
        self.config = config = vllm_config.model_config.hf_config
        self.language_model = Qwen2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model"))
        # All parameters are supplied by the checkpoint, not HF initialization.
        with no_init_weights():
            self.audio_tower = VibeVoiceAcousticTokenizerModel(config.audio_config)
            self.semantic_tokenizer_encoder = VibeVoiceAcousticTokenizerEncoderModel(config.semantic_model_config)
        self.multi_modal_projector = VibeVoiceMultiModalProjector(
            config.audio_config.hidden_size, config.text_config.hidden_size
        )
        self.semantic_connector = VibeVoiceMultiModalProjector(
            config.semantic_model_config.hidden_size, config.text_config.hidden_size
        )
        self.diffusion_head = VibeVoiceDiffusionHead(config.diffusion_head_config)
        self.latent_scaling_factor = nn.Parameter(torch.tensor(1.0))
        self.latent_bias_factor = nn.Parameter(torch.tensor(0.0))
        self.samples_per_token = math.prod(config.audio_config.decoder_config.upsampling_ratios)

    def reference_embeddings(
        self,
        input_values: torch.Tensor | Sequence[torch.Tensor],
        audio_num_tokens: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Encode a ragged reference batch with the official sampled VAE path.

        Token counts stay on CPU. Acoustic latents alone are used for reference
        conditioning; semantic feedback applies only to generated waveform.
        """
        counts = []
        for count in audio_num_tokens:
            if count.device.type != "cpu" or count.numel() != 1:
                raise ValueError("VibeVoice reference token counts must be CPU scalars.")
            counts.append(int(count.item()))
        rows = list(input_values)
        if not rows or len(rows) != len(counts):
            raise ValueError("VibeVoice reference waveform/count batches must be non-empty and aligned.")
        lengths = [row.shape[-1] for row in rows]
        hop = self.samples_per_token
        for row, count in zip(rows, counts, strict=True):
            if row.ndim != 2 or row.shape[0] != 1 or not 0 < count * hop <= row.shape[-1]:
                raise ValueError("VibeVoice reference waveform shape does not match its token count.")
        parameter = next(self.audio_tower.parameters())
        batch = torch.stack([nn.functional.pad(row, (0, max(lengths) - row.shape[-1])) for row in rows])
        latents = self.audio_tower.encode(batch.to(parameter), sample=True).latents
        latents = (latents + self.latent_bias_factor) * self.latent_scaling_factor
        embeddings = self.multi_modal_projector(latents)
        return tuple(row[:count] for row, count in zip(embeddings, counts, strict=True))

    def sample_audio_latent(
        self,
        positive_condition: torch.Tensor,
        negative_condition: torch.Tensor,
        noise: torch.Tensor,
        *,
        guidance_scale: float = VIBEVOICE_DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS,
    ) -> torch.Tensor:
        """Denoise one acoustic latent per request; returns ``(B, 1, latent)``.

        ``noise`` is the official single ``(2B, latent)`` draw per active subset;
        only its first half seeds both CFG branches at every step.
        """
        batch_size = positive_condition.shape[0]
        if noise.shape != (2 * batch_size, self.config.audio_config.hidden_size):
            raise ValueError(f"VibeVoice noise must have shape (2B, latent), got {tuple(noise.shape)}.")
        condition = torch.cat([positive_condition, negative_condition], dim=0)
        latent = noise.to(condition)
        scheduler = create_dpm_scheduler()
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        for timestep in scheduler.timesteps:
            shared = latent[:batch_size]
            combined = torch.cat([shared, shared], dim=0)
            prediction = self.diffusion_head(combined, timestep.repeat(2 * batch_size).to(combined), condition)
            cond, uncond = prediction[:batch_size], prediction[batch_size:]
            guided = uncond + guidance_scale * (cond - uncond)
            latent = scheduler.step(torch.cat([guided, guided], dim=0), timestep, latent).prev_sample
        return latent[:batch_size].unsqueeze(1)

    def decode_audio_tokens(
        self,
        audio_latents: torch.Tensor,
        acoustic_caches: list[PaddingCache | None],
        semantic_caches: list[PaddingCache | None],
    ) -> VibeVoiceAudioTokenDecodeOutput:
        """Decode one latent token per request as one batch and build the next Qwen input embeddings."""
        batch_size = audio_latents.shape[0]
        decoded = self.audio_tower.decode(
            audio_latents / self.latent_scaling_factor - self.latent_bias_factor,
            padding_cache=_stack_padding_caches(acoustic_caches),
            use_cache=True,
        )
        semantic = self.semantic_tokenizer_encoder(
            decoded.audio, padding_cache=_stack_padding_caches(semantic_caches), use_cache=True
        )
        next_embeddings = self.multi_modal_projector(audio_latents) + self.semantic_connector(semantic.latents)
        return VibeVoiceAudioTokenDecodeOutput(
            audio=decoded.audio,
            next_embeddings=next_embeddings,
            acoustic_caches=_split_padding_cache(decoded.padding_cache, batch_size),
            semantic_caches=_split_padding_cache(semantic.padding_cache, batch_size),
        )


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceMultiModalProcessor,
    info=VibeVoiceProcessingInfo,
    dummy_inputs=VibeVoiceDummyInputsBuilder,
)
class VibeVoiceForConditionalGeneration(nn.Module, SupportsMultiModal):
    """Eager single-stage TTS through the runner's synchronous model hooks.

    Positive Qwen runs in ``forward``; the CFG negative branch, diffusion and
    decode run after sampling in ``post_sample_multimodal_outputs``.
    """

    has_preprocess = True
    supports_sampled_token_ids_cpu_override = True
    omni_pooler_payload_include_hidden = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        _check_supported_config(vllm_config)
        self._vllm_config = vllm_config
        self._kv_capacity_checked = False
        if not self.config.tie_word_embeddings:
            raise ValueError("VibeVoice requires the native checkpoint's tied embeddings.")
        with self._mark_language_model(vllm_config, targets=Qwen2Model):
            self.model = VibeVoiceModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = self.model.language_model.embed_tokens
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.configure_mm_token_handling(self.config.vocab_size, [self.config.audio_token_id])
        self.negative_kv = VibeVoiceNegativeKV(language_model=self.model.language_model, vllm_config=vllm_config)
        self.inference = VibeVoiceStatefulInference(
            model=self.model,
            negative_kv=self.negative_kv,
            audio_bos_token_id=self.config.audio_bos_token_id,
            audio_token_id=self.config.audio_token_id,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality != "audio":
            raise ValueError("VibeVoice supports only reference audio.")
        return AUDIO_TOKEN

    def embed_multimodal(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        input_values = kwargs.get("input_values")
        if input_values is None:
            return ()
        return self.model.reference_embeddings(input_values, kwargs["audio_num_tokens"])

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # The worker sets num_gpu_blocks before its warmup forward, so this fails at startup.
        if not self._kv_capacity_checked and self._vllm_config.cache_config.num_gpu_blocks is not None:
            _check_positive_kv_capacity(self._vllm_config)
            self._kv_capacity_checked = True
        return self.model.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    # Runner hooks, in call order. Dummy/profile runs call only forward and
    # compute_logits, so no request state is created outside preprocess.
    def on_requests_finished(self, finished_req_ids: set[str]) -> None:
        # Called before this step's preprocess; finished requests are no
        # longer scheduled and their audio is already owned by the output.
        self.inference.release(finished_req_ids)

    def preprocess_batch(self, **_: Any) -> None:
        self.inference.begin_step()

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        embeds = self.inference.add_row(
            info["request_id"],
            num_computed=info["_omni_num_computed_tokens"],
            num_scheduled=input_ids.shape[0],
            prompt_len=info["_omni_prompt_len"],
            input_embeds=input_embeds,
        )
        return input_ids, embeds, {}

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self.inference.has_rows:
            self.inference.record_hidden(hidden_states)
        return self.logits_processor(self.lm_head, hidden_states)

    def consume_sampled_token_ids_cpu_override(self, sampled_token_ids: torch.Tensor) -> list[list[int]] | None:
        if not self.inference.has_rows:
            return None
        sampled = sampled_token_ids.tolist()
        self.inference.record_sampled(sampled)
        return sampled

    def post_sample_multimodal_outputs(
        self,
        *,
        req_ids: list[str],
        invalid_req_indices: list[int],
        multimodal_outputs: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not self.inference.has_rows:
            return multimodal_outputs
        return self.inference.finish_step(list(req_ids))

    def model_local_kv_specs(self) -> Sequence[ModelLocalKVSpec]:
        return self.negative_kv.model_local_kv_specs()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(weights)
        # Allocate once after loading, before memory profiling measures the model.
        parameter = next(self.parameters())
        self.negative_kv.allocate(device=parameter.device, dtype=parameter.dtype)
        return loaded


def _check_supported_config(vllm_config: VllmConfig) -> None:
    """Fail fast on options this eager, synchronous implementation does not support."""
    parallel = vllm_config.parallel_config
    unsupported = {
        "enforce_eager=False": not vllm_config.model_config.enforce_eager,
        # The step loop relies on V1 runner sampling hooks; the negative pool is BF16.
        "model runner V2": vllm_config.use_v2_model_runner,
        "a quantized KV cache": vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"),
        "async_scheduling": vllm_config.scheduler_config.async_scheduling,
        "enable_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
        "speculative decoding": vllm_config.speculative_config is not None,
        "tensor/pipeline parallelism": parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1,
    }
    enabled = [name for name, active in unsupported.items() if active]
    if enabled:
        raise ValueError(f"VibeVoice eager offline support does not allow: {', '.join(enabled)}.")


def _check_positive_kv_capacity(vllm_config: VllmConfig) -> None:
    """Every admitted request must fit at full length: preemption-recompute is unsupported."""
    cache = vllm_config.cache_config
    required = vllm_config.scheduler_config.max_num_seqs * math.ceil(
        vllm_config.model_config.max_model_len / cache.block_size
    )
    usable = cache.num_gpu_blocks - 1  # the block pool reserves one null block
    if usable < required:
        raise ValueError(
            f"VibeVoice needs {required} KV blocks (max_num_seqs x max_model_len) because it cannot be "
            f"preempted, but only {usable} are available. Increase kv_cache_memory_bytes or lower "
            "max_model_len / max_num_seqs."
        )

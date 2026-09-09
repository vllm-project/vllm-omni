# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VibeVoice model implementation helpers.

The model class will live in this module. Keep checkpoint compatibility next to
that class so the model remains the owner of its loading semantics.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import Any

import regex as re
import torch
import torch.nn as nn
from transformers import AutoModel
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.model_runner_metadata import (
    OMNI_INPUT_TOKEN_IDS_CPU_KEY,
    OMNI_IS_PREFILL_KEY,
    OMNI_NUM_COMPUTED_TOKENS_KEY,
    OMNI_PROMPT_LEN_KEY,
    OMNI_REQUEST_ID_KEY,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.named_kv_branch import (
    NamedCausalKVBranch,
    NamedKVBranchRequest,
)

from .audio_decode import (
    VibeVoiceAudioTokenDecodeOutput,
    VibeVoiceAudioTokenDecoder,
)
from .diffusion import (
    VibeVoiceDiffusionHead,
    VibeVoiceDiffusionSampler,
    VibeVoiceRMSNorm,
)
from .negative_branch import VibeVoiceNegativeBranch
from .processing_vibevoice import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_HOP_LENGTH,
    AUDIO_TOKEN,
    SAMPLE_RATE,
    VibeVoiceDummyInputsBuilder,
    VibeVoiceMultiModalProcessor,
    VibeVoiceProcessingInfo,
)
from .runtime_config import (
    VIBEVOICE_DEFAULT_GUIDANCE_SCALE,
    VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS,
    VibeVoiceRuntimeConfig,
)
from .stateful import (
    VibeVoiceNegativeKVBranch,
    VibeVoiceStatefulInference,
)
from .vllm_compat import merge_multimodal_embeddings

logger = init_logger(__name__)


def _flatten_audio_items(
    value: object,
    field_name: str,
    *,
    item_ndim: int,
) -> list[torch.Tensor]:
    """Flatten vLLM's tensor-or-ragged-list MM batch into item tensors."""
    if isinstance(value, torch.Tensor):
        if value.ndim == item_ndim:
            return [value]
        if value.ndim == item_ndim + 1:
            return list(value.unbind(dim=0))
        raise ValueError(
            f"VibeVoice {field_name} items must have rank {item_ndim} "
            f"(or rank {item_ndim + 1} when batched), got shape={tuple(value.shape)}."
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items: list[torch.Tensor] = []
        for item in value:
            items.extend(_flatten_audio_items(item, field_name, item_ndim=item_ndim))
        if items:
            return items
    raise TypeError(f"VibeVoice {field_name} must be a tensor or a non-empty sequence of tensors.")


def _pad_ragged_audio_batch(
    input_values: object,
    padding_mask: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length reference audios batched by vLLM.

    vLLM stacks equal-shaped multimodal fields into tensors, but intentionally
    preserves fields from different requests as a list when their reference
    audio lengths differ. The Microsoft Acoustic Encoder accepts one padded
    tensor batch, so materialize that equivalent representation here while
    preserving each item's padding mask and item order.
    """
    if isinstance(input_values, torch.Tensor) and isinstance(padding_mask, torch.Tensor):
        return input_values, padding_mask

    input_items = _flatten_audio_items(input_values, "input_values", item_ndim=2)
    mask_items = _flatten_audio_items(padding_mask, "padding_mask", item_ndim=1)
    if len(input_items) != len(mask_items):
        raise ValueError(
            "VibeVoice ragged audio batch has different input/mask item counts: "
            f"input_values={len(input_items)}, padding_mask={len(mask_items)}."
        )
    channels = input_items[0].shape[0]
    for item_idx, (input_item, mask_item) in enumerate(zip(input_items, mask_items, strict=True)):
        if input_item.shape[0] != channels:
            raise ValueError(
                f"VibeVoice ragged audio item {item_idx} has {input_item.shape[0]} channels; expected {channels}."
            )
        if input_item.shape[-1] != mask_item.numel():
            raise ValueError(
                f"VibeVoice ragged audio item {item_idx} has mismatched sample lengths: "
                f"input_values={input_item.shape[-1]}, padding_mask={mask_item.numel()}."
            )

    max_samples = max(item.shape[-1] for item in input_items)
    padded_inputs = input_items[0].new_zeros((len(input_items), channels, max_samples))
    padded_masks = mask_items[0].new_zeros((len(mask_items), max_samples))
    for item_idx, (input_item, mask_item) in enumerate(zip(input_items, mask_items, strict=True)):
        samples = input_item.shape[-1]
        padded_inputs[item_idx, :, :samples] = input_item
        padded_masks[item_idx, :samples] = mask_item.reshape(-1)
    return padded_inputs, padded_masks


def _flatten_audio_token_counts(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pieces = [_flatten_audio_token_counts(item) for item in value]
        if pieces:
            device = pieces[0].device
            return torch.cat([piece.to(device=device) for piece in pieces])
    raise TypeError("VibeVoice audio_num_tokens must be a tensor or a non-empty sequence of tensors.")


def _num_tokenizer_stages(config: Any, child_config_name: str) -> int:
    """Return the tokenizer stage count needed for N -> N-1 key mappings."""
    child_config = getattr(config, child_config_name, None)
    depths = getattr(child_config, "depths", None)
    if not isinstance(depths, (list, tuple)) or not depths:
        raise ValueError(f"VibeVoice {child_config_name}.depths must be a non-empty list or tuple, got {depths!r}.")
    return len(depths)


def _add_encoder_mappings(
    mappings: dict[re.Pattern[str], str | None],
    *,
    source: str,
    target: str,
    num_stages: int,
) -> None:
    """Add Microsoft tokenizer-encoder to HF Acoustic Encoder mappings."""
    mappings[re.compile(rf"{source}\.encoder\.downsample_layers\.0\.0\.conv\.")] = f"{target}.stem.conv.conv."
    mappings[re.compile(rf"{source}\.encoder\.stages\.0\.")] = f"{target}.stem.stage."

    for source_idx in range(1, num_stages):
        target_idx = source_idx - 1
        mappings[re.compile(rf"{source}\.encoder\.downsample_layers\.{source_idx}\.0\.conv\.")] = (
            f"{target}.conv_layers.{target_idx}.conv.conv."
        )
        mappings[re.compile(rf"{source}\.encoder\.stages\.{source_idx}\.")] = (
            f"{target}.conv_layers.{target_idx}.stage."
        )

    mappings[re.compile(rf"{source}\.encoder\.head\.conv\.")] = f"{target}.head."


def _add_decoder_mappings(
    mappings: dict[re.Pattern[str], str | None],
    *,
    source: str,
    target: str,
    num_stages: int,
) -> None:
    """Add Microsoft tokenizer-decoder to HF Acoustic Decoder mappings."""
    mappings[re.compile(rf"{source}\.decoder\.upsample_layers\.0\.0\.conv\.conv\.")] = f"{target}.stem.conv.conv."
    mappings[re.compile(rf"{source}\.decoder\.stages\.0\.")] = f"{target}.stem.stage."

    for source_idx in range(1, num_stages):
        target_idx = source_idx - 1
        mappings[re.compile(rf"{source}\.decoder\.upsample_layers\.{source_idx}\.0\.convtr\.convtr\.")] = (
            f"{target}.conv_layers.{target_idx}.convtr.convtr."
        )
        mappings[re.compile(rf"{source}\.decoder\.stages\.{source_idx}\.")] = (
            f"{target}.conv_layers.{target_idx}.stage."
        )

    mappings[re.compile(rf"{source}\.decoder\.head\.conv\.")] = f"{target}.head."


def _build_vibevoice_weights_mapper(config: Any) -> WeightsMapper:
    """Build the Microsoft-checkpoint to HF-runtime name mapper.

    The mapper is also safe for checkpoints already converted to the PR #40546
    layout: none of the source patterns match those names, so they pass through
    unchanged.

    A builder is needed because ``WeightsMapper`` regex replacements cannot
    express the tokenizer's ``source_index - 1`` transformation. We generate
    exact regex entries from the normalized child configs, while
    ``WeightsMapper`` still performs every runtime key conversion.
    """
    acoustic_stages = _num_tokenizer_stages(config, "audio_config")
    semantic_stages = _num_tokenizer_stages(config, "semantic_model_config")

    # dict preserves insertion order. Ordering is significant: specific
    # tokenizer/diffusion mappings must run before their generic cleanup rules.
    mappings: dict[re.Pattern[str], str | None] = {}

    _add_encoder_mappings(
        mappings,
        source=r"semantic_tokenizer",
        target="semantic_tokenizer_encoder",
        num_stages=semantic_stages,
    )
    _add_encoder_mappings(
        mappings,
        source=r"acoustic_tokenizer",
        target="audio_tower.encoder",
        num_stages=acoustic_stages,
    )
    _add_decoder_mappings(
        mappings,
        source=r"acoustic_tokenizer",
        target="audio_tower.decoder",
        num_stages=acoustic_stages,
    )

    mappings.update(
        {
            # Any remaining Acoustic Tokenizer keys belong below audio_tower.
            re.compile(r"acoustic_tokenizer\."): "audio_tower.",
            # Diffusion Head.
            re.compile(r"prediction_head\.t_embedder\.mlp\.0\."): ("diffusion_head.timestep_proj.layer_1."),
            re.compile(r"prediction_head\.t_embedder\.mlp\.2\."): ("diffusion_head.timestep_proj.layer_2."),
            re.compile(r"prediction_head\.layers\.(\d+)\.adaLN_modulation\.1\."): (r"diffusion_head.layers.\1.linear."),
            re.compile(r"prediction_head\.final_layer\.adaLN_modulation\.1\."): (
                "diffusion_head.final_layer.linear_1."
            ),
            re.compile(r"prediction_head\.final_layer\.linear\."): ("diffusion_head.final_layer.linear_2."),
            re.compile(r"prediction_head\."): "diffusion_head.",
            # Acoustic and semantic connectors.
            re.compile(r"acoustic_connector\.fc1\."): "multi_modal_projector.linear_1.",
            re.compile(r"acoustic_connector\.norm\."): "multi_modal_projector.act.",
            re.compile(r"acoustic_connector\.fc2\."): "multi_modal_projector.linear_2.",
            re.compile(r"semantic_connector\.fc1\."): "semantic_connector.linear_1.",
            re.compile(r"semantic_connector\.norm\."): "semantic_connector.act.",
            re.compile(r"semantic_connector\.fc2\."): "semantic_connector.linear_2.",
            # Latent normalization factors.
            re.compile(r"^model\.speech_scaling_factor$"): "model.latent_scaling_factor",
            re.compile(r"^model\.speech_bias_factor$"): "model.latent_bias_factor",
            # Original modules contain one extra nested Conv1d wrapper.
            re.compile(r"mixer\.conv\.conv\.conv\."): "mixer.conv.",
            re.compile(r"\.conv\.conv\.conv\."): ".conv.conv.",
        }
    )

    return WeightsMapper(orig_to_new_regex=mappings)


class VibeVoiceMultiModalProjector(nn.Module):
    """Project a continuous Acoustic/Semantic latent into Qwen2 hidden space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.act = VibeVoiceRMSNorm(output_dim)
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(features)))


class VibeVoiceModel(nn.Module):
    """Weight-complete VibeVoice backbone scaffold.

    Forward-time multimodal replacement and per-request decoder state are added
    separately; this class already mirrors the released checkpoint hierarchy.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.language_model = Qwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.multi_modal_projector = VibeVoiceMultiModalProjector(
            config.audio_config.hidden_size, config.text_config.hidden_size
        )
        self.semantic_tokenizer_encoder = AutoModel.from_config(config.semantic_model_config)
        self.semantic_connector = VibeVoiceMultiModalProjector(
            config.semantic_model_config.hidden_size, config.text_config.hidden_size
        )
        self.diffusion_head = VibeVoiceDiffusionHead(config)
        # Pure model-side numerical helper. It creates a fresh DPM solver for
        # each audio token and owns no request/KV/cache state.
        self.diffusion_sampler = VibeVoiceDiffusionSampler.from_model_config(config)
        # Lazily-created CUDA-graph replay of the denoising loop.
        # The flag is set by the outer model from the deployment runtime
        # config; the executor itself falls back to eager on any capture
        # failure or non-CUDA input.
        self.diffusion_graph_enabled = False
        self._diffusion_graph_executor = None
        # Lazily-created CUDA-graph replay of the audio decode path.
        self.decode_graph_enabled = False
        self.cuda_graph_capture_failure_fatal = False
        self._decode_graph_executor = None
        # Shared graph pool for diffusion and decode graphs: PyTorch's
        # CUDACachingAllocator requires co-resident graphs to share one pool
        # (capture_begin checks global pool use_count; separate pools trigger
        # the ``use_count > 0`` assertion). See make_graphed_callables.
        self._shared_graph_pool = None
        # Like the diffusion sampler, this kernel receives and returns caches;
        # it never owns mutable request state.
        self.audio_token_decoder = VibeVoiceAudioTokenDecoder.from_model_config(config)
        self.latent_scaling_factor = nn.Parameter(torch.tensor(1.0))
        self.latent_bias_factor = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def warmup_diffusion_graphs(
        self,
        batch_sizes: Iterable[int],
        *,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> None:
        """Pre-capture configured diffusion graph keys without consuming RNG."""
        if not self.diffusion_graph_enabled:
            return
        requested_batch_sizes = tuple(batch_sizes)
        if any(
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0
            for batch_size in requested_batch_sizes
        ):
            raise ValueError(
                f"VibeVoice diffusion graph warmup batch sizes must be positive integers: {requested_batch_sizes!r}."
            )
        resolved_batch_sizes = tuple(sorted(set(requested_batch_sizes)))
        if not resolved_batch_sizes:
            return
        executor = self._diffusion_graph_executor
        if executor is not None and executor.disabled:
            return

        head_parameter = next(self.diffusion_head.parameters(), None)
        if head_parameter is None or not head_parameter.is_cuda:
            return
        device = head_parameter.device
        dtype = head_parameter.dtype
        device_index = device.index if device.index is not None else torch.accelerator.current_device_index()
        condition_size = self.diffusion_sampler.condition_size
        latent_size = self.diffusion_sampler.latent_size

        logger.info(
            "Warming VibeVoice diffusion CUDA graphs for batch sizes %s.",
            list(resolved_batch_sizes),
        )
        with (
            torch.random.fork_rng(devices=[device_index], device_type=device.type),
            torch.inference_mode(),
        ):
            for batch_size in resolved_batch_sizes:
                positive_condition = torch.randn(
                    batch_size,
                    condition_size,
                    device=device,
                    dtype=dtype,
                )
                negative_condition = torch.randn_like(positive_condition)
                noise = torch.randn(
                    2 * batch_size,
                    latent_size,
                    device=device,
                    dtype=dtype,
                )
                self.sample_audio_latent(
                    positive_condition,
                    negative_condition,
                    noise,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )
                executor = self._diffusion_graph_executor
                if executor is not None and executor.disabled:
                    break
        if executor is not None and not executor.disabled:
            logger.info(
                "Warmed VibeVoice diffusion CUDA graphs for batch sizes %s.",
                list(resolved_batch_sizes),
            )

    def sample_audio_latent(
        self,
        positive_condition: torch.Tensor,
        negative_condition: torch.Tensor,
        noise: torch.Tensor,
        *,
        guidance_scale: float,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Run the model-local diffusion numerical kernel for one AR step."""
        steps = (
            self.diffusion_sampler.default_num_inference_steps
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        if self.diffusion_graph_enabled and noise.is_cuda:
            from .diffusion import VibeVoiceDiffusionGraphExecutor

            if self._diffusion_graph_executor is None:
                self._diffusion_graph_executor = VibeVoiceDiffusionGraphExecutor(
                    self.diffusion_sampler,
                    self.diffusion_head,
                    capture_failure_fatal=self.cuda_graph_capture_failure_fatal,
                )
                if self._shared_graph_pool is None:
                    self._shared_graph_pool = torch.cuda.graph_pool_handle()
                self._diffusion_graph_executor._pool = self._shared_graph_pool
            replayed = self._diffusion_graph_executor.sample(
                positive_condition,
                negative_condition,
                noise,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
            )
            if replayed is not None:
                return replayed
        return self.diffusion_sampler.sample_audio_latent(
            self.diffusion_head,
            positive_condition,
            negative_condition,
            noise,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

    def decode_audio_token(
        self,
        audio_latent: torch.Tensor,
        *,
        acoustic_cache: Any = None,
        semantic_cache: Any = None,
    ) -> VibeVoiceAudioTokenDecodeOutput:
        """Decode one acoustic latent and produce semantic AR feedback."""
        if self.decode_graph_enabled and audio_latent.is_cuda and acoustic_cache is not None:
            from .audio_decode import VibeVoiceDecodeGraphExecutor

            if self._decode_graph_executor is None:
                self._decode_graph_executor = VibeVoiceDecodeGraphExecutor(
                    self.audio_token_decoder,
                    capture_failure_fatal=self.cuda_graph_capture_failure_fatal,
                )
            replayed = self._decode_graph_executor.decode(
                audio_tower=self.audio_tower,
                semantic_encoder=self.semantic_tokenizer_encoder,
                acoustic_projector=self.multi_modal_projector,
                semantic_connector=self.semantic_connector,
                latent_scaling_factor=self.latent_scaling_factor,
                latent_bias_factor=self.latent_bias_factor,
                audio_latent=audio_latent,
                acoustic_cache=acoustic_cache,
                semantic_cache=semantic_cache,
            )
            if replayed is not None:
                return replayed
        return self.audio_token_decoder.decode_audio_token(
            audio_tower=self.audio_tower,
            semantic_encoder=self.semantic_tokenizer_encoder,
            acoustic_projector=self.multi_modal_projector,
            semantic_connector=self.semantic_connector,
            latent_scaling_factor=self.latent_scaling_factor,
            latent_bias_factor=self.latent_bias_factor,
            audio_latent=audio_latent,
            acoustic_cache=acoustic_cache,
            semantic_cache=semantic_cache,
        )


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceMultiModalProcessor,
    info=VibeVoiceProcessingInfo,
    dummy_inputs=VibeVoiceDummyInputsBuilder,
)
class VibeVoiceForConditionalGeneration(nn.Module, SupportsMultiModal):
    """vLLM VibeVoice model with reference-audio prefill support."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "audio":
            return f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.has_preprocess = True
        self.has_postprocess = True
        self.requires_omni_request_id = True
        self.requires_omni_input_token_ids_cpu = True
        self.have_multimodal_outputs = True
        # VibeVoice serves decoded waveform only. Hidden rows remain internal
        # positive conditions and must never be exposed as audio payloads.
        self.omni_pooler_payload_include_hidden = False
        # Only the final scheduled row is needed as the positive diffusion
        # condition; never reconstruct a full prefix-cache hidden span.
        self.requires_full_prefix_cached_hidden_states = False
        self.postprocess_uses_multimodal_outputs = False
        # Sparse waveform routing may include only the decode subset of a
        # mixed prefill/decode batch. Every scheduled hidden tail is still a
        # request-local positive condition required by the next AR step.
        self.postprocess_requires_all_scheduled_requests = True
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model = VibeVoiceModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        # The released checkpoint ties the output head to the token embedding
        # and consequently contains no independent lm_head tensor.
        self.lm_head = self.model.language_model.embed_tokens
        self.logits_processor = LogitsProcessor(self.config.text_config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.language_model.make_empty_intermediate_tensors

        self._stateful = VibeVoiceStatefulInference(
            audio_bos_token_id=int(self.config.audio_bos_token_id),
            audio_eos_token_id=int(self.config.audio_eos_token_id),
            audio_token_id=int(self.config.audio_token_id),
            eos_token_id=int(self.config.eos_token_id),
            latent_size=int(self.config.audio_config.hidden_size),
            condition_size=int(self.config.text_config.hidden_size),
            default_guidance_scale=VIBEVOICE_DEFAULT_GUIDANCE_SCALE,
            default_num_diffusion_steps=VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS,
        )
        self._negative_kv_branch: VibeVoiceNegativeKVBranch | None = None
        self._runtime_config = VibeVoiceRuntimeConfig.from_vllm_config(vllm_config)
        self._diffusion_graph_warmup_batch_sizes = self._runtime_config.resolve_diffusion_graph_warmup_batch_sizes(
            vllm_config.scheduler_config.max_num_seqs,
        )
        self.model.diffusion_graph_enabled = self._runtime_config.diffusion_cuda_graph
        self.model.decode_graph_enabled = self._runtime_config.decode_cuda_graph
        self.model.cuda_graph_capture_failure_fatal = self._runtime_config.cuda_graph_capture_failure_fatal
        self.named_kv_branch_request = NamedKVBranchRequest(
            name="negative",
            memory_bytes=self._runtime_config.negative_kv_cache_memory_bytes,
            activation_margin_bytes=(self._runtime_config.negative_kv_activation_margin_bytes),
        )
        self._pending_request_ids: list[str] = []
        self._pending_request_spans: list[tuple[str, int, int]] = []
        self._pending_audio_transitions: list[tuple[str, int]] = []
        self._pending_num_input_rows = 0
        # GPUARModelRunner consumes these declarative hooks only after sampling
        # a token that reaches a hard length cap. Other models pay no cost.
        self.terminal_sample_drain_token_ids = frozenset({self._stateful.audio_token_id})

    def get_language_model(self) -> Qwen2Model:
        return self.model.language_model

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens

    def warmup_side_graphs(self) -> None:
        """Pre-capture model-local graphs after runner-owned graph capture."""
        self.model.warmup_diffusion_graphs(
            self._diffusion_graph_warmup_batch_sizes,
            num_inference_steps=self._stateful.default_num_diffusion_steps,
            guidance_scale=self._stateful.default_guidance_scale,
        )

    def _get_audio_embeddings(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor,
        audio_num_tokens: torch.Tensor | None = None,
        *,
        sample: bool,
    ) -> list[torch.Tensor]:
        """Encode, project, and crop every reference-audio item.

        ``sample`` is explicit so tests can exercise a deterministic parity
        path. Runtime ``embed_multimodal`` always uses the official
        ``sample=True`` behavior.
        """
        if input_values.ndim != 3:
            raise ValueError(
                f"VibeVoice input_values must have shape (batch, channels, samples), got {tuple(input_values.shape)}."
            )
        if padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if padding_mask.ndim != 2:
            raise ValueError(
                f"VibeVoice padding_mask must have shape (batch, samples), got {tuple(padding_mask.shape)}."
            )
        if input_values.shape[0] != padding_mask.shape[0]:
            raise ValueError(
                "VibeVoice audio batch mismatch: "
                f"input_values={input_values.shape[0]}, "
                f"padding_mask={padding_mask.shape[0]}."
            )
        if input_values.shape[1] != 1:
            raise ValueError(f"VibeVoice Acoustic Encoder requires mono input, got {input_values.shape[1]} channels.")
        if input_values.shape[-1] != padding_mask.shape[-1]:
            raise ValueError(
                "VibeVoice waveform/mask length mismatch: "
                f"input_values={input_values.shape[-1]}, "
                f"padding_mask={padding_mask.shape[-1]}."
            )

        tower_param = next(self.model.audio_tower.parameters())
        input_values = input_values.to(
            device=tower_param.device,
            dtype=tower_param.dtype,
        )
        padding_mask = padding_mask.to(device=tower_param.device)
        counts_from_mask = torch.div(
            padding_mask.to(torch.long).sum(dim=-1) + AUDIO_HOP_LENGTH - 1,
            AUDIO_HOP_LENGTH,
            rounding_mode="floor",
        )
        if audio_num_tokens is None:
            audio_num_tokens = counts_from_mask
        else:
            audio_num_tokens = torch.as_tensor(
                audio_num_tokens,
                device=counts_from_mask.device,
                dtype=torch.long,
            ).reshape(-1)
            if audio_num_tokens.shape != counts_from_mask.shape or not torch.equal(
                audio_num_tokens,
                counts_from_mask,
            ):
                raise ValueError(
                    "VibeVoice audio_num_tokens does not match padding_mask: "
                    f"provided={audio_num_tokens.tolist()}, "
                    f"expected={counts_from_mask.tolist()}."
                )

        with torch.no_grad():
            acoustic_latents = self.model.audio_tower.encode(
                input_values,
                sample=sample,
            ).latents
            acoustic_features = (
                acoustic_latents + self.model.latent_bias_factor.to(acoustic_latents.device)
            ) * self.model.latent_scaling_factor.to(acoustic_latents.device)
            projected = self.model.multi_modal_projector(acoustic_features)

        if projected.ndim != 3 or projected.shape[0] != input_values.shape[0]:
            raise ValueError(f"VibeVoice Acoustic Encoder returned an unexpected shape: {tuple(projected.shape)}.")

        embeddings: list[torch.Tensor] = []
        for item_idx, num_tokens in enumerate(audio_num_tokens.tolist()):
            if num_tokens < 1 or num_tokens > projected.shape[1]:
                raise ValueError(
                    f"VibeVoice audio item {item_idx} requires {num_tokens} "
                    f"embeddings, but the encoder produced {projected.shape[1]}."
                )
            item = projected[item_idx, :num_tokens]
            if item.shape[0] != num_tokens:
                raise AssertionError(
                    "VibeVoice audio embedding/placeholder length mismatch: "
                    f"item={item_idx}, embeddings={item.shape[0]}, "
                    f"placeholders={num_tokens}."
                )
            embeddings.append(item)
        return embeddings

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor]:
        input_values, padding_mask = _pad_ragged_audio_batch(
            kwargs.get("input_values"),
            kwargs.get("padding_mask"),
        )
        audio_num_tokens = kwargs.get("audio_num_tokens")
        if audio_num_tokens is not None:
            audio_num_tokens = _flatten_audio_token_counts(audio_num_tokens)
        return self._get_audio_embeddings(
            input_values,
            padding_mask,
            audio_num_tokens,
            sample=True,
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or is_multimodal is None:
            return inputs_embeds
        return merge_multimodal_embeddings(
            inputs_embeds,
            multimodal_embeddings,
            is_multimodal,
        )

    def bind_named_kv_branch(
        self,
        store: NamedCausalKVBranch,
    ) -> None:
        """Wrap and bind the runner-owned negative-Qwen PagedAttention store."""
        if self._negative_kv_branch is not None:
            raise RuntimeError("VibeVoice negative KV branch was bound twice.")
        branch = VibeVoiceNegativeBranch(
            store=store,
            language_model=self.model.language_model,
            hidden_size=int(self.config.text_config.hidden_size),
        )
        self._stateful.bind_negative_branch(branch)
        self._negative_kv_branch = branch

    def record_negative_condition(
        self,
        request_id: str,
        condition: torch.Tensor,
    ) -> None:
        """Publish one aligned hidden row from the negative Qwen branch."""
        self._stateful.record_negative_condition(request_id, condition)

    @staticmethod
    def _input_token_ids_cpu(
        input_ids: torch.Tensor,
        info_dict: dict[str, Any],
    ) -> tuple[int, ...]:
        values = info_dict.get(OMNI_INPUT_TOKEN_IDS_CPU_KEY)
        expected_count = int(input_ids.numel())
        if not isinstance(values, (list, tuple)) or len(values) != expected_count:
            actual_count = len(values) if isinstance(values, (list, tuple)) else None
            raise ValueError(
                "VibeVoice preprocess requires request-aligned "
                f"_omni_input_token_ids_cpu (expected {expected_count}, got {actual_count})."
            )
        return tuple(int(value) for value in values)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply control-token transitions and continuous feedback embeddings."""
        request_id = info_dict.get(OMNI_REQUEST_ID_KEY)
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("VibeVoice preprocess requires a non-empty _omni_req_id.")
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        input_token_ids_cpu = self._input_token_ids_cpu(input_ids, info_dict)
        is_prefill = bool(info_dict.get(OMNI_IS_PREFILL_KEY, input_ids.numel() > 1))
        num_computed = int(info_dict.get(OMNI_NUM_COMPUTED_TOKENS_KEY, 0) or 0)
        prompt_len = int(info_dict.get(OMNI_PROMPT_LEN_KEY, input_ids.numel()) or 0)
        state = self._stateful.get_or_create(
            request_id,
            reset=is_prefill and num_computed == 0,
        )

        if is_prefill:
            # Serving prompts already end in audio BOS. Initialize the segment
            # at the final prefill chunk so the first sampled audio token is a
            # valid transition, matching Transformers generation.
            is_final_prefill = num_computed + int(input_ids.numel()) >= prompt_len
            if (
                is_final_prefill
                and input_token_ids_cpu
                and input_token_ids_cpu[-1] == self._stateful.audio_bos_token_id
            ):
                self._stateful.start_audio_segment(state.request_id)
        elif len(input_token_ids_cpu) == 1:
            token_id = input_token_ids_cpu[0]
            if token_id == self._stateful.audio_token_id:
                # Defer audio-token decode so all active audio-token requests
                # step consume one official [2B, latent] RNG draw.
                self._pending_audio_transitions.append((request_id, self._pending_num_input_rows))
            else:
                next_embedding, _ = self._stateful.process_sampled_token(
                    request_id=request_id,
                    token_id=token_id,
                    token_embedding=input_embeds.reshape(1, -1),
                    kernel=self.model,
                )
                input_embeds = next_embedding

        span_start = self._pending_num_input_rows
        span_end = span_start + int(input_ids.numel())
        self._pending_request_ids.append(request_id)
        self._pending_request_spans.append((request_id, span_start, span_end))
        self._pending_num_input_rows = span_end
        return input_ids, input_embeds, {OMNI_REQUEST_ID_KEY: request_id}

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        **info_dict: Any,
    ) -> dict[str, Any]:
        """Retain only the positive hidden row needed by the next transition."""
        request_id = info_dict.get(OMNI_REQUEST_ID_KEY)
        if not isinstance(request_id, str) or not request_id:
            return {}
        if hidden_states.numel() > 0:
            condition = hidden_states[-1].detach().reshape(1, -1).contiguous()
            self._stateful.record_positive_condition(request_id, condition)
        self._stateful.finish_postprocess(request_id)
        return {OMNI_REQUEST_ID_KEY: request_id}

    def on_requests_finished(
        self,
        finished_req_ids: Collection[str],
        *,
        scheduled_req_ids: Collection[str] = (),
    ) -> None:
        self._stateful.on_requests_finished(
            finished_req_ids,
            scheduled_req_ids=scheduled_req_ids,
        )

    def clear_runtime_state(self) -> None:
        """Release all request-owned state before runner resource teardown."""
        self._stateful.clear()
        self._pending_request_ids = []
        self._pending_request_spans = []
        self._pending_audio_transitions = []
        self._pending_num_input_rows = 0

    def _warn_if_named_kv_capability_unavailable(self) -> None:
        if (
            not bool(
                getattr(
                    self,
                    "named_kv_branch_capability_acknowledged",
                    False,
                )
            )
            and self._negative_kv_branch is None
        ):
            logger.warning_once(
                "VibeVoice waveform generation requires the vLLM-Omni named-KV "
                "runner capability. This runner did not acknowledge or bind "
                "that capability; stock vllm.LLM may emit tokens but cannot "
                "execute the complete VibeVoice waveform path. Use "
                "AsyncOmni with GPUARModelRunner."
            )

    def drain_terminal_sampled_tokens(
        self,
        request_ids: list[str],
        multimodal_outputs: Any,
    ) -> dict[str, Any]:
        """Decode hard-capped sampled audio tokens without another AR step."""
        if not request_ids:
            return {}
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("VibeVoice terminal drain contains duplicate request IDs.")
        if self._negative_kv_branch is None:
            raise RuntimeError("VibeVoice terminal audio-token drain requires the bound negative Qwen branch.")

        negative_inputs: list[torch.Tensor] = []
        for request_id in request_ids:
            state = self._stateful.get(request_id)
            negative_input = state.negative_input_embedding if state is not None else None
            if negative_input is None:
                raise RuntimeError(
                    f"VibeVoice terminal audio-token drain has no preceding input embedding for request {request_id!r}."
                )
            negative_inputs.append(negative_input)

        negative_conditions = self._negative_kv_branch.forward_step(
            request_ids,
            negative_inputs,
        )
        if len(negative_conditions) != len(request_ids):
            raise RuntimeError("VibeVoice terminal negative branch returned the wrong number of conditions.")
        for request_id, condition in zip(
            request_ids,
            negative_conditions,
            strict=True,
        ):
            self._stateful.record_negative_condition(request_id, condition)

        token_embedding_batch = self.get_input_embeddings()(
            torch.full(
                (len(request_ids),),
                self._stateful.audio_token_id,
                device=negative_inputs[0].device,
                dtype=torch.long,
            )
        )
        # Match the regular forward path exactly: preserve first-seen group
        # order and issue one official [2B, latent] RNG draw per shared
        # (guidance_scale, num_diffusion_steps) contract.
        transition_groups: dict[tuple[float, int], list[int]] = {}
        for index, request_id in enumerate(request_ids):
            state = self._stateful.get(request_id)
            if state is None:
                raise RuntimeError(f"Missing VibeVoice terminal request state for {request_id!r}.")
            transition_groups.setdefault(
                (state.guidance_scale, state.num_diffusion_steps),
                [],
            ).append(index)
        for indices in transition_groups.values():
            group_request_ids = [request_ids[index] for index in indices]
            group_embeddings = [token_embedding_batch[index : index + 1] for index in indices]
            self._stateful.process_audio_tokens_batch(
                request_ids=group_request_ids,
                token_embeddings=group_embeddings,
                kernel=self.model,
            )

        terminal_waveforms: dict[str, torch.Tensor] = {}
        for request_id in request_ids:
            waveform = self._stateful.drain_waveform_chunks(request_id)
            if waveform is None:
                raise RuntimeError(f"VibeVoice terminal audio-token drain produced no waveform for {request_id!r}.")
            terminal_waveforms[request_id] = (
                waveform.detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .contiguous()
            )
            # A hard cap has no later preprocess/safe point. Its final negative
            # condition is consumed, so release Paged KV immediately; deferred
            # parent cleanup remains idempotent for the other side state.
            self._negative_kv_branch.free(request_id)

        if not multimodal_outputs:
            merged_audio: list[torch.Tensor] = []
            merged_sample_rates: list[torch.Tensor] = []
            merged_request_ids: list[str] = []
            meta: dict[str, Any] = {}
        else:
            if not isinstance(multimodal_outputs, dict):
                raise TypeError("VibeVoice terminal drain requires dictionary multimodal output.")
            meta_value = multimodal_outputs.get("meta")
            audio = multimodal_outputs.get("audio")
            sample_rates = multimodal_outputs.get("sr")
            if (
                not isinstance(meta_value, dict)
                or not isinstance(meta_value.get("req_id"), list)
                or not isinstance(audio, list)
                or not isinstance(sample_rates, list)
                or len(audio) != len(meta_value["req_id"])
                or len(sample_rates) != len(audio)
            ):
                raise ValueError("VibeVoice terminal drain received malformed sparse audio output.")
            meta = meta_value
            merged_audio = list(audio)
            merged_sample_rates = list(sample_rates)
            merged_request_ids = list(meta["req_id"])

        for request_id in request_ids:
            terminal_waveform = terminal_waveforms[request_id]
            if request_id in merged_request_ids:
                index = merged_request_ids.index(request_id)
                prior_waveform = merged_audio[index]
                if not isinstance(prior_waveform, torch.Tensor):
                    raise TypeError("VibeVoice sparse audio output must contain tensors.")
                merged_audio[index] = torch.cat(
                    [prior_waveform.to(terminal_waveform), terminal_waveform],
                    dim=0,
                ).contiguous()
            else:
                merged_request_ids.append(request_id)
                merged_audio.append(terminal_waveform)
                merged_sample_rates.append(torch.tensor(SAMPLE_RATE, dtype=torch.int32))
        return {
            **(multimodal_outputs or {}),
            "audio": merged_audio,
            "sr": merged_sample_rates,
            "meta": {
                **meta,
                "req_id": merged_request_ids,
                "sparse_audio": ["1"],
                "audio_chunk_semantics": ["delta" for _ in merged_audio],
            },
        }

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Publish each newly decoded mono 24 kHz waveform chunk once."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        runtime_info = kwargs.get("model_intermediate_buffer")
        if runtime_info is None:
            runtime_info = kwargs.get("runtime_additional_information")
        if runtime_info is None:
            runtime_info = []
        if not isinstance(runtime_info, list):
            raise TypeError("VibeVoice make_omni_output requires request-aligned runtime information.")

        ready_request_ids: list[str] = []
        audio_chunks: list[torch.Tensor] = []
        for info in runtime_info:
            if not isinstance(info, dict):
                continue
            request_id = info.get(OMNI_REQUEST_ID_KEY)
            if not isinstance(request_id, str) or not request_id:
                continue
            waveform = self._stateful.drain_waveform_chunks(request_id)
            if waveform is None:
                continue
            if waveform.ndim != 1 or not waveform.is_floating_point():
                raise ValueError("VibeVoice published waveform must be a one-dimensional floating-point tensor.")
            ready_request_ids.append(request_id)
            audio_chunks.append(waveform.detach().to(device="cpu", dtype=torch.float32).contiguous())

        multimodal_outputs: dict[str, Any] = {}
        if audio_chunks:
            sample_rate = torch.tensor(SAMPLE_RATE, dtype=torch.int32)
            multimodal_outputs = {
                "audio": audio_chunks,
                "sr": [sample_rate for _ in audio_chunks],
                "meta": {
                    "req_id": ready_request_ids,
                    "sparse_audio": ["1"],
                    "audio_chunk_semantics": ["delta" for _ in audio_chunks],
                },
            }
        return OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs=multimodal_outputs,
        )

    def preprocess_finalize(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        req_ids: list[str],
        sampling_extra_args: list[dict[str, Any]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run audio-token transitions before the graph-capturable forward.

        Moving the negative-branch advance, diffusion, audio decode,
        embed splice, and negative-input recording out of ``forward()`` makes
        ``forward()`` a pure ``self.model(...)`` call that vLLM can capture as a
        FULL decode CUDA graph. The deferred-RNG ordering is preserved because
        the runner calls this hook after every per-request ``preprocess``
        call, so the complete scheduled set is known.
        """
        pending_request_ids = self._pending_request_ids
        pending_request_spans = self._pending_request_spans
        pending_audio_transitions = self._pending_audio_transitions
        alignment_error: str | None = None
        if pending_request_ids != req_ids:
            alignment_error = (
                "VibeVoice preprocess request metadata is misaligned: "
                f"preprocess={pending_request_ids!r}, runner={req_ids!r}."
            )
        elif sampling_extra_args is not None and len(sampling_extra_args) != len(pending_request_ids):
            alignment_error = (
                "VibeVoice sampling controls are misaligned: "
                f"requests={len(pending_request_ids)}, controls={len(sampling_extra_args)}."
            )
        self._stateful.flush_deferred_cleanup(
            exclude_request_ids=set(pending_request_ids),
        )
        self._pending_request_ids = []
        self._pending_request_spans = []
        self._pending_audio_transitions = []
        self._pending_num_input_rows = 0
        if alignment_error is not None:
            raise ValueError(alignment_error)
        if sampling_extra_args is not None:
            for request_id, extra_args in zip(
                pending_request_ids,
                sampling_extra_args,
                strict=True,
            ):
                self._stateful.set_runtime_controls(request_id, extra_args)

        if pending_audio_transitions and inputs_embeds is not None:
            if self._negative_kv_branch is not None:
                negative_request_ids = [request_id for request_id, _ in pending_audio_transitions]
                negative_inputs: list[torch.Tensor] = []
                for request_id in negative_request_ids:
                    state = self._stateful.get(request_id)
                    negative_input = state.negative_input_embedding if state is not None else None
                    if negative_input is None:
                        raise RuntimeError(
                            "VibeVoice negative Qwen branch has no preceding "
                            f"input embedding for request {request_id!r}."
                        )
                    negative_inputs.append(negative_input)
                negative_conditions = self._negative_kv_branch.forward_step(
                    negative_request_ids,
                    negative_inputs,
                )
                if len(negative_conditions) != len(negative_request_ids):
                    raise RuntimeError(
                        "VibeVoice negative Qwen branch returned a condition batch with the wrong length."
                    )
                for request_id, condition in zip(
                    negative_request_ids,
                    negative_conditions,
                    strict=True,
                ):
                    self._stateful.record_negative_condition(
                        request_id,
                        condition,
                    )

            transition_groups: dict[
                tuple[float, int],
                list[tuple[str, int]],
            ] = {}
            for request_id, row_offset in pending_audio_transitions:
                state = self._stateful.get(request_id)
                if state is None:
                    raise RuntimeError(f"Missing VibeVoice request state for {request_id!r}.")
                transition_groups.setdefault(
                    (state.guidance_scale, state.num_diffusion_steps),
                    [],
                ).append((request_id, row_offset))

            for transitions in transition_groups.values():
                request_ids = [item[0] for item in transitions]
                offsets = [item[1] for item in transitions]
                token_embeddings = [inputs_embeds[offset : offset + 1] for offset in offsets]
                next_embeddings, _ = self._stateful.process_audio_tokens_batch(
                    request_ids=request_ids,
                    token_embeddings=token_embeddings,
                    kernel=self.model,
                )
                offset_tensor = torch.tensor(
                    offsets,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )
                inputs_embeds.index_copy_(
                    0,
                    offset_tensor,
                    torch.cat(next_embeddings, dim=0).to(inputs_embeds),
                )

        if inputs_embeds is not None:
            for request_id, span_start, span_end in pending_request_spans:
                state = self._stateful.get(request_id)
                if state is not None and state.in_audio_segment and span_end > span_start:
                    self._stateful.record_negative_input_embedding(
                        request_id,
                        inputs_embeds[span_end - 1 : span_end],
                    )
        return input_ids, inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_extra_args: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Run the positive AR branch for Delta waveform generation.

        Every decoded audio-token transition publishes only its newly generated
        3,200-sample mono float32 chunk. Request-level output processing owns
        final concatenation; this model never republishes cumulative waveform
        history.
        """
        self._warn_if_named_kv_capability_unavailable()
        # All audio-token transition work (negative branch,
        # diffusion, audio decode, embed splice, negative-input recording) has
        # moved to preprocess_finalize, which the runner calls before this
        # forward. sampling_extra_args is now consumed there; it is accepted
        # here only for backward compatibility with runners that have not
        # adopted the hook yet.
        if self._pending_request_ids:
            self.preprocess_finalize(
                input_ids=input_ids if input_ids is not None else torch.empty(0),
                inputs_embeds=inputs_embeds,
                req_ids=self._pending_request_ids,
                sampling_extra_args=sampling_extra_args,
            )
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = _build_vibevoice_weights_mapper(self.config)
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)


__all__ = [
    "VibeVoiceAudioTokenDecodeOutput",
    "VibeVoiceAudioTokenDecoder",
    "VibeVoiceDiffusionHead",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceModel",
    "VibeVoiceMultiModalProjector",
    "VibeVoiceNegativeBranch",
    "VibeVoiceRMSNorm",
    "_build_vibevoice_weights_mapper",
]

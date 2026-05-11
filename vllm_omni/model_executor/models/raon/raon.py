# SPDX-License-Identifier: Apache-2.0

"""Raon vLLM model: AR thinker + talker with audio codec integration."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio.functional
from torch import nn
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed.parallel_state import get_pp_group
from vllm.inputs import ModalityData, MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    AudioItem,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import MultiModalProcessingInfo
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.raon.raon_adaptors import (
    EmbeddingAdaptor,
    ThinkerToTalkerProjection,
)
from vllm_omni.model_executor.models.raon.raon_audio_encoder import (
    Qwen3OmniAuTWrapper,
    compute_num_audio_input_tokens,
    compute_samples_per_frame,
)
from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import StreamingMimiModel
from vllm_omni.model_executor.models.raon.raon_code_predictor import (
    RaonCodePredictor,
    RepetitionAwareSampler,
)
from vllm_omni.model_executor.models.raon.raon_speaker_encoder import (
    PretrainedSpeakerEncoder,
    build_speaker_encoder,
    compute_speaker_embeds,
    load_speaker_ref_audio,
    normalize_speaker_ref_audio,
)
from vllm_omni.model_executor.models.raon.raon_utils import (
    IGNORE_WEIGHT_SUFFIXES,
    coerce_optional_int,
    collapse_exact_repeated_codec_snapshot,
    module_device,
    module_dtype,
    normalize_runtime_request_id,
    strip_raon_audio_markers,
    unwrap_singleton_list,
)
from vllm_omni.model_executor.models.raon.serving_utils import RaonServingHooks
from vllm_omni.tokenizers.raon_tokenizer import (
    AUDIO_END,
    AUDIO_END_TOKEN,
    AUDIO_INPUT_PAD_TOKEN,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_OPEN_SEQ,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_OUTPUT_PLACEHOLDER_SEQ,
    AUDIO_PLACEHOLDER_PATTERN,
    AUDIO_PLACEHOLDER_SEQ,
    AUDIO_START,
    AUDIO_START_TOKEN,
    LEGACY_AUDIO_PLACEHOLDER_SEQ,
    USER_PROMPT_MARKER,
    RaonResolvedIds,
    align_tokenizer,
    inject_placeholders_into_str,
    inject_placeholders_into_token_ids,
    resolve_raon_special_ids,
    resolve_speaker_token_id,
)
from vllm_omni.transformers_utils.configs.raon import (
    ENV,
    REQUEST_STATE_CLEANUP_KEYS,
    REQUEST_STATE_CLEANUP_PREFIXES,
    SpeakerEncoderConfig,
    coerce_speaker_encoder_config,
    get_mimi_frame_rate,
)

logger = init_logger(__name__)

RaonServingHooks.apply_default_modalities()


# Token ID used by older checkpoints that encoded audio pads as <|audio_pad|>
# (vocab index 151673) instead of the current AUDIO_INPUT_PLACEHOLDER token.
LEGACY_AUDIO_PAD_TOKEN_ID = 151673
LEGACY_AUDIO_PLACEHOLDER_TOKEN_IDS = [
    AUDIO_START.id,
    LEGACY_AUDIO_PAD_TOKEN_ID,
    AUDIO_END.id,
]


def build_audio_input_placeholder_ids(
    *,
    num_audio_tokens: int,
    audio_start_token_id: int,
    audio_input_token_id: int,
    audio_end_token_id: int,
    close_with_end: bool = True,
) -> list[int]:
    if num_audio_tokens <= 0:
        raise ValueError(f"num_audio_tokens must be positive, got {num_audio_tokens}.")

    replacement_ids = [
        audio_start_token_id,
        *([audio_input_token_id] * num_audio_tokens),
    ]
    if close_with_end:
        replacement_ids.append(audio_end_token_id)
    return replacement_ids


def _infer_audio_placeholder_token_ids(
    prompt: str,
    *,
    num_audios: int,
    audio_input_token_id: int,
    audio_output_token_id: int,
) -> torch.Tensor:
    """Detect per-audio placeholder type from the prompt text."""
    token_ids: list[int] = []
    for match in AUDIO_PLACEHOLDER_PATTERN.finditer(prompt):
        text = match.group(0)
        if text in (AUDIO_PLACEHOLDER_SEQ, LEGACY_AUDIO_PLACEHOLDER_SEQ):
            token_ids.append(audio_input_token_id)
        else:
            token_ids.append(audio_output_token_id)

    if len(token_ids) != num_audios:
        logger.warning(
            "Expected %d audio placeholders, found %d; trimming/defaulting to input",
            num_audios,
            len(token_ids),
        )
        if len(token_ids) > num_audios:
            token_ids = token_ids[:num_audios]
        else:
            token_ids.extend([audio_input_token_id] * (num_audios - len(token_ids)))

    return torch.tensor(token_ids, dtype=torch.long)


def normalize_audio_waveforms_and_lengths(
    audio_waveforms: torch.Tensor | list[Any],
    audio_lengths: torch.Tensor | list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch and pad waveforms to ``[N, T]`` float32 and align lengths."""
    if isinstance(audio_waveforms, list):
        waveforms = [torch.as_tensor(w, dtype=torch.float32).reshape(-1) for w in audio_waveforms]
        max_len = max((w.shape[0] for w in waveforms), default=0)
        waveforms = [F.pad(w, (0, max_len - w.shape[0])) if w.shape[0] < max_len else w for w in waveforms]
        if waveforms:
            stacked = torch.stack(waveforms, dim=0)
        else:
            stacked = torch.empty((0, 0), dtype=torch.float32)
    elif isinstance(audio_waveforms, torch.Tensor):
        if audio_waveforms.ndim == 1:
            stacked = audio_waveforms.unsqueeze(0)
        elif audio_waveforms.ndim == 2:
            stacked = audio_waveforms
        else:
            raise ValueError(
                f"audio_waveforms must be 1D/2D tensor or list of 1D tensors, got shape={tuple(audio_waveforms.shape)}."
            )
        stacked = stacked.to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported audio_waveforms type: {type(audio_waveforms)}")

    num_audios = int(stacked.shape[0])
    if audio_lengths is None:
        lengths = torch.full((num_audios,), int(stacked.shape[1]), dtype=torch.long)
    elif isinstance(audio_lengths, list):
        lengths = torch.as_tensor(audio_lengths, dtype=torch.long)
    elif isinstance(audio_lengths, torch.Tensor):
        lengths = audio_lengths.to(dtype=torch.long).reshape(-1)
    else:
        raise ValueError(f"Unsupported audio_lengths type: {type(audio_lengths)}")

    if int(lengths.shape[0]) != num_audios:
        raise ValueError(f"audio_lengths size mismatch: got {int(lengths.shape[0])}, expected {num_audios}.")
    return stacked, lengths


def flatten_audio_embeddings(
    multimodal_embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    *,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(multimodal_embeddings, torch.Tensor):
        if multimodal_embeddings.ndim == 2:
            flat = multimodal_embeddings
        elif multimodal_embeddings.ndim == 3:
            flat = multimodal_embeddings.reshape(-1, multimodal_embeddings.shape[-1])
        else:
            raise ValueError(
                f"multimodal_embeddings tensor must be 2D or 3D, got shape={tuple(multimodal_embeddings.shape)}."
            )
    else:
        if len(multimodal_embeddings) == 0:
            return torch.empty((0, hidden_size), device=device, dtype=dtype)
        flat = torch.cat(multimodal_embeddings, dim=0)

    if flat.shape[-1] != hidden_size:
        raise ValueError(f"audio embedding hidden size mismatch: expected {hidden_size}, got {flat.shape[-1]}.")
    return flat.to(device=device, dtype=dtype)


def scatter_audio_input_embeddings(
    *,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_input_embeddings: torch.Tensor,
    audio_input_token_id: int,
    is_multimodal: torch.Tensor | None,
) -> torch.Tensor:
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape={tuple(input_ids.shape)}.")
    if inputs_embeds.ndim != 2:
        raise ValueError(f"inputs_embeds must be 2D, got shape={tuple(inputs_embeds.shape)}.")

    audio_mask = input_ids == audio_input_token_id
    if is_multimodal is not None:
        is_multimodal = is_multimodal.to(device=input_ids.device, non_blocking=True)
        non_audio_mm = is_multimodal & (~audio_mask)
        if bool(non_audio_mm.any().item()):
            raise ValueError(
                "is_multimodal marks tokens that are not audio_input_token_id. "
                "This model only supports audio multimodal placeholders."
            )
        audio_mask = audio_mask & is_multimodal

    num_placeholders = int(audio_mask.sum().item())
    num_embeddings = int(audio_input_embeddings.shape[0])
    if num_placeholders != num_embeddings:
        raise ValueError(
            "Audio embedding alignment error: "
            f"found {num_placeholders} audio_input_token_id placeholders but "
            f"got {num_embeddings} audio embedding frames."
        )

    if num_placeholders == 0:
        return inputs_embeds

    if audio_input_embeddings.ndim != 2:
        raise ValueError(
            f"audio_input_embeddings must be 2D after flattening, got shape={tuple(audio_input_embeddings.shape)}."
        )

    expanded_mask = audio_mask[:, None].expand_as(inputs_embeds)
    if int(expanded_mask.sum().item()) != int(audio_input_embeddings.numel()):
        raise ValueError(
            "Audio embedding scatter shape mismatch: "
            f"mask_elements={int(expanded_mask.sum().item())}, "
            f"embedding_elements={int(audio_input_embeddings.numel())}."
        )
    return inputs_embeds.masked_scatter(expanded_mask, audio_input_embeddings)


def _raon_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    return {
        "audio_waveforms": MultiModalFieldConfig.batched("audio"),
        "audio_lengths": MultiModalFieldConfig.batched("audio"),
        "audio_placeholder_token_ids": MultiModalFieldConfig.batched("audio"),
    }


class RaonMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_waveforms", "audio_lengths"},
                fields_factory=_raon_field_config,
            )
        return super()._parse_audio_data(data)


class RaonProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        hf_config = self.get_hf_config()
        sampling_rate = int(hf_config.audio_tokenizer_config.sampling_rate)
        return RaonMultiModalDataParser(
            target_sr=float(sampling_rate),
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class RaonDummyInputsBuilder(BaseDummyInputsBuilder[RaonProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return AUDIO_PLACEHOLDER_SEQ * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        hf_config = self.info.get_hf_config()
        sampling_rate = int(hf_config.audio_tokenizer_config.sampling_rate)
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(
                length=sampling_rate,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        }


class RaonMultiModalProcessor(BaseMultiModalProcessor[RaonProcessingInfo]):
    @cached_property
    def _ids(self) -> RaonResolvedIds:
        """Resolve Raon special-token IDs from the live tokenizer once."""
        tokenizer = self.info.get_tokenizer()
        return resolve_raon_special_ids(tokenizer)

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        num_audios = mm_items.get_count("audio", strict=False)
        if num_audios == 0:
            return BatchFeature({}, tensor_type="pt")

        audios = mm_items.get_items("audio", AudioProcessorItems)
        waveform_tensors: list[torch.Tensor] = []
        lengths: list[int] = []
        for i in range(num_audios):
            audio = audios.get(i)
            tensor = torch.as_tensor(audio, dtype=torch.float32).reshape(-1)
            waveform_tensors.append(tensor)
            lengths.append(int(tensor.shape[0]))

        audio_waveforms, audio_lengths = normalize_audio_waveforms_and_lengths(waveform_tensors, lengths)
        return BatchFeature(
            {
                "audio_waveforms": audio_waveforms,
                "audio_lengths": audio_lengths,
            },
            tensor_type="pt",
        )

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], BatchFeature, bool]:
        del enable_hf_prompt_update

        num_audios = mm_items.get_count("audio", strict=False)
        if isinstance(prompt, str):
            tokenizer = self.info.get_tokenizer()
            align_tokenizer(tokenizer)
            prompt = inject_placeholders_into_str(prompt, num_audios=num_audios)
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = self._apply_hf_processor_tokens_only(prompt)
            if num_audios > 0:
                tokenizer = self.info.get_tokenizer()
                align_tokenizer(tokenizer)
                ph_ids = tokenizer.encode(AUDIO_PLACEHOLDER_SEQ, add_special_tokens=False)
                legacy_ph_ids = tokenizer.encode(
                    f"{AUDIO_START_TOKEN}<|audio_pad|>{AUDIO_END_TOKEN}",
                    add_special_tokens=False,
                )
                marker_ids = tokenizer.encode(f"{USER_PROMPT_MARKER}", add_special_tokens=False)
                prompt_ids = inject_placeholders_into_token_ids(
                    prompt_ids,
                    num_audios=num_audios,
                    ph_ids=ph_ids,
                    legacy_ph_ids=legacy_ph_ids,
                    marker_ids=marker_ids,
                )

        mm_processed_data = self._apply_hf_processor_mm_only(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )
        if num_audios > 0:
            tokenizer = self.info.get_tokenizer() if not isinstance(prompt, str) else tokenizer
            align_tokenizer(tokenizer)
            audio_input_tid = int(tokenizer.encode(AUDIO_INPUT_PLACEHOLDER.text, add_special_tokens=False)[0])
            audio_output_tid = int(tokenizer.encode(AUDIO_OUTPUT_PLACEHOLDER.text, add_special_tokens=False)[0])
            prompt_text = prompt if isinstance(prompt, str) else tokenizer.decode(prompt_ids)
            mm_processed_data["audio_placeholder_token_ids"] = _infer_audio_placeholder_token_ids(
                prompt_text,
                num_audios=num_audios,
                audio_input_token_id=audio_input_tid,
                audio_output_token_id=audio_output_tid,
            )
        return prompt_ids, mm_processed_data, False

    def _build_cache_signature_prompt_text(
        self,
        prompt: str | list[int],
        num_audios: int,
    ) -> str:
        """Mirror prompt injection so cache signatures see post-processed placeholders."""
        if isinstance(prompt, str):
            tokenizer = self.info.get_tokenizer()
            align_tokenizer(tokenizer)
            return inject_placeholders_into_str(prompt, num_audios=num_audios)

        # Token-list path — mirror the exact sequence from
        # ``_apply_hf_processor_main`` so variants resolve identically.
        prompt_ids = self._apply_hf_processor_tokens_only(prompt)
        if num_audios > 0:
            tokenizer = self.info.get_tokenizer()
            align_tokenizer(tokenizer)
            ph_ids = tokenizer.encode(AUDIO_PLACEHOLDER_SEQ, add_special_tokens=False)
            legacy_ph_ids = tokenizer.encode(
                f"{AUDIO_START_TOKEN}<|audio_pad|>{AUDIO_END_TOKEN}",
                add_special_tokens=False,
            )
            marker_ids = tokenizer.encode(f"{USER_PROMPT_MARKER}", add_special_tokens=False)
            prompt_ids = inject_placeholders_into_token_ids(
                prompt_ids,
                num_audios=num_audios,
                ph_ids=ph_ids,
                legacy_ph_ids=legacy_ph_ids,
                marker_ids=marker_ids,
            )
        else:
            tokenizer = self.info.get_tokenizer()
            align_tokenizer(tokenizer)
        return tokenizer.decode(prompt_ids)

    @staticmethod
    def _extract_audio_placeholder_variants(
        prompt_text: str,
        num_audios: int,
    ) -> tuple[str, ...]:
        """Classify post-injection placeholders for cache signatures."""
        variants: list[str] = []
        for match in AUDIO_PLACEHOLDER_PATTERN.finditer(prompt_text):
            text = match.group(0)
            if text == AUDIO_PLACEHOLDER_SEQ:
                variants.append("input")
            elif text == LEGACY_AUDIO_PLACEHOLDER_SEQ:
                variants.append("legacy_input")
            elif text == AUDIO_OUTPUT_PLACEHOLDER_SEQ:
                variants.append("output_closed")
            elif text == AUDIO_OUTPUT_OPEN_SEQ:
                variants.append("output_open")
            else:
                variants.append("unknown")

        if len(variants) != num_audios:
            return (
                f"mismatch:expected={num_audios}:found={len(variants)}",
                *variants,
            )
        return tuple(variants)

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        """Salt multimodal cache keys with per-audio placeholder variants."""
        num_audios = inputs.mm_data_items.get_count("audio", strict=False)
        if num_audios == 0:
            return super()._cached_apply_hf_processor(inputs, timing_ctx)

        prompt_text = self._build_cache_signature_prompt_text(inputs.prompt, num_audios)
        variants = self._extract_audio_placeholder_variants(prompt_text, num_audios)

        # Per-item UUIDs keep both audio identity and prompt role in the cache key.
        # A request-level variant tuple would invalidate sibling items unnecessarily.
        mismatch_prefix = bool(variants) and variants[0].startswith("mismatch:")
        mismatch_marker = variants[0] if mismatch_prefix else ""

        audio_data_items = inputs.mm_data_items.get("audio")
        items_for_hash = list(audio_data_items.get_all_items_for_hash()) if audio_data_items is not None else []

        existing_uuids = (inputs.mm_uuid_items or {}).get("audio") or [None] * num_audios
        salted_audio_uuids: list[str] = []
        for i in range(num_audios):
            if i < len(items_for_hash):
                content_hash = MultiModalHasher.hash_kwargs(audio=items_for_hash[i])
            else:
                content_hash = f"no-item-{i}"
            if mismatch_prefix:
                # variants = (mismatch_marker, v0, v1, ...) — shift index by 1
                per_item_variant = variants[i + 1] if i + 1 < len(variants) else "unknown"
                variant_field = f"{mismatch_marker}|{per_item_variant}"
            else:
                per_item_variant = variants[i] if i < len(variants) else "unknown"
                variant_field = per_item_variant
            caller_base = existing_uuids[i] or ""
            prefix = f"{caller_base}|" if caller_base else ""
            salted_audio_uuids.append(f"{prefix}raon-mm-v2|content={content_hash}|idx={i}|variant={variant_field}")

        new_uuids: dict[str, list[str | None] | None] = {
            modality: list(uuids) if uuids is not None else None
            for modality, uuids in (inputs.mm_uuid_items or {}).items()
        }
        new_uuids["audio"] = salted_audio_uuids

        # Version marker for future cache-signature schema changes.
        new_kwargs = dict(inputs.hf_processor_mm_kwargs)
        new_kwargs["_raon_mm_sig_version"] = "v2"

        inputs = dataclasses.replace(
            inputs,
            hf_processor_mm_kwargs=new_kwargs,
            mm_uuid_items=new_uuids,
        )
        return super()._cached_apply_hf_processor(inputs, timing_ctx)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _raon_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_audios = mm_items.get_count("audio", strict=False)
        if num_audios == 0:
            return []

        hf_config = self.info.get_hf_config()
        atc = hf_config.audio_tokenizer_config
        sampling_rate = int(atc.sampling_rate)
        frame_rate = get_mimi_frame_rate(atc)

        ids = self._ids
        audio_start_token_id = ids.audio_start
        audio_end_token_id = ids.audio_end
        audio_input_token_id = ids.audio_input_placeholder
        audio_output_token_id = ids.audio_output_placeholder

        audios = mm_items.get_items("audio", AudioProcessorItems)

        def _build_replacement(
            item_idx: int,
            *,
            placeholder_token_id: int,
            close_with_end: bool,
        ) -> PromptUpdateDetails[list[int]]:
            if hasattr(audios, "get_audio_length"):
                audio_len_samples = int(audios.get_audio_length(item_idx))
            else:
                audio = audios.get(item_idx)
                audio_len_samples = int(len(audio))
            num_audio_tokens = compute_num_audio_input_tokens(
                audio_len_samples,
                sampling_rate=sampling_rate,
                frame_rate=frame_rate,
            )
            if num_audio_tokens == 0:
                raise ValueError(
                    f"The audio input is too short to be represented by Raon: audio_len_samples={audio_len_samples}."
                )
            replacement_ids = build_audio_input_placeholder_ids(
                num_audio_tokens=num_audio_tokens,
                audio_start_token_id=audio_start_token_id,
                audio_input_token_id=placeholder_token_id,
                audio_end_token_id=audio_end_token_id,
                close_with_end=close_with_end,
            )
            return PromptUpdateDetails.select_token_id(
                replacement_ids,
                embed_token_id=placeholder_token_id,
            )

        def get_input_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            return _build_replacement(
                item_idx,
                placeholder_token_id=audio_input_token_id,
                close_with_end=True,
            )

        def get_output_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            return _build_replacement(
                item_idx,
                placeholder_token_id=audio_output_token_id,
                close_with_end=True,
            )

        def get_output_open_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            return _build_replacement(
                item_idx,
                placeholder_token_id=audio_output_token_id,
                close_with_end=False,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=AUDIO_PLACEHOLDER_SEQ,
                replacement=get_input_replacement,
            ),
            PromptReplacement(
                modality="audio",
                target=LEGACY_AUDIO_PLACEHOLDER_TOKEN_IDS,
                replacement=get_input_replacement,
            ),
            PromptReplacement(
                modality="audio",
                target=AUDIO_OUTPUT_PLACEHOLDER_SEQ,
                replacement=get_output_replacement,
            ),
            PromptReplacement(
                modality="audio",
                target=AUDIO_OUTPUT_OPEN_SEQ,
                replacement=get_output_open_replacement,
            ),
        ]


@dataclass
class AudioDecodeState:
    pending_audio_codes: torch.Tensor | None = None
    forced_audio_bootstrap_done: bool = False
    is_generating_audio: bool = False
    audio_step_index: int = 0
    continuation_silence_frames: int = 0
    _talker_hidden_row: torch.Tensor | None = None
    code_history: list[int] = field(default_factory=list)


@dataclass
class _StepContext:
    """Step-scoped context: set in forward(), consumed in compute_logits()/postprocess()."""

    talker_hidden: torch.Tensor
    runtime_info: list[dict[str, Any]]


@MULTIMODAL_REGISTRY.register_processor(
    RaonMultiModalProcessor,
    info=RaonProcessingInfo,
    dummy_inputs=RaonDummyInputsBuilder,
)
class RaonModel(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
    CustomProcessMixin,
):
    """Raon wrapper supporting Stage-0 AR.

    Pipeline per decode step::

        audio_preprocess()   -- inject previous step's audio codes as embeddings
        forward()            -- (1) Thinker: text_model() -> thinker_hidden
                                (2) Talker: thinker_to_talker_proj + talker() -> talker_hidden
                                Returns: text_hidden_states; stashes talker_hidden in _step_ctx
        compute_logits()     -- (3) Text head: lm_head(text_hidden) -> text logits
                                (4) Audio head: audio_lm_head(talker_hidden) -> layer-0 code
                                (5) Code predictor: predict remaining RVQ groups
        postprocess()        -- propagate talker_hidden + decode state to next step
    """

    supported_languages = ISO639_1_SUPPORTED_LANGS
    request_state_cleanup_keys = REQUEST_STATE_CLEANUP_KEYS
    request_state_cleanup_prefixes = REQUEST_STATE_CLEANUP_PREFIXES
    gpu_resident_buffer_keys: set[str] = {"duplex_prev_hidden", "speaker_embeds"}

    # Stage-1 only consumes codec payloads; skip unused thinker hidden.
    emit_hidden_in_payload: bool = False

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return AUDIO_PLACEHOLDER_SEQ
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        text_config = self.config.text_model_config
        talker_config = self.config.talker_config
        code_predictor_config = self.config.code_predictor_config
        audio_encoder_config = self.config.audio_encoder_config
        audio_tokenizer_config = self.config.audio_tokenizer_config
        input_adaptor_config = self.config.input_adaptor_config
        output_adaptor_config = self.config.output_adaptor_config

        self.model_stage = "stage0"
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.requires_raw_input_tokens = True

        dtype = vllm_config.model_config.dtype
        self.dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16

        self.audio_output_token_id = int(self.config.audio_output_token_id)
        self.audio_input_token_id = int(self.config.audio_input_token_id)
        self.hidden_size = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self.num_thinker_layers = int(text_config.num_hidden_layers)
        self.codebook_size = int(audio_tokenizer_config.codebook_size)
        self.sampling_rate = int(audio_tokenizer_config.sampling_rate)
        self.num_code_groups = int(code_predictor_config.num_code_groups)
        self.accept_hidden_layer = self.config.accept_hidden_layer_resolved()
        self.lm_head_layer_index = self.accept_hidden_layer
        self.audio_lm_head_vocab_size = self.codebook_size + 1
        self.frame_rate = get_mimi_frame_rate(audio_tokenizer_config)
        self.samples_per_frame = compute_samples_per_frame(
            sampling_rate=self.sampling_rate,
            frame_rate=self.frame_rate,
        )

        audio_hidden_size = int(talker_config.hidden_size)
        pp_last = get_pp_group().is_last_rank

        projection_mode = self.config.thinker_to_talker_projection_mode
        projection_intermediate_size = self.config.thinker_to_talker_intermediate_size
        if projection_mode == "mlp" and projection_intermediate_size is None:
            projection_intermediate_size = getattr(talker_config, "intermediate_size", None)

        if float(input_adaptor_config.output_time_scale) != 1.0:
            raise NotImplementedError("Only `output_time_scale == 1` is supported.")

        text_vllm_config = vllm_config.with_hf_config(text_config, architectures=["Qwen3ForCausalLM"])
        talker_vllm_config = vllm_config.with_hf_config(talker_config, architectures=["Qwen3ForCausalLM"])
        # Disable AWQ quantization on BF16 talker (fused layer name mismatch).
        if talker_vllm_config.quant_config is not None:
            object.__setattr__(talker_vllm_config, "quant_config", None)

        self._ras = RepetitionAwareSampler()
        self._step_decode_states: dict[str, AudioDecodeState] = {}
        self._step_talker_hidden_rows: dict[str, torch.Tensor] = {}
        self._step_ctx: _StepContext | None = None
        self._captured_thinker_hidden: torch.Tensor | None = None
        self._cached_silence_codes: torch.Tensor | None = None
        self._N_ICL_SILENCE_FRAMES: int = 2
        self.speaker_token_id: int | None = None
        self.speaker_encoder: PretrainedSpeakerEncoder | None = None
        self.is_pretrained_speaker_encoder = False
        self.eos_token_id: int | None = None
        self._audio_only_allowed_text_token_ids: tuple[int, ...] = ()
        self._tokenizer: Any | None = None
        self._tokenizer_len: int | None = None
        self.audio_end_token_id: int | None = None

        def _prefixed(name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

        self.text_model = Qwen3Model(vllm_config=text_vllm_config, prefix=_prefixed("text_model"))
        self.talker = Qwen3Model(vllm_config=talker_vllm_config, prefix=_prefixed("talker"))
        # The talker runs inside the main compiled RaonModel graph; compiling it
        # as a separate sub-model causes a KeyError on its KV-cache layer index
        # when enforce_eager=False.  Disable independent compilation here so it
        # is inlined into the parent's compiled graph.
        self.talker.do_not_compile = True
        del self.talker.embed_tokens
        self.talker.embed_tokens = None  # type: ignore[assignment]

        self.lm_head = (
            ParallelLMHead(
                self.vocab_size,
                int(text_config.hidden_size),
                quant_config=vllm_config.quant_config,
                prefix=_prefixed("lm_head"),
            )
            if pp_last
            else PPMissingLayer()
        )
        self.logits_processor = LogitsProcessor(self.vocab_size)
        self.audio_lm_head = nn.Linear(
            audio_hidden_size,
            self.audio_lm_head_vocab_size,
            bias=False,
            dtype=self.dtype,
        )
        self.proj_code = nn.Linear(
            audio_hidden_size,
            int(code_predictor_config.hidden_size),
            bias=bool(self.config.proj_code_bias),
            dtype=self.dtype,
        )
        self.proj_speaker_code: nn.Linear | None = None
        if bool(getattr(self.config, "speaker_embedding_to_code_predictor", False)):
            self.proj_speaker_code = nn.Linear(
                audio_hidden_size,
                int(code_predictor_config.hidden_size),
                bias=False,
                dtype=self.dtype,
            )
        self.thinker_to_talker_proj = ThinkerToTalkerProjection(
            thinker_hidden_size=int(text_config.hidden_size),
            talker_hidden_size=audio_hidden_size,
            intermediate_size=projection_intermediate_size,
            mode=projection_mode,
            use_norm=bool(self.config.thinker_to_talker_pre_norm),
            rms_norm_eps=float(getattr(text_config, "rms_norm_eps", 1e-6)),
            dtype=self.dtype,
        )
        self.audio_encoder = Qwen3OmniAuTWrapper.from_config(config=audio_encoder_config, dtype=self.dtype)
        self.audio_tokenizer = StreamingMimiModel._from_config(audio_tokenizer_config, dtype=self.dtype)

        def _build_adaptor(cfg: Any) -> EmbeddingAdaptor:
            return EmbeddingAdaptor(
                input_size=int(cfg.input_size),
                output_size=int(cfg.output_size),
                output_time_scale=float(cfg.output_time_scale),
                num_layers=int(cfg.num_layers),
                hidden_size=cfg.hidden_size,
                decoder_config=cfg.decoder_config,
                use_post_norm=bool(cfg.use_post_norm),
                norm_eps=float(cfg.norm_eps),
                dtype=self.dtype,
            )

        self.input_adaptor = _build_adaptor(input_adaptor_config)
        self.output_adaptor = _build_adaptor(output_adaptor_config)
        self.code_predictor = RaonCodePredictor(vllm_config=self.vllm_config, config=code_predictor_config)
        self.make_empty_intermediate_tensors = self.text_model.make_empty_intermediate_tensors

        self.set_custom_preprocess(self.audio_preprocess)
        self._register_thinker_hook()
        self._init_speaker()
        self._resolve_tokenizer_ids()

        from vllm_omni.model_executor.models.raon.logits_routing import LogitsRouter

        self._logits_router = LogitsRouter(self)

    @cached_property
    def sampler(self) -> Sampler:
        return Sampler()

    @staticmethod
    def _get_audio_decode_state(info_dict: dict) -> AudioDecodeState:
        state = info_dict.get("_decode_state")
        if state is None:
            state = AudioDecodeState()
            info_dict["_decode_state"] = state
        return state

    def audio_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        update_dict: dict[str, Any] = {}

        if input_embeds is None and input_ids is not None:
            input_embeds = self.embed_input_ids(input_ids)
        if input_ids is None or input_ids.numel() == 0:
            return input_ids, input_embeds, update_dict

        if self.speaker_token_id is not None:
            speaker_positions = torch.nonzero(input_ids == self.speaker_token_id, as_tuple=False).flatten()
            if speaker_positions.numel() > 0:
                speaker_embeds = info_dict.get("speaker_embeds")
                while isinstance(speaker_embeds, list) and len(speaker_embeds) == 1:
                    speaker_embeds = speaker_embeds[0]

                # ICL / voice-cache path: prefer pre-computed embedding.
                if not isinstance(speaker_embeds, torch.Tensor):
                    cached_spk = unwrap_singleton_list(info_dict.get("cached_spk_embedding"))
                    # Coerce plain list (from msgspec IPC) back to tensor.
                    if isinstance(cached_spk, list):
                        cached_spk = torch.tensor(cached_spk, dtype=torch.float32)
                    if isinstance(cached_spk, torch.Tensor):
                        flat = cached_spk
                        if flat.ndim == 3:
                            flat = flat[:, 0, :]
                        if flat.ndim == 2:
                            flat = flat[0]
                        if flat.ndim == 1 and int(flat.shape[0]) == int(input_embeds.shape[-1]):
                            speaker_embeds = cached_spk
                            update_dict["speaker_embeds"] = speaker_embeds
                        elif (
                            flat.ndim == 1
                            and self.speaker_encoder is not None
                            and self.is_pretrained_speaker_encoder
                            and int(flat.shape[0]) == getattr(self.speaker_encoder, "pretrained_dim", -1)
                        ):
                            # Raw ECAPA embedding from voices API cache; project to hidden size.
                            projected = self.speaker_encoder.projection(
                                flat.to(
                                    device=self.speaker_encoder.projection.weight.device,
                                    dtype=self.speaker_encoder.projection.weight.dtype,
                                )
                            )
                            speaker_embeds = projected.unsqueeze(0)
                            update_dict["speaker_embeds"] = speaker_embeds
                        else:
                            logger.warning(
                                "cached_spk_embedding shape %s incompatible with hidden=%d; falling back",
                                tuple(cached_spk.shape),
                                int(input_embeds.shape[-1]),
                            )

                if not isinstance(speaker_embeds, torch.Tensor):
                    speaker_ref_audio = normalize_speaker_ref_audio(
                        info_dict.get("speaker_ref_audio", info_dict.get("ref_audio"))
                    )
                    if speaker_ref_audio is not None and self.speaker_encoder is not None:
                        if (loaded := load_speaker_ref_audio(speaker_ref_audio)) is not None:
                            audio, sr = loaded
                            if audio.ndim > 1 and audio.shape[0] > 1:
                                audio = audio.mean(dim=0, keepdim=True)
                            speaker_audio = audio.to(dtype=torch.float32).transpose(0, 1).contiguous().view(1, -1)
                            speaker_lengths = torch.tensor([speaker_audio.shape[1]], dtype=torch.long)
                            speaker_embeds = compute_speaker_embeds(
                                self.speaker_encoder,
                                audio=speaker_audio,
                                audio_lengths=speaker_lengths,
                                sampling_rate=int(sr),
                                model_sampling_rate=self.sampling_rate,
                            )
                            update_dict["speaker_embeds"] = speaker_embeds
                            update_dict["speaker_ref_audio"] = None
                        else:
                            logger.warning(
                                "Skipping speaker conditioning; unable to load ref audio: %s",
                                speaker_ref_audio,
                            )

                if isinstance(speaker_embeds, torch.Tensor):
                    embeds = speaker_embeds
                    if embeds.ndim == 3:
                        embeds = embeds[:, 0, :]
                    if embeds.ndim == 2:
                        embeds = embeds[0]
                    if embeds.ndim != 1 or int(embeds.shape[0]) != int(input_embeds.shape[-1]):
                        raise AssertionError(
                            f"speaker_embeds must resolve to a single vector matching hidden size: "
                            f"got shape={tuple(speaker_embeds.shape)}, hidden={int(input_embeds.shape[-1])}."
                        )
                    replacement = embeds.to(device=input_embeds.device, dtype=input_embeds.dtype)
                    input_embeds = input_embeds.clone()
                    input_embeds[speaker_positions] = replacement.expand(int(speaker_positions.numel()), -1)

        # Audio-code feedback only applies to decode ticks (1 token/request).
        if int(input_ids.shape[0]) != 1:
            return input_ids, input_embeds, update_dict

        audio_positions = torch.nonzero(input_ids == self.audio_output_token_id, as_tuple=False).flatten()
        if audio_positions.numel() == 0:
            return input_ids, input_embeds, update_dict

        req_id = (
            normalize_runtime_request_id(info_dict.get("global_request_id", info_dict.get("_omni_req_id")))
            or "__unknown__"
        )
        req_state = self._get_audio_decode_state(info_dict)

        # First audio step: keep learned trigger embedding when no pending codes yet (bootstrap).
        if int(req_state.audio_step_index) == 0:
            forced_first_token = bool(torch.all(input_ids[audio_positions] == int(self.audio_output_token_id)))
            if forced_first_token and req_state.pending_audio_codes is None:
                update_dict["_decode_state"] = req_state
                return input_ids, input_embeds, update_dict

        if req_state.pending_audio_codes is None:
            logger.warning("Missing pending_audio_codes for req_id=%s; leaving embed unchanged", req_id)
            update_dict["_decode_state"] = req_state
            return input_ids, input_embeds, update_dict
        full_codes = req_state.pending_audio_codes
        req_state.pending_audio_codes = None
        input_embeds = input_embeds.clone()

        codes_tensor = full_codes.unsqueeze(0)
        codes_mask = torch.ones((1, codes_tensor.shape[1]), device=codes_tensor.device, dtype=torch.bool)
        audio_embeds, _ = self.get_audio_output_embeds(codes_tensor, codes_mask)
        input_embeds[audio_positions] = audio_embeds[0].to(device=input_embeds.device, dtype=input_embeds.dtype)

        step_chunk = full_codes.detach().to(torch.long).to("cpu").contiguous()
        req_state.is_generating_audio = True
        req_state.audio_step_index += int(step_chunk.shape[0])
        existing = coerce_optional_int(unwrap_singleton_list(info_dict.get("codec_total_rows"))) or 0
        total_rows = int(existing) + int(step_chunk.shape[0])
        update_dict.update(
            _decode_state=req_state,
            codec_codes=None,
            codec_codes_chunk=step_chunk,
            codec_total_rows=total_rows,
            codec_seq=max(0, total_rows - 1),
        )
        return input_ids, input_embeds, update_dict

    def _init_speaker(self) -> None:
        stid = self.config.speaker_token_id
        if isinstance(stid, int):
            self.speaker_token_id = int(stid)
        elif isinstance(stid, (list, tuple)) and stid:
            self.speaker_token_id = int(stid[0])

        speaker_cfg = coerce_speaker_encoder_config(self.config.speaker_encoder_config)
        if isinstance(speaker_cfg, SpeakerEncoderConfig):
            self.speaker_encoder = build_speaker_encoder(speaker_cfg, dtype=self.dtype)
            self.is_pretrained_speaker_encoder = isinstance(self.speaker_encoder, PretrainedSpeakerEncoder)
            # Warm artifacts eagerly to avoid stalling on first TTS request.
            if self.is_pretrained_speaker_encoder and isinstance(self.speaker_encoder, PretrainedSpeakerEncoder):
                try:
                    self.speaker_encoder.warm_backend_artifacts()
                    logger.info(
                        "Warmed pretrained speaker artifacts: encoder_type=%s model_id=%s",
                        self.speaker_encoder.encoder_type,
                        self.speaker_encoder.pretrained_model_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to warm pretrained speaker artifacts; "
                        "speaker-conditioned TTS will use fully lazy init: %s",
                        exc,
                    )

    def _register_thinker_hook(self) -> None:
        def _hook(_module: Any, _input: Any, output: Any) -> None:
            if isinstance(output, tuple) and len(output) >= 2:
                thinker_hidden = output[0] + output[1]
            elif isinstance(output, tuple):
                thinker_hidden = output[0]
            else:
                thinker_hidden = output
            if isinstance(thinker_hidden, torch.Tensor):
                self._captured_thinker_hidden = thinker_hidden

        self.text_model.layers[self.accept_hidden_layer].register_forward_hook(_hook)

    def _resolve_tokenizer_ids(self) -> None:
        eos_token_id = getattr(self.config.text_model_config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        self.eos_token_id = int(eos_token_id) if isinstance(eos_token_id, int) else None
        self._audio_only_allowed_text_token_ids = (self.eos_token_id,) if self.eos_token_id is not None else ()

        try:
            tok_path = self.vllm_config.model_config.tokenizer or self.vllm_config.model_config.model
            tokenizer = AutoTokenizer.from_pretrained(
                tok_path,
                trust_remote_code=self.vllm_config.model_config.trust_remote_code,
                fix_mistral_regex=True,
            )
            align_tokenizer(tokenizer)
            self._tokenizer = tokenizer
            self._tokenizer_len = len(tokenizer)

            tok_eos = getattr(tokenizer, "eos_token_id", None)
            if isinstance(tok_eos, (list, tuple)):
                tok_eos = tok_eos[0] if tok_eos else None
            if isinstance(tok_eos, int):
                self.eos_token_id = int(tok_eos)

            allowed: list[int] = []
            if self.eos_token_id is not None:
                allowed.append(self.eos_token_id)
            for tok in (AUDIO_START_TOKEN, AUDIO_END_TOKEN):
                ids = tokenizer.encode(tok, add_special_tokens=False)
                if len(ids) == 1 and isinstance(ids[0], int):
                    if tok == AUDIO_END_TOKEN:
                        self.audio_end_token_id = int(ids[0])
                    allowed.append(int(ids[0]))
            allowed.append(int(self.audio_output_token_id))

            for attr, special_token, resolver in (("audio_input_token_id", AUDIO_INPUT_PAD_TOKEN, None),):
                enc_ids = tokenizer.encode(special_token, add_special_tokens=False)
                if len(enc_ids) == 1 and isinstance(enc_ids[0], int):
                    tok_id = int(enc_ids[0])
                    if getattr(self, attr) != tok_id:
                        logger.warning("Overriding %s from %s to tokenizer id %s", attr, getattr(self, attr), tok_id)
                        setattr(self, attr, tok_id)

            if self.speaker_encoder is not None:
                try:
                    tok_spk_id = resolve_speaker_token_id(tokenizer, expected_speaker_token_id=self.speaker_token_id)
                    if self.speaker_token_id != tok_spk_id:
                        logger.warning(
                            "Overriding speaker_token_id from %s to tokenizer id %s",
                            self.speaker_token_id,
                            tok_spk_id,
                        )
                        self.speaker_token_id = tok_spk_id
                except Exception as exc:
                    logger.warning("Speaker token lookup failed; speaker-conditioned TTS may be disabled: %s", exc)

            if allowed:
                seen: set[int] = set()
                self._audio_only_allowed_text_token_ids = tuple(
                    tid for tid in allowed if not (tid in seen or seen.add(tid))
                )
        except Exception as exc:
            logger.warning("Tokenizer length lookup failed; detok bound checks disabled: %s", exc)

    @staticmethod
    def _ensure_3d_audio(audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [B, 1, T] shape."""
        if audio.ndim == 1:
            return audio[None, None]
        if audio.ndim == 2:
            return audio[:, None]
        if audio.ndim == 3:
            return audio
        raise ValueError(f"audio must be 1D/2D/3D tensor, got shape={tuple(audio.shape)}.")

    def _validate_audio_length(self, audio: torch.Tensor) -> torch.Tensor:
        max_samples = ENV.max_audio_duration_s * self.sampling_rate
        if max_samples > 0 and audio.shape[-1] > max_samples:
            raise ValueError(
                f"Audio input too long: {audio.shape[-1]} samples "
                f"({audio.shape[-1] / self.sampling_rate:.1f}s), max allowed {ENV.max_audio_duration_s}s"
            )
        return audio

    @torch.inference_mode()
    def get_audio_output_embeds_from_audio(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference audio -> Mimi codes -> output-adaptor embeddings (thinker ICL)."""
        audio = self._ensure_3d_audio(audio)
        audio = self._validate_audio_length(audio)

        target_device = module_device(self.audio_tokenizer)
        target_dtype = module_dtype(self.audio_tokenizer)
        audio = audio.to(device=target_device, dtype=target_dtype)

        if audio_lengths is None:
            audio_lengths = torch.full(
                (int(audio.shape[0]),),
                int(audio.shape[-1]),
                dtype=torch.long,
                device=audio.device,
            )
        else:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long).reshape(-1)

        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=int(sampling_rate),
                new_freq=int(self.sampling_rate),
            )
            ratio = float(self.sampling_rate) / float(sampling_rate)
            audio_lengths = (audio_lengths.float() * ratio).long()

        audio_lengths = audio_lengths.clamp(min=1, max=int(audio.shape[-1]))
        arange = torch.arange(audio.shape[-1], device=audio.device)[None]
        audio_mask = (arange < audio_lengths[:, None]).long()

        outputs = self.audio_tokenizer.encode(
            audio,
            padding_mask=audio_mask,
            num_quantizers=int(self.num_code_groups),
            return_dict=True,
        )
        if getattr(outputs, "audio_codes", None) is None:
            raise RuntimeError("audio_tokenizer.encode returned no audio_codes for ICL output embeddings.")

        audio_codes = outputs.audio_codes.view(
            outputs.audio_codes.shape[-3:],
        ).transpose(1, 2)
        pad_len = int(audio_codes.shape[1]) * int(self.samples_per_frame) - int(audio_mask.shape[1])
        padded_audio_mask = F.pad(audio_mask, (0, pad_len))
        audio_codes_mask = padded_audio_mask.view(
            -1,
            audio_codes.shape[1],
            self.samples_per_frame,
        ).any(dim=-1)
        return self.get_audio_output_embeds(audio_codes, audio_codes_mask)

    def _get_silence_codes(self) -> torch.Tensor:
        """Encode actual silence through Mimi and cache the resulting codes [N, G]."""
        if self._cached_silence_codes is None:
            dur = int(self._N_ICL_SILENCE_FRAMES * self.samples_per_frame)
            dev = next(self.audio_tokenizer.parameters()).device
            wav = torch.zeros(1, 1, dur, device=dev, dtype=torch.bfloat16)
            mask = torch.ones(1, dur, device=dev, dtype=torch.long)
            with torch.inference_mode():
                out = self.audio_tokenizer.encode(wav, mask, num_quantizers=int(self.num_code_groups), return_dict=True)
            codes = out.audio_codes.view(out.audio_codes.shape[-3:])
            self._cached_silence_codes = codes.transpose(1, 2)[0].to(torch.long).cpu().contiguous()
        return self._cached_silence_codes

    def build_request_payload(
        self,
        req_id: str,
        req_info: dict[str, Any],
        is_finished: bool,
    ) -> dict[str, object]:
        """Stage bridge codec payload: per-step chunk while running; full codes on finish."""
        codec_payload: torch.Tensor | None = None
        if is_finished:
            codec_full = req_info.get("codec_codes")
            if isinstance(codec_full, torch.Tensor) and codec_full.numel() > 0:
                codec_payload = collapse_exact_repeated_codec_snapshot(codec_full)
                req_info["codec_codes"] = codec_payload
                req_info["codec_total_rows"] = int(codec_payload.shape[0])
                req_info["codec_seq"] = max(0, int(codec_payload.shape[0]) - 1)
            req_info.pop("codec_codes_chunk", None)
        else:
            codec_chunk = req_info.pop("codec_codes_chunk", None)
            if isinstance(codec_chunk, torch.Tensor) and codec_chunk.numel() > 0:
                codec_payload = codec_chunk

        result: dict[str, object] = {}
        if isinstance(codec_payload, torch.Tensor) and codec_payload.numel() > 0:
            result["codec_codes"] = codec_payload.detach().to("cpu").contiguous()
        for key in ("global_request_id", "source_text"):
            if (val := req_info.get(key)) is not None:
                result[key] = val
        return result

    @torch.inference_mode()
    def get_audio_input_embeds(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = self._ensure_3d_audio(audio)
        audio = self._validate_audio_length(audio)

        target_device = module_device(self.audio_encoder)
        target_dtype = module_dtype(self.audio_encoder)
        audio = audio.to(device=target_device, dtype=target_dtype)

        encoder_sampling_rate = int(getattr(self.audio_encoder.config, "sampling_rate", self.sampling_rate))
        if sampling_rate is not None and sampling_rate != encoder_sampling_rate:
            audio = torchaudio.functional.resample(
                waveform=audio.float(),
                orig_freq=sampling_rate,
                new_freq=encoder_sampling_rate,
            )
            audio = audio.to(dtype=target_dtype)

        encoder_outputs = self.audio_encoder(audio, use_streaming=False)
        audio_embeds = getattr(encoder_outputs, "embeds", None)
        if not isinstance(audio_embeds, torch.Tensor):
            raise RuntimeError("audio_encoder returned no audio embeddings.")

        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long).reshape(-1)
            if int(audio_lengths.shape[0]) != int(audio.shape[0]):
                raise ValueError(
                    f"audio_lengths batch mismatch: got {int(audio_lengths.shape[0])}, expected {int(audio.shape[0])}."
                )
            audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])
            arange = torch.arange(audio.shape[-1], device=audio.device)[None]
            audio_embeds_mask = (arange < audio_lengths[:, None]).long()
            target_audio_samples = audio_embeds.shape[1] * self.samples_per_frame
            if target_audio_samples < audio_embeds_mask.shape[1]:
                raise ValueError(
                    f"audio_encoder produced fewer frames than expected: "
                    f"target_samples={target_audio_samples}, input_samples={audio_embeds_mask.shape[1]}."
                )
            pad_len = target_audio_samples - audio_embeds_mask.shape[1]
            padded = F.pad(audio_embeds_mask, (0, pad_len))
            audio_embeds_mask = padded.view(
                -1,
                audio_embeds.shape[1],
                self.samples_per_frame,
            ).any(dim=-1)
        else:
            audio_embeds_mask = torch.ones(
                audio_embeds.shape[:2],
                dtype=torch.bool,
                device=audio_embeds.device,
            )

        out = self.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        if out.mask is None:
            out.mask = torch.ones(out.outputs_embeds.shape[:2], dtype=torch.bool, device=out.outputs_embeds.device)
        return out.outputs_embeds, out.mask

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_waveforms = kwargs.pop("audio_waveforms", None)
        audio_lengths = kwargs.pop("audio_lengths", None)
        if audio_waveforms is None:
            return []

        audio_waveforms, audio_lengths = normalize_audio_waveforms_and_lengths(audio_waveforms, audio_lengths)
        target_device = module_device(self.audio_encoder)
        audio_waveforms = audio_waveforms.to(device=target_device)
        audio_lengths = audio_lengths.to(device=target_device)

        # Per-audio placeholder routing for ICL prefill via Mimi codec.
        audio_placeholder_token_ids = kwargs.pop("audio_placeholder_token_ids", None)
        if isinstance(audio_placeholder_token_ids, list):
            audio_placeholder_token_ids = torch.as_tensor(audio_placeholder_token_ids, dtype=torch.long)
        elif isinstance(audio_placeholder_token_ids, torch.Tensor):
            audio_placeholder_token_ids = audio_placeholder_token_ids.to(dtype=torch.long).reshape(-1)
        if audio_placeholder_token_ids is None:
            audio_placeholder_token_ids = torch.full(
                (int(audio_waveforms.shape[0]),),
                int(self.audio_input_token_id),
                dtype=torch.long,
            )

        per_audio: list[torch.Tensor] = []
        for idx in range(int(audio_waveforms.shape[0])):
            item_audio = audio_waveforms[idx : idx + 1]
            item_lengths = audio_lengths[idx : idx + 1]
            valid_len = int(item_lengths[0].item())
            if valid_len < 0:
                raise ValueError(f"audio_lengths must be >= 0, got {valid_len}.")
            if valid_len < int(item_audio.shape[-1]):
                # Each item is sliced out of a padded batch; trim back to the
                # true waveform length so encoder length math is per-audio.
                item_audio = item_audio[..., :valid_len].contiguous()
            placeholder_tid = int(audio_placeholder_token_ids[idx].item())

            if placeholder_tid == int(self.audio_output_token_id):
                cached_codes_raw = unwrap_singleton_list(kwargs.get("cached_ref_codec_codes"))
                cached_mask_raw = unwrap_singleton_list(kwargs.get("cached_ref_codec_codes_mask"))

                if isinstance(cached_codes_raw, torch.Tensor) and cached_codes_raw.numel() > 0:
                    _cached_codes = cached_codes_raw.to(device=target_device, dtype=torch.long)
                    if _cached_codes.ndim == 2:
                        _cached_codes = _cached_codes.unsqueeze(0)
                    if isinstance(cached_mask_raw, torch.Tensor):
                        _cached_mask = cached_mask_raw.to(device=target_device)
                        if _cached_mask.ndim == 1:
                            _cached_mask = _cached_mask.unsqueeze(0)
                    else:
                        _cached_mask = torch.ones(_cached_codes.shape[:2], dtype=torch.bool, device=target_device)
                    item_embeds, item_mask = self.get_audio_output_embeds(_cached_codes, _cached_mask)
                else:
                    item_embeds, item_mask = self.get_audio_output_embeds_from_audio(
                        audio=item_audio,
                        audio_lengths=item_lengths,
                    )
            else:
                item_embeds, item_mask = self.get_audio_input_embeds(audio=item_audio, audio_lengths=item_lengths)
            per_audio.append(item_embeds[0][item_mask[0]])
        return tuple(per_audio)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.text_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if not multimodal_embeddings:
            return inputs_embeds

        audio_embeddings = flatten_audio_embeddings(
            multimodal_embeddings,
            hidden_size=inputs_embeds.shape[-1],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        scatter_token_id = self.audio_input_token_id
        if is_multimodal is not None:
            is_multimodal = is_multimodal.to(device=input_ids.device, non_blocking=True)
            mm_token_ids = input_ids[is_multimodal]
            unique_ids = mm_token_ids.unique()
            if unique_ids.numel() == 1:
                scatter_token_id = int(unique_ids.item())
            elif unique_ids.numel() > 1:
                # Mixed placeholder types: scatter each type separately.
                for tid in unique_ids.tolist():
                    type_mm = is_multimodal & (input_ids == tid)
                    inputs_embeds = scatter_audio_input_embeddings(
                        inputs_embeds=inputs_embeds,
                        input_ids=input_ids,
                        audio_input_embeddings=audio_embeddings[mm_token_ids == tid],
                        audio_input_token_id=tid,
                        is_multimodal=type_mm,
                    )
                return inputs_embeds

        return scatter_audio_input_embeddings(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            audio_input_embeddings=audio_embeddings,
            audio_input_token_id=scatter_token_id,
            is_multimodal=is_multimodal,
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.text_model

    @staticmethod
    def _gather_last_hidden_per_request(
        hidden: torch.Tensor,
        runtime_info: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        """Select the last token per request from *hidden* [total_tokens, D].

        Uses ``_num_tokens`` from the model runner to compute cumulative
        last-token indices -- the same positions used for ``compute_logits``.
        """
        if not isinstance(runtime_info, list) or not runtime_info:
            return hidden
        if hidden.shape[0] <= len(runtime_info):
            return hidden
        cum = 0
        indices: list[int] = []
        for info in runtime_info:
            cum += int(info.get("_num_tokens", 1)) if isinstance(info, dict) else 1
            indices.append(cum - 1)
        if not indices or indices[-1] >= hidden.shape[0]:
            logger.warning("_gather_last_hidden: index %s >= total %d, skipping", indices[-1:], hidden.shape[0])
            return hidden
        return hidden[torch.tensor(indices, device=hidden.device, dtype=torch.long)]

    def _run_thinker(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor | IntermediateTensors, torch.Tensor | None]:
        """Run the thinker (text_model) and return ``(text_hidden_states, thinker_hidden)``.

        The thinker hidden state is captured via the forward hook registered in
        ``_register_thinker_hook()`` at the ``accept_hidden_layer`` index.
        Returns ``(IntermediateTensors, None)`` on pipeline-parallel non-last ranks.
        """
        self._captured_thinker_hidden = None
        text_hidden_states = self.text_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        thinker_hidden = self._captured_thinker_hidden
        self._captured_thinker_hidden = None
        return text_hidden_states, thinker_hidden

    def _run_talker(
        self,
        thinker_hidden: torch.Tensor,
        positions: torch.Tensor,
        runtime_info: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        """Project thinker hidden into talker space and run the talker model."""
        talker_input = self.thinker_to_talker_proj(thinker_hidden.to(dtype=self.dtype))
        talker_hidden_states = self.talker(input_ids=None, positions=positions, inputs_embeds=talker_input)
        talker_hidden_states = self._gather_last_hidden_per_request(talker_hidden_states, runtime_info)
        return talker_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        text_hidden_states, thinker_hidden = self._run_thinker(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )
        if not isinstance(text_hidden_states, torch.Tensor):
            return text_hidden_states
        if thinker_hidden is None:
            raise RuntimeError("Thinker hook did not fire – check accept_hidden_layer config")

        talker_hidden_states = self._run_talker(
            thinker_hidden,
            positions,
            runtime_additional_information,
        )

        self._step_ctx = _StepContext(
            talker_hidden=talker_hidden_states,
            runtime_info=runtime_additional_information or [],
        )
        return text_hidden_states

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | IntermediateTensors | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput | IntermediateTensors:
        if isinstance(model_outputs, (OmniOutput, IntermediateTensors)):
            return model_outputs

        runtime_info = kwargs.get("runtime_additional_information", [])
        per_req_full: list[torch.Tensor | None] = []
        per_req_chunk: list[torch.Tensor | None] = []
        if isinstance(runtime_info, list):

            def _parse_codec_slot(
                info: dict[str, Any],
            ) -> tuple[
                torch.Tensor | None,
                torch.Tensor | None,
                str | None,
                int,
            ]:
                full = info.get("codec_codes")
                full = full if isinstance(full, torch.Tensor) and full.numel() > 0 else None
                chunk = info.get("codec_codes_chunk")
                chunk = chunk if isinstance(chunk, torch.Tensor) and chunk.numel() > 0 else None
                req_id = normalize_runtime_request_id(info.get("global_request_id", info.get("_omni_req_id")))
                seq = coerce_optional_int(info.get("codec_seq"))
                if seq is None and (tr := coerce_optional_int(info.get("codec_total_rows"))) is not None:
                    seq = max(0, int(tr) - 1)
                if seq is None:
                    seq = max(0, int((full or chunk).shape[0]) - 1) if (full is not None or chunk is not None) else -1
                return full, chunk, req_id, seq

            parsed = [_parse_codec_slot(info) if isinstance(info, dict) else None for info in runtime_info]

            # Dedupe duplicate req_ids: keep the entry with the largest seq.
            selected_by_req: dict[str, tuple[int, torch.Tensor | None, torch.Tensor | None]] = {}
            total_entries = fallback_entries = 0
            for slot in parsed:
                if slot is None:
                    continue
                full, chunk, req_id, seq = slot
                if full is None and chunk is None:
                    continue
                total_entries += 1
                if req_id is None:
                    fallback_entries += 1
                    continue
                prev = selected_by_req.get(req_id)
                if prev is None or seq >= prev[0]:
                    selected_by_req[req_id] = (seq, full, chunk)

            for slot in parsed:
                if slot is None:
                    per_req_full.append(None)
                    per_req_chunk.append(None)
                    continue
                full, chunk, req_id, _ = slot
                if req_id is not None and (sel := selected_by_req.get(req_id)) is not None:
                    _, full, chunk = sel
                per_req_full.append(full)
                per_req_chunk.append(chunk)

            selected_entries = len(selected_by_req) + fallback_entries
            if total_entries > selected_entries:
                logger.warning(
                    "Deduped runtime codec payload entries: total=%d selected=%d",
                    total_entries,
                    selected_entries,
                )

        multimodal_outputs: dict[str, Any] = {}
        if any(item is not None for item in per_req_full):
            multimodal_outputs["codec_codes"] = per_req_full
        if any(item is not None for item in per_req_chunk):
            multimodal_outputs["codec_codes_chunk"] = per_req_chunk
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=multimodal_outputs)

    @staticmethod
    def _normalize_output_mode(req_info: dict[str, Any] | None) -> str:
        _default = "text_and_audio"
        if not isinstance(req_info, dict):
            return _default
        value = req_info.get("output_mode", _default)
        if isinstance(value, list):
            value = value[0] if value else _default
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else _default
        if not isinstance(value, str):
            return _default
        value = value.lower().strip()
        return value if value in {"text_only", "audio_only", _default} else _default

    def _mask_audio_logits_for_text_mode(self, logits: torch.Tensor, row_idx: int) -> None:
        audio_start_idx = int(self.audio_output_token_id)
        if 0 <= audio_start_idx < int(logits.shape[-1]):
            logits[row_idx, audio_start_idx:] = float("-inf")

    @staticmethod
    def _apply_audio_sampling_params(logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature / top-k / top-p warping to audio_lm_head logits."""
        temperature = ENV.tts_temperature
        top_k = ENV.tts_top_k
        top_p = ENV.tts_top_p

        scores = logits.float()
        if temperature > 0 and temperature != 1.0:
            scores = scores / temperature
        if top_k > 0:
            kth, _ = torch.topk(scores, min(top_k, scores.shape[-1]))
            scores = scores.masked_fill(scores < kth[..., -1:], float("-inf"))
        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scores, descending=False)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cum_probs <= (1.0 - top_p)
            remove[..., -1:] = False
            remove_orig = remove.scatter(-1, sorted_idx, remove)
            scores = scores.masked_fill(remove_orig, float("-inf"))
        return scores

    def _force_token_in_logits(self, logits: torch.Tensor, row_idx: int, token_id: int) -> None:
        """Set logits so only *token_id* is selectable at *row_idx*."""
        logits[row_idx, :] = float("-inf")
        if 0 <= token_id < int(logits.shape[-1]):
            logits[row_idx, token_id] = 0.0

    def _forced_bootstrap_audio(
        self,
        logits: torch.Tensor,
        row_idx: int,
        req_state: AudioDecodeState | None,
    ) -> None:
        """Force audio_output_token_id on the first audio step (bootstrap)."""
        if req_state is not None:
            req_state.forced_audio_bootstrap_done = True
        if 0 <= int(self.audio_output_token_id) < int(logits.shape[-1]):
            self._force_token_in_logits(logits, row_idx, int(self.audio_output_token_id))
        else:
            logger.warning(
                "Invalid audio_output_token_id=%s for vocab=%s; skipping forced audio bootstrap",
                self.audio_output_token_id,
                int(logits.shape[-1]),
            )

    def _predict_rvq_codes(
        self,
        first_code: int,
        audio_hidden_row: torch.Tensor,
        device: torch.device,
        speaker_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the code predictor to generate all RVQ groups from a layer-0 code.

        Returns a ``[1, num_code_groups]`` long tensor of audio codes.
        """
        hidden_for_code = audio_hidden_row.to(dtype=self.proj_code.weight.dtype)
        if hidden_for_code.ndim == 2:
            hidden_for_code = hidden_for_code.unsqueeze(1)
        full_codes = self.generate_audio_codes(
            input_ids=torch.tensor([[first_code]], device=device, dtype=torch.long),
            inputs_embeds=self.proj_code(hidden_for_code),
            num_code_groups=self.num_code_groups,
            speaker_embeds=speaker_embeds,
        ).to(torch.long)
        return full_codes

    def _is_all_audio_only(self, queued_runtime_info: list[Any]) -> bool:
        if not queued_runtime_info:
            return False
        return all(
            self._normalize_output_mode(info if isinstance(info, dict) else None) == "audio_only"
            for info in queued_runtime_info
        )

    def _build_audio_only_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (hidden_states.shape[0], self.vocab_size),
            float("-inf"),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        special_ids = self._audio_only_allowed_text_token_ids
        if not special_ids:
            return logits

        lm_weight = self.lm_head.weight
        normed = hidden_states.squeeze(1) if hidden_states.dim() == 3 else hidden_states
        for token_id in special_ids:
            if token_id == self.audio_end_token_id:
                continue
            if 0 <= token_id < lm_weight.shape[0]:
                logits[:, token_id] = torch.mv(normed, lm_weight[token_id])
        return logits

    def _compute_base_logits(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor | None,
        queued_runtime_info: list[Any],
    ) -> torch.Tensor | None:
        use_split = (
            isinstance(audio_hidden_states, torch.Tensor) and audio_hidden_states.shape[0] == hidden_states.shape[0]
        )
        if use_split and self._is_all_audio_only(queued_runtime_info):
            return self._build_audio_only_logits(hidden_states)
        return self.logits_processor(self.lm_head, hidden_states)

    def _resolve_row_runtime_info(self, runtime_info: list[Any], row_count: int) -> list[Any]:
        if not runtime_info:
            return [{"output_mode": ["text_only"]}] * row_count
        if len(runtime_info) == row_count:
            return runtime_info
        if len(runtime_info) == 1:
            return [runtime_info[0]] * row_count
        modes = [self._normalize_output_mode(info if isinstance(info, dict) else None) for info in runtime_info]
        if modes and all(m == modes[0] for m in modes):
            logger.warning(
                "Runtime_info/logits row mismatch: runtime=%d logits=%d; broadcasting mode=%s",
                len(runtime_info),
                row_count,
                modes[0],
            )
            return [runtime_info[-1]] * row_count
        return [{"output_mode": ["text_only"]}] * row_count

    @staticmethod
    def _resolve_row_sampling_state(
        output_token_ids: Any,
        row_idx: int,
        row_count: int,
    ) -> tuple[list[int], bool]:
        is_sampled_row = not (
            isinstance(output_token_ids, list)
            and len(output_token_ids) == 1
            and row_count > 1
            and row_idx != row_count - 1
        )
        row_output_ids: list[int] = []
        if isinstance(output_token_ids, list) and output_token_ids:
            if len(output_token_ids) == row_count and row_idx < row_count:
                row_output_ids = output_token_ids[row_idx]
            elif len(output_token_ids) == 1:
                row_output_ids = output_token_ids[0]
            elif row_idx < len(output_token_ids):
                row_output_ids = output_token_ids[row_idx]
        return row_output_ids, is_sampled_row

    def _suppress_first_step_eos(self, logits: torch.Tensor, output_token_ids: Any) -> None:
        eos = self.eos_token_id
        if not isinstance(eos, int) or not (0 <= eos < logits.shape[-1]):
            return
        if not isinstance(output_token_ids, list) or len(output_token_ids) != logits.shape[0]:
            return
        tiny = torch.finfo(logits.dtype).tiny
        for row_idx, out_ids in enumerate(output_token_ids):
            if not out_ids:
                logits[row_idx, eos] = tiny

    def _compute_text_logits(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor | None,
        runtime_info: list[Any],
    ) -> torch.Tensor | None:
        """Compute text logits from thinker hidden states via the LM head.

        Returns the raw logits tensor with out-of-vocab positions masked.
        """
        logits = self._compute_base_logits(hidden_states, audio_hidden_states, runtime_info)
        if logits is None:
            return None
        if int(logits.shape[-1]) > int(self.vocab_size):
            logits[:, int(self.vocab_size) :] = float("-inf")
        return logits

    def _compute_audio_and_apply_modes(
        self,
        logits: torch.Tensor,
        audio_hidden_states: torch.Tensor | None,
        runtime_info: list[Any],
        sampling_metadata: SamplingMetadata | None,
    ) -> torch.Tensor:
        """Apply audio head + code predictor + per-row mode routing.

        Mutates *logits* in-place (audio token sampling, mode masking,
        EOS suppression) and returns the same tensor.
        """
        if not isinstance(runtime_info, list):
            runtime_info = []
        output_token_ids = (
            getattr(sampling_metadata, "output_token_ids", None) if sampling_metadata is not None else None
        )
        if (row_ri := self._resolve_row_runtime_info(runtime_info, int(logits.shape[0]))) is not None:
            self._logits_router.apply_row_mode_adjustments(
                logits=logits,
                row_runtime_info=row_ri,
                output_token_ids=output_token_ids,
                audio_hidden_states=audio_hidden_states,
            )
        self._suppress_first_step_eos(logits, output_token_ids)
        return logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput | IntermediateTensors,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        self._step_decode_states.clear()
        self._step_talker_hidden_rows.clear()
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if not isinstance(hidden_states, torch.Tensor):
            return None

        ctx = self._step_ctx
        runtime_info = ctx.runtime_info if ctx else []
        audio_hidden_states = ctx.talker_hidden if ctx else None

        if audio_hidden_states is not None and audio_hidden_states.shape[0] != hidden_states.shape[0]:
            audio_hidden_states = audio_hidden_states[-hidden_states.shape[0] :]

        logits = self._compute_text_logits(hidden_states, audio_hidden_states, runtime_info)
        if logits is None:
            return None

        return self._compute_audio_and_apply_modes(
            logits,
            audio_hidden_states,
            runtime_info,
            sampling_metadata,
        )

    def sample(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)

    def postprocess(
        self,
        hidden_states: torch.Tensor | None,
        multimodal_outputs: dict[str, Any] | None = None,
        **info_dict: Any,
    ) -> dict[str, Any]:
        del multimodal_outputs
        req_id = (
            normalize_runtime_request_id(info_dict.get("global_request_id", info_dict.get("_omni_req_id")))
            or "__unknown__"
        )
        # Prefer step-scoped state written by compute_logits, fall back to info_dict.
        req_state: AudioDecodeState | None = None
        if req_id and req_id != "__unknown__":
            req_state = self._step_decode_states.pop(req_id, None)
        if req_state is None:
            req_state = info_dict.get("_decode_state")

        # Use per-request talker hidden stored by compute_logits, not batch-level _step_ctx.
        per_request_hidden = None
        if req_id and req_id != "__unknown__":
            per_request_hidden = self._step_talker_hidden_rows.pop(req_id, None)
        if per_request_hidden is None and req_state is not None:
            per_request_hidden = req_state._talker_hidden_row
        if per_request_hidden is not None and isinstance(per_request_hidden, torch.Tensor):
            hidden_states = per_request_hidden
        else:
            ctx = self._step_ctx
            if ctx is not None and isinstance(ctx.talker_hidden, torch.Tensor):
                hidden_states = ctx.talker_hidden
        result: dict[str, Any] = {}
        if req_state is not None:
            # Keep this transient tensor out of persistent per-request state.
            req_state._talker_hidden_row = None
            result["_decode_state"] = req_state
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.numel() == 0:
            return result
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(0)
        result["prev_hidden"] = hidden_states[-1:].detach().contiguous().clone()
        return result

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return strip_raon_audio_markers(text)

    @torch.inference_mode()
    def get_audio_output_embeds(
        self,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if audio_codes.ndim != 3 or audio_codes_mask.ndim != 2:
            raise ValueError(
                f"Expected audio_codes [B,T,G] and mask [B,T], "
                f"got {tuple(audio_codes.shape)} and "
                f"{tuple(audio_codes_mask.shape)}"
            )
        latent = self.audio_tokenizer.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)
        adaptor_out = self.output_adaptor(
            latent,
            mask=audio_codes_mask,
        )
        return adaptor_out.outputs_embeds, adaptor_out.mask

    @torch.inference_mode()
    def generate_audio_codes(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        num_code_groups: int,
        speaker_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """All RVQ groups from layer-0 token ids + ``proj_code`` hidden state → ``[B, num_code_groups]``."""
        if num_code_groups == 1:
            return input_ids

        input_ids = input_ids.to(torch.long)
        last_hidden = inputs_embeds
        if speaker_embeds is not None and self.proj_speaker_code is not None:
            if speaker_embeds.ndim == 2:
                speaker_embeds = speaker_embeds.unsqueeze(1)
            if speaker_embeds.ndim != 3:
                raise AssertionError(
                    "speaker_embeds must be 2D/3D tensor for code predictor conditioning, "
                    f"got shape={tuple(speaker_embeds.shape)}."
                )
            if int(speaker_embeds.shape[0]) != int(inputs_embeds.shape[0]):
                raise AssertionError(
                    "speaker_embeds batch size must match audio-code rows: "
                    f"speaker_batch={int(speaker_embeds.shape[0])}, code_rows={int(inputs_embeds.shape[0])}."
                )
            speaker_embeds = speaker_embeds.to(device=inputs_embeds.device, dtype=self.proj_speaker_code.weight.dtype)
            speaker_cond = self.proj_speaker_code(speaker_embeds.squeeze(1)).to(dtype=inputs_embeds.dtype)
            last_hidden = inputs_embeds + speaker_cond.unsqueeze(1)

        layer0_code = input_ids.reshape(-1)
        bsz = layer0_code.shape[0]
        layer0_embed = self.code_predictor.get_input_embeddings()[0](layer0_code.unsqueeze(-1)).squeeze(1)

        return self.code_predictor.predict_codes(
            layer0_code=layer0_code,
            layer0_embed=layer0_embed,
            last_hidden=last_hidden.reshape(bsz, -1),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["text_output_norm."]
        if getattr(self, "speaker_encoder", None) is None:
            skip_prefixes.append("speaker_encoder.")
        if getattr(self, "proj_speaker_code", None) is None:
            skip_prefixes.append("proj_speaker_code.")

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
            ignore_unexpected_suffixes=list(IGNORE_WEIGHT_SUFFIXES),
        )
        loaded = loader.load_weights(weights)

        # Talker uses projected embeddings, not token ids.
        loaded.discard("talker.embed_tokens.weight")
        loaded.add("talker.embed_tokens.weight")

        logger.info("Loaded weights: stage=%s total=%d", self.model_stage, len(loaded))

        # Truncate RoPE cos/sin caches to bf16 precision to match training.
        for module in self.modules():
            if (cache := getattr(module, "cos_sin_cache", None)) is not None and isinstance(cache, torch.Tensor):
                module.cos_sin_cache = cache.to(torch.bfloat16).to(cache.dtype)

        return loaded

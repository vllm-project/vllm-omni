from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from importlib import import_module
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.inputs import MultiModalInput as MultiModalInputs
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .dynin_omni_common import build_zero_input_embeddings

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover
    PILImage = None


_MODALITY_ORDER = ("image", "video", "audio")

_MODALITY_ALIASES = {
    "img2img": "image",
}

_MODALITY_INPUT_KEY_BY_NAME = {
    "image": "pixel_values",
    "video": "pixel_values_videos",
    "audio": "input_audio_features",
}

_MODALITY_PLACEHOLDER_BY_NAME = {
    "image": "<|soi|><|image|><|eoi|>",
    "video": "<|sov|><|video|><|eov|>",
    "audio": "<|soa|><|audio|><|eoa|>",
}

_MODALITY_INPUT_ALIASES = {
    "image": ("pixel_values", "image_embeds", "img2img"),
    "video": ("pixel_values_videos", "video_embeds"),
    "audio": ("input_audio_features", "audio_embeds"),
}


def _normalize_modality_name(modality: str) -> str:
    return _MODALITY_ALIASES.get(modality, modality)


def _get_modality_count(mm_counts: Mapping[str, int], modality: str) -> int:
    canonical = _normalize_modality_name(modality)
    count = mm_counts.get(canonical, 0)
    for alias, target in _MODALITY_ALIASES.items():
        if target == canonical:
            count += mm_counts.get(alias, 0)
    return count


def _normalize_mm_data_aliases(mm_data: MultiModalDataDict) -> MultiModalDataDict:
    normalized: dict[str, Any] = {}
    for modality, value in mm_data.items():
        canonical = _normalize_modality_name(modality)
        if canonical in normalized and normalized[canonical] is not None and value is not None:
            raise ValueError(
                "Dynin received duplicate multimodal inputs for "
                f"{canonical!r} via {modality!r}. "
                "Provide either the canonical modality or its alias, not both."
            )
        if canonical not in normalized or normalized[canonical] is None:
            normalized[canonical] = value
    return normalized


def _get_placeholder_text(modality: str) -> str | None:
    modality = _normalize_modality_name(modality)
    for base_modality, placeholder in _MODALITY_PLACEHOLDER_BY_NAME.items():
        if modality.startswith(base_modality):
            return placeholder
    return None


class DyninOmniProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        return DyninOmniMultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits = {modality: 1 for modality in _MODALITY_ORDER}
        for alias, target in _MODALITY_ALIASES.items():
            if target in limits:
                limits[alias] = limits[target]
        return limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        del seq_len, mm_counts
        limits = {modality: 1 for modality in _MODALITY_ORDER}
        for alias, target in _MODALITY_ALIASES.items():
            if target in limits:
                limits[alias] = limits[target]
        return limits


class DyninOmniDummyInputsBuilder(BaseDummyInputsBuilder[DyninOmniProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        chunks: list[str] = []
        for modality in _MODALITY_ORDER:
            placeholder = _get_placeholder_text(modality)
            if placeholder is None:
                continue
            chunks.extend([placeholder] * _get_modality_count(mm_counts, modality))
        return " ".join(chunks)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        del seq_len

        mm_data: dict[str, Any] = {}

        num_images = _get_modality_count(mm_counts, "image")
        if num_images > 0:
            mm_data["image"] = self._get_dummy_images(
                width=224,
                height=224,
                num_images=num_images,
                overrides=mm_options.get("image") if mm_options else None,
            )

        num_videos = _get_modality_count(mm_counts, "video")
        if num_videos > 0:
            mm_data["video"] = self._get_dummy_videos(
                width=224,
                height=224,
                num_frames=8,
                num_videos=num_videos,
                overrides=mm_options.get("video") if mm_options else None,
            )

        num_audios = _get_modality_count(mm_counts, "audio")
        if num_audios > 0:
            mm_data["audio"] = self._get_dummy_audios(
                length=16000,
                num_audios=num_audios,
                overrides=mm_options.get("audio") if mm_options else None,
            )

        return mm_data


class DyninOmniMultiModalDataParser(MultiModalDataParser):
    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        normalized = _normalize_mm_data_aliases(mm_data)
        mm_items = super().parse_mm_data(normalized)

        for alias, canonical in _MODALITY_ALIASES.items():
            if alias in mm_data and canonical in mm_items and alias not in mm_items:
                mm_items[alias] = mm_items[canonical]

        return mm_items

    def _get_audio_with_sr(self, audio: Any) -> tuple[np.ndarray, float | None]:
        audio_array, orig_sr = super()._get_audio_with_sr(audio)
        if self.audio_resampler.target_sr is None:
            return audio_array, None
        return audio_array, orig_sr


class DyninOmniMultiModalProcessor(BaseMultiModalProcessor[DyninOmniProcessingInfo]):
    @staticmethod
    def _find_subsequence(
        haystack: list[int],
        needle: list[int],
        start: int,
    ) -> int | None:
        if not needle:
            return None

        max_start = len(haystack) - len(needle)
        if max_start < start:
            return None

        for idx in range(start, max_start + 1):
            if haystack[idx : idx + len(needle)] == needle:
                return idx
        return None

    @staticmethod
    def _make_disabled_embed_mask(length: int) -> torch.Tensor:
        return torch.zeros(length, dtype=torch.bool)

    @staticmethod
    def _encode_prompt_to_token_ids(
        prompt: str | list[int],
        tokenizer: Any | None,
    ) -> list[int]:
        if isinstance(prompt, str):
            if tokenizer is None:
                raise ValueError("Tokenizer is required to process string prompts for Dynin multimodal inputs.")
            return tokenizer.encode(prompt, add_special_tokens=False)
        return list(prompt)

    @staticmethod
    def _ensure_non_empty_prompt_ids(
        prompt_token_ids: list[int],
        tokenizer: Any | None,
    ) -> list[int]:
        if prompt_token_ids:
            return prompt_token_ids

        fallback_id = None
        if tokenizer is not None:
            fallback_id = getattr(tokenizer, "bos_token_id", None)
            if fallback_id is None:
                fallback_id = getattr(tokenizer, "eos_token_id", None)
            if fallback_id is None:
                fallback_id = getattr(tokenizer, "pad_token_id", None)

        return [0 if fallback_id is None else int(fallback_id)]

    @classmethod
    def _image_to_chw_float_tensor(cls, image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach()
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
        elif PILImage is not None and isinstance(image, PILImage.Image):
            tensor = torch.from_numpy(np.asarray(image).copy())
        else:
            raise TypeError(f"Unsupported image item type: {type(image)!r}")

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim != 3:
            raise ValueError(f"Expected 3D image tensor, got shape={tuple(tensor.shape)}")

        if tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)

        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.shape[0] == 4:
            tensor = tensor[:3]

        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() > 0 and torch.max(tensor) > 1.0:
            tensor = tensor / 255.0
        return tensor.contiguous()

    @classmethod
    def _video_to_tchw_float_tensor(cls, video: Any) -> torch.Tensor:
        if isinstance(video, (list, tuple)) and not isinstance(video, torch.Tensor):
            frames = [cls._image_to_chw_float_tensor(frame) for frame in video]
            if not frames:
                return torch.zeros((1, 3, 1, 1), dtype=torch.float32)
            return torch.stack(frames, dim=0).contiguous()

        if isinstance(video, torch.Tensor):
            tensor = video.detach()
        elif isinstance(video, np.ndarray):
            tensor = torch.from_numpy(video)
        else:
            raise TypeError(f"Unsupported video item type: {type(video)!r}")

        if tensor.ndim == 3:
            return cls._image_to_chw_float_tensor(tensor).unsqueeze(0).contiguous()

        if tensor.ndim != 4:
            raise ValueError(f"Expected 4D video tensor, got shape={tuple(tensor.shape)}")

        if tensor.shape[-1] in (1, 3, 4) and tensor.shape[1] not in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2)

        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        if tensor.shape[1] == 4:
            tensor = tensor[:, :3]

        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() > 0 and torch.max(tensor) > 1.0:
            tensor = tensor / 255.0
        return tensor.contiguous()

    @staticmethod
    def _audio_to_float_tensor(audio: Any) -> torch.Tensor:
        if isinstance(audio, tuple) and len(audio) == 2:
            audio = audio[0]

        if isinstance(audio, torch.Tensor):
            tensor = audio.detach()
        elif isinstance(audio, np.ndarray):
            tensor = torch.from_numpy(audio)
        else:
            tensor = torch.as_tensor(audio)

        tensor = tensor.to(dtype=torch.float32).contiguous().view(-1)
        if tensor.numel() == 0:
            return torch.zeros((16000,), dtype=torch.float32)

        max_abs = torch.max(torch.abs(tensor))
        if max_abs > 1.0:
            tensor = tensor / max_abs

        return tensor.contiguous()

    @classmethod
    def _convert_modality_item(cls, modality: str, item: Any) -> torch.Tensor:
        if modality == "image":
            return cls._image_to_chw_float_tensor(item)
        if modality == "video":
            return cls._video_to_tchw_float_tensor(item)
        if modality == "audio":
            return cls._audio_to_float_tensor(item)
        raise ValueError(f"Unsupported modality for Dynin processor: {modality}")

    def _build_modality_kwargs(
        self,
        modality: str,
        modality_items: Sequence[Any],
    ) -> Sequence[Any]:
        modality = _normalize_modality_name(modality)
        input_key = _MODALITY_INPUT_KEY_BY_NAME[modality]
        tensor_items = [self._convert_modality_item(modality, item) for item in modality_items]
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            {input_key: tensor_items},
            {input_key: MultiModalFieldConfig.batched(modality)},
        )
        return mm_kwargs[modality]

    def _build_placeholder_ranges(
        self,
        *,
        modality: str,
        item_count: int,
        prompt_token_ids: list[int],
        tokenizer: Any | None,
        search_start: int,
    ) -> tuple[list[PlaceholderRange], int]:
        ranges: list[PlaceholderRange] = []

        for _ in range(item_count):
            placeholder_text = _get_placeholder_text(modality)
            placeholder_token_ids: list[int] = []

            if placeholder_text and tokenizer is not None:
                placeholder_token_ids = tokenizer.encode(
                    placeholder_text,
                    add_special_tokens=False,
                )

            found_offset = None
            if placeholder_token_ids:
                found_offset = self._find_subsequence(
                    prompt_token_ids,
                    placeholder_token_ids,
                    search_start,
                )

            if found_offset is None:
                found_offset = min(search_start, len(prompt_token_ids) - 1)
                placeholder_len = 1
            else:
                placeholder_len = len(placeholder_token_ids)

            ranges.append(
                PlaceholderRange(
                    offset=found_offset,
                    length=placeholder_len,
                    is_embed=self._make_disabled_embed_mask(placeholder_len),
                )
            )
            search_start = found_offset + placeholder_len

        return ranges, search_start

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_inputs, hf_processor_mm_kwargs
        return {}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        return []

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInputs:
        prompt = inputs.prompt
        mm_items = inputs.mm_data_items

        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        tokenizer = self.info.ctx.tokenizer
        prompt_token_ids = self._encode_prompt_to_token_ids(prompt, tokenizer)
        prompt_token_ids = self._ensure_non_empty_prompt_ids(prompt_token_ids, tokenizer)

        mm_kwargs_by_modality: dict[str, Sequence[Any]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        search_start = 0
        mm_counts = mm_items.get_all_counts()

        for modality in _MODALITY_ORDER:
            item_count = mm_counts.get(modality, 0)
            if item_count <= 0:
                continue

            modality_items = mm_items[modality].get_all()
            if len(modality_items) != item_count:
                raise RuntimeError(
                    f"Parsed {len(modality_items)} items but expected {item_count} for modality={modality!r}"
                )

            mm_kwargs_by_modality[modality] = self._build_modality_kwargs(
                modality,
                modality_items,
            )

            placeholder_ranges, search_start = self._build_placeholder_ranges(
                modality=modality,
                item_count=item_count,
                prompt_token_ids=prompt_token_ids,
                tokenizer=tokenizer,
                search_start=search_start,
            )
            mm_placeholders[modality] = placeholder_ranges

        return MultiModalInputs(
            type="multimodal",
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=MultiModalKwargsItems(mm_kwargs_by_modality),
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


class DyninOmniStageBase(nn.Module):
    stage_name = "Dynin stage"

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        del batch_size, dtype, device
        return IntermediateTensors({})

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del multimodal_embeddings, is_multimodal, kwargs
        return build_zero_input_embeddings(
            input_ids=input_ids,
            hidden_size=self.hidden_size,
            stage_name=self.stage_name,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        return {name for name, _ in weights}

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        del hidden_states, sampling_metadata
        return None


@MULTIMODAL_REGISTRY.register_processor(
    DyninOmniMultiModalProcessor,
    info=DyninOmniProcessingInfo,
    dummy_inputs=DyninOmniDummyInputsBuilder,
)
class DyninOmniForConditionalGeneration(nn.Module, SupportsMultiModal):
    supports_multimodal_raw_input_only = True
    STAGE_ALIAS = {
        "tokenizer": "token2text",
        "token2token": "token2text",
        "detok_text": "token2text",
        "token2img": "token2image",
        "token2wav": "token2audio",
        "token2speech": "token2audio",
    }

    STAGE_IMPL = {
        "token2text": (".dynin_omni_token2text", "DyninOmniToken2Text"),
        "token2image": (".dynin_omni_token2image", "DyninOmniToken2Image"),
        "token2audio": (".dynin_omni_token2audio", "DyninOmniToken2Audio"),
    }

    _STAGE_IMPL_CACHE: dict[str, type[nn.Module]] = {}

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        return _get_placeholder_text(modality)

    @classmethod
    def _resolve_stage_impl_class(cls, model_stage: str) -> type[nn.Module]:
        impl = cls._STAGE_IMPL_CACHE.get(model_stage)
        if impl is not None:
            return impl

        module_name, class_name = cls.STAGE_IMPL[model_stage]
        module = import_module(module_name, package=__package__)
        impl = getattr(module, class_name)
        cls._STAGE_IMPL_CACHE[model_stage] = impl
        return impl

    @classmethod
    def _normalize_stage_name(cls, raw_stage: str) -> str:
        normalized = cls.STAGE_ALIAS.get(raw_stage, raw_stage)
        if normalized not in cls.STAGE_IMPL:
            raise ValueError(
                "Unsupported DYNIN omni model_stage: "
                f"{raw_stage} (normalized={normalized}). "
                f"Supported: {sorted(cls.STAGE_IMPL.keys())}"
            )
        return normalized

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        raw_stage = str(getattr(vllm_config.model_config, "model_stage", "token2text")).lower()
        self.model_stage = self._normalize_stage_name(raw_stage)

        impl_cls = self._resolve_stage_impl_class(self.model_stage)
        self.impl = impl_cls(vllm_config=vllm_config, prefix=prefix)
        self.model = self.impl

        self.has_preprocess = False
        self.has_postprocess = False
        self.have_multimodal_outputs = getattr(self.impl, "have_multimodal_outputs", True)
        self.requires_raw_input_tokens = getattr(self.impl, "requires_raw_input_tokens", True)
        self.language_model = self._resolve_language_model()

    def _resolve_language_model(self) -> Any | None:
        if hasattr(self.impl, "get_language_model"):
            language_model = self.impl.get_language_model()
            if language_model is not None:
                return language_model

        if hasattr(self.impl, "language_model"):
            language_model = getattr(self.impl, "language_model")
            if language_model is not None:
                return language_model

        if self.model_stage == "token2text":
            return getattr(self.impl, "model", None)

        return None

    def get_language_model(self) -> Any | None:
        return self.language_model

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        if self.language_model is not None and hasattr(self.language_model, "sampler"):
            return self.language_model.sampler
        return Sampler()

    def init_multi_modal(self, thinker_config: Any = None) -> None:
        if hasattr(self.model, "init_multi_modal"):
            self.model.init_multi_modal(thinker_config)

    def _collect_multimodal_inputs(self, **kwargs: Any) -> dict[str, Any]:
        mm_inputs: dict[str, Any] = {}
        for modality, aliases in _MODALITY_INPUT_ALIASES.items():
            for alias in aliases:
                if alias in kwargs and kwargs[alias] is not None:
                    mm_inputs[modality] = kwargs[alias]
                    break
        return mm_inputs

    def _normalize_loaded_weight_names(
        self,
        loaded: set[str],
        expected_param_names: set[str],
    ) -> set[str]:
        if self.model_stage != "token2text":
            return loaded

        normalized_loaded: set[str] = set()
        prefixes = ("", "impl.", "impl.model.")

        for name in loaded:
            for prefix in prefixes:
                candidate = f"{prefix}{name}" if prefix else name
                if candidate in expected_param_names:
                    normalized_loaded.add(candidate)
                    break

        if len(normalized_loaded) < len(expected_param_names):
            normalized_loaded.update(expected_param_names)

        return normalized_loaded

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        squeezed_batch = False
        staged_input_ids = input_ids

        if input_ids.ndim == 0:
            staged_input_ids = input_ids.view(1, 1)
            squeezed_batch = True
        elif input_ids.ndim == 1:
            staged_input_ids = input_ids.unsqueeze(0)
            squeezed_batch = True

        embeddings = self.model.embed_input_ids(
            staged_input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            **kwargs,
        )

        if squeezed_batch and isinstance(embeddings, torch.Tensor):
            if embeddings.ndim == 3 and embeddings.shape[0] == 1:
                return embeddings.squeeze(0)
            if embeddings.ndim == 2 and input_ids.ndim == 0 and embeddings.shape[0] == 1:
                return embeddings

        return embeddings

    def embed_multimodal(self, **kwargs: Any) -> Any:
        if hasattr(self.model, "embed_multimodal"):
            return self.model.embed_multimodal(**kwargs)

        self._collect_multimodal_inputs(**kwargs)
        return None

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loaded = self.model.load_weights(weights)
        if loaded is None:
            loaded = set()

        expected_param_names = {name for name, _ in self.named_parameters()}
        if not expected_param_names:
            return loaded

        return self._normalize_loaded_weight_names(loaded, expected_param_names)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, sampling_metadata=sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if hasattr(self.model, "sample"):
            return self.model.sample(logits, sampling_metadata)
        if self.language_model is not None and hasattr(self.language_model, "sample"):
            return self.language_model.sample(logits, sampling_metadata)
        return None

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
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
    PlaceholderRange,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - defensive fallback
    PILImage = None


_DYNIN_MM_INPUT_KEY_BY_MODALITY = {
    "image": "pixel_values",
    "video": "pixel_values_videos",
    "audio": "input_audio_features",
}


def _dynin_placeholder_str(modality: str, i: int) -> str | None:
    del i
    if modality.startswith("image"):
        return "<|soi|><|image|><|eoi|>"
    if modality.startswith("video"):
        return "<|sov|><|video|><|eov|>"
    if modality.startswith("audio"):
        return "<|soa|><|audio|><|eoa|>"
    return None


class DyninOmniProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        return DyninOmniMultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Keep limits conservative until DYNIN has modality-specific prompt
        # expansion and embedding lengths wired for online serving.
        return {
            "image": 1,
            "video": 1,
            "audio": 1,
        }

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        del seq_len, mm_counts
        # Use one encoder token per modality item in current compatibility path.
        return {
            "image": 1,
            "video": 1,
            "audio": 1,
        }


class DyninOmniDummyInputsBuilder(BaseDummyInputsBuilder[DyninOmniProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        placeholder_chunks: list[str] = []
        for modality in ("image", "video", "audio"):
            for item_idx in range(mm_counts.get(modality, 0)):
                placeholder = _dynin_placeholder_str(modality, item_idx)
                if placeholder:
                    placeholder_chunks.append(placeholder)
        return " ".join(placeholder_chunks)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        del seq_len
        mm_data: dict[str, Any] = {}

        num_images = mm_counts.get("image", 0)
        if num_images > 0:
            image_overrides = mm_options.get("image") if mm_options else None
            mm_data["image"] = self._get_dummy_images(
                width=224,
                height=224,
                num_images=num_images,
                overrides=image_overrides,
            )

        num_videos = mm_counts.get("video", 0)
        if num_videos > 0:
            video_overrides = mm_options.get("video") if mm_options else None
            mm_data["video"] = self._get_dummy_videos(
                width=224,
                height=224,
                num_frames=8,
                num_videos=num_videos,
                overrides=video_overrides,
            )

        num_audios = mm_counts.get("audio", 0)
        if num_audios > 0:
            audio_overrides = mm_options.get("audio") if mm_options else None
            mm_data["audio"] = self._get_dummy_audios(
                length=16000,
                num_audios=num_audios,
                overrides=audio_overrides,
            )

        return mm_data


class DyninOmniMultiModalDataParser(MultiModalDataParser):
    def _get_audio_with_sr(self, audio: Any) -> tuple[np.ndarray, float | None]:
        audio_array, orig_sr = super()._get_audio_with_sr(audio)
        # Keep parity with image/video parsing in Dynin compatibility path:
        # consume raw waveform directly unless a target sampling rate is
        # explicitly configured for resampling.
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
    def _is_embed_mask(length: int) -> torch.Tensor:
        # Dynin stage-0 currently consumes multimodal payload through custom
        # runtime logic, not vLLM encoder-cache embeddings. Keep placeholder
        # tokens but disable embed indices to avoid encoder cache dependency.
        return torch.zeros(length, dtype=torch.bool)

    @staticmethod
    def _to_prompt_token_ids(prompt: str | list[int], tokenizer: Any | None) -> list[int]:
        if isinstance(prompt, str):
            if tokenizer is None:
                raise ValueError(
                    "Tokenizer is required to process string prompts "
                    "for Dynin multimodal inputs."
                )
            return tokenizer.encode(prompt, add_special_tokens=False)
        return list(prompt)

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

        # Convert HWC -> CHW if needed.
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

        # Convert THWC -> TCHW if needed.
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

    def _build_modality_kwargs(self, modality: str, modality_items: Sequence[Any]) -> Sequence[Any]:
        input_key = _DYNIN_MM_INPUT_KEY_BY_MODALITY[modality]
        tensor_items = [self._convert_modality_item(modality, item) for item in modality_items]
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            {input_key: tensor_items},
            {input_key: MultiModalFieldConfig.batched(modality)},
        )
        return mm_kwargs[modality]

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_inputs, hf_processor_mm_kwargs
        # Unused in this custom `apply` path.
        return {}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        # Prompt updates are handled explicitly in `apply`.
        return []

    def apply(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict | MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInputs:
        if tokenization_kwargs is None:
            tokenization_kwargs = {}

        # vLLM >= 0.16 removed BaseMultiModalProcessor._to_mm_items.
        # Normalize here using processing info for compatibility.
        if isinstance(mm_data, MultiModalDataItems):
            mm_items = mm_data
        else:
            mm_items = self.info.parse_mm_data(mm_data)
        mm_hashes = self._hash_mm_items(
            mm_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        tokenizer = self.info.ctx.tokenizer
        prompt_token_ids = self._to_prompt_token_ids(prompt, tokenizer)
        if not prompt_token_ids:
            prompt_token_ids = [0]

        mm_kwargs_by_modality: dict[str, Sequence[Any]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        search_start = 0

        for modality, item_count in mm_items.get_all_counts().items():
            input_key = _DYNIN_MM_INPUT_KEY_BY_MODALITY.get(modality)
            if input_key is None or item_count <= 0:
                continue

            modality_items = mm_items[modality].get_all()
            if len(modality_items) != item_count:
                raise RuntimeError(
                    f"Parsed {len(modality_items)} items but expected {item_count} "
                    f"for modality={modality!r}"
                )

            mm_kwargs_by_modality[modality] = self._build_modality_kwargs(
                modality,
                modality_items,
            )

            placeholder_ranges: list[PlaceholderRange] = []
            for item_idx in range(item_count):
                placeholder = _dynin_placeholder_str(modality, item_idx)
                placeholder_token_ids: list[int] = []
                if placeholder and tokenizer is not None:
                    placeholder_token_ids = tokenizer.encode(
                        placeholder,
                        add_special_tokens=False,
                    )

                if placeholder_token_ids:
                    found_offset = self._find_subsequence(
                        prompt_token_ids,
                        placeholder_token_ids,
                        search_start,
                    )
                else:
                    found_offset = None

                if found_offset is None:
                    found_offset = min(search_start, len(prompt_token_ids) - 1)
                    placeholder_len = 1
                else:
                    placeholder_len = len(placeholder_token_ids)

                placeholder_ranges.append(
                    PlaceholderRange(
                        offset=found_offset,
                        length=placeholder_len,
                        is_embed=self._is_embed_mask(placeholder_len),
                    )
                )
                search_start = found_offset + placeholder_len

            mm_placeholders[modality] = placeholder_ranges

        return MultiModalInputs(
            type="multimodal",
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=MultiModalKwargsItems(mm_kwargs_by_modality),
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


@MULTIMODAL_REGISTRY.register_processor(
    DyninOmniMultiModalProcessor,
    info=DyninOmniProcessingInfo,
    dummy_inputs=DyninOmniDummyInputsBuilder,
)
class DyninOmniForConditionalGeneration(nn.Module, SupportsMultiModal):
    """DYNIN omni router.

    Primary inference graph:
      token2text -> token2image -> token2audio

    - `token2text`: DYNIN generation + optional text detokenization
    - `token2image`: image detokenization (or pass-through)
    - `token2audio`: audio/speech detokenization (or pass-through)

    Backward-compatible aliases (e.g., token2token/tokenizer/token2wav/token2img)
    are normalized by `STAGE_ALIAS`.
    """

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
        # NOTE: DYNIN currently uses placeholders primarily for OpenAI chat
        # ingestion compatibility. Native DYNIN prompting is task-driven.
        return _dynin_placeholder_str(modality, i)

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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        raw_stage = str(getattr(vllm_config.model_config, "model_stage", "token2text")).lower()
        model_stage = self.STAGE_ALIAS.get(raw_stage, raw_stage)
        if model_stage not in self.STAGE_IMPL:
            raise ValueError(
                "Unsupported DYNIN omni model_stage: "
                f"{raw_stage} (normalized={model_stage}). "
                f"Supported: {sorted(self.STAGE_IMPL.keys())}"
            )

        self.model_stage = model_stage
        impl_cls = self._resolve_stage_impl_class(model_stage)
        self.impl = impl_cls(vllm_config=vllm_config, prefix=prefix)
        # Keep parity with Qwen omni wrappers: active stage module as `self.model`.
        self.model = self.impl
        self.has_preprocess = False
        self.has_postprocess = False
        self.have_multimodal_outputs = getattr(self.impl, "have_multimodal_outputs", True)
        self.requires_raw_input_tokens = getattr(self.impl, "requires_raw_input_tokens", True)
        # DYNIN token2text stage loads DyninOmniModelLM (LLaDA backbone) into `impl.model`.
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
        # token2text keeps the backbone model in `impl.model`.
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
        # DYNIN stages currently do not require explicit multimodal tower init,
        # but keep this hook for API parity with Qwen omni models.
        if hasattr(self.model, "init_multi_modal"):
            self.model.init_multi_modal(thinker_config)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: Any) -> dict[str, Any]:
        mm_input_by_modality: dict[str, Any] = {}
        for input_key, value in kwargs.items():
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = value
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = value
            if input_key in ("input_audio_features", "audio_embeds") and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = value
        return mm_input_by_modality

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
        # vLLM V1 generation path can pass scheduled token IDs as a 1D tensor.
        # Some stage implementations still assume 2D input_ids.
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
        # API parity path; DYNIN currently does not materialize embeddings here.
        self._parse_and_validate_multimodal_inputs(**kwargs)
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

        if self.model_stage != "token2text":
            return loaded

        # token2text stage preloads the local DyninOmniModelLM submodel inside
        # impl.__init__ via DyninOmniModelLM.from_pretrained(). vLLM still asks for load_weights()
        # and validates returned parameter names against this wrapper module.
        # Normalize checkpoint keys to wrapper parameter names and fallback to
        # "all expected loaded" for already-preloaded parameters.
        normalized_loaded: set[str] = set()
        for name in loaded:
            if name in expected_param_names:
                normalized_loaded.add(name)
                continue
            impl_name = f"impl.{name}"
            if impl_name in expected_param_names:
                normalized_loaded.add(impl_name)
                continue
            impl_model_name = f"impl.model.{name}"
            if impl_model_name in expected_param_names:
                normalized_loaded.add(impl_model_name)
                continue

        if len(normalized_loaded) < len(expected_param_names):
            normalized_loaded.update(expected_param_names)

        return normalized_loaded

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

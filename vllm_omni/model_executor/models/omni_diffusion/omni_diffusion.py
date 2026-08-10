from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from transformers import AutoConfig, AutoModel, AutoTokenizer
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.inputs import MultiModalInput as MultiModalInputs
from vllm.logger import init_logger
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

from vllm_omni.model_executor.models.omni_diffusion.audio_tokenizer import OmniDiffusionAudioTokenizer
from vllm_omni.model_executor.models.omni_diffusion.chat_template import (
    OMNI_DIFFUSION_CHAT_TEMPLATE,
    normalize_chat_template_token_ids,
)
from vllm_omni.model_executor.models.omni_diffusion.component_paths import (
    OMNI_DIFFUSION_FLOW_DECODER_REPO_ID,
    OMNI_DIFFUSION_IMAGE_TOKENIZER_REPO_ID,
    OMNI_DIFFUSION_SENSEVOICE_REPO_ID,
    resolve_omni_diffusion_component_path,
)
from vllm_omni.model_executor.models.omni_diffusion.dream_compat import (
    ensure_default_rope_init_function,
    ensure_dream_generation_config_fields,
    ensure_dream_rope_parameters,
    initialize_dream_generation_config,
    patch_legacy_dream_generation_config_validate,
    patch_remote_dream_generation_config_validate,
    repair_default_dream_rope_buffers,
)
from vllm_omni.model_executor.models.omni_diffusion.image_tokenizer import OmniDiffusionImageTokenizer
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_AUDIO_CODEBOOK_SIZE,
    OMNI_DIFFUSION_AUDIO_START_TOKEN,
    OMNI_DIFFUSION_END_OF_TEXT_TOKEN,
    OMNI_DIFFUSION_IM_END_TOKEN,
    OMNI_DIFFUSION_IMAGE_CODEBOOK_SIZE,
    OMNI_DIFFUSION_IMAGE_START_TOKEN,
    OMNI_DIFFUSION_INPUT_SAMPLE_RATE,
    OMNI_DIFFUSION_OUTPUT_SAMPLE_RATE,
    OmniDiffusionModelSpecialTokens,
    OmniDiffusionTokenizerBaseData,
    set_generation_seed,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

_MODALITY_ORDER = ("image", "audio")
_MODALITY_INPUT_KEY_BY_NAME = {
    "image": "omni_images",
    "audio": "omni_audios",
}
_MODALITY_METADATA_KEY_BY_NAME = {
    "image": "omni_image_sizes",
    "audio": "omni_audio_sample_rates",
}
_MODALITY_PLACEHOLDER_BY_NAME = {
    "image": OmniDiffusionModelSpecialTokens.IMG_TAG.value,
    "audio": OmniDiffusionModelSpecialTokens.AUD_TAG.value,
}

# Each task initializes only the side components used by its input or output.
_IMAGE_TOKENIZER_TASKS = frozenset({"T2I", "VQA", "SVQA"})
_AUDIO_TOKENIZER_TASKS = frozenset({"ASR", "TTS", "SVQA"})
_SENSEVOICE_TASKS = frozenset({"ASR", "SVQA"})
_GLM4VOICE_DECODER_TASKS = frozenset({"TTS"})
_SUPPORTED_TASKS = _IMAGE_TOKENIZER_TASKS | _AUDIO_TOKENIZER_TASKS
logger = init_logger(__name__)


def _is_dummy_run(runtime_additional_information: Any) -> bool:
    """Return whether vLLM marked this forward call as a profiling run."""
    if isinstance(runtime_additional_information, list):
        if not runtime_additional_information:
            return False
        runtime_additional_information = runtime_additional_information[0]
    if not isinstance(runtime_additional_information, dict):
        return False
    return bool(runtime_additional_information.get("_is_dummy", False))


class OmniDiffusionProcessingInfo(BaseProcessingInfo):
    """Processing metadata for Omni-Diffusion.

    Omni-Diffusion's official inference accepts at most one input image and
    one input audio item in the demo path we are matching first. Image inputs
    are converted to discrete image tokens, while user audio is passed through
    the contiguous-audio path.
    """

    def get_data_parser(self) -> MultiModalDataParser:
        return OmniDiffusionMultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            target_sr=OMNI_DIFFUSION_INPUT_SAMPLE_RATE,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {modality: 1 for modality in _MODALITY_ORDER}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        del seq_len, mm_counts
        return {modality: 1 for modality in _MODALITY_ORDER}


class OmniDiffusionDummyInputsBuilder(BaseDummyInputsBuilder[OmniDiffusionProcessingInfo]):
    """Dummy inputs for vLLM profiling.

    Keep this small initially; fill it once the processor contract is settled.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        parts = ["hello"]
        if mm_counts.get("image", 0) > 0:
            parts.append("<|image|>")
        if mm_counts.get("audio", 0) > 0:
            parts.append("<|audio|>")
        return "\n".join(parts)

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
            mm_data["image"] = self._get_dummy_images(
                width=512,
                height=512,
                num_images=num_images,
                overrides=mm_options.get("image") if mm_options else None,
            )

        num_audios = mm_counts.get("audio", 0)
        if num_audios > 0:
            mm_data["audio"] = self._get_dummy_audios(
                length=16000,
                num_audios=num_audios,
                overrides=mm_options.get("audio") if mm_options else None,
            )

        return mm_data


class OmniDiffusionMultiModalDataParser(MultiModalDataParser):
    """Parse Omni-Diffusion raw multimodal inputs.

    Accepts vLLM's standard modality keys:
    - ``image`` / ``img2img`` for a single input image.
    - ``audio`` for a single input audio item.

    ``img2img`` is the serving-layer alias used when a request has both an
    input image and image output. Omni-Diffusion consumes it like a normal
    image input, so normalize it here instead of teaching the shared pipeline
    another model-specific spelling.

    Model-specific conversion, such as MAGVIT image tokenization and
    contiguous-audio encoding, belongs to the processor/model wrapper.
    """

    def _get_subparsers(self):
        parsers = super()._get_subparsers()
        parsers["img2img"] = self._parse_image_data
        return parsers

    def parse_mm_data(self, mm_data: MultiModalDataDict, **kwargs: object) -> MultiModalDataItems:
        normalized = {"image" if modality == "img2img" else modality: data for modality, data in mm_data.items()}
        return super().parse_mm_data(normalized, **kwargs)


class OmniDiffusionMultiModalProcessor(BaseMultiModalProcessor[OmniDiffusionProcessingInfo]):
    """Convert vLLM multimodal inputs into Omni-Diffusion model inputs.

    Intended first implementation:
    - accept prompt + image/audio from vLLM renderer,
    - preserve task/steps/cfg/alg in runtime additional information,
    - return MultiModalInputs that let forward call the official HF generate.
    """

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

    @staticmethod
    def _ensure_official_chat_template(tokenizer: Any | None) -> None:
        if tokenizer is not None:
            tokenizer.chat_template = OMNI_DIFFUSION_CHAT_TEMPLATE

    @staticmethod
    def _encode_prompt_to_token_ids(
        prompt: str | list[int],
        tokenizer: Any,
    ) -> list[int]:
        if isinstance(prompt, str):
            if "<|im_start|>" not in prompt:
                rendered_token_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
                return normalize_chat_template_token_ids(rendered_token_ids)
            return tokenizer.encode(prompt, add_special_tokens=False)
        return list(prompt)

    @staticmethod
    def _ensure_audio_placeholder(
        prompt: str | list[int],
        mm_counts: Mapping[str, int],
    ) -> str | list[int]:
        if not isinstance(prompt, str) or mm_counts.get("audio", 0) <= 0:
            return prompt

        audio_placeholder = OmniDiffusionModelSpecialTokens.AUD_TAG.value
        if audio_placeholder in prompt:
            return prompt

        user_to_assistant_marker = "<|im_end|>\n<|im_start|>assistant\n"
        if user_to_assistant_marker in prompt:
            # Some paths hand us an already-rendered chat prompt. Keep the
            # audio marker inside the user turn so Dream can replace it with
            # contiguous audio tokens before generation.
            return prompt.replace(
                user_to_assistant_marker,
                f"\n{audio_placeholder}{user_to_assistant_marker}",
                1,
            )

        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in prompt:
            return prompt.replace(assistant_marker, f"{audio_placeholder}\n{assistant_marker}", 1)
        return f"{prompt}\n{audio_placeholder}" if prompt else audio_placeholder

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
    def _image_to_chw_float_tensor(image: torch.Tensor | np.ndarray | PILImage.Image) -> torch.Tensor:
        channels_last = False
        if isinstance(image, torch.Tensor):
            tensor = image.detach()
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
            channels_last = True
        elif isinstance(image, PILImage.Image):
            tensor = torch.from_numpy(np.asarray(image).copy())
            channels_last = True
        else:
            raise TypeError(f"Unsupported image item type: {type(image)!r}")

        match tensor.ndim:
            # [H, W, C]
            case 3:
                pass
            # [H, W] -> [H, W, 1]
            case 2:
                tensor = tensor.unsqueeze(-1)
                channels_last = True
            case _:
                raise ValueError(f"Expected 3D image tensor, got shape={tuple(tensor.shape)}")

        # PIL/NumPy inputs have known channel-last layouts. Tensor inputs use
        # PyTorch's channel-first convention unless only their last dimension
        # can represent gray/RGB/RGBA channels.
        channels_last = channels_last or (tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4))
        if channels_last:
            tensor = tensor.permute(2, 0, 1)

        # Grayscale [1, H, W] -> RGB-like [3, H, W] by copying the gray channel.
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        # RGBA [4, H, W] -> RGB [3, H, W] by dropping the alpha channel.
        if tensor.shape[0] == 4:
            tensor = tensor[:3]

        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() > 0 and torch.max(tensor) > 1.0:
            tensor = tensor / 255.0
        return tensor.contiguous()

    @staticmethod
    def _image_size_hw_tensor(image: PILImage.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, PILImage.Image):
            width, height = image.size
            return torch.tensor([height, width], dtype=torch.int32)

        if isinstance(image, np.ndarray):
            shape = image.shape
            channels_last = True
        elif isinstance(image, torch.Tensor):
            shape = tuple(image.shape)
            channels_last = False
        else:
            raise TypeError(f"Unsupported image item type: {type(image)!r}")

        if len(shape) == 2:
            height, width = shape
        elif len(shape) == 3:
            if channels_last or (shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4)):
                height, width = shape[0], shape[1]
            else:
                height, width = shape[1], shape[2]
        else:
            raise ValueError(f"Expected 2D or 3D image shape, got shape={shape}")

        return torch.tensor([int(height), int(width)], dtype=torch.int32)

    @staticmethod
    def _audio_to_float_tensor(
        audio: tuple[torch.Tensor | np.ndarray, Any] | torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        # vLLM audio items may be (audio_waveform, sample_rate); sample rate is
        # handled separately by _audio_sample_rate_tensor.
        if isinstance(audio, tuple) and len(audio) == 2:
            audio = audio[0]

        if isinstance(audio, torch.Tensor):
            tensor = audio.detach()
        elif isinstance(audio, np.ndarray):
            tensor = torch.from_numpy(audio)
        else:
            raise TypeError(f"Unsupported audio item type: {type(audio)!r}")

        tensor = tensor.to(dtype=torch.float32).contiguous()
        # [T] is mono waveform; [C, T] keeps multi-channel audio for the tokenizer to normalize.
        if tensor.ndim not in (1, 2):
            raise ValueError(f"Expected audio tensor with shape [T] or [C, T], got {tuple(tensor.shape)}.")
        return tensor

    @staticmethod
    def _audio_sample_rate_tensor(audio: Any) -> torch.Tensor:
        sample_rate = OMNI_DIFFUSION_INPUT_SAMPLE_RATE
        if isinstance(audio, tuple) and len(audio) == 2:
            sample_rate = int(audio[1])
        return torch.tensor(sample_rate, dtype=torch.int32)

    @classmethod
    def _convert_modality_item(cls, modality: str, item: Any) -> torch.Tensor:
        match modality:
            case "image":
                return cls._image_to_chw_float_tensor(item)
            case "audio":
                return cls._audio_to_float_tensor(item)
            case _:
                raise ValueError(f"Unsupported modality for Omni-Diffusion processor: {modality}")

    @classmethod
    def _convert_modality_metadata(cls, modality: str, item: Any) -> torch.Tensor:
        match modality:
            case "image":
                return cls._image_size_hw_tensor(item)
            case "audio":
                return cls._audio_sample_rate_tensor(item)
            case _:
                raise ValueError(f"Unsupported modality for Omni-Diffusion processor: {modality}")

    def _build_modality_kwargs(
        self,
        modality: str,
        modality_items: Sequence[Any],
    ) -> Sequence[Any]:
        input_key = _MODALITY_INPUT_KEY_BY_NAME[modality]
        metadata_key = _MODALITY_METADATA_KEY_BY_NAME[modality]
        tensor_items = [self._convert_modality_item(modality, item) for item in modality_items]
        metadata_items = [self._convert_modality_metadata(modality, item) for item in modality_items]
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            {
                input_key: tensor_items,
                metadata_key: metadata_items,
            },
            {
                input_key: MultiModalFieldConfig.batched(modality),
                metadata_key: MultiModalFieldConfig.batched(modality),
            },
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
            placeholder_text = _MODALITY_PLACEHOLDER_BY_NAME.get(modality)
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
                    is_embed=torch.zeros(placeholder_len, dtype=torch.bool),
                )
            )
            search_start = found_offset + placeholder_len

        return ranges, search_start

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
        self._ensure_official_chat_template(tokenizer)
        mm_counts = mm_items.get_all_counts()
        prompt = self._ensure_audio_placeholder(prompt, mm_counts)
        prompt_token_ids = self._encode_prompt_to_token_ids(prompt, tokenizer)
        prompt_token_ids = self._ensure_non_empty_prompt_ids(prompt_token_ids, tokenizer)

        mm_kwargs_by_modality: dict[str, Sequence[Any]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        search_start = 0

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


@dataclass(frozen=True)
class OmniDiffusionAdditionalConfig:
    """Model and generation settings parsed from an Omni-Diffusion deploy config."""

    image_tokenizer_path: str | None
    audio_tokenizer_type: str
    flow_path: str | None
    sensevoice_path: str | None
    attn_implementation: str
    output_text_only: bool
    seed: int | None
    task: str
    steps: int
    max_new_tokens: int
    alg: str
    cfg: float
    temperature: float
    top_p: float
    add_boa_token: int
    max_position_penalty: float
    repeat_penalty: float
    top_k: int | None

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> OmniDiffusionAdditionalConfig:
        """Parse Omni-Diffusion's additional_config once during model startup."""

        additional_config = getattr(vllm_config, "additional_config", {})
        logger.info("Initializing Omni-Diffusion additional config from vLLM additional config: %s", additional_config)

        if not isinstance(additional_config, Mapping):
            raise TypeError(f"Omni-Diffusion additional_config must be a mapping, got {type(additional_config)!r}.")
        additional_config = dict(additional_config)

        task = additional_config.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Omni-Diffusion additional_config.task must be a non-empty string, got {task!r}.")
        parsed_task = task.strip().upper()
        if parsed_task not in _SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported Omni-Diffusion task {parsed_task!r}; expected one of {sorted(_SUPPORTED_TASKS)}."
            )

        image_tokenizer_path = (
            resolve_omni_diffusion_component_path(
                additional_config.get("image_tokenizer_path"),
                config_key="additional_config.image_tokenizer_path",
                default_repo_id=OMNI_DIFFUSION_IMAGE_TOKENIZER_REPO_ID,
            )
            if parsed_task in _IMAGE_TOKENIZER_TASKS
            else None
        )

        audio_tokenizer_type = str(additional_config.get("audio_tokenizer_type", "sensevoice_glm4voice"))
        if audio_tokenizer_type != "sensevoice_glm4voice":
            raise ValueError(
                "Omni-Diffusion currently supports only additional_config.audio_tokenizer_type='sensevoice_glm4voice'."
            )

        flow_path = (
            resolve_omni_diffusion_component_path(
                additional_config.get("flow_path"),
                config_key="additional_config.flow_path",
                default_repo_id=OMNI_DIFFUSION_FLOW_DECODER_REPO_ID,
            )
            if parsed_task in _GLM4VOICE_DECODER_TASKS
            else None
        )
        sensevoice_path = (
            resolve_omni_diffusion_component_path(
                additional_config.get("sensevoice_path"),
                config_key="additional_config.sensevoice_path",
                default_repo_id=OMNI_DIFFUSION_SENSEVOICE_REPO_ID,
            )
            if parsed_task in _SENSEVOICE_TASKS
            else None
        )

        attn_implementation = str(additional_config.get("attn_implementation", "flash_attention_2"))
        output_text_only = bool(additional_config.get("output_text_only", False))

        seed = additional_config.get("seed")
        if seed is None:
            parsed_seed = None
        elif isinstance(seed, int):
            parsed_seed = seed
        else:
            logger.warning("Omni-Diffusion additional_config.seed must be an integer, got %r.", seed)
            parsed_seed = None

        steps = additional_config.get("steps")
        if steps is not None and isinstance(steps, int) and steps >= 0:
            parsed_steps = steps
        else:
            logger.warning("Omni-Diffusion additional_config.steps must be a non-negative integer, got %r.", steps)
            parsed_steps = 128

        max_new_tokens = additional_config.get("max_new_tokens")
        if max_new_tokens is not None and isinstance(max_new_tokens, int) and max_new_tokens >= 0:
            parsed_max_new_tokens = max_new_tokens
        else:
            logger.warning(
                "Omni-Diffusion additional_config.max_new_tokens must be a non-negative integer, got %r.",
                max_new_tokens,
            )
            parsed_max_new_tokens = 128

        alg = additional_config.get("alg")
        if alg is not None and str(alg).strip():
            parsed_alg = str(alg)
        else:
            logger.warning("Omni-Diffusion additional_config.alg must be a non-empty string, got %r.", alg)
            parsed_alg = "entropy"

        cfg = additional_config.get("cfg")
        if cfg is not None and isinstance(cfg, (int, float)):
            parsed_cfg = float(cfg)
        else:
            logger.warning("Omni-Diffusion additional_config.cfg must be a number, got %r.", cfg)
            parsed_cfg = 0.0

        temperature = additional_config.get("temperature")
        if temperature is not None and isinstance(temperature, (int, float)):
            parsed_temperature = float(temperature)
        else:
            logger.warning("Omni-Diffusion additional_config.temperature must be a number, got %r.", temperature)
            parsed_temperature = 0.0

        top_p = additional_config.get("top_p")
        if top_p is not None and isinstance(top_p, (int, float)):
            parsed_top_p = float(top_p)
        else:
            logger.warning("Omni-Diffusion additional_config.top_p must be a number, got %r.", top_p)
            parsed_top_p = 0.9

        add_boa_token = additional_config.get("add_boa_token", 0)
        if isinstance(add_boa_token, int) and add_boa_token >= 0:
            parsed_add_boa_token = add_boa_token
        else:
            logger.warning(
                "Omni-Diffusion additional_config.add_boa_token must be a non-negative integer, got %r.",
                add_boa_token,
            )
            parsed_add_boa_token = 0

        max_position_penalty = additional_config.get("max_position_penalty")
        if max_position_penalty is not None and isinstance(max_position_penalty, (int, float)):
            parsed_max_position_penalty = float(max_position_penalty)
        else:
            logger.warning(
                "Omni-Diffusion additional_config.max_position_penalty must be a number, got %r.",
                max_position_penalty,
            )
            parsed_max_position_penalty = 1.0

        repeat_penalty = additional_config.get("repeat_penalty")
        if repeat_penalty is not None and isinstance(repeat_penalty, (int, float)):
            parsed_repeat_penalty = float(repeat_penalty)
        else:
            logger.warning(
                "Omni-Diffusion additional_config.repeat_penalty must be a number, got %r.",
                repeat_penalty,
            )
            parsed_repeat_penalty = 1.0

        top_k = additional_config.get("top_k")
        if top_k is None:
            parsed_top_k = None
        elif isinstance(top_k, int) and top_k >= 0:
            parsed_top_k = top_k
        else:
            logger.warning(
                "Omni-Diffusion additional_config.top_k must be a non-negative integer, got %r.",
                top_k,
            )
            parsed_top_k = None

        config = cls(
            image_tokenizer_path=image_tokenizer_path,
            audio_tokenizer_type=audio_tokenizer_type,
            flow_path=flow_path,
            sensevoice_path=sensevoice_path,
            attn_implementation=attn_implementation,
            output_text_only=output_text_only,
            seed=parsed_seed,
            task=parsed_task,
            steps=parsed_steps,
            max_new_tokens=parsed_max_new_tokens,
            alg=parsed_alg,
            cfg=parsed_cfg,
            temperature=parsed_temperature,
            top_p=parsed_top_p,
            add_boa_token=parsed_add_boa_token,
            max_position_penalty=parsed_max_position_penalty,
            repeat_penalty=parsed_repeat_penalty,
            top_k=parsed_top_k,
        )

        logger.info("Omni-Diffusion additional config initialized: %s", config)
        return config


@dataclass(frozen=True)
class OmniDiffusionForwardKwargs:
    omni_images: torch.Tensor | Sequence | None
    omni_audios: Any | None
    omni_audio_sample_rates: Any | None

    @classmethod
    def from_forward_kwargs(cls, kwargs: dict[str, Any]) -> OmniDiffusionForwardKwargs:
        omni_images = kwargs.pop("omni_images", None)
        omni_audios = kwargs.pop("omni_audios", None)
        omni_audio_sample_rates = kwargs.pop("omni_audio_sample_rates", None)
        return OmniDiffusionForwardKwargs(
            omni_images=omni_images,
            omni_audios=omni_audios,
            omni_audio_sample_rates=omni_audio_sample_rates,
        )


@MULTIMODAL_REGISTRY.register_processor(
    OmniDiffusionMultiModalProcessor,
    info=OmniDiffusionProcessingInfo,
    dummy_inputs=OmniDiffusionDummyInputsBuilder,
)
class OmniDiffusionForConditionalGeneration(nn.Module, SupportsMultiModal):
    """HF-wrapper skeleton for lijiang/Omni-Diffusion.

    This file is intentionally minimal for now so the processor/model shape can
    be built while reading Dynin side by side.
    """

    supports_multimodal_raw_input_only = True
    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    logitsprocs_need_output_token_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        self.prefix = prefix
        self.additional_config = OmniDiffusionAdditionalConfig.from_vllm_config(vllm_config)

        model_config = vllm_config.model_config
        self.model_path = model_config.model
        self.device = vllm_config.device_config.device
        self.dtype = model_config.dtype

        self.hidden_size = int(model_config.get_hidden_size())
        self.add_generation_prompt = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=model_config.trust_remote_code,
            chat_template=OMNI_DIFFUSION_CHAT_TEMPLATE,
        )
        self.tokenizer_base_data = OmniDiffusionTokenizerBaseData(self.tokenizer)

        hf_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=model_config.trust_remote_code,
        )
        ensure_dream_rope_parameters(hf_config)
        ensure_default_rope_init_function()
        patch_remote_dream_generation_config_validate(
            self.model_path,
            model_config.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            config=hf_config,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=self.dtype,
            attn_implementation=self.additional_config.attn_implementation,
        ).to(self.device)
        self.model.eval()

        repair_default_dream_rope_buffers(self.model)

        logger.info(
            "Omni-Diffusion model load config: vllm_dtype=%s dtype=%s "
            "attn_implementation=%s model_class=%s.%s model_source=%s",
            self.dtype,
            self.dtype,
            self.additional_config.attn_implementation,
            type(self.model).__module__,
            type(self.model).__name__,
            inspect.getsourcefile(type(self.model)),
        )

        initialize_dream_generation_config(
            model=self.model,
            tokenizer=self.tokenizer,
            model_path=self.model_path,
            trust_remote_code=model_config.trust_remote_code,
            top_k=self.additional_config.top_k,
        )

        self.image_tokenizer: OmniDiffusionImageTokenizer | None = None
        if self.additional_config.task in _IMAGE_TOKENIZER_TASKS:
            image_tokenizer_path = self.additional_config.image_tokenizer_path
            if image_tokenizer_path is None:
                raise RuntimeError(
                    f"Omni-Diffusion task {self.additional_config.task!r} requires an image tokenizer path."
                )
            self.image_tokenizer = OmniDiffusionImageTokenizer(
                model_path=image_tokenizer_path,
                device=self.device,
            )

        self.audio_tokenizer: OmniDiffusionAudioTokenizer | None = None
        if self.additional_config.task in _AUDIO_TOKENIZER_TASKS:
            self.audio_tokenizer = OmniDiffusionAudioTokenizer(
                sensevoice_path=self.additional_config.sensevoice_path,
                flow_path=self.additional_config.flow_path,
                device=self.device,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds

        if input_ids is None:
            raise ValueError("Omni-Diffusion requires input_ids.")

        if _is_dummy_run(kwargs.get("runtime_additional_information")):
            return self._empty_omni_output(input_ids, cpu_hidden_states=True)

        kwargs_data = OmniDiffusionForwardKwargs.from_forward_kwargs(kwargs=kwargs)

        raw_input_ids = self._get_single_prompt_token_ids(input_ids)
        prepared_input_ids = self._prepare_image_inputs(
            input_ids=raw_input_ids,
            omni_images=kwargs_data.omni_images,
        )
        prepared_input_ids, audios, audio_indices = self._prepare_audio_inputs(
            input_ids=prepared_input_ids,
            omni_audios=kwargs_data.omni_audios,
            omni_audio_sample_rates=kwargs_data.omni_audio_sample_rates,
        )
        input_ids = torch.tensor(
            [prepared_input_ids],
            dtype=torch.long,
            device=input_ids.device,
        )

        set_generation_seed(self.additional_config.seed)
        patch_legacy_dream_generation_config_validate(self.model.generation_config)
        ensure_dream_generation_config_fields(
            self.model.generation_config,
            self.model.config,
            self.tokenizer,
        )
        outputs, histories = self.model.generate(
            input_ids,
            generation_config=self.model.generation_config,
            audios=audios,
            audio_indices=audio_indices,
            temperature=self.additional_config.temperature,
            top_p=self.additional_config.top_p,
            steps=self.additional_config.steps,
            max_new_tokens=self.additional_config.max_new_tokens,
            alg=self.additional_config.alg,
            cfg=self.additional_config.cfg,
            tokenizer=self.tokenizer,
            add_boa_token=self.additional_config.add_boa_token,
            max_position_penalty=self.additional_config.max_position_penalty,
            repeat_penalty=self.additional_config.repeat_penalty,
            output_text_only=self.additional_config.output_text_only,
            task=self.additional_config.task,
        )
        del histories

        generated_token_ids = outputs[0][input_ids.shape[1] :]
        text_token_ids, audio_token_ids, image_token_ids = self._split_generated_token_ids(generated_token_ids)
        output_text_token_ids = self._trim_generated_text_token_ids(text_token_ids)
        multimodal_outputs: dict[str, torch.Tensor] = {
            "token_ids": generated_token_ids,
            "text_token_ids": output_text_token_ids,
            "audio_token_ids": audio_token_ids,
            "image_token_ids": image_token_ids,
        }
        if audio_token_ids.numel() > 0 and not self.additional_config.output_text_only:
            decoded_audio = self._get_audio_tokenizer().decode(audio_token_ids)
            multimodal_outputs["audio"] = decoded_audio
            multimodal_outputs["sr"] = torch.tensor(OMNI_DIFFUSION_OUTPUT_SAMPLE_RATE)

        if image_token_ids.numel() > 0 and not self.additional_config.output_text_only:
            decoded_image = self._get_image_tokenizer().decode(image_token_ids)
            image = decoded_image.squeeze(0)
            multimodal_outputs["image"] = image

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs=multimodal_outputs,
        )

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        del batch_size, dtype, device
        return IntermediateTensors({})

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor | None:
        del hidden_states, sampling_metadata
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        del logits, sampling_metadata
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The HF model has already loaded its checkpoint in __init__. Report
        # the wrapper-level names so vLLM does not treat those parameters as
        # uninitialized after exhausting its own checkpoint iterator.
        parameter_names = {name for name, _ in self.named_parameters()}
        loaded_names = set()
        for name, _ in weights:
            wrapper_name = f"model.{name}"
            loaded_names.add(wrapper_name if wrapper_name in parameter_names else name)
        return loaded_names

    def get_language_model(self) -> nn.Module:
        # vLLM 0.22 probes wrappers for their language model during load_model()
        # for MoE-related setup. Omni-Diffusion drives generation through the HF
        # DreamModel.generate path, but this wrapper owns the vLLM-facing
        # embed_input_ids implementation, so expose the wrapper itself.
        return self

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # The generation worker requires input embeddings while preparing a
        # multimodal batch. Omni-Diffusion's HF generate path consumes the
        # original token IDs instead, so only the shape is relevant here.
        del multimodal_embeddings, is_multimodal, kwargs
        shape = (1, self.hidden_size) if input_ids.ndim == 0 else (*input_ids.shape, self.hidden_size)
        return torch.zeros(
            shape,
            dtype=self.dtype,
            device=input_ids.device,
        )

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, bool]]:
        return [{"_is_dummy": True} for _ in range(num_reqs)]

    def _empty_omni_output(
        self,
        input_ids: torch.Tensor,
        *,
        cpu_hidden_states: bool = False,
    ) -> OmniOutput:
        hidden_states_device = torch.device("cpu") if cpu_hidden_states else input_ids.device
        return OmniOutput(
            text_hidden_states=torch.zeros(
                (input_ids.numel(), self.hidden_size),
                dtype=self.dtype,
                device=hidden_states_device,
            ),
            multimodal_outputs={},
        )

    def _prepare_image_inputs(
        self,
        input_ids: list[int],
        omni_images: Any,
    ) -> list[int]:
        if omni_images is None:
            return input_ids

        if not self._has_image_placeholder(input_ids):
            logger.debug("Ignoring Omni-Diffusion image tensors because the prompt contains no <|image|> placeholder.")
            return input_ids

        return self._get_image_tokenizer().prepare_image_token_inputs(
            input_ids=input_ids,
            images=omni_images,
            tokenizer=self.tokenizer,
            tokenizer_base_data=self.tokenizer_base_data,
        )

    def _prepare_audio_inputs(
        self,
        input_ids: Sequence[int],
        omni_audios: Any,
        omni_audio_sample_rates: Any,
    ) -> tuple[list[int], list[torch.Tensor] | None, list[torch.Tensor] | None]:
        input_ids = list(input_ids)
        if omni_audios is None:
            return input_ids, None, None

        if not self._has_audio_placeholder(input_ids):
            logger.debug("Ignoring Omni-Diffusion audio tensors because the prompt contains no <|audio|> placeholder.")
            return input_ids, None, None

        return self._get_audio_tokenizer().prepare_contiguous_audio_inputs(
            input_ids=input_ids,
            omni_audios=omni_audios,
            omni_audio_sample_rates=omni_audio_sample_rates,
            tokenizer_base_data=self.tokenizer_base_data,
        )

    def _get_image_tokenizer(self) -> OmniDiffusionImageTokenizer:
        """Return the task's image tokenizer, failing on inconsistent inputs."""
        if self.image_tokenizer is None:
            raise RuntimeError(
                f"Omni-Diffusion task {self.additional_config.task!r} does not initialize an image tokenizer."
            )
        return self.image_tokenizer

    def _get_audio_tokenizer(self) -> OmniDiffusionAudioTokenizer:
        """Return the task's audio tokenizer, failing on inconsistent inputs."""
        if self.audio_tokenizer is None:
            raise RuntimeError(
                f"Omni-Diffusion task {self.additional_config.task!r} does not initialize an audio tokenizer."
            )
        return self.audio_tokenizer

    def _split_generated_token_ids(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_offset = self.tokenizer.convert_tokens_to_ids(OMNI_DIFFUSION_AUDIO_START_TOKEN)
        image_offset = self.tokenizer.convert_tokens_to_ids(OMNI_DIFFUSION_IMAGE_START_TOKEN)

        marker_token_ids = (
            self.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_START),
            self.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_END),
            self.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_START),
            self.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_END),
        )
        marker_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for marker_token_id in marker_token_ids:
            marker_mask |= token_ids == marker_token_id

        content_token_ids = token_ids[~marker_mask]
        audio_mask = (content_token_ids >= audio_offset) & (
            content_token_ids < audio_offset + OMNI_DIFFUSION_AUDIO_CODEBOOK_SIZE
        )
        image_mask = (content_token_ids >= image_offset) & (
            content_token_ids < image_offset + OMNI_DIFFUSION_IMAGE_CODEBOOK_SIZE
        )
        text_mask = ~(audio_mask | image_mask)
        return (
            content_token_ids[text_mask],
            content_token_ids[audio_mask] - audio_offset,
            content_token_ids[image_mask] - image_offset,
        )

    def _trim_generated_text_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.numel() == 0:
            return token_ids

        stop_token_ids: set[int] = set()
        for token_id in (
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.convert_tokens_to_ids(OMNI_DIFFUSION_IM_END_TOKEN),
            self.tokenizer.convert_tokens_to_ids(OMNI_DIFFUSION_END_OF_TEXT_TOKEN),
        ):
            if token_id is not None and int(token_id) >= 0:
                stop_token_ids.add(int(token_id))
        if not stop_token_ids:
            return token_ids

        stop_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for stop_token_id in stop_token_ids:
            stop_mask |= token_ids == stop_token_id
        stop_positions = stop_mask.nonzero(as_tuple=False).flatten()
        if stop_positions.numel() == 0:
            return token_ids
        return token_ids[: int(stop_positions[0].item())]

    def _get_single_prompt_token_ids(self, input_ids: torch.Tensor) -> list[int]:
        match input_ids.ndim:
            case 2:
                pass
            case 1:
                input_ids = input_ids.unsqueeze(0)
            case _:
                raise ValueError(
                    f"Omni-Diffusion currently requires input_ids with shape [1, T], got {tuple(input_ids.shape)}."
                )
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"Omni-Diffusion currently requires input_ids with shape [1, T], got {tuple(input_ids.shape)}."
            )
        return input_ids[0].tolist()

    def _has_image_placeholder(self, input_ids: Sequence[int]) -> bool:
        img_tag = self.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_TAG)
        return img_tag in input_ids

    def _has_audio_placeholder(self, input_ids: Sequence[int]) -> bool:
        aud_tag = self.tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_TAG)
        return aud_tag in input_ids

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        return _MODALITY_PLACEHOLDER_BY_NAME.get(modality)

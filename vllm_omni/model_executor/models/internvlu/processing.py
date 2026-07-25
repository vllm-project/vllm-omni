# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal input processing for InternVL-U Stage 0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from vllm.inputs import MultiModalDataDict, MultiModalInput
from vllm.model_executor.models.internvl import (
    BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import ProcessorInputs, TimingContext
from vllm.transformers_utils.processors.internvl import (
    InternVLImageProcessor,
    InternVLProcessor,
)

DEFAULT_SYSTEM_MESSAGE = (
    "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
)
IMAGE_GENERATION_SYSTEM_MESSAGE = (
    DEFAULT_SYSTEM_MESSAGE + "请通过详细描述图像中物体和背景的颜色、形状、大小、纹理、数量、"
    "文字内容以及空间位置关系等来对图像进行全面描述："
)
IMAGE_EDITING_SYSTEM_MESSAGE = (
    DEFAULT_SYSTEM_MESSAGE + "请描述输入图像的关键特征（颜色、形状、大小、纹理、物体、背景等），"
    "然后解释用户的文本指令应该如何改变或修改图像，"
    "从而生成一个满足用户要求的新图像，并适当保持与原始输入的一致性。："
)

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_IMAGE_PLACEHOLDER = "<image>"
_GENERATION_TOKEN = "<img>"


def _normalize_user_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?,:;":
        text += "."
    return text


def _is_formatted_chat(text: str) -> bool:
    return _IM_START in text and f"{_IM_START}assistant\n" in text


def _prepend_missing_images(text: str, num_images: int) -> str:
    represented_images = text.count(_IMAGE_PLACEHOLDER)
    if "<IMG_CONTEXT>" in text:
        represented_images += text.count("</img>")

    if represented_images > num_images:
        raise ValueError(f"InternVL-U prompt has {represented_images} image placeholders but only {num_images} images")

    missing = num_images - represented_images
    if missing == 0:
        return text

    prefix = f"{_IMAGE_PLACEHOLDER}\n" * missing
    if not _is_formatted_chat(text):
        return prefix + text

    user_marker = f"{_IM_START}user\n"
    user_start = text.rfind(user_marker)
    if user_start < 0:
        raise ValueError("InternVL-U formatted prompt has no user turn for image insertion")
    insert_at = user_start + len(user_marker)
    return text[:insert_at] + prefix + text[insert_at:]


def _append_generation_token(text: str) -> str:
    text = text.rstrip(" \t\r")
    while text.endswith(_GENERATION_TOKEN):
        text = text[: -len(_GENERATION_TOKEN)].rstrip(" \t\r")
    return text + _GENERATION_TOKEN


def _build_chat_prompt(user_text: str, system_message: str) -> str:
    return (
        f"{_IM_START}system\n{system_message}{_IM_END}\n{_IM_START}user\n{user_text}{_IM_END}\n{_IM_START}assistant\n"
    )


def _normalize_last_user_turn(text: str) -> str:
    user_marker = f"{_IM_START}user\n"
    user_start = text.rfind(user_marker)
    if user_start < 0:
        raise ValueError("InternVL-U formatted prompt has no user turn")
    content_start = user_start + len(user_marker)
    content_end = text.find(_IM_END, content_start)
    if content_end < 0:
        raise ValueError("InternVL-U formatted prompt has an unterminated user turn")
    normalized = _normalize_user_text(text[content_start:content_end])
    return text[:content_start] + normalized + text[content_end:]


def _ensure_system_turn(
    text: str,
    system_message: str,
    *,
    replace_existing: bool,
) -> str:
    system_marker = f"{_IM_START}system\n"
    system_start = text.find(system_marker)
    if system_start < 0:
        return f"{system_marker}{system_message}{_IM_END}\n" + text
    if not replace_existing:
        return text

    content_start = system_start + len(system_marker)
    content_end = text.find(_IM_END, content_start)
    if content_end < 0:
        raise ValueError("InternVL-U formatted prompt has an unterminated system turn")
    return text[:content_start] + system_message + text[content_end:]


def _prepare_prompt(
    prompt: str,
    *,
    num_images: int,
    modalities: tuple[str, ...],
    generation_mode: str | None,
    system_prompt: str | None,
    has_generation_kwargs: bool = False,
    think: bool = False,
) -> str:
    is_image_generation = (
        generation_mode in {"image", "editing"}
        or any(modality in {"image", "img2img"} for modality in modalities)
        # REST /v1/images paths pass only generation-target kwargs
        # (target_h/target_w/vae_generator_seed) without a modalities entry.
        or has_generation_kwargs
    )
    # ``think`` is the single deployment-wide text-then-image knob
    # (internvlu_chat_think.yaml sets it through engine mm_processor_kwargs).
    # It only changes HOW an image is generated: plain understanding requests
    # are unaffected, and an explicit per-request generation mode (the CFG
    # companions pin "image"/"editing") keeps the direct <img>-terminated
    # form.
    is_think = think and is_image_generation and generation_mode is None
    is_editing = generation_mode == "editing" or "img2img" in modalities or (is_image_generation and num_images > 0)

    default_system = (
        IMAGE_EDITING_SYSTEM_MESSAGE
        if is_editing
        else IMAGE_GENERATION_SYSTEM_MESSAGE
        if is_image_generation
        else DEFAULT_SYSTEM_MESSAGE
    )

    if not _is_formatted_chat(prompt):
        if is_image_generation:
            prompt = prompt.rstrip()
            while prompt.endswith(_GENERATION_TOKEN):
                prompt = prompt[: -len(_GENERATION_TOKEN)].rstrip()

        prompt = _prepend_missing_images(prompt, num_images)
        prompt = _normalize_user_text(prompt)
        prompt = _build_chat_prompt(
            prompt,
            system_prompt if system_prompt is not None else default_system,
        )
    else:
        prompt = _prepend_missing_images(prompt, num_images)
        prompt = _normalize_last_user_turn(prompt)
        prompt = _ensure_system_turn(
            prompt,
            system_prompt if system_prompt is not None else default_system,
            replace_existing=system_prompt is not None,
        )

    # Think mode generates CoT text first; the model-local sampler appends
    # the actual <img> after the CoT instead of the prompt carrying it.
    if is_image_generation and not is_think:
        prompt = _append_generation_token(prompt)
    return prompt


class InternVLUDataParser(MultiModalDataParser):
    """Treat the framework's editing-only ``img2img`` key as an image alias."""

    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        mm_data = dict(mm_data)
        if "img2img" in mm_data:
            if "image" in mm_data:
                raise ValueError("Use either 'image' or its 'img2img' alias, not both")
            mm_data["image"] = mm_data.pop("img2img")
        return super().parse_mm_data(mm_data)


class InternVLUProcessingInfo(BaseInternVLProcessingInfo):
    """Fixed-resolution, image-only processing information for InternVL-U."""

    def get_data_parser(self) -> MultiModalDataParser:
        return InternVLUDataParser(expected_hidden_size=self._get_expected_hidden_size())

    def get_hf_processor(self, **kwargs: object) -> InternVLProcessor:  # noqa: ARG002
        config = self.get_hf_config()
        vision_config = config.vision_config
        image_size = int(config.force_image_size or vision_config.image_size)
        patch_size = int(vision_config.patch_size)
        downsample_ratio = float(config.downsample_ratio)
        image_seq_length = int((image_size // patch_size) ** 2 * downsample_ratio**2)

        image_processor = InternVLImageProcessor(
            image_size=image_size,
            min_dynamic_patch=1,
            max_dynamic_patch=1,
            dynamic_image_size=False,
            use_thumbnail=False,
        )
        return InternVLProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=image_processor,
            image_seq_length=image_seq_length,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,  # noqa: ARG002
        mm_counts: Mapping[str, int],  # noqa: ARG002
    ) -> Mapping[str, int]:
        return {"image": 256}


class InternVLUDummyInputsBuilder(BaseInternVLDummyInputsBuilder[InternVLUProcessingInfo]):
    """Profile InternVL-U with the same fixed 448px image contract."""


class InternVLUMultiModalProcessor(BaseInternVLMultiModalProcessor[InternVLUProcessingInfo]):
    """Apply InternVL-U chat semantics before native InternVL processing."""

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        prompt = inputs.prompt
        if not isinstance(prompt, str):
            prompt = self.info.get_tokenizer().decode(prompt, skip_special_tokens=False)

        processor_kwargs = self.info.ctx.get_merged_mm_kwargs(inputs.hf_processor_mm_kwargs)
        modalities_value = processor_kwargs.get("modalities") or ()
        if isinstance(modalities_value, str):
            modalities = (modalities_value,)
        else:
            modalities = tuple(str(item) for item in modalities_value)

        generation_mode_value = processor_kwargs.get("internvlu_generation_mode")
        if generation_mode_value is None:
            generation_mode_value = processor_kwargs.get("generation_mode")
        generation_mode = str(generation_mode_value) if generation_mode_value is not None else None

        system_prompt_value = processor_kwargs.get("system_prompt")
        system_prompt = str(system_prompt_value) if system_prompt_value is not None else None
        num_images = inputs.mm_data_items.get_count("image", strict=False)
        has_generation_kwargs = any(
            processor_kwargs.get(key) is not None for key in ("target_h", "target_w", "vae_generator_seed")
        )

        # ``think`` is deployment-only; per-request kwargs cannot toggle it.
        engine_mm_kwargs = getattr(self.info.ctx.model_config.multimodal_config, "mm_processor_kwargs", None) or {}
        think = bool(engine_mm_kwargs.get("think"))

        prepared_prompt = _prepare_prompt(
            prompt,
            num_images=num_images,
            modalities=modalities,
            generation_mode=generation_mode,
            system_prompt=system_prompt,
            has_generation_kwargs=has_generation_kwargs,
            think=think,
        )
        return super().apply(replace(inputs, prompt=prepared_prompt), timing_ctx)


__all__ = [
    "InternVLUDataParser",
    "InternVLUDummyInputsBuilder",
    "InternVLUMultiModalProcessor",
    "InternVLUProcessingInfo",
]

from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs.data import TextPrompt
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalInputs, MultiModalUUIDDict

from vllm_omni.inputs.data import OmniTokenInputs, token_inputs_omni
from vllm_omni.inputs.preprocess import OmniInputPreprocessor
from vllm_omni.model_executor.stage_input_processors.apertus import merge_image_placeholders

logger = init_logger(__name__)


def is_apertus_model_config(model_config: Any) -> bool:
    model_arch = getattr(model_config, "model_arch", None)
    if isinstance(model_arch, str) and "ApertusForCausalLM" in model_arch:
        return True

    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return False

    if getattr(hf_config, "model_type", None) == "apertus":
        return True

    architectures = getattr(hf_config, "architectures", None) or []
    return any("ApertusForCausalLM" in arch for arch in architectures)


class ApertusOmniInputPreprocessor(OmniInputPreprocessor):
    """Apertus-specialized input preprocessor.

    This adapter converts text+image prompts into token-only prompts by:
    1) Encoding image(s) with EMU3 VQ tokenizer
    2) Replacing image placeholders with visual token strings
    3) Tokenizing merged text for ApertusForCausalLM
    """

    _APERTUS_DEFAULT_VQ_HUB = "BAAI/Emu3-VisionTokenizer"
    _APERTUS_DEFAULT_MIN_PIXELS = 128 * 128
    _APERTUS_DEFAULT_MAX_PIXELS = 1024 * 1024
    _APERTUS_DEFAULT_IMAGE_PLACEHOLDER = "<|image|>"
    _APERTUS_VISUAL_TEMPLATE = "<|visual token {token_id:0>6d}|>"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apertus_vision_encoder_cache: dict[
            tuple[str, int, int, str, torch.dtype],
            tuple[Any, Any],
        ] = {}

    def _is_apertus_text_image_input(self, multi_modal_data: Mapping[str, Any]) -> bool:
        if "image" not in multi_modal_data:
            return False

        unsupported_modalities = [k for k, v in multi_modal_data.items() if k != "image" and v]
        if unsupported_modalities:
            raise ValueError(
                "Apertus Omni adapter currently supports text and image inputs only. "
                f"Unsupported modalities: {unsupported_modalities}"
            )

        return True

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_dtype(value: Any) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value

        if value is None:
            return torch.bfloat16

        value_str = str(value).lower().strip()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(value_str, torch.bfloat16)

    def _get_apertus_vision_components(
        self,
        mm_processor_kwargs: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoModel

        vq_hub = str(
            mm_processor_kwargs.get(
                "apertus_vq_hub",
                mm_processor_kwargs.get("vq_hub", self._APERTUS_DEFAULT_VQ_HUB),
            )
        )
        min_pixels = self._coerce_int(
            mm_processor_kwargs.get(
                "apertus_min_pixels",
                mm_processor_kwargs.get("emu_min_pixels", self._APERTUS_DEFAULT_MIN_PIXELS),
            ),
            default=self._APERTUS_DEFAULT_MIN_PIXELS,
        )
        max_pixels = self._coerce_int(
            mm_processor_kwargs.get(
                "apertus_max_pixels",
                mm_processor_kwargs.get("emu_max_pixels", self._APERTUS_DEFAULT_MAX_PIXELS),
            ),
            default=self._APERTUS_DEFAULT_MAX_PIXELS,
        )
        vision_device = str(mm_processor_kwargs.get("apertus_vision_tokenizer_device", "cpu"))
        vision_dtype = self._coerce_dtype(mm_processor_kwargs.get("apertus_vision_tokenizer_dtype"))
        if vision_device == "cpu" and vision_dtype in (torch.float16, torch.bfloat16):
            vision_dtype = torch.float32
        trust_remote_code = bool(mm_processor_kwargs.get("apertus_vq_trust_remote_code", True))

        cache_key = (vq_hub, min_pixels, max_pixels, vision_device, vision_dtype)
        if cache_key in self._apertus_vision_encoder_cache:
            return self._apertus_vision_encoder_cache[cache_key]

        image_processor = AutoImageProcessor.from_pretrained(
            vq_hub,
            trust_remote_code=trust_remote_code,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        vision_tokenizer = AutoModel.from_pretrained(
            vq_hub,
            trust_remote_code=trust_remote_code,
            torch_dtype=vision_dtype,
        ).eval()
        vision_tokenizer = vision_tokenizer.to(vision_device)

        self._apertus_vision_encoder_cache[cache_key] = (image_processor, vision_tokenizer)
        return image_processor, vision_tokenizer

    def _apertus_special_token(self, attr_name: str, fallback: str) -> str:
        token = getattr(self.tokenizer, attr_name, None)
        return token if isinstance(token, str) and token else fallback

    def _build_apertus_image_prompt(self, image_tokens: torch.Tensor) -> str:
        if image_tokens.ndim != 2:
            raise ValueError(f"Apertus image tokens must be 2D, got shape {tuple(image_tokens.shape)}")

        h, w = image_tokens.shape
        rows = [
            "".join(
                self._APERTUS_VISUAL_TEMPLATE.format(token_id=int(token_id))
                for token_id in row
            )
            for row in image_tokens.detach().to("cpu").tolist()
        ]
        imgstr = self._apertus_special_token("eol_token", "<|img_end_of_row|>").join(rows)

        boi_token = self._apertus_special_token("boi_token", "<|img_start|>")
        img_token = self._apertus_special_token("img_token", "<|img_token_start|>")
        eol_token = self._apertus_special_token("eol_token", "<|img_end_of_row|>")
        eof_token = self._apertus_special_token("eof_token", "<|img_end_of_frame|>")
        eoi_token = self._apertus_special_token("eoi_token", "<|img_end|>")

        return f"{boi_token}{h}*{w}{img_token}{imgstr}{eol_token}{eof_token}{eoi_token}"

    def _resolve_apertus_image_placeholder(
        self,
        prompt_text: str,
        mm_processor_kwargs: Mapping[str, Any],
    ) -> str:
        configured_placeholder = mm_processor_kwargs.get("apertus_image_placeholder")
        if isinstance(configured_placeholder, str) and configured_placeholder:
            return configured_placeholder

        tokenizer_placeholder = getattr(self.tokenizer, "image_token", None)
        if isinstance(tokenizer_placeholder, str) and tokenizer_placeholder in prompt_text:
            return tokenizer_placeholder

        for candidate in (self._APERTUS_DEFAULT_IMAGE_PLACEHOLDER, "<image>"):
            if candidate in prompt_text:
                return candidate

        if isinstance(tokenizer_placeholder, str) and tokenizer_placeholder:
            return tokenizer_placeholder

        return self._APERTUS_DEFAULT_IMAGE_PLACEHOLDER

    def _normalize_apertus_images(self, image_data: Any) -> list[Any]:
        if image_data is None:
            return []

        images = image_data if isinstance(image_data, list) else [image_data]
        if not images:
            return []

        try:
            from PIL.Image import Image as PILImage
        except Exception:
            PILImage = None

        if PILImage is not None:
            for image in images:
                if not isinstance(image, PILImage):
                    raise TypeError(
                        "Apertus text+image adapter expects PIL images in multi_modal_data['image']."
                    )

        return images

    def _encode_apertus_images_to_strings(
        self,
        images: list[Any],
        mm_processor_kwargs: Mapping[str, Any],
    ) -> list[str]:
        if not images:
            return []

        image_processor, vision_tokenizer = self._get_apertus_vision_components(mm_processor_kwargs)
        vision_device = next(vision_tokenizer.parameters()).device
        vision_dtype = next(vision_tokenizer.parameters()).dtype
        image_prompts: list[str] = []

        for image in images:
            encoded = image_processor(image, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device=vision_device, dtype=vision_dtype)
            image_sizes = encoded.get("image_sizes")
            if isinstance(image_sizes, torch.Tensor):
                image_sizes = image_sizes.to(device=vision_device)
            elif image_sizes is not None:
                image_sizes = torch.tensor(image_sizes, device=vision_device)

            with torch.inference_mode():
                image_tokens = None
                if image_sizes is not None:
                    try:
                        image_tokens = vision_tokenizer.encode(
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                        )
                    except TypeError:
                        image_tokens = None
                if image_tokens is None:
                    try:
                        image_tokens = vision_tokenizer.encode(pixel_values)
                    except TypeError:
                        image_tokens = vision_tokenizer.encode(pixel_values=pixel_values)

            if image_tokens is None:
                raise ValueError("Apertus image encoding produced no image tokens.")

            if isinstance(image_tokens, torch.Tensor):
                if image_tokens.ndim == 4:
                    image_tokens = image_tokens.squeeze(0)
                if image_tokens.ndim == 3:
                    image_token_grid = image_tokens[0]
                elif image_tokens.ndim == 2:
                    image_token_grid = image_tokens
                else:
                    raise ValueError(
                        "Unexpected tensor rank from Apertus image encoding: "
                        f"{tuple(image_tokens.shape)}"
                    )
            else:
                if not image_tokens:
                    raise ValueError("Apertus image encoding produced an empty token sequence.")
                image_token_grid = image_tokens[0]

            if not isinstance(image_token_grid, torch.Tensor):
                image_token_grid = torch.tensor(image_token_grid)
            if image_token_grid.ndim == 3:
                image_token_grid = image_token_grid.squeeze(0)

            image_prompts.append(self._build_apertus_image_prompt(image_token_grid))

        return image_prompts

    def _process_apertus_text_with_images(
        self,
        prompt_text: str,
        multi_modal_data: Mapping[str, Any],
        mm_processor_kwargs: Mapping[str, Any],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> OmniTokenInputs:
        images = self._normalize_apertus_images(multi_modal_data.get("image"))
        image_prompts = self._encode_apertus_images_to_strings(images, mm_processor_kwargs)
        placeholder = self._resolve_apertus_image_placeholder(prompt_text, mm_processor_kwargs)
        merged_prompt = merge_image_placeholders(
            prompt_text,
            image_prompts=image_prompts,
            image_placeholder=placeholder,
        )
        prompt_token_ids = self._tokenize_prompt(
            merged_prompt,
            tokenization_kwargs=tokenization_kwargs,
        )
        return token_inputs_omni(prompt_token_ids)

    def _process_text(
        self,
        parsed_content: TextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> OmniTokenInputs | MultiModalInputs:
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            if isinstance(multi_modal_data, Mapping) and self._is_apertus_text_image_input(multi_modal_data):
                prompt_text = parsed_content["prompt"]
                mm_processor_kwargs = parsed_content.get("mm_processor_kwargs") or {}
                inputs = self._process_apertus_text_with_images(
                    prompt_text,
                    multi_modal_data=multi_modal_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                    tokenization_kwargs=tokenization_kwargs,
                )
                prompt_embeds = parsed_content.get("prompt_embeds")
                if prompt_embeds is not None:
                    inputs["prompt_embeds"] = prompt_embeds
                additional_information = parsed_content.get("additional_information")
                if additional_information is not None:
                    inputs["additional_information"] = additional_information
                if cache_salt := parsed_content.get("cache_salt"):
                    inputs["cache_salt"] = cache_salt
                return inputs

        return super()._process_text(
            parsed_content,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

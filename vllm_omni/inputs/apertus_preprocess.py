from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image
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
    1) Encoding image(s) with EMU3.5 IBQ vision tokenizer
    2) Replacing image placeholders with visual token strings
    3) Tokenizing merged text for ApertusForCausalLM
    """

    _APERTUS_DEFAULT_VQ_HUB = "BAAI/Emu3.5-VisionTokenizer"
    _APERTUS_DEFAULT_MIN_PIXELS = 512 * 512
    _APERTUS_DEFAULT_MAX_PIXELS = 1024 * 1024
    _APERTUS_DEFAULT_IMAGE_PLACEHOLDER = "<|image|>"
    _APERTUS_VISUAL_TEMPLATE = "<|visual token {token_id}|>"
    _APERTUS_EMU35_DS_FACTOR = 16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apertus_vision_encoder_cache: dict[tuple[str, str, torch.dtype, bool], Any] = {}

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

    @staticmethod
    def _smart_resize(image: Image.Image, area: int, ds_factor: int) -> Image.Image:
        width, height = image.size
        aspect_ratio = width / height
        new_height = int((area / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        new_height = ((new_height + ds_factor // 2) // ds_factor) * ds_factor
        new_width = ((new_width + ds_factor // 2) // ds_factor) * ds_factor
        return image.resize((new_width, new_height), Image.BICUBIC)

    @staticmethod
    def _extract_emu35_token_grid(
        encode_out: Any,
        token_height: int,
        token_width: int,
    ) -> torch.Tensor:
        def _unwrap_token_payload(payload: Any) -> Any:
            token = payload
            if isinstance(token, tuple):
                token = token[2] if len(token) >= 3 else token[-1]

            while isinstance(token, (list, tuple)):
                if not token:
                    raise ValueError("Apertus emu3.5 encoding produced an empty token sequence.")
                non_none = [item for item in token if item is not None]
                if not non_none:
                    raise ValueError("Apertus emu3.5 encoding produced only None token entries.")
                token = non_none[-1]

            if isinstance(token, Mapping):
                for key in ("token_ids", "indices", "codes", "tokens"):
                    value = token.get(key)
                    if value is not None:
                        return _unwrap_token_payload(value)
                non_none_values = [value for value in token.values() if value is not None]
                if not non_none_values:
                    raise ValueError("Apertus emu3.5 encoding produced an empty token mapping.")
                token = non_none_values[-1]

            return token

        token = _unwrap_token_payload(encode_out)

        if not isinstance(token, torch.Tensor):
            token = torch.tensor(token)

        # Drop leading dims (e.g. codebook/batch) until we have a 2D grid or 1D flattened tokens.
        while token.ndim > 2:
            token = token[0] if token.shape[0] == 1 else token[-1]

        if token.ndim == 1:
            expected = token_height * token_width
            if token.numel() != expected:
                raise ValueError(
                    "Apertus emu3.5 token length mismatch: "
                    f"got {token.numel()}, expected {expected}."
                )
            token = token.view(token_height, token_width)
        elif token.ndim == 2:
            if token.shape == (token_height, token_width):
                pass
            elif token.numel() == token_height * token_width:
                token = token.reshape(token_height, token_width)
            else:
                raise ValueError(
                    "Apertus emu3.5 token grid shape mismatch: "
                    f"got {tuple(token.shape)}, expected {(token_height, token_width)}."
                )
        else:
            raise ValueError(f"Unexpected emu3.5 token rank: {token.ndim}.")

        return token.to(dtype=torch.int64)

    def _get_apertus_vision_components(
        self,
        mm_processor_kwargs: Mapping[str, Any],
    ) -> Any:
        from transformers import AutoModel

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

        del min_pixels, max_pixels
        cache_key = (vq_hub, vision_device, vision_dtype, trust_remote_code)
        if cache_key in self._apertus_vision_encoder_cache:
            return self._apertus_vision_encoder_cache[cache_key]

        vision_tokenizer = AutoModel.from_pretrained(
            vq_hub,
            trust_remote_code=trust_remote_code,
            torch_dtype=vision_dtype,
        ).eval()
        vision_tokenizer = vision_tokenizer.to(vision_device)

        self._apertus_vision_encoder_cache[cache_key] = vision_tokenizer
        return vision_tokenizer

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
        eol_token = self._apertus_special_token("eol_token", "<|extra_200|>")
        imgstr = eol_token.join(rows)

        boi_token = self._apertus_special_token("boi_token", "<|image start|>")
        img_token = self._apertus_special_token("img_token", "<|image token|>")
        eoi_token = self._apertus_special_token("eoi_token", "<|image end|>")

        # Emu3.5 format: no trailing EOL, no EOF token.
        return f"{boi_token}{h}*{w}{img_token}{imgstr}{eoi_token}"

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

        for idx, image in enumerate(images):
            if not isinstance(image, Image.Image):
                raise TypeError(
                    "Apertus text+image adapter expects PIL images in multi_modal_data['image']."
                )
            images[idx] = image.convert("RGB")

        return images

    def _encode_apertus_images_to_strings(
        self,
        images: list[Any],
        mm_processor_kwargs: Mapping[str, Any],
    ) -> list[str]:
        if not images:
            return []

        vision_tokenizer = self._get_apertus_vision_components(mm_processor_kwargs)
        vision_device = next(vision_tokenizer.parameters()).device
        vision_dtype = next(vision_tokenizer.parameters()).dtype
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
        image_prompts: list[str] = []

        for image in images:
            width, height = image.size
            current_area = width * height
            target_area = max(min(max_pixels, current_area), min_pixels)
            resized_image = self._smart_resize(image, target_area, self._APERTUS_EMU35_DS_FACTOR)
            resized_w, resized_h = resized_image.size
            image_tensor = torch.tensor((np.array(resized_image) / 127.5 - 1.0)).to(
                device=vision_device,
                dtype=vision_dtype,
            ).permute(2, 0, 1)
            with torch.inference_mode():
                try:
                    encode_out = vision_tokenizer.encode(image_tensor[None])
                except TypeError:
                    try:
                        encode_out = vision_tokenizer.encode(pixel_values=image_tensor[None])
                    except TypeError:
                        encode_out = vision_tokenizer.encode(images=image_tensor[None])

            token_h = resized_h // self._APERTUS_EMU35_DS_FACTOR
            token_w = resized_w // self._APERTUS_EMU35_DS_FACTOR
            image_token_grid = self._extract_emu35_token_grid(encode_out, token_h, token_w)

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
        effective_tokenization_kwargs = dict(tokenization_kwargs or {})
        effective_tokenization_kwargs.setdefault("add_special_tokens", False)
        prompt_token_ids = self._tokenize_prompt(
            merged_prompt,
            tokenization_kwargs=effective_tokenization_kwargs,
        )

        logger.info(
            "Apertus Omni adapter merged %d image(s) into %d tokens.",
            len(image_prompts),
            len(prompt_token_ids),
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

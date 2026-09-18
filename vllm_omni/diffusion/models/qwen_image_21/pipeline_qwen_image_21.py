# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from diffusers' pipeline_qwenimage21.py (Qwen-Image 2.1 T2I + image-conditioned generation).

import copy
import dataclasses
import json
import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage21 import (
    DistributedAutoencoderKLQwenImage21,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.lora.loader import QwenImageLoraLoaderMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.model_metadata import QWEN_IMAGE_21_MAX_INPUT_IMAGES
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    calculate_shift,
    retrieve_timesteps,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import (
    calculate_dimensions,
    retrieve_latents,
)
from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import QwenImage21CFGParallelMixin
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Transformer2DModel,
)
from vllm_omni.diffusion.models.utils import create_transformers_model
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.prompt_utils import (
    validate_prompt_sequence_lengths,
)
from vllm_omni.diffusion.utils.size_utils import (
    normalize_min_aligned_size,
)
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.quantization import resolve_component_quant_config
from vllm_omni.quantization.component_config import resolve_encoder_quant_config

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)

# Target area (in pixels) used to derive the output size from a condition
# image's aspect ratio and to resize condition images before encoding.
OUTPUT_RESOLUTION = 1024

# Re-export the shared metadata value locally so this pipeline keeps a nearby,
# descriptive constant for validation without becoming the source of truth.
MAX_QWEN_IMAGE_21_INPUT_IMAGES = QWEN_IMAGE_21_MAX_INPUT_IMAGES


def _read_vae_scale_factor(od_config: OmniDiffusionConfig) -> int:
    """Read the VAE spatial compression ratio from the checkpoint's vae/config.json."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
    # The architecture compresses 2x spatially per downsampling stage (one per
    # `temperal_downsample` entry), e.g. 16x for Qwen-Image 2.1. The shipped
    # `scale_factor_spatial` (8) under-reports it, so prefer the
    # architecture-derived value.
    if "temperal_downsample" in vae_config:
        return 2 ** len(vae_config["temperal_downsample"])
    if vae_config.get("scale_factor_spatial"):
        return int(vae_config["scale_factor_spatial"])
    return 16


def _resolve_text_encoder_quant_config(
    quant_config: "QuantizationConfig | None",
) -> "QuantizationConfig | None":
    """Resolve the text-encoder slice of a global or per-component quant config.

    Pre-quantized formats (modelopt, svdquant, ...) require serialized scale
    tensors the BF16 text-encoder checkpoint does not ship, so they are stripped
    and the encoder stays in checkpoint precision.
    """
    resolved = resolve_component_quant_config(quant_config, "text_encoder")
    return resolve_encoder_quant_config(resolved)


# Subtrees of the HF Qwen3-VL text encoder that must stay in checkpoint
# precision: the vision tower (condition-image encoding is numerically
# sensitive) and the LM head (unused — the pipeline reads hidden states).
_TEXT_ENCODER_QUANT_EXCLUDED_PREFIXES = ("model.visual", "lm_head")


def _exclude_text_encoder_subtrees_from_quant(
    quant_config: "QuantizationConfig | None",
) -> "QuantizationConfig | None":
    """Keep the vision tower and LM head in checkpoint precision.

    Uses the config's standard ``ignored_layers`` mechanism, with substring
    matching so the prefixes cover whole subtrees (same convention as the
    DiT's ``_enable_pattern_ignored_layers``).
    """
    if quant_config is None:
        return None
    if not hasattr(quant_config, "ignored_layers"):
        logger.warning(
            "Quantization config %s has no ignored_layers; the Qwen3-VL vision tower "
            "and LM head cannot be excluded from quantization.",
            type(quant_config).__name__,
        )
        return quant_config
    config = copy.copy(quant_config)
    config.ignored_layers = [*config.ignored_layers, *_TEXT_ENCODER_QUANT_EXCLUDED_PREFIXES]
    if hasattr(config, "ignored_layers_match_mode"):
        config.ignored_layers_match_mode = "substring"
    return config


def get_qwen_image_21_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for QwenImage21Pipeline.

    Reads condition images from ``request.prompt["multi_modal_data"]["image"]``
    (single image or list), resizes them once for both the vision-language
    prompt encoder and the VAE, and stores the results under
    ``prompt["additional_information"]``. Pure text-to-image requests (no
    image) pass through unchanged.
    """
    vae_scale_factor = _read_vae_scale_factor(od_config)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    # The Qwen-Image 2.1 VAE is RGBA (in/out_channels=4): condition images get an
    # opaque alpha channel unless the input already carries one.
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def _to_pil(image: Any) -> PIL.Image.Image:
        if isinstance(image, PIL.Image.Image):
            return image
        if isinstance(image, str):
            return PIL.Image.open(image)
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu()
            if arr.dim() == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.permute(1, 2, 0)
            arr = (arr * 255).round().to(torch.uint8) if arr.dtype.is_floating_point else arr
            arr = arr.numpy()
            if arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            return PIL.Image.fromarray(arr)
        if isinstance(image, np.ndarray):
            return PIL.Image.fromarray(image)
        raise TypeError(f"Unsupported condition image type: {type(image)}")

    def pre_process_func(
        request: OmniDiffusionRequest,
    ):
        """Pre-process requests for QwenImage21Pipeline."""
        request.allow_mixed_step_phases = False
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        if "additional_information" not in prompt:
            prompt["additional_information"] = {}

        if raw_image is None or (isinstance(raw_image, list) and len(raw_image) == 0):
            request.batch_compatibility_key = ("qwen_image_21", ())
            request.prompt = prompt
            return request

        if not isinstance(raw_image, list):
            raw_image = [raw_image]
        if len(raw_image) > MAX_QWEN_IMAGE_21_INPUT_IMAGES:
            raise ValueError(
                f"Received {len(raw_image)} input images. "
                f"At most {MAX_QWEN_IMAGE_21_INPUT_IMAGES} images are supported by this model."
            )
        images = [_to_pil(im) for im in raw_image]

        # One resize feeds both the text encoder and the VAE.
        input_image_sizes = []
        prompt_images = []
        vae_images = []
        for img in images:
            image_width, image_height = img.size
            input_width, input_height = calculate_dimensions(
                OUTPUT_RESOLUTION * OUTPUT_RESOLUTION, image_width / image_height
            )
            input_image_sizes.append((input_width, input_height))
            prompt_images.append(image_processor.resize(img, height=input_height, width=input_width))
            vae_images.append(
                vae_image_processor.preprocess(img.convert("RGBA"), height=input_height, width=input_width).unsqueeze(2)
            )

        # The generated image derives its aspect ratio from the last condition image.
        last_width, last_height = images[-1].size
        calculated_width, calculated_height = calculate_dimensions(
            OUTPUT_RESOLUTION * OUTPUT_RESOLUTION, last_width / last_height
        )
        height = request.sampling_params.height or calculated_height
        width = request.sampling_params.width or calculated_width
        height, width = normalize_min_aligned_size(height, width, vae_scale_factor * 2)
        request.sampling_params.height = height
        request.sampling_params.width = width

        prompt["additional_information"]["prompt_image"] = prompt_images
        prompt["additional_information"]["vae_images"] = vae_images
        prompt["additional_information"]["input_image_sizes"] = input_image_sizes
        prompt["additional_information"]["calculated_height"] = calculated_height
        prompt["additional_information"]["calculated_width"] = calculated_width
        request.batch_compatibility_key = ("qwen_image_21", tuple(input_image_sizes))
        request.prompt = prompt
        return request

    return pre_process_func


def get_qwen_image_21_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-processing function for QwenImage21Pipeline."""
    vae_scale_factor = _read_vae_scale_factor(od_config)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(
        images: torch.Tensor | dict[str, Any],
    ):
        if isinstance(images, dict) and isinstance(images.get("payload"), dict):
            payload = dict(images["payload"])
            image_payload = payload.get("image")
            if image_payload is None:
                raise ValueError("Qwen-Image 2.1 postprocess expected payload['image'] in output envelope.")
            payload["image"] = image_processor.postprocess(image_payload)
            metadata = images.get("metadata") or {}
            return {
                "payload": payload,
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        return image_processor.postprocess(images)

    return post_process_func


class QwenImage21Pipeline(
    nn.Module,
    QwenImage21CFGParallelMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
    QwenImageLoraLoaderMixin,
):
    """Text-to-image and image-conditioned generation with Qwen-Image 2.1.

    Prompt and condition images are encoded together by a Qwen3-VL model, so a
    condition image occupies the vision slots the encoder reserved for it and
    the transformer sees one interleaved text/image sequence.
    """

    supports_request_batch = True
    supports_step_execution: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.device = get_local_device()
        model = od_config.model
        # Check if model is a local path
        local_files_only = os.path.isdir(model)

        # Guard against transformers v5 multi-worker race on partial subfolder
        # shard sets (see pipeline_qwen_image_edit_plus, Buildkite #1043).
        qwen_subfolders = ["scheduler", "text_encoder", "vae", "processor"]
        prefetch_subfolders(
            model,
            qwen_subfolders,
            local_files_only=local_files_only,
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        # Unlike the 2.0 T2I pipeline, the vision tower is required: condition
        # images are encoded jointly with the prompt by the VLM.
        text_encoder_quant_config = _resolve_text_encoder_quant_config(od_config.quantization_config)
        if text_encoder_quant_config is None:
            self.text_encoder = from_pretrained_with_prefetch(
                Qwen3VLForConditionalGeneration.from_pretrained,
                model,
                subfolder="text_encoder",
                prefetch_list=qwen_subfolders,
                local_files_only=local_files_only,
            ).to(self.device)
        else:
            # Quantized path: build the HF model on meta, swap its nn.Linear
            # modules for vLLM quantizable linears, and stream the weights
            # through the pipeline loader (same integration as Z-Image). The
            # encoder-scoped quant config rides on a copied od_config so the
            # shared create_transformers_model signature stays untouched.
            text_encoder_config = AutoConfig.from_pretrained(
                model, subfolder="text_encoder", local_files_only=local_files_only
            )
            encoder_od_config = dataclasses.replace(
                od_config,
                quantization_config=_exclude_text_encoder_subtrees_from_quant(text_encoder_quant_config),
            )
            self.text_encoder = create_transformers_model(
                AutoModelForImageTextToText,
                encoder_od_config,
                hf_config=text_encoder_config,
            ).to(self.device)
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=model,
                    subfolder="text_encoder",
                    revision=None,
                    prefix="text_encoder.",
                )
            )
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLQwenImage21.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=qwen_subfolders,
            local_files_only=local_files_only,
        ).to(self.device)
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImage21Transformer2DModel)
        self.transformer = QwenImage21Transformer2DModel(
            od_config=od_config,
            quant_config=resolve_component_quant_config(od_config.quantization_config, "transformer"),
            **transformer_kwargs,
        )
        self.processor = from_pretrained_with_prefetch(
            Qwen3VLProcessor.from_pretrained,
            model,
            subfolder="processor",
            prefetch_list=qwen_subfolders,
            local_files_only=local_files_only,
        )

        self.stage = None

        # The VAE compresses 16x spatially and the transformer consumes latents
        # unpatched, so one token covers a 16x16 pixel tile.
        self.vae_scale_factor = 16
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 64

        self.sys_prompt = "Comprehend and analyze the provided prompt."
        # The prompt is built as a raw template string and passed straight to
        # `self.processor(text=..., images=...)`, rather than going through
        # `apply_chat_template`: the two tokenize differently and the checkpoint
        # expects this one. The "<image1>..." vision prefix only appears in
        # the image-conditioned template.
        self.prompt_template_t2i = (
            f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{{}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prompt_template_ti2i = (
            f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        # Number of leading system-role tokens to drop from the hidden states.
        # Derived from the tokenized system message rather than hardcoded, so it
        # tracks the processor's template.
        sys_message = [{"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]}]
        sys_tokens = self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
        self._drop_idx = len(sys_tokens[0])
        self._img_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]
        self._max_length = 8192

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if prompt is None:
            raise ValueError("Provide `prompt`. Cannot leave `prompt` undefined.")
        elif not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if max_sequence_length is not None and max_sequence_length > self._max_length:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than {self._max_length} but is {max_sequence_length}"
            )

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    @staticmethod
    def _downsample_image_pad_tokens(
        hidden_states_list,
        image_pad_mask_list,
        expected_slots_list,
        image_sizes_list,
    ):
        """Reconcile each contiguous ``<|image_pad|>`` run with one slot per 2x2 latent group.

        The transformer's joint sequence reserves one vision-language token slot for every 2x2 group
        of latent tokens and expands each slot 4x when substituting the latent embeddings, so a
        condition image resized to ``W x H`` pixels needs exactly ``(H // 16) * (W // 16) // 4``
        slots in ``img_mask``.

        A standard Qwen3-VL processor (patch=16, merge=2) already emits exactly that many tokens per
        image, so the run is kept unchanged (identity). A processor without spatial merging (merge=1)
        emits 4x as many; those runs are folded by keeping the first token of every 4 consecutive
        tokens. Any other count raises: if an official checkpoint's processor tokenizes differently,
        this is the place to adjust.

        Args:
            hidden_states_list: Per-sample hidden states (after `drop_idx`, before padding).
            image_pad_mask_list: Per-sample bool masks marking the processor's image tokens.
            expected_slots_list: Per-sample list of expected slot counts, one per condition image.
            image_sizes_list: Per-sample list of `(width, height)` pixel sizes, for error messages.

        Returns:
            `tuple` of `(out_hs, out_mask)`, same per-sample list structure as the inputs.
        """
        out_hs, out_mask = [], []
        for hidden_state, pad_mask, expected_slots, image_sizes in zip(
            hidden_states_list, image_pad_mask_list, expected_slots_list, image_sizes_list
        ):
            pad_indices = torch.where(pad_mask)[0]
            runs: list[tuple[int, int]] = []  # contiguous image-token runs, inclusive (start, end)
            if len(pad_indices) > 0:
                run_start = pad_indices[0].item()
                prev = run_start
                for idx in pad_indices[1:].tolist():
                    if idx != prev + 1:
                        runs.append((run_start, prev))
                        run_start = idx
                    prev = idx
                runs.append((run_start, prev))

            if len(runs) != len(expected_slots):
                raise ValueError(
                    f"Prompt encodes {len(runs)} condition-image token run(s) but "
                    f"{len(expected_slots)} condition image(s) were provided (sizes: {image_sizes})."
                )
            if not runs:
                out_hs.append(hidden_state)
                out_mask.append(pad_mask)
                continue

            keep = ~pad_mask
            for (run_start, run_end), expected, size in zip(runs, expected_slots, image_sizes):
                run_len = run_end - run_start + 1
                if run_len == expected:
                    # Standard Qwen3-VL processor (patch=16, merge=2): already one
                    # token per 2x2 latent group.
                    keep[run_start : run_end + 1] = True
                elif run_len == expected * 4:
                    # Processor without spatial merging (merge=1): fold every 4
                    # consecutive tokens into the first one.
                    keep[run_start : run_end + 1 : 4] = True
                else:
                    raise ValueError(
                        f"Condition image resized to {size[0]}x{size[1]} produced {run_len} image tokens from the "
                        f"processor, but the transformer expects {expected} slots (one per 2x2 latent group) or "
                        f"{expected * 4} unmerged tokens. If the official checkpoint's processor tokenizes "
                        "differently, adjust `_downsample_image_pad_tokens` accordingly."
                    )
            out_hs.append(hidden_state[keep])
            out_mask.append(pad_mask[keep])
        return out_hs, out_mask

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        images_per_prompt: list[list[PIL.Image.Image] | None] | None = None,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
        """Encode prompts (and optional per-prompt condition images) with the Qwen3-VL encoder.

        Returns:
            `tuple` of `(prompt_embeds, encoder_attention_mask, image_pad_mask)`, all padded to the
            batch's maximum (post-downsample) sequence length. `image_pad_mask` is `True` at the
            vision-language image slots.
        """
        dtype = self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [" " if not p else p for p in prompt]

        has_images = images_per_prompt is not None and any(images_per_prompt)
        if not has_images:
            txt = [self.prompt_template_t2i.format(e) for e in prompt]
        else:
            if images_per_prompt is None or len(images_per_prompt) != len(prompt):
                raise ValueError("`images_per_prompt` must align with `prompt`.")
            txt = []
            condition_pil_per_prompt: list[list[PIL.Image.Image]] = []
            for t, images in zip(prompt, images_per_prompt):
                images = [
                    img if isinstance(img, PIL.Image.Image) else PIL.Image.fromarray(img) for img in (images or [])
                ]
                condition_pil_per_prompt.append(images)
                if not images:
                    txt.append(self.prompt_template_t2i.format(t))
                    continue
                n_imgs = len(images)
                replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
                for i in range(2, n_imgs + 1):
                    replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
                template = self.prompt_template_ti2i.replace(
                    "<image1><|vision_start|><|image_pad|><|vision_end|>", replace
                )
                txt.append(template.format(t))
            condition_pil_list = []
            for images in condition_pil_per_prompt:
                for img in images:
                    if img.mode == "RGBA":
                        white = PIL.Image.new("RGB", img.size, (255, 255, 255))
                        white.paste(img, mask=img.getchannel("A"))
                        img = white
                    condition_pil_list.append(img)

        # Validate only the user prompt contribution against the text budget;
        # image placeholder expansion happens later inside the processor.
        template_tokens = self.processor.tokenizer(
            [self.prompt_template_t2i.format("")],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        txt_tokens = self.processor.tokenizer(
            txt,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        validate_prompt_sequence_lengths(
            txt_tokens.attention_mask,
            max_sequence_length=max_sequence_length or self._max_length,
            supported_max_sequence_length=self._max_length,
            prompt_name=prompt_name,
            baseline_attention_mask=template_tokens.attention_mask,
            error_context="after applying the Qwen-Image 2.1 prompt template",
        )

        processor_kwargs: dict[str, Any] = {
            "text": txt,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if has_images:
            processor_kwargs["images"] = condition_pil_list
        model_inputs = self.processor(**processor_kwargs).to(self.device)

        forward_kwargs: dict[str, Any] = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
        }
        if has_images and hasattr(model_inputs, "pixel_values"):
            forward_kwargs.update(
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
            )
        # transformers>=5 Qwen3-VL computes multimodal RoPE from the token type
        # ids the processor emits alongside input_ids.
        if getattr(model_inputs, "mm_token_type_ids", None) is not None:
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        text_model = getattr(self.text_encoder.model, "language_model", self.text_encoder.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            outputs = self.text_encoder(**forward_kwargs)
        finally:
            handle.remove()
        hidden_states = outputs.hidden_states[-1]

        split_hidden_states = list(self._extract_masked_hidden(hidden_states, model_inputs.attention_mask))
        split_hidden_states = [e[self._drop_idx :] for e in split_hidden_states]

        image_pad_mask = [
            (sample_ids[sample_mask.bool()] == self._img_token_id)
            for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
        ]
        image_pad_mask = [e[self._drop_idx :] for e in image_pad_mask]

        if has_images:
            # Expected slot counts: one VLM token slot per 2x2 group of latent
            # tokens, i.e. (H // vae_scale_factor) * (W // vae_scale_factor) // 4
            # for a condition image resized to W x H pixels.
            expected_slots = [
                [(img.height // self.vae_scale_factor) * (img.width // self.vae_scale_factor) // 4 for img in images]
                for images in condition_pil_per_prompt
            ]
            image_sizes = [[img.size for img in images] for images in condition_pil_per_prompt]
            split_hidden_states, image_pad_mask = self._downsample_image_pad_tokens(
                split_hidden_states, image_pad_mask, expected_slots, image_sizes
            )

        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        image_pad_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), dtype=torch.bool)]) for u in image_pad_mask]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask, image_pad_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        images_per_prompt: list[list[PIL.Image.Image] | None] | None = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
        r"""

        Args:
            prompt (`str` or `list[str]`):
                prompt to be encoded
            images_per_prompt (`list[list[PIL.Image.Image]]`, *optional*):
                Per-prompt condition images, encoded jointly with the prompt.
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds, prompt_embeds_mask, image_pad_mask = self._get_qwen_prompt_embeds(
            prompt,
            images_per_prompt,
            max_sequence_length=max_sequence_length,
            prompt_name=prompt_name,
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        image_pad_mask = image_pad_mask.repeat(1, num_images_per_prompt)
        image_pad_mask = image_pad_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask, image_pad_mask

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        # 2.1 consumes latents unpatched, so packing is a plain spatial flatten.
        return latents.view(batch_size, num_channels_latents, height * width).transpose(1, 2)

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, _, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.transpose(1, 2).reshape(batch_size, channels, 1, height, width)
        return latents

    @staticmethod
    def _append_target_slots(image_pad_mask: torch.Tensor, num_target_tokens: int) -> torch.Tensor:
        """Append one True slot per 2x2 group of target-image tokens to the VLM image-slot mask."""
        target_slots = torch.ones(
            [image_pad_mask.shape[0], num_target_tokens // 4],
            dtype=image_pad_mask.dtype,
            device=image_pad_mask.device,
        )
        return torch.cat([image_pad_mask, target_slots], dim=1)

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def prepare_latents(
        self,
        vae_images: list[torch.Tensor] | None,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        image_latents = None
        if vae_images is not None:
            all_image_latents = []
            for image in vae_images:
                image = image.to(device=device, dtype=dtype)
                encoded = image if image.shape[1] == self.latent_channels else self._encode_vae_image(image, generator)
                if batch_size > encoded.shape[0]:
                    if batch_size % encoded.shape[0] != 0:
                        raise ValueError(
                            f"Cannot duplicate `image` of batch size {encoded.shape[0]} to {batch_size} text prompts."
                        )
                    encoded = encoded.repeat_interleave(batch_size // encoded.shape[0], dim=0)
                image_latent_height, image_latent_width = encoded.shape[3:]
                all_image_latents.append(
                    self._pack_latents(
                        encoded, batch_size, num_channels_latents, image_latent_height, image_latent_width
                    )
                )
            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            shape = (batch_size, 1, num_channels_latents, height, width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device=self.device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def _extract_prompts(self, prompts):
        """Extract prompt and negative_prompt from OmniPromptType list."""
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts] or None
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in prompts):
            negative_prompt = None
        elif prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
        else:
            negative_prompt = None
        return prompt, negative_prompt

    @staticmethod
    def _extract_request_images(prompts) -> list[dict[str, Any]]:
        """Pull per-request condition-image data staged by the pre-process func."""
        per_request = []
        for p in prompts:
            info = p.get("additional_information", {}) if isinstance(p, dict) else {}
            per_request.append(
                {
                    "prompt_images": info.get("prompt_image"),
                    "vae_images": info.get("vae_images"),
                    "input_image_sizes": info.get("input_image_sizes"),
                }
            )
        return per_request

    def _prepare_generation_context(
        self,
        *,
        prompt,
        negative_prompt,
        height,
        width,
        num_inference_steps,
        sigmas,
        num_images_per_prompt,
        generator,
        true_cfg_scale,
        max_sequence_length,
        per_request_images: list[dict[str, Any]],
        latents=None,
        attention_kwargs=None,
    ):
        """Shared preparation logic for forward() and prepare_encode().

        Validates inputs, encodes prompts (jointly with condition images),
        VAE-encodes condition images, prepares latents and timesteps, and
        returns all intermediate values as a dict.
        """
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            max_sequence_length,
        )

        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        batch_size = len(prompt)

        has_neg_prompt = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)
        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no "
                f"negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                "negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        images_per_prompt = [entry["prompt_images"] for entry in per_request_images]
        has_images = any(images_per_prompt)
        if has_images and not all(images_per_prompt):
            raise ValueError(
                "Cannot batch a mix of image-conditioned and text-to-image requests; "
                "the transformer's joint sequence requires a shared layout."
            )

        prompt_embeds, prompt_embeds_mask, image_pad_mask = self.encode_prompt(
            prompt=prompt,
            images_per_prompt=images_per_prompt if has_images else None,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask, negative_image_pad_mask = self.encode_prompt(
                prompt=negative_prompt,
                images_per_prompt=images_per_prompt if has_images else None,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                prompt_name="negative_prompt",
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None
            negative_image_pad_mask = None

        # Collate condition images per input slot across requests. Batched VAE
        # encoding requires every request to share the same per-slot sizes.
        vae_images_batched = None
        input_image_sizes: list[tuple[int, int]] = []
        if has_images:
            num_slots = len(per_request_images[0]["vae_images"])
            for entry in per_request_images:
                if entry["input_image_sizes"] != per_request_images[0]["input_image_sizes"]:
                    raise ValueError(
                        "All image-conditioned requests in a batch must share condition image sizes "
                        "(the transformer's joint sequence requires a shared layout)."
                    )
            input_image_sizes = per_request_images[0]["input_image_sizes"] or []
            vae_images_batched = [
                torch.cat([entry["vae_images"][slot] for entry in per_request_images], dim=0)
                for slot in range(num_slots)
            ]

        num_channels_latents = self.transformer.in_channels
        latents, image_latents = self.prepare_latents(
            vae_images_batched,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        img_shapes = [
            [
                *[(1, h // self.vae_scale_factor, w // self.vae_scale_factor) for w, h in input_image_sizes],
                (1, height // self.vae_scale_factor, width // self.vae_scale_factor),
            ]
        ] * batch_size

        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps,
            sigmas,
            latents.shape[1],
        )
        self._num_timesteps = len(timesteps)

        # The transformer's `img_mask` spans the joint sequence, so append one
        # slot per 2x2 group of target latents.
        img_mask = self._append_target_slots(image_pad_mask, latents.shape[1])
        if do_true_cfg:
            negative_img_mask = self._append_target_slots(negative_image_pad_mask, latents.shape[1])
        else:
            negative_img_mask = None

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "image_pad_mask": image_pad_mask,
            "negative_image_pad_mask": negative_image_pad_mask,
            "img_mask": img_mask,
            "negative_img_mask": negative_img_mask,
            "latents": latents,
            "image_latents": image_latents,
            "img_shapes": img_shapes,
            "timesteps": timesteps,
            "do_true_cfg": do_true_cfg,
        }

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        """Unpack, denormalize, and VAE-decode latents into a DiffusionOutput."""
        if output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        sampling_params_list = req.sampling_params_list
        common_sampling_params = sampling_params_list[0]
        prompt, negative_prompt = self._extract_prompts(req.prompts)
        per_request_images = self._extract_request_images(req.prompts)

        height = common_sampling_params.height or OUTPUT_RESOLUTION
        width = common_sampling_params.width or OUTPUT_RESOLUTION
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        num_inference_steps = common_sampling_params.num_inference_steps or 50
        sigmas = common_sampling_params.sigmas
        max_sequence_length = common_sampling_params.max_sequence_length or self._max_length
        num_images_per_prompt = (
            common_sampling_params.num_outputs_per_prompt if common_sampling_params.num_outputs_per_prompt > 0 else 1
        )
        generator = req.collate_request_generators(num_images_per_prompt, None)
        latents = req.collate_request_tensors("latents", None)
        true_cfg_scale = (
            common_sampling_params.true_cfg_scale if common_sampling_params.true_cfg_scale is not None else 4.0
        )
        output_type = common_sampling_params.output_type or "pil"

        # Batch homogeneity decides the attention path: a homogeneous batch
        # (same condition-image count and sizes per request) can use the
        # piecewise-span backend; heterogeneous batches take the dense 4D mask.
        signatures = [
            (
                len(entry["vae_images"] or []),
                tuple(entry["input_image_sizes"] or ()),
            )
            for entry in per_request_images
        ]
        homogeneous = all(sig == signatures[0] for sig in signatures)
        attention_kwargs = {"attn_path": "auto" if homogeneous else "mask"}

        ctx = self._prepare_generation_context(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            max_sequence_length=max_sequence_length,
            per_request_images=per_request_images,
            latents=latents,
            attention_kwargs=attention_kwargs,
        )

        latents = self.diffuse(
            ctx["prompt_embeds"],
            ctx["prompt_embeds_mask"],
            ctx["negative_prompt_embeds"],
            ctx["negative_prompt_embeds_mask"],
            ctx["img_mask"],
            ctx["negative_img_mask"],
            ctx["latents"],
            ctx["image_latents"],
            ctx["img_shapes"],
            ctx["timesteps"],
            ctx["do_true_cfg"],
            true_cfg_scale,
            attention_kwargs=self.attention_kwargs,
            additional_transformer_kwargs={
                "return_dict": False,
            },
        )

        self._current_timestep = None

        result = self._decode_latents(latents, height, width, output_type)
        return split_diffusion_output_by_request(
            result,
            req,
            num_outputs_per_prompt=num_images_per_prompt,
        )

    # ── Step-execution protocol ──

    def prepare_encode(
        self,
        state: "StepRequestState",
        **kwargs: Any,
    ) -> "StepRequestState":
        """Populate *state* with encoded prompts, latents, timesteps, and CFG config."""
        sampling = state.sampling
        prompt, negative_prompt = self._extract_prompts([state.prompt] if state.prompt is not None else [])
        per_request_images = self._extract_request_images([state.prompt] if state.prompt is not None else [])

        height = sampling.height or OUTPUT_RESOLUTION
        width = sampling.width or OUTPUT_RESOLUTION
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1

        ctx = self._prepare_generation_context(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=sampling.num_inference_steps or 50,
            sigmas=sampling.sigmas,
            num_images_per_prompt=num_images_per_prompt,
            generator=sampling.generator,
            true_cfg_scale=sampling.true_cfg_scale if sampling.true_cfg_scale is not None else 4.0,
            max_sequence_length=sampling.max_sequence_length or self._max_length,
            per_request_images=per_request_images,
            attention_kwargs=kwargs.get("attention_kwargs"),
        )

        # prepare_timesteps() has already materialized request-specific timestep
        # state on self.scheduler, so deepcopy preserves dynamic-shifting state
        # without replaying set_timesteps() on the per-request scheduler.
        # Per-request scheduler (must not share state with self.scheduler)
        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)

        state.prompt_embeds = ctx["prompt_embeds"]
        state.prompt_embeds_mask = ctx["prompt_embeds_mask"]
        state.negative_prompt_embeds = ctx["negative_prompt_embeds"]
        state.negative_prompt_embeds_mask = ctx["negative_prompt_embeds_mask"]
        state.latents = ctx["latents"]
        state.timesteps = ctx["timesteps"]
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = ctx["do_true_cfg"]
        state.guidance = None
        state.img_shapes = ctx["img_shapes"]
        # 2.1 applies plain CFG (no output normalization, unlike 2.0).
        state.sampling.cfg_normalize = False

        # Pipeline-specific state: the image-slot mask is stored *before* the
        # target slots are appended so a step batch can right-pad the prompt
        # region to a common length first. The KV cache is per request —
        # the pipeline is a singleton shared across requests.
        state.extra["image_pad_mask"] = ctx["image_pad_mask"]
        state.extra["negative_image_pad_mask"] = ctx["negative_image_pad_mask"]
        state.extra["image_latents"] = ctx["image_latents"]
        state.extra["height"] = height
        state.extra["width"] = width
        cache_enabled = getattr(self.transformer, "causal_condition", False)
        state.extra["kv_cache"] = (
            [{} for _ in range(len(self.transformer.transformer_blocks))] if cache_enabled else None
        )

        return state

    @staticmethod
    def _kv_cache_phase(kv_cache: list[dict] | None) -> str:
        if kv_cache is None:
            return "none"
        first = kv_cache[0]
        return "decode" if any("key" in branch for branch in first.values()) else "prefill"

    @staticmethod
    def _assemble_kv_cache(
        states: list["StepRequestState"],
    ) -> tuple[list[dict] | None, bool]:
        """Merge per-request KV caches for a batched forward.

        Returns `(kv_cache, take_ownership)`. When `take_ownership` is True the
        returned cache is the freshly written prefill cache and its entries must
        be scattered back into the per-request caches by `_scatter_kv_cache`.
        """
        caches = [state.extra.get("kv_cache") for state in states]
        if all(cache is None for cache in caches):
            return None, False
        phases = {QwenImage21Pipeline._kv_cache_phase(cache) for cache in caches}
        if phases == {"prefill"}:
            # Batched prefill: a fresh merged cache collects all requests' prefix
            # K/V; written entries are split back per request after the step.
            num_blocks = len(caches[0])
            return [{} for _ in range(num_blocks)], True
        if phases == {"decode"}:
            # Decode only reads the cache, so a batched view is enough and the
            # per-request caches stay untouched.
            num_blocks = len(caches[0])
            merged = []
            for block_idx in range(num_blocks):
                branches: dict[str, dict[str, torch.Tensor]] = {}
                branch_names = {branch for cache in caches for branch in cache[block_idx]}
                for branch in branch_names:
                    branches[branch] = {
                        part: torch.cat([cache[block_idx][branch][part] for cache in caches], dim=0)
                        for part in ("key", "value")
                    }
                merged.append(branches)
            return merged, False
        raise ValueError(
            "Cannot batch requests at mixed KV-cache phases (a request starting its first denoise step "
            "joined a batch of in-flight requests). Schedule them separately."
        )

    @staticmethod
    def _scatter_kv_cache(
        states: list["StepRequestState"],
        row_counts: list[int],
        kv_cache: list[dict],
    ) -> None:
        """Split a batched prefill cache back into the per-request caches."""
        start = 0
        for state, rows in zip(states, row_counts):
            cache = state.extra["kv_cache"]
            for block_idx, block_cache in enumerate(kv_cache):
                for branch, parts in block_cache.items():
                    cache[block_idx][branch] = {
                        part: tensor[start : start + rows].contiguous() for part, tensor in parts.items()
                    }
            start += rows

    def denoise_step(
        self,
        input_batch: "InputBatch",
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """One denoise step: read from *input_batch*, delegate to CFGParallelMixin."""
        del kwargs
        if self.interrupt:
            return None

        t = input_batch.timesteps
        self._current_timestep = t
        self.transformer.do_true_cfg = input_batch.do_true_cfg

        states = list(input_batch.states)
        if not states:
            raise ValueError("QwenImage21Pipeline.denoise_step requires per-request states on the InputBatch.")

        latents = input_batch.latents
        timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

        # Assemble the joint image-slot mask: right-pad each request's VLM mask
        # to the batch's padded prompt length, then append the target slots.
        max_seq_len = input_batch.prompt_embeds.shape[1]
        mask_rows = []
        for state in states:
            row = state.extra["image_pad_mask"]
            if row.shape[1] < max_seq_len:
                row = F.pad(row, (0, max_seq_len - row.shape[1]), value=False)
            mask_rows.append(row)
        img_mask = self._append_target_slots(torch.cat(mask_rows, dim=0), latents.shape[1])
        negative_img_mask = None
        if input_batch.do_true_cfg:
            negative_seq_len = input_batch.negative_prompt_embeds.shape[1]
            neg_rows = []
            for state in states:
                row = state.extra["negative_image_pad_mask"]
                if row.shape[1] < negative_seq_len:
                    row = F.pad(row, (0, negative_seq_len - row.shape[1]), value=False)
                neg_rows.append(row)
            negative_img_mask = self._append_target_slots(torch.cat(neg_rows, dim=0), latents.shape[1])

        image_latents_list = [state.extra["image_latents"] for state in states]
        if all(image_latent is None for image_latent in image_latents_list):
            image_latents = None
        elif all(image_latent is not None for image_latent in image_latents_list):
            image_latents = torch.cat(image_latents_list, dim=0)
        else:
            raise ValueError("Cannot batch a mix of image-conditioned and text-to-image requests in step mode.")

        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([image_latents, latents], dim=1)

        row_counts = [state.latents.shape[0] for state in states]
        kv_cache, take_ownership = self._assemble_kv_cache(states)

        positive_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timestep / 1000,
            "encoder_hidden_states_mask": input_batch.prompt_embeds_mask,
            "encoder_hidden_states": input_batch.prompt_embeds,
            "img_shapes": input_batch.img_shapes,
            "img_mask": img_mask,
            "attention_kwargs": self.attention_kwargs,
            "kv_cache": kv_cache,
            "cache_branch": "cond",
            "return_dict": False,
        }
        if input_batch.do_true_cfg:
            negative_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep / 1000,
                "encoder_hidden_states_mask": input_batch.negative_prompt_embeds_mask,
                "encoder_hidden_states": input_batch.negative_prompt_embeds,
                "img_shapes": input_batch.img_shapes,
                "img_mask": negative_img_mask,
                "attention_kwargs": self.attention_kwargs,
                "kv_cache": kv_cache,
                "cache_branch": "uncond",
                "return_dict": False,
            }
        else:
            negative_kwargs = None

        noise_pred = self.predict_noise_maybe_with_cfg(
            input_batch.do_true_cfg,
            input_batch.true_cfg_scale,
            positive_kwargs,
            negative_kwargs,
            input_batch.cfg_normalize,
            output_slice=latents.size(1),
        )

        if take_ownership:
            self._scatter_kv_cache(states, row_counts, kv_cache)

        return noise_pred

    def step_scheduler(
        self,
        state: "StepRequestState",
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """One scheduler step: update ``state.latents`` and advance ``step_index``."""
        if self.interrupt:
            return

        t = state.current_timestep
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        )

        state.step_index += 1

    def post_decode(
        self,
        state: "StepRequestState",
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final latents from *state*."""
        self._current_timestep = None

        height = state.extra.get("height") or state.sampling.height or OUTPUT_RESOLUTION
        width = state.extra.get("width") or state.sampling.width or OUTPUT_RESOLUTION
        output_type = kwargs.get("output_type") or state.sampling.output_type or "pil"

        return self._decode_latents(state.latents, height, width, output_type)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

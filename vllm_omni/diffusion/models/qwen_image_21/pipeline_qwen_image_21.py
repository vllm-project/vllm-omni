# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from diffusers' pipeline_qwenimage21.py (Qwen-Image 2.1 T2I + image-conditioned generation).

import copy
import dataclasses
import json
import logging
import math
import os
from collections import OrderedDict
from collections.abc import Iterable
from numbers import Real
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

from vllm_omni.config.stage_config import DiffusionStageOperator
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage21 import (
    DistributedAutoencoderKLQwenImage21,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.lora.loader import QwenImageLoraLoaderMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.model_metadata import QWEN_IMAGE_21_MAX_INPUT_IMAGES
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery, role_loads_component
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
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)
from vllm_omni.diffusion.models.stage_mixin import DiffusionStageMixin
from vllm_omni.diffusion.models.utils import create_transformers_model
from vllm_omni.diffusion.offloader.config import offload_enabled
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


def _should_encode_negative_prompt(true_cfg_scale: float | None, negative_prompt: str | list[str] | None) -> bool:
    """Share the negative-conditioning policy between scheduling and encoding.

    An absent negative prompt does not imply an empty/default negative prompt.
    Explicit empty strings do request negative conditioning when CFG is enabled.
    """
    if true_cfg_scale is not None:
        try:
            valid_scale = isinstance(true_cfg_scale, Real) and math.isfinite(true_cfg_scale)
        except OverflowError:
            valid_scale = False
        if not valid_scale:
            raise ValueError("true_cfg_scale must be a finite real number or None.")
    if true_cfg_scale is None or true_cfg_scale <= 1 or negative_prompt is None:
        return False
    if isinstance(negative_prompt, list):
        if not negative_prompt or any(not isinstance(prompt, str) for prompt in negative_prompt):
            raise ValueError("negative_prompt must be a string, a non-empty list of strings, or None.")
    elif negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValueError("negative_prompt must be a string, a non-empty list of strings, or None.")
    return True


def get_qwen_image_21_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for QwenImage21Pipeline.

    Reads condition images from ``request.prompt["multi_modal_data"]["image"]``
    (single image or list), resizes them once for both the vision-language
    prompt encoder and the VAE, and stores the results under
    ``prompt["additional_information"]``. Unused negative prompts are cleared
    before admission and cross-role handoff, including for text-only requests.
    """
    vae_scale_factor = _read_vae_scale_factor(od_config)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    # The Qwen-Image 2.1 VAE is RGBA (in/out_channels=4): condition images get an
    # opaque alpha channel unless the input already carries one.
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    sample_sigmas = (getattr(od_config, "extras", None) or {}).get("sample_sigmas")

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
        if request.sampling_params.true_cfg_scale is None:
            request.sampling_params.true_cfg_scale = 1.0
        # Request-level sigmas may arrive via whitelisted extra-body params,
        # which the serving layer stores in ``extra_args`` rather than the
        # typed field (same convention as LTX2). Hoist them so the pipeline
        # sees the explicit override before the preset grid is injected.
        if request.sampling_params.sigmas is None:
            extra_sigmas = (request.sampling_params.extra_args or {}).get("sigmas")
            if extra_sigmas is not None:
                request.sampling_params.sigmas = [float(s) for s in extra_sigmas]
        # Inject the model-level sampling grid as request sigmas so the
        # step scheduler and cache refresh resolve the total step count from
        # len(sigmas). prepare_timesteps applies the same grid as a fallback
        # for paths that bypass this hook; request-level sigmas always win.
        if request.sampling_params.sigmas is None and sample_sigmas:
            request.sampling_params.sigmas = list(sample_sigmas)
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
        raw_image = None
        if multi_modal_data is not None:
            # The multistage OpenAI image-edit path uses ``img2img`` for a
            # single reference image, while the direct path uses ``image``.
            # Normalize both forms here so stage separation does not drop the
            # image condition before ENCODE.
            raw_image = multi_modal_data.get("image")
            if raw_image is None:
                raw_image = multi_modal_data.get("img2img")
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        else:
            # CFG sweeps and fan-out can share a caller-owned prompt dict.
            # Copy its top-level fields without cloning conditioning tensors.
            prompt = dict(prompt)
        encode_negative_prompt = _should_encode_negative_prompt(
            request.sampling_params.true_cfg_scale, prompt.get("negative_prompt")
        )
        if encode_negative_prompt and not isinstance(prompt.get("negative_prompt"), str):
            raise ValueError("A single request's negative_prompt must be a string or None.")
        if not encode_negative_prompt and prompt.get("negative_prompt") is not None:
            # Preserve CFG-off ignore semantics beyond encoding: compatibility
            # keys, cache preparation and role handoff must not see placeholders.
            prompt["negative_prompt"] = None
        # Image preparation writes into this container. Keep its fields local
        # to the request, while retaining references to tensor payloads. The
        # preparation contract is replace-only: assign complete field values;
        # do not append to or mutate nested caller-owned values in place.
        prompt["additional_information"] = dict(prompt.get("additional_information") or {})

        metadata = DiffusionRequestBatch.get_prompt_field(prompt, "qwen_image_21_metadata")
        conditioning = DiffusionRequestBatch.get_prompt_field(prompt, "qwen_image_21_conditioning")
        if metadata is not None or conditioning is not None or "_stage_payload_transfer" in prompt:
            # The connector transfers tensors on the worker. Geometry remains
            # inline so scheduling can group requests before that receive.
            sizes = (metadata or conditioning or {}).get("input_image_sizes", [])
            if metadata is not None:
                request.sampling_params.height = metadata["height"]
                request.sampling_params.width = metadata["width"]
            request.batch_compatibility_key = (
                "qwen_image_21",
                tuple(tuple(size) for size in sizes),
                request.sampling_params.max_sequence_length,
                (metadata or {}).get(
                    "has_negative_conditioning", (conditioning or {}).get("negative_prompt_embeds") is not None
                ),
            )
            request.prompt = prompt
            return request

        if isinstance(raw_image, tuple):
            raw_image = list(raw_image)
        if raw_image is None or (isinstance(raw_image, list) and len(raw_image) == 0):
            request.batch_compatibility_key = (
                "qwen_image_21",
                (),
                request.sampling_params.max_sequence_length,
                encode_negative_prompt,
            )
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
        prompt["additional_information"]["vae_images_are_latents"] = False
        prompt["additional_information"]["input_image_sizes"] = input_image_sizes
        prompt["additional_information"]["calculated_height"] = calculated_height
        prompt["additional_information"]["calculated_width"] = calculated_width
        request.batch_compatibility_key = (
            "qwen_image_21",
            tuple(input_image_sizes),
            request.sampling_params.max_sequence_length,
            encode_negative_prompt,
        )
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
    DiffusionStageMixin,
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
    _stage_dummy_methods = {
        DiffusionStageOperator.DIFFUSE: "_prepare_dummy_denoise_payload",
        DiffusionStageOperator.DECODE: "_prepare_dummy_decode_payload",
    }
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def release_captured_graphs(self) -> None:
        """Drop decode graphs before sleep level 2 discards their capture storage.

        The runner looks this up on the pipeline, not on the transformer.
        """
        self._prompt_embedding_cache = OrderedDict()
        transformer = getattr(self, "transformer", None)
        release = getattr(transformer, "release_captured_graphs", None)
        if callable(release):
            release()

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__(od_config=od_config)
        self.parallel_config = od_config.parallel_config
        # Model-level sampling grid from model_index.json (diffusers PR
        # 14950): when the request carries no explicit sigmas, this grid
        # drives the schedule and its length decides the step count.
        sample_sigmas = od_config.extras.get("sample_sigmas")
        self.sample_sigmas = [float(s) for s in sample_sigmas] if sample_sigmas else None
        load_encoder = role_loads_component(self.stage_role, "encoder")
        load_dit = role_loads_component(self.stage_role, "dit")
        load_decoder = role_loads_component(self.stage_role, "vae")
        load_vae = load_encoder or load_decoder
        self.weights_sources = []
        if load_dit:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer",
                    revision=None,
                    prefix="transformer.",
                    fall_back_to_pt=True,
                )
            )

        self.device = get_local_device()
        model = od_config.model
        # Check if model is a local path
        local_files_only = os.path.isdir(model)

        # Guard against transformers v5 multi-worker race on partial subfolder
        # shard sets (see pipeline_qwen_image_edit_plus, Buildkite #1043).
        qwen_subfolders = ["scheduler"]
        if load_encoder:
            qwen_subfolders.extend(["text_encoder", "processor"])
        if load_vae:
            qwen_subfolders.append("vae")
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
        if not load_encoder:
            self.text_encoder = None
        elif text_encoder_quant_config is None:
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
        self.vae = None
        if load_vae:
            self.vae = from_pretrained_with_prefetch(
                DistributedAutoencoderKLQwenImage21.from_pretrained,
                model,
                subfolder="vae",
                prefetch_list=qwen_subfolders,
                local_files_only=local_files_only,
            )
            if load_encoder != load_decoder:
                self.vae.retain_stage_components(encode=load_encoder)
            self.vae = self.vae.to(self.device)
        self.transformer = None
        if load_dit:
            transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImage21Transformer2DModel)
            self.transformer = QwenImage21Transformer2DModel(
                od_config=od_config,
                quant_config=resolve_component_quant_config(od_config.quantization_config, "transformer"),
                **transformer_kwargs,
            )
        self.processor = None
        if load_encoder:
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
        self.latent_channels = self.vae.config.z_dim if self.vae is not None else self.transformer.in_channels

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
        if load_encoder:
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
        cacheable = (
            len(prompt) == 1
            and isinstance(prompt[0], str)
            and not (images_per_prompt is not None and any(images_per_prompt))
            and not torch.is_grad_enabled()
            and not self.text_encoder.training
            and not getattr(self.od_config, "lora_config", None)
            and not offload_enabled(self.od_config)
        )
        cache = getattr(self, "_prompt_embedding_cache", None)
        if cache is None or not cacheable:
            cache = self._prompt_embedding_cache = OrderedDict()
        key = (
            (
                tuple(prompt),
                max_sequence_length,
                prompt_name,
                id(self.text_encoder),
                id(self.processor),
                self.text_encoder.dtype,
                str(self.device),
                self.prompt_template_t2i,
                self._drop_idx,
            )
            if cacheable
            else None
        )
        encoded = cache.get(key) if cacheable else None
        if encoded is None:
            encoded = self._get_qwen_prompt_embeds(
                prompt,
                images_per_prompt,
                max_sequence_length=max_sequence_length,
                prompt_name=prompt_name,
            )
            size = sum(t.numel() * t.element_size() for t in encoded)
            if cacheable and size <= 16 * 1024 * 1024:
                cache[key] = encoded
                while (
                    len(cache) > 16
                    or sum(t.numel() * t.element_size() for entry in cache.values() for t in entry) > 16 * 1024 * 1024
                ):
                    cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        prompt_embeds, prompt_embeds_mask, image_pad_mask = encoded

        # Request-major expansion (A,A,B,B). ``repeat`` on the 2D masks would
        # prepend a dimension and reshape to A,B,A,B — wrong pairing with embeds.
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)
        image_pad_mask = image_pad_mask.repeat_interleave(num_images_per_prompt, dim=0)

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
        *,
        image_is_latent: bool = False,
    ):
        """Prepare target noise and pack explicitly identified condition images.

        Raw RGBA pixels and normalized VAE latents can have the same channel
        count; their representation must be supplied by the caller.
        """
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        image_latents = None
        if vae_images is not None:
            all_image_latents = []
            for image in vae_images:
                image = image.to(device=device, dtype=dtype)
                if image_is_latent:
                    if image.ndim != 5 or image.shape[1] != self.latent_channels:
                        raise ValueError("Normalized Qwen image latents must have BCTHW layout and latent_channels.")
                    encoded = image
                else:
                    encoded = self._encode_vae_image(image, generator)
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
        if sigmas is None:
            sigmas = getattr(self, "sample_sigmas", None)
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
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
    def default_num_inference_steps(self):
        """Step count implied by the model-level sampling grid, when present.

        The model runner falls back to this when the request carries neither
        num_inference_steps, timesteps, nor sigmas.
        """
        sample_sigmas = getattr(self, "sample_sigmas", None)
        if sample_sigmas is not None:
            return len(sample_sigmas)
        return None

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
                    "vae_images_are_latents": info.get("vae_images_are_latents", False),
                    "input_image_sizes": info.get("input_image_sizes"),
                }
            )
        return per_request

    def _encode_conditioning(
        self, prompt, negative_prompt, per_request_images, true_cfg_scale, max_sequence_length, *, generator=None
    ):
        """Encode one row per prompt, keeping condition latents in BCTHW VAE layout."""
        encode_negative = _should_encode_negative_prompt(true_cfg_scale, negative_prompt)
        prompt_count = len(prompt) if isinstance(prompt, list) else 1
        if encode_negative and isinstance(negative_prompt, list) and len(negative_prompt) != prompt_count:
            raise ValueError("negative_prompt must contain one string per prompt.")
        images_per_prompt = [entry["prompt_images"] for entry in per_request_images]
        has_images = any(images_per_prompt)
        if has_images and not all(images_per_prompt):
            raise ValueError("Cannot batch a mix of image-conditioned and text-to-image requests.")
        prompt_embeds, prompt_embeds_mask, image_pad_mask = self.encode_prompt(
            prompt=prompt,
            images_per_prompt=images_per_prompt if has_images else None,
            max_sequence_length=max_sequence_length,
        )
        negative_prompt_embeds = negative_prompt_embeds_mask = negative_image_pad_mask = None
        if encode_negative:
            negative_prompt_embeds, negative_prompt_embeds_mask, negative_image_pad_mask = self.encode_prompt(
                prompt=negative_prompt,
                images_per_prompt=images_per_prompt if has_images else None,
                max_sequence_length=max_sequence_length,
                prompt_name="negative_prompt",
            )

        image_latents = None
        input_image_sizes = []
        if has_images:
            input_image_sizes = per_request_images[0]["input_image_sizes"] or []
            if any(entry["input_image_sizes"] != input_image_sizes for entry in per_request_images):
                raise ValueError("All image-conditioned requests in a batch must share condition image sizes.")
            latent_flags = [entry.get("vae_images_are_latents", False) for entry in per_request_images]
            if any(type(flag) is not bool for flag in latent_flags):
                raise ValueError("vae_images_are_latents must be an explicit boolean.")
            if len(set(latent_flags)) != 1:
                raise ValueError("Cannot batch raw pixels with pre-encoded image latents.")
            image_latents = []
            for slot in range(len(input_image_sizes)):
                image = torch.cat([entry["vae_images"][slot] for entry in per_request_images], dim=0).to(
                    device=self.device, dtype=prompt_embeds.dtype
                )
                if latent_flags[0]:
                    if image.ndim != 5 or image.shape[1] != self.latent_channels:
                        raise ValueError("Normalized Qwen image latents must have BCTHW layout and latent_channels.")
                    image_latents.append(image)
                else:
                    image_latents.append(self._encode_vae_image(image, generator=generator))
        return {
            "version": 1,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "image_pad_mask": image_pad_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "negative_image_pad_mask": negative_image_pad_mask,
            "image_latents": image_latents,
            "input_image_sizes": input_image_sizes,
        }

    def _stage_conditioning(self, prompt) -> dict[str, Any]:
        conditioning = DiffusionRequestBatch.get_prompt_field(prompt, "qwen_image_21_conditioning")
        if not isinstance(conditioning, dict) or conditioning.get("version") != 1:
            raise ValueError("QwenImage21 generation requires version-1 'qwen_image_21_conditioning' from ENCODE.")
        for prefix in ("", "negative_"):
            embeds = conditioning.get(f"{prefix}prompt_embeds")
            if prefix and embeds is None:
                continue
            mask = conditioning.get(f"{prefix}prompt_embeds_mask")
            image_mask = conditioning.get(f"{prefix}image_pad_mask")
            if (
                not isinstance(embeds, torch.Tensor)
                or embeds.ndim != 3
                or embeds.shape[0] != 1
                or not isinstance(mask, torch.Tensor)
                or mask.shape != embeds.shape[:2]
                or not isinstance(image_mask, torch.Tensor)
                or image_mask.shape != mask.shape
            ):
                raise ValueError(f"Invalid QwenImage21 {prefix}conditioning embeddings or masks.")
        sizes = conditioning.get("input_image_sizes")
        images = conditioning.get("image_latents")
        if not isinstance(sizes, (list, tuple)) or len(sizes) > MAX_QWEN_IMAGE_21_INPUT_IMAGES:
            raise ValueError("Invalid QwenImage21 conditioning image sizes.")
        if sizes and (not isinstance(images, (list, tuple)) or len(images) != len(sizes)):
            raise ValueError("QwenImage21 conditioning image sizes and latents must align.")
        for (width, height), image in zip(sizes, images or [], strict=True):
            if (
                not isinstance(image, torch.Tensor)
                or image.ndim != 5
                or image.shape
                != (1, self.latent_channels, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
            ):
                raise ValueError("Invalid QwenImage21 conditioning image latent geometry.")
        return conditioning

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
        conditioning=None,
    ):
        """Shared preparation logic for forward() and prepare_encode().

        Reuse supplied conditioning or encode it, then prepare noise and schedule.
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

        has_neg_prompt = negative_prompt is not None or (
            conditioning is not None and conditioning.get("negative_prompt_embeds") is not None
        )
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

        if conditioning is None:
            conditioning = self._encode_conditioning(
                prompt, negative_prompt, per_request_images, true_cfg_scale, max_sequence_length, generator=generator
            )
        if do_true_cfg and conditioning.get("negative_prompt_embeds") is None:
            raise ValueError("QwenImage21 CFG requires negative conditioning from the ENCODE stage.")
        expanded = {
            key: value.to(self.device).repeat_interleave(num_images_per_prompt, dim=0)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in conditioning.items()
        }
        prompt_embeds = expanded["prompt_embeds"]
        prompt_embeds_mask = expanded["prompt_embeds_mask"]
        image_pad_mask = expanded["image_pad_mask"]
        negative_prompt_embeds = expanded["negative_prompt_embeds"] if do_true_cfg else None
        negative_prompt_embeds_mask = expanded["negative_prompt_embeds_mask"] if do_true_cfg else None
        negative_image_pad_mask = expanded["negative_image_pad_mask"] if do_true_cfg else None
        input_image_sizes = conditioning["input_image_sizes"]
        latents, image_latents = self.prepare_latents(
            conditioning["image_latents"],
            batch_size * num_images_per_prompt,
            self.latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
            image_is_latent=True,
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

    def _generate_latents(self, req: DiffusionRequestBatch, *, conditioning=None):
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
            common_sampling_params.true_cfg_scale if common_sampling_params.true_cfg_scale is not None else 1.0
        )

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
            conditioning=conditioning,
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
        return latents, height, width, num_images_per_prompt

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Run the complete native pipeline, independently of deployment role."""
        latents, height, width, num_images_per_prompt = self._generate_latents(req)
        result = self._decode_latents(latents, height, width, req.sampling_params_list[0].output_type or "pil")
        return split_diffusion_output_by_request(
            result,
            req,
            num_outputs_per_prompt=num_images_per_prompt,
        )

    @staticmethod
    def _validate_stage_batch_settings(batch: DiffusionRequestBatch, fields: tuple[str, ...]) -> None:
        """Reject incompatible direct callers before running shared model computation.

        RequestScheduler groups all these sampling fields. Atomic E/D calls
        under StepScheduler additionally validate the fields they consume.
        """
        if not batch.requests:
            return
        first = batch.sampling_params_list[0]
        for sampling in batch.sampling_params_list[1:]:
            for field in fields:
                value, reference = getattr(sampling, field), getattr(first, field)
                if field == "sigmas":
                    # Direct callers may supply tensors/arrays despite the
                    # public list annotation. Preserve the input precision
                    # when comparing schedules across these representations.
                    if value is None or reference is None:
                        matches = value is reference
                    elif isinstance(value, torch.Tensor) or isinstance(reference, torch.Tensor):
                        # Deliberately compare stored values across dtypes;
                        # casting to value.dtype would hide precision mismatches.
                        value = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
                        reference = (
                            reference
                            if isinstance(reference, torch.Tensor)
                            else torch.from_numpy(np.asarray(reference))
                        )
                        matches = torch.equal(value, reference.to(device=value.device))
                    else:
                        matches = np.array_equal(value, reference)
                else:
                    matches = value == reference
                if not matches:
                    raise ValueError(f"QwenImage21 stage batch requires matching {field}.")

    def encode_batch(self, batch: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Encode compatible requests together, then retain each request's geometry."""
        if not batch.requests:
            return []
        self._validate_stage_batch_settings(batch, ("height", "width", "max_sequence_length", "true_cfg_scale"))
        prompt, negative_prompt = self._extract_prompts(batch.prompts)
        sampling = batch.sampling_params_list[0]
        height, width = normalize_min_aligned_size(
            sampling.height or OUTPUT_RESOLUTION,
            sampling.width or OUTPUT_RESOLUTION,
            self.vae_scale_factor * 2,
        )
        max_sequence_length = sampling.max_sequence_length or self._max_length
        self.check_inputs(prompt, height, width, negative_prompt, max_sequence_length)
        conditioning = self._encode_conditioning(
            prompt,
            negative_prompt,
            self._extract_request_images(batch.prompts),
            sampling.true_cfg_scale if sampling.true_cfg_scale is not None else 1.0,
            max_sequence_length,
            generator=batch.collate_request_generators(max(sampling.num_outputs_per_prompt, 1), None),
        )
        outputs = []
        for index, request in enumerate(batch.requests):
            request_conditioning = {
                key: value[index : index + 1] if isinstance(value, torch.Tensor) else value
                for key, value in conditioning.items()
            }
            if conditioning["image_latents"] is not None:
                request_conditioning["image_latents"] = [
                    image[index : index + 1] for image in conditioning["image_latents"]
                ]
            height, width = normalize_min_aligned_size(
                request.sampling_params.height or OUTPUT_RESOLUTION,
                request.sampling_params.width or OUTPUT_RESOLUTION,
                self.vae_scale_factor * 2,
            )
            outputs.append(
                DiffusionOutput(
                    custom_output={
                        "qwen_image_21_conditioning": request_conditioning,
                        "qwen_image_21_metadata": {
                            "height": height,
                            "width": width,
                            "input_image_sizes": conditioning["input_image_sizes"],
                            "has_negative_conditioning": conditioning["negative_prompt_embeds"] is not None,
                        },
                    }
                )
            )
        return outputs

    def _collate_stage_conditioning(self, batch: DiffusionRequestBatch) -> dict[str, Any]:
        """Right-pad text sequences and concatenate request rows for native DiT batching."""
        rows = [self._stage_conditioning(request.prompt) for request in batch.requests]
        result = dict(rows[0])
        for prefix in ("", "negative_"):
            fields = tuple(prefix + suffix for suffix in ("prompt_embeds", "prompt_embeds_mask", "image_pad_mask"))
            present = [row[fields[0]] is not None for row in rows]
            if not any(present):
                continue
            if not all(present):
                raise ValueError("QwenImage21 stage batch requires matching negative conditioning availability.")
            length = max(row[fields[0]].shape[1] for row in rows)
            for field in fields:
                padded = []
                for row in rows:
                    value = row[field].to(self.device)
                    padding = (0, 0, 0, length - value.shape[1]) if value.ndim == 3 else (0, length - value.shape[1])
                    padded.append(F.pad(value, padding))
                result[field] = torch.cat(padded, dim=0)
        sizes = result["input_image_sizes"]
        if any(row["input_image_sizes"] != sizes for row in rows):
            raise ValueError("QwenImage21 stage batch requires matching condition image sizes.")
        if result["image_latents"] is not None:
            result["image_latents"] = [
                torch.cat([row["image_latents"][slot].to(self.device) for row in rows], dim=0)
                for slot in range(len(sizes))
            ]
        return result

    def diffuse_batch(self, batch: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Denoise the scheduler batch in one native DiT call, preserving row order."""
        if not batch.requests:
            return []
        self._validate_stage_batch_settings(
            batch,
            (
                "height",
                "width",
                "max_sequence_length",
                "true_cfg_scale",
                "num_outputs_per_prompt",
                "num_inference_steps",
                "sigmas",
            ),
        )
        conditioning = self._collate_stage_conditioning(batch)
        latents, height, width, num_outputs = self._generate_latents(batch, conditioning=conditioning)
        return [
            DiffusionOutput(
                custom_output={
                    "latents": latents[index * num_outputs : (index + 1) * num_outputs],
                    "qwen_image_21_metadata": {"height": height, "width": width, "input_image_sizes": []},
                }
            )
            for index in range(batch.num_reqs)
        ]

    def decode_batch(self, batch: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Decode compatible request latents together through the native VAE path."""
        if not batch.requests:
            return []
        self._validate_stage_batch_settings(batch, ("height", "width", "output_type", "num_outputs_per_prompt"))
        sampling = batch.sampling_params_list[0]
        height, width = normalize_min_aligned_size(
            sampling.height or OUTPUT_RESOLUTION,
            sampling.width or OUTPUT_RESOLUTION,
            self.vae_scale_factor * 2,
        )
        num_outputs = max(sampling.num_outputs_per_prompt, 1)
        expected_shape = (
            num_outputs,
            height // self.vae_scale_factor * (width // self.vae_scale_factor),
            self.latent_channels,
        )
        latents = []
        for request in batch.requests:
            value = DiffusionRequestBatch.get_prompt_field(request.prompt, "latents")
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError("QwenImage21 DECODE requires packed latents matching the output geometry and count.")
            latents.append(value.to(self.device))
        result = self._decode_latents(torch.cat(latents, dim=0), height, width, sampling.output_type or "pil")
        return split_diffusion_output_by_request(result, batch, num_outputs_per_prompt=num_outputs)

    def _prepare_dummy_denoise_payload(self, batch: DiffusionRequestBatch) -> None:
        """Profile G conservatively at the configured maximum text length.

        G has no encoder to determine the real prompt length during local
        warmup. This upper bound can reserve more memory than FULL's warmup
        with encoded text; it does not change runtime request sequence lengths.
        """
        for request in batch.requests:
            if DiffusionRequestBatch.get_prompt_field(request.prompt, "qwen_image_21_conditioning") is not None:
                continue
            sequence_length = request.sampling_params.max_sequence_length or self._max_length
            context_dim = self.od_config.tf_model_config.get("context_in_dim", 4096)
            embeds = torch.zeros(1, sequence_length, context_dim, device=self.device, dtype=self.od_config.dtype)
            mask = torch.ones(1, sequence_length, device=self.device, dtype=torch.bool)
            conditioning = {
                "version": 1,
                "prompt_embeds": embeds,
                "prompt_embeds_mask": mask,
                "image_pad_mask": torch.zeros_like(mask),
                "negative_prompt_embeds": embeds,
                "negative_prompt_embeds_mask": mask,
                "negative_image_pad_mask": torch.zeros_like(mask),
                "image_latents": None,
                "input_image_sizes": [],
            }
            prompt = dict(request.prompt) if isinstance(request.prompt, dict) else {"prompt": request.prompt}
            prompt["qwen_image_21_conditioning"] = conditioning
            request.prompt = prompt

    def _prepare_dummy_decode_payload(self, batch: DiffusionRequestBatch) -> None:
        """Supply target latents absent from a decoder-only warmup."""
        for request in batch.requests:
            if DiffusionRequestBatch.get_prompt_field(request.prompt, "latents") is not None:
                continue
            sampling = request.sampling_params
            height, width = normalize_min_aligned_size(
                sampling.height or OUTPUT_RESOLUTION,
                sampling.width or OUTPUT_RESOLUTION,
                self.vae_scale_factor * 2,
            )
            prompt = dict(request.prompt) if isinstance(request.prompt, dict) else {"prompt": request.prompt}
            prompt["latents"] = torch.zeros(
                max(sampling.num_outputs_per_prompt, 1),
                height // self.vae_scale_factor * (width // self.vae_scale_factor),
                self.latent_channels,
                device=self.device,
                dtype=self.vae.dtype,
            )
            request.prompt = prompt

    # ── Step-execution protocol ──

    def prepare_encode(
        self,
        state: "StepRequestState",
        **kwargs: Any,
    ) -> "StepRequestState":
        """Populate *state* with encoded prompts, latents, timesteps, and CFG config."""
        return self._prepare_step_state(state, **kwargs)

    def prepare_denoise(self, state: "StepRequestState", **kwargs: Any) -> "StepRequestState":
        """Initialize step state from upstream conditioning without running encoders."""
        if OmniDiffusionRequest.is_dummy_run_request_id(state.request_id):
            request = OmniDiffusionRequest(
                prompt=state.prompt, sampling_params=state.sampling, request_id=state.request_id
            )
            self._prepare_dummy_denoise_payload(DiffusionRequestBatch(requests=[request]))
            state.prompt = request.prompt
        return self._prepare_step_state(state, conditioning=self._stage_conditioning(state.prompt), **kwargs)

    def _prepare_step_state(self, state: "StepRequestState", *, conditioning=None, **kwargs: Any) -> "StepRequestState":
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
            true_cfg_scale=sampling.true_cfg_scale if sampling.true_cfg_scale is not None else 1.0,
            max_sequence_length=sampling.max_sequence_length or self._max_length,
            per_request_images=per_request_images,
            attention_kwargs=kwargs.get("attention_kwargs"),
            conditioning=conditioning,
            latents=sampling.latents,
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
            QwenImage21KVCache({} for _ in self.transformer.transformer_blocks) if cache_enabled else None
        )

        return state

    @staticmethod
    def _kv_cache_phase(kv_cache: list[dict] | None) -> str:
        if kv_cache is None:
            return "none"
        first = kv_cache[0]
        return "decode" if any("key" in branch for branch in first.values()) else "prefill"

    def _release_idle_decode_graphs(self) -> None:
        """Free finished requests' static prefix copies before the next prefill allocates."""
        manager = getattr(self.transformer, "_decode_graph_manager", None)
        if manager is None:
            return
        manager.release_idle_entries()

    def release_step_request(self, state: "StepRequestState") -> None:
        """Drop this request's prefix KV and geometry when the runner retires it.

        The graph table is left alone here. The owner weakref dies with this
        cache, and the next prefill drops that dead entry before it allocates.
        """
        extra = getattr(state, "extra", None)
        if isinstance(extra, dict):
            extra["kv_cache"] = None
        release_geometry = getattr(self.transformer, "release_request_geometry", None)
        if release_geometry is not None:
            release_geometry()

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
            return QwenImage21KVCache({} for _ in range(num_blocks)), True
        if phases == {"decode"}:
            if len(caches) == 1:
                # Keep the request owner stable across decode steps. Prefill
                # still scatters and records the padded prefix lengths above.
                return caches[0], False
            # Decode only reads the cache, so a batched view is enough and the
            # per-request caches stay untouched. Merge every stored part — FP8
            # cache entries carry per-request scale tensors alongside "key"/"value".
            # Callers must pass a homogeneous baked-prefix-length group: denoise_step
            # sub-batches by prefill_prompt_seq_len before assembling.
            prefix_lens = {state.extra.get("prefill_prompt_seq_len") for state in states}
            if len(prefix_lens) > 1:
                raise ValueError(
                    "Cannot merge decode KV for requests baked at different padded prompt "
                    "lengths. denoise_step must sub-batch by prefill_prompt_seq_len first."
                )
            num_blocks = len(caches[0])
            merged = QwenImage21KVCache()
            for block_idx in range(num_blocks):
                branches: dict[str, dict[str, torch.Tensor]] = {}
                branch_names = {branch for cache in caches for branch in cache[block_idx]}
                for branch in branch_names:
                    part_names = {part for cache in caches for part in cache[block_idx][branch]}
                    branches[branch] = {
                        part: torch.cat([cache[block_idx][branch][part] for cache in caches], dim=0)
                        for part in sorted(part_names)
                    }
                merged.append(branches)
            return merged, False
        raise ValueError(
            "Cannot batch requests at mixed KV-cache phases (a request starting its first denoise step "
            "joined a batch of in-flight requests). Schedule them separately."
        )

    @staticmethod
    def _decode_group_key(state: "StepRequestState") -> tuple[Any, ...]:
        """Group decode requests by the padded VLM length baked into their KV."""
        return (
            state.extra.get("prefill_prompt_seq_len"),
            state.extra.get("prefill_negative_prompt_seq_len"),
        )

    @staticmethod
    def _split_decode_groups(
        states: list["StepRequestState"],
    ) -> list[list["StepRequestState"]]:
        """Split a decode batch into homogeneous baked-prefix-length groups.

        Preserves first-seen group order and relative order within each group.
        Callers stitch per-group noise preds back into the input request order.
        """
        groups: dict[tuple[Any, ...], list[StepRequestState]] = {}
        order: list[tuple[Any, ...]] = []
        for state in states:
            key = QwenImage21Pipeline._decode_group_key(state)
            if key not in groups:
                order.append(key)
                groups[key] = []
            groups[key].append(state)
        return [groups[key] for key in order]

    @staticmethod
    def _needs_decode_seq_len_split(states: list["StepRequestState"]) -> bool:
        if len(states) <= 1:
            return False
        phases = {QwenImage21Pipeline._kv_cache_phase(state.extra.get("kv_cache")) for state in states}
        if phases != {"decode"}:
            return False
        return len({QwenImage21Pipeline._decode_group_key(state) for state in states}) > 1

    @staticmethod
    def _scatter_kv_cache(
        states: list["StepRequestState"],
        row_counts: list[int],
        kv_cache: list[dict],
    ) -> None:
        """Split a batched prefill cache back into the per-request caches.

        Clone (not ``.contiguous()``) so each request owns independent storage.
        Dim-0 slices of a contiguous merged prefill tensor are already contiguous
        views; keeping those views alive would pin the merged tensor and break
        the ``owner_prefix is not cached_key`` identity check in step-mode
        CUDA-graph decode (every decode would fall back to eager).
        """
        start = 0
        for state, rows in zip(states, row_counts):
            cache = state.extra["kv_cache"]
            for block_idx, block_cache in enumerate(kv_cache):
                for branch, parts in block_cache.items():
                    cache[block_idx][branch] = {
                        part: tensor[start : start + rows].clone() for part, tensor in parts.items()
                    }
            start += rows

    @staticmethod
    def _resolve_prompt_seq_len(
        states: list["StepRequestState"],
        current_seq_len: int,
        extra_key: str = "prefill_prompt_seq_len",
    ) -> int:
        """Prefer the padded VLM length baked into each request's KV at prefill."""
        seq_len = current_seq_len
        for state in states:
            stored = state.extra.get(extra_key)
            if stored is not None:
                seq_len = max(seq_len, int(stored))
        return seq_len

    @staticmethod
    def _pad_rows_to_seq_len(rows: list[torch.Tensor], seq_len: int) -> torch.Tensor:
        padded = []
        for row in rows:
            if row.shape[1] < seq_len:
                row = F.pad(row, (0, seq_len - row.shape[1]), value=False)
            elif row.shape[1] > seq_len:
                row = row[:, :seq_len]
            padded.append(row)
        return torch.cat(padded, dim=0)

    def denoise_step(
        self,
        input_batch: "InputBatch",
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """One denoise step: read from *input_batch*, delegate to CFGParallelMixin.

        Decode batches that mix different baked ``prefill_prompt_seq_len`` values
        (common when the mixed-phase gate forces separate prefills) are
        sub-batched by that length so both requests keep running.
        """
        del kwargs
        if self.interrupt:
            return None

        t = input_batch.timesteps
        self._current_timestep = t
        self.transformer.do_true_cfg = input_batch.do_true_cfg

        states = list(input_batch.states)
        if not states:
            raise ValueError("QwenImage21Pipeline.denoise_step requires per-request states on the InputBatch.")

        if self._needs_decode_seq_len_split(states):
            # Local import: InputBatch is only TYPE_CHECKING at module scope.
            from vllm_omni.diffusion.worker.input_batch import InputBatch as _InputBatch

            preds_by_id: dict[str, torch.Tensor] = {}
            for group in self._split_decode_groups(states):
                group_pred = self._denoise_step_homogeneous(_InputBatch.make_batch(group))
                if group_pred is None:
                    return None
                offset = 0
                for state in group:
                    rows = state.latents.shape[0]
                    preds_by_id[state.request_id] = group_pred[offset : offset + rows]
                    offset += rows
            return torch.cat([preds_by_id[state.request_id] for state in states], dim=0)

        return self._denoise_step_homogeneous(input_batch)

    def _denoise_step_homogeneous(
        self,
        input_batch: "InputBatch",
    ) -> torch.Tensor | None:
        """Run one denoise forward for a KV-compatible state group."""
        if self.interrupt:
            return None

        t = input_batch.timesteps
        self._current_timestep = t
        self.transformer.do_true_cfg = input_batch.do_true_cfg

        states = list(input_batch.states)
        if not states:
            raise ValueError("QwenImage21Pipeline.denoise_step requires per-request states on the InputBatch.")
        if self._kv_cache_phase(states[0].extra.get("kv_cache")) == "prefill":
            self._release_idle_decode_graphs()

        latents = input_batch.latents
        timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

        # Assemble the joint image-slot mask: right-pad each request's VLM mask
        # to the batch's padded prompt length, then append the target slots.
        # After a mixed-length prefill, each request's KV prefix stays at that
        # prefill's padded length — keep using it even if later decode batches
        # shrink (longest prompt finished) so mask / RoPE layout still match.
        max_seq_len = self._resolve_prompt_seq_len(states, input_batch.prompt_embeds.shape[1])
        img_mask = self._append_target_slots(
            self._pad_rows_to_seq_len([state.extra["image_pad_mask"] for state in states], max_seq_len),
            latents.shape[1],
        )
        prompt_embeds = input_batch.prompt_embeds
        prompt_embeds_mask = input_batch.prompt_embeds_mask
        if prompt_embeds is not None and prompt_embeds.shape[1] < max_seq_len:
            pad = max_seq_len - prompt_embeds.shape[1]
            prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, pad), value=0.0)
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = F.pad(prompt_embeds_mask, (0, pad), value=False)
        negative_img_mask = None
        negative_prompt_embeds = input_batch.negative_prompt_embeds
        negative_prompt_embeds_mask = input_batch.negative_prompt_embeds_mask
        negative_seq_len = 0
        if input_batch.do_true_cfg:
            negative_seq_len = self._resolve_prompt_seq_len(
                states,
                input_batch.negative_prompt_embeds.shape[1],
                extra_key="prefill_negative_prompt_seq_len",
            )
            negative_img_mask = self._append_target_slots(
                self._pad_rows_to_seq_len(
                    [state.extra["negative_image_pad_mask"] for state in states], negative_seq_len
                ),
                latents.shape[1],
            )
            if negative_prompt_embeds is not None and negative_prompt_embeds.shape[1] < negative_seq_len:
                pad = negative_seq_len - negative_prompt_embeds.shape[1]
                negative_prompt_embeds = F.pad(negative_prompt_embeds, (0, 0, 0, pad), value=0.0)
                if negative_prompt_embeds_mask is not None:
                    negative_prompt_embeds_mask = F.pad(negative_prompt_embeds_mask, (0, pad), value=False)

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
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
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
                "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                "encoder_hidden_states": negative_prompt_embeds,
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
            # Remember the padded VLM length baked into the per-request KV so a
            # later smaller decode batch can rebuild matching masks.
            for state in states:
                state.extra["prefill_prompt_seq_len"] = max_seq_len
                if input_batch.do_true_cfg:
                    state.extra["prefill_negative_prompt_seq_len"] = negative_seq_len
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

    def step_output(self, state: "StepRequestState", **kwargs: Any) -> DiffusionOutput:
        output = super().step_output(state, **kwargs)
        output.custom_output["qwen_image_21_metadata"] = {
            "height": state.extra["height"],
            "width": state.extra["width"],
            "input_image_sizes": [],
        }
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:
        if not self.weights_sources:
            # Encoder/decoder-only workers loaded their components through
            # from_pretrained; there is no streamed checkpoint to account for.
            return None
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native vLLM-Omni pipeline for Boogu-Image-0.1.

Ported from the upstream ``boogu`` package
(``boogu/pipelines/boogu/pipeline_boogu.py``) with the following changes:

- Diffusers ``DiffusionPipeline``/``register_modules`` machinery replaced by a
  plain ``nn.Module`` constructed from ``OmniDiffusionConfig`` (components are
  loaded from the checkpoint subfolders; transformer weights arrive later via
  ``weights_sources`` + ``load_weights``).
- Upstream ``encode_instruction`` is exposed as ``encode_prompt`` (the
  vLLM-Omni convention, also hooked by the prompt-embed cache); only the
  text-to-image path is ported. Instruction rewriting, prompt tuning,
  reference-image (TI2I/I2I) encoding, double guidance, and vision-token
  stripping are not ported.
- ``forward()`` (denoise loop + VAE decode) is intentionally not implemented
  yet; it lands together with the post-process registration.
"""

import json
import os
from collections.abc import Iterable
from typing import ClassVar, cast

import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
    BooguImageDoubleStreamRotaryPosEmbed,
    BooguImageTransformer2DModel,
)
from vllm_omni.diffusion.models.boogu_image.image_processor import BooguImageProcessor
from vllm_omni.diffusion.models.boogu_image.scheduling_flow_match_euler_discrete_time_shifting import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)

# Reference-image preprocessing limits (upstream ``BooguImagePipeline.__call__``
# defaults). The VLM copy is aggressively downscaled for the Qwen3VL encoder;
# the VAE copy keeps near-native resolution for the reference latents.
_MAX_VLM_INPUT_PIL_PIXELS = 384 * 384
_MAX_VLM_INPUT_PIL_SIDE_LENGTH = 384 * 2
_MAX_INPUT_IMAGE_PIXELS = 2048 * 2048
_MAX_INPUT_IMAGE_SIDE_LENGTH = 2048 * 2


def _load_vae_scale_factor(model_path: str) -> int:
    vae_config_path = os.path.join(model_path, "vae/config.json")
    try:
        with open(vae_config_path) as f:
            vae_config = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to load Boogu VAE config from {vae_config_path}: {exc}") from exc

    if "block_out_channels" not in vae_config:
        return 8
    return 2 ** (len(vae_config["block_out_channels"]) - 1)


def get_boogu_image_post_process_func(od_config: OmniDiffusionConfig):
    """Build the post-process callable that converts decoded tensors to images.

    Upstream ``BooguImageProcessor`` only customizes *pre*-processing; the
    ``postprocess`` path is inherited from the stock diffusers
    ``VaeImageProcessor``, so we reuse it directly here.
    """
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_scale_factor = _load_vae_scale_factor(model_path)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


def get_boogu_image_pre_process_func(od_config: OmniDiffusionConfig):
    """Build the pre-process callable for Boogu-Image reference (edit) input.

    Text-to-image requests carry no image and are passed through unchanged (the
    Base checkpoint shares this pipeline class). Edit (TI2I) requests carry a
    single reference PIL image on ``prompt["multi_modal_data"]["image"]``; it is
    resized twice — once for the Qwen3VL encoder (``prompt_image``) and once for
    the VAE reference latents (``preprocessed_image``) — and stashed in
    ``additional_information`` for ``forward`` to consume. Mirrors upstream
    ``preprocess_vlm_input_pil_images`` + ``prepare_image``.

    For a single reference image, upstream ``align_res`` (default ``True``)
    derives the output resolution from the VAE-encoded reference dimensions, so
    the request height/width are overwritten accordingly.
    """
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_scale_factor = _load_vae_scale_factor(model_path)

    # Upstream builds ``BooguImageProcessor(vae_scale_factor=vae_scale_factor*2)``
    # so all resize targets align to multiples of ``vae_scale_factor * 2``.
    image_processor = BooguImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_resize=True)

    def pre_process_func(request: OmniDiffusionRequest):
        prompt = request.prompt
        if isinstance(prompt, str):
            # Plain-text prompt cannot carry an image -> text-to-image, no-op.
            return request

        multi_modal_data = prompt.get("multi_modal_data") or {}
        raw_image = multi_modal_data.get("image")
        if not raw_image:
            # No reference image -> text-to-image, no-op (Base checkpoint).
            return request

        if isinstance(raw_image, list):
            if len(raw_image) > 1:
                raise ValueError(f"Boogu-Image editing supports a single reference image; received {len(raw_image)}.")
            raw_image = raw_image[0]

        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image)
        else:
            image = cast(PIL.Image.Image, raw_image)
        image = image.convert("RGB")

        if "additional_information" not in prompt:
            prompt["additional_information"] = {}

        # VLM-resized copy (PIL) for the Qwen3VL instruction encoder.
        vlm_height, vlm_width = image_processor.get_new_height_width(
            image, None, None, _MAX_VLM_INPUT_PIL_PIXELS, _MAX_VLM_INPUT_PIL_SIDE_LENGTH
        )
        prompt_image = image_processor.resize(image, vlm_height, vlm_width)

        # VAE-ready copy (normalized [1, C, H, W] tensor) for reference latents.
        preprocessed_image = image_processor.preprocess(
            image, max_pixels=_MAX_INPUT_IMAGE_PIXELS, max_side_length=_MAX_INPUT_IMAGE_SIDE_LENGTH
        )

        # align_res: single-image output resolution follows the reference dims.
        request.sampling_params.height = int(preprocessed_image.shape[-2])
        request.sampling_params.width = int(preprocessed_image.shape[-1])

        prompt["additional_information"]["preprocessed_image"] = preprocessed_image
        prompt["additional_information"]["prompt_image"] = prompt_image
        request.prompt = prompt
        return request

    return pre_process_func


# System prompts matching upstream dataset logic (ported verbatim from
# ``BooguImagePipeline.__init__``).
SYSTEM_PROMPT_4_TI2I_UNIFIED = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
)
SYSTEM_PROMPT_4_T2I_UNIFIED = (
    "You are a helpful assistant that generates high-quality images based on user instructions. "
    "The instructions are as follows."
)


class BooguImagePipeline(nn.Module, ProgressBarMixin, SupportsComponentDiscovery, SupportImageInput):
    """Boogu-Image text-to-image and image-editing (TI2I) pipeline.

    Native vLLM-Omni implementation. A request with a reference image (edit /
    TI2I) is served by the same class as text-to-image; the reference latents
    and Qwen3VL image tokens are threaded through ``forward`` and the ported
    transformer's reference-image refiner path.
    """

    supports_request_batch = False

    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["mllm"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self._raise_unsupported_features()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # See ``hub_prefetch.py`` for the transformers v5 multi-worker subfolder
        # race; prefetch the whole component set before any from_pretrained.
        boogu_subfolders = ["scheduler", "vae", "mllm", "processor"]
        prefetch_subfolders(model, boogu_subfolders, local_files_only=local_files_only)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        mllm = from_pretrained_with_prefetch(
            Qwen3VLForConditionalGeneration.from_pretrained,
            model,
            subfolder="mllm",
            prefetch_list=boogu_subfolders,
            local_files_only=local_files_only,
            torch_dtype=od_config.dtype,
        )
        # Upstream reuses the full VLM as an optional instruction rewriter and
        # encodes with its inner model (no ``lm_head``); the rewriter is not
        # ported, so keep only the inner ``Qwen3VLModel`` as the encoder.
        if hasattr(mllm, "lm_head"):
            mllm = mllm.model
        self.mllm = mllm.to(self._execution_device)

        self.processor = Qwen3VLProcessor.from_pretrained(
            model, subfolder="processor", local_files_only=local_files_only
        )

        self.vae = from_pretrained_with_prefetch(
            AutoencoderKL.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=boogu_subfolders,
            local_files_only=local_files_only,
        ).to(self._execution_device)

        self.transformer = BooguImageTransformer2DModel(od_config=od_config)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.default_sample_size = 128

        self.SYSTEM_PROMPT_4_T2I = SYSTEM_PROMPT_4_T2I_UNIFIED
        # Upstream uses the TI2I prompt for empty instructions (the default
        # negative prompt "" hits this path).
        self.SYSTEM_PROMPT_DROP = SYSTEM_PROMPT_4_TI2I_UNIFIED
        # Edit (TI2I / I2I) system prompts (image present in the chat template).
        self.SYSTEM_PROMPT_4_TI2I = SYSTEM_PROMPT_4_TI2I_UNIFIED
        self.SYSTEM_PROMPT_4_I2I = SYSTEM_PROMPT_4_TI2I_UNIFIED

    def _raise_unsupported_features(self) -> None:
        """Reject execution modes that do not have Boogu-specific support."""
        parallel_config = self.od_config.parallel_config
        if parallel_config.tensor_parallel_size > 1:
            raise NotImplementedError("Tensor parallelism is not supported by BooguImagePipeline.")
        if (parallel_config.sequence_parallel_size or 1) > 1:
            raise NotImplementedError("Sequence parallelism is not supported by BooguImagePipeline.")
        if parallel_config.cfg_parallel_size > 1:
            raise NotImplementedError("CFG parallelism is not supported by BooguImagePipeline.")
        if parallel_config.use_hsdp:
            raise NotImplementedError("HSDP is not supported by BooguImagePipeline.")
        if self.od_config.cache_backend not in (None, "", "none"):
            raise NotImplementedError(
                f"Cache backend '{self.od_config.cache_backend}' is not supported by BooguImagePipeline."
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    # ------------------------------------------------------------------
    # Prompt encoding (upstream ``encode_instruction``, t2i path)
    # ------------------------------------------------------------------

    def _apply_chat_template(
        self,
        instruction: str,
        input_pil_images: list[PIL.Image.Image] | None = None,
    ) -> list[dict]:
        """Build the chat messages for one instruction (text-to-image or edit).

        Mirrors upstream ``_apply_chat_template`` (``system_prompt_follows_task_type``
        is always ``False`` here): the system prompt is picked by whether images
        are present and whether the instruction is empty, and reference images
        are placed *before* the instruction text in the user turn.
        """
        user_text_content = [{"type": "text", "text": instruction}]

        has_images = input_pil_images is not None and len(input_pil_images) > 0
        instruction_empty = instruction is None or len(instruction.strip()) == 0

        if not has_images:
            system_prompt = self.SYSTEM_PROMPT_DROP if instruction_empty else self.SYSTEM_PROMPT_4_T2I
        else:
            system_prompt = self.SYSTEM_PROMPT_4_I2I if instruction_empty else self.SYSTEM_PROMPT_4_TI2I

        system_role = {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        if not has_images:
            return [system_role, {"role": "user", "content": user_text_content}]

        images_content = [{"type": "image", "image": pil_img} for pil_img in input_pil_images]
        return [system_role, {"role": "user", "content": images_content + user_text_content}]

    def _get_instruction_feature_embeds(
        self,
        instruction: str | list[str],
        input_pil_images: list[list[PIL.Image.Image] | None] | None = None,
        device: torch.device | None = None,
        max_sequence_length: int = 256,
        truncate_instruction_sequence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode instructions (and optional reference images) with Qwen3VL.

        ``input_pil_images`` is a per-sample list (outer length == batch size);
        each entry is the sample's already-VLM-resized reference images or
        ``None``. Returns the last hidden state (or the last-N layers as a list
        when the transformer config asks for more than one) and the attention
        mask.
        """
        device = device or self._execution_device
        instruction = [instruction] if isinstance(instruction, str) else instruction

        if input_pil_images is None:
            per_sample_images: list[list[PIL.Image.Image] | None] = [None] * len(instruction)
        else:
            assert len(input_pil_images) == len(instruction), (
                "`input_pil_images` outer length must match the instruction batch size."
            )
            per_sample_images = input_pil_images

        prompts = [self._apply_chat_template(text, per_sample_images[i]) for i, text in enumerate(instruction)]

        vlm_inputs = self.processor.apply_chat_template(
            prompts,
            padding="longest",
            max_length=max_sequence_length,
            truncation=truncate_instruction_sequence,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        for k in vlm_inputs.keys():
            if isinstance(vlm_inputs[k], torch.Tensor):
                vlm_inputs[k] = vlm_inputs[k].to(device)

        final_instruction_mask = vlm_inputs["attention_mask"]

        num_instruction_feature_layers = self.transformer.instruction_feature_configs.get(
            "num_instruction_feature_layers", 1
        )

        with torch.no_grad():
            text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)
            if num_instruction_feature_layers > 1:
                instruction_feats = list(text_encoder_outputs.hidden_states)[-num_instruction_feature_layers:]
            else:
                instruction_feats = text_encoder_outputs.hidden_states[-1]

        dtype = self.mllm.dtype if self.mllm is not None else self.transformer.dtype

        if isinstance(instruction_feats, (list, tuple)):
            final_instruction_feats = [feat.to(dtype=dtype, device=device) for feat in instruction_feats]
        else:
            final_instruction_feats = instruction_feats.to(dtype=dtype, device=device)
        final_instruction_mask = final_instruction_mask.to(device=device)

        return final_instruction_feats, final_instruction_mask

    def _reshape_embeds_and_mask(self, embeds, mask, num_images_per_prompt: int):
        """Duplicate embeddings/mask for each generation per prompt (mps-friendly)."""
        if isinstance(embeds, (list, tuple)):
            batch_size, seq_len, _ = embeds[0].shape
            reshaped_embeds = []
            for embed in embeds:
                embed = embed.repeat(1, num_images_per_prompt, 1)
                reshaped_embeds.append(embed.view(batch_size * num_images_per_prompt, seq_len, -1))
        else:
            batch_size, seq_len, _ = embeds.shape
            embeds = embeds.repeat(1, num_images_per_prompt, 1)
            reshaped_embeds = embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        mask = mask.repeat(num_images_per_prompt, 1)
        reshaped_mask = mask.view(batch_size * num_images_per_prompt, -1)

        return batch_size, seq_len, reshaped_embeds, reshaped_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        device: torch.device | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1280,
        truncate_instruction_sequence: bool = False,
        input_images: list[list[PIL.Image.Image] | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encode prompt (and negative prompt for CFG) into Qwen3VL hidden states.

        Port of upstream ``encode_instruction`` for text-to-image and the
        text-guided image-editing (TI2I) path. Reference images are attached to
        the *positive* instruction only (upstream default
        ``use_input_images_4_neg_instruct=False``). Instruction rewriting,
        prompt tuning, and double-guidance empty instructions are not ported.
        The default ``max_sequence_length`` matches the upstream ``__call__``
        default (1280), not the upstream ``encode_instruction`` default (256).

        Args:
            input_images: Per-sample list (outer length == batch size) of
                already-VLM-resized reference images, or ``None`` for pure
                text-to-image.

        Returns:
            ``(prompt_embeds, prompt_attention_mask, negative_prompt_embeds,
            negative_prompt_attention_mask)`` where each embeds tensor has shape
            ``[batch_size * num_images_per_prompt, seq_len, dim]``. The negative
            pair is ``None`` when ``do_classifier_free_guidance`` is off and no
            precomputed negative embeddings were passed.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_instruction_feature_embeds(
                instruction=prompt,
                input_pil_images=input_images,
                device=device,
                max_sequence_length=max_sequence_length,
                truncate_instruction_sequence=truncate_instruction_sequence,
            )

        batch_size, _, prompt_embeds, prompt_attention_mask = self._reshape_embeds_and_mask(
            prompt_embeds, prompt_attention_mask, num_images_per_prompt
        )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt if negative_prompt is not None else ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompt)}, but `prompt` has"
                    f" batch size {batch_size}. Please make sure that passed `negative_prompt`"
                    " matches the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_instruction_feature_embeds(
                instruction=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                truncate_instruction_sequence=truncate_instruction_sequence,
            )

            _, _, negative_prompt_embeds, negative_prompt_attention_mask = self._reshape_embeds_and_mask(
                negative_prompt_embeds, negative_prompt_attention_mask, num_images_per_prompt
            )

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    # ------------------------------------------------------------------
    # Denoise loop + VAE decode (upstream ``__call__`` / ``processing``, t2i)
    # ------------------------------------------------------------------

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        """Sample initial noise latents (upstream ``prepare_latents``)."""
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def _resolve_output_size(self, height, width):
        """t2i branch of upstream ``_resolve_output_and_original_size``.

        Clamps the working resolution to ``max_input_image_pixels`` (2048**2),
        rounding down to a multiple of ``vae_scale_factor * 2``; the requested
        size is remembered so the decoded image can be resized back.
        """
        img_scale_num = self.vae_scale_factor * 2
        ori_height, ori_width = height, width
        max_pixels = 2048 * 2048
        cur_pixels = height * width
        ratio = min((max_pixels / cur_pixels) ** 0.5, 1.0)
        height = int(height * ratio) // img_scale_num * img_scale_num
        width = int(width * ratio) // img_scale_num * img_scale_num
        return height, width, ori_height, ori_width

    def predict(
        self, t, latents, instruction_embeds, freqs_cis, instruction_attention_mask, ref_image_hidden_states=None
    ):
        """One transformer velocity prediction (upstream ``predict``).

        ``ref_image_hidden_states`` is ``None`` for text-to-image, or the
        per-sample reference latents (``list[list[Tensor[C, H, W]]]``) for the
        image-editing path.
        """
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        return self.transformer(
            latents,
            timestep,
            instruction_embeds,
            freqs_cis,
            instruction_attention_mask,
            ref_image_hidden_states=ref_image_hidden_states,
        )

    def _encode_vae_image(self, img: torch.Tensor, generator=None) -> torch.Tensor:
        """Encode an image tensor into the VAE latent space (upstream ``encode_vae``).

        Upstream leaves ``latent_dist.sample()`` unseeded; the native path
        threads the request generator through so a fixed seed gives a
        reproducible reference latent.
        """
        z0 = self.vae.encode(img.to(dtype=self.vae.dtype)).latent_dist.sample(generator=generator)
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        return z0.to(dtype=self.vae.dtype)

    def _build_ref_latents(
        self,
        preprocessed_images: list[torch.Tensor | None],
        num_images_per_prompt: int,
        device: torch.device,
        generator=None,
    ) -> list[list[torch.Tensor] | None]:
        """VAE-encode per-sample reference images into the transformer's format.

        Mirrors upstream ``prepare_image``: returns a list of length
        ``batch_size * num_images_per_prompt`` where each entry is either
        ``None`` (no reference / text-to-image) or a list of ``[C, H, W]``
        reference latents (one per reference image). Boogu editing uses a single
        reference image, so each non-empty entry is a one-element list.
        """
        # ``latent_dist.sample`` accepts only a single generator; a per-output
        # generator list (num_outputs_per_prompt > 1) falls back to unseeded.
        vae_generator = generator if isinstance(generator, torch.Generator) else None

        ref_latents: list[list[torch.Tensor] | None] = []
        for image in preprocessed_images:
            if image is None:
                sample_latents: list[torch.Tensor] | None = None
            else:
                latent = self._encode_vae_image(image.to(device=device), generator=vae_generator).squeeze(0)
                sample_latents = [latent]
            for _ in range(num_images_per_prompt):
                ref_latents.append(sample_latents)
        return ref_latents

    @staticmethod
    def _extract_reference_images(
        prompts: list,
    ) -> tuple[list[PIL.Image.Image | None], list[torch.Tensor | None]]:
        """Pull per-sample reference images out of ``additional_information``.

        Returns ``(prompt_images, preprocessed_images)`` where entries are
        ``None`` for pure text-to-image samples. Populated by
        :func:`get_boogu_image_pre_process_func`.
        """
        prompt_images: list[PIL.Image.Image | None] = []
        preprocessed_images: list[torch.Tensor | None] = []
        for p in prompts:
            if isinstance(p, str):
                prompt_images.append(None)
                preprocessed_images.append(None)
                continue
            ai = p.get("additional_information") or {}
            prompt_images.append(ai.get("prompt_image"))
            preprocessed_images.append(ai.get("preprocessed_image"))
        return prompt_images, preprocessed_images

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        # Prompt / negative-prompt extraction (mirrors the Ovis pattern; the
        # online API sometimes passes ``{"negative_prompt": None}``).
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts]
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        else:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        # Reference (edit / TI2I) images, if any.
        prompt_images, preprocessed_images = self._extract_reference_images(req.prompts)
        has_reference = any(img is not None for img in preprocessed_images)
        task_type = "ti2i" if has_reference else "t2i"

        sp = req.sampling_params
        device = self._execution_device

        height = sp.height or self.default_sample_size * self.vae_scale_factor
        width = sp.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sp.num_inference_steps or 50
        # Upstream default text guidance is 4.0; the engine coerces an unset
        # guidance_scale to 1.0, so only honor a caller-provided value.
        text_guidance_scale = sp.guidance_scale if sp.guidance_scale_provided else 4.0
        # Image guidance rides on ``guidance_scale_2`` (upstream default 1.0 =
        # off); only a caller-provided value enables the double-guidance path.
        image_guidance_scale = sp.guidance_scale_2 if sp.guidance_scale_2_provided else 1.0
        if not has_reference:
            image_guidance_scale = 1.0
        num_images_per_prompt = sp.num_outputs_per_prompt if sp.num_outputs_per_prompt > 0 else 1
        generator = sp.generator
        max_sequence_length = sp.max_sequence_length or 1280
        output_type = sp.output_type or "pil"
        cfg_range = (0.0, 1.0)

        # Negative instruction embeddings are needed whenever text guidance is
        # active (t2i text CFG, ti2i text-only, and ti2i double guidance).
        do_classifier_free_guidance = text_guidance_scale > 1.0

        batch_size = len(prompt)

        # Per-sample VLM reference images for the positive instruction only
        # (upstream default ``use_input_images_4_neg_instruct=False``).
        input_images = None
        if has_reference:
            input_images = [[img] if img is not None else None for img in prompt_images]

        # 1. Encode prompts.
        (
            instruction_embeds,
            instruction_attention_mask,
            negative_instruction_embeds,
            negative_instruction_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            input_images=input_images,
        )

        # 2. Resolve working / output resolution.
        height, width, ori_height, ori_width = self._resolve_output_size(height, width)

        # 3. Reference latents (edit path) and initial noise latents.
        dtype = self.vae.dtype
        latent_channels = self.transformer.in_channels
        ref_latents = None
        if has_reference:
            ref_latents = self._build_ref_latents(preprocessed_images, num_images_per_prompt, device, generator)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            instruction_embeds.dtype,
            device,
            generator,
        )

        freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(
            self.transformer.axes_dim_rope,
            self.transformer.axes_lens,
            theta=10000,
        )

        # 4. Timesteps (the ported scheduler consumes ``num_tokens``).
        num_tokens = latents.shape[-2] * latents.shape[-1]
        self.scheduler.set_timesteps(num_inference_steps, device=device, num_tokens=num_tokens)
        timesteps = self.scheduler.timesteps
        num_timesteps = len(timesteps)

        # 5. Denoise loop with (sequential) classifier-free guidance. Reproduces
        # the branch priority of upstream ``processing`` (double > text-only >
        # image-only > t2i text). Reference latents are kept in the conditional
        # (and, for text-only ti2i, the unconditional) predictions.
        with self.progress_bar(total=num_timesteps) as progress_bar:
            for i, t in enumerate(timesteps):
                in_cfg_range = cfg_range[0] <= i / num_timesteps <= cfg_range[1]
                text_gs = text_guidance_scale if in_cfg_range else 1.0
                image_gs = image_guidance_scale if in_cfg_range else 1.0

                model_pred = self.predict(
                    t, latents, instruction_embeds, freqs_cis, instruction_attention_mask, ref_latents
                )

                if task_type == "ti2i" and text_gs > 1.0 and image_gs > 1.0:
                    # Double guidance: 3 predictions (cond+ref, neg+ref, neg+no-ref).
                    model_pred_drop_text = self.predict(
                        t,
                        latents,
                        negative_instruction_embeds,
                        freqs_cis,
                        negative_instruction_attention_mask,
                        ref_latents,
                    )
                    model_pred_drop_all = self.predict(
                        t,
                        latents,
                        negative_instruction_embeds,
                        freqs_cis,
                        negative_instruction_attention_mask,
                        None,
                    )
                    delta_text = model_pred - model_pred_drop_text
                    delta_image = model_pred_drop_text - model_pred_drop_all
                    model_pred = model_pred + (text_gs - 1) * delta_text + (image_gs - 1) * delta_image
                elif task_type == "ti2i" and text_gs > 1.0:
                    # Text-only ti2i guidance: reference kept in the uncond pred.
                    model_pred_drop_text = self.predict(
                        t,
                        latents,
                        negative_instruction_embeds,
                        freqs_cis,
                        negative_instruction_attention_mask,
                        ref_latents,
                    )
                    model_pred = model_pred + (text_gs - 1) * (model_pred - model_pred_drop_text)
                elif task_type == "ti2i" and image_gs > 1.0:
                    # Image-only ti2i guidance: drop the reference in the uncond pred.
                    model_pred_drop_image = self.predict(
                        t, latents, instruction_embeds, freqs_cis, instruction_attention_mask, None
                    )
                    model_pred = model_pred + (image_gs - 1) * (model_pred - model_pred_drop_image)
                elif text_gs > 1.0:
                    # Text-to-image classifier-free guidance.
                    model_pred_drop_all = self.predict(
                        t,
                        latents,
                        negative_instruction_embeds,
                        freqs_cis,
                        negative_instruction_attention_mask,
                        None,
                    )
                    model_pred = model_pred + (text_gs - 1) * (model_pred - model_pred_drop_all)

                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]
                latents = latents.to(dtype=instruction_embeds.dtype)
                progress_bar.update()

        # 6. Decode.
        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(dtype=dtype)
            if self.vae.config.scaling_factor is not None:
                latents = latents / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                latents = latents + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            if (ori_height, ori_width) != (height, width):
                image = F.interpolate(image, size=(ori_height, ori_width), mode="bilinear")

        return DiffusionOutput(output=image)

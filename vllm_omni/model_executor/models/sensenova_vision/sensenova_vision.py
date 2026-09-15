# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova-Vision-7B-MoT omni model.

SenseNova-Vision is a fork of Bagel with identical parameter-bearing modules.
This class reuses the MoT/ViT/VAE embedding logic from the BAGEL integration
and only overrides the SenseNovaVision checkpoint defaults plus the official
VAE/ViT resize chain (``ImageTransform(1024, 512, 16)`` then
``ImageTransform(980, 224, 14)``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
from vllm.transformers_utils.processors.bagel import BagelProcessorKwargs

from vllm_omni.model_executor.models.bagel.bagel import (
    Img2ImgProcessorItems,
    OmniBagelDummyInputsBuilder,
    OmniBagelForConditionalGeneration,
    OmniBagelMultiModalProcessor,
    OmniBagelProcessingInfo,
    OmniBagelProcessor,
)

logger = init_logger(__name__)

# Official SenseNova-Vision VAE image transform, transcribed from the upstream
# ``ImageTransform(1024, 512, 16)`` (``sensenova_vision.py`` ``vae_transform``).
SENSENOVA_VISION_VAE_MAX_SIZE = 1024
SENSENOVA_VISION_VAE_MIN_SIZE = 512
SENSENOVA_VISION_VAE_STRIDE = 16

# Official SenseNova-Vision ViT image transform (``ImageTransform(980, 224, 14)``).
# Upstream applies this to the already-VAE-resized image.
SENSENOVA_VISION_VIT_MAX_SIZE = 980
SENSENOVA_VISION_VIT_MIN_SIZE = 224
SENSENOVA_VISION_VIT_STRIDE = 14
SENSENOVA_VISION_VIT_MAX_PIXELS = 14 * 14 * 9 * 1024

# Per-task image target side for recon3d generation output grid.
RECON3D_VAE_SIDE = 512
RECON3D_DEFAULT_NUM_VIEWS = 4

# SenseNova-Vision-7B-MoT defaults.
SENSENOVA_VISION_DEFAULT_LAYER_MODULE = "Qwen2MoTDecoderLayer"
SENSENOVA_VISION_DEFAULT_QK_NORM = True
SENSENOVA_VISION_DEFAULT_TIE_WORD_EMBEDDINGS = False
SENSENOVA_VISION_DEFAULT_VISUAL_GEN = True
SENSENOVA_VISION_DEFAULT_VISUAL_UND = True
SENSENOVA_VISION_DEFAULT_MAX_LATENT_SIZE = 64
SENSENOVA_VISION_DEFAULT_VIT_MAX_NUM_PATCH_PER_SIDE = 70


def _sensenova_vae_resize_dims(img_h: int, img_w: int) -> tuple[int, int]:
    """Stride-aligned ``(new_h, new_w)`` for ``ImageTransform(1024, 512, 16)``."""
    stride = SENSENOVA_VISION_VAE_STRIDE
    max_size = SENSENOVA_VISION_VAE_MAX_SIZE
    min_size = SENSENOVA_VISION_VAE_MIN_SIZE

    scale = min(max_size / max(img_h, img_w), 1.0)
    scale = max(scale, min_size / min(img_h, img_w))
    new_h = max(stride, int(round(img_h * scale / stride) * stride))
    new_w = max(stride, int(round(img_w * scale / stride) * stride))
    if max(new_h, new_w) > max_size:
        scale = max_size / max(new_h, new_w)
        new_h = max(stride, int(round(new_h * scale / stride) * stride))
        new_w = max(stride, int(round(new_w * scale / stride) * stride))
    return new_h, new_w


def _sensenova_make_divisible(value: int, stride: int) -> int:
    """Mirror ``MaxLongEdgeMinShortEdgeResize._make_divisible``."""
    return max(stride, int(round(value / stride)) * stride)


def _sensenova_vit_resize_dims(vae_h: int, vae_w: int) -> tuple[int, int]:
    """Stride-aligned ``(vit_h, vit_w)`` for ``ImageTransform(980, 224, 14)``.

    Input is the already-VAE-resized image, matching upstream
    ``update_context_image``.
    """
    max_size = SENSENOVA_VISION_VIT_MAX_SIZE
    min_size = SENSENOVA_VISION_VIT_MIN_SIZE
    stride = SENSENOVA_VISION_VIT_STRIDE

    def apply_scale(width: int, height: int, scale: float) -> tuple[int, int]:
        return (
            _sensenova_make_divisible(round(width * scale), stride),
            _sensenova_make_divisible(round(height * scale), stride),
        )

    scale = min(max_size / max(vae_h, vae_w), 1.0)
    scale = max(scale, min_size / min(vae_h, vae_w))
    vit_w, vit_h = apply_scale(vae_w, vae_h, scale)

    if vit_w * vit_h > SENSENOVA_VISION_VIT_MAX_PIXELS:
        shrink = SENSENOVA_VISION_VIT_MAX_PIXELS / (vit_w * vit_h)
        vit_w, vit_h = apply_scale(vit_w, vit_h, shrink)
    if max(vit_w, vit_h) > max_size:
        shrink = max_size / max(vit_w, vit_h)
        vit_w, vit_h = apply_scale(vit_w, vit_h, shrink)
    return vit_h, vit_w


def _sensenova_vit_patch_count(vae_h: int, vae_w: int) -> int:
    """Aspect-aware ViT patch count for a VAE-resized image."""
    vit_h, vit_w = _sensenova_vit_resize_dims(int(vae_h), int(vae_w))
    return (vit_h // SENSENOVA_VISION_VIT_STRIDE) * (vit_w // SENSENOVA_VISION_VIT_STRIDE)


def _sensenova_understanding_patch_count(img_h: int, img_w: int) -> int:
    """Aspect-aware ViT patch count for an original understanding image.

    Mirrors the model chain: VAE transform then ViT transform.
    """
    vae_h, vae_w = _sensenova_vae_resize_dims(int(img_h), int(img_w))
    return _sensenova_vit_patch_count(vae_h, vae_w)


def _sensenova_img2img_token_counts(img_h: int, img_w: int) -> tuple[int, int, int, int]:
    """Return ``(num_vae_total, num_vit_total, vae_h, vae_w)`` for one img2img item.

    Totals include the ``<|vision_start|>`` / ``<|vision_end|>`` marker slots.
    """
    vae_h, vae_w = _sensenova_vae_resize_dims(int(img_h), int(img_w))
    num_vae_patches = (vae_h // SENSENOVA_VISION_VAE_STRIDE) * (vae_w // SENSENOVA_VISION_VAE_STRIDE)
    num_vit_patches = _sensenova_vit_patch_count(vae_h, vae_w)
    return num_vae_patches + 2, num_vit_patches + 2, vae_h, vae_w


class OmniSenseNovaVisionProcessor(OmniBagelProcessor):
    """Pass original-res pixels for both understanding and img2img.

    The AR model applies the official VAE then ViT resize chain itself.
    """

    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __call__(self, text=None, images=None, **kwargs):
        is_img2img = kwargs.pop("is_img2img", False)
        if images is not None:
            output_kwargs = self._merge_kwargs(
                BagelProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
                **kwargs,
            )
            image_kwargs = dict(output_kwargs["images_kwargs"])
            image_kwargs["do_resize"] = False
            image_kwargs["do_rescale"] = True
            image_kwargs.setdefault("return_tensors", "pt")
            pixel_values = self.image_processor(images, **image_kwargs)

            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"]) if text is not None else None
            if pixel_values is not None and text_inputs is not None:
                combined = dict(text_inputs)
                combined["pixel_values"] = pixel_values["pixel_values"]
                return BatchFeature(combined)
            if pixel_values is not None:
                return pixel_values
            if text_inputs is not None:
                return BatchFeature(dict(text_inputs))
            return BatchFeature({})

        return super().__call__(text, images, is_img2img=is_img2img, **kwargs)


class OmniSenseNovaVisionProcessingInfo(OmniBagelProcessingInfo):
    """Multi-modal limits for SenseNova-Vision."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 10, "img2img": 10}

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(OmniSenseNovaVisionProcessor, **kwargs)


class OmniSenseNovaVisionMultiModalProcessor(OmniBagelMultiModalProcessor):
    """Placeholder sizing locked to ``_sensenova_*_resize_dims``."""

    def _mm_kwargs_for_bagel_img2img_hf(self, mm_kwargs):
        # Preserve target_h/target_w for recon3d view sizing.
        return dict(mm_kwargs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        tokenizer = self.info.get_tokenizer()
        replacements: list[PromptReplacement] = []

        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
        if image_token_id is not None:

            def get_image_replacement(item_idx: int):
                size = mm_items.get_items("image", ImageProcessorItems).get_image_size(item_idx)
                # Markers + ViT patches after the official VAE->ViT chain.
                count = _sensenova_understanding_patch_count(int(size.height), int(size.width)) + 2
                return [image_token_id] * count

            replacements.append(
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_image_replacement,
                )
            )

        img2img_token_id = tokenizer.get_vocab().get("<|fim_middle|>")
        if img2img_token_id is not None:
            vit_config = self.info.get_hf_config().vit_config
            default_h = default_w = int(vit_config.image_size)

            def get_img2img_replacement(item_idx: int):
                h, w = default_h, default_w
                if "img2img" in mm_items:
                    item = mm_items.get_items("img2img", (Img2ImgProcessorItems, ImageEmbeddingItems))
                    if hasattr(item, "get_image_size"):
                        size = item.get_image_size(item_idx)
                        h, w = int(size.height), int(size.width)

                num_vae_total, num_vit_total, _, _ = _sensenova_img2img_token_counts(h, w)
                # Keep BAGEL's separator so extract_embeds_range() yields two
                # distinct mm_prefix ranges (VAE vs ViT) for M-RoPE / MoT.
                total = num_vae_total + 1 + num_vit_total
                tokens = [img2img_token_id] * total
                embed_mask = [True] * num_vae_total + [False] + [True] * num_vit_total
                return PromptUpdateDetails(
                    full=tokens,
                    is_embed=lambda _tok, _seq, _m=embed_mask: torch.tensor(_m, dtype=torch.bool),
                )

            replacements.append(
                PromptReplacement(
                    modality="img2img",
                    target=[img2img_token_id],
                    replacement=get_img2img_replacement,
                )
            )

        return replacements


@MULTIMODAL_REGISTRY.register_processor(
    OmniSenseNovaVisionMultiModalProcessor,
    info=OmniSenseNovaVisionProcessingInfo,
    dummy_inputs=OmniBagelDummyInputsBuilder,
)
class OmniSenseNovaVisionForConditionalGeneration(OmniBagelForConditionalGeneration):
    """SenseNova-Vision-7B-MoT omni model with official VAE/ViT resize dims."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        from vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision import (
            register_vllm_sensenova_vision_tokenizer,
        )

        register_vllm_sensenova_vision_tokenizer()
        config = vllm_config.model_config.hf_config
        self._apply_sensenova_vision_config_defaults(config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        tok = getattr(self, "_probe_tokenizer", None) or getattr(self, "tokenizer", None)
        if tok is not None:
            self._img2text_token_id = int(tok.convert_tokens_to_ids("<|image_pad|>"))
        else:
            self._img2text_token_id = -1

    @staticmethod
    def _apply_sensenova_vision_config_defaults(config) -> None:
        """Force SenseNovaVision checkpoint defaults on the HF config in place."""
        config.visual_gen = SENSENOVA_VISION_DEFAULT_VISUAL_GEN
        config.visual_und = SENSENOVA_VISION_DEFAULT_VISUAL_UND
        config.max_latent_size = SENSENOVA_VISION_DEFAULT_MAX_LATENT_SIZE
        config.vit_max_num_patch_per_side = SENSENOVA_VISION_DEFAULT_VIT_MAX_NUM_PATCH_PER_SIDE

        llm_config = config.llm_config
        llm_config.layer_module = SENSENOVA_VISION_DEFAULT_LAYER_MODULE
        llm_config.qk_norm = SENSENOVA_VISION_DEFAULT_QK_NORM
        llm_config.tie_word_embeddings = SENSENOVA_VISION_DEFAULT_TIE_WORD_EMBEDDINGS

    def get_raw_latent(self) -> None:
        """AR stage produces KV caches, never latents."""
        return None

    def _resize_to_stride(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Resize to the official SenseNova-Vision VAE grid.

        Overrides BAGEL's short-edge floor of ``min(256, max)`` with
        ``ImageTransform(1024, 512, 16)``.
        """
        h, w = pixel_values.shape[2], pixel_values.shape[3]
        new_h, new_w = _sensenova_vae_resize_dims(h, w)
        if new_h != h or new_w != w:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(new_h, new_w), mode="bicubic", align_corners=False
            )
        return pixel_values

    def _resize_for_vit(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Resize an already-VAE-resized image with ``ImageTransform(980, 224, 14)``."""
        h, w = pixel_values.shape[2], pixel_values.shape[3]
        vit_h, vit_w = _sensenova_vit_resize_dims(h, w)
        if vit_h != h or vit_w != w:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(vit_h, vit_w), mode="bicubic", align_corners=False
            )
        return pixel_values

    def _process_img2text_input(self, multimodal_input) -> tuple[torch.Tensor, ...]:
        """Understanding path: VAE resize then ViT resize before SigLIP."""
        images = self._image_list(multimodal_input["pixel_values"])
        vit_images = []
        for img in images:
            vae_resized = self._resize_to_stride(img[None])
            vit_images.append(self._resize_for_vit(vae_resized)[0])

        marker_ids = torch.tensor([self._start_of_image_id, self._end_of_image_id], device=images[0].device)
        start, end = self.language_model.model.embed_tokens(marker_ids).split(1)
        return tuple(torch.cat([start.to(e.dtype), e, end.to(e.dtype)]) for e in self._vit_embeddings(vit_images))

    def _process_img2img_input(self, multimodal_input):
        """img2img path: VAE encode at VAE dims; ViT encode at ViT-of-VAE dims."""
        pixel_values = self._image_list(multimodal_input["pixel_values"])
        num_images = len(pixel_values)
        p = self.latent_patch_size
        timestep = 0

        if self._ropes_pending:
            self._ropes_pending.clear()

        # Upstream runs the ViT transform on the VAE-transformed image.
        vae_resized = [self._resize_to_stride(pv[None]) for pv in pixel_values]
        vit_embeddings_tuple = self._vit_embeddings([self._resize_for_vit(pv)[0] for pv in vae_resized])

        marker_ids = torch.tensor(
            [self._start_of_image_id, self._end_of_image_id],
            device=pixel_values[0].device,
        )
        start_embed, end_embed = self.language_model.model.embed_tokens(marker_ids).split(1)

        results = []
        for i in range(num_images):
            single_pv = vae_resized[i]
            h_px, w_px = single_pv.shape[2:]

            padded_latent = self.vae.encode(single_pv)
            h = h_px // self.latent_downsample
            w = w_px // self.latent_downsample

            latent = padded_latent[0][:, : h * p, : w * p]
            latent = latent.reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)

            vae_position_ids = self.get_flattened_position_ids(
                h_px,
                w_px,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            pos_embed = self.latent_pos_embed([vae_position_ids])
            packed_timesteps = torch.tensor([timestep], device=padded_latent.device)
            with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
                timestep_embeds = self.time_embedder(packed_timesteps.to(padded_latent))
            vae_embeds = self.vae2llm(latent) + timestep_embeds + pos_embed

            vit_emb = vit_embeddings_tuple[i] if i < len(vit_embeddings_tuple) else vit_embeddings_tuple[0]
            se = start_embed.to(vae_embeds.dtype)
            ee = end_embed.to(vae_embeds.dtype)
            combined = torch.cat([se, vae_embeds, ee, se, vit_emb, ee], dim=0)
            results.append(combined)

            num_vae = h * w + 2
            num_vit = vit_emb.shape[0] + 2
            info = (num_vae, num_vit, int(h_px), int(w_px))
            # Register in the pending list AND the cross-request size cache.  A
            # later request whose image the encoder/prefix cache serves (no
            # embed run) resolves its (H, W) via ``_match_img2img_info`` ->
            # ``_img2img_info_by_size``; without the size-cache entry the DiT
            # stage falls back to a square 1024x1024 output instead of the
            # aspect-preserving VAE dims.
            self._register_img2img_info(info)
            self._last_img2img_info = info

        return tuple(results)

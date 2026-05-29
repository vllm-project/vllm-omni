from collections.abc import Iterable, Mapping, Sequence
import logging
import os
from math import isqrt
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import ModalityData, MultiModalDataDict
from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bagel import BagelForConditionalGeneration
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2MLP
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
)
from vllm.transformers_utils.processors.bagel import BagelProcessor

from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.bagel.autoencoder import (
    AutoEncoderParams,
    DiagonalGaussian,
    Encoder,
)
from vllm_omni.diffusion.models.bagel.bagel_transformer import (
    PositionEmbedding,
    TimestepEmbedder,
    get_flattened_position_ids_extrapolate,
    patchify,
)
from vllm_omni.diffusion.models.bagel.pipeline_bagel import (
    SiglipNaViTWrapper,
    _ImageTransform,
    default_ae_params,
)
from vllm_omni.utils.bagel_vqa import (
    bagel_vqa_reference_layout_enabled,
    bagel_vqa_reference_prefill_enabled,
    build_bagel_vqa_image_spans,
    build_bagel_vqa_rope_positions,
)


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _bagel_img2img_vit_separator_enabled() -> bool:
    return _env_flag("BAGEL_IMG2IMG_VIT_SEPARATOR", False)


def _bagel_img2img_local_preprocess_enabled() -> bool:
    return _env_flag("BAGEL_IMG2IMG_LOCAL_PREPROCESS", False)


def _bagel_img2img_local_preprocess_vit_enabled() -> bool:
    return _env_flag("BAGEL_IMG2IMG_LOCAL_PREPROCESS_VIT", False)


def _bagel_img2img_debug_enabled() -> bool:
    return _env_flag("BAGEL_IMG2IMG_DEBUG", False)


def _bagel_force_input_img2img_vit_enabled() -> bool:
    input_vit = os.environ.get("BAGEL_FORCE_INPUT_IMG2IMG_VIT")
    if input_vit is not None and input_vit.strip() != "":
        return _env_flag("BAGEL_FORCE_INPUT_IMG2IMG_VIT", True)
    return _env_flag("BAGEL_FORCE_IMG2IMG_VIT", True)


_BAGEL_IMAGE_MAX_PIXELS = 14 * 14 * 9 * 1024
_BAGEL_FORCE_VAE_MAX_SIZE = 1024
_BAGEL_FORCE_VAE_MIN_SIZE = 512
_BAGEL_FORCE_VIT_MAX_SIZE = 980
_BAGEL_FORCE_VIT_MIN_SIZE = 224
_BAGEL_FORCE_VIT_STRIDE = 14


def _bagel_make_divisible(value: float, stride: int) -> int:
    return max(stride, int(round(value / stride) * stride))


def _bagel_apply_scale_to_stride(width: int, height: int, scale: float, stride: int) -> tuple[int, int]:
    return (
        _bagel_make_divisible(round(width * scale), stride),
        _bagel_make_divisible(round(height * scale), stride),
    )


def _bagel_reference_resize_hw(
    height: int,
    width: int,
    *,
    max_size: int,
    min_size: int,
    stride: int,
    max_pixels: int = _BAGEL_IMAGE_MAX_PIXELS,
    img_num: int = 1,
) -> tuple[int, int]:
    """Match BAGEL ImageTransform resize geometry, returning (H, W)."""
    scale = min(max_size / max(width, height), 1.0)
    scale = max(scale, min_size / min(width, height))
    new_width, new_height = _bagel_apply_scale_to_stride(width, height, scale, stride)

    if new_width * new_height > max_pixels / img_num:
        scale = max_pixels / img_num / (new_width * new_height)
        new_width, new_height = _bagel_apply_scale_to_stride(new_width, new_height, scale, stride)

    if max(new_width, new_height) > max_size:
        scale = max_size / max(new_width, new_height)
        new_width, new_height = _bagel_apply_scale_to_stride(new_width, new_height, scale, stride)

    return int(new_height), int(new_width)


def _bagel_force_img2img_vae_hw(
    height: int,
    width: int,
    *,
    latent_downsample: int,
    max_latent_size: int,
) -> tuple[int, int]:
    max_size = min(_BAGEL_FORCE_VAE_MAX_SIZE, int(max_latent_size * latent_downsample))
    min_size = min(_BAGEL_FORCE_VAE_MIN_SIZE, max_size)
    return _bagel_reference_resize_hw(
        height,
        width,
        max_size=max_size,
        min_size=min_size,
        stride=latent_downsample,
    )


def _bagel_force_img2img_vit_hw(height: int, width: int) -> tuple[int, int]:
    return _bagel_reference_resize_hw(
        height,
        width,
        max_size=_BAGEL_FORCE_VIT_MAX_SIZE,
        min_size=_BAGEL_FORCE_VIT_MIN_SIZE,
        stride=_BAGEL_FORCE_VIT_STRIDE,
    )


class OmniBagelProcessor(BagelProcessor):
    # transformers>=5.0 ProcessorMixin.get_attributes() only scans the leaf
    # class's __dict__ for ``<attribute>_class`` hints; redeclare them here
    # so from_pretrained() correctly sets ``self.image_processor`` and
    # ``self.tokenizer`` on the OmniBagelProcessor instance.
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # PATCH (navit-thinker-fix): Canonical aspect-preserving ViT transform for the
    # VQA / understanding path. Mirrors VLMEvalKit's ImageTransform(980, 224, 14) and
    # the pipeline_bagel.py DiT path, producing variable-shape ViT input the
    # checkpoint was trained on (NaViT-style packed sequences). Replaces the prior
    # fixed-980×980 SigLIP path that destroyed aspect ratio.
    _vit_transform_cls = _ImageTransform
    _vit_patch_size = 14
    _img2img_vae_stride = 16

    def _get_vit_transform(self) -> _ImageTransform:
        # Cached instance — stride-14 ladder, mean/std = 0.5 (matches canonical).
        cached = getattr(self, "_cached_vit_transform", None)
        if cached is None:
            cached = self._vit_transform_cls(980, 224, self._vit_patch_size)
            object.__setattr__(self, "_cached_vit_transform", cached)
        return cached

    def _get_img2img_vae_transform(self) -> _ImageTransform:
        # Local force_interleave first resizes source images with the VAE ladder,
        # then feeds that image to both VAE and ViT context updates.
        cached = getattr(self, "_cached_img2img_vae_transform", None)
        if cached is None:
            cached = self._vit_transform_cls(1024, 512, self._img2img_vae_stride)
            object.__setattr__(self, "_cached_img2img_vae_transform", cached)
        return cached

    def _get_img2img_vit_transform(self) -> _ImageTransform:
        cached = getattr(self, "_cached_img2img_vit_transform", None)
        if cached is None:
            cached = self._vit_transform_cls(980, 224, self._vit_patch_size)
            object.__setattr__(self, "_cached_img2img_vit_transform", cached)
        return cached

    @staticmethod
    def _to_image_list(images) -> list:
        if images is None:
            return []
        if isinstance(images, Image.Image):
            return [images]
        if isinstance(images, (list, tuple)):
            return list(images)
        # numpy / tensor inputs unsupported on this path; let parent handle.
        return list(images) if hasattr(images, "__iter__") else [images]

    def __call__(self, text=None, images=None, **kwargs):
        is_img2img = kwargs.pop("is_img2img", False)

        if is_img2img and images is not None:
            # transformers>=5.0 enforces strict kwarg typing on image
            # processors, so split generic kwargs into text/image buckets
            # via the standard ProcessorMixin helper before dispatch.
            from vllm.transformers_utils.processors.bagel import BagelProcessorKwargs

            output_kwargs = self._merge_kwargs(
                BagelProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
                **kwargs,
            )
            image_list = self._to_image_list(images)
            use_local_preprocess = _bagel_img2img_local_preprocess_enabled()
            precompute_local_vit = _bagel_img2img_local_preprocess_vit_enabled()
            pixel_values = None
            if use_local_preprocess and image_list and all(isinstance(img, Image.Image) for img in image_list):
                vae_transform = self._get_img2img_vae_transform()
                vit_transform = self._get_img2img_vit_transform()
                vae_tensors = []
                vit_chunks: list[torch.Tensor] = []
                vit_grid_rows: list[list[int]] = []
                patch_size = self._vit_patch_size
                for img in image_list:
                    img = img.convert("RGB") if img.mode != "RGB" else img
                    vae_img = vae_transform.resize_only(img)
                    vae_tensors.append(vae_transform(vae_img))

                    if precompute_local_vit:
                        vit_tensor = vit_transform(vae_img)
                        _, vit_h, vit_w = vit_tensor.shape
                        if vit_h % patch_size != 0 or vit_w % patch_size != 0:
                            pad_h = (-vit_h) % patch_size
                            pad_w = (-vit_w) % patch_size
                            vit_tensor = torch.nn.functional.pad(vit_tensor, (0, pad_w, 0, pad_h))
                            _, vit_h, vit_w = vit_tensor.shape
                        vit_chunks.append(patchify(vit_tensor, patch_size))
                        vit_grid_rows.append([1, vit_h // patch_size, vit_w // patch_size])

                if len({tuple(t.shape) for t in vae_tensors}) == 1:
                    data = {"pixel_values": torch.stack(vae_tensors, dim=0)}
                    if precompute_local_vit and vit_chunks:
                        data["pixel_values_img2img_vit"] = torch.cat(vit_chunks, dim=0)
                        data["image_grid_thw_img2img_vit"] = torch.tensor(
                            vit_grid_rows,
                            dtype=torch.long,
                        )
                    pixel_values = BatchFeature(data)

            if pixel_values is None:
                image_kwargs = dict(output_kwargs["images_kwargs"])
                image_kwargs["do_resize"] = False
                image_kwargs["do_rescale"] = True
                image_kwargs.setdefault("return_tensors", "pt")
                pixel_values = self.image_processor(images, **image_kwargs)

            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"]) if text is not None else None

            if pixel_values is not None and text_inputs is not None:
                combined = dict(text_inputs)
                for key, value in pixel_values.items():
                    combined[key] = value
                return BatchFeature(combined)
            elif pixel_values is not None:
                return pixel_values
            elif text_inputs is not None:
                return BatchFeature(dict(text_inputs))
            else:
                return BatchFeature({})

        # PATCH (navit-thinker-fix): VQA path — apply ImageTransform(980, 224, 14)
        # + patchify, emit packed pixel_values + image_grid_thw, so the model's
        # _process_image_input override can run a NaViT-style ViT.
        image_list = self._to_image_list(images)
        if image_list:
            from vllm.transformers_utils.processors.bagel import BagelProcessorKwargs

            output_kwargs = self._merge_kwargs(
                BagelProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
                **kwargs,
            )

            vit_transform = self._get_vit_transform()
            patch_size = self._vit_patch_size
            packed_chunks: list[torch.Tensor] = []
            grid_thw_rows: list[list[int]] = []
            for img in image_list:
                tensor = vit_transform(img)  # (3, H, W) in [-1, 1]
                _, H, W = tensor.shape
                if H % patch_size != 0 or W % patch_size != 0:
                    # ImageTransform's resize already aligns to stride=14, but
                    # be defensive: pad to multiple of patch_size if upstream
                    # ever changes.
                    pad_h = (-H) % patch_size
                    pad_w = (-W) % patch_size
                    tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
                    _, H, W = tensor.shape
                patches = patchify(tensor, patch_size)  # (L, 3 * p²)
                packed_chunks.append(patches)
                grid_thw_rows.append([1, H // patch_size, W // patch_size])

            packed_pixel_values = torch.cat(packed_chunks, dim=0)
            image_grid_thw = torch.tensor(grid_thw_rows, dtype=torch.long)

            text_inputs = (
                self.tokenizer(text, **output_kwargs["text_kwargs"]) if text is not None else {}
            )
            result = dict(text_inputs) if text_inputs else {}
            result["pixel_values"] = packed_pixel_values
            result["image_grid_thw"] = image_grid_thw
            return BatchFeature(result)

        # Text-only / no-image path: defer to parent (tokenizer-only).
        return super().__call__(text, images, **kwargs)


class OmniBagelProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        image_limit = 1
        raw_limit = os.environ.get("BAGEL_MAX_MULTIMODAL_IMAGE_INPUTS")
        if raw_limit:
            try:
                image_limit = max(1, int(raw_limit))
            except ValueError:
                image_limit = 1
        return {"image": image_limit, "img2img": 1}

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(OmniBagelProcessor, **kwargs)

    def get_hf_config(self):
        config = super().get_hf_config()
        if not getattr(self, "_latent_size_patched", False):
            self._latent_size_patched = True
            self._patch_max_latent_size(config)
        return config

    def _patch_max_latent_size(self, config):
        """Infer correct max_latent_size from the model's latent_pos_embed
        weight, since the HF config value may be stale (e.g. 32 vs 64)."""
        import json
        from pathlib import Path

        model_name = self.ctx.model_config.model
        try:
            p = Path(model_name)
            if p.is_dir():
                index_path = p / "model.safetensors.index.json"
            else:
                from huggingface_hub import hf_hub_download

                index_path = Path(hf_hub_download(model_name, "model.safetensors.index.json"))

            if not index_path.exists():
                return

            with open(index_path) as f:
                index = json.load(f)

            shard = index.get("weight_map", {}).get("latent_pos_embed.pos_embed")
            if not shard:
                return

            from safetensors import safe_open

            with safe_open(str(index_path.parent / shard), framework="pt") as f:
                if "latent_pos_embed.pos_embed" in f.keys():
                    npos = f.get_slice("latent_pos_embed.pos_embed").get_shape()[0]
                    side = isqrt(npos)
                    if side * side == npos:
                        old = getattr(config, "max_latent_size", 32)
                        if old != side:
                            config.max_latent_size = side
        except Exception:
            pass

    def get_data_parser(self) -> "OmniBagelDataParser":
        return OmniBagelDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class OmniBagelDummyInputsBuilder(BaseDummyInputsBuilder[OmniBagelProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        dummy_text = ""
        if "image" in mm_counts:
            dummy_text += "<|image_pad|>" * mm_counts["image"]
        if "img2img" in mm_counts:
            dummy_text += "<|fim_middle|>" * mm_counts["img2img"]
        return dummy_text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        hf_config = self.info.get_hf_config()
        vit_config = hf_config.vit_config

        image_size = vit_config.image_size
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "img2img": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=mm_counts.get("img2img", 0),
                overrides=image_overrides,
            ),
        }


class Img2ImgProcessorItems(ImageProcessorItems):
    def __init__(self, data):
        super().__init__(data)
        self.modality = "img2img"

    def get_processor_data(self):
        return {"pixel_values_img2img": self.get_all()}


class OmniBagelDataParser(MultiModalDataParser):
    def _parse_img2img_data(self, data: ModalityData) -> ModalityDataItems | None:
        items = self._parse_image_data(data)
        if items is None:
            return None
        return Img2ImgProcessorItems(items.data)

    def _get_subparsers(self):
        parsers = super()._get_subparsers()
        parsers["img2img"] = self._parse_img2img_data
        return parsers


class OmniBagelMultiModalProcessor(BaseMultiModalProcessor[OmniBagelProcessingInfo]):
    IMG2IMG_PLACEHOLDER = "<|fim_middle|>"

    @staticmethod
    def _mm_kwargs_for_bagel_img2img_hf(mm_kwargs: Mapping[str, object]) -> dict[str, object]:
        # OpenAI / GLM-style serving may pass target_h/target_w for output grid sizing.
        # BagelProcessor does not accept these in img2img mode; strip here so callers
        # (e.g. serving_chat) can stay model-agnostic.
        return {k: v for k, v in mm_kwargs.items() if k not in ("target_h", "target_w")}

    def _cached_apply_hf_processor(self, inputs, timing_ctx):
        # img2img: prompt text must be modified based on mm data presence,
        # so text and mm data cannot be tokenized separately — bypass cache.
        if inputs.mm_data_items.get_all_counts().get("img2img", 0) > 0:
            return self._apply_hf_processor(inputs, timing_ctx)
        return super()._cached_apply_hf_processor(inputs, timing_ctx)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        # PATCH (navit-thinker-fix): VQA pixel_values now arrive as packed
        # variable-shape tokens — slice with `flat_from_sizes` keyed off
        # `image_grid_thw[:, 1] * image_grid_thw[:, 2]`. img2img path still
        # produces uniform fixed-shape tensors.
        image_grid_thw = hf_inputs.get("image_grid_thw") if isinstance(hf_inputs, Mapping) else None
        if image_grid_thw is not None:
            image_pixel_grid_sizes = image_grid_thw[:, 1] * image_grid_thw[:, 2]
            pixel_values_cfg = MultiModalFieldConfig.flat_from_sizes("image", image_pixel_grid_sizes)
        else:
            # Fallback for warmup / dummy paths that may not populate image_grid_thw.
            pixel_values_cfg = MultiModalFieldConfig.batched("image")
        config = {
            "pixel_values": pixel_values_cfg,
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
            "pixel_values_img2img": MultiModalFieldConfig.batched("img2img"),
        }
        img2img_vit_grid_thw = (
            hf_inputs.get("image_grid_thw_img2img_vit")
            if isinstance(hf_inputs, Mapping)
            else None
        )
        if img2img_vit_grid_thw is not None:
            img2img_vit_grid_sizes = img2img_vit_grid_thw[:, 1] * img2img_vit_grid_thw[:, 2]
            config["pixel_values_img2img_vit"] = MultiModalFieldConfig.flat_from_sizes(
                "img2img",
                img2img_vit_grid_sizes,
            )
            config["image_grid_thw_img2img_vit"] = MultiModalFieldConfig.batched("img2img")
        return config

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        has_image = "images" in mm_data
        has_img2img = "pixel_values_img2img" in mm_data

        if has_img2img and self.IMG2IMG_PLACEHOLDER not in prompt:
            prompt = f"{self.IMG2IMG_PLACEHOLDER}{prompt}"

        if has_image and has_img2img:
            outputs = BatchFeature()

            img_data = dict(mm_data)
            if "pixel_values_img2img" in img_data:
                del img_data["pixel_values_img2img"]
            kwargs_img = dict(mm_kwargs)
            kwargs_img["is_img2img"] = False
            out_img = super()._call_hf_processor(prompt, img_data, kwargs_img, tok_kwargs)
            if "pixel_values" in out_img:
                outputs["pixel_values"] = out_img["pixel_values"]
            for k, v in out_img.items():
                if k != "pixel_values":
                    outputs[k] = v

            img2img_data = dict(mm_data)
            if "images" in img2img_data:
                del img2img_data["images"]
            img2img_data["images"] = img2img_data.pop("pixel_values_img2img")
            kwargs_img2img = self._mm_kwargs_for_bagel_img2img_hf(mm_kwargs)
            kwargs_img2img["is_img2img"] = True
            out_img2img = super()._call_hf_processor(prompt, img2img_data, kwargs_img2img, tok_kwargs)
            if "pixel_values" in out_img2img:
                outputs["pixel_values_img2img"] = out_img2img["pixel_values"]
            for k, v in out_img2img.items():
                if k not in outputs:
                    outputs[k] = v

            return outputs

        elif has_img2img:
            mm_data = dict(mm_data)
            mm_data["images"] = mm_data.pop("pixel_values_img2img")
            mm_kwargs = self._mm_kwargs_for_bagel_img2img_hf(mm_kwargs)
            mm_kwargs["is_img2img"] = True
            outputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
            if "pixel_values" in outputs:
                outputs["pixel_values_img2img"] = outputs.pop("pixel_values")
            return outputs

        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()

        replacements: list[PromptReplacement] = []

        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
        vision_start_id = tokenizer.get_vocab().get("<|vision_start|>")
        vision_end_id = tokenizer.get_vocab().get("<|vision_end|>")
        reference_layout = bagel_vqa_reference_layout_enabled()
        if image_token_id is not None:
            # PATCH (navit-thinker-fix): with variable-shape ViT, each image
            # consumes its own actual patch count (grid_h × grid_w from
            # image_grid_thw), not the maximum 70×70=4900 slots. Read the
            # grid from out_mm_kwargs which our HF processor populated.
            # Pattern mirrors Qwen2-VL (`qwen2_5_vl.py:_get_prompt_updates`).
            max_num_patches = hf_config.vit_max_num_patch_per_side**2

            def _get_image_patch_count(item_idx: int) -> int:
                try:
                    out_item = out_mm_kwargs["image"][item_idx]
                    grid = out_item["image_grid_thw"].data
                    return int(grid[1].item()) * int(grid[2].item())
                except Exception:
                    return max_num_patches

            def get_image_replacement(item_idx: int):
                count = _get_image_patch_count(item_idx)
                if not reference_layout or vision_start_id is None or vision_end_id is None:
                    return [image_token_id] * count

                full = [vision_start_id] + [image_token_id] * count + [vision_end_id]
                return PromptUpdateDetails.select_token_id(full, image_token_id)

            replacements.append(
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_image_replacement,
                )
            )

        img2img_token_id = tokenizer.get_vocab().get("<|fim_middle|>")
        if img2img_token_id is not None:
            include_vit = _bagel_force_input_img2img_vit_enabled()
            vit_config = hf_config.vit_config
            image_size = vit_config.image_size

            latent_patch_size = getattr(hf_config, "latent_patch_size", 2)
            downsample = hf_config.vae_config.get("downsample", 8)
            latent_downsample = downsample * latent_patch_size

            def get_img2img_processed_item(item_idx: int) -> dict[str, object]:
                if "img2img" not in out_mm_kwargs:
                    return {}
                try:
                    items = out_mm_kwargs["img2img"]
                    if item_idx >= len(items) or items[item_idx] is None:
                        return {}
                    return items[item_idx].get_data()
                except Exception:
                    return {}

            def get_img2img_replacement(item_idx: int):
                h, w = image_size, image_size
                processed = get_img2img_processed_item(item_idx)
                processed_vae_hw: tuple[int, int] | None = None
                processed_vit_tokens: int | None = None

                if _bagel_img2img_local_preprocess_enabled():
                    pv = processed.get("pixel_values_img2img")
                    if isinstance(pv, torch.Tensor) and pv.ndim >= 3:
                        processed_vae_hw = (int(pv.shape[-2]), int(pv.shape[-1]))

                    grid = processed.get("image_grid_thw_img2img_vit")
                    if isinstance(grid, torch.Tensor) and grid.numel() >= 3:
                        row = grid.reshape(-1, 3)[0]
                        processed_vit_tokens = int(row[1].item()) * int(row[2].item()) + 2

                if "img2img" in mm_items:
                    item = mm_items.get_items("img2img", (Img2ImgProcessorItems, ImageEmbeddingItems))
                    if hasattr(item, "get_image_size"):
                        size = item.get_image_size(item_idx)
                        h, w = size.height, size.width

                max_latent_size = getattr(hf_config, "max_latent_size", 32)
                if processed_vae_hw is not None:
                    vae_h, vae_w = processed_vae_hw
                else:
                    vae_h, vae_w = _bagel_force_img2img_vae_hw(
                        h,
                        w,
                        latent_downsample=latent_downsample,
                        max_latent_size=max_latent_size,
                    )
                vit_h, vit_w = _bagel_force_img2img_vit_hw(vae_h, vae_w)

                num_vae_patches = (vae_h // latent_downsample) * (vae_w // latent_downsample)
                num_vae_total = num_vae_patches + 2
                num_vit_total = (
                    processed_vit_tokens
                    if processed_vit_tokens is not None
                    else (vit_h // vit_config.patch_size) * (vit_w // vit_config.patch_size) + 2
                )
                if include_vit:
                    if _bagel_img2img_vit_separator_enabled():
                        # Historical vLLM-Omni img2img layout: VAE image block,
                        # one raw <|fim_middle|> separator token, then the ViT
                        # image block. Local BAGEL's force path performs VAE and
                        # ViT cache updates back-to-back without this token, so
                        # keep the separator opt-in for diagnostics only.
                        total = num_vae_total + 1 + num_vit_total
                        embed_mask = [True] * num_vae_total + [False] + [True] * num_vit_total
                    else:
                        total = num_vae_total + num_vit_total
                        embed_mask = [True] * total
                else:
                    total = num_vae_total
                    embed_mask = [True] * num_vae_total
                tokens = [img2img_token_id] * total

                if _bagel_img2img_debug_enabled():
                    logger.info(
                        "BAGEL img2img replacement idx=%d raw_hw=%dx%d vae_hw=%dx%d "
                        "vit_hw=%dx%d vae_tokens=%d vit_tokens=%d total=%d "
                        "processed_vae=%s processed_vit=%s sep=%s",
                        item_idx,
                        h,
                        w,
                        vae_h,
                        vae_w,
                        vit_h,
                        vit_w,
                        num_vae_total,
                        num_vit_total if include_vit else 0,
                        total,
                        processed_vae_hw is not None,
                        processed_vit_tokens is not None,
                        _bagel_img2img_vit_separator_enabled(),
                    )

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


class VAEEncoder(nn.Module):
    """Lightweight VAE encoder (no decoder) for embedding images in the AR stage."""

    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z


def _bagel_vqa_layer_attention_metadata(layer_name: str):
    from vllm.forward_context import get_forward_context

    raw = get_forward_context().attn_metadata
    if isinstance(raw, dict):
        return raw.get(layer_name)
    if isinstance(raw, list):
        return raw[0].get(layer_name)
    return raw


def _bagel_vqa_recompute_noncausal_image_blocks(
    qwen_attn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
) -> torch.Tensor:
    owner = getattr(qwen_attn, "_bagel_vqa_owner", None)
    spans = getattr(owner, "_bagel_vqa_image_spans_current", None)
    if not spans:
        return attn_output
    return _bagel_recompute_noncausal_image_blocks_paged(
        qwen_attn,
        query,
        key,
        value,
        attn_output,
        spans,
    )


def _bagel_recompute_noncausal_image_blocks_paged(
    qwen_attn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    spans: list[dict[str, int]],
) -> torch.Tensor:
    if not spans:
        return attn_output
    attn_layer = qwen_attn.attn
    backend_name = attn_layer.attn_backend.get_name()

    impl = attn_layer.impl
    if getattr(impl, "dcp_world_size", 1) > 1:
        raise RuntimeError("BAGEL reference prefill does not support DCP yet")

    metadata = _bagel_vqa_layer_attention_metadata(attn_layer.layer_name)
    if metadata is None:
        return attn_output
    if getattr(metadata, "use_cascade", False):
        raise RuntimeError(
            "BAGEL reference prefill does not support cascade attention; "
            "disable prefix caching for this mode"
        )

    if backend_name != "FLASH_ATTN":
        return _bagel_vqa_recompute_noncausal_image_blocks_direct(
            qwen_attn,
            query,
            key,
            value,
            attn_output,
            spans,
            backend_name,
        )

    try:
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            is_flash_attn_varlen_func_available,
        )
    except Exception as exc:
        raise RuntimeError("BAGEL reference prefill requires FlashAttention") from exc
    if not is_flash_attn_varlen_func_available():
        raise RuntimeError("BAGEL reference prefill requires flash_attn_varlen_func")

    kv_cache = attn_layer.kv_cache
    if kv_cache.numel() == 0:
        return attn_output
    key_cache, value_cache = kv_cache.unbind(0)

    out = attn_output.clone()
    block_table = metadata.block_table
    device = query.device
    q_descale = None

    for span in spans:
        q_start = int(span["q_start"])
        q_end = int(span["q_end"])
        req_idx = int(span["req_idx"])
        kv_end = int(span["kv_end"])
        if q_end <= q_start:
            continue

        q_len = q_end - q_start
        q = query[q_start:q_end].view(q_len, attn_layer.num_heads, attn_layer.head_size)
        block_out = torch.empty(
            (q_len, attn_layer.num_heads, attn_layer.head_size_v),
            dtype=out.dtype,
            device=device,
        )
        cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
        seqused_k = torch.tensor([kv_end], dtype=torch.int32, device=device)

        descale_shape = (1, attn_layer.num_kv_heads)
        if getattr(impl, "supports_quant_query_input", False):
            q_descale = attn_layer._q_scale.expand(descale_shape)
        k_descale = attn_layer._k_scale.expand(descale_shape)
        v_descale = attn_layer._v_scale.expand(descale_shape)
        sliding_window = (
            list(impl.sliding_window)
            if getattr(impl, "sliding_window", None) is not None
            else None
        )

        flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=block_out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=q_len,
            seqused_k=seqused_k,
            max_seqlen_k=max(kv_end, 1),
            softmax_scale=impl.scale,
            causal=False,
            alibi_slopes=impl.alibi_slopes,
            window_size=sliding_window,
            block_table=block_table[req_idx : req_idx + 1],
            softcap=impl.logits_soft_cap,
            scheduler_metadata=None,
            fa_version=impl.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=1,
            s_aux=getattr(impl, "sinks", None),
        )
        out[q_start:q_end] = block_out.reshape(q_len, -1)

    return out


def _bagel_vqa_recompute_noncausal_image_blocks_direct(
    qwen_attn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    spans: list[dict[str, int]],
    backend_name: str,
) -> torch.Tensor:
    """Backend-independent recompute for full-prefill image blocks.

    Triton paged attention in vLLM 0.20 only exposes causal decoder attention.
    In reference-prefill mode we disable chunked prefill, so BAGEL VQA image
    spans are scheduled in the first prompt pass and the current K/V tensors
    contain all keys needed up to the end of the image block.
    """
    attn_layer = qwen_attn.attn
    impl = attn_layer.impl
    if getattr(impl, "alibi_slopes", None) is not None:
        raise RuntimeError(
            "BAGEL reference prefill direct recompute does not support ALiBi"
        )
    if getattr(impl, "sinks", None) is not None:
        raise RuntimeError(
            "BAGEL reference prefill direct recompute does not support sinks"
        )

    sliding_window = getattr(impl, "sliding_window", None)
    if sliding_window not in (None, (-1, -1), [-1, -1]):
        raise RuntimeError(
            "BAGEL reference prefill direct recompute does not support "
            f"sliding window attention on backend {backend_name}"
        )

    out = attn_output.clone()
    q_all = query.view(-1, attn_layer.num_heads, attn_layer.head_size)
    k_all = key.view(-1, attn_layer.num_kv_heads, attn_layer.head_size)
    v_all = value.view(-1, attn_layer.num_kv_heads, attn_layer.head_size_v)
    repeat = attn_layer.num_heads // attn_layer.num_kv_heads
    softcap = getattr(impl, "logits_soft_cap", 0) or 0

    for span in spans:
        computed = int(span.get("num_computed_tokens", 0))
        if computed != 0:
            raise RuntimeError(
                "BAGEL reference prefill direct recompute needs the complete "
                "image prefix in the current scheduled batch; disable chunked "
                f"prefill or use a paged non-causal backend (got {computed} "
                "computed tokens)"
            )

        q_start = int(span["q_start"])
        q_end = int(span["q_end"])
        request_start = int(span["request_start"])
        kv_local_end = int(span["kv_local_end"])
        kv_end = request_start + kv_local_end
        if q_end <= q_start or kv_end <= request_start:
            continue

        q = q_all[q_start:q_end]
        k = k_all[request_start:kv_end]
        v = v_all[request_start:kv_end]
        if repeat != 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if softcap <= 0:
            q_sdpa = q.transpose(0, 1).unsqueeze(0).to(dtype=v.dtype)
            k_sdpa = k.transpose(0, 1).unsqueeze(0).to(dtype=v.dtype)
            v_sdpa = v.transpose(0, 1).unsqueeze(0)
            block_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=0.0,
                is_causal=False,
                scale=impl.scale,
            )
            block_out = block_out.squeeze(0).transpose(0, 1)
            out[q_start:q_end] = block_out.reshape(q_end - q_start, -1).to(out.dtype)
            continue

        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * impl.scale
        if softcap > 0:
            scores = torch.tanh(scores / softcap) * softcap
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        block_out = torch.einsum("hqk,khd->qhd", probs, v)
        out[q_start:q_end] = block_out.reshape(q_end - q_start, -1).to(out.dtype)

    return out


def _bagel_vqa_reference_qwen2_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    if self.qk_norm:
        total_tokens = q.shape[0]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(total_tokens, self.q_size)
        k = k.view(total_tokens, self.kv_size)

    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    attn_output = _bagel_vqa_recompute_noncausal_image_blocks(
        self,
        q,
        k,
        v,
        attn_output,
    )
    output, _ = self.o_proj(attn_output)
    return output


@MULTIMODAL_REGISTRY.register_processor(
    OmniBagelMultiModalProcessor,
    info=OmniBagelProcessingInfo,
    dummy_inputs=OmniBagelDummyInputsBuilder,
)
class OmniBagelForConditionalGeneration(BagelForConditionalGeneration):
    """
    Omni version of BagelForConditionalGeneration.

    Extends the base model with a VAE encoder so that img2img can embed
    both VAE latents and ViT features within the AR stage, producing a
    combined KV cache that is then transferred to the DiT stage.

    Position IDs are adjusted so that:
      - VAE tokens all share position 0
      - ViT tokens all share position 1
      - Text tokens use sequential positions starting from 2
    This matches the position scheme used by the single-stage DiT pipeline,
    ensuring the transferred KV cache + ropes are directly compatible with
    the DiT's denoising loop.
    """

    # LoRA packed→sublayer mapping for both standard Qwen2 projections
    # and the MoE generation-mode projections added by _install_mot_modules().
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv_proj_moe_gen": [
            "q_proj_moe_gen",
            "k_proj_moe_gen",
            "v_proj_moe_gen",
        ],
        "mlp_moe_gen.gate_up_proj": [
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        # PATCH (navit-thinker-fix): `self.vit_model` (vllm's
        # SiglipVisionModel) stays untouched so the upstream fixed-shape
        # `_process_image_input` (used by img2img and warmup) still works.
        # The NaViT path runs a per-image vllm-encoder loop in
        # `_process_image_input_navit` (Session 5 winning state). Session 6
        # tried swapping in HF SiglipVisionModel via SiglipNaViTWrapper —
        # closed `know` by +16.67 but regressed text-heavy categories
        # (math/ocr/spat -3 to -9 each), net -3.25 overall. Reverted; helper
        # `_load_hf_navit_vit` is retained as a known-good loader for
        # potential future hybrid experiments.

        self.latent_patch_size = getattr(config, "latent_patch_size", 2)
        self.downsample = config.vae_config.get("downsample")
        self.latent_downsample = self.downsample * self.latent_patch_size
        self.max_latent_size = getattr(config, "max_latent_size", 32)
        self.latent_channel = config.vae_config.get("z_channels")

        hidden_size = config.llm_config.hidden_size
        patch_latent_dim = self.latent_patch_size**2 * self.latent_channel
        self.vae = VAEEncoder(default_ae_params())
        self.vae2llm = nn.Linear(patch_latent_dim, hidden_size)
        self.latent_pos_embed = PositionEmbedding(self.max_latent_size, hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)

        self._pending_img2img_info: list[tuple[int, ...]] = []
        self._img2img_noncausal_spans_current: list[dict[str, int]] = []
        self._ropes_pending: list[dict[str, Any]] = []
        self._ropes_metadata: dict[str, dict[str, Any]] = {}
        self._last_img2img_info: tuple[int, ...] | None = None
        self._bagel_vqa_rope_states: dict[str, dict[str, Any]] = {}
        self._bagel_vqa_rope_positions_current: torch.Tensor | None = None
        self._bagel_vqa_image_spans_current: list[dict[str, int]] = []
        self._bagel_runner_num_computed_tokens_current: list[int] = []
        self._bagel_vqa_reference_prefill_enabled = (
            bagel_vqa_reference_prefill_enabled()
        )
        self._bagel_vqa_logical_rope_enabled = (
            self._bagel_vqa_reference_prefill_enabled
            or os.environ.get("BAGEL_VQA_LOGICAL_ROPE", "").lower()
            in {"1", "true", "yes", "on"}
        )

        from transformers import AutoTokenizer

        tok_name = getattr(vllm_config.model_config, "tokenizer", None) or vllm_config.model_config.model
        _tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        for t in ["<|vision_start|>", "<|vision_end|>"]:
            if t not in _tok.get_vocab():
                _tok.add_tokens([t])
        self._start_of_image_id = int(_tok.convert_tokens_to_ids("<|vision_start|>"))
        self._end_of_image_id = int(_tok.convert_tokens_to_ids("<|vision_end|>"))
        self._img2img_token_id = int(_tok.convert_tokens_to_ids("<|fim_middle|>"))
        self._vae_token_mask: torch.Tensor | None = None
        self.device = get_local_device()
        self._install_mot_modules(config)
        if self._bagel_vqa_reference_prefill_enabled:
            self._install_vqa_reference_attention()

    def _install_mot_modules(self, config):
        """Add generation-mode (MoT) weight modules to each Qwen2 decoder layer.

        The single-stage DiT routes VAE latent tokens through separate
        ``qkv_proj_moe_gen / o_proj_moe_gen / mlp_moe_gen`` weight matrices
        (``mode="gen"``).  We replicate that structure here so the AR stage
        produces the same KV cache values.
        """
        llm_cfg = config.llm_config
        hidden_size = llm_cfg.hidden_size
        intermediate_size = llm_cfg.intermediate_size
        num_heads = llm_cfg.num_attention_heads
        num_kv_heads = llm_cfg.num_key_value_heads
        head_dim = hidden_size // num_heads
        rms_eps = llm_cfg.rms_norm_eps

        qwen2_model = self.language_model.model  # Qwen2Model

        qwen2_model.norm_moe_gen = VllmRMSNorm(hidden_size, eps=rms_eps)

        for layer in qwen2_model.layers:
            if not isinstance(layer, Qwen2DecoderLayer):
                continue
            attn = layer.self_attn

            attn.qkv_proj_moe_gen = QKVParallelLinear(
                hidden_size,
                head_dim,
                num_heads,
                num_kv_heads,
                bias=True,
            )
            attn.o_proj_moe_gen = RowParallelLinear(
                num_heads * head_dim,
                hidden_size,
                bias=False,
            )
            attn.q_norm_moe_gen = VllmRMSNorm(head_dim, eps=rms_eps)
            attn.k_norm_moe_gen = VllmRMSNorm(head_dim, eps=rms_eps)

            layer.mlp_moe_gen = Qwen2MLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=llm_cfg.hidden_act,
            )
            layer.input_layernorm_moe_gen = VllmRMSNorm(hidden_size, eps=rms_eps)
            layer.post_attention_layernorm_moe_gen = VllmRMSNorm(hidden_size, eps=rms_eps)

    def _install_vqa_reference_attention(self) -> None:
        """Bind BAGEL-only image-block non-causal attention recompute hooks."""
        qwen2_model = self.language_model.model
        for layer in qwen2_model.layers:
            if not isinstance(layer, Qwen2DecoderLayer):
                continue
            attn = layer.self_attn
            if getattr(attn, "_bagel_vqa_reference_attention_installed", False):
                object.__setattr__(attn, "_bagel_vqa_owner", self)
                continue
            object.__setattr__(attn, "_bagel_vqa_owner", self)
            attn._bagel_vqa_original_forward = attn.forward
            attn.forward = _bagel_vqa_reference_qwen2_attention_forward.__get__(
                attn,
                attn.__class__,
            )
            attn._bagel_vqa_reference_attention_installed = True

    def _load_hf_navit_vit(self, device: torch.device, dtype: torch.dtype) -> SiglipNaViTWrapper:
        """Lazily build a HuggingFace SiglipVisionModel + NaViT wrapper for the
        VQA path. Loads weights from the BAGEL checkpoint (keys
        ``vit_model.vision_model.*``) directly — no dependence on vllm's
        already-loaded ``self.vit_model``. This bypasses vllm's adapted
        ``SiglipEncoder`` (which lacks ``attention_mask`` support and produces
        unnormalized output) and matches the reference torch-native bagel_local
        pipeline exactly.

        PATCH (hf-sigvit): the only forward path that hits this is
        ``_process_image_input_navit`` (VQA). img2img + warmup still use
        vllm's ``self.vit_model`` via the parent's ``_process_image_input``.
        """
        from pathlib import Path

        from safetensors import safe_open
        from transformers import SiglipVisionConfig, SiglipVisionModel

        vit_cfg = self.config.vit_config
        hf_cfg = SiglipVisionConfig(
            hidden_size=vit_cfg.hidden_size,
            intermediate_size=vit_cfg.intermediate_size,
            num_attention_heads=vit_cfg.num_attention_heads,
            num_hidden_layers=vit_cfg.num_hidden_layers,
            patch_size=vit_cfg.patch_size,
            image_size=vit_cfg.image_size,
            num_channels=vit_cfg.num_channels,
            attention_dropout=getattr(vit_cfg, "attention_dropout", 0.0),
            layer_norm_eps=getattr(vit_cfg, "layer_norm_eps", 1e-6),
            hidden_act=getattr(vit_cfg, "hidden_act", "gelu_pytorch_tanh"),
            vision_use_head=False,
        )
        # `meta` device + later `to_empty` + state_dict load avoids materialising
        # a full random-init ViT just to overwrite it.
        with torch.device("meta"):
            hf_model = SiglipVisionModel(hf_cfg)
        hf_model = hf_model.to_empty(device=device).to(dtype=dtype)

        ckpt_dir = Path(self._hf_navit_ckpt_path)
        index_path = ckpt_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise RuntimeError(
                f"HF NaViT ViT load: missing safetensors index at {index_path}"
            )
        import json as _json
        with open(index_path) as f:
            weight_map = _json.load(f)["weight_map"]

        # Collect every `vit_model.vision_model.*` key, mapping prefix away.
        wanted = {k: v for k, v in weight_map.items() if k.startswith("vit_model.vision_model.")}
        if not wanted:
            raise RuntimeError("HF NaViT ViT load: no vit_model.vision_model.* keys found")

        # Group by shard for efficient I/O.
        by_shard: dict[str, list[str]] = {}
        for k, shard in wanted.items():
            by_shard.setdefault(shard, []).append(k)

        # Determine prefix handling: transformers >=5.x flattens
        # SiglipVisionModel (state_dict starts with `embeddings.*`); pre-5.x
        # kept the inner wrapper (`vision_model.embeddings.*`). Inspect what
        # the model actually wants.
        hf_keys = set(hf_model.state_dict().keys())
        needs_strip = "embeddings.patch_embedding.weight" in hf_keys

        state: dict[str, torch.Tensor] = {}
        for shard, keys in by_shard.items():
            with safe_open(str(ckpt_dir / shard), framework="pt") as sf:
                for k in keys:
                    # Strip BAGEL's `vit_model.` prefix.
                    new_key = k[len("vit_model."):]
                    # Optionally also strip `vision_model.` prefix on
                    # transformers >=5.x.
                    if needs_strip:
                        new_key = new_key[len("vision_model."):]
                    t = sf.get_tensor(k).to(dtype=dtype)
                    # BAGEL stores patch_embedding flat as (out, kh*kw*in) with
                    # inner order (kh, kw, c); HF Conv2d expects (out, in, kh, kw).
                    is_patch_w = new_key.endswith("embeddings.patch_embedding.weight")
                    if is_patch_w and t.ndim == 2:
                        patch_size = vit_cfg.patch_size
                        in_ch = vit_cfg.num_channels
                        out_ch = t.shape[0]
                        t = t.reshape(out_ch, patch_size, patch_size, in_ch).permute(0, 3, 1, 2).contiguous()
                    state[new_key] = t

        missing, unexpected = hf_model.load_state_dict(state, strict=False)
        # `head` is added by HF for some configurations and is absent in BAGEL;
        # filter it out of the unexpected list before warning.
        unexpected = [u for u in unexpected if "head" not in u]
        if missing or unexpected:
            import logging
            logging.getLogger(__name__).warning(
                "HF NaViT ViT load: missing=%s unexpected=%s (transformers_strip=%s)",
                missing[:5], unexpected[:5], needs_strip,
            )

        hf_model.eval()
        for p in hf_model.parameters():
            p.requires_grad_(False)
        return SiglipNaViTWrapper(hf_model)

    def _resize_to_stride(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Resize pixel values to stride-aligned dimensions
        (matches DiT's ``_resize_images_to_stride``)."""
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        stride = self.latent_downsample
        max_img_size = int(self.max_latent_size * stride)

        scale = min(max_img_size / max(H, W), 1.0)
        min_img_size = min(256, max_img_size)
        scale = max(scale, min_img_size / min(H, W))
        new_H = max(stride, int(round(H * scale / stride) * stride))
        new_W = max(stride, int(round(W * scale / stride) * stride))
        new_H = min(new_H, max_img_size)
        new_W = min(new_W, max_img_size)

        if new_H != H or new_W != W:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(new_H, new_W), mode="bicubic", align_corners=False
            )
        return pixel_values

    def _clear_warmup_state(self):
        """Clear stale state accumulated during warmup/profiling runs."""
        self._ropes_pending.clear()
        self._ropes_metadata.clear()
        self._pending_img2img_info.clear()
        self._last_img2img_info = None
        self._vae_token_mask = None
        self._bagel_vqa_rope_states.clear()
        self._bagel_vqa_rope_positions_current = None
        self._bagel_vqa_image_spans_current = []
        self._bagel_runner_num_computed_tokens_current = []

    def get_kv_transfer_metadata(
        self,
        req_id: str,
        *,
        num_computed_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        # NOTE: num_computed_tokens will not include async placeholders
        meta = self._ropes_metadata.pop(req_id, None)
        if meta is None:
            return None
        if num_computed_tokens is not None and "image_shape" in meta:
            prefill_rope = meta["ropes"][0] if meta.get("ropes") else 0
            prefill_position_count = meta.get("prefill_position_count")
            if prefill_position_count is not None:
                num_decoded = num_computed_tokens - prefill_position_count
                if num_decoded > 0:
                    meta["ropes"] = [prefill_rope + num_decoded]
            elif num_computed_tokens > prefill_rope:
                meta["ropes"] = [num_computed_tokens]
        return meta

    def prepare_runner_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        req_ids: list[str],
        num_computed_tokens: list[int],
        num_scheduled_tokens: list[int],
        input_ids_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Restore input_ids so _adjust_positions_for_img2img can locate
        the <|fim_middle|> placeholder for thinking-mode pre_text_len
        detection."""
        if inputs_embeds is not None and input_ids is None and input_ids_buffer is not None:
            input_ids = input_ids_buffer
        self._bagel_vqa_rope_positions_current = self._build_bagel_vqa_rope_positions(
            input_ids=input_ids,
            positions=positions,
            req_ids=req_ids,
            num_computed_tokens=num_computed_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        self._bagel_vqa_image_spans_current = self._build_bagel_vqa_image_spans(
            input_ids=input_ids,
            req_ids=req_ids,
            num_computed_tokens=num_computed_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        self._bagel_runner_num_computed_tokens_current = [
            int(x) for x in num_computed_tokens
        ]
        return input_ids, positions

    def _build_bagel_vqa_rope_positions(
        self,
        *,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        req_ids: list[str],
        num_computed_tokens: list[int],
        num_scheduled_tokens: list[int],
    ) -> torch.Tensor | None:
        """Compute BAGEL logical RoPE positions without touching vLLM slots.

        vLLM's `positions` tensor remains the physical paged-KV position used by
        slot mapping and attention metadata. This side-channel mirrors BAGEL
        reference's logical rule for VQA: every token in a vision block shares
        one RoPE position, and text/decode tokens continue from the collapsed
        logical length. State is per request so chunked prefill and continuous
        batching can cross image-block boundaries safely.
        """
        if not self._bagel_vqa_logical_rope_enabled:
            return None

        return build_bagel_vqa_rope_positions(
            input_ids=input_ids,
            positions=positions,
            req_ids=req_ids,
            num_computed_tokens=num_computed_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            rope_states=self._bagel_vqa_rope_states,
            start_of_image_id=self._start_of_image_id,
            end_of_image_id=self._end_of_image_id,
        )

    def _build_bagel_vqa_image_spans(
        self,
        *,
        input_ids: torch.Tensor | None,
        req_ids: list[str],
        num_computed_tokens: list[int],
        num_scheduled_tokens: list[int],
    ) -> list[dict[str, int]]:
        if not self._bagel_vqa_reference_prefill_enabled:
            return []
        return build_bagel_vqa_image_spans(
            input_ids=input_ids,
            req_ids=req_ids,
            num_computed_tokens=num_computed_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            start_of_image_id=self._start_of_image_id,
            end_of_image_id=self._end_of_image_id,
        )

    def flush_pending_metadata(self, req_ids: list[str]) -> None:
        """Map pending metadata (batch order) to req_ids after forward().

        Guard: if a request already has metadata with ``image_shape``
        (written during img2img prefill), don't overwrite it with
        decode-step metadata that lacks ``image_shape``.
        """
        pending = self._ropes_pending
        self._ropes_pending = []
        for i, meta in enumerate(pending):
            if i < len(req_ids):
                rid = req_ids[i]
                existing = self._ropes_metadata.get(rid)
                if existing and "image_shape" in existing and "image_shape" not in meta:
                    continue
                ropes = meta.get("ropes")
                if ropes:
                    meta["ropes"] = [int(r.item()) if isinstance(r, torch.Tensor) else r for r in ropes]
                self._ropes_metadata[rid] = meta

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        if any(k in kwargs for k in ("pixel_values", "image_embeds")):
            parsed = self._parse_and_validate_image_input(**kwargs)
            # PATCH (navit-thinker-fix): propagate image_grid_thw alongside
            # pixel_values so `_process_img2text_input` can dispatch to the
            # NaViT branch when present. The parent returns an immutable
            # ``BagelImagePixelInputs`` TensorSchema, so re-pack into a plain
            # dict (still subscriptable downstream) before adding extra keys.
            grid_thw = kwargs.get("image_grid_thw")
            if parsed is not None and grid_thw is not None:
                parsed_dict = {
                    "type": parsed["type"],
                    "pixel_values": parsed["pixel_values"],
                    "image_grid_thw": grid_thw,
                }
                mm_input_by_modality["img2text"] = parsed_dict
            else:
                mm_input_by_modality["img2text"] = parsed

        img2img_keys = {
            "pixel_values_img2img": "pixel_values",
            "image_embeds_img2img": "image_embeds",
            "pixel_values_img2img_vit": "pixel_values_vit",
            "image_grid_thw_img2img_vit": "image_grid_thw_vit",
        }
        img2img_kwargs = {img2img_keys[k]: v for k, v in kwargs.items() if k in img2img_keys}

        if img2img_kwargs:
            combined_kwargs = kwargs.copy()
            combined_kwargs.update(img2img_kwargs)
            parsed = self._parse_and_validate_image_input(**combined_kwargs)
            if parsed is not None:
                parsed_dict = {
                    "type": parsed["type"],
                    "pixel_values": parsed["pixel_values"],
                }
                if "pixel_values_vit" in img2img_kwargs:
                    parsed_dict["pixel_values_vit"] = img2img_kwargs["pixel_values_vit"]
                if "image_grid_thw_vit" in img2img_kwargs:
                    parsed_dict["image_grid_thw_vit"] = img2img_kwargs["image_grid_thw_vit"]
                mm_input_by_modality["img2img"] = parsed_dict

        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "img2text":
                image_embeddings = self._process_img2text_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "img2img":
                img2img_embeddings = self._process_img2img_input(multimodal_input)
                multimodal_embeddings += tuple(img2img_embeddings)
        return multimodal_embeddings

    def get_flattened_position_ids(self, img_h, img_w, patch_size, max_num_patches_per_side):
        num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
        coords_h = torch.arange(0, num_patches_h)
        coords_w = torch.arange(0, num_patches_w)
        pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
        return pos_ids

    def _process_img2text_input(self, multimodal_input):
        # PATCH (navit-thinker-fix): when our processor produced packed
        # variable-shape pixel_values + image_grid_thw, take the NaViT path.
        # Otherwise fall back to the upstream fixed-shape path (used by
        # warmup / dummy inputs that pre-date this patch).
        if "image_grid_thw" in multimodal_input:
            return self._process_image_input_navit(multimodal_input)
        return self._process_image_input(multimodal_input)

    def _process_image_input_navit(self, multimodal_input) -> tuple[torch.Tensor, ...]:
        """NaViT-style ViT forward for variable-shape packed pixel_values.

        Per-image vllm-encoder loop (the Session 5 winning configuration).
        Tried Session 6 HF SigLIP via SiglipNaViTWrapper: improved `know`
        (+16.67) but regressed `math/ocr/spat` by 3-9pt each, net -3.25
        overall on full 100 vs this baseline. Reverted because the gain on
        `know` doesn't compensate the loss on text-heavy categories.

        Sequence: linear patch_embed + pos_embed → vllm SiglipEncoder per
        image → post_layernorm → connector → vit_pos_embed. post_layernorm
        IS load-bearing here — removing it broke the path entirely (mean=0.0
        on smoke).
        """
        pixel_values = multimodal_input["pixel_values"]
        grid_thw = multimodal_input["image_grid_thw"]
        # `flat_from_sizes` slices into a list of per-image (L_i, ...) tensors;
        # concatenate back to packed (total_patches, ...) for NaViT.
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat([t for t in pixel_values], dim=0)
        if isinstance(grid_thw, (list, tuple)):
            grid_thw = torch.cat([t.unsqueeze(0) if t.ndim == 1 else t for t in grid_thw], dim=0)

        device = pixel_values.device
        grid_thw = grid_thw.to(device)
        patch_counts = grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            patch_counts.cumsum(0).to(torch.int32),
        ])

        # Shared bits: patch embedding weight and pos table.
        vit = self.vit_model  # vllm's SiglipVisionModel (unwrapped — see __init__)
        patch_embed = vit.vision_model.embeddings.patch_embedding
        patch_embed_weight = patch_embed.weight.view(patch_embed.weight.shape[0], -1)
        position_embedding = vit.vision_model.embeddings.position_embedding
        target_dtype = patch_embed.weight.dtype
        patch_size = self.config.vit_config.patch_size
        max_per_side = self.config.vit_max_num_patch_per_side

        # Forward each image independently through patch_embed + pos + encoder.
        per_image_features: list[torch.Tensor] = []
        per_image_pos_ids: list[torch.Tensor] = []
        cu_list = cu_seqlens.tolist()
        for i in range(grid_thw.shape[0]):
            start, end = cu_list[i], cu_list[i + 1]
            patches = pixel_values[start:end].to(dtype=target_dtype)

            h_pixels = int(grid_thw[i, 1].item()) * patch_size
            w_pixels = int(grid_thw[i, 2].item()) * patch_size
            pos_ids = get_flattened_position_ids_extrapolate(
                h_pixels, w_pixels, patch_size, max_per_side
            ).to(device=device, dtype=torch.long)
            per_image_pos_ids.append(pos_ids)

            x = torch.nn.functional.linear(patches, patch_embed_weight, patch_embed.bias)
            x = x + position_embedding(pos_ids)
            x_3d = x.unsqueeze(0)

            encoder_out = vit.vision_model.encoder(
                inputs_embeds=x_3d, return_all_hidden_states=False
            )
            if isinstance(encoder_out, list):
                encoder_out = encoder_out[-1]
            post_ln = getattr(vit.vision_model, "post_layernorm", None)
            if post_ln is not None:
                encoder_out = post_ln(encoder_out)
            per_image_features.append(encoder_out.squeeze(0))

        vision_features = torch.cat(per_image_features, dim=0)
        vision_embeds = self.connector(vision_features)
        packed_pos_ids = torch.cat(per_image_pos_ids)
        pos_emb = self.vit_pos_embed(packed_pos_ids.cpu()).to(
            device=vision_embeds.device, dtype=vision_embeds.dtype
        )
        vision_embeds = vision_embeds + pos_emb

        return tuple(
            vision_embeds[cu_list[i]: cu_list[i + 1]]
            for i in range(len(cu_list) - 1)
        )

    def _process_img2img_input(self, multimodal_input):
        pixel_values = multimodal_input["pixel_values"]
        pixel_value_items: list[torch.Tensor] = []
        raw_items = pixel_values if isinstance(pixel_values, (list, tuple)) else [pixel_values]
        for item in raw_items:
            if item.ndim == 5:
                b, n, c, h, w = item.shape
                item = item.reshape(b * n, c, h, w)
            elif item.ndim == 3:
                item = item.unsqueeze(0)
            if item.ndim != 4:
                raise ValueError(f"Unsupported img2img pixel_values shape: {tuple(item.shape)}")
            pixel_value_items.extend(item[i : i + 1] for i in range(item.shape[0]))

        if not pixel_value_items:
            return ()

        num_images = len(pixel_value_items)
        pixel_device = pixel_value_items[0].device
        include_vit = _bagel_force_input_img2img_vit_enabled()
        use_vit_separator = include_vit and _bagel_img2img_vit_separator_enabled()
        p = self.latent_patch_size
        timestep = 0

        if self._ropes_pending:
            self._ropes_pending.clear()

        marker_ids = torch.tensor(
            [self._start_of_image_id, self._end_of_image_id],
            device=pixel_device,
            dtype=torch.long,
        )
        marker_embeds = self.language_model.model.embed_tokens(marker_ids)
        start_embed = marker_embeds[0:1]
        end_embed = marker_embeds[1:2]

        results = []
        prepared_vae_inputs: list[tuple[torch.Tensor, int, int]] = []
        vit_patch_chunks: list[torch.Tensor] = []
        vit_grid_rows: list[list[int]] = []
        precomputed_vit_pixel_values = multimodal_input.get("pixel_values_vit")
        precomputed_vit_grid_thw = multimodal_input.get("image_grid_thw_vit")
        debug_img2img = _bagel_img2img_debug_enabled()

        for i, single_pv in enumerate(pixel_value_items):
            raw_h, raw_w = single_pv.shape[2:]
            vae_h, vae_w = _bagel_force_img2img_vae_hw(
                raw_h,
                raw_w,
                latent_downsample=self.latent_downsample,
                max_latent_size=self.max_latent_size,
            )
            if (raw_h, raw_w) != (vae_h, vae_w):
                single_pv = torch.nn.functional.interpolate(
                    single_pv,
                    size=(vae_h, vae_w),
                    mode="bicubic",
                    align_corners=False,
            )
            prepared_vae_inputs.append((single_pv, vae_h, vae_w))

            if include_vit and precomputed_vit_pixel_values is None:
                vit_h, vit_w = _bagel_force_img2img_vit_hw(vae_h, vae_w)
                vit_pixel_values = torch.nn.functional.interpolate(
                    single_pv,
                    size=(vit_h, vit_w),
                    mode="bicubic",
                    align_corners=False,
                )
                vit_patch_chunks.append(
                    patchify(vit_pixel_values[0], self.config.vit_config.patch_size)
                )
                vit_grid_rows.append(
                    [
                        1,
                        vit_h // self.config.vit_config.patch_size,
                        vit_w // self.config.vit_config.patch_size,
                    ]
                )

        vit_embeddings_tuple = ()
        if include_vit:
            if precomputed_vit_pixel_values is not None and precomputed_vit_grid_thw is not None:
                vit_pixel_values = precomputed_vit_pixel_values
                vit_grid_thw = precomputed_vit_grid_thw
                if isinstance(vit_pixel_values, (list, tuple)):
                    vit_pixel_values = torch.cat([t for t in vit_pixel_values], dim=0)
                if isinstance(vit_grid_thw, (list, tuple)):
                    vit_grid_thw = torch.cat(
                        [t.unsqueeze(0) if t.ndim == 1 else t for t in vit_grid_thw],
                        dim=0,
                    )
                if vit_grid_thw.ndim == 1:
                    vit_grid_thw = vit_grid_thw.unsqueeze(0)
                vit_embeddings_tuple = self._process_image_input_navit(
                    {
                        "pixel_values": vit_pixel_values,
                        "image_grid_thw": vit_grid_thw.to(
                            device=pixel_device,
                            dtype=torch.long,
                        ),
                    }
                )
            elif vit_patch_chunks:
                vit_embeddings_tuple = self._process_image_input_navit(
                    {
                        "pixel_values": torch.cat(vit_patch_chunks, dim=0),
                        "image_grid_thw": torch.tensor(
                            vit_grid_rows,
                            dtype=torch.long,
                            device=pixel_values.device,
                        ),
                    }
                )

        for i, (single_pv, H, W) in enumerate(prepared_vae_inputs):

            padded_latent = self.vae.encode(single_pv)
            h = H // self.latent_downsample
            w = W // self.latent_downsample

            latent = padded_latent[0][:, : h * p, : w * p]
            latent = latent.reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)

            vae_position_ids = self.get_flattened_position_ids(
                H,
                W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            pos_embed = self.latent_pos_embed([vae_position_ids])
            packed_timesteps = torch.tensor([timestep], device=padded_latent.device)
            with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
                timestep_embeds = self.time_embedder(packed_timesteps.to(padded_latent))
            vae_embeds = self.vae2llm(latent) + timestep_embeds + pos_embed

            se = start_embed.to(vae_embeds.dtype)
            ee = end_embed.to(vae_embeds.dtype)
            if include_vit:
                vit_emb = vit_embeddings_tuple[i] if i < len(vit_embeddings_tuple) else vit_embeddings_tuple[0]
                combined = torch.cat([se, vae_embeds, ee, se, vit_emb, ee], dim=0)
                num_vit = vit_emb.shape[0] + 2 + (1 if use_vit_separator else 0)
            else:
                combined = torch.cat([se, vae_embeds, ee], dim=0)
                num_vit = 0
            results.append(combined)

            num_vae = h * w + 2  # +2 for start/end markers
            info = (num_vae, num_vit, int(H), int(W), 1 if use_vit_separator else 0)
            if debug_img2img:
                logger.info(
                    "BAGEL img2img embeddings idx=%d raw_hw=%dx%d vae_hw=%dx%d "
                    "vae_tokens=%d vit_tokens=%d combined=%d precomputed_vit=%s sep=%s",
                    i,
                    raw_h,
                    raw_w,
                    H,
                    W,
                    num_vae,
                    num_vit,
                    combined.shape[0],
                    precomputed_vit_pixel_values is not None,
                    use_vit_separator,
                )
            self._pending_img2img_info.append(info)
            self._last_img2img_info = info

        return tuple(results)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        use_mot = False
        seq_len = inputs_embeds.shape[0] if inputs_embeds is not None else positions.shape[0]

        if self._pending_img2img_info:
            positions = self._adjust_positions_for_img2img(positions, input_ids)
            use_mot = True

        elif self._last_img2img_info is not None:
            info = self._last_img2img_info
            num_vae = int(info[0])
            num_vit = int(info[1])
            num_img2img = num_vae + num_vit

            if seq_len >= num_img2img:
                self._pending_img2img_info = [info]
                positions = self._adjust_positions_for_img2img(positions, input_ids)
                use_mot = True
            else:
                rope = positions[seq_len - 1] + 1
                self._ropes_pending.append({"ropes": [rope]})

        if use_mot:
            return self._mot_forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

        rope_positions = self._bagel_vqa_rope_positions_current
        self._bagel_vqa_rope_positions_current = None
        try:
            if rope_positions is not None and rope_positions.shape == positions.shape:
                return super().forward(input_ids, rope_positions, intermediate_tensors, inputs_embeds, **kwargs)
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        finally:
            self._bagel_vqa_image_spans_current = []

    def _adjust_positions_for_img2img(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rewrite position IDs for img2img.

        Supports an optional ``pre_text_len`` prefix (thinking-mode) detected
        via the ``<|fim_middle|>`` token in *input_ids*:

            pre_text -> 0 .. M-1
            VAE      -> M       (all share)
            ViT      -> M+1     (all share)
            post_text-> M+2, M+3, ...

        When M=0 (standard img2img) this reduces to VAE->0, ViT->1, text->2..
        """
        info_list = self._pending_img2img_info
        self._pending_img2img_info = []

        if not info_list:
            self._vae_token_mask = None
            return positions

        boundaries = [0]
        for i in range(1, len(positions)):
            if positions[i] < positions[i - 1]:
                boundaries.append(i)
        boundaries.append(len(positions))

        num_requests = len(boundaries) - 1
        new_positions = positions.clone()
        vae_mask = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)

        img2img_idx = 0
        noncausal_spans: list[dict[str, int]] = []
        num_computed_tokens = self._bagel_runner_num_computed_tokens_current
        for req_idx in range(num_requests):
            start = boundaries[req_idx]
            end = boundaries[req_idx + 1]
            req_len = end - start
            num_computed = (
                int(num_computed_tokens[req_idx])
                if req_idx < len(num_computed_tokens)
                else 0
            )

            if img2img_idx < len(info_list):
                cur_info = info_list[img2img_idx]
            elif self._last_img2img_info is not None:
                cur_info = self._last_img2img_info
            else:
                cur_info = None

            if cur_info is not None:
                if len(cur_info) >= 5:
                    num_vae, num_vit, img_H, img_W, sep_count = cur_info
                else:
                    num_vae, num_vit, img_H, img_W = cur_info
                    sep_count = 0
                include_vit = num_vit > 0
                num_img2img = num_vae + num_vit if include_vit else num_vae

                if req_len >= num_img2img:
                    pre_text_len = 0
                    if input_ids is not None:
                        req_ids_slice = input_ids[start:end]
                        indices = (req_ids_slice == self._img2img_token_id).nonzero(as_tuple=True)[0]
                        if indices.numel() > 0:
                            pre_text_len = int(indices[0].item())

                    M = pre_text_len
                    img_start = start + M
                    post_text_start = img_start + num_img2img

                    if M > 0:
                        new_positions[start:img_start] = torch.arange(
                            0, M, device=positions.device, dtype=positions.dtype
                        )

                    new_positions[img_start : img_start + num_vae] = M
                    if include_vit:
                        vit_group_start = img_start + num_vae
                        vit_start = vit_group_start + sep_count
                        new_positions[vit_group_start : vit_group_start + num_vit] = M + 1

                    num_post_text = end - post_text_start
                    post_text_rope_start = M + 2 if include_vit else M + 1
                    if num_post_text > 0:
                        new_positions[post_text_start:end] = torch.arange(
                            post_text_rope_start,
                            post_text_rope_start + num_post_text,
                            device=positions.device,
                            dtype=positions.dtype,
                        )

                    vae_patches_start = img_start + 1
                    vae_patches_end = img_start + num_vae - 1
                    if vae_patches_end > vae_patches_start:
                        vae_mask[vae_patches_start:vae_patches_end] = True

                    noncausal_spans.append(
                        {
                            "req_idx": req_idx,
                            "q_start": img_start,
                            "q_end": img_start + num_vae,
                            "request_start": start,
                            "num_computed_tokens": num_computed,
                            "kv_local_end": M + num_vae,
                            "kv_end": num_computed + M + num_vae,
                        }
                    )
                    if include_vit:
                        noncausal_spans.append(
                            {
                                "req_idx": req_idx,
                                "q_start": vit_start,
                                "q_end": img_start + num_vae + num_vit,
                                "request_start": start,
                                "num_computed_tokens": num_computed,
                                "kv_local_end": M + num_vae + num_vit,
                                "kv_end": num_computed + M + num_vae + num_vit,
                            }
                        )

                    rope = post_text_rope_start + num_post_text
                    self._ropes_pending.append(
                        {
                            "ropes": [rope],
                            "image_shape": [img_H, img_W],
                            "prefill_position_count": req_len,
                            "cfg_text_kv_len": num_img2img + M,
                            "cfg_text_rope": post_text_rope_start,
                        }
                    )
                    if _bagel_img2img_debug_enabled():
                        logger.info(
                            "BAGEL img2img positions req=%d req_len=%d pre_text=%d "
                            "img_start=%d num_vae=%d num_vit=%d post_text=%d rope=%d",
                            req_idx,
                            req_len,
                            M,
                            img_start,
                            num_vae,
                            num_vit,
                            num_post_text,
                            rope,
                        )
                    img2img_idx += 1
                    continue

            rope = int(new_positions[end - 1].item()) + 1
            self._ropes_pending.append({"ropes": [rope]})

        self._vae_token_mask = vae_mask if vae_mask.any() else None
        self._img2img_noncausal_spans_current = noncausal_spans
        return new_positions

    # Session 8 (2026-05-18) attempted to add a VQA-mode counterpart to
    # _adjust_positions_for_img2img that collapsed <|vision_start|>...<|vision_end|>
    # tokens to a shared position (mirroring vlmeval reference's bagel.py:340
    # which assigns ALL vision-block tokens the same packed_position_id and
    # advances RoPE by only 1 after the whole block).
    #
    # Result: NET REGRESSION on full 100-sample MMVet:
    #   - Smoke 10-sample (concurrency=4): 77.78 (looked great)
    #   - Full 100 (chunked prefill on, concurrency=16):  61.14 (vs S5 baseline 64.66)
    #   - Full 100 (chunked prefill off, concurrency=16): 60.22
    # The smoke's 77.78 was a sample artifact: first-10 indices are all
    # equation/math problems where collapsed positions happened to help. On
    # the broader 100-sample set every category regressed (math −4.17,
    # ocr −4.71, spat −6.94, rec −4.44, know flat).
    #
    # Reverted. `_build_bagel_vqa_rope_positions` above keeps the safer
    # side-channel version available behind BAGEL_VQA_LOGICAL_ROPE=1 for
    # future experiments, but default VQA serving stays on physical positions.
    # Token/RoPE layout alone is not enough to match reference BAGEL because
    # reference image blocks are prefetched with non-causal attention.
    #
    # Diff archive: vllm_omni_serving/patches/omni_bagel_vqa_position_remap.patch
    # Snapshot:     vllm_omni_serving/patches/omni_bagel_thinker_vqa_position_fix.py.patched

    # ------------------------------------------------------------------
    # MoT (Mixture-of-Transformers) forward path
    # ------------------------------------------------------------------

    def _mot_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors,
        inputs_embeds: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        """Full forward pass with MoT routing for img2img requests.

        VAE latent patches are routed through ``*_moe_gen`` weight matrices
        while all other tokens (markers, ViT, text) use the
        standard understanding-mode weights.
        """
        qwen2_model = self.language_model.model  # Qwen2Model

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = qwen2_model.embed_input_ids(input_ids)

        residual = None
        vae_mask = self._vae_token_mask
        self._vae_token_mask = None  # consumed

        for layer in qwen2_model.layers:
            if not isinstance(layer, Qwen2DecoderLayer):
                continue  # skip PPMissingLayer (pipeline parallelism)
            hidden_states, residual = self._mot_layer_forward(
                layer,
                positions,
                hidden_states,
                residual,
                vae_mask,
            )

        # Final norm with MoT routing
        if residual is not None:
            hidden_states = hidden_states + residual
        if vae_mask is not None and vae_mask.any():
            out = torch.empty_like(hidden_states)
            non_vae = ~vae_mask
            if non_vae.any():
                out[non_vae] = qwen2_model.norm(hidden_states[non_vae])
            out[vae_mask] = qwen2_model.norm_moe_gen(hidden_states[vae_mask])
            hidden_states = out
        else:
            hidden_states = qwen2_model.norm(hidden_states)

        return hidden_states

    def _mot_layer_forward(
        self,
        layer: Qwen2DecoderLayer,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        vae_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single decoder-layer forward with MoT routing."""
        if vae_mask is None or not vae_mask.any():
            return layer(positions, hidden_states, residual)

        non_vae = ~vae_mask

        # ---- input layernorm (split) ----
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        normed = torch.empty_like(hidden_states)
        if non_vae.any():
            normed[non_vae] = layer.input_layernorm(hidden_states[non_vae])
        normed[vae_mask] = layer.input_layernorm_moe_gen(hidden_states[vae_mask])
        hidden_states = normed

        # ---- attention (split QKV / O projections) ----
        hidden_states = self._mot_attn_forward(layer.self_attn, positions, hidden_states, vae_mask)

        # ---- post-attention layernorm (split) ----
        hidden_states = hidden_states + residual
        residual = hidden_states
        normed = torch.empty_like(hidden_states)
        if non_vae.any():
            normed[non_vae] = layer.post_attention_layernorm(hidden_states[non_vae])
        normed[vae_mask] = layer.post_attention_layernorm_moe_gen(hidden_states[vae_mask])
        hidden_states = normed

        # ---- MLP (split) ----
        mlp_out = torch.empty_like(hidden_states)
        if non_vae.any():
            mlp_out[non_vae] = layer.mlp(hidden_states[non_vae])
        mlp_out[vae_mask] = layer.mlp_moe_gen(hidden_states[vae_mask])
        hidden_states = mlp_out

        return hidden_states, residual

    def _mot_attn_forward(
        self,
        attn,  # Qwen2Attention
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        vae_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention forward with MoT routing for QKV and O projections."""
        non_vae = ~vae_mask
        qkv_dim = attn.q_size + 2 * attn.kv_size

        # ---- QKV projection (split) ----
        qkv = torch.empty(
            hidden_states.shape[0],
            qkv_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if non_vae.any():
            qkv_und, _ = attn.qkv_proj(hidden_states[non_vae])
            qkv[non_vae] = qkv_und
        qkv_gen, _ = attn.qkv_proj_moe_gen(hidden_states[vae_mask])
        qkv[vae_mask] = qkv_gen

        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

        # ---- QK normalization (split) ----
        if attn.qk_norm:
            n_tok = q.shape[0]
            q = q.view(n_tok, attn.num_heads, attn.head_dim)
            k = k.view(n_tok, attn.num_kv_heads, attn.head_dim)

            q_out = torch.empty_like(q)
            k_out = torch.empty_like(k)
            if non_vae.any():
                q_out[non_vae] = attn.q_norm(q[non_vae])
                k_out[non_vae] = attn.k_norm(k[non_vae])
            q_out[vae_mask] = attn.q_norm_moe_gen(q[vae_mask])
            k_out[vae_mask] = attn.k_norm_moe_gen(k[vae_mask])

            q = q_out.reshape(n_tok, attn.q_size)
            k = k_out.reshape(n_tok, attn.kv_size)

        # ---- RoPE + attention (same for all tokens) ----
        q, k = attn.rotary_emb(positions, q, k)
        attn_output = attn.attn(q, k, v)
        spans = self._img2img_noncausal_spans_current
        if spans:
            if _env_flag("BAGEL_IMG2IMG_NONCAUSAL_RECOMPUTE", False):
                try:
                    attn_output = _bagel_recompute_noncausal_image_blocks_paged(
                        attn,
                        q,
                        k,
                        v,
                        attn_output,
                        spans,
                    )
                except Exception:
                    # Keep the experiment debuggable on non-paged/mock
                    # backends; real vLLM prefill should use the paged KV path
                    # above so image tokens can still see transferred/prefix KV.
                    attn_output = _bagel_vqa_recompute_noncausal_image_blocks_direct(
                        attn,
                        q,
                        k,
                        v,
                        attn_output,
                        spans,
                        "img2img",
                    )

        # ---- O projection (split) ----
        output = torch.empty(
            hidden_states.shape[0],
            attn.hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if non_vae.any():
            o_und, _ = attn.o_proj(attn_output[non_vae])
            output[non_vae] = o_und
        o_gen, _ = attn.o_proj_moe_gen(attn_output[vae_mask])
        output[vae_mask] = o_gen

        return output

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        generation_keywords_to_skip = [
            "llm2vae",
            "decoder.",
        ]

        def _map_vae_weight_name(name: str) -> str:
            if name.startswith("encoder."):
                return "vae." + name
            if name.startswith("reg."):
                return "vae." + name
            return name

        moe_gen_weights: list[tuple[str, torch.Tensor]] = []
        filtered_weights = []

        for name, tensor in weights:
            if any(skip in name for skip in generation_keywords_to_skip):
                continue

            mapped_name = _map_vae_weight_name(name)

            if "moe_gen" in mapped_name:
                moe_gen_weights.append((mapped_name, tensor))
                continue

            if "patch_embedding.weight" in mapped_name and tensor.ndim == 2:
                out_channels = tensor.shape[0]
                in_features = tensor.shape[1]
                patch_size = self.config.vit_config.patch_size
                in_channels = self.config.vit_config.num_channels
                if in_features == in_channels * patch_size * patch_size:
                    tensor = tensor.reshape(out_channels, patch_size, patch_size, in_channels)
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()

            if "latent_pos_embed.pos_embed" in mapped_name and tensor.ndim == 2:
                npos, hdim = tensor.shape
                current_param = self.latent_pos_embed.pos_embed
                if current_param.shape != tensor.shape:
                    side = isqrt(int(npos))
                    if side * side == int(npos) and hdim == current_param.shape[1]:
                        current_param.data = current_param.data.new_empty((npos, hdim))
                        self.max_latent_size = int(side)
                        setattr(self.config, "max_latent_size", int(side))
                        if hasattr(self.latent_pos_embed, "max_num_patch_per_side"):
                            self.latent_pos_embed.max_num_patch_per_side = int(side)

            filtered_weights.append((mapped_name, tensor))

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["vit_pos_embed.pos_embed"],
            ignore_unexpected_prefixes=["vae.", "latent_pos_embed.", "time_embedder.", "vae2llm."],
        )
        loaded = loader.load_weights(filtered_weights, mapper=self.hf_to_vllm_mapper)

        loaded |= self._load_moe_gen_weights(moe_gen_weights)

        return loaded

    def _load_moe_gen_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load generation-mode MoT weights with proper stacked-param mapping."""
        stacked_params = [
            ("qkv_proj_moe_gen", "q_proj_moe_gen", "q"),
            ("qkv_proj_moe_gen", "k_proj_moe_gen", "k"),
            ("qkv_proj_moe_gen", "v_proj_moe_gen", "v"),
            ("mlp_moe_gen.gate_up_proj", "mlp_moe_gen.gate_proj", 0),
            ("mlp_moe_gen.gate_up_proj", "mlp_moe_gen.up_proj", 1),
        ]

        mapper = self.hf_to_vllm_mapper
        prefix_map = getattr(mapper, "orig_to_new_prefix", {})

        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, tensor in weights:
            mapped = name
            for orig, new in prefix_map.items():
                if mapped.startswith(orig):
                    mapped = new + mapped[len(orig) :]
                    break

            found_stacked = False
            for param_name, weight_name, shard_id in stacked_params:
                if weight_name not in mapped:
                    continue
                mapped = mapped.replace(weight_name, param_name)
                if mapped in params_dict:
                    param = params_dict[mapped]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor, shard_id)
                    loaded.add(mapped)
                found_stacked = True
                break

            if not found_stacked:
                if mapped in params_dict:
                    param = params_dict[mapped]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)
                    loaded.add(mapped)

        return loaded

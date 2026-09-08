# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable, Mapping, Sequence
from math import isqrt
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import ModalityData, MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bagel import BagelForConditionalGeneration
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMRoPE
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
)
from vllm_omni.diffusion.models.bagel.pipeline_bagel import bagel_image_size, bagel_vit_transform, default_ae_params

logger = init_logger(__name__)

# One img2img image: tokens of its VAE and ViT blocks including the <|vision_start|> / <|vision_end|>
# markers, and the stride-aligned (H, W) the DiT stage generates at.
Img2ImgInfo = tuple[int, int, int, int]


class OmniBagelProcessor(BagelProcessor):
    # transformers>=5.0 ProcessorMixin.get_attributes() only scans the leaf
    # class's __dict__ for ``<attribute>_class`` hints; redeclare them here
    # so from_pretrained() correctly sets ``self.image_processor`` and
    # ``self.tokenizer`` on the OmniBagelProcessor instance.
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __call__(self, text=None, images=None, **kwargs):
        is_img2img = kwargs.pop("is_img2img", False)

        from vllm.transformers_utils.processors.bagel import BagelProcessorKwargs

        if is_img2img and images is not None:
            # transformers>=5.0 enforces strict kwarg typing on image
            # processors, so split generic kwargs into text/image buckets
            # via the standard ProcessorMixin helper before dispatch.
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
            elif pixel_values is not None:
                return pixel_values
            elif text_inputs is not None:
                return BatchFeature(dict(text_inputs))
            else:
                return BatchFeature({})

        if images is not None:
            output_kwargs = self._merge_kwargs(
                BagelProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
            )
            pixel_values = [bagel_vit_transform(img) for img in (images if isinstance(images, list) else [images])]
            text_inputs = dict(self.tokenizer(text, **output_kwargs["text_kwargs"])) if text is not None else {}
            return BatchFeature({"pixel_values": pixel_values, **text_inputs})
        return super().__call__(text, images, **kwargs)


class OmniBagelProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "img2img": 1}

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
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "pixel_values_img2img": MultiModalFieldConfig.batched("img2img"),
        }

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

        vit_config = hf_config.vit_config
        image_size, patch_size = vit_config.image_size, vit_config.patch_size

        def num_vit_tokens(width: int, height: int) -> int:
            """<|vision_start|> + patches of the aspect-preserving resize + <|vision_end|>."""
            w, h = bagel_image_size(width, height, image_size, 224, patch_size)
            return (h // patch_size) * (w // patch_size) + 2

        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
        if image_token_id is not None:

            def get_image_replacement(item_idx: int):
                size = mm_items.get_items("image", ImageProcessorItems).get_image_size(item_idx)
                return [image_token_id] * num_vit_tokens(size.width, size.height)

            replacements.append(
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_image_replacement,
                )
            )

        img2img_token_id = tokenizer.get_vocab().get("<|fim_middle|>")
        if img2img_token_id is not None:
            latent_patch_size = getattr(hf_config, "latent_patch_size", 2)
            downsample = hf_config.vae_config.get("downsample", 8)
            latent_downsample = downsample * latent_patch_size

            def get_img2img_replacement(item_idx: int):
                h, w = image_size, image_size
                if "img2img" in mm_items:
                    item = mm_items.get_items("img2img", (Img2ImgProcessorItems, ImageEmbeddingItems))
                    if hasattr(item, "get_image_size"):
                        size = item.get_image_size(item_idx)
                        h, w = size.height, size.width

                max_latent_size = getattr(hf_config, "max_latent_size", 32)
                max_img_size = int(max_latent_size * latent_downsample)
                stride = latent_downsample
                scale = min(max_img_size / max(h, w), 1.0)
                min_img_size = min(256, max_img_size)
                scale = max(scale, min_img_size / min(h, w))
                new_h = max(stride, int(round(h * scale / stride) * stride))
                new_w = max(stride, int(round(w * scale / stride) * stride))
                new_h = min(new_h, max_img_size)
                new_w = min(new_w, max_img_size)

                num_vae_patches = (new_h // latent_downsample) * (new_w // latent_downsample)
                num_vae_total = num_vae_patches + 2
                num_vit_total = num_vit_tokens(w, h)
                # +1 separator between VAE and ViT blocks so that
                # extract_embeds_range() produces two distinct mm_prefix_range
                # entries, preventing VAE tokens from attending to ViT.
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


@MULTIMODAL_REGISTRY.register_processor(
    OmniBagelMultiModalProcessor,
    info=OmniBagelProcessingInfo,
    dummy_inputs=OmniBagelDummyInputsBuilder,
)
class OmniBagelForConditionalGeneration(BagelForConditionalGeneration, SupportsMRoPE):
    """
    Omni version of BagelForConditionalGeneration.

    Extends the base model with a VAE encoder so that img2img can embed
    both VAE latents and ViT features within the AR stage, producing a
    combined KV cache that is then transferred to the DiT stage.

    RoPE positions follow BAGEL (modeling/bagel/bagel.py): every image block --
    <|vision_start|> + ViT patches + <|vision_end|>, or the VAE / ViT block of
    img2img -- occupies one position and the text continues from the next one,
    so the KV cache + ropes handed to the DiT stage match the single-stage
    pipeline. They are computed per request in get_mrope_input_positions (the
    LLM itself uses plain 1-D RoPE; the three rows are identical).
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

        self._ropes_pending: list[dict[str, Any]] = []
        self._ropes_metadata: dict[str, dict[str, Any]] = {}
        self._reset_img2img_state()

        from transformers import AutoTokenizer

        tok_name = getattr(vllm_config.model_config, "tokenizer", None) or vllm_config.model_config.model
        _tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        for t in ["<|vision_start|>", "<|vision_end|>"]:
            if t not in _tok.get_vocab():
                _tok.add_tokens([t])
        self._start_of_image_id = int(_tok.convert_tokens_to_ids("<|vision_start|>"))
        self._end_of_image_id = int(_tok.convert_tokens_to_ids("<|vision_end|>"))
        self._img2img_token_id = int(_tok.convert_tokens_to_ids("<|fim_middle|>"))
        self.device = get_local_device()
        self._install_mot_modules(config)

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

    def _reset_img2img_state(self) -> None:
        """img2img bookkeeping, see _route_img2img."""
        # Each img2img image the encoder embedded in the current step, in encoder order.
        self._pending_img2img_info: list[Img2ImgInfo] = []
        # The same keyed by (num_vae, num_vit), most recent last, for requests whose image the encoder
        # cache already held: a CFG companion shares its parent's image, a resumed request its own.
        self._img2img_info_by_size: dict[tuple[int, int], Img2ImgInfo] = {}
        # (block start in the prompt, info) per request, so every chunk of a chunked prefill routes
        # the same tokens through the generation expert.
        self._img2img_by_req: dict[str, tuple[int, Img2ImgInfo]] = {}
        # (req_id, first row, end row, tokens computed before this step) per request of the batch
        # about to run, recorded by prepare_runner_inputs; None on dummy runs.
        self._batch_layout: list[tuple[str, int, int, int]] | None = None
        # VAE patches of the batch about to run (None: none) and, as plain bools so the per-layer MoT
        # routing branches without a device sync, whether it holds any VAE / non-VAE rows.
        self._vae_token_mask: torch.Tensor | None = None
        self._has_vae_tokens: bool = False
        self._has_non_vae_tokens: bool = True

    def _clear_warmup_state(self):
        """Clear stale state accumulated during warmup/profiling runs."""
        self._ropes_pending.clear()
        self._ropes_metadata.clear()
        self._reset_img2img_state()

    def get_kv_transfer_metadata(
        self,
        req_id: str,
        *,
        num_computed_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        # NOTE: num_computed_tokens will not include async placeholders
        self._img2img_by_req.pop(req_id, None)
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
        req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        input_ids_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Restore input_ids (the runner hands a multimodal batch over as embeddings only) so that
        _route_img2img sees the <|fim_middle|> placeholders, and record which rows of the batch
        belong to which request."""
        if inputs_embeds is not None and input_ids is None and input_ids_buffer is not None:
            input_ids = input_ids_buffer
        layout: list[tuple[str, int, int, int]] = []
        start = 0
        for req_id, computed, scheduled in zip(req_ids, num_computed_tokens, num_scheduled_tokens):
            layout.append((req_id, start, start + int(scheduled), int(computed)))
            start += int(scheduled)
        self._batch_layout = layout
        return input_ids, positions

    def get_mrope_input_positions(self, input_tokens: list[int], mm_features: list) -> tuple[torch.Tensor, int]:
        """One RoPE position per image block (BAGEL prepare_vit_images / prepare_vae_images: markers and
        patches share it): a token advances the position unless it continues a multimodal placeholder;
        a placeholder's second embedded range (img2img's ViT block after its VAE block) starts a new one,
        and the plain <|vision_end|> that closes the VAE block keeps that block's position."""
        step = torch.ones(len(input_tokens), dtype=torch.long)
        for f in mm_features:
            step[f.mm_position.offset + 1 : f.mm_position.offset + f.mm_position.length] = 0
            for start, _ in f.mm_position.extract_embeds_range()[1:]:
                step[start] = 1
        positions = step.cumsum(0) - 1
        return positions.unsqueeze(0).repeat(3, 1), int(positions[-1]) + 1 - len(input_tokens)

    def flush_pending_metadata(self, req_ids: Sequence[str]) -> None:
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

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        pixel_values = kwargs.pop("pixel_values", None)
        return None if pixel_values is None else {"type": "pixel_values", "pixel_values": pixel_values}

    @staticmethod
    def _image_list(pixel_values) -> list[torch.Tensor]:
        """(N, 3, H, W) / (B, N, 3, H, W) tensors, or lists of them for mixed sizes -> one (3, H, W) per image."""
        if not isinstance(pixel_values, (list, tuple)):
            pixel_values = [pixel_values]
        return [img for t in pixel_values for img in t.reshape(-1, *t.shape[-3:])]

    def _vit_embeddings(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """SigLIP NaViT as in modeling/bagel/siglip_navit.py: each image is its own sequence with
        2-D position ids into the 70x70 table, then post_layernorm, connector and BAGEL's sin-cos
        position embedding on the same grid."""
        vit = self.vit_model.vision_model
        patch, side = self.config.vit_config.patch_size, self.config.vit_max_num_patch_per_side
        out = []
        assert len(images) > 0
        for img in images:
            ids = self.get_flattened_position_ids(img.shape[-2], img.shape[-1], patch, side).to(img.device)
            x = vit.embeddings.patch_embedding(img[None].to(vit.embeddings.patch_embedding.weight.dtype))
            x = x.flatten(2).transpose(1, 2) + vit.embeddings.position_embedding(ids)
            x = vit.maybe_layer_norm_and_apply_head(vit.encoder(inputs_embeds=x, return_all_hidden_states=False))
            out.append(self.connector(x[0]) + self.vit_pos_embed(ids).to(x.device))
        return out

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        if any(k in kwargs for k in ("pixel_values", "image_embeds")):
            mm_input_by_modality["img2text"] = self._parse_and_validate_image_input(**kwargs)

        img2img_keys = {"pixel_values_img2img": "pixel_values", "image_embeds_img2img": "image_embeds"}
        img2img_kwargs = {img2img_keys[k]: v for k, v in kwargs.items() if k in img2img_keys}

        if img2img_kwargs:
            combined_kwargs = kwargs.copy()
            combined_kwargs.update(img2img_kwargs)
            mm_input_by_modality["img2img"] = self._parse_and_validate_image_input(**combined_kwargs)

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

    def _process_img2text_input(self, multimodal_input) -> tuple[torch.Tensor, ...]:
        images = self._image_list(multimodal_input["pixel_values"])
        marker_ids = torch.tensor([self._start_of_image_id, self._end_of_image_id], device=images[0].device)
        start, end = self.language_model.model.embed_tokens(marker_ids).split(1)
        return tuple(torch.cat([start.to(e.dtype), e, end.to(e.dtype)]) for e in self._vit_embeddings(images))

    def _process_img2img_input(self, multimodal_input):
        pixel_values = self._image_list(multimodal_input["pixel_values"])
        num_images = len(pixel_values)
        image_size, patch_size = self.config.vit_config.image_size, self.config.vit_config.patch_size
        p = self.latent_patch_size
        timestep = 0

        if self._ropes_pending:
            self._ropes_pending.clear()

        vit_embeddings_tuple = self._vit_embeddings(
            [
                torch.nn.functional.interpolate(
                    pv[None],
                    size=bagel_image_size(pv.shape[-1], pv.shape[-2], image_size, 224, patch_size)[::-1],
                    mode="bicubic",
                    align_corners=False,
                )[0]
                for pv in pixel_values
            ]
        )
        marker_ids = torch.tensor([self._start_of_image_id, self._end_of_image_id], device=pixel_values[0].device)
        start_embed, end_embed = self.language_model.model.embed_tokens(marker_ids).split(1)

        results = []

        for i in range(num_images):
            single_pv = pixel_values[i][None]
            single_pv = self._resize_to_stride(single_pv)
            H, W = single_pv.shape[2:]

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

            vit_emb = vit_embeddings_tuple[i] if i < len(vit_embeddings_tuple) else vit_embeddings_tuple[0]

            se = start_embed.to(vae_embeds.dtype)
            ee = end_embed.to(vae_embeds.dtype)
            combined = torch.cat([se, vae_embeds, ee, se, vit_emb, ee], dim=0)
            results.append(combined)

            num_vae = h * w + 2  # +2 for start/end markers
            num_vit = vit_emb.shape[0] + 2
            self._register_img2img_info((num_vae, num_vit, int(H), int(W)))

        return tuple(results)

    def _register_img2img_info(self, info: Img2ImgInfo) -> None:
        """Record one encoder run for this step's routing and, by size, for the requests the encoder
        cache serves later."""
        self._pending_img2img_info.append(info)
        key = (info[0], info[1])
        # most recent last: _match_img2img_info prefers it when several sizes fit a cut block
        self._img2img_info_by_size.pop(key, None)
        self._img2img_info_by_size[key] = info

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if positions.ndim == 2:
            positions = positions[0]
        if self._route_img2img(input_ids, positions):
            return self._mot_forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def _route_img2img(self, input_ids: torch.Tensor | None, positions: torch.Tensor) -> bool:
        """Set up the MoT routing of this batch; True when it holds VAE patches for the generation
        expert. Gated per request on the <|fim_middle|> placeholder in the request's own rows of
        *input_ids*, so a text or img2text request batched with an img2img one, or running after it,
        keeps the understanding expert. Only batches the runner described through
        prepare_runner_inputs are inspected: a dummy run (profiling, CUDA-graph capture) carries no
        layout and must not sync the device."""
        layout = self._batch_layout
        self._batch_layout = None
        self._vae_token_mask = None
        self._has_vae_tokens = False
        self._has_non_vae_tokens = True
        is_img2img = None
        if (
            layout
            and input_ids is not None
            and (self._pending_img2img_info or self._img2img_info_by_size or self._img2img_by_req)
        ):
            # Rows past the batch's own are CUDA-graph padding; the buffer still holds earlier tokens there.
            is_img2img = input_ids[: layout[-1][2]] == self._img2img_token_id
            if not bool(is_img2img.any()):
                is_img2img = None
        if is_img2img is None:
            # Encoder output no request of this batch consumes (profiling, an aborted request) must
            # not leak onto a later batch.
            self._pending_img2img_info.clear()
            return False
        self._prepare_img2img(positions, is_img2img, layout)
        return self._has_vae_tokens

    def _prepare_img2img(
        self,
        positions: torch.Tensor,
        is_img2img: torch.Tensor,
        layout: list[tuple[str, int, int, int]],
    ) -> None:
        """Per request of the batch (``layout``: req_id, first row, end row, tokens computed before
        this step): mark its VAE patches in ``_vae_token_mask`` and queue the rope the DiT stage
        continues from. A request is img2img only where *is_img2img* marks its own rows. Its
        ``(num_vae, num_vit, H, W)`` is this step's encoder output matched on the block layout the
        rows show -- exact once the block is complete, a lower bound while a chunk cuts it -- or,
        when the encoder cache served the image, the same size seen before; the block start is kept
        per request so later chunks of the same prefill mark the right rows."""
        pending = self._pending_img2img_info
        self._pending_img2img_info = []
        num_tokens = layout[-1][2]
        # Two host copies for the batch instead of a device sync per request.
        pos_list = positions[:num_tokens].tolist()
        tok_list = is_img2img.tolist()

        # [req_id, first row, end row, computed, (block start, info) | None]
        slices: list[list[Any]] = []
        fresh: list[tuple[int, tuple[int, int, bool, int, bool]]] = []
        for req_id, start, end, computed in layout:
            state = self._img2img_by_req.get(req_id)
            if state is not None and computed == 0:
                # prefilled again from scratch (preemption): the earlier prefill's block is gone
                del self._img2img_by_req[req_id]
                state = None
            slices.append([req_id, start, end, computed, state])
            if state is None:
                block = self._visible_img2img_block(tok_list[start:end], pos_list[start:end])
                if block is not None:
                    fresh.append((len(slices) - 1, block))

        # Complete blocks match their encoder output exactly and go first; a block a chunk cuts
        # only bounds its size and takes what is left.
        for idx, block in sorted(fresh, key=lambda item: not (item[1][2] and item[1][4])):
            req_id, _, _, computed, _ = slices[idx]
            info = self._match_img2img_info(block, pending)
            if info is None:
                info = self._provisional_img2img_info(block)
                logger.warning(
                    "No encoder output matches the img2img block of request %s (visible layout %s); "
                    "routing its visible VAE rows and leaving the DiT image size to the request",
                    req_id,
                    block,
                )
            state = (computed + block[0], info)
            slices[idx][4] = state
            self._img2img_by_req[req_id] = state
        if pending:
            logger.debug("Dropping %d img2img encoder outputs no request of this batch consumed", len(pending))

        vae_mask = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
        num_vae_rows = 0
        for req_id, start, end, computed, state in slices:
            rope = pos_list[end - 1] + 1
            if state is None:
                self._ropes_pending.append({"ropes": [rope]})
                continue
            block_start, (num_vae, _, img_h, img_w) = state
            # the VAE block without its markers, clipped to this step's chunk of the request
            lo = max(block_start + 1, computed)
            hi = min(block_start + num_vae - 1, computed + end - start)
            if hi > lo:
                vae_mask[start + lo - computed : start + hi - computed] = True
                num_vae_rows += hi - lo
            meta: dict[str, Any] = {"ropes": [rope], "prefill_position_count": computed + end - start}
            if img_h and img_w:
                meta["image_shape"] = [img_h, img_w]
            self._ropes_pending.append(meta)

        self._has_vae_tokens = num_vae_rows > 0
        self._has_non_vae_tokens = num_vae_rows < positions.shape[0]
        self._vae_token_mask = vae_mask if self._has_vae_tokens else None

    @staticmethod
    def _visible_img2img_block(tok: list[bool], pos: list[int]) -> tuple[int, int, bool, int, bool] | None:
        """Layout of the img2img block that starts in these rows, or None: (first row, rows of the VAE
        group, whether it is complete, rows of the ViT group, whether it is complete). The VAE block
        and its separator share one RoPE position and the ViT block takes the next
        (get_mrope_input_positions), which tells the groups apart; a group is complete when the rows
        continue past it."""
        if True not in tok:
            return None
        first = tok.index(True)
        n = len(tok)
        i = first
        while i < n and tok[i] and pos[i] == pos[first]:
            i += 1
        j = i
        while j < n and tok[j] and pos[j] == pos[first] + 1:
            j += 1
        return first, i - first, i < n, j - i, j < n

    def _match_img2img_info(
        self, block: tuple[int, int, bool, int, bool], pending: list[Img2ImgInfo]
    ) -> Img2ImgInfo | None:
        """The encoder output describing *block*: one of this step's, else the same size seen before
        (the encoder cache serves a CFG companion, which shares its parent's image, and a resumed
        request without running the encoder again). A complete group must match exactly, a cut one
        bounds the size."""
        _, n_vae, vae_done, n_vit, vit_done = block

        def fits(info: Img2ImgInfo) -> bool:
            vae_rows, vit_rows = info[0] + 1, info[1]  # the VAE group also holds the separator
            return (vae_rows == n_vae if vae_done else vae_rows >= n_vae) and (
                vit_rows == n_vit if vit_done else vit_rows >= n_vit
            )

        for i, info in enumerate(pending):
            if fits(info):
                return pending.pop(i)
        for info in reversed(self._img2img_info_by_size.values()):
            if fits(info):
                return info
        return None

    @staticmethod
    def _provisional_img2img_info(block: tuple[int, int, bool, int, bool]) -> Img2ImgInfo:
        """Best guess from the visible rows alone: every VAE-group row after the start marker is a
        patch, so a cut group is read as ending right after its last visible row; no image size."""
        _, n_vae, vae_done, n_vit, _ = block
        return (n_vae - 1 if vae_done else n_vae + 1), n_vit, 0, 0

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
        while all other tokens (markers, ViT, separator, text) use the
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
        if vae_mask is not None and self._has_vae_tokens:
            out = torch.empty_like(hidden_states)
            non_vae = ~vae_mask
            if self._has_non_vae_tokens:
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
        if vae_mask is None or not self._has_vae_tokens:
            return layer(positions, hidden_states, residual)

        non_vae = ~vae_mask

        # ---- input layernorm (split) ----
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        normed = torch.empty_like(hidden_states)
        if self._has_non_vae_tokens:
            normed[non_vae] = layer.input_layernorm(hidden_states[non_vae])
        normed[vae_mask] = layer.input_layernorm_moe_gen(hidden_states[vae_mask])
        hidden_states = normed

        # ---- attention (split QKV / O projections) ----
        hidden_states = self._mot_attn_forward(layer.self_attn, positions, hidden_states, vae_mask)

        # ---- post-attention layernorm (split) ----
        hidden_states = hidden_states + residual
        residual = hidden_states
        normed = torch.empty_like(hidden_states)
        if self._has_non_vae_tokens:
            normed[non_vae] = layer.post_attention_layernorm(hidden_states[non_vae])
        normed[vae_mask] = layer.post_attention_layernorm_moe_gen(hidden_states[vae_mask])
        hidden_states = normed

        # ---- MLP (split) ----
        mlp_out = torch.empty_like(hidden_states)
        if self._has_non_vae_tokens:
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
        if self._has_non_vae_tokens:
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
            if self._has_non_vae_tokens:
                q_out[non_vae] = attn.q_norm(q[non_vae])
                k_out[non_vae] = attn.k_norm(k[non_vae])
            q_out[vae_mask] = attn.q_norm_moe_gen(q[vae_mask])
            k_out[vae_mask] = attn.k_norm_moe_gen(k[vae_mask])

            q = q_out.reshape(n_tok, attn.q_size)
            k = k_out.reshape(n_tok, attn.kv_size)

        # ---- RoPE + attention (same for all tokens) ----
        q, k = attn.rotary_emb(positions, q, k)
        attn_output = attn.attn(q, k, v)

        # ---- O projection (split) ----
        output = torch.empty(
            hidden_states.shape[0],
            attn.hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if self._has_non_vae_tokens:
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

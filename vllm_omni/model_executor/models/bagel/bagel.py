from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from math import isqrt
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
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
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ModalityData,
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
from vllm_omni.diffusion.models.bagel.pipeline_bagel import default_ae_params


class OmniBagelProcessor(BagelProcessor):
    def __call__(self, text=None, images=None, **kwargs):
        is_img2img = kwargs.pop("is_img2img", False)

        if is_img2img and images is not None:
            image_kwargs = kwargs.copy()
            image_kwargs["do_resize"] = False
            image_kwargs["do_rescale"] = True
            if "return_tensors" not in image_kwargs:
                image_kwargs["return_tensors"] = "pt"

            pixel_values = self.image_processor(images, **image_kwargs)

            text_inputs = self.tokenizer(text, **kwargs) if text is not None else None

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

        return super().__call__(text, images, **kwargs)


class OmniBagelProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1, "img2img": 1}

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
            kwargs_img2img = dict(mm_kwargs)
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
            mm_kwargs = dict(mm_kwargs)
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
        if image_token_id is not None:
            num_patches = hf_config.vit_max_num_patch_per_side**2

            def get_image_replacement(item_idx: int):
                return [image_token_id] * num_patches

            replacements.append(
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_image_replacement,
                )
            )

        img2img_token_id = tokenizer.get_vocab().get("<|fim_middle|>")
        if img2img_token_id is not None:
            vit_config = hf_config.vit_config
            image_size = vit_config.image_size
            num_vit_patches = (image_size // vit_config.patch_size) ** 2

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
                num_vit_total = num_vit_patches + 2
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

        self._pending_img2img_info: list[tuple[int, int, int, int]] = []
        self._ropes_pending: list[dict[str, Any]] = []
        self._ropes_metadata: dict[str, dict[str, Any]] = {}
        self._cfg_companion_queue: deque[tuple[tuple[int, int, int, int], int]] = deque()

        # Per-request position offset for decode after img2img prefill.
        # Prefill rewrites positions (VAE→0, ViT→1, text→2..N) but the model
        # runner assigns decode positions starting from prefill_len, not N+1.
        # offset = rope - prefill_len (a negative number).
        self._pending_decode_offsets: list[int] = []
        self._decode_position_offsets: dict[str, int] = {}

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

    def _clear_warmup_state(self):
        """Clear stale state accumulated during warmup/profiling runs."""
        self._ropes_pending.clear()
        self._ropes_metadata.clear()
        self._pending_img2img_info.clear()
        self._cfg_companion_queue.clear()
        self._pending_decode_offsets.clear()
        self._decode_position_offsets.clear()
        self._vae_token_mask = None

    def get_kv_transfer_metadata(
        self,
        req_id: str,
        *,
        num_computed_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        meta = self._ropes_metadata.pop(req_id, None)
        if meta is None:
            return None
        # In think-mode img2img the prefill rope doesn't account for decoded
        # thinking tokens; correct it to num_computed_tokens + offset.
        # Skip correction when num_computed_tokens is unavailable (None).
        offset = self._decode_position_offsets.pop(req_id, 0)
        if offset != 0 and "ropes" in meta and num_computed_tokens is not None:
            meta["ropes"] = [num_computed_tokens + offset]
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
        """Model-runner hook: adjust inputs before ``forward()``.

        Returns ``(input_ids, positions)`` — possibly modified.

        Two adjustments for BAGEL img2img:

        1. **Restore input_ids** when ``inputs_embeds`` is present so that
           ``_adjust_positions_for_img2img`` can locate the
           ``<|fim_middle|>`` placeholder.
        2. **Decode position offset**: prefill rewrites positions to a
           compact scheme (rope ≪ prefill_len).  The runner assigns decode
           positions from ``num_computed_tokens``, which is far too large;
           apply the stored per-request offset.
        """
        if inputs_embeds is not None and input_ids is None and input_ids_buffer is not None:
            input_ids = input_ids_buffer

        if self._decode_position_offsets and positions is not None:
            token_start = 0
            for i, rid in enumerate(req_ids):
                sched = num_scheduled_tokens[i]
                offset = self._decode_position_offsets.get(rid, 0)
                if offset != 0 and num_computed_tokens[i] > 0:
                    positions[token_start : token_start + sched] += offset
                token_start += sched

        return input_ids, positions

    def flush_pending_metadata(self, req_ids: list[str]) -> None:
        """Map pending metadata (batch order) to req_ids after forward()."""
        pending = self._ropes_pending
        self._ropes_pending = []
        for i, meta in enumerate(pending):
            if i < len(req_ids):
                if req_ids[i] not in self._ropes_metadata:
                    self._ropes_metadata[req_ids[i]] = meta

        pending_offsets = self._pending_decode_offsets
        self._pending_decode_offsets = []
        for i, offset in enumerate(pending_offsets):
            if i < len(req_ids) and offset != 0:
                self._decode_position_offsets[req_ids[i]] = offset

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

    def _process_img2text_input(self, multimodal_input):
        return self._process_image_input(multimodal_input)

    def _process_img2img_input(self, multimodal_input):
        pixel_values = multimodal_input["pixel_values"]
        if pixel_values.ndim == 5:
            b, n, c, h, w = pixel_values.shape
            pixel_values = pixel_values.reshape(b * n, c, h, w)

        num_images = pixel_values.shape[0]
        image_size = self.config.vit_config.image_size
        p = self.latent_patch_size
        timestep = 0

        if self._ropes_pending:
            self._ropes_pending.clear()

        vit_pixel_values = torch.nn.functional.interpolate(
            pixel_values,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        )

        vit_embeddings_tuple = self._process_image_input({"pixel_values": vit_pixel_values})

        marker_ids = torch.tensor(
            [self._start_of_image_id, self._end_of_image_id],
            device=pixel_values.device,
            dtype=torch.long,
        )
        marker_embeds = self.language_model.model.embed_tokens(marker_ids)
        start_embed = marker_embeds[0:1]
        end_embed = marker_embeds[1:2]

        results = []

        for i in range(num_images):
            single_pv = pixel_values[i : i + 1]
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
            info = (num_vae, num_vit, int(H), int(W))
            self._pending_img2img_info.append(info)
            # Only the gen (main) request should add a companion queue entry.
            # Companion requests (cfg_text, cfg_img) also call this method with
            # the same image, so guard by checking whether this exact info
            # tuple is already enqueued.  For batched img2img with multiple
            # concurrent gen requests this correctly adds one entry per unique
            # image; images with identical (num_vae, num_vit, H, W) that arrive
            # in the same batch are indistinguishable here and will share one
            # entry, but that is an uncommon edge case.
            if not any(entry[0] == info for entry in self._cfg_companion_queue):
                self._cfg_companion_queue.append((info, 2))  # cfg_text + cfg_img

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

        elif self._cfg_companion_queue:
            # Guard: if this looks like a pure decode step (small token count,
            # no multimodal embeddings), the queue has stale entries from a
            # previous prefill cycle — clear them instead of consuming.
            if inputs_embeds is None and seq_len <= 2:
                self._cfg_companion_queue.clear()
            else:
                cached, remaining = self._cfg_companion_queue[0]
                remaining -= 1
                num_vae, num_vit, img_H, img_W = cached
                num_img2img = num_vae + 1 + num_vit  # +1 separator
                seq_len = inputs_embeds.shape[0] if inputs_embeds is not None else positions.shape[0]

                if inputs_embeds is not None and seq_len >= num_img2img:
                    self._pending_img2img_info = [cached]
                    positions = self._adjust_positions_for_img2img(positions, input_ids)
                    use_mot = True
                else:
                    rope = int(positions[seq_len - 1].item()) + 1
                    self._ropes_pending.append({"ropes": [rope]})

                if remaining == 0:
                    self._cfg_companion_queue.popleft()
                else:
                    self._cfg_companion_queue[0] = (cached, remaining)

        if use_mot:
            return self._mot_forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def _adjust_positions_for_img2img(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rewrite position IDs to match the original BAGEL position scheme:

        If there are ``pre_text_len`` text tokens before the img2img block::

            pre_text → 0, 1, ..., M-1
            VAE      → M       (all share)
            separator→ M
            ViT      → M+1     (all share)
            post_text→ M+2, M+3, ...

        When no text precedes the img2img block (M=0), this reduces to the
        simpler scheme: VAE→0, ViT→1, text→2, 3, ...

        Also computes ``self._vae_token_mask`` (bool tensor, True for actual
        VAE latent patches that should use gen-mode weights) and pushes
        per-request ropes + image_shape to the FIFO consumed by
        ``get_kv_transfer_metadata``.

        For img2img requests, also stores a decode position offset so that
        subsequent autoregressive decode steps use positions that continue
        from the rewritten scheme rather than from the original prefill length.
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
        for req_idx in range(num_requests):
            start = boundaries[req_idx]
            end = boundaries[req_idx + 1]
            req_len = end - start

            if img2img_idx < len(info_list):
                num_vae, num_vit, img_H, img_W = info_list[img2img_idx]
                num_img2img = num_vae + 1 + num_vit  # +1 separator

                if req_len >= num_img2img:
                    # Detect offset of img2img tokens within this request
                    # by searching for the img2img placeholder token ID.
                    pre_text_len = 0
                    if input_ids is not None:
                        req_ids = input_ids[start:end]
                        mask = req_ids == self._img2img_token_id
                        indices = mask.nonzero(as_tuple=True)[0]
                        if indices.numel() > 0:
                            pre_text_len = int(indices[0].item())

                    img_start = start + pre_text_len
                    post_text_start = img_start + num_img2img
                    # pre_text_pos: position base for image tokens
                    pre_text_pos = pre_text_len

                    # Pre-image text: sequential positions 0..pre_text_pos-1
                    if pre_text_len > 0:
                        new_positions[start:img_start] = torch.arange(
                            0, pre_text_pos, device=positions.device, dtype=positions.dtype
                        )

                    # VAE tokens: all share position pre_text_pos
                    new_positions[img_start : img_start + num_vae] = pre_text_pos
                    # Separator: position pre_text_pos
                    new_positions[img_start + num_vae] = pre_text_pos
                    # ViT tokens: all share position pre_text_pos+1
                    vit_start = img_start + num_vae + 1
                    new_positions[vit_start : vit_start + num_vit] = pre_text_pos + 1

                    # Post-image text: sequential positions pre_text_pos+2, pre_text_pos+3, ...
                    num_post_text = end - post_text_start
                    if num_post_text > 0:
                        new_positions[post_text_start:end] = torch.arange(
                            pre_text_pos + 2,
                            pre_text_pos + 2 + num_post_text,
                            device=positions.device,
                            dtype=positions.dtype,
                        )

                    # VAE gen-mode mask: only actual VAE latent patches (not markers)
                    vae_patches_start = img_start + 1  # skip start_marker
                    vae_patches_end = img_start + num_vae - 1  # before end_marker
                    if vae_patches_end > vae_patches_start:
                        vae_mask[vae_patches_start:vae_patches_end] = True

                    rope = pre_text_pos + 2 + num_post_text
                    self._ropes_pending.append(
                        {
                            "ropes": [rope],
                            "image_shape": [img_H, img_W],
                        }
                    )
                    decode_offset = rope - req_len
                    self._pending_decode_offsets.append(decode_offset)
                    img2img_idx += 1
                    continue

            rope = int(new_positions[end - 1].item()) + 1
            self._ropes_pending.append({"ropes": [rope]})
            self._pending_decode_offsets.append(0)

        self._vae_token_mask = vae_mask if vae_mask.any() else None
        return new_positions

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

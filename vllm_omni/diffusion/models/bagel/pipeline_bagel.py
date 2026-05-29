# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
BagelPipeline implementation for vLLM-Omni.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from math import isqrt
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.configs.bagel import BagelConfig

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .autoencoder import AutoEncoder, AutoEncoderParams
from .bagel_transformer import Bagel, NaiveCache, Qwen2MoTConfig, Qwen2MoTForCausalLM

logger = init_logger(__name__)

GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind and then "
    "generate the image.\n"
    "The planning process is enclosed within <think> </think> tags, i.e. "
    "<think> planning process here </think> image here"
)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class BagelGenParams:
    num_timesteps: int = 50
    timestep_shift: float = 3.0
    # Defaults aligned with ThinkMorphInterleaved / Bagel.py canonical values
    # (see VLMEvalKit_Thinkmorph/vlmeval/vlm/ThinkMorphInterleaved.py:171-176).
    # Previously cfg_text_scale=4.0 / cfg_renorm_min=0.0 — diverged from canonical.
    cfg_text_scale: float = 3.0
    cfg_img_scale: float = 1.5
    cfg_interval: tuple = (0.4, 1.0)
    cfg_renorm_min: float = 1.0
    cfg_renorm_type: str = "global"


# ---------------------------------------------------------------------------
# Canonical Bagel image preprocessing — port of bagel/src/models/bagel/transforms.py
# (MaxLongEdgeMinShortEdgeResize + ImageTransform). The transforms themselves
# are separate (VAE: 1024/512/16, ViT: 980/224/14), but the local
# force-interleaved path first VAE-resizes image inputs and then feeds that
# resized image into both VAE and ViT context updates.
# ---------------------------------------------------------------------------


class _MaxLongEdgeMinShortEdgeResize:
    def __init__(self, max_size: int, min_size: int, stride: int, max_pixels: int):
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels

    def _make_divisible(self, v: float) -> int:
        return max(self.stride, int(round(v / self.stride) * self.stride))

    def _apply_scale(self, w: int, h: int, scale: float) -> tuple[int, int]:
        return self._make_divisible(round(w * scale)), self._make_divisible(round(h * scale))

    def __call__(self, img: Image.Image, img_num: int = 1) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        scale = min(self.max_size / max(w, h), 1.0)
        scale = max(scale, self.min_size / min(w, h))
        new_w, new_h = self._apply_scale(w, h, scale)
        if new_w * new_h > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_w * new_h)
            new_w, new_h = self._apply_scale(new_w, new_h, scale)
        if max(new_w, new_h) > self.max_size:
            scale = self.max_size / max(new_w, new_h)
            new_w, new_h = self._apply_scale(new_w, new_h, scale)
        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h), Image.BICUBIC)
        return img


class _ImageTransform:
    """Resize PIL → [-1, 1] CHW tensor. Mirrors bagel/src/models/bagel/transforms.py:92."""

    def __init__(
        self,
        max_image_size: int,
        min_image_size: int,
        image_stride: int,
        max_pixels: int = 14 * 14 * 9 * 1024,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.resize = _MaxLongEdgeMinShortEdgeResize(max_image_size, min_image_size, image_stride, max_pixels)
        self.mean = torch.tensor(image_mean).view(3, 1, 1)
        self.std = torch.tensor(image_std).view(3, 1, 1)

    def resize_only(self, img: Image.Image, img_num: int = 1) -> Image.Image:
        return self.resize(img, img_num=img_num)

    def __call__(self, img: Image.Image, img_num: int = 1) -> torch.Tensor:
        img = self.resize(img, img_num=img_num)
        arr = torch.from_numpy(np.array(img)).float() / 255.0  # HWC, [0, 1]
        arr = arr.permute(2, 0, 1)  # CHW
        mean = self.mean.to(device=arr.device, dtype=arr.dtype)
        std = self.std.to(device=arr.device, dtype=arr.dtype)
        arr = (arr - mean) / std  # → [-1, 1] with mean=std=0.5
        return arr


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if "<|im_start|>" not in all_special_tokens:
        new_tokens.append("<|im_start|>")

    if "<|im_end|>" not in all_special_tokens:
        new_tokens.append("<|im_end|>")

    if "<|vision_start|>" not in all_special_tokens:
        new_tokens.append("<|vision_start|>")

    if "<|vision_end|>" not in all_special_tokens:
        new_tokens.append("<|vision_end|>")

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
    )

    return tokenizer, new_token_ids, num_new_tokens


def get_bagel_post_process_func(od_config: OmniDiffusionConfig):
    # BagelPipeline returns PIL.Image.Image directly.
    def post_process_func(x):
        return x

    return post_process_func


@dataclass
class _VaeCfg:
    z_channels: int = 16
    downsample: int = 8


@dataclass
class _VitCfg:
    patch_size: int = 14
    hidden_size: int = 1152


def default_ae_params() -> AutoEncoderParams:
    return AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )


class SiglipNaViTWrapper(nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        # If input is SiglipVisionModel, unwrap it to get SiglipVisionTransformer
        if hasattr(vision_model, "vision_model"):
            self.vision_model = vision_model.vision_model
        else:
            self.vision_model = vision_model

    def forward(self, packed_pixel_values, packed_flattened_position_ids, cu_seqlens, max_seqlen):
        patch_embed = self.vision_model.embeddings.patch_embedding
        w = patch_embed.weight.view(patch_embed.weight.shape[0], -1)
        x = F.linear(packed_pixel_values, w, patch_embed.bias)
        pos = self.vision_model.embeddings.position_embedding(packed_flattened_position_ids)
        x = x + pos
        hidden_states = x.unsqueeze(0)
        seq_len = x.shape[0]
        mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype)
        cu_seqlens_list = cu_seqlens.tolist()
        for i in range(len(cu_seqlens_list) - 1):
            start = cu_seqlens_list[i]
            end = cu_seqlens_list[i + 1]
            mask[..., start:end, start:end] = 0.0

        outputs = self.vision_model.encoder(inputs_embeds=hidden_states, attention_mask=mask)
        hidden_states = outputs.last_hidden_state.squeeze(0)
        return self.vision_model.post_layernorm(hidden_states)


class BagelPipeline(nn.Module, SupportsComponentDiscovery, DiffusionPipelineProfilerMixin):
    """Bagel generation pipeline (MoT) packaged for vllm-omni diffusion engine.

    This pipeline is self-contained and uses the ported Bagel core files.
    """

    SUPPORTS_REQUEST_BATCHING: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["language_model.model"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = [
        "bagel.time_embedder",
        "bagel.vae2llm",
        "bagel.llm2vae",
        "bagel.latent_pos_embed",
        "bagel.vit_model",
        "bagel.connector",
        "bagel.vit_pos_embed",
    ]
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "force_interleaved",
            "force_stage1_continue",
            "force_batch_text",
            "force_reencode_text",
            "max_think_token_n",
            "max_think_tokens",
            "max_answer_token_n",
            "max_answer_tokens",
            "do_sample",
            "answer_do_sample",
            "text_temperature",
            "text_repeat_stop_n",
            "cfg_text_scale",
            "cfg_img_scale",
            "cfg_interval",
            "cfg_renorm_min",
            "cfg_renorm_type",
            "timestep_shift",
            "num_timesteps",
        }
    )
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "force_interleaved",
            "think_text",
        }
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        self.scheduler: object | None = None
        self.scheduler_kwargs: dict = {}

        model = od_config.model
        local_files_only = os.path.exists(model)
        if local_files_only:
            model_path = model
        else:
            # Download everything required (ema.safetensors, ae.safetensors, tokenizer files, configs).
            model_path = download_weights_from_hf_specific(model, od_config.revision, ["*"])

        # Load Bagel top-level config for VAE settings.
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            bagel_cfg = json.load(f)

        vae_cfg_dict = bagel_cfg.get("vae_config") or {}
        vae_cfg = _VaeCfg(
            z_channels=int(vae_cfg_dict.get("z_channels", 16)),
            downsample=int(vae_cfg_dict.get("downsample", 8)),
        )

        # LLM config: Bagel MoT requires explicitly setting layer_module
        llm_cfg_path = os.path.join(model_path, "llm_config.json")
        llm_config = Qwen2MoTConfig.from_json_file(llm_cfg_path)
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        # Allow overriding from vllm-omni config if user wants MoE/vanilla.
        llm_config.layer_module = od_config.override_transformer_cls_name or "Qwen2MoTDecoderLayer"

        # Tokenizer and special tokens.
        # Bagel uses a Qwen2 tokenizer variant; prefer trust_remote_code to get the
        # correct tokenizer implementation from the checkpoint repo when available.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Try finding vision_config or interpolate from top-level config
        vit_cfg_dict = bagel_cfg.get("vit_config") or {}
        vit_cfg = _VitCfg(
            patch_size=int(vit_cfg_dict.get("patch_size", 14)),
            hidden_size=int(vit_cfg_dict.get("hidden_size", 1152)),
        )
        vit_config_path = os.path.join(model_path, "vit_config.json")
        vit_conf = SiglipVisionConfig.from_json_file(vit_config_path)
        if vit_conf.num_hidden_layers == 27:
            vit_conf.num_hidden_layers = 26
        vit_conf.vision_use_head = False
        self.vit_model = SiglipVisionModel(vit_conf)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_path, local_files_only=True)

        if self.vit_model:
            self.vit_model = SiglipNaViTWrapper(self.vit_model)
            vit_cfg.hidden_size = self.vit_model.vision_model.config.hidden_size
            vit_cfg.patch_size = self.vit_model.vision_model.config.patch_size

        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        # Register the ThinkMorph action token <gen_image>. Canonical
        # ThinkMorphInterleaved adapter does this at init time
        # (VLMEvalKit_Thinkmorph/vlmeval/vlm/ThinkMorphInterleaved.py:131-136).
        # Without it, the model never emits the token id that triggers
        # interleaved image generation, even if the checkpoint expects it.
        _action_token = "<gen_image>"
        if _action_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([_action_token])
        self.new_token_ids["image_action_token_id"] = self.tokenizer.convert_tokens_to_ids(_action_token)
        self.new_token_ids["register_action_token"] = True

        # Two canonical image transforms — VAE-side and ViT-side use different
        # max/min/stride ladders (1024/512/16 vs 980/224/14). See
        # bagel/src/models/bagel/transforms.py and
        # VLMEvalKit_Thinkmorph/vlmeval/vlm/Bagel.py:226-227.
        self._vae_transform = _ImageTransform(1024, 512, 16)
        self._vit_transform = _ImageTransform(980, 224, 14)

        tok_len = len(self.tokenizer)
        required_max_id = max(int(v) for v in self.new_token_ids.values())
        llm_config.vocab_size = max(
            int(getattr(llm_config, "vocab_size", tok_len)),
            int(tok_len),
            int(required_max_id + 1),
        )

        parallel_config = od_config.parallel_config if od_config else None
        quant_config = od_config.quantization_config
        # Bagel uses explicit prefixes ("bagel.language_model", "bagel") because
        # its model structure nests components under a top-level "bagel" module,
        # unlike other pipelines where the transformer is the root module.
        # This ensures ComponentQuantizationConfig prefix matching works correctly.
        self.language_model = Qwen2MoTForCausalLM(
            llm_config, parallel_config=parallel_config, quant_config=quant_config, prefix="bagel.language_model"
        )
        self.transformer = self.language_model.model
        ae_params: AutoEncoderParams = default_ae_params()
        self.vae = AutoEncoder(ae_params)

        self.bagel = Bagel(
            language_model=self.language_model,
            vit_model=self.vit_model,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix="bagel",
            config=BagelConfig(
                llm_config=llm_config,
                vae_config=vae_cfg,
                vit_config=vit_cfg,
                vit_max_num_patch_per_side=int(bagel_cfg.get("vit_max_num_patch_per_side", 70)),
                connector_act=str(bagel_cfg.get("connector_act", "gelu_pytorch_tanh")),
                interpolate_pos=bool(bagel_cfg.get("interpolate_pos", False)),
                latent_patch_size=int(bagel_cfg.get("latent_patch_size", 2)),
                # The public BAGEL config.json says 32, but the released
                # checkpoint and the canonical VLMEvalKit loader use a 64x64
                # latent positional table. Using 32 makes 1024px VAE inputs
                # index past latent_pos_embed and trips a CUDA gather assert.
                max_latent_size=int(os.environ.get(
                    "BAGEL_MAX_LATENT_SIZE",
                    max(64, int(bagel_cfg.get("max_latent_size", 64))),
                )),
                timestep_shift=float(bagel_cfg.get("timestep_shift", 1.0)),
            ),
        )

        # Let vLLM loader download and stream all *.safetensors under model root.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=False,
            )
        ]

        # Defer device placement to the weight-loading/offload path in three cases:
        # 1. Quantization: When quantization is enabled, vLLM linear layers live on meta
        #    device until the weight loader materializes them. Calling .to(device) would fail on those meta tensors,
        #    so we skip it entirely and let the weight loader handle device placement.
        # 2. Layerwise offload: modules should be initialized on CPU first, then
        #    selectively materialized/moved by the offloader.
        # 3. HSDP: weights should be loaded on CPU first and sharded afterwards,
        #    rather than eagerly placing the full model on one GPU.
        if quant_config is None and not (od_config.enable_layerwise_offload or od_config.parallel_config.use_hsdp):
            self.to(self.device)
        self._bagel_global_seed: int | None = None
        self._bagel_reseeded_after_warmup = False
        seed_raw = os.environ.get("BAGEL_GLOBAL_SEED", "").strip().lower()
        if seed_raw not in {"", "none", "null", "-1"}:
            self._bagel_global_seed = int(seed_raw)
            self._seed_global_rng(self._bagel_global_seed)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @staticmethod
    def _seed_global_rng(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _is_dummy_request(req: OmniDiffusionRequest) -> bool:
        request_ids = getattr(req, "request_ids", None) or ()
        return len(request_ids) == 1 and request_ids[0] == "dummy_req_id"

    def _maybe_reseed_after_warmup(self, reqs: OmniDiffusionRequest | list[OmniDiffusionRequest]) -> None:
        if self._bagel_global_seed is None or self._bagel_reseeded_after_warmup:
            return
        req_list = reqs if isinstance(reqs, list) else [reqs]
        if req_list and all(self._is_dummy_request(req) for req in req_list):
            return
        self._seed_global_rng(self._bagel_global_seed)
        self._bagel_reseeded_after_warmup = True
        logger.info("BAGEL global RNG reseeded after warmup with seed=%d", self._bagel_global_seed)

    @staticmethod
    def _decode_image_from_latent(
        bagel: Bagel, vae: AutoEncoder, latent: torch.Tensor, image_shape: tuple[int, int]
    ) -> Image.Image:
        H, W = image_shape
        h, w = H // bagel.latent_downsample, W // bagel.latent_downsample
        p = bagel.latent_patch_size
        c = bagel.latent_channel
        latent = latent.reshape(1, h, w, p, p, c)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, c, h * p, w * p)

        # Cast to VAE dtype (e.g. bfloat16) as latents might remain float32 from generation loop
        vae_dtype = next(vae.parameters()).dtype
        latent = latent.to(vae_dtype)

        image = vae.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        return Image.fromarray(image.to(torch.uint8).cpu().numpy())

    def _move_generation_input_to_device(self, generation_input: dict[str, Any]) -> dict[str, Any]:
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(self.device)
        return generation_input

    def _decode_generated_text(self, token_ids: torch.Tensor) -> str:
        decoded = self.tokenizer.decode(token_ids[:, 0].tolist())
        text = decoded.split("<|im_end|>")[0]
        if "<|im_start|>" in text:
            text = text.split("<|im_start|>")[-1]
        return text

    @staticmethod
    def _force_visible_answer(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if "<think>" in text and "</think>" in text:
            before_end, after_end = text.split("</think>", 1)
            inner = before_end.split("<think>", 1)[-1].strip()
            rest = after_end.strip()
            return rest or inner
        return text.replace("<think>", "").replace("</think>", "").strip()

    def _update_context_text(
        self,
        gen_context: dict[str, Any],
        text: str,
    ) -> dict[str, Any]:
        clean_text = text.removeprefix("<|im_start|>").removesuffix("<|im_end|>")
        generation_input, newlens, new_rope = self.bagel.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            prompts=[clean_text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        max_tid = int(generation_input["packed_text_ids"].max().item())
        emb_n = int(self.language_model.vocab_size)
        if max_tid >= emb_n:
            raise ValueError(
                "Tokenizer/model vocab mismatch: max token id "
                f"{max_tid} >= embed_tokens size {emb_n}. "
                "This usually means the tokenizer token IDs do not match the checkpoint embeddings."
            )
        generation_input = self._move_generation_input_to_device(generation_input)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_text(
                gen_context["past_key_values"], **generation_input
            )
        gen_context["kv_lens"] = newlens
        gen_context["ropes"] = new_rope
        return gen_context

    def _update_context_image(
        self,
        gen_context: dict[str, Any],
        image: Image.Image,
        *,
        vae: bool,
        vit: bool,
    ) -> tuple[dict[str, Any], tuple[int, int] | None]:
        if not (vae or vit):
            raise ValueError("At least one of vae/vit must be enabled.")

        image_shape: tuple[int, int] | None = None
        rgb_image = image.convert("RGB") if image.mode != "RGB" else image
        vae_image = rgb_image
        if vae:
            vae_image = self._vae_transform.resize_only(rgb_image)
            resized_w, resized_h = vae_image.size
            image_shape = (resized_h, resized_w)

            gen_input_vae, newlens_vae, new_rope_vae = self.bagel.prepare_vae_images(
                curr_kvlens=gen_context["kv_lens"],
                curr_rope=gen_context["ropes"],
                images=[vae_image],
                transforms=self._vae_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input_vae = self._move_generation_input_to_device(gen_input_vae)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                gen_context["past_key_values"] = self.bagel.forward_cache_update_vae(
                    self.vae, gen_context["past_key_values"], **gen_input_vae
                )
            gen_context["kv_lens"] = newlens_vae
            gen_context["ropes"] = new_rope_vae

        if vit:
            # Local force_interleave_inference VAE-resizes image inputs before
            # calling update_context_image(..., vae=True, vit=True). Preserve
            # that behavior when both branches are active; pure ViT updates
            # still use the original RGB image.
            vit_image = vae_image if vae else rgb_image
            gen_input_img, newlens_img, new_rope_img = self.bagel.prepare_vit_images(
                curr_kvlens=gen_context["kv_lens"],
                curr_rope=gen_context["ropes"],
                images=[vit_image],
                transforms=self._vit_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input_img = self._move_generation_input_to_device(gen_input_img)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                gen_context["past_key_values"] = self.bagel.forward_cache_update_vit(
                    gen_context["past_key_values"], **gen_input_img
                )
            gen_context["kv_lens"] = newlens_img
            gen_context["ropes"] = new_rope_img

        return gen_context, image_shape

    def _decode_generated_text_sequence(self, token_ids: torch.Tensor) -> str:
        if token_ids.numel() == 0:
            return ""
        if token_ids.ndim == 2:
            ids = token_ids[:, 0].tolist()
        else:
            ids = token_ids.tolist()
        decoded = self.tokenizer.decode(ids)
        text = decoded.split("<|im_end|>")[0]
        if "<|im_start|>" in text:
            text = text.split("<|im_start|>")[-1]
        return text

    def _empty_context(self, batch_size: int) -> dict[str, Any]:
        return {
            "kv_lens": [0] * batch_size,
            "ropes": [0] * batch_size,
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

    def _merge_contexts(self, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        merged_kv_lens: list[int] = []
        merged_ropes: list[int] = []
        caches = []
        for context in contexts:
            merged_kv_lens.extend(int(x) for x in context["kv_lens"])
            merged_ropes.extend(int(x) for x in context["ropes"])
            caches.append(context["past_key_values"])
        return {
            "kv_lens": merged_kv_lens,
            "ropes": merged_ropes,
            "past_key_values": self.bagel._merge_naive_caches(caches),
        }

    def _adopt_generated_text_batch_context(
        self,
        target_context: dict[str, Any],
        generated_context: dict[str, Any],
        original_kv_lens: list[int],
        original_ropes: list[int],
        token_sequences: list[torch.Tensor],
    ) -> None:
        """Keep exact batched text-generation KV, trimming finished-row padding.

        `batch_generate_text` must keep every row in the packed cache until all
        rows finish, so shorter rows receive extra post-finish tokens. Local
        single-sample force mode continues from the exact generated-token KV,
        without re-encoding decoded text or adding an EOS prompt token. Trim the
        packed cache back to each row's real generated length to preserve that
        contract for the batched Stage 1 path.
        """
        generated_lengths = [int(seq.shape[0]) if torch.is_tensor(seq) else len(seq) for seq in token_sequences]
        if not generated_lengths:
            target_context["past_key_values"] = generated_context["past_key_values"]
            target_context["kv_lens"] = list(original_kv_lens)
            target_context["ropes"] = list(original_ropes)
            return

        source_cache = generated_context["past_key_values"]
        total_cache_len = self._cache_seq_len(source_cache)
        original_total = sum(int(x) for x in original_kv_lens)
        batch_size = len(original_kv_lens)
        if batch_size <= 0:
            return

        # In the common case all rows finish at the same step. If the cache
        # shape is unavailable, falling back to max length is still consistent.
        global_steps = max(generated_lengths)
        if total_cache_len >= original_total:
            delta = total_cache_len - original_total
            if delta % batch_size == 0:
                global_steps = delta // batch_size

        full_lens = [int(kv_len) + global_steps for kv_len in original_kv_lens]
        keep_lens = [
            int(kv_len) + int(gen_len)
            for kv_len, gen_len in zip(original_kv_lens, generated_lengths)
        ]

        trimmed_cache = NaiveCache(source_cache.num_layers)
        for layer_idx in range(source_cache.num_layers):
            key_cache = source_cache.key_cache[layer_idx]
            value_cache = source_cache.value_cache[layer_idx]
            if key_cache is None or value_cache is None:
                continue
            key_chunks = []
            value_chunks = []
            cursor = 0
            for full_len, keep_len in zip(full_lens, keep_lens):
                key_chunks.append(key_cache[cursor : cursor + keep_len])
                value_chunks.append(value_cache[cursor : cursor + keep_len])
                cursor += full_len
            trimmed_cache.key_cache[layer_idx] = torch.cat(key_chunks, dim=0)
            trimmed_cache.value_cache[layer_idx] = torch.cat(value_chunks, dim=0)

        target_context["past_key_values"] = trimmed_cache
        target_context["kv_lens"] = keep_lens
        target_context["ropes"] = [
            int(rope) + int(gen_len)
            for rope, gen_len in zip(original_ropes, generated_lengths)
        ]

    @staticmethod
    def _cache_seq_len(past_key_values: NaiveCache) -> int:
        key_cache = past_key_values.key_cache
        values = key_cache.values() if hasattr(key_cache, "values") else key_cache
        for key_cache in values:
            if key_cache is not None:
                return int(key_cache.shape[0])
        return 0

    @staticmethod
    def _slice_cache_prefix(past_key_values: NaiveCache, keep_len: int) -> NaiveCache:
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        num_layers = (
            past_key_values.num_layers
            if hasattr(past_key_values, "num_layers")
            else len(key_cache)
        )
        sliced = NaiveCache(num_layers)
        for layer_idx in range(num_layers):
            k = key_cache[layer_idx]
            v = value_cache[layer_idx]
            if k is None or v is None:
                continue
            sliced.key_cache[layer_idx] = k[:keep_len].clone()
            sliced.value_cache[layer_idx] = v[:keep_len].clone()
        return sliced

    @staticmethod
    def _first_rope(value: Any, default: int) -> int:
        if value is None:
            return int(default)
        if isinstance(value, (list, tuple)):
            return int(value[0]) if value else int(default)
        return int(value)

    def _update_context_text_batch(
        self,
        gen_context: dict[str, Any],
        texts: list[str],
    ) -> dict[str, Any]:
        clean_texts = [
            (text or "").removeprefix("<|im_start|>").removesuffix("<|im_end|>")
            for text in texts
        ]
        generation_input, newlens, new_rope = self.bagel.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            prompts=clean_texts,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        if generation_input["packed_text_ids"].numel() > 0:
            max_tid = int(generation_input["packed_text_ids"].max().item())
            emb_n = int(self.language_model.vocab_size)
            if max_tid >= emb_n:
                raise ValueError(
                    "Tokenizer/model vocab mismatch: max token id "
                    f"{max_tid} >= embed_tokens size {emb_n}."
                )
        generation_input = self._move_generation_input_to_device(generation_input)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_text(
                gen_context["past_key_values"], **generation_input
            )
        gen_context["kv_lens"] = newlens
        gen_context["ropes"] = new_rope
        return gen_context

    def _update_context_image_batch(
        self,
        gen_context: dict[str, Any],
        images: list[Image.Image],
        *,
        vae: bool,
        vit: bool,
    ) -> tuple[dict[str, Any], list[tuple[int, int] | None]]:
        if not (vae or vit):
            raise ValueError("At least one of vae/vit must be enabled.")

        rgb_images = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in images]
        image_shapes: list[tuple[int, int] | None] = [None] * len(rgb_images)
        vae_images = rgb_images

        if vae:
            vae_images = [self._vae_transform.resize_only(img) for img in rgb_images]
            image_shapes = [(img.size[1], img.size[0]) for img in vae_images]

            gen_input_vae, newlens_vae, new_rope_vae = self.bagel.prepare_vae_images(
                curr_kvlens=gen_context["kv_lens"],
                curr_rope=gen_context["ropes"],
                images=vae_images,
                transforms=self._vae_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input_vae = self._move_generation_input_to_device(gen_input_vae)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                gen_context["past_key_values"] = self.bagel.forward_cache_update_vae(
                    self.vae, gen_context["past_key_values"], **gen_input_vae
                )
            gen_context["kv_lens"] = newlens_vae
            gen_context["ropes"] = new_rope_vae

        if vit:
            # Match local force-interleaved context updates: VAE+ViT receives
            # the VAE-resized image; ViT-only receives the original image.
            vit_images = vae_images if vae else rgb_images
            gen_input_img, newlens_img, new_rope_img = self.bagel.prepare_vit_images(
                curr_kvlens=gen_context["kv_lens"],
                curr_rope=gen_context["ropes"],
                images=vit_images,
                transforms=self._vit_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input_img = self._move_generation_input_to_device(gen_input_img)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                gen_context["past_key_values"] = self.bagel.forward_cache_update_vit(
                    gen_context["past_key_values"], **gen_input_img
                )
            gen_context["kv_lens"] = newlens_img
            gen_context["ropes"] = new_rope_img

        return gen_context, image_shapes

    def _generate_texts_from_context_batch(
        self,
        gen_context: dict[str, Any],
        *,
        max_tokens: int,
        do_sample: bool,
        text_temperature: float,
        sampling_generators: list[torch.Generator | None] | None = None,
        repeat_stop_n: int | None = None,
        adopt_context: bool = True,
    ) -> list[str]:
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            start_input = self.bagel.prepare_start_tokens(
                gen_context["kv_lens"], gen_context["ropes"], self.new_token_ids
            )
            start_input = self._move_generation_input_to_device(start_input)
            gen_ctx_copy = deepcopy(gen_context)
            original_kv_lens = list(gen_context["kv_lens"])
            original_ropes = list(gen_context["ropes"])
            token_sequences = self.bagel.batch_generate_text(
                past_key_values=gen_ctx_copy["past_key_values"],
                max_length=max_tokens,
                do_sample=do_sample,
                temperature=text_temperature,
                end_token_id=self.new_token_ids["eos_token_id"],
                sampling_generators=sampling_generators,
                repeat_stop_n=repeat_stop_n,
                **start_input,
            )
        texts = [self._decode_generated_text_sequence(seq) for seq in token_sequences]
        if adopt_context:
            self._adopt_generated_text_batch_context(
                gen_context,
                gen_ctx_copy,
                original_kv_lens,
                original_ropes,
                token_sequences,
            )
        return texts

    def _make_text_sampling_generators(
        self,
        reqs: list[OmniDiffusionRequest],
    ) -> list[torch.Generator | None] | None:
        if not any(req.sampling_params.seed is not None for req in reqs):
            return None
        generators: list[torch.Generator | None] = []
        for req in reqs:
            seed = req.sampling_params.seed
            if seed is None:
                generators.append(None)
                continue
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            generators.append(generator)
        return generators

    def _validate_image_generation_input(
        self,
        generation_input: dict[str, Any],
        image_shapes: list[tuple[int, int]],
    ) -> None:
        if generation_input["packed_text_ids"].numel() > 0:
            max_tid_img = int(generation_input["packed_text_ids"].max().item())
            emb_n = int(self.language_model.vocab_size)
            if max_tid_img >= emb_n:
                raise ValueError(
                    "Tokenizer/model vocab mismatch (image path): max token id "
                    f"{max_tid_img} >= embed_tokens size {emb_n}."
                )
        min_pid = int(generation_input["packed_position_ids"].min().item())
        if min_pid < 0:
            raise ValueError(f"Invalid packed_position_ids: min={min_pid} (must be >= 0)")
        max_lat_pid = int(generation_input["packed_vae_position_ids"].max().item())
        max_lat_pid_allowed = int(self.bagel.max_latent_size * self.bagel.max_latent_size) - 1
        if max_lat_pid > max_lat_pid_allowed:
            raise ValueError(
                "Invalid packed_vae_position_ids (latent position embedding OOB): "
                f"max={max_lat_pid} > allowed_max={max_lat_pid_allowed}. "
                f"Requested image_shapes={image_shapes}, max_latent_size={self.bagel.max_latent_size}."
            )

    def _generate_images_from_context_batch(
        self,
        reqs: list[OmniDiffusionRequest],
        gen_context: dict[str, Any],
        cfg_text_context: dict[str, Any],
        cfg_img_context: dict[str, Any],
        image_shapes: list[tuple[int, int]],
        gen_params: BagelGenParams,
    ) -> list[Image.Image]:
        generation_input = self.bagel.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            image_sizes=image_shapes,
            new_token_ids=self.new_token_ids,
            noise_seeds=[req.sampling_params.seed for req in reqs],
        )
        self._validate_image_generation_input(generation_input, image_shapes)
        generation_input = self._move_generation_input_to_device(generation_input)

        generation_input_cfg_text = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            image_sizes=image_shapes,
        )
        generation_input_cfg_img = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_context["kv_lens"],
            curr_rope=cfg_img_context["ropes"],
            image_sizes=image_shapes,
        )
        generation_input_cfg_text = self._move_generation_input_to_device(generation_input_cfg_text)
        generation_input_cfg_img = self._move_generation_input_to_device(generation_input_cfg_img)

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            latents, _, _, _ = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_img_past_key_values=cfg_img_context["past_key_values"],
                num_timesteps=gen_params.num_timesteps,
                timestep_shift=gen_params.timestep_shift,
                cfg_text_scale=gen_params.cfg_text_scale,
                cfg_img_scale=gen_params.cfg_img_scale,
                cfg_interval=gen_params.cfg_interval,
                cfg_renorm_min=gen_params.cfg_renorm_min,
                cfg_renorm_type=gen_params.cfg_renorm_type,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
                cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
                return_trajectory_latents=False,
                scheduler=self.scheduler,
                scheduler_kwargs=self.scheduler_kwargs,
            )
        return [
            self._decode_image_from_latent(self.bagel, self.vae, latent, image_shape)
            for latent, image_shape in zip(latents, image_shapes)
        ]

    def _generate_text_from_context(
        self,
        gen_context: dict[str, Any],
        *,
        max_tokens: int,
        do_sample: bool,
        text_temperature: float,
        adopt_context: bool = True,
        repeat_stop_n: int | None = None,
    ) -> str:
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            start_input = self.bagel.prepare_start_tokens(
                gen_context["kv_lens"], gen_context["ropes"], self.new_token_ids
            )
            start_input = self._move_generation_input_to_device(start_input)
            gen_ctx_copy = deepcopy(gen_context)
            token_ids = self.bagel.generate_text(
                past_key_values=gen_ctx_copy["past_key_values"],
                max_length=max_tokens,
                do_sample=do_sample,
                temperature=text_temperature,
                end_token_id=self.new_token_ids["eos_token_id"],
                repeat_stop_n=repeat_stop_n,
                **start_input,
            )
        if adopt_context:
            gen_context["past_key_values"] = gen_ctx_copy["past_key_values"]
            gen_context["kv_lens"] = [
                kv_len + token_ids.shape[0] for kv_len in gen_context["kv_lens"]
            ]
            gen_context["ropes"] = [
                rope + token_ids.shape[0] for rope in gen_context["ropes"]
            ]
        return self._decode_generated_text(token_ids)

    def _generate_image_from_context(
        self,
        req: OmniDiffusionRequest,
        gen_context: dict[str, Any],
        cfg_text_context: dict[str, Any],
        cfg_img_context: dict[str, Any],
        image_shape: tuple[int, int],
        gen_params: BagelGenParams,
    ) -> Image.Image:
        generation_input = self.bagel.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        max_tid_img = int(generation_input["packed_text_ids"].max().item())
        emb_n = int(self.language_model.vocab_size)
        if max_tid_img >= emb_n:
            raise ValueError(
                "Tokenizer/model vocab mismatch (image path): max token id "
                f"{max_tid_img} >= embed_tokens size {emb_n}."
            )
        min_pid = int(generation_input["packed_position_ids"].min().item())
        if min_pid < 0:
            raise ValueError(f"Invalid packed_position_ids: min={min_pid} (must be >= 0)")
        max_lat_pid = int(generation_input["packed_vae_position_ids"].max().item())
        max_lat_pid_allowed = int(self.bagel.max_latent_size * self.bagel.max_latent_size) - 1
        if max_lat_pid > max_lat_pid_allowed:
            raise ValueError(
                "Invalid packed_vae_position_ids (latent position embedding OOB): "
                f"max={max_lat_pid} > allowed_max={max_lat_pid_allowed}. "
                f"Requested image_shape={image_shape}, max_latent_size={self.bagel.max_latent_size}."
            )
        generation_input = self._move_generation_input_to_device(generation_input)

        generation_input_cfg_text = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            image_sizes=[image_shape],
        )
        generation_input_cfg_img = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_context["kv_lens"],
            curr_rope=cfg_img_context["ropes"],
            image_sizes=[image_shape],
        )
        generation_input_cfg_text = self._move_generation_input_to_device(generation_input_cfg_text)
        generation_input_cfg_img = self._move_generation_input_to_device(generation_input_cfg_img)

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            latents, _, _, _ = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_img_past_key_values=cfg_img_context["past_key_values"],
                num_timesteps=gen_params.num_timesteps,
                timestep_shift=gen_params.timestep_shift,
                cfg_text_scale=gen_params.cfg_text_scale,
                cfg_img_scale=gen_params.cfg_img_scale,
                cfg_interval=gen_params.cfg_interval,
                cfg_renorm_min=gen_params.cfg_renorm_min,
                cfg_renorm_type=gen_params.cfg_renorm_type,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
                cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
                return_trajectory_latents=req.sampling_params.return_trajectory_latents,
                scheduler=self.scheduler,
                scheduler_kwargs=self.scheduler_kwargs,
            )
        return self._decode_image_from_latent(self.bagel, self.vae, latents[0], image_shape)

    def _force_generation_settings(
        self,
        req: OmniDiffusionRequest,
        extra_args: dict[str, Any],
    ) -> tuple[BagelGenParams, int, int, bool, bool, float, int | None]:
        force_params = BagelGenParams(
            num_timesteps=int(
                extra_args.get("num_timesteps")
                or req.sampling_params.num_inference_steps
                or 50
            ),
            timestep_shift=float(extra_args.get("timestep_shift", 3.0)),
            # Keep force-interleaved defaults aligned with
            # VLMEvalKit's active bagel_local_force_interleaved config.
            cfg_text_scale=float(extra_args.get("cfg_text_scale", 4.0)),
            cfg_img_scale=float(extra_args.get("cfg_img_scale", 2.0)),
            cfg_interval=tuple(extra_args.get("cfg_interval", (0.0, 1.0))),
            cfg_renorm_min=float(extra_args.get("cfg_renorm_min", 0.0)),
            cfg_renorm_type=str(extra_args.get("cfg_renorm_type", "text_channel")),
        )
        max_think_tokens = int(
            extra_args.get("max_think_token_n")
            or extra_args.get("max_think_tokens")
            or 4096
        )
        max_answer_tokens = int(
            extra_args.get("max_answer_token_n")
            or extra_args.get("max_answer_tokens")
            or 4096
        )
        do_sample = _truthy(extra_args.get("do_sample", True))
        answer_do_sample_raw = extra_args.get("answer_do_sample")
        answer_do_sample = do_sample if answer_do_sample_raw is None else _truthy(answer_do_sample_raw)
        text_temperature = float(extra_args.get("text_temperature", 0.3))
        repeat_stop_n_raw = extra_args.get("text_repeat_stop_n", 64)
        repeat_stop_n = int(repeat_stop_n_raw) if repeat_stop_n_raw is not None else None
        return (
            force_params,
            max_think_tokens,
            max_answer_tokens,
            do_sample,
            answer_do_sample,
            text_temperature,
            repeat_stop_n,
        )

    def _force_stage1_continue_forward(
        self,
        req: OmniDiffusionRequest,
        gen_context: dict[str, Any],
        cfg_text_context: dict[str, Any],
        cfg_img_context: dict[str, Any],
        image_shape: tuple[int, int],
        extra_args: dict[str, Any],
    ) -> DiffusionOutput:
        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        (
            force_params,
            max_think_tokens,
            max_answer_tokens,
            do_sample,
            answer_do_sample,
            text_temperature,
            repeat_stop_n,
        ) = self._force_generation_settings(req, extra_args)

        reencode_text = _truthy(extra_args.get("force_reencode_text", False))
        think_text = self._generate_text_from_context(
            gen_context,
            max_tokens=max_think_tokens,
            do_sample=do_sample,
            text_temperature=text_temperature,
            adopt_context=not reencode_text,
            repeat_stop_n=repeat_stop_n,
        )
        if reencode_text:
            # Mirrors VLMEvalKit's batched force path: decoded think text is
            # fed back through prepare_prompts before image generation.
            gen_context = self._update_context_text(gen_context, think_text)

        img = self._generate_image_from_context(
            req,
            gen_context,
            cfg_text_context,
            cfg_img_context,
            image_shape,
            force_params,
        )

        gen_context, _ = self._update_context_image(
            gen_context,
            img,
            vae=True,
            vit=True,
        )

        answer_text = self._generate_text_from_context(
            gen_context,
            max_tokens=max_answer_tokens,
            do_sample=answer_do_sample,
            text_temperature=text_temperature,
            repeat_stop_n=repeat_stop_n,
        )
        answer_visible = self._force_visible_answer(answer_text)
        trace_text = "\n".join(
            part for part in (think_text, "[Generated Image]", answer_visible) if part
        )

        return DiffusionOutput(
            output=img,
            custom_output={
                "text_output": trace_text,
                "answer_text": answer_visible,
                "raw_answer_text": answer_text,
                "think_text": think_text,
                "force_interleaved": True,
                "stage1_continue": True,
                "stage1_reencode_text": reencode_text,
                "answer_do_sample": answer_do_sample,
            },
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def _resolve_request_image_shape(self, req: OmniDiffusionRequest) -> tuple[int, int]:
        max_hw = int(self.bagel.max_latent_size * self.bagel.latent_downsample)
        if req.sampling_params.height is None and req.sampling_params.width is None:
            image_shape = (max_hw, max_hw)
        else:
            height = int(req.sampling_params.height) if req.sampling_params.height is not None else max_hw
            width = int(req.sampling_params.width) if req.sampling_params.width is not None else max_hw
            image_shape = (height, width)
        if req.sampling_params.kv_metadata and "image_shape" in req.sampling_params.kv_metadata:
            image_shape = tuple(req.sampling_params.kv_metadata["image_shape"])
        height, width = image_shape
        if height > max_hw or width > max_hw:
            raise ValueError(
                f"Requested resolution {height}x{width} exceeds Bagel checkpoint limit "
                f"{max_hw}x{max_hw} (max_latent_size={self.bagel.max_latent_size}, "
                f"latent_downsample={self.bagel.latent_downsample})."
            )
        return int(height), int(width)

    def _force_stage1_contexts_from_request(
        self,
        req: OmniDiffusionRequest,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[int, int]]:
        injected_kv = req.sampling_params.past_key_values
        if injected_kv is None:
            raise ValueError("force_stage1_continue requires injected KV cache from Stage 0.")

        image_shape = self._resolve_request_image_shape(req)
        seq_len = self._cache_seq_len(injected_kv)
        ropes = (
            req.sampling_params.kv_metadata["ropes"]
            if req.sampling_params.kv_metadata and "ropes" in req.sampling_params.kv_metadata
            else seq_len
        )
        gen_context = {
            "kv_lens": [seq_len],
            "ropes": [self._first_rope(ropes, seq_len)],
            "past_key_values": injected_kv,
        }

        cfg_text_context = self._empty_context(1)
        cfg_img_context = self._empty_context(1)

        branch_kvs = getattr(req.sampling_params, "cfg_branch_past_key_values", None) or {}
        branch_metadata = getattr(req.sampling_params, "cfg_branch_kv_metadata", None) or {}
        cfg_text_kv = getattr(req.sampling_params, "cfg_text_past_key_values", None) or branch_kvs.get("cfg_text")
        cfg_text_metadata = getattr(req.sampling_params, "cfg_text_kv_metadata", None) or branch_metadata.get(
            "cfg_text"
        )
        cfg_img_kv = getattr(req.sampling_params, "cfg_img_past_key_values", None) or branch_kvs.get("cfg_img")
        cfg_img_metadata = getattr(req.sampling_params, "cfg_img_kv_metadata", None) or branch_metadata.get("cfg_img")

        cfg_text_prefix_len = None
        cfg_text_prefix_rope = None
        if req.sampling_params.kv_metadata:
            cfg_text_prefix_len = req.sampling_params.kv_metadata.get("cfg_text_kv_len")
            cfg_text_prefix_rope = req.sampling_params.kv_metadata.get("cfg_text_rope")

        if cfg_text_prefix_len:
            cfg_text_seq_len = int(cfg_text_prefix_len)
            cfg_text_context = {
                "kv_lens": [cfg_text_seq_len],
                "ropes": [self._first_rope(cfg_text_prefix_rope, cfg_text_seq_len)],
                "past_key_values": self._slice_cache_prefix(injected_kv, cfg_text_seq_len),
            }
        elif cfg_text_kv is not None:
            cfg_text_seq_len = self._cache_seq_len(cfg_text_kv)
            cfg_text_ropes = (
                cfg_text_metadata["ropes"] if cfg_text_metadata and "ropes" in cfg_text_metadata else cfg_text_seq_len
            )
            cfg_text_context = {
                "kv_lens": [cfg_text_seq_len],
                "ropes": [self._first_rope(cfg_text_ropes, cfg_text_seq_len)],
                "past_key_values": cfg_text_kv,
            }

        if cfg_img_kv is None:
            cfg_img_context = {
                "kv_lens": [seq_len],
                "ropes": [self._first_rope(ropes, seq_len)],
                "past_key_values": injected_kv,
            }
        else:
            cfg_img_seq_len = self._cache_seq_len(cfg_img_kv)
            cfg_img_ropes = (
                cfg_img_metadata["ropes"] if cfg_img_metadata and "ropes" in cfg_img_metadata else cfg_img_seq_len
            )
            cfg_img_context = {
                "kv_lens": [cfg_img_seq_len],
                "ropes": [self._first_rope(cfg_img_ropes, cfg_img_seq_len)],
                "past_key_values": cfg_img_kv,
            }

        return gen_context, cfg_text_context, cfg_img_context, image_shape

    def _can_batch_force_stage1(self, reqs: list[OmniDiffusionRequest]) -> bool:
        if not reqs:
            return False
        first_extra = getattr(reqs[0].sampling_params, "extra_args", {}) or {}
        if not _truthy(first_extra.get("force_stage1_continue")):
            return False
        try:
            first_settings = self._force_generation_settings(reqs[0], first_extra)
        except Exception:
            return False
        for req in reqs:
            extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
            if not _truthy(extra_args.get("force_stage1_continue")):
                return False
            if _truthy(extra_args.get("force_batch_text", False)) != _truthy(
                first_extra.get("force_batch_text", False)
            ):
                return False
            if req.sampling_params.past_key_values is None:
                return False
            if req.sampling_params.return_trajectory_latents or req.sampling_params.return_trajectory_decoded:
                return False
            try:
                if self._force_generation_settings(req, extra_args) != first_settings:
                    return False
            except Exception:
                return False
        return True

    def _force_stage1_continue_forward_batch(
        self,
        reqs: list[OmniDiffusionRequest],
    ) -> list[DiffusionOutput]:
        if not self._can_batch_force_stage1(reqs):
            return [self.forward(req) for req in reqs]

        first_extra = getattr(reqs[0].sampling_params, "extra_args", {}) or {}
        (
            force_params,
            max_think_tokens,
            max_answer_tokens,
            do_sample,
            answer_do_sample,
            text_temperature,
            repeat_stop_n,
        ) = self._force_generation_settings(reqs[0], first_extra)

        contexts = [self._force_stage1_contexts_from_request(req) for req in reqs]
        image_shapes = [ctx[3] for ctx in contexts]

        batch_text = _truthy(first_extra.get("force_batch_text", False))
        can_batch_text = batch_text and len(reqs) > 1
        if can_batch_text:
            logger.info(
                "Running BAGEL force_stage1_continue batch size=%d "
                "(batched text, batched image)",
                len(reqs),
            )
            sampling_generators = self._make_text_sampling_generators(reqs)

            phase_t0 = time.perf_counter()
            gen_context = self._merge_contexts([ctx[0] for ctx in contexts])
            cfg_text_context = self._merge_contexts([ctx[1] for ctx in contexts])
            cfg_img_context = self._merge_contexts([ctx[2] for ctx in contexts])
            reencode_text = _truthy(first_extra.get("force_reencode_text", False))

            think_texts = self._generate_texts_from_context_batch(
                gen_context,
                max_tokens=max_think_tokens,
                do_sample=do_sample,
                text_temperature=text_temperature,
                sampling_generators=sampling_generators,
                repeat_stop_n=repeat_stop_n,
                adopt_context=not reencode_text,
            )
            if reencode_text:
                # VLMEvalKit's batched force path decodes think text, then
                # re-encodes it into the context before image generation.
                gen_context = self._update_context_text_batch(gen_context, think_texts)
            logger.info(
                "BAGEL force_stage1_continue batched think done in %.2fs",
                time.perf_counter() - phase_t0,
            )
            phase_t0 = time.perf_counter()
            images = self._generate_images_from_context_batch(
                reqs,
                gen_context,
                cfg_text_context,
                cfg_img_context,
                image_shapes,
                force_params,
            )
            logger.info(
                "BAGEL force_stage1_continue batched image done in %.2fs",
                time.perf_counter() - phase_t0,
            )
            phase_t0 = time.perf_counter()
            gen_context, _ = self._update_context_image_batch(
                gen_context,
                images,
                vae=True,
                vit=True,
            )
            answer_texts = self._generate_texts_from_context_batch(
                gen_context,
                max_tokens=max_answer_tokens,
                do_sample=answer_do_sample,
                text_temperature=text_temperature,
                sampling_generators=sampling_generators,
                repeat_stop_n=repeat_stop_n,
            )
            logger.info(
                "BAGEL force_stage1_continue batched answer done in %.2fs",
                time.perf_counter() - phase_t0,
            )

            outputs: list[DiffusionOutput] = []
            for think_text, img, answer_text in zip(think_texts, images, answer_texts):
                answer_visible = self._force_visible_answer(answer_text)
                trace_text = "\n".join(
                    part for part in (think_text, "[Generated Image]", answer_visible) if part
                )
                outputs.append(
                    DiffusionOutput(
                        output=img,
                        custom_output={
                            "text_output": trace_text,
                            "answer_text": answer_visible,
                            "raw_answer_text": answer_text,
                            "think_text": think_text,
                            "force_interleaved": True,
                            "stage1_continue": True,
                            "stage1_batch_text": True,
                            "stage1_reencode_text": reencode_text,
                            "answer_do_sample": answer_do_sample,
                        },
                        stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
                    )
            )
            return outputs

        logger.info(
            "Running BAGEL force_stage1_continue batch size=%d "
            "(per-request text, batched image)",
            len(reqs),
        )

        total_t0 = time.perf_counter()
        think_texts: list[str] = []
        think_durations: list[float] = []
        text_rng_states: list[tuple[torch.Tensor, torch.Tensor | None] | None] = []
        reencode_text = _truthy(first_extra.get("force_reencode_text", False))
        for req, (gen_context_i, _, _, _) in zip(reqs, contexts):
            phase_t0 = time.perf_counter()
            seed = req.sampling_params.seed
            if seed is not None:
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(seed)
            think_texts.append(
                self._generate_text_from_context(
                    gen_context_i,
                    max_tokens=max_think_tokens,
                    do_sample=do_sample,
                    text_temperature=text_temperature,
                    adopt_context=not reencode_text,
                    repeat_stop_n=repeat_stop_n,
                )
            )
            think_durations.append(time.perf_counter() - phase_t0)
            if seed is None:
                text_rng_states.append(None)
            else:
                text_rng_states.append(
                    (
                        torch.random.get_rng_state(),
                        torch.cuda.get_rng_state(self.device)
                        if self.device.type == "cuda"
                        else None,
                    )
                )

        think_total = sum(think_durations)
        logger.info(
            "BAGEL force_stage1_continue per-request think done in %.2fs "
            "(sum %.2fs, avg %.2fs, max %.2fs)",
            time.perf_counter() - total_t0,
            think_total,
            think_total / max(len(think_durations), 1),
            max(think_durations, default=0.0),
        )

        if reencode_text:
            for think_text, (gen_context_i, _, _, _) in zip(think_texts, contexts):
                self._update_context_text(gen_context_i, think_text)

        gen_context = self._merge_contexts([ctx[0] for ctx in contexts])
        cfg_text_context = self._merge_contexts([ctx[1] for ctx in contexts])
        cfg_img_context = self._merge_contexts([ctx[2] for ctx in contexts])

        image_t0 = time.perf_counter()
        images = self._generate_images_from_context_batch(
            reqs,
            gen_context,
            cfg_text_context,
            cfg_img_context,
            image_shapes,
            force_params,
        )
        image_duration = time.perf_counter() - image_t0
        logger.info(
            "BAGEL force_stage1_continue batched image done in %.2fs",
            image_duration,
        )

        answer_texts: list[str] = []
        answer_t0 = time.perf_counter()
        answer_durations: list[float] = []
        for (gen_context_i, _, _, _), img, rng_state in zip(contexts, images, text_rng_states):
            phase_t0 = time.perf_counter()
            gen_context_i, _ = self._update_context_image(
                gen_context_i,
                img,
                vae=True,
                vit=True,
            )
            if rng_state is not None:
                cpu_state, cuda_state = rng_state
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, self.device)
            answer_texts.append(
                self._generate_text_from_context(
                    gen_context_i,
                    max_tokens=max_answer_tokens,
                    do_sample=answer_do_sample,
                    text_temperature=text_temperature,
                    repeat_stop_n=repeat_stop_n,
                )
            )
            answer_durations.append(time.perf_counter() - phase_t0)

        answer_total = sum(answer_durations)
        logger.info(
            "BAGEL force_stage1_continue per-request answer done in %.2fs "
            "(sum %.2fs, avg %.2fs, max %.2fs)",
            time.perf_counter() - answer_t0,
            answer_total,
            answer_total / max(len(answer_durations), 1),
            max(answer_durations, default=0.0),
        )
        logger.info(
            "BAGEL force_stage1_continue batch total %.2fs "
            "(think sum %.2fs, image %.2fs, answer sum %.2fs)",
            time.perf_counter() - total_t0,
            think_total,
            image_duration,
            answer_total,
        )

        outputs: list[DiffusionOutput] = []
        for think_text, img, answer_text in zip(think_texts, images, answer_texts):
            answer_visible = self._force_visible_answer(answer_text)
            trace_text = "\n".join(
                part for part in (think_text, "[Generated Image]", answer_visible) if part
            )
            outputs.append(
                DiffusionOutput(
                    output=img,
                    custom_output={
                        "text_output": trace_text,
                        "answer_text": answer_visible,
                        "raw_answer_text": answer_text,
                        "think_text": think_text,
                        "force_interleaved": True,
                        "stage1_continue": True,
                        "stage1_reencode_text": reencode_text,
                        "answer_do_sample": answer_do_sample,
                    },
                    stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
                )
            )
        return outputs

    def _force_interleave_forward(
        self,
        req: OmniDiffusionRequest,
        first_prompt: Any,
        prompt: str,
        image_shape: tuple[int, int],
        extra_args: dict[str, Any],
    ) -> DiffusionOutput:
        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        (
            force_params,
            max_think_tokens,
            max_answer_tokens,
            do_sample,
            answer_do_sample,
            text_temperature,
            repeat_stop_n,
        ) = self._force_generation_settings(req, extra_args)

        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        gen_context = self._update_context_text(gen_context, GEN_THINK_SYSTEM_PROMPT)
        cfg_img_context = self._update_context_text(cfg_img_context, GEN_THINK_SYSTEM_PROMPT)

        image_input = None
        if not isinstance(first_prompt, str):
            image_input = (
                (first_prompt.get("multi_modal_data") or {}).get("image")
                or (first_prompt.get("multi_modal_data") or {}).get("img2img")
            )
        if image_input and not isinstance(image_input, list):
            image_input = [image_input]
        if image_input:
            image_input = [
                Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
                for image in image_input
            ]
            input_vit = _truthy(
                extra_args.get(
                    "force_input_vit",
                    os.environ.get("BAGEL_FORCE_IMG2IMG_VIT", "1"),
                )
            )
            for image in image_input:
                gen_context, maybe_shape = self._update_context_image(
                    gen_context,
                    image,
                    vae=True,
                    vit=input_vit,
                )
                if maybe_shape is not None:
                    image_shape = maybe_shape
            cfg_text_context = deepcopy(gen_context)

        if prompt:
            cfg_text_context = deepcopy(gen_context)
            gen_context = self._update_context_text(gen_context, prompt)
            cfg_img_context = self._update_context_text(cfg_img_context, prompt)

        think_text = self._generate_text_from_context(
            gen_context,
            max_tokens=max_think_tokens,
            do_sample=do_sample,
            text_temperature=text_temperature,
            repeat_stop_n=repeat_stop_n,
        )

        img = self._generate_image_from_context(
            req,
            gen_context,
            cfg_text_context,
            cfg_img_context,
            image_shape,
            force_params,
        )

        gen_context, _ = self._update_context_image(
            gen_context,
            img,
            vae=True,
            vit=True,
        )

        answer_text = self._generate_text_from_context(
            gen_context,
            max_tokens=max_answer_tokens,
            do_sample=answer_do_sample,
            text_temperature=text_temperature,
            repeat_stop_n=repeat_stop_n,
        )
        answer_visible = self._force_visible_answer(answer_text)
        trace_text = "\n".join(
            part for part in (think_text, "[Generated Image]", answer_visible) if part
        )

        return DiffusionOutput(
            output=trace_text,
            custom_output={
                "text_output": trace_text,
                "answer_text": answer_visible,
                "raw_answer_text": answer_text,
                "think_text": think_text,
                "force_interleaved": True,
                "answer_do_sample": answer_do_sample,
            },
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @torch.inference_mode()
    def forward_batch(self, reqs: list[OmniDiffusionRequest]) -> list[DiffusionOutput]:
        if not reqs:
            return []
        self._maybe_reseed_after_warmup(reqs)
        if self._can_batch_force_stage1(reqs):
            return self._force_stage1_continue_forward_batch(reqs)
        return [self.forward(req) for req in reqs]

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        self._maybe_reseed_after_warmup(req)
        if len(req.prompts) > 1:
            logger.warning(
                """This model only supports a single prompt, not a batched request.""",
                """Taking only the first image for now.""",
            )
        # TODO: In online mode, sometimes it receives [{"prompts": None}, {...}], so cannot use .get("...", "")
        # TODO: May be some data formatting operations on the API side. Hack for now.
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(req.prompts[0], str) else (req.prompts[0].get("prompt") or "")

        max_hw = int(self.bagel.max_latent_size * self.bagel.latent_downsample)
        if req.sampling_params.height is None and req.sampling_params.width is None:
            height = width = max_hw
        else:
            height = int(req.sampling_params.height) if req.sampling_params.height is not None else max_hw
            width = int(req.sampling_params.width) if req.sampling_params.width is not None else max_hw
        if height > max_hw or width > max_hw:
            raise ValueError(
                f"Requested resolution {height}x{width} exceeds Bagel checkpoint limit "
                f"{max_hw}x{max_hw} (max_latent_size={self.bagel.max_latent_size}, "
                f"latent_downsample={self.bagel.latent_downsample})."
            )
        image_shape = (height, width)

        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        if _truthy(extra_args.get("force_interleaved")):
            if req.sampling_params.past_key_values is not None:
                raise ValueError("force_interleaved does not support injected KV cache.")
            return self._force_interleave_forward(
                req,
                first_prompt,
                prompt,
                image_shape,
                extra_args,
            )

        # Canonical ThinkMorphInterleaved/Bagel.py defaults: cfg_text_scale=3.0,
        # cfg_renorm_min=1.0. Prior 4.0/0.0 defaults caused image-quality drift
        # vs the HF baseline.
        cfg_text_scale = extra_args.get("cfg_text_scale", 3.0)
        cfg_img_scale = extra_args.get("cfg_img_scale", 1.5)

        cfg_interval = extra_args.get("cfg_interval", (0.4, 1.0))
        cfg_renorm_type = extra_args.get("cfg_renorm_type", "global")
        cfg_renorm_min = extra_args.get("cfg_renorm_min", 1.0)

        gen_params = BagelGenParams(
            num_timesteps=int(req.sampling_params.num_inference_steps or 50),
            timestep_shift=3.0,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_type=cfg_renorm_type,
            cfg_renorm_min=cfg_renorm_min,
        )

        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        injected_kv = req.sampling_params.past_key_values
        if injected_kv is not None:
            logger.info("Using injected KV Cache (direct)")
            gen_context["past_key_values"] = injected_kv
            seq_len = injected_kv.key_cache[0].shape[0]
            gen_context["kv_lens"] = [seq_len]
            if req.sampling_params.kv_metadata and "ropes" in req.sampling_params.kv_metadata:
                gen_context["ropes"] = req.sampling_params.kv_metadata["ropes"]
            else:
                gen_context["ropes"] = [seq_len]

            if req.sampling_params.kv_metadata and "image_shape" in req.sampling_params.kv_metadata:
                image_shape = tuple(req.sampling_params.kv_metadata["image_shape"])

            branch_kvs = getattr(req.sampling_params, "cfg_branch_past_key_values", None) or {}
            branch_metadata = getattr(req.sampling_params, "cfg_branch_kv_metadata", None) or {}
            active_branch = getattr(req.sampling_params, "cfg_active_branch", None)
            branch_roles = getattr(req.sampling_params, "cfg_branch_roles", None) or list(branch_kvs.keys())

            cfg_text_kv = getattr(req.sampling_params, "cfg_text_past_key_values", None) or branch_kvs.get("cfg_text")
            cfg_text_metadata = getattr(req.sampling_params, "cfg_text_kv_metadata", None) or branch_metadata.get(
                "cfg_text"
            )
            cfg_img_kv = getattr(req.sampling_params, "cfg_img_past_key_values", None) or branch_kvs.get("cfg_img")
            cfg_img_metadata = getattr(req.sampling_params, "cfg_img_kv_metadata", None) or branch_metadata.get(
                "cfg_img"
            )

            cfg_parallel_contract = (
                active_branch is not None or bool(branch_roles) or cfg_text_kv is not None or cfg_img_kv is not None
            )
            if cfg_parallel_contract:
                logger.info(
                    "CFG enabled with injected branch KV context roles=%s active=%s",
                    branch_roles,
                    active_branch,
                )

            cfg_text_prefix_len = None
            cfg_text_prefix_rope = None
            if req.sampling_params.kv_metadata:
                cfg_text_prefix_len = req.sampling_params.kv_metadata.get("cfg_text_kv_len")
                cfg_text_prefix_rope = req.sampling_params.kv_metadata.get("cfg_text_rope")

            if cfg_text_prefix_len:
                cfg_text_seq_len = int(cfg_text_prefix_len)
                cfg_text_context["past_key_values"] = self._slice_cache_prefix(
                    injected_kv,
                    cfg_text_seq_len,
                )
                cfg_text_context["kv_lens"] = [cfg_text_seq_len]
                cfg_text_context["ropes"] = [
                    self._first_rope(cfg_text_prefix_rope, cfg_text_seq_len)
                ]
            elif cfg_text_kv is not None:
                cfg_text_seq_len = cfg_text_kv.key_cache[0].shape[0]
                cfg_text_context["past_key_values"] = cfg_text_kv
                cfg_text_context["kv_lens"] = [cfg_text_seq_len]
                if cfg_text_metadata and "ropes" in cfg_text_metadata:
                    cfg_text_context["ropes"] = cfg_text_metadata["ropes"]
                else:
                    cfg_text_context["ropes"] = [cfg_text_seq_len]
            else:
                # No cfg_text companion received.  For text2img this is the
                # expected path: original BAGEL uses an empty KV cache (0
                # tokens) as the text-unconditional branch.  Keep the default
                # empty NaiveCache in cfg_text_context and preserve the
                # original cfg_text_scale so CFG still applies.
                pass

            if cfg_img_kv is None:
                # text2img multi-stage: cfg_img reuses gen KV (positive prompt,
                # no image), mirroring forward_cache_update_text on cfg_img_context
                # in the single-stage path.
                cfg_img_seq_len = injected_kv.key_cache[0].shape[0]
                cfg_img_context["past_key_values"] = injected_kv
                cfg_img_context["kv_lens"] = [cfg_img_seq_len]
                if req.sampling_params.kv_metadata and "ropes" in req.sampling_params.kv_metadata:
                    cfg_img_context["ropes"] = req.sampling_params.kv_metadata["ropes"]
                else:
                    cfg_img_context["ropes"] = [cfg_img_seq_len]
            else:
                cfg_img_seq_len = cfg_img_kv.key_cache[0].shape[0]
                cfg_img_context["past_key_values"] = cfg_img_kv
                cfg_img_context["kv_lens"] = [cfg_img_seq_len]
                if cfg_img_metadata and "ropes" in cfg_img_metadata:
                    cfg_img_context["ropes"] = cfg_img_metadata["ropes"]
                else:
                    cfg_img_context["ropes"] = [cfg_img_seq_len]

            if _truthy(extra_args.get("force_stage1_continue")):
                return self._force_stage1_continue_forward(
                    req,
                    gen_context,
                    cfg_text_context,
                    cfg_img_context,
                    image_shape,
                    extra_args,
                )

        else:
            if _truthy(extra_args.get("force_stage1_continue")):
                raise ValueError("force_stage1_continue requires injected KV cache from Stage 0.")

            image_input = (
                None
                if isinstance(first_prompt, str)
                else (
                    (first_prompt.get("multi_modal_data") or {}).get("image")
                    or (first_prompt.get("multi_modal_data") or {}).get("img2img")
                )
            )
            if image_input and not isinstance(image_input, list):
                image_input = [image_input]
            if image_input:
                image_input = [Image.open(image) if isinstance(image, str) else image for image in image_input]

            if image_input:
                # If we have an image, we prefill with it
                if self.vae:
                    # Canonical preprocessing: VAE and ViT use *independent*
                    # MaxLongEdgeMinShortEdgeResize ladders applied to the same
                    # source image. The prior shared `_resize_to_stride` (with
                    # stride=latent_downsample=8 and max=max_latent_size*8=512)
                    # forced ViT to receive 8-aligned dims and capped at 512px,
                    # diverging from the canonical 14-aligned dynamic ladder.

                    vit_transforms = self._vit_transform  # callable: PIL → CHW tensor in [-1,1]
                    vae_transforms = self._vae_transform  # callable: PIL → CHW tensor in [-1,1]

                    image_input_vae_resized = [
                        self._vae_transform.resize_only(img) for img in image_input
                    ]
                    # Local force_interleave_inference passes the VAE-resized
                    # image into update_context_image(..., vae=True, vit=True),
                    # so ViT also starts from this resized image.
                    image_input_vit = list(image_input_vae_resized)
                    resized_w, resized_h = image_input_vae_resized[0].size
                    image_shape = (resized_h, resized_w)
                    logger.info(
                        "img2img: VAE-resized image to %dx%d (stride 16, max 1024); ViT path follows local force input",
                        resized_w,
                        resized_h,
                    )
                    image_input = image_input_vae_resized

                    # Update gen_context with image (VAE + ViT)
                    gen_input_vae, newlens_vae, new_rope_vae = self.bagel.prepare_vae_images(
                        curr_kvlens=gen_context["kv_lens"],
                        curr_rope=gen_context["ropes"],
                        images=image_input,
                        transforms=vae_transforms,
                        new_token_ids=self.new_token_ids,
                    )
                    for k, v in gen_input_vae.items():
                        if torch.is_tensor(v):
                            gen_input_vae[k] = v.to(self.device)
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=self.device.type != "cpu",
                        dtype=self.od_config.dtype,
                    ):
                        gen_context["past_key_values"] = self.bagel.forward_cache_update_vae(
                            self.vae, gen_context["past_key_values"], **gen_input_vae
                        )
                    gen_context["kv_lens"] = newlens_vae
                    gen_context["ropes"] = new_rope_vae

                    gen_input_img, newlens_img, new_rope_img = self.bagel.prepare_vit_images(
                        curr_kvlens=gen_context["kv_lens"],
                        curr_rope=gen_context["ropes"],
                        images=image_input_vit,
                        transforms=vit_transforms,
                        new_token_ids=self.new_token_ids,
                    )
                    for k, v in gen_input_img.items():
                        if torch.is_tensor(v):
                            gen_input_img[k] = v.to(self.device)
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=self.device.type != "cpu",
                        dtype=self.od_config.dtype,
                    ):
                        gen_context["past_key_values"] = self.bagel.forward_cache_update_vit(
                            gen_context["past_key_values"], **gen_input_img
                        )
                    gen_context["kv_lens"] = newlens_img
                    gen_context["ropes"] = new_rope_img

                    cfg_text_context = deepcopy(gen_context)

            # Strip <|im_start|>/<|im_end|> wrappers that end2end.py may have
            # already added, so prepare_prompts doesn't double-add bos/eos.
            clean_prompt = prompt.removeprefix("<|im_start|>").removesuffix("<|im_end|>")

            # Update gen_context with text prompt
            generation_input, newlens, new_rope = self.bagel.prepare_prompts(
                curr_kvlens=gen_context["kv_lens"],
                curr_rope=gen_context["ropes"],
                prompts=[clean_prompt],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            # Fail fast with a clear error instead of CUDA gather OOB.
            max_tid = int(generation_input["packed_text_ids"].max().item())
            emb_n = int(self.language_model.vocab_size)
            if max_tid >= emb_n:
                raise ValueError(
                    "Tokenizer/model vocab mismatch: max token id "
                    f"{max_tid} >= embed_tokens size {emb_n}. "
                    "This usually means you're not using the tokenizer shipped with the Bagel checkpoint, "
                    "or llm_config.vocab_size is smaller than the tokenizer vocab."
                )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                gen_context["past_key_values"] = self.bagel.forward_cache_update_text(
                    gen_context["past_key_values"], **generation_input
                )
            gen_context["kv_lens"] = newlens
            gen_context["ropes"] = new_rope

            # cfg_text_context: update with negative prompt (no text condition).
            # When empty, keep cfg_text_context as-is (kv_lens=0) to match
            # original BAGEL; _merge_naive_caches handles None KV entries.
            neg_prompt = extra_args.get("negative_prompt", "")
            if neg_prompt:
                neg_input, neg_newlens, neg_rope = self.bagel.prepare_prompts(
                    curr_kvlens=cfg_text_context["kv_lens"],
                    curr_rope=cfg_text_context["ropes"],
                    prompts=[neg_prompt],
                    tokenizer=self.tokenizer,
                    new_token_ids=self.new_token_ids,
                )
                for k, v in neg_input.items():
                    if torch.is_tensor(v):
                        neg_input[k] = v.to(self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type != "cpu",
                    dtype=self.od_config.dtype,
                ):
                    cfg_text_context["past_key_values"] = self.bagel.forward_cache_update_text(
                        cfg_text_context["past_key_values"], **neg_input
                    )
                cfg_text_context["kv_lens"] = neg_newlens
                cfg_text_context["ropes"] = neg_rope

            # cfg_img_context: update with text prompt (no image condition)
            cfg_img_generation_input, cfg_img_newlens, cfg_img_new_rope = self.bagel.prepare_prompts(
                curr_kvlens=cfg_img_context["kv_lens"],
                curr_rope=cfg_img_context["ropes"],
                prompts=[clean_prompt],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            for k, v in cfg_img_generation_input.items():
                if torch.is_tensor(v):
                    cfg_img_generation_input[k] = v.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                cfg_img_context["past_key_values"] = self.bagel.forward_cache_update_text(
                    cfg_img_context["past_key_values"], **cfg_img_generation_input
                )
            cfg_img_context["kv_lens"] = cfg_img_newlens
            cfg_img_context["ropes"] = cfg_img_new_rope

        # ---- Detect output modality and think mode ----
        modalities = first_prompt.get("modalities", []) if isinstance(first_prompt, dict) else []
        is_text_output = "text" in modalities
        think_enabled = extra_args.get("think", False)
        think_text = None

        if think_enabled and injected_kv is None:
            max_think_tokens = int(extra_args.get("max_think_tokens", 1000))
            do_sample = bool(extra_args.get("do_sample", False))
            text_temperature = float(extra_args.get("text_temperature", 0.3))

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                start_input = self.bagel.prepare_start_tokens(
                    gen_context["kv_lens"], gen_context["ropes"], self.new_token_ids
                )
                for k, v in start_input.items():
                    if torch.is_tensor(v):
                        start_input[k] = v.to(self.device)

                gen_ctx_copy = deepcopy(gen_context)
                token_ids = self.bagel.generate_text(
                    past_key_values=gen_ctx_copy["past_key_values"],
                    max_length=max_think_tokens,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    end_token_id=self.new_token_ids["eos_token_id"],
                    **start_input,
                )
                # token_ids shape: (seq_len, batch=1)
                decoded = self.tokenizer.decode(token_ids[:, 0].tolist())
                # Strip chat markers to get clean text
                think_text = decoded.split("<|im_end|>")[0]
                if "<|im_start|>" in think_text:
                    think_text = think_text.split("<|im_start|>")[-1]
                logger.info("Think mode generated %d tokens", token_ids.shape[0])

            if not is_text_output:
                # Use the autoregressive KV cache from think generation
                # directly, instead of decode→re-encode which adds extra
                # bos/eos and may alter tokenization.
                num_think_tokens = token_ids.shape[0]
                gen_context["past_key_values"] = gen_ctx_copy["past_key_values"]
                gen_context["kv_lens"] = [kl + num_think_tokens for kl in gen_context["kv_lens"]]
                gen_context["ropes"] = [r + num_think_tokens for r in gen_context["ropes"]]

        # ---- Text-only output (text2text / img2text) ----
        if is_text_output and injected_kv is None:
            if think_text is not None:
                # Think mode already generated the text (including reasoning)
                text_output = think_text
            else:
                max_text_tokens = int(extra_args.get("max_think_tokens", 500))
                do_sample = bool(extra_args.get("do_sample", False))
                text_temperature = float(extra_args.get("text_temperature", 0.3))

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type != "cpu",
                    dtype=self.od_config.dtype,
                ):
                    start_input = self.bagel.prepare_start_tokens(
                        gen_context["kv_lens"], gen_context["ropes"], self.new_token_ids
                    )
                    for k, v in start_input.items():
                        if torch.is_tensor(v):
                            start_input[k] = v.to(self.device)
                    token_ids = self.bagel.generate_text(
                        past_key_values=gen_context["past_key_values"],
                        max_length=max_text_tokens,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        end_token_id=self.new_token_ids["eos_token_id"],
                        **start_input,
                    )
                    decoded = self.tokenizer.decode(token_ids[:, 0].tolist())
                    text_output = decoded.split("<|im_end|>")[0]
                    if "<|im_start|>" in text_output:
                        text_output = text_output.split("<|im_start|>")[-1]

            return DiffusionOutput(
                output=text_output,
                custom_output={"text_output": text_output},
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        # ---- Image generation (text2img / img2img) ----
        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        generation_input = self.bagel.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        # Fail fast for special tokens used by the image path as well.
        max_tid_img = int(generation_input["packed_text_ids"].max().item())
        emb_n = int(self.language_model.vocab_size)
        if max_tid_img >= emb_n:
            raise ValueError(
                "Tokenizer/model vocab mismatch (image path): max token id "
                f"{max_tid_img} >= embed_tokens size {emb_n}. "
                "This indicates the tokenizer token IDs do not match the checkpoint embeddings."
            )
        # Position ids must be non-negative; negative ids can trigger CUDA gather OOB inside RoPE.
        min_pid = int(generation_input["packed_position_ids"].min().item())
        if min_pid < 0:
            raise ValueError(f"Invalid packed_position_ids: min={min_pid} (must be >= 0)")
        # Latent position embedding bounds check: ids must be < max_latent_size^2.
        max_lat_pid = int(generation_input["packed_vae_position_ids"].max().item())
        max_lat_pid_allowed = int(self.bagel.max_latent_size * self.bagel.max_latent_size) - 1
        if max_lat_pid > max_lat_pid_allowed:
            raise ValueError(
                "Invalid packed_vae_position_ids (latent position embedding OOB): "
                f"max={max_lat_pid} > allowed_max={max_lat_pid_allowed}. "
                f"Requested image_shape={image_shape}, max_latent_size={self.bagel.max_latent_size}."
            )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(self.device)

        # text cfg
        generation_input_cfg_text = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            image_sizes=[image_shape],
        )
        # img cfg
        generation_input_cfg_img = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_context["kv_lens"],
            curr_rope=cfg_img_context["ropes"],
            image_sizes=[image_shape],
        )
        for k, v in generation_input_cfg_text.items():
            if torch.is_tensor(v):
                generation_input_cfg_text[k] = v.to(self.device)
        for k, v in generation_input_cfg_img.items():
            if torch.is_tensor(v):
                generation_input_cfg_img[k] = v.to(self.device)

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            latents, trajectory_latents, trajectory_timesteps, trajectory_log_probs = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_img_past_key_values=cfg_img_context["past_key_values"],
                num_timesteps=gen_params.num_timesteps,
                timestep_shift=gen_params.timestep_shift,
                cfg_text_scale=gen_params.cfg_text_scale,
                cfg_img_scale=gen_params.cfg_img_scale,
                cfg_interval=gen_params.cfg_interval,
                cfg_renorm_min=gen_params.cfg_renorm_min,
                cfg_renorm_type=gen_params.cfg_renorm_type,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
                cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
                return_trajectory_latents=req.sampling_params.return_trajectory_latents,
                scheduler=self.scheduler,
                scheduler_kwargs=self.scheduler_kwargs,
            )

        img = self._decode_image_from_latent(self.bagel, self.vae, latents[0], image_shape)

        # Build trajectory output when requested
        trajectory_latents_stacked: torch.Tensor | None = None
        trajectory_timesteps_stacked: torch.Tensor | None = None
        trajectory_decoded: list[Image.Image] | None = None
        if trajectory_latents:
            trajectory_latents_stacked = torch.stack(trajectory_latents)
            trajectory_timesteps_stacked = torch.stack(trajectory_timesteps)
            if req.sampling_params.return_trajectory_decoded:
                trajectory_decoded = [
                    self._decode_image_from_latent(self.bagel, self.vae, lat, image_shape) for lat in trajectory_latents
                ]

        trajectory_log_probs_stacked: torch.Tensor | None = None
        if trajectory_log_probs:
            trajectory_log_probs_stacked = torch.stack(trajectory_log_probs)

        custom = {}
        if think_text is not None:
            custom["think_text"] = think_text

        return DiffusionOutput(
            output=img,
            trajectory_latents=trajectory_latents_stacked,
            trajectory_timesteps=trajectory_timesteps_stacked,
            trajectory_log_probs=trajectory_log_probs_stacked,
            trajectory_decoded=trajectory_decoded,
            custom_output=custom,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state = self.state_dict()
        allowed = set(state.keys())
        shapes = {k: tuple(v.shape) for k, v in state.items()}

        tp_aware_params = {name for name, p in self.named_parameters() if hasattr(p, "weight_loader")}

        # Expand allowed/tp_aware_params with stacked param source names.
        # QKVParallelLinear merges q_proj+k_proj+v_proj into qkv_proj; the
        # checkpoint stores the original separate names.  We must recognise
        # those names so _filtered_weights does not drop them.
        _stacked_expansions = [
            (".qkv_proj", ".q_proj"),
            (".qkv_proj", ".k_proj"),
            (".qkv_proj", ".v_proj"),
            (".qkv_proj_moe_gen", ".q_proj_moe_gen"),
            (".qkv_proj_moe_gen", ".k_proj_moe_gen"),
            (".qkv_proj_moe_gen", ".v_proj_moe_gen"),
            (".gate_up_proj", ".gate_proj"),
            (".gate_up_proj", ".up_proj"),
        ]
        stacked_source_names: set[str] = set()
        for name in list(allowed):
            for target_suffix, source_suffix in _stacked_expansions:
                if target_suffix in name:
                    stacked_source_names.add(name.replace(target_suffix, source_suffix))
        allowed.update(stacked_source_names)
        tp_aware_params.update(stacked_source_names)

        def _normalize_name(name: str) -> str:
            # Common wrappers/prefixes in checkpoints.
            for pfx in ("module.", "model."):
                if name.startswith(pfx):
                    name = name[len(pfx) :]
            # Common component renames across repos.
            if name.startswith("vae_model."):
                name = "vae." + name[len("vae_model.") :]
            # Bagel `ae.safetensors` commonly stores AE weights without a top-level prefix.
            # Map them into this pipeline's `vae.*` namespace.
            if name.startswith("encoder.") or name.startswith("decoder."):
                name = "vae." + name
            return name

        def _iter_candidate_names(name: str) -> Iterable[str]:
            """Yield candidate parameter names in this pipeline for a checkpoint key.

            The upstream Bagel repo typically stores Bagel-core layers (time_embedder,
            latent_pos_embed, vae2llm, llm2vae, etc.) at the top-level of the model,
            while this vllm-omni integration nests them under `self.bagel`.
            """
            n = _normalize_name(name)
            yield n

            # Map Bagel core layers from top-level -> `bagel.*` namespace.
            for pfx in ("time_embedder.", "latent_pos_embed.", "vae2llm.", "llm2vae."):
                if n.startswith(pfx):
                    yield "bagel." + n
                    break

            # Map connector and vit_pos_embed to `bagel.*`
            for pfx in ("connector.", "vit_pos_embed."):
                if n.startswith(pfx):
                    yield "bagel." + n
                    break

            if n.startswith("vit_model."):
                yield "bagel." + n  # matches self.bagel.vit_model
            elif n.startswith("vision_model."):
                yield "bagel.vit_model." + n
            elif n.startswith("model.vision_model."):
                yield "bagel.vit_model." + n[len("model.") :]

        def _filtered_weights():
            total = 0
            kept = 0
            shape_mismatch = 0
            for name, tensor in weights:
                total += 1
                picked = None
                for cand in _iter_candidate_names(name):
                    if cand in allowed:
                        # Only accept if tensor shape matches target param/buffer shape.
                        if tuple(tensor.shape) == shapes.get(cand) or cand in tp_aware_params:
                            picked = cand
                            break
                        else:
                            if cand.endswith("bagel.latent_pos_embed.pos_embed") and tensor.ndim == 2:
                                npos, hdim = tensor.shape
                                side = isqrt(int(npos))
                                if side * side == int(npos) and hdim == int(self.bagel.hidden_size):
                                    param = self.bagel.latent_pos_embed.pos_embed
                                    # Resize in-place to keep the same Parameter object.
                                    param.data = param.data.new_empty((npos, hdim))
                                    # Update model bookkeeping so position-id generation matches.
                                    self.bagel.max_latent_size = int(side)
                                    if hasattr(self.bagel, "config"):
                                        setattr(self.bagel.config, "max_latent_size", int(side))
                                    if hasattr(self.bagel.latent_pos_embed, "max_num_patch_per_side"):
                                        self.bagel.latent_pos_embed.max_num_patch_per_side = int(side)
                                    shapes[cand] = (npos, hdim)
                                    picked = cand
                                    break
                            # Handle flattened patch embedding for SigLIP
                            if cand.endswith("embeddings.patch_embedding.weight") and tensor.ndim == 2:
                                # Checkpoint has (Hidden, C*P*P), model expects (Hidden, C, P, P)
                                if shapes.get(cand) is not None:
                                    target_shape = shapes[cand]
                                    if tensor.numel() == torch.prod(torch.tensor(target_shape)):
                                        # Reshape tensor to match target
                                        tensor = tensor.view(target_shape)
                                        picked = cand
                                        break

                            shape_mismatch += 1
                            # Keep this quiet; shape mismatches are expected for ignored modules.
                if picked is not None:
                    kept += 1
                    yield picked, tensor
                # else: ignore extra weights (e.g. connector/vision/und)
            logger.info_once(
                "BagelPipeline weight filter kept %d/%d tensors (shape mismatches seen: %d)",
                kept,
                total,
                shape_mismatch,
            )

        loader = AutoWeightsLoader(self)
        return loader.load_weights(_filtered_weights())

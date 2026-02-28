# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
BagelPipeline implementation for vLLM-Omni.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from math import isqrt

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
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .autoencoder import AutoEncoder, AutoEncoderParams
from .bagel_transformer import Bagel, NaiveCache, Qwen2MoTConfig, Qwen2MoTForCausalLM

logger = init_logger(__name__)


@dataclass
class BagelGenParams:
    num_timesteps: int = 50
    timestep_shift: float = 3.0
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.5
    cfg_interval: tuple = (0.4, 1.0)
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"


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
        return outputs.last_hidden_state.squeeze(0)


class BagelPipeline(nn.Module):
    """Bagel generation pipeline (MoT) packaged for vllm-omni diffusion engine.

    This pipeline is self-contained and uses the ported Bagel core files.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

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

        tok_len = len(self.tokenizer)
        required_max_id = max(int(v) for v in self.new_token_ids.values())
        llm_config.vocab_size = max(
            int(getattr(llm_config, "vocab_size", tok_len)),
            int(tok_len),
            int(required_max_id + 1),
        )

        self.language_model = Qwen2MoTForCausalLM(llm_config)
        ae_params: AutoEncoderParams = default_ae_params()
        self.vae = AutoEncoder(ae_params)

        self.bagel = Bagel(
            language_model=self.language_model,
            vit_model=self.vit_model,
            config=BagelConfig(
                llm_config=llm_config,
                vae_config=vae_cfg,
                vit_config=vit_cfg,
                vit_max_num_patch_per_side=int(bagel_cfg.get("vit_max_num_patch_per_side", 70)),
                connector_act=str(bagel_cfg.get("connector_act", "gelu_pytorch_tanh")),
                interpolate_pos=bool(bagel_cfg.get("interpolate_pos", False)),
                latent_patch_size=int(bagel_cfg.get("latent_patch_size", 2)),
                max_latent_size=int(bagel_cfg.get("max_latent_size", 32)),
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

        self.to(self.device)

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

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
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
        cfg_text_scale = extra_args.get("cfg_text_scale", 4.0)
        cfg_img_scale = extra_args.get("cfg_img_scale", 1.5)

        gen_params = BagelGenParams(
            num_timesteps=int(req.sampling_params.num_inference_steps or 50),
            timestep_shift=3.0,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
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
            gen_context["ropes"] = [seq_len]

            # Disable CFG: single KV cache cannot support 3-branch CFG
            logger.warning("CFG is disabled when using injected KV Cache")
            gen_params = BagelGenParams(
                num_timesteps=gen_params.num_timesteps,
                timestep_shift=gen_params.timestep_shift,
                cfg_text_scale=1.0,
                cfg_img_scale=1.0,
            )

        else:
            image_input = (
                None if isinstance(first_prompt, str) else (first_prompt.get("multi_modal_data") or {}).get("image")
            )
            if image_input and not isinstance(image_input, list):
                image_input = [image_input]
            if image_input:
                image_input = [Image.open(image) if isinstance(image, str) else image for image in image_input]

            if image_input:
                # If we have an image, we prefill with it
                if self.image_processor and self.vae:

                    def vit_transforms(img):
                        return self.image_processor(images=img, return_tensors="pt").pixel_values[0]

                    stride = self.bagel.latent_downsample
                    max_img_size = int(self.bagel.max_latent_size * stride)

                    def _resize_to_stride(img):
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        w, h = img.size
                        # Scale down if longest edge exceeds max
                        scale = min(max_img_size / max(w, h), 1.0)
                        # Scale up if shortest edge is too small (min 256)
                        min_img_size = min(256, max_img_size)
                        scale = max(scale, min_img_size / min(w, h))
                        new_w = max(stride, int(round(w * scale / stride) * stride))
                        new_h = max(stride, int(round(h * scale / stride) * stride))
                        # Clamp to max
                        new_w = min(new_w, max_img_size)
                        new_h = min(new_h, max_img_size)
                        if new_w != w or new_h != h:
                            img = img.resize((new_w, new_h), Image.BICUBIC)
                        return img

                    image_input = [_resize_to_stride(img) for img in image_input]

                    resized_w, resized_h = image_input[0].size
                    image_shape = (resized_h, resized_w)
                    logger.info(f"img2img: resized image to {resized_w}x{resized_h}")

                    def vae_transforms(img):
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        # Convert to [-1, 1] tensor (H, W, C) -> (C, H, W)
                        arr = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
                        return arr.permute(2, 0, 1)

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
                        images=image_input,
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

            # Update gen_context with text prompt
            generation_input, newlens, new_rope = self.bagel.prepare_prompts(
                curr_kvlens=gen_context["kv_lens"],
                curr_rope=gen_context["ropes"],
                prompts=[prompt],
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

            # cfg_text_context: update with negative prompt (no text condition)
            neg_prompt = extra_args.get("negative_prompt", "")
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
                prompts=[prompt],
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
            latents = self.bagel.generate_image(
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
            )

        img = self._decode_image_from_latent(self.bagel, self.vae, latents[0], image_shape)
        return DiffusionOutput(output=img)

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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
BagelPipeline implementation for vLLM-Omni.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from math import isqrt
from typing import TYPE_CHECKING, Any, ClassVar

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
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .autoencoder import AutoEncoder, AutoEncoderParams, DistributedAutoEncoder
from .bagel_transformer import Bagel, NaiveCache, Qwen2MoTConfig, Qwen2MoTForCausalLM

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


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


def _resolve_bagel_image_geometry(od_config: OmniDiffusionConfig) -> tuple[int, int]:
    """Return the effective ``(latent_downsample, max_latent_size)``.

    Some BAGEL checkpoints advertise a stale ``max_latent_size`` in
    ``config.json``.  Model loading already corrects it from the positional
    embedding weight, so admission preprocessing must do the same or it can
    compute a different img2img shape from the Worker.
    """

    model = getattr(od_config, "model", None)
    if not model:
        raise ValueError("BAGEL img2img preprocessing requires od_config.model.")
    if os.path.exists(model):
        model_path = model
    else:
        model_path = download_weights_from_hf_specific(
            model,
            None,
            ["*"],
            revision=getattr(od_config, "revision", None),
        )

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        bagel_cfg = json.load(f)

    vae_config = bagel_cfg.get("vae_config") or {}
    latent_downsample = int(vae_config.get("downsample", 8)) * int(bagel_cfg.get("latent_patch_size", 2))
    max_latent_size = int(bagel_cfg.get("max_latent_size", 32))

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map") or {}
        position_key = next(
            (key for key in ("latent_pos_embed.pos_embed", "bagel.latent_pos_embed.pos_embed") if key in weight_map),
            None,
        )
        if position_key is not None:
            from safetensors import safe_open

            shard_path = os.path.join(model_path, weight_map[position_key])
            with safe_open(shard_path, framework="pt") as f:
                num_positions = int(f.get_slice(position_key).get_shape()[0])
            inferred_size = isqrt(num_positions)
            if inferred_size * inferred_size != num_positions:
                raise ValueError(f"BAGEL latent position embedding length must be a square, got {num_positions}.")
            max_latent_size = inferred_size

    return latent_downsample, max_latent_size


def _bagel_effective_image_size(
    image_size: tuple[int, int],
    *,
    latent_downsample: int,
    max_latent_size: int,
) -> tuple[int, int]:
    """Match BAGEL's img2img resize and return ``(height, width)``."""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"BAGEL img2img input must have positive dimensions, got {width}x{height}.")

    max_image_size = int(max_latent_size * latent_downsample)
    scale = min(max_image_size / max(width, height), 1.0)
    min_image_size = min(256, max_image_size)
    scale = max(scale, min_image_size / min(width, height))
    resized_width = max(
        latent_downsample,
        int(round(width * scale / latent_downsample) * latent_downsample),
    )
    resized_height = max(
        latent_downsample,
        int(round(height * scale / latent_downsample) * latent_downsample),
    )
    return min(resized_height, max_image_size), min(resized_width, max_image_size)


def get_bagel_pre_process_func(od_config: OmniDiffusionConfig):
    """Resolve BAGEL execution mode and step-batch compatibility."""

    step_execution = bool(getattr(od_config, "step_execution", False))
    image_geometry: tuple[int, int] | None = None

    def pre_process_func(request: OmniDiffusionRequest):
        nonlocal image_geometry

        sampling = request.sampling_params
        kv_metadata = getattr(sampling, "kv_metadata", None) or {}
        image_shape = kv_metadata.get("image_shape")
        if image_shape is not None:
            sampling.height, sampling.width = (int(value) for value in image_shape)
        elif isinstance(request.prompt, dict):
            modalities = request.prompt.get("modalities") or []
            multi_modal_data = request.prompt.get("multi_modal_data") or {}
            image_input = multi_modal_data.get("img2img")
            if image_input is None and "text" not in modalities:
                image_input = multi_modal_data.get("image")
            if isinstance(image_input, list):
                image_input = image_input[0] if image_input else None
            if image_input is not None:
                if isinstance(image_input, str):
                    with Image.open(image_input) as image:
                        input_size = image.size
                else:
                    input_size = image_input.size
                if image_geometry is None:
                    image_geometry = _resolve_bagel_image_geometry(od_config)
                latent_downsample, max_latent_size = image_geometry
                sampling.height, sampling.width = _bagel_effective_image_size(
                    input_size,
                    latent_downsample=latent_downsample,
                    max_latent_size=max_latent_size,
                )

        extra_args = request.sampling_params.extra_args or {}
        cfg_interval = tuple(extra_args.get("cfg_interval", (0.4, 1.0)))
        request.batch_compatibility_key = (
            "bagel_cfg",
            float(extra_args.get("cfg_text_scale", 4.0)),
            float(extra_args.get("cfg_img_scale", 1.5)),
            cfg_interval,
            extra_args.get("cfg_renorm_type", "global"),
            float(extra_args.get("cfg_renorm_min", 0.0)),
        )

        # Preserve legacy plain-string behavior: only an explicit modality
        # request selects Bagel.generate_text() instead of stepwise denoising.
        if step_execution and isinstance(request.prompt, dict):
            modalities = request.prompt.get("modalities") or []
            if "text" in modalities:
                request.use_step_execution = False
        return request

    return pre_process_func


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


class BagelPipeline(nn.Module, SupportsComponentDiscovery, DiffusionPipelineProfilerMixin):
    """Bagel generation pipeline (MoT) packaged for vllm-omni diffusion engine.

    This pipeline is self-contained and uses the ported Bagel core files.
    """

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
    supports_step_execution: ClassVar[bool] = True

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
        self.vae = DistributedAutoEncoder(ae_params)

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
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

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

    def _regen_init_noise_on_device(self, gen_input: dict, seed: int | None) -> None:
        """Resample ``gen_input["packed_init_noises"]`` on-device with a fresh
        per-call ``torch.Generator``.

        ``Bagel.prepare_input`` (and the Lance video equivalent) call
        ``torch.randn`` with no device or generator, falling back to CPU+fp32
        via the global RNG.  Upstream Lance samples directly on CUDA+bf16 via
        ``torch.Generator(device=cuda).manual_seed(seed)`` (lance.py:1536),
        so for the same seed the two sides land on different noise streams.
        Mutates ``gen_input`` in place; no-op if seed is unset or device is CPU.
        """
        if seed is None or self.device.type != "cuda":
            return
        ref = gen_input["packed_init_noises"]
        gen_input["packed_init_noises"] = torch.randn(
            ref.shape,
            generator=torch.Generator(device=self.device).manual_seed(int(seed)),
            device=self.device,
            dtype=self.od_config.dtype,
        )

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning(
                """This model only supports a single prompt, not a batched request.""",
                """Taking only the first image for now.""",
            )
        return self._forward_single(req.prompts[0], req.sampling_params)

    @torch.inference_mode()
    def _forward_single(
        self,
        first_prompt: OmniPromptType,
        sampling: OmniDiffusionSamplingParams,
        *,
        prepare_only: bool = False,
    ) -> DiffusionOutput | dict[str, Any]:
        # TODO: In online mode, sometimes it receives [{"prompts": None}, {...}], so cannot use .get("...", "")
        # TODO: May be some data formatting operations on the API side. Hack for now.
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")

        max_hw = int(self.bagel.max_latent_size * self.bagel.latent_downsample)
        if sampling.height is None and sampling.width is None:
            height = width = max_hw
        else:
            height = int(sampling.height) if sampling.height is not None else max_hw
            width = int(sampling.width) if sampling.width is not None else max_hw
        if height > max_hw or width > max_hw:
            raise ValueError(
                f"Requested resolution {height}x{width} exceeds Bagel checkpoint limit "
                f"{max_hw}x{max_hw} (max_latent_size={self.bagel.max_latent_size}, "
                f"latent_downsample={self.bagel.latent_downsample})."
            )
        image_shape = (height, width)

        extra_args = getattr(sampling, "extra_args", {}) or {}
        cfg_text_scale = extra_args.get("cfg_text_scale", 4.0)
        cfg_img_scale = extra_args.get("cfg_img_scale", 1.5)

        cfg_interval = extra_args.get("cfg_interval", (0.4, 1.0))
        cfg_renorm_type = extra_args.get("cfg_renorm_type", "global")
        cfg_renorm_min = extra_args.get("cfg_renorm_min", 0.0)

        gen_params = BagelGenParams(
            num_timesteps=int(sampling.num_inference_steps or 50),
            timestep_shift=float(extra_args.get("timestep_shift", 3.0)),
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

        injected_kv = sampling.past_key_values
        if injected_kv is not None:
            logger.info("Using injected KV Cache (direct)")
            injected_kv = NaiveCache.from_object(injected_kv)
            gen_context["past_key_values"] = injected_kv
            seq_len = injected_kv.key_cache[0].shape[0]
            gen_context["kv_lens"] = [seq_len]
            if sampling.kv_metadata and "ropes" in sampling.kv_metadata:
                gen_context["ropes"] = sampling.kv_metadata["ropes"]
            else:
                gen_context["ropes"] = [seq_len]

            if sampling.kv_metadata and "image_shape" in sampling.kv_metadata:
                image_shape = tuple(sampling.kv_metadata["image_shape"])

            branch_kvs = getattr(sampling, "cfg_branch_past_key_values", None) or {}
            branch_metadata = getattr(sampling, "cfg_branch_kv_metadata", None) or {}
            active_branch = getattr(sampling, "cfg_active_branch", None)
            branch_roles = getattr(sampling, "cfg_branch_roles", None) or list(branch_kvs.keys())

            cfg_text_kv = getattr(sampling, "cfg_text_past_key_values", None) or branch_kvs.get("cfg_text")
            cfg_text_metadata = getattr(sampling, "cfg_text_kv_metadata", None) or branch_metadata.get("cfg_text")
            cfg_img_kv = getattr(sampling, "cfg_img_past_key_values", None) or branch_kvs.get("cfg_img")
            cfg_img_metadata = getattr(sampling, "cfg_img_kv_metadata", None) or branch_metadata.get("cfg_img")

            cfg_parallel_contract = (
                active_branch is not None or bool(branch_roles) or cfg_text_kv is not None or cfg_img_kv is not None
            )
            if cfg_parallel_contract:
                logger.info(
                    "CFG enabled with injected branch KV context roles=%s active=%s",
                    branch_roles,
                    active_branch,
                )

            if cfg_text_kv is not None:
                cfg_text_kv = NaiveCache.from_object(cfg_text_kv)
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
                if sampling.kv_metadata and "ropes" in sampling.kv_metadata:
                    cfg_img_context["ropes"] = sampling.kv_metadata["ropes"]
                else:
                    cfg_img_context["ropes"] = [cfg_img_seq_len]
            else:
                cfg_img_kv = NaiveCache.from_object(cfg_img_kv)
                cfg_img_seq_len = cfg_img_kv.key_cache[0].shape[0]
                cfg_img_context["past_key_values"] = cfg_img_kv
                cfg_img_context["kv_lens"] = [cfg_img_seq_len]
                if cfg_img_metadata and "ropes" in cfg_img_metadata:
                    cfg_img_context["ropes"] = cfg_img_metadata["ropes"]
                else:
                    cfg_img_context["ropes"] = [cfg_img_seq_len]

        else:
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
                    for k in ("packed_indexes", "packed_key_value_indexes", "key_values_lens"):
                        gen_input_img.pop(k, None)
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
            # original BAGEL.
            prompt_negative = first_prompt.get("negative_prompt") if isinstance(first_prompt, dict) else None
            neg_prompt = prompt_negative if prompt_negative is not None else extra_args.get("negative_prompt", "")
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
                output={
                    "payload": {"text": text_output},
                    "metadata": {"text": {"text_output": text_output}},
                },
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        # ---- Image generation (text2img / img2img) ----
        if sampling.seed is not None:
            torch.manual_seed(sampling.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(sampling.seed)

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

        # NOTE: For now we disable device specific noise regeneration so that e2e tests can run
        # on both CUDA and ROCm. Context: https://github.com/vllm-project/vllm-omni/pull/4081
        # self._regen_init_noise_on_device(generation_input, sampling.seed)

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

        if prepare_only:
            return {
                "generation_input": generation_input,
                "cfg_text_packed_position_ids": generation_input_cfg_text["cfg_packed_position_ids"],
                "cfg_img_packed_position_ids": generation_input_cfg_img["cfg_packed_position_ids"],
                "gen_context": gen_context,
                "cfg_text_context": cfg_text_context,
                "cfg_img_context": cfg_img_context,
                "gen_params": gen_params,
                "image_shape": image_shape,
                "think_text": think_text,
            }

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
                return_trajectory_latents=sampling.return_trajectory_latents,
                scheduler=self.scheduler,
                scheduler_kwargs=self.scheduler_kwargs,
            )

        return self._build_image_output(
            latents[0],
            image_shape,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            trajectory_log_probs=trajectory_log_probs,
            return_trajectory_decoded=sampling.return_trajectory_decoded,
            think_text=think_text,
        )

    def _build_image_output(
        self,
        latent: torch.Tensor,
        image_shape: tuple[int, int],
        *,
        trajectory_latents: list[torch.Tensor] | None = None,
        trajectory_timesteps: list[torch.Tensor] | None = None,
        trajectory_log_probs: list[torch.Tensor] | None = None,
        return_trajectory_decoded: bool = False,
        think_text: str | None = None,
    ) -> DiffusionOutput:
        img = self._decode_image_from_latent(self.bagel, self.vae, latent, image_shape)

        # Build trajectory output when requested
        trajectory_latents_stacked: torch.Tensor | None = None
        trajectory_timesteps_stacked: torch.Tensor | None = None
        trajectory_decoded: list[Image.Image] | None = None
        if trajectory_latents:
            trajectory_latents_stacked = torch.stack(trajectory_latents)
            trajectory_timesteps_stacked = torch.stack(trajectory_timesteps)
            if return_trajectory_decoded:
                trajectory_decoded = [
                    self._decode_image_from_latent(self.bagel, self.vae, lat, image_shape) for lat in trajectory_latents
                ]

        trajectory_log_probs_stacked: torch.Tensor | None = None
        if trajectory_log_probs:
            trajectory_log_probs_stacked = torch.stack(trajectory_log_probs)

        payload = {"image": img}
        metadata = {}
        if think_text is not None:
            metadata["text"] = {"think_text": think_text}
        trajectory_payload = {}
        if trajectory_latents_stacked is not None:
            trajectory_payload["latents"] = trajectory_latents_stacked
        if trajectory_timesteps_stacked is not None:
            trajectory_payload["timesteps"] = trajectory_timesteps_stacked
        if trajectory_log_probs_stacked is not None:
            trajectory_payload["log_probs"] = trajectory_log_probs_stacked
        if trajectory_decoded is not None:
            trajectory_payload["decoded"] = trajectory_decoded
        if trajectory_payload:
            payload["trajectory"] = trajectory_payload
            metadata["trajectory"] = {"type": "denoising"}

        return DiffusionOutput(
            output={
                "payload": payload,
                "metadata": metadata,
            },
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def prepare_encode(
        self,
        state: StepRequestState,
        **kwargs: object,
    ) -> StepRequestState:
        """Populate *state* with BAGEL inputs, latents, timesteps, and CFG config."""
        del kwargs
        if self.bagel._sp_size > 1:
            raise NotImplementedError("BAGEL step execution does not currently support sequence parallelism.")

        sampling = state.sampling
        prompt = state.prompt if state.prompt is not None else ""
        if isinstance(prompt, dict):
            if "text" in (prompt.get("modalities") or []):
                raise NotImplementedError("BAGEL text output is not supported by step execution.")

        ctx = self._forward_single(prompt, sampling, prepare_only=True)
        if not isinstance(ctx, dict):
            raise RuntimeError("BAGEL step preparation did not produce an image-generation context.")

        generation_input = ctx["generation_input"]
        latents = generation_input["packed_init_noises"]
        gen_params = ctx["gen_params"]
        timesteps, dts = self.bagel.prepare_denoise_schedule(
            latents,
            gen_params.num_timesteps,
            gen_params.timestep_shift,
        )

        # Keep request-local scheduler progress while sharing immutable config
        # and tensor buffers with the pipeline scheduler.
        req_scheduler = copy(self.scheduler)

        # Populate state from generation context.
        state.latents = latents
        state.timesteps = timesteps
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = gen_params.cfg_text_scale > 1.0
        state.img_shapes = [ctx["image_shape"]]
        state.extra.update(
            {
                "bagel_generation_input": {
                    key: value for key, value in generation_input.items() if key != "packed_init_noises"
                },
                "bagel_cfg_text_packed_position_ids": ctx["cfg_text_packed_position_ids"],
                "bagel_cfg_img_packed_position_ids": ctx["cfg_img_packed_position_ids"],
                "bagel_gen_context": ctx["gen_context"],
                "bagel_cfg_text_context": ctx["cfg_text_context"],
                "bagel_cfg_img_context": ctx["cfg_img_context"],
                "bagel_gen_params": gen_params,
                "bagel_dts": dts,
                "bagel_image_shape": ctx["image_shape"],
                "bagel_scheduler_kwargs": dict(self.scheduler_kwargs),
                "bagel_think_text": ctx["think_text"],
                "bagel_trajectory_latents": (
                    # Trajectory output needs immutable snapshots because each
                    # later scheduler update replaces or mutates the latents.
                    [latents.clone()] if sampling.return_trajectory_latents and len(timesteps) > 0 else []
                ),
                "bagel_trajectory_timesteps": [],
                "bagel_trajectory_log_probs": [],
            }
        )
        return state

    @staticmethod
    def _pack_step_generation_inputs(
        states: Sequence[StepRequestState],
    ) -> dict[str, torch.Tensor]:
        generation_inputs = [state.extra["bagel_generation_input"] for state in states]
        packed_seqlens = [generation_input["packed_seqlens"] for generation_input in generation_inputs]
        if any(seqlens.numel() != 1 for seqlens in packed_seqlens):
            raise ValueError("BAGEL step execution expects one packed sequence per request.")
        seq_lengths = [int(seqlens.item()) for seqlens in packed_seqlens]
        if len(set(seq_lengths)) != 1:
            raise ValueError("BAGEL step batching requires matching packed sequence lengths.")

        query_offset = 0
        text_indexes = []
        vae_indexes = []
        for generation_input, seq_len in zip(generation_inputs, seq_lengths, strict=True):
            text_indexes.append(generation_input["packed_text_indexes"] + query_offset)
            vae_indexes.append(generation_input["packed_vae_token_indexes"] + query_offset)
            query_offset += seq_len

        return {
            "packed_text_ids": torch.cat([item["packed_text_ids"] for item in generation_inputs]),
            "packed_text_indexes": torch.cat(text_indexes),
            "packed_vae_position_ids": torch.cat([item["packed_vae_position_ids"] for item in generation_inputs]),
            "packed_vae_token_indexes": torch.cat(vae_indexes),
            "packed_seqlens": torch.cat(packed_seqlens),
            "packed_position_ids": torch.cat([item["packed_position_ids"] for item in generation_inputs]),
        }

    def _build_denoise_kwargs(
        self,
        input_batch: InputBatch,
        states: Sequence[StepRequestState],
    ) -> dict[str, Any]:
        parallel_config = getattr(self.bagel, "parallel_config", None)
        cfg_parallel_size = getattr(parallel_config, "cfg_parallel_size", 1)

        packed = self._pack_step_generation_inputs(states)
        vae_lengths = [int(state.latents.shape[0]) for state in states]
        if sum(vae_lengths) != int(input_batch.latents.shape[0]):
            raise ValueError("BAGEL packed latent rows do not match InputBatch.latents.")

        cfg_settings = {
            (
                state.extra["bagel_gen_params"].cfg_text_scale,
                state.extra["bagel_gen_params"].cfg_img_scale,
                tuple(state.extra["bagel_gen_params"].cfg_interval),
                state.extra["bagel_gen_params"].cfg_renorm_type,
                state.extra["bagel_gen_params"].cfg_renorm_min,
            )
            for state in states
        }
        if len(cfg_settings) != 1:
            raise ValueError("Mixed BAGEL CFG settings cannot share one step batch.")
        cfg_text_scale, cfg_img_scale, _cfg_interval, cfg_renorm_type, cfg_renorm_min = next(iter(cfg_settings))

        cfg_text_scales = []
        cfg_img_scales = []
        for state in states:
            gen_params = state.extra["bagel_gen_params"]
            timestep = state.current_timestep
            if timestep is None:
                raise ValueError(f"BAGEL request {state.request_id} has no current timestep.")
            t_value = float(timestep.item())
            in_cfg_window = t_value > gen_params.cfg_interval[0] and t_value <= gen_params.cfg_interval[1]
            text_scale = gen_params.cfg_text_scale if in_cfg_window else 1.0
            cfg_text_scales.append(text_scale)
            cfg_img_scales.append(gen_params.cfg_img_scale if in_cfg_window and text_scale > 1.0 else 1.0)

        use_cfg_text = any(scale > 1.0 for scale in cfg_text_scales)
        use_cfg_img = use_cfg_text and any(scale > 1.0 for scale in cfg_img_scales)
        configured_cfg_text = any(state.extra["bagel_gen_params"].cfg_text_scale > 1.0 for state in states)
        configured_cfg_img = any(
            state.extra["bagel_gen_params"].cfg_text_scale > 1.0 and state.extra["bagel_gen_params"].cfg_img_scale > 1.0
            for state in states
        )
        build_cfg_text = use_cfg_text or (cfg_parallel_size > 1 and configured_cfg_text)
        build_cfg_img = use_cfg_img or (cfg_parallel_size > 1 and configured_cfg_img)
        gen_cache = NaiveCache.merge([state.extra["bagel_gen_context"]["past_key_values"] for state in states])
        cfg_branch_pids = None
        cfg_branch_caches = None
        if build_cfg_text:
            cfg_branch_pids = [
                packed["packed_position_ids"],
                torch.cat([state.extra["bagel_cfg_text_packed_position_ids"] for state in states]),
            ]
            cfg_branch_caches = [
                gen_cache,
                NaiveCache.merge([state.extra["bagel_cfg_text_context"]["past_key_values"] for state in states]),
            ]
            if build_cfg_img:
                cfg_branch_pids.append(
                    torch.cat([state.extra["bagel_cfg_img_packed_position_ids"] for state in states])
                )
                cfg_branch_caches.append(
                    NaiveCache.merge([state.extra["bagel_cfg_img_context"]["past_key_values"] for state in states])
                )

        return {
            "x_t": input_batch.latents,
            "timestep": input_batch.timesteps,
            "past_key_values": gen_cache,
            "cfg_renorm_min": cfg_renorm_min,
            "cfg_renorm_type": cfg_renorm_type,
            "cfg_text_scale": cfg_text_scale,
            "cfg_img_scale": cfg_img_scale,
            "cfg_branch_pids": cfg_branch_pids,
            "cfg_branch_caches": cfg_branch_caches,
            "cfg_vae_lengths": vae_lengths if use_cfg_text else None,
            "cfg_text_scales": cfg_text_scales if use_cfg_text else None,
            "cfg_img_scales": cfg_img_scales if use_cfg_text else None,
            **packed,
        }

    def _denoise_step_cfg_parallel(self, denoise_kwargs: dict[str, Any]) -> torch.Tensor:
        cfg_branch_pids = denoise_kwargs["cfg_branch_pids"]
        cfg_branch_caches = denoise_kwargs["cfg_branch_caches"]
        if cfg_branch_pids is None or cfg_branch_caches is None:
            return self.bagel.forward(**denoise_kwargs)

        cfg_world_size = get_classifier_free_guidance_world_size()
        num_branches = len(cfg_branch_pids)
        if cfg_world_size == 2 and num_branches == 3:
            raise ValueError(
                f"Image CFG (cfg_img_scale={denoise_kwargs['cfg_img_scale']}) requires cfg_parallel_size=3, "
                "but got cfg_parallel_size=2. Use cfg_parallel_size=3 to enable image CFG in parallel mode."
            )

        common_keys = (
            "x_t",
            "timestep",
            "packed_vae_token_indexes",
            "packed_vae_position_ids",
            "packed_text_ids",
            "packed_text_indexes",
            "packed_seqlens",
        )
        common = {key: denoise_kwargs[key] for key in common_keys}
        branches_kwargs = [
            {**common, "packed_position_ids": position_ids, "past_key_values": cache}
            for position_ids, cache in zip(cfg_branch_pids, cfg_branch_caches, strict=True)
        ]
        result = self.bagel.predict_noise_with_multi_branch_cfg(
            do_true_cfg=denoise_kwargs["cfg_text_scales"] is not None,
            true_cfg_scale={
                "cfg_text_scale": denoise_kwargs["cfg_text_scale"],
                "cfg_img_scale": denoise_kwargs["cfg_img_scale"],
                "cfg_renorm_type": denoise_kwargs["cfg_renorm_type"],
                "cfg_renorm_min": denoise_kwargs["cfg_renorm_min"],
                "cfg_vae_lengths": denoise_kwargs["cfg_vae_lengths"],
                "cfg_text_scales": denoise_kwargs["cfg_text_scales"],
                "cfg_img_scales": denoise_kwargs["cfg_img_scales"],
            },
            branches_kwargs=branches_kwargs,
        )
        if not isinstance(result, torch.Tensor):
            raise TypeError("BAGEL CFG parallel step must return one velocity tensor.")
        return result

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """One denoise step: pack request-local BAGEL state and run the model."""
        del kwargs
        states = tuple(states or input_batch.states)
        if not states:
            raise ValueError("BAGEL denoise_step requires at least one request state.")
        denoise_kwargs = self._build_denoise_kwargs(input_batch, states)
        parallel_config = getattr(self.bagel, "parallel_config", None)
        use_cfg_parallel = (
            getattr(parallel_config, "cfg_parallel_size", 1) > 1 and denoise_kwargs["cfg_branch_pids"] is not None
        )

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            if use_cfg_parallel:
                if any(state.step_index == 0 for state in states):
                    get_cfg_group().broadcast(input_batch.latents, src=0)
                return self._denoise_step_cfg_parallel(denoise_kwargs)
            return self.bagel.forward(**denoise_kwargs)

    def step_scheduler(
        self,
        state: StepRequestState,
        noise_pred: torch.Tensor,
        **kwargs: object,
    ) -> None:
        """One scheduler step: update ``state.latents`` and advance ``step_index``."""
        del kwargs
        t = state.current_timestep
        if t is None or state.latents is None:
            raise ValueError(f"BAGEL request {state.request_id} is not ready for a scheduler step.")
        dt = state.extra["bagel_dts"][state.step_index]
        log_prob = None
        if state.scheduler is not None:
            output = state.scheduler.step(
                noise_pred.to(state.latents.device),
                t,
                state.latents,
                dt,
                **state.extra["bagel_scheduler_kwargs"],
            )
            state.latents = output.prev_sample
            log_prob = getattr(output, "log_prob", None)
        else:
            state.latents = state.latents - noise_pred.to(state.latents.device) * dt

        if state.sampling.return_trajectory_latents:
            # Preserve this opt-in step snapshot before the next update.
            state.extra["bagel_trajectory_latents"].append(state.latents.clone())
            state.extra["bagel_trajectory_timesteps"].append(t)
            if log_prob is not None:
                state.extra["bagel_trajectory_log_probs"].append(log_prob)
        state.step_index += 1

    def post_decode(
        self,
        state: StepRequestState,
        **kwargs: object,
    ) -> DiffusionOutput:
        """Decode final latents from *state*."""
        del kwargs
        if state.latents is None:
            raise ValueError(f"BAGEL request {state.request_id} has no latents to decode.")
        image_shape = state.extra["bagel_image_shape"]
        return_trajectory_decoded = state.sampling.return_trajectory_decoded
        return self._build_image_output(
            state.latents,
            image_shape,
            trajectory_latents=state.extra["bagel_trajectory_latents"],
            trajectory_timesteps=state.extra["bagel_trajectory_timesteps"],
            trajectory_log_probs=state.extra["bagel_trajectory_log_probs"],
            return_trajectory_decoded=return_trajectory_decoded,
            think_text=state.extra["bagel_think_text"],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state = self.state_dict()
        allowed = set(state.keys())
        shapes = {k: tuple(v.shape) for k, v in state.items()}

        tp_aware_params = {name for name, p in self.named_parameters() if hasattr(p, "weight_loader")}

        # Expand allowed/tp_aware_params with stacked param source names.
        # The model fuses several checkpoint projections into merged layers:
        #   QKV: q/k/v_proj → qkv_proj, q/k/v_proj_moe_gen → qkv_proj.gen_exp
        # and remaps non-stacked weights like:
        #   {norm}_moe_gen.weight → {norm}.gen_weight  (MoTRMSNorm layers)
        # We expand allowed names so _filtered_weights does not drop them.
        _stacked_expansions = [
            # text QKV
            (".qkv_proj", ".q_proj"),
            (".qkv_proj", ".k_proj"),
            (".qkv_proj", ".v_proj"),
            # gen QKV
            (".qkv_proj.gen_exp", ".q_proj_moe_gen"),
            (".qkv_proj.gen_exp", ".k_proj_moe_gen"),
            (".qkv_proj.gen_exp", ".v_proj_moe_gen"),
            # gen o_proj (non-stacked, but still remapped)
            (".o_proj.gen_exp", ".o_proj_moe_gen"),
            # text FFN gate+up
            (".mlp.gate_up_proj", ".mlp.gate_proj"),
            (".mlp.gate_up_proj", ".mlp.up_proj"),
            # gen FFN gate+up
            (".mlp_moe_gen.gate_up_proj", ".mlp_moe_gen.gate_proj"),
            (".mlp_moe_gen.gate_up_proj", ".mlp_moe_gen.up_proj"),
            # MoTRMSNorm gen_weight ← checkpoint _moe_gen.weight
            (".input_layernorm.gen_", ".input_layernorm_moe_gen."),
            (".post_attention_layernorm.gen_", ".post_attention_layernorm_moe_gen."),
            (".q_norm.gen_", ".q_norm_moe_gen."),
            (".k_norm.gen_", ".k_norm_moe_gen."),
            (".norm.gen_", ".norm_moe_gen."),
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

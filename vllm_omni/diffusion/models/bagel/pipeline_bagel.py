# SPDX-License-Identifier: Apache-2.0
"""
BagelPipeline implementation for vLLM Omni.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from math import isqrt

import torch
from PIL import Image
from torch import nn
from transformers import AutoTokenizer
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .autoencoder import AutoEncoder, AutoEncoderParams
from .bagel_core import Bagel, BagelConfig
from .qwen2_navit import NaiveCache, Qwen2Config, Qwen2ForCausalLM
from .utils import BagelGenParams, add_special_tokens

logger = init_logger(__name__)


def get_bagel_post_process_func(od_config: OmniDiffusionConfig):
    # BagelPipeline returns PIL.Image.Image directly.
    def post_process_func(x):
        return x

    return post_process_func


@dataclass
class _VaeCfg:
    z_channels: int = 16
    downsample: int = 8


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
        llm_config = Qwen2Config.from_json_file(llm_cfg_path)
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
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        # IMPORTANT: ensure model vocab_size can cover tokenizer ids. Otherwise embedding
        # lookup will hit CUDA gather OOB during inference.
        try:
            tok_len = len(self.tokenizer)
        except Exception:  # pragma: no cover - very old tokenizers
            tok_len = getattr(self.tokenizer, "vocab_size", llm_config.vocab_size)
        required_max_id = max(int(v) for v in self.new_token_ids.values())
        llm_config.vocab_size = max(
            int(getattr(llm_config, "vocab_size", tok_len)),
            int(tok_len),
            int(required_max_id + 1),
        )

        # Build modules (weights loaded later by DiffusersPipelineLoader/AutoWeightsLoader)
        self.language_model = Qwen2ForCausalLM(llm_config)

        # AutoEncoder architecture is fixed for Bagel release; weights come from `ae.safetensors`.
        # Keep params compatible with original.
        ae_params: AutoEncoderParams = default_ae_params()
        self.vae = AutoEncoder(ae_params)

        self.bagel = Bagel(
            language_model=self.language_model,
            config=BagelConfig(
                visual_gen=True,
                visual_und=False,  # Explicitly disabled
                llm_config=llm_config,
                vae_config=vae_cfg,
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
        prompt = req.prompt or ""
        if isinstance(prompt, list):
            # vllm-omni request supports list; Bagel pipeline currently supports first prompt.
            prompt = prompt[0] if prompt else ""

        # Choose target resolution.
        # Bagel latent positional embedding supports up to:
        #   max_hw = max_latent_size * latent_downsample
        # Exceeding this will cause index-out-of-bounds in latent position embedding (CUDA gather).
        max_hw = int(self.bagel.max_latent_size * self.bagel.latent_downsample)
        if req.height is None and req.width is None:
            height = width = max_hw
        else:
            height = int(req.height) if req.height is not None else max_hw
            width = int(req.width) if req.width is not None else max_hw
        if height > max_hw or width > max_hw:
            raise ValueError(
                f"Requested resolution {height}x{width} exceeds Bagel checkpoint limit "
                f"{max_hw}x{max_hw} (max_latent_size={self.bagel.max_latent_size}, "
                f"latent_downsample={self.bagel.latent_downsample})."
            )
        image_shape = (height, width)

        # Map request params to Bagel gen params (defaults follow Bagel inferencer)
        gen_params = BagelGenParams(
            cfg_text_scale=float(req.guidance_scale or 4.0),
            cfg_img_scale=float(req.guidance_scale_2 or 1.5),
            cfg_interval=(0.4, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            num_timesteps=int(req.num_inference_steps or 50),
            timestep_shift=3.0,
        )

        # Context init. Bagel uses per-layer KV cache; keep three contexts for (gen / cfg_text / cfg_img).
        # NOTE: In Bagel, CFG paths still expect consistent KV cache objects + kv_lens/rope; otherwise
        # varlen attention merge will fail when past K/V are missing.
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

        # Optional image conditioning: if provided, just treat it as "present image" input.
        # NOTE: Full image-conditioned editing requires prepare_vae_images + cache update; not yet wired here.
        if req.pil_image is not None:
            # Keep prompt consistent; for now we only use it to set output size.
            _ = req.pil_image  # reserved
            # In practice you would encode the image into context here.
            gen_params.cfg_img_scale = 1.0

        # Initialize cfg_text_context BEFORE text update (unconditional on text).
        cfg_text_context = copy.deepcopy(gen_context)

        # Add text prompt (prefill) on gen context.
        generation_input, newlens, new_rope = self.bagel.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            prompts=[prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        # Fail fast with a clear error instead of CUDA gather OOB.
        max_tid = int(generation_input["packed_text_ids"].max().item())
        emb_n = int(self.language_model.model.embed_tokens.weight.shape[0])
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
        with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_text(
                gen_context["past_key_values"], **generation_input
            )
        gen_context["kv_lens"] = newlens
        gen_context["ropes"] = new_rope

        # Initialize cfg_img_context AFTER text update (conditional on text, but maybe unconditional on image later).
        # Typically cfg_img_context mirrors gen_context for text-to-image.
        cfg_img_context = copy.deepcopy(gen_context)

        # Prepare latent query and run flow
        generation_input = self.bagel.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        # Fail fast for special tokens used by the image path as well.
        max_tid_img = int(generation_input["packed_text_ids"].max().item())
        emb_n = int(self.language_model.model.embed_tokens.weight.shape[0])
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

        cfg_text_latent = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            image_sizes=[image_shape],
        )
        cfg_img_latent = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_context["kv_lens"],
            curr_rope=cfg_img_context["ropes"],
            image_sizes=[image_shape],
        )
        for d in (cfg_text_latent, cfg_img_latent):
            for k, v in d.items():
                if torch.is_tensor(v):
                    d[k] = v.to(self.device)

        with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
            latents = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_img_past_key_values=cfg_img_context["past_key_values"],
                num_timesteps=gen_params.num_timesteps,
                cfg_text_scale=gen_params.cfg_text_scale,
                cfg_img_scale=gen_params.cfg_img_scale,
                cfg_interval=gen_params.cfg_interval,
                cfg_renorm_min=gen_params.cfg_renorm_min,
                cfg_renorm_type=gen_params.cfg_renorm_type,
                timestep_shift=gen_params.timestep_shift,
                cfg_text_packed_position_ids=cfg_text_latent["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=cfg_text_latent["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=cfg_text_latent["cfg_key_values_lens"],
                cfg_text_packed_key_value_indexes=cfg_text_latent["cfg_packed_key_value_indexes"],
                cfg_img_packed_position_ids=cfg_img_latent["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=cfg_img_latent["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=cfg_img_latent["cfg_key_values_lens"],
                cfg_img_packed_key_value_indexes=cfg_img_latent["cfg_packed_key_value_indexes"],
                **generation_input,
            )

        # Decode first sample
        img = self._decode_image_from_latent(self.bagel, self.vae, latents[0], image_shape)
        return DiffusionOutput(output=img)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Bagel checkpoints often include extra modules for "understanding" (e.g.
        # `connector.*`, vision towers, etc.). This vendored BagelPipeline is a
        # minimal *generation* pipeline, so those keys won't exist here.
        #
        # Some loaders are strict and will error on unexpected keys; filter them.
        state = self.state_dict()
        allowed = set(state.keys())
        shapes = {k: tuple(v.shape) for k, v in state.items()}

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
                        if tuple(tensor.shape) == shapes.get(cand):
                            picked = cand
                            break
                        else:
                            # Special-case: Bagel checkpoints may have different `max_latent_size`,
                            # which changes `latent_pos_embed.pos_embed` length (max_latent_size^2).
                            # If we detect that mismatch, resize the existing parameter *in-place*
                            # (preserving Parameter identity) so the loader can populate it.
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

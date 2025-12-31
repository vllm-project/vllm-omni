# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
BagelPipeline implementation for vLLM-Omni.
"""

from __future__ import annotations

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
from .bagel_transformer import Bagel, BagelConfig, NaiveCache, Qwen2MoTConfig, Qwen2MoTForCausalLM

logger = init_logger(__name__)


@dataclass
class BagelGenParams:
    num_timesteps: int = 50
    timestep_shift: float = 1.0


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
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

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

        self.language_model = Qwen2MoTForCausalLM(llm_config)
        ae_params: AutoEncoderParams = default_ae_params()
        self.vae = AutoEncoder(ae_params)

        self.bagel = Bagel(
            language_model=self.language_model,
            config=BagelConfig(
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
            num_timesteps=int(req.num_inference_steps or 50),
            timestep_shift=3.0,
        )

        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

        # Add text prompt (prefill) on gen context.
        # [Omni] Check for injected KV Cache from remote transfer
        injected_kv = getattr(req, "past_key_values", None)
        injected_metadata = getattr(req, "kv_metadata", None)

        if injected_kv is not None and injected_metadata is not None:
            logger.info("Using injected KV Cache from remote transfer")

            # [Fix] Reconstruct NaiveCache if injected_kv is a dict of tensors
            current_cache = gen_context["past_key_values"]
            if isinstance(current_cache, NaiveCache) and isinstance(injected_kv, dict):
                # injected_kv keys are like "0_k", "0_v", "1_k", ...
                for key_name, tensor in injected_kv.items():
                    try:
                        # Parse layer index and type
                        parts = key_name.split("_")
                        if len(parts) < 2:
                            continue

                        layer_idx = int(parts[0])
                        cache_type = parts[1]  # 'k' or 'v'

                        # Ensure tensor is on correct device
                        if tensor.device != self.device:
                            tensor = tensor.to(self.device)

                        if layer_idx in current_cache.key_cache:
                            if cache_type == "k":
                                current_cache.key_cache[layer_idx] = tensor
                            elif cache_type == "v":
                                current_cache.value_cache[layer_idx] = tensor
                            elif cache_type == "kv":
                                # Fallback if sender sent mixed/packed (less ideal)
                                current_cache.key_cache[layer_idx] = tensor
                                current_cache.value_cache[layer_idx] = tensor
                    except Exception as e:
                        logger.warning(f"Failed to load injected KV part {key_name}: {e}")

            if "kv_lens" in injected_metadata:
                val = injected_metadata["kv_lens"]
                if isinstance(val, (int, float)):
                    gen_context["kv_lens"] = [int(val)]
                else:
                    gen_context["kv_lens"] = list(val)

            if "ropes" in injected_metadata:
                val = injected_metadata["ropes"]
                if isinstance(val, (int, float)):
                    gen_context["ropes"] = [int(val)]
                else:
                    gen_context["ropes"] = list(val)

        else:
            # Standard local prefill path
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

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.seed)

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

        with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda", dtype=torch.bfloat16):
            latents = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                num_timesteps=gen_params.num_timesteps,
                timestep_shift=gen_params.timestep_shift,
                **generation_input,
            )

        # Decode first sample
        img = self._decode_image_from_latent(self.bagel, self.vae, latents[0], image_shape)
        return DiffusionOutput(output=img)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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

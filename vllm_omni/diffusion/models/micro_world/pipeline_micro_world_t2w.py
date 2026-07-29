# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
AMD Micro-World Text-to-World (T2W) pipeline.

Generates action-controlled video from a text prompt and keyboard/mouse actions.
Based on the Wan2.1-T2V-1.3B architecture with ControlNet-style action injection.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.platforms import current_omni_platform

from .micro_world_transformer import MicroWorldControlNetTransformer

logger = logging.getLogger(__name__)


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=model_path, filename=f"{subfolder}/config.json")
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(config: dict) -> MicroWorldControlNetTransformer:
    """Create MicroWorldControlNetTransformer from a HF config dict."""
    kwargs: dict[str, Any] = {}

    # Map HF config keys to vllm-omni WanTransformer3DModel constructor args
    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_heads" in config:
        # Micro-World uses 'num_heads' and 'dim' rather than
        # 'num_attention_heads' and 'attention_head_dim'
        kwargs["num_attention_heads"] = config["num_heads"]
        if "dim" in config:
            kwargs["attention_head_dim"] = config["dim"] // config["num_heads"]
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "in_dim" in config:
        kwargs["in_channels"] = config["in_dim"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_dim" in config:
        kwargs["out_channels"] = config["out_dim"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "text_dim" in config:
        kwargs["text_dim"] = config["text_dim"]
    if "freq_dim" in config:
        kwargs["freq_dim"] = config["freq_dim"]
    if "ffn_dim" in config:
        kwargs["ffn_dim"] = config["ffn_dim"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "cross_attn_norm" in config:
        kwargs["cross_attn_norm"] = config["cross_attn_norm"]
    if "eps" in config:
        kwargs["eps"] = config["eps"]
    if "image_dim" in config:
        kwargs["image_dim"] = config["image_dim"]
    if "added_kv_proj_dim" in config:
        kwargs["added_kv_proj_dim"] = config["added_kv_proj_dim"]
    if "rope_max_seq_len" in config:
        kwargs["rope_max_seq_len"] = config["rope_max_seq_len"]

    # Micro-World specific action params
    action_kwargs: dict[str, Any] = {}
    if "keyboard_dim" in config:
        action_kwargs["keyboard_dim"] = config["keyboard_dim"]
    if "mouse_dim" in config:
        action_kwargs["mouse_dim"] = config["mouse_dim"]
    if "action_dim" in config:
        action_kwargs["action_dim"] = config.get("action_dim", 1536)
    if "action_layers" in config and config["action_layers"] is not None:
        action_kwargs["action_layers"] = config["action_layers"]

    return MicroWorldControlNetTransformer(**action_kwargs, **kwargs)


def get_micro_world_t2w_post_process_func(od_config: OmniDiffusionConfig):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_micro_world_t2w_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process: validate action inputs from extra_params."""

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        # Actions come via extra_args; nothing to validate at this stage
        return request

    return pre_process_func


class MicroWorldT2WPipeline(nn.Module, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin):
    """AMD Micro-World Text-to-World pipeline.

    Generates action-controlled video from text prompts with keyboard/mouse inputs.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Weight sources
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Check for LoRA weights at model root — merged into transformer
        # weights at the end of load_weights().
        # Skip via MICRO_WORLD_SKIP_LORA=1 for debugging the LoRA-free baseline.
        self._lora_path: str | None = None
        lora_path = os.path.join(model, "lora_diffusion_pytorch_model.safetensors") if local_files_only else None
        if lora_path and os.path.exists(lora_path):
            if os.environ.get("MICRO_WORLD_SKIP_LORA") == "1":
                logger.info(f"Skipping LoRA weights at {lora_path} (MICRO_WORLD_SKIP_LORA=1)")
            else:
                logger.info(f"Found LoRA weights at {lora_path}")
                self._lora_path = lora_path
        self._lora_weight: float = 1.0

        # Text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)

        # VAE
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
        ).to(self.device)

        # Transformer
        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(transformer_config)

        # Store config for latent shape computation
        self.transformer_config = self.transformer.config
        # Reference re-pads T5 output to text_len with zeros before cross-attn.
        self._text_len = int(transformer_config.get("text_len", 512))

        # Scheduler
        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 3.0
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if hasattr(self.vae, "config") else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if hasattr(self.vae, "config") else 8

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def load_weights(self, weights):
        """Load weights using AutoWeightsLoader for vLLM integration.

        If a Kohya-style LoRA was found at pipeline init, merge it into the
        transformer weights once the base checkpoint has finished loading.
        """
        from vllm.model_executor.models.utils import AutoWeightsLoader

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights)
        if self._lora_path is not None:
            self._merge_lora(self._lora_path, self._lora_weight)
        return loaded

    def _merge_lora(self, lora_path: str, multiplier: float = 1.0) -> None:
        """Merge a Kohya-style LoRA checkpoint into the transformer weights.

        LoRA keys look like ``lora_unet__blocks_N_cross_attn_k.lora_up.weight``
        and encode ``transformer.blocks.N.attn2.to_k`` after applying the same
        Wan2.1 → diffusers remapping rules used by
        :meth:`MicroWorldControlNetTransformer.load_weights`. For each target
        Linear layer we apply ``W += multiplier * alpha * (up @ down)``.
        """
        from collections import defaultdict

        from safetensors.torch import load_file

        state_dict = load_file(lora_path)
        layers: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for key, value in state_dict.items():
            # Keys are either "<layer>.alpha" or "<layer>.lora_up.weight" / "<layer>.lora_down.weight".
            if ".lora_up." in key or ".lora_down." in key:
                layer, sub = key.split(".", 1)
                layers[layer][sub] = value
            elif key.endswith(".alpha"):
                layer = key[: -len(".alpha")]
                layers[layer]["alpha"] = value

        # Top-level name remapping (same as transformer.load_weights).
        _TOP_LEVEL_REMAP = [
            ("head_head", "proj_out"),
            ("time_embedding_0", "condition_embedder.time_embedder.linear_1"),
            ("time_embedding_2", "condition_embedder.time_embedder.linear_2"),
            ("time_projection_1", "condition_embedder.time_proj"),
            ("text_embedding_0", "condition_embedder.text_embedder.linear_1"),
            ("text_embedding_2", "condition_embedder.text_embedder.linear_2"),
        ]

        def _remap(lora_key: str) -> str | None:
            if not lora_key.startswith("lora_unet__"):
                return None
            name = lora_key[len("lora_unet__") :]
            # Top-level patterns first (they contain underscores we must
            # not turn into dots).
            for old, new in _TOP_LEVEL_REMAP:
                if name == old:
                    return new
            # Block-level patterns: blocks_N_<type_attn>_<qkvo|0|2>
            # Replace structural underscores with dots, keeping attention
            # subtype names like self_attn / cross_attn intact.
            import re

            m = re.match(r"^(blocks|action_blocks)_(\d+)_(.+)$", name)
            if not m:
                return None
            prefix, idx, rest = m.group(1), m.group(2), m.group(3)
            # rest is something like "cross_attn_k", "self_attn_o", "ffn_0", "ffn_2"
            rest_map = {
                "self_attn_q": "attn1.to_q",
                "self_attn_k": "attn1.to_k",
                "self_attn_v": "attn1.to_v",
                "self_attn_o": "attn1.to_out",
                "cross_attn_q": "attn2.to_q",
                "cross_attn_k": "attn2.to_k",
                "cross_attn_v": "attn2.to_v",
                "cross_attn_o": "attn2.to_out",
                "ffn_0": "ffn.net_0.proj",
                "ffn_2": "ffn.net_2",
            }
            sub = rest_map.get(rest)
            if sub is None:
                return None
            return f"{prefix}.{idx}.{sub}"

        # Self-attention q/k/v are FUSED into ``attn1.to_qkv`` (QKVParallelLinear).
        # Collect those LoRA deltas and apply them via the param's weight_loader
        # so each shard lands at the right offset of the fused weight.
        # Key: "blocks.N.attn1.to_qkv"  Value: {"q": (up, down, alpha), ...}
        from collections import defaultdict as _dd

        fused_qkv: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, float]]] = _dd(dict)
        _qkv_re = re.compile(r"^lora_unet__(blocks|action_blocks)_(\d+)_self_attn_(q|k|v)$")

        named_modules = dict(self.transformer.named_modules())
        merged = 0
        skipped = 0
        for lora_key, elems in layers.items():
            up = elems.get("lora_up.weight")
            down = elems.get("lora_down.weight")
            if up is None or down is None:
                skipped += 1
                continue
            alpha_t = elems.get("alpha")
            alpha = float(alpha_t.item()) / up.shape[1] if alpha_t is not None else 1.0

            # Defer fused-QKV updates so we can read-modify-write each shard.
            qkv_match = _qkv_re.match(lora_key)
            if qkv_match:
                prefix, idx, shard_id = qkv_match.group(1), qkv_match.group(2), qkv_match.group(3)
                fused_target = f"{prefix}.{idx}.attn1.to_qkv"
                fused_qkv[fused_target][shard_id] = (up, down, alpha)
                continue

            target = _remap(lora_key)
            if target is None or target not in named_modules:
                skipped += 1
                continue
            module = named_modules[target]
            if not hasattr(module, "weight") or module.weight is None:
                skipped += 1
                continue
            with torch.no_grad():
                w = module.weight
                up_t = up.to(device=w.device, dtype=w.dtype)
                down_t = down.to(device=w.device, dtype=w.dtype)
                delta = torch.mm(up_t, down_t)
                w.data.add_(multiplier * alpha * delta)
            merged += 1

        # Apply fused-QKV deltas via QKVParallelLinear.weight_loader(param, w, shard_id).
        # The loader writes ``w`` into the right slice of the fused weight, so we
        # read the existing slice first, add the LoRA delta, and write it back.
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        for fused_target, shards in fused_qkv.items():
            module = named_modules.get(fused_target)
            if module is None or not hasattr(module, "weight"):
                skipped += len(shards)
                continue
            qkv_param = module.weight
            head_size = getattr(module, "head_size", None)
            n_heads_q = getattr(module, "total_num_heads", None)
            n_heads_kv = getattr(module, "total_num_kv_heads", n_heads_q)
            if head_size is None or n_heads_q is None:
                skipped += len(shards)
                continue
            q_size = n_heads_q * head_size
            kv_size = n_heads_kv * head_size
            offsets = {
                "q": (0, q_size),
                "k": (q_size, q_size + kv_size),
                "v": (q_size + kv_size, q_size + 2 * kv_size),
            }
            weight_loader = getattr(qkv_param, "weight_loader", default_weight_loader)
            with torch.no_grad():
                for shard_id, (up, down, alpha) in shards.items():
                    start, end = offsets[shard_id]
                    up_t = up.to(device=qkv_param.device, dtype=qkv_param.dtype)
                    down_t = down.to(device=qkv_param.device, dtype=qkv_param.dtype)
                    delta = multiplier * alpha * torch.mm(up_t, down_t)
                    existing = qkv_param.data[start:end].clone()
                    weight_loader(qkv_param, existing + delta, shard_id)
                    merged += 1

        logger.info("LoRA merge: %d layers merged, %d skipped (of %d)", merged, skipped, len(layers))

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode text prompts using UMT5 text encoder."""
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        def _encode(text_list):
            inputs = self.tokenizer(
                text_list,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(device)
            mask = inputs.attention_mask.to(device)
            embeds = self.text_encoder(input_ids, attention_mask=mask)[0].to(dtype=dtype, device=device)
            # Trim to actual non-padding length per item — matches reference T5
            # behavior. With padding tokens left in, cross-attention spreads
            # mass over 500+ garbage tokens and dilutes prompt conditioning.
            seq_lens = mask.gt(0).sum(dim=1).long().tolist()
            if len(text_list) == 1:
                return embeds[:1, : seq_lens[0]]
            # Batch>1: trim to longest non-padding length so it stays one tensor.
            max_len = max(seq_lens) if seq_lens else 1
            return embeds[:, :max_len]

        prompt_embeds = _encode(prompt)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = _encode(negative_prompt)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare noise latents for diffusion."""
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def _parse_actions(
        self,
        extra_args: dict[str, Any] | None,
        device: torch.device,
        dtype: torch.dtype,
        num_action_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract action tensors from extra_args.

        If actions are missing and ``num_action_frames`` is given, returns
        zero-filled defaults (used for warmup / dummy run, where actions
        are not meaningful).
        """
        mouse_actions = extra_args.get("mouse_actions") if extra_args else None
        keyboard_actions = extra_args.get("keyboard_actions") if extra_args else None

        if mouse_actions is None or keyboard_actions is None:
            if num_action_frames is not None:
                # Warmup path: supply zero-filled no-op actions so the model
                # can still trace its forward pass.
                mouse_actions = torch.zeros((num_action_frames, 2), dtype=dtype, device=device)
                keyboard_actions = torch.zeros((num_action_frames, 7), dtype=dtype, device=device)
            else:
                raise ValueError("Both 'mouse_actions' and 'keyboard_actions' must be provided in extra_params.")

        # Convert to tensors if needed
        if not isinstance(mouse_actions, torch.Tensor):
            mouse_actions = torch.tensor(mouse_actions, dtype=dtype, device=device)
        else:
            mouse_actions = mouse_actions.to(dtype=dtype, device=device)

        if not isinstance(keyboard_actions, torch.Tensor):
            keyboard_actions = torch.tensor(keyboard_actions, dtype=dtype, device=device)
        else:
            keyboard_actions = keyboard_actions.to(dtype=dtype, device=device)

        # Ensure batch dimension
        if mouse_actions.ndim == 2:
            mouse_actions = mouse_actions.unsqueeze(0)
        if keyboard_actions.ndim == 2:
            keyboard_actions = keyboard_actions.unsqueeze(0)

        return mouse_actions, keyboard_actions

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        mouse_actions: torch.Tensor | None = None,
        keyboard_actions: torch.Tensor | None = None,
        action_context_scale: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through transformer to predict noise."""
        if current_model is None:
            current_model = self.transformer
        return current_model(
            mouse_actions=mouse_actions,
            keyboard_actions=keyboard_actions,
            action_context_scale=action_context_scale,
            **kwargs,
        )[0]

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        frame_num: int = 49,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Extract prompt from request
        if len(req.prompts) > 1:
            raise ValueError("This model only supports a single prompt.")
        if len(req.prompts) == 1:
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        self._guidance_scale = guidance_scale

        # Ensure dimensions are compatible
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self.transformer.dtype

        # Generator
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        # Parse actions (num_action_frames = temporal_ratio * latent_frames + 1)
        extra_args = req.sampling_params.extra_args if hasattr(req.sampling_params, "extra_args") else None
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        num_action_frames = num_latent_frames * self.vae_scale_factor_temporal + 1
        mouse_actions, keyboard_actions = self._parse_actions(
            extra_args, device, dtype, num_action_frames=num_action_frames
        )

        # Encode prompt
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                device=device,
                dtype=dtype,
            )

        # Re-pad to text_len with zeros so cross-attention sees the same K/V
        # count as the reference (which projects the padded zeros and attends
        # over all 512 positions without a key mask). Without this, every
        # transformer block diverges from reference starting at attn2.
        import torch.nn.functional as _F

        if prompt_embeds.shape[1] < self._text_len:
            prompt_embeds = _F.pad(prompt_embeds, (0, 0, 0, self._text_len - prompt_embeds.shape[1]))
        if negative_prompt_embeds is not None and negative_prompt_embeds.shape[1] < self._text_len:
            negative_prompt_embeds = _F.pad(
                negative_prompt_embeds, (0, 0, 0, self._text_len - negative_prompt_embeds.shape[1])
            )

        # ---- DEBUG DUMPS ----
        _dump_dir = os.environ.get("MICRO_WORLD_DUMP_DIR")
        if _dump_dir:
            from pathlib import Path

            _dd = Path(_dump_dir)
            _dd.mkdir(parents=True, exist_ok=True)
            torch.save(prompt_embeds.detach().cpu(), _dd / "prompt_embeds.pt")
            if negative_prompt_embeds is not None:
                torch.save(negative_prompt_embeds.detach().cpu(), _dd / "neg_prompt_embeds.pt")
            torch.save(mouse_actions.detach().cpu(), _dd / "mouse_actions.pt")
            torch.save(keyboard_actions.detach().cpu(), _dd / "keyboard_actions.pt")
            logger.info(
                "DUMP: prompt_embeds shape=%s, neg=%s",
                tuple(prompt_embeds.shape),
                tuple(negative_prompt_embeds.shape) if negative_prompt_embeds is not None else None,
            )

        # Prepare latents
        num_channels_latents = self.transformer_config.in_channels
        latents = self.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        if _dump_dir:
            torch.save(latents.detach().cpu(), _dd / "initial_latents.pt")
            logger.info(
                "DUMP: initial_latents shape=%s mean=%.4f std=%.4f",
                tuple(latents.shape),
                float(latents.float().mean()),
                float(latents.float().std()),
            )

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        # Denoising loop
        with self.progress_bar(total=len(timesteps)) as pbar:
            for _step_idx, t in enumerate(timesteps):
                if _dump_dir and (_step_idx % 5 == 0 or _step_idx == len(timesteps) - 1):
                    torch.save(latents.detach().cpu(), _dd / f"latent_step_{_step_idx:02d}.pt")
                self._current_timestep = t
                latent_model_input = latents.to(dtype)
                timestep = t.expand(latents.shape[0])

                # Duplicate actions for CFG
                if do_cfg:
                    mouse_actions_cfg = torch.cat([mouse_actions] * 2)
                    keyboard_actions_cfg = torch.cat([keyboard_actions] * 2)
                else:
                    mouse_actions_cfg = mouse_actions
                    keyboard_actions_cfg = keyboard_actions

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "return_dict": False,
                    "current_model": self.transformer,
                    "mouse_actions": mouse_actions_cfg if not do_cfg else mouse_actions,
                    "keyboard_actions": keyboard_actions_cfg if not do_cfg else keyboard_actions,
                }
                if do_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "return_dict": False,
                        "current_model": self.transformer,
                        "mouse_actions": mouse_actions,
                        "keyboard_actions": keyboard_actions,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_cfg)
                pbar.update()

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        # Decode
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]
            if _dump_dir:
                torch.save(output.detach().cpu(), _dd / "frames.pt")
                logger.info("DUMP: frames shape=%s", tuple(output.shape))

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

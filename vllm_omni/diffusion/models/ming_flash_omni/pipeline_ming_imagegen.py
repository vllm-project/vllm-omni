# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming-flash-omni-2.0 imagegen (text-to-image / img2img) diffusion pipeline.

Cross-stage data flow:

    Stage 0 (thinker, llm)           Stage 1 (imagegen, diffusion)
    ────────────────────             ──────────────────────────────
    forward returns                  thinker2imagegen hook slices
    multimodal_output[               final_hidden_states at
      "final_hidden_states"]         <imagePatch> positions,
       ↓                             returns list[dict] with
    shared_memory_connector          {"extra": {"thinker_hidden_states"}}
       ↓                             ──── via OmniMsgpackEncoder ────>
                                     MingImagePipeline.forward(req):
                                       hidden = req.prompts[0]["extra"][...]
                                       cond = condition_encoder(hidden)
                                       img = ZImagePipeline-style loop
                                       return DiffusionOutput(output=img)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_ref_latent
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.ming_flash_omni.byte5_encoder import (
    MingByT5Encoder,
)
from vllm_omni.diffusion.models.ming_flash_omni.condition_encoder import (
    MingConditionEncoder,
)
from vllm_omni.diffusion.models.ming_flash_omni.ming_zimage_transformer import (
    MingZImageTransformer2DModel,
)
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from vllm_omni.transformers_utils.configs.ming_flash_omni import MingImageGenConfig

logger = logging.getLogger(__name__)


@dataclass
class _ZPipelineSamplingParams:
    """Typed shim satisfying the attributes ZImagePipeline.forward reads
    from ``req.sampling_params``.  Unlike SimpleNamespace this fails loudly
    at construction if a required field is missing."""

    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    generator: torch.Generator | None = None
    strength: float | None = None
    sigmas: list[float] | None = None
    max_sequence_length: int = 512
    guidance_rescale: float | None = None
    num_outputs_per_prompt: int = 1


@dataclass
class _ZPipelineRequest:
    """Typed shim satisfying the attributes ZImagePipeline.forward reads
    from ``req``."""

    request_id: str
    sampling_params: _ZPipelineSamplingParams
    prompts: list[dict[str, Any]] = field(default_factory=lambda: [{"prompt": "", "negative_prompt": ""}])


class MingImagePipeline(ZImagePipeline):
    """Ming-flash-omni-2.0 text-to-image diffusion pipeline.

    Ming-specific components added on top of the inherited contract:
      * ``condition_encoder`` — Qwen2 connector + proj_in/out + F.normalize×1000
      * ``byte5``             — Optional ByT5 glyph encoder (loaded if checkpoint
                                ships ``byt5/``)
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",  # noqa: ARG002
    ) -> None:
        # Skip ZImagePipeline.__init__ (it would eagerly load the Z-Image text
        # encoder/tokenizer that Ming replaces with its own condition_encoder).
        nn.Module.__init__(self)

        model_path = od_config.model
        if not os.path.exists(model_path):
            model_path = download_weights_from_hf_specific(model_path, od_config.revision, ["*"])

        dtype = getattr(od_config, "dtype", torch.bfloat16)
        local_files_only = os.path.exists(model_path)

        self.od_config = od_config
        self._execution_device = get_local_device()
        self.device = self._execution_device  # Ming convention alias
        self._dtype = dtype

        # Ming's per-checkpoint image-gen configuration. We cannot rely on
        # ``od_config.hf_config.image_gen_config`` because the diffusion
        # stage is started with ``hf_config_name: thinker_config`` (the
        # BailingMM2Config), which does not carry a MingImageGenConfig.
        # Fall back to defaults that match the released checkpoint.
        self.image_gen_config = MingImageGenConfig()
        logger.info(
            "[MingImagePipeline] init: model=%s dtype=%s image_gen_config=%s",
            model_path,
            dtype,
            self.image_gen_config,
        )

        # ----- weights_sources: DiT transformer + VAE.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=self.image_gen_config.transformer_subfolder,
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=self.image_gen_config.vae_subfolder,
                revision=od_config.revision,
                prefix="vae.",
            ),
        ]

        prefetch_subfolders(
            model_path,
            [self.image_gen_config.scheduler_subfolder, self.image_gen_config.vae_subfolder],
            local_files_only=local_files_only,
        )

        # ----- Scheduler: load config-only from disk + Ming-specific override.
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder=self.image_gen_config.scheduler_subfolder,
            local_files_only=local_files_only,
        )
        # Ming forces use_dynamic_shifting=True at runtime regardless of what
        # the checkpoint scheduler_config.json ships.
        self.scheduler.config["use_dynamic_shifting"] = True
        logger.info(
            "[MingImagePipeline] scheduler: %s (use_dynamic_shifting=True)",
            type(self.scheduler).__name__,
        )

        # ----- VAE: DistributedAutoencoderKL.
        vae_config = DistributedAutoencoderKL.load_config(
            model_path, subfolder=self.image_gen_config.vae_subfolder, local_files_only=local_files_only
        )
        self.vae = DistributedAutoencoderKL.from_config(vae_config).to(self._execution_device, dtype=dtype)
        self.vae.eval()

        # ----- DiT transformer.
        self.transformer = MingZImageTransformer2DModel(quant_config=None)

        # Ming brings its own conditioning path — no Z-Image text_encoder /
        # tokenizer.
        self.text_encoder = None
        self.tokenizer = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=getattr(od_config, "enable_diffusion_pipeline_profiler", False)
        )

        # ----- Condition encoder (Qwen2 connector + proj_in/out + norm×1000).
        self.condition_encoder = MingConditionEncoder(
            self.image_gen_config,
            thinker_hidden_size=self.image_gen_config.thinker_hidden_size,
            device=self.device,
            dtype=dtype,
        )
        self.condition_encoder.load_from_checkpoint(model_path)

        # Optional ByT5 glyph/text encoder. Only loaded when the checkpoint
        # ships a byt5/ subfolder; otherwise byte5_text requests are ignored
        byte5_dir = Path(model_path) / "byt5"
        if byte5_dir.exists():
            self.byte5 = MingByT5Encoder.from_checkpoint(byte5_dir, device=self.device, dtype=dtype)
        else:
            self.byte5 = None
            logger.info("[MingImagePipeline] no byt5/ subfolder at %s; ByT5 enhancement disabled", byte5_dir)

        logger.info("[MingImagePipeline] ready — vae_scale_factor=%d", self.vae_scale_factor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_byte5_texts(extra: dict, sampling_params) -> list[str]:
        """Resolve byte5 glyph texts.

        Two sources, in order of priority:
        1. `extra["byte5_text"]`: auto-extracted from the user prompt's quoted spans
            by thinker2imagegen (already wrapped as `'Text "<glyph>". '`
            by Ming's get_text_from_prompt).
        2. `sampling_params.extra_args["byte5_text"]`:
            raw strings without the `Text "..."` wrapper are auto-wrapped here
            to match the distribution ByT5 was trained on.
        """
        # Source 1: auto-extracted, already wrapped. Return as-is if non-empty.
        raw = extra.get("byte5_text")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            cleaned = [t for t in raw if isinstance(t, str) and t.strip()]
            if cleaned:
                return cleaned

        # Source 2: explicit override — wrap raw strings so the byte5 encoder
        # sees the same ``Text "<glyph>". `` format Ming used during training.
        raw = (getattr(sampling_params, "extra_args", None) or {}).get("byte5_text")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            out: list[str] = []
            for t in raw:
                if not isinstance(t, str):
                    continue
                s = t.strip()
                if not s:
                    continue
                # Don't double-wrap if the caller already supplied ``Text "...". ``.
                out.append(s if s.startswith('Text "') else f'Text "{s}". ')
            if out:
                return out
        return []

    @torch.inference_mode()
    def _encode_reference_image(self, ref, height: int, width: int) -> torch.Tensor | None:
        """Turn a PIL/tensor reference image into a VAE latent for ``ref_x``.

        Applies the same shift/scale Ming uses (``(z - shift_factor) * scaling_factor``)
        so the concatenated frame lives in the DiT's latent space.
        """
        if ref is None:
            return None
        if not isinstance(ref, torch.Tensor):
            ref = self.image_processor.preprocess(ref, height, width)
        ref = ref.to(device=self.device, dtype=self.vae.dtype)
        latent = self.vae.encode(ref).latent_dist.mode()
        return (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """Run one text-to-image generation request.

        Args:
            req: Diffusion request. The cross-stage thinker hidden states
                must be present at
                ``req.prompts[0]["extra"]["thinker_hidden_states"]`` as a
                ``[N, H]`` (or ``[1, N, H]``) tensor, placed there by
                ``thinker2imagegen``.

        Returns:
            DiffusionOutput with ``.output`` set to a ``[B, 3, H, W]``
            image tensor in ``[-1, 1]``. The vllm-omni diffusion engine's
            output adapter converts this to PIL/base64 downstream.
        """
        first_prompt = req.prompts[0] if req.prompts else None
        if isinstance(first_prompt, str):
            prompt_dict: dict[str, Any] = {}
        elif isinstance(first_prompt, dict):
            prompt_dict = first_prompt
        elif first_prompt is not None and hasattr(first_prompt, "_asdict"):
            prompt_dict = first_prompt._asdict()
        elif first_prompt is not None and hasattr(first_prompt, "__dict__"):
            prompt_dict = vars(first_prompt)
        else:
            prompt_dict = {}

        extra = prompt_dict.get("extra") or {}
        hidden = extra.get("thinker_hidden_states")
        if hidden is None:
            # Same dual-path convention as glm_image: also check
            # ``sampling_params.extra_args``.
            hidden = (req.sampling_params.extra_args or {}).get("thinker_hidden_states")
        if hidden is None:
            scale = self.image_gen_config.img_gen_scales[-1]
            num_query_tokens = scale * scale
            hidden = torch.zeros(
                (num_query_tokens, self.image_gen_config.thinker_hidden_size),
                dtype=self._dtype,
                device=self.device,
            )
            logger.warning(
                "[MingImagePipeline.forward] 'thinker_hidden_states' missing "
                "from request; falling back to zero-conditioning %s. This is "
                "expected during warmup; for real requests verify that "
                "`custom_process_input_func: thinker2imagegen` is set on the "
                "diffusion stage in the YAML.",
                tuple(hidden.shape),
            )

        if not isinstance(hidden, torch.Tensor):
            raise TypeError(
                f"[MingImagePipeline] 'thinker_hidden_states' must be a Tensor, got {type(hidden).__name__}"
            )

        # Move to the pipeline's device+dtype.
        target_device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype
        hidden = hidden.to(device=target_device, dtype=target_dtype)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)  # [N, H] -> [1, N, H]
        logger.debug(
            "[MingImagePipeline.forward] thinker_hidden_states=%s on %s (%s)",
            tuple(hidden.shape),
            target_device,
            target_dtype,
        )

        # ----- Condition encoder → cap_feats
        cap_feats = self.condition_encoder(hidden)
        logger.debug("[MingImagePipeline.forward] cap_feats=%s", tuple(cap_feats.shape))

        # Real negative CFG conditioning (opt-in). See expand_cfg_prompts.
        negative_hidden = extra.get("negative_thinker_hidden_states")
        negative_cap_feats = None
        if isinstance(negative_hidden, torch.Tensor):
            negative_hidden = negative_hidden.to(device=target_device, dtype=target_dtype)
            if negative_hidden.dim() == 2:
                negative_hidden = negative_hidden.unsqueeze(0)
            negative_cap_feats = self.condition_encoder(negative_hidden)
            logger.debug("[MingImagePipeline.forward] negative_cap_feats=%s", tuple(negative_cap_feats.shape))

        # ByT5 text enhancement (opt-in). Appends glyph-aware features along
        # the sequence dim; negative side gets zeros for the byte5 portion so
        # CFG doesn't push away from the rendered text.
        byte5_texts = self._resolve_byte5_texts(extra, req.sampling_params)
        if byte5_texts and self.byte5 is not None:
            byte5_feats = self.byte5(byte5_texts).to(device=target_device, dtype=target_dtype)
            cap_feats = torch.cat((cap_feats, byte5_feats), dim=1)
            if negative_cap_feats is not None:
                negative_cap_feats = torch.cat((negative_cap_feats, torch.zeros_like(byte5_feats)), dim=1)
            logger.debug("[MingImagePipeline.forward] byte5 cat'd: cap_feats=%s", tuple(cap_feats.shape))

        # Sampling knobs, in priority order:
        #   top-level extra_args[key] > sampling_params.* attr >
        #   MingImageGenConfig default. Knobs live flat on extra_args
        sp = req.sampling_params
        cfg = self.image_gen_config
        ea = sp.extra_args or {}
        resolved: dict[str, Any] = {}
        for ea_key, sp_attr, default in (
            ("height", "height", cfg.default_height),
            ("width", "width", cfg.default_width),
            ("steps", "num_inference_steps", cfg.num_inference_steps),
            ("cfg", "guidance_scale", cfg.guidance_scale),
            ("seed", "seed", None),
        ):
            for v in (ea.get(ea_key), getattr(sp, sp_attr), default):
                if v is not None:
                    resolved[ea_key] = v
                    break

        height = int(resolved["height"])
        width = int(resolved["width"])
        num_inference_steps = int(resolved["steps"])
        guidance_scale = float(resolved["cfg"])
        seed = resolved.get("seed")

        # Always rebuild the generator from the resolved seed. Reusing
        # ``sp.generator`` causes two problems:
        #   (1) if the caller pre-seeded it with sp.seed (e.g. the top-level
        #       ``seed`` key on OmniDiffusionSamplingParams), any override via
        #       `extra_args["seed"]` would be silently ignored; and
        #   (2) a persistent generator instance accumulates state across
        #       requests → same-seed replays produce different outputs.
        if seed is not None:
            generator = torch.Generator(device=target_device).manual_seed(int(seed))
        else:
            generator = sp.generator

        # Format prompt_embeds / negative_prompt_embeds as list[Tensor]
        # (one entry per request) — matches ZImagePipeline's contract when
        # prompt_embeds are pre-computed.
        prompt_embeds = [cap_feats[i] for i in range(cap_feats.shape[0])]
        if negative_cap_feats is not None:
            negative_prompt_embeds = [negative_cap_feats[i] for i in range(negative_cap_feats.shape[0])]
        else:
            negative_prompt_embeds = [self.condition_encoder.zero_negative(e) for e in prompt_embeds]

        z_sp = _ZPipelineSamplingParams(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        z_req = _ZPipelineRequest(
            request_id=req.request_id or "ming-imagegen",
            sampling_params=z_sp,
        )

        # Reference image (img2img) → VAE-encoded latent published on the
        # active ForwardContext so MingZImageTransformer2DModel can read it
        # from request scope inside its forward().
        ref_latent = self._encode_reference_image(extra.get("reference_image"), height, width)
        set_forward_context_ref_latent(ref_latent)

        logger.debug(
            "[MingImagePipeline.forward] running z_pipeline hw=(%d,%d) steps=%d cfg=%.2f seed=%s overrides=%s ref=%s",
            height,
            width,
            num_inference_steps,
            guidance_scale,
            seed,
            ea,
            None if ref_latent is None else tuple(ref_latent.shape),
        )
        try:
            output = super().forward(
                z_req,
                prompt=None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                generator=generator,
                output_type="pt",
                return_dict=True,
            )
        finally:
            set_forward_context_ref_latent(None)

        if hasattr(output, "output") and output.output is not None:
            raw = output.output
        elif hasattr(output, "images"):
            raw = output.images
        else:
            raw = output
        if not isinstance(raw, torch.Tensor):
            raise RuntimeError(f"ZImagePipeline returned non-tensor output: {type(raw).__name__}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[MingImagePipeline.forward] produced image tensor shape=%s range=[%.3f,%.3f]",
                tuple(raw.shape),
                raw.float().min().item(),
                raw.float().max().item(),
            )
        return DiffusionOutput(output=raw)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def get_ming_image_post_process_func(od_config: OmniDiffusionConfig):
    """Return a post-process callable that converts the raw VAE tensor to PIL.

    The diffusion engine calls ``post_process_func(output_data)`` where
    ``output_data`` is the ``DiffusionOutput.output`` tensor returned by
    ``MingImagePipeline.forward``. It has shape ``[B, 3, H, W]`` in ``[-1, 1]``
    (Z-image VAE convention). We run the standard ``VaeImageProcessor``
    postprocess to convert it to ``list[PIL.Image]`` which vllm-omni's
    ``OmniRequestOutput.from_diffusion`` then bubbles up as
    ``omni_outputs.images`` for serving_chat to base64-encode.

    Registered via ``_DIFFUSION_POST_PROCESS_FUNCS["MingImagePipeline"]``
    in vllm_omni/diffusion/registry.py.
    """
    import json

    model_path = od_config.model
    vae_config_path = os.path.join(model_path, "vae", "config.json")
    try:
        with open(vae_config_path) as f:
            vae_cfg = json.load(f)
        block_out_channels = vae_cfg.get("block_out_channels", [128, 256, 512, 512])
        vae_scale_factor = 2 ** (len(block_out_channels) - 1)
    except Exception:
        vae_scale_factor = 8  # Ming's Flux-format VAE default

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(images: torch.Tensor):
        # VaeImageProcessor.postprocess with default output_type="pil"
        # returns ``list[PIL.Image]``.
        return image_processor.postprocess(images.float())

    return post_process_func


__all__ = [
    "MingImagePipeline",
    "get_ming_image_post_process_func",
]

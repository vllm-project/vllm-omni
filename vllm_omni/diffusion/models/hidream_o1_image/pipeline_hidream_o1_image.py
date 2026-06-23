# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiDream-O1-Image pipeline -- Phase 1 scope (see the project roadmap):
text-to-image only, single GPU, no reference-image conditioning, no
parallelism/Cache-DiT. Reference images, CFG-Parallel, TP/SP, and
Cache-DiT are explicitly out of scope here and land in later phases.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hidream_o1_image.qwen3_vl_uit_transformer import HiDreamO1UiTModel
from vllm_omni.diffusion.models.hidream_o1_image.utils_hidream_o1 import (
    PATCH_SIZE,
    build_packed_attention_metadata,
    depatchify,
    find_closest_resolution,
    get_rope_index_fix_point,
    patchify,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

# Special tokens used to build the packed sequence. Resolved against the
# checkpoint's own tokenizer at pipeline init time (see __init__) -- these
# string literals are tokenizer *vocabulary entries*, not ids, so they are
# stable across the HiDream-O1-Image checkpoint family even though the
# numeric ids they resolve to are not (do not hardcode ids).
_BOI_TOKEN = "<|boi_token|>"
_TMS_TOKEN = "<|tms_token|>"

DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_SHIFT = 3.0
# The initial noise tensor is scaled by this factor before patchifying --
# the model's flow-matching sigma schedule was trained assuming this
# noise variance at t~999, not unit variance. Confirmed empirically:
# omitting this causes the denoised latent to monotonically collapse
# toward zero (flat gray output) regardless of step count.
DEFAULT_NOISE_SCALE = 8.0
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024


def get_hidream_o1_image_post_process_func(od_config: OmniDiffusionConfig):
    """Converts the pipeline's raw ``[-1, 1]`` pixel tensor output to PIL images.

    No VAE is involved, so there is no ``VaeImageProcessor``/scale-factor
    concept here -- this is a plain denormalize + clamp + to-uint8 step.
    """
    del od_config

    def post_process_func(images: torch.Tensor) -> list[Image.Image]:
        images = ((images.float() + 1) / 2).clamp(0, 1)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        images = np.round(images * 255).astype(np.uint8)
        return [Image.fromarray(img).convert("RGB") for img in images]

    return post_process_func


class HiDreamO1ImagePipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()
        self.dtype = od_config.dtype

        model_path = od_config.model
        if model_path is None:
            raise ValueError("HiDreamO1ImagePipeline requires od_config.model.")
        local_files_only = os.path.exists(model_path)

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

        tms_token_id = self.tokenizer.convert_tokens_to_ids(_TMS_TOKEN)
        if tms_token_id is None or tms_token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Could not resolve {_TMS_TOKEN!r} to a valid token id in this checkpoint's tokenizer -- "
                "refusing to fall back to a hardcoded id."
            )

        self.transformer = HiDreamO1UiTModel(od_config=od_config, tms_token_id=tms_token_id)

        # weights_sources triggers vLLM-Omni's standard checkpoint discovery
        # (download + safetensors iteration); load_weights() below does the
        # actual name remap (checkpoint keys are `model.*`/`lm_head.*`, this
        # pipeline's module tree is `transformer.*` with no lm_head).
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=False,
            )
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        skipped_lm_head = 0
        for name, tensor in weights:
            if name.startswith("lm_head."):
                # Text-generation head from the upstream Qwen3VLForConditionalGeneration
                # checkpoint -- this pipeline never decodes text, so it's never
                # instantiated and these weights are intentionally dropped.
                skipped_lm_head += 1
                continue
            if not name.startswith("model."):
                logger.warning("Skipping unexpected HiDream-O1-Image weight key %s", name)
                continue
            local_name = "transformer." + name[len("model.") :]
            if local_name not in params_dict:
                logger.warning("Skipping HiDream-O1-Image weight %s -- no matching local parameter", local_name)
                continue
            param = params_dict[local_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(local_name)
        logger.info(
            "HiDreamO1ImagePipeline loaded %d/%d local parameters (%d lm_head.* keys dropped)",
            len(loaded),
            len(params_dict),
            skipped_lm_head,
        )
        return loaded

    # ------------------------------------------------------------------
    # Packed-sequence construction (text-to-image only for Phase 1)
    # ------------------------------------------------------------------

    def _build_t2i_sample(self, prompt: str, height: int, width: int) -> dict[str, Any]:
        """Build one branch's (cond or uncond) packed sequence: text tokens
        followed by a target-image patch span. Mirrors the reference repo's
        ``build_t2i_text_sample`` (no VAE, no ref images for Phase 1).
        """
        config = self.transformer.config
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id

        h_patches = height // PATCH_SIZE
        w_patches = width // PATCH_SIZE
        image_len = h_patches * w_patches

        messages = [{"role": "user", "content": prompt}]
        template = (
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + _BOI_TOKEN
            + _TMS_TOKEN
        )
        input_ids = self.tokenizer.encode(template, return_tensors="pt", add_special_tokens=False).to(self.device)

        image_grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.int64, device=self.device)

        vision_tokens = torch.full((1, image_len), image_token_id, dtype=input_ids.dtype, device=self.device)
        vision_tokens[0, 0] = vision_start_token_id
        input_ids_padded = torch.cat([input_ids, vision_tokens], dim=-1)

        position_ids, _ = get_rope_index_fix_point(
            spatial_merge_size=1,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            input_ids=input_ids_padded,
            image_grid_thw=image_grid_thw,
            skip_vision_start_token=[1],
        )

        txt_seq_len = input_ids.shape[-1]
        total_seq_len = position_ids.shape[-1]

        # Raw token types distinguish the timestep placeholder (3) from
        # actual target-image patches (1) so vinput_mask can select only
        # the image-patch outputs; the binarized version (>0) is what
        # drives attention masking (both are "gen", attending bidirectionally).
        token_types_raw = torch.zeros((1, total_seq_len), dtype=torch.long, device=self.device)
        bgn = txt_seq_len - 1
        token_types_raw[0, bgn : bgn + image_len + 1] = 1
        token_types_raw[0, txt_seq_len - 1 : txt_seq_len] = 3
        vinput_mask = token_types_raw == 1
        token_types = (token_types_raw > 0).long()

        dense_mask, full_attn_spans = build_packed_attention_metadata(token_types)
        attn_metadata = AttentionMetadata(attn_mask=dense_mask, full_attn_spans=full_attn_spans)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_types": token_types,
            "attn_metadata": attn_metadata,
            "vinput_mask": vinput_mask,
        }

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def _build_scheduler(self, scheduler_name: str, num_inference_steps: int, shift: float):
        if scheduler_name == "flow_match":
            sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
        else:
            sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
        sched.set_timesteps(num_inference_steps, device=self.device)
        return sched

    def predict_noise(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_types: torch.Tensor,
        attn_metadata: AttentionMetadata,
        vinput_mask: torch.Tensor,
        z: torch.Tensor,
        t_pixeldit: torch.Tensor,
        sigma: torch.Tensor,
        **_unused: Any,
    ) -> torch.Tensor:
        out = self.transformer.forward_generation(
            input_ids=input_ids,
            position_ids=position_ids,
            vinputs=z,
            timestep=t_pixeldit.reshape(-1),
            token_types=token_types,
            attn_metadata=attn_metadata,
        )
        # Batch size is 1 throughout Phase 1 (see roadmap Phase 0 design
        # lock re: piecewise_attn's batch-homogeneity constraint).
        x_pred = out.x_pred[:, vinput_mask[0]]
        return (x_pred.float() - z.float()) / sigma

    def diffuse(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        shift: float,
        seed: int,
        scheduler_name: str = "default",
        noise_scale: float = DEFAULT_NOISE_SCALE,
    ) -> torch.Tensor:
        do_true_cfg = guidance_scale > 1.0

        cond_sample = self._build_t2i_sample(prompt, height, width)
        negative_sample = self._build_t2i_sample(" ", height, width) if do_true_cfg else None

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = noise_scale * torch.randn((1, 3, height, width), generator=generator).to(self.device, self.dtype)
        z = patchify(noise)

        scheduler = self._build_scheduler(scheduler_name, num_inference_steps, shift)
        self.scheduler = scheduler  # scheduler_step() reads self.scheduler -- must be set before the loop

        with self.progress_bar(total=len(scheduler.timesteps)) as pbar:
            for step_t in scheduler.timesteps:
                t_pixeldit = 1.0 - step_t.float() / 1000.0
                sigma = (step_t.float() / 1000.0).clamp_min(1e-3)

                positive_kwargs = {**cond_sample, "z": z, "t_pixeldit": t_pixeldit, "sigma": sigma}
                negative_kwargs = (
                    {**negative_sample, "z": z, "t_pixeldit": t_pixeldit, "sigma": sigma} if do_true_cfg else None
                )

                v_guided = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,  # HiDream's CFG combine is a plain linear blend, no norm-rescale
                )
                model_output = -v_guided

                z = self.scheduler_step_maybe_with_cfg(
                    model_output.float(), step_t.float(), z.float(), do_true_cfg=do_true_cfg
                ).to(self.dtype)
                pbar.update()

        return z

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def forward(self, req: OmniDiffusionRequest, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else first_prompt.get("prompt", "")

        sp = req.sampling_params
        width, height = find_closest_resolution(sp.width or DEFAULT_WIDTH, sp.height or DEFAULT_HEIGHT)
        guidance_scale = sp.guidance_scale if sp.guidance_scale_provided else DEFAULT_GUIDANCE_SCALE
        num_inference_steps = sp.num_inference_steps or DEFAULT_NUM_INFERENCE_STEPS
        shift = sp.extra_args.get("shift", DEFAULT_SHIFT)
        scheduler_name = sp.extra_args.get("scheduler_name", "default")
        noise_scale = sp.extra_args.get("noise_scale", DEFAULT_NOISE_SCALE)
        seed = sp.seed if sp.seed is not None else 42

        latents = self.diffuse(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            seed=seed,
            scheduler_name=scheduler_name,
            noise_scale=noise_scale,
        )

        h_patches, w_patches = height // PATCH_SIZE, width // PATCH_SIZE
        images = depatchify(latents, h_patches, w_patches)
        return DiffusionOutput(output=images)

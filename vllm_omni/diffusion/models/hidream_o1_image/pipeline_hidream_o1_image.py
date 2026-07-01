# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiDream-O1-Image pipeline — Phase 2 (text-to-image + image editing +
multi-reference personalization, single GPU, no Cache-DiT / TP / SP yet).

Phase 1 added text-to-image. Phase 2 extends it with reference-image
conditioning via HiDream-O1's dual-pathway: semantic conditioning through the
Qwen3-VL vision tower (with DeepStack intermediate feature injection) and
pixel-level conditioning through raw pixel patches appended to the noise
latent.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, ClassVar

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
    adaptive_ref_max_size,
    build_packed_attention_metadata,
    calculate_dimensions,
    depatchify,
    find_closest_resolution,
    get_rope_index_fix_point,
    patchify,
    preprocess_ref_patches,
    resize_pilimage,
)
from vllm_omni.diffusion.models.interface import SupportImageInput
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


class HiDreamO1ImagePipeline(nn.Module, CFGParallelMixin, ProgressBarMixin, SupportImageInput):
    # SupportImageInput protocol -- enables reference-image conditioning for
    # editing (1 ref) and subject-driven personalization (N refs).
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"

    # Max side-length (pixels) for ref images fed to the VLM vision tower.
    # The vision tower encodes semantic conditioning; it doesn't need full
    # resolution, so we cap it to avoid excessive token counts there.
    _VLM_COND_IMG_SIZE: ClassVar[int] = 448

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

    def _build_edit_sample(
        self, prompt: str, ref_pils: list[Image.Image], height: int, width: int
    ) -> dict[str, Any]:
        """Build one branch's packed sequence for editing / personalization.

        Ref images condition the model through two independent paths:
        1. **VLM path** — each ref is processed through the Qwen3-VL processor
           (at reduced ``_VLM_COND_IMG_SIZE`` resolution), producing:
           - ``pixel_values`` for the vision tower (semantic embedding +
             DeepStack intermediate features, computed once and cached).
           - Vision-token placeholders in ``proc.input_ids`` (token_type=0,
             causal — the vision tower handles their embedding via
             ``masked_scatter`` in ``forward_generation``).
        2. **Pixel-patch path** — each ref is patchified (at adaptive full
           resolution) and concatenated after the target-noise patches as
           ``ref_patches`` (token_type=2, bidirectional gen tokens).

        Returns a dict compatible with ``predict_noise``'s kwargs, extended
        with ``pixel_values``, ``image_grid_thw_vlm``, ``ref_patches``, and
        ``ref_image_lens``.
        """
        K = len(ref_pils)
        config = self.transformer.config
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id

        h_patches = height // PATCH_SIZE
        w_patches = width // PATCH_SIZE
        tgt_image_len = h_patches * w_patches

        # --- VLM path: resize refs for the vision tower ---
        ref_pils_vlm: list[Image.Image] = []
        for pil in ref_pils:
            vw, vh = calculate_dimensions(self._VLM_COND_IMG_SIZE, pil.width / pil.height)
            ref_pils_vlm.append(pil.convert("RGB").resize((vw, vh), resample=Image.LANCZOS))

        # Build the chat template that embeds K reference images + prompt.
        content = [{"type": "image"} for _ in range(K)]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        template = (
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + _BOI_TOKEN
            + _TMS_TOKEN
        )
        proc = self.processor(
            text=[template],
            images=ref_pils_vlm,
            padding="longest",
            return_tensors="pt",
        )
        proc = {k: v.to(self.device) for k, v in proc.items() if isinstance(v, torch.Tensor)}

        # proc.input_ids: [1, txt+vlm_ref_len]
        # proc.pixel_values: pixel patches for the vision tower
        # proc.image_grid_thw: [K, 3] grid sizes for the K VLM refs
        input_ids_text = proc["input_ids"]
        pixel_values_vlm = proc.get("pixel_values")
        image_grid_thw_vlm = proc.get("image_grid_thw")

        # --- Pixel-patch path: patchify refs for vinputs ---
        ref_max = adaptive_ref_max_size(K, max(height, width))
        ref_patches, ref_image_lens = preprocess_ref_patches(
            ref_pils, max_size=ref_max, patch_size=PATCH_SIZE, dtype=self.dtype, device=self.device
        )

        # Build target + ref pixel-patch placeholder tokens for input_ids.
        def _make_image_span(n_patches: int) -> torch.Tensor:
            span = torch.full((1, n_patches), image_token_id, dtype=input_ids_text.dtype, device=self.device)
            span[0, 0] = vision_start_token_id
            return span

        target_tokens = _make_image_span(tgt_image_len)
        ref_pixel_tokens = [_make_image_span(n) for n in ref_image_lens]
        input_ids_padded = torch.cat([input_ids_text, target_tokens] + ref_pixel_tokens, dim=-1)

        # --- Position IDs ---
        # K VLM-ref spans: skip_vision_start_token=0 (sequential, no fix_point)
        # 1 target span:   skip_vision_start_token=1 (fix_point anchored)
        # K ref-pixel spans: skip_vision_start_token=1 (fix_point anchored)
        image_grid_thw_tgt = torch.tensor(
            [[1, h_patches, w_patches]], dtype=torch.int64, device=self.device
        )
        ref_pixel_grid_thw = torch.tensor(
            [[1, pil.size[1] // PATCH_SIZE, pil.size[0] // PATCH_SIZE]
             for pil, n in zip(ref_pils, ref_image_lens)
             for _ in [resize_pilimage(pil, ref_max)]],
            dtype=torch.int64,
            device=self.device,
        )
        # Build per-ref pixel grid from actual resized dimensions.
        ref_pixel_grid_thw_list = []
        for pil, n in zip(ref_pils, ref_image_lens):
            pil_r = resize_pilimage(pil.convert("RGB"), ref_max)
            rw, rh = pil_r.size
            ref_pixel_grid_thw_list.append([1, rh // PATCH_SIZE, rw // PATCH_SIZE])
        ref_pixel_grid_thw = torch.tensor(ref_pixel_grid_thw_list, dtype=torch.int64, device=self.device)

        image_grid_thw_all = torch.cat(
            [image_grid_thw_vlm, image_grid_thw_tgt, ref_pixel_grid_thw], dim=0
        ) if image_grid_thw_vlm is not None else torch.cat([image_grid_thw_tgt, ref_pixel_grid_thw], dim=0)

        skip_flags = [0] * K + [1] + [1] * K

        position_ids, _ = get_rope_index_fix_point(
            spatial_merge_size=1,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            input_ids=input_ids_padded,
            image_grid_thw=image_grid_thw_all,
            skip_vision_start_token=skip_flags,
        )

        txt_seq_len = input_ids_text.shape[-1]
        total_seq_len = position_ids.shape[-1]

        # --- Token types ---
        # 0 = text + VLM-ref tokens (proc.input_ids), 3 = tms, 1 = target patches, 2 = ref pixel patches
        token_types_raw = torch.zeros((1, total_seq_len), dtype=torch.long, device=self.device)
        tms_pos = txt_seq_len - 1  # tms_token is the last token in input_ids_text
        token_types_raw[0, tms_pos] = 3
        tgt_start = txt_seq_len
        tgt_end = tgt_start + tgt_image_len
        token_types_raw[0, tgt_start:tgt_end] = 1
        ref_start = tgt_end
        for n in ref_image_lens:
            token_types_raw[0, ref_start: ref_start + n] = 2
            ref_start += n

        vinput_mask = token_types_raw == 1  # only target positions
        token_types = (token_types_raw > 0).long()

        dense_mask, full_attn_spans = build_packed_attention_metadata(token_types)
        attn_metadata = AttentionMetadata(attn_mask=dense_mask, full_attn_spans=full_attn_spans)

        return {
            "input_ids": input_ids_text,
            "position_ids": position_ids,
            "token_types": token_types,
            "attn_metadata": attn_metadata,
            "vinput_mask": vinput_mask,
            "pixel_values": pixel_values_vlm,
            "image_grid_thw": image_grid_thw_all,
            "ref_patches": ref_patches,
            "ref_image_lens": ref_image_lens,
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
        # Phase 2: reference-image conditioning --------------------------
        vinputs: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        precomputed_image_embeds: torch.Tensor | None = None,
        precomputed_deepstack_image_embeds: list[torch.Tensor] | None = None,
        _embed_storage: dict | None = None,
        **_unused: Any,
    ) -> torch.Tensor:
        # For editing/personalization ``vinputs`` is cat([z, ref_patches]);
        # for T2I it's just z (both forms are accepted for backward compat).
        if vinputs is None:
            vinputs = z

        out = self.transformer.forward_generation(
            input_ids=input_ids,
            position_ids=position_ids,
            vinputs=vinputs,
            timestep=t_pixeldit.reshape(-1),
            token_types=token_types,
            attn_metadata=attn_metadata,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            precomputed_image_embeds=precomputed_image_embeds,
            precomputed_deepstack_image_embeds=precomputed_deepstack_image_embeds,
        )

        # On the first denoising step (no precomputed embeds yet), store the
        # just-computed embeds in the mutable side-channel dict so diffuse()
        # can promote them to precomputed_* on subsequent steps.
        if _embed_storage is not None and out.cond_image_embeds is not None:
            _embed_storage["image_embeds"] = out.cond_image_embeds.detach()
            _embed_storage["deepstack"] = (
                [e.detach() for e in out.cond_deepstack_image_embeds]
                if out.cond_deepstack_image_embeds else []
            )

        # vinput_mask selects only the *target* patch positions (type=1);
        # ref-pixel-patch positions (type=2) are present in out.x_pred but
        # discarded -- they were conditioning, not generation targets.
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
        ref_pils: list[Image.Image] | None = None,
    ) -> torch.Tensor:
        do_true_cfg = guidance_scale > 1.0
        has_refs = ref_pils is not None and len(ref_pils) > 0

        if has_refs:
            cond_sample = self._build_edit_sample(prompt, ref_pils, height, width)
            # For CFG with ref images, the uncond branch uses the same ref patches
            # (no ref → zero conditioning would cause different masking, so we keep
            # refs but use a null text prompt).  Most editing uses guidance=1.0 (no
            # CFG) so this path is rarely exercised in practice.
            negative_sample = self._build_edit_sample(" ", ref_pils, height, width) if do_true_cfg else None
        else:
            cond_sample = self._build_t2i_sample(prompt, height, width)
            negative_sample = self._build_t2i_sample(" ", height, width) if do_true_cfg else None

        ref_patches = cond_sample.pop("ref_patches", None)
        # ref_image_lens consumed inside _build_edit_sample; not needed by predict_noise
        cond_sample.pop("ref_image_lens", None)
        if negative_sample is not None:
            negative_sample.pop("ref_patches", None)
            negative_sample.pop("ref_image_lens", None)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = noise_scale * torch.randn((1, 3, height, width), generator=generator).to(self.device, self.dtype)
        z = patchify(noise)

        scheduler = self._build_scheduler(scheduler_name, num_inference_steps, shift)
        self.scheduler = scheduler  # scheduler_step() reads self.scheduler -- must be set before the loop

        # Mutable side-channel for vision-tower embed caching across steps.
        # On step 0, predict_noise fills embed_storage via _embed_storage kwarg.
        # On step 1+, we promote the cached embeds to precomputed_* so the
        # vision tower is never called again (one forward per request total).
        embed_storage: dict[str, Any] = {}

        with self.progress_bar(total=len(scheduler.timesteps)) as pbar:
            for step_t in scheduler.timesteps:
                t_pixeldit = 1.0 - step_t.float() / 1000.0
                sigma = (step_t.float() / 1000.0).clamp_min(1e-3)

                cached_img_emb = embed_storage.get("image_embeds")
                cached_ds_emb = embed_storage.get("deepstack")

                # Concatenate target noise + ref pixel patches for vinputs.
                vinputs = torch.cat([z, ref_patches], dim=1) if ref_patches is not None else None

                positive_kwargs: dict[str, Any] = {
                    **cond_sample,
                    "z": z,
                    "t_pixeldit": t_pixeldit,
                    "sigma": sigma,
                    # Phase 2 fields (None for T2I, populated for editing)
                    "vinputs": vinputs,
                    "precomputed_image_embeds": cached_img_emb,
                    "precomputed_deepstack_image_embeds": cached_ds_emb if cached_ds_emb else None,
                    # Only pass _embed_storage on step 0 (when embeds not yet cached)
                    "_embed_storage": embed_storage if cached_img_emb is None and has_refs else None,
                }

                if do_true_cfg and negative_sample is not None:
                    neg_vinputs = torch.cat([z, ref_patches], dim=1) if ref_patches is not None else None
                    negative_kwargs: dict[str, Any] | None = {
                        **negative_sample,
                        "z": z,
                        "t_pixeldit": t_pixeldit,
                        "sigma": sigma,
                        "vinputs": neg_vinputs,
                        # CFG uncond branch reuses the same cached embeds (same refs)
                        "precomputed_image_embeds": cached_img_emb,
                        "precomputed_deepstack_image_embeds": cached_ds_emb if cached_ds_emb else None,
                        "_embed_storage": None,  # only cond branch populates the cache
                    }
                else:
                    negative_kwargs = None

                v_guided = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,  # HiDream's CFG combine is plain linear blend, no norm-rescale
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
        if isinstance(first_prompt, str):
            prompt = first_prompt
            mm_data: dict = {}
        else:
            prompt = first_prompt.get("prompt", "")
            mm_data = first_prompt.get("multi_modal_data", {}) or {}

        # Extract reference images if provided (editing / personalization).
        raw_images = mm_data.get("image")
        if raw_images is None:
            ref_pils: list[Image.Image] | None = None
        elif isinstance(raw_images, Image.Image):
            ref_pils = [raw_images.convert("RGB")]
        elif isinstance(raw_images, str):
            ref_pils = [Image.open(raw_images).convert("RGB")]
        else:
            # Assume iterable of PIL / path
            ref_pils = [
                img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
                for img in raw_images
            ]

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
            ref_pils=ref_pils,
        )

        h_patches, w_patches = height // PATCH_SIZE, width // PATCH_SIZE
        images = depatchify(latents, h_patches, w_patches)
        return DiffusionOutput(output=images)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multi-anchor frame conditioning for LTX-2 / LTX-2.3 (FLF2V / FMLF).

Keyframe-interpolation building blocks for ``LTX2ConditionPipeline``. Mirrors the
Diffusers ``LTX2ConditionPipeline`` (``diffusers/pipelines/ltx2``):

* First-frame anchors (``latent_idx == 0``) overwrite the base latent timeline
  in place (``apply_first_frame_conditioning``).
* Non-first-frame anchors (``latent_idx > 0``) are appended as extra *keyframe
  tokens* with their own RoPE coordinates (``_prepare_keyframe_coords``), so a
  middle/last anchor is positioned at its own pixel-frame index. The sequence
  grows by those tokens for the denoise loop and is trimmed before VAE decode.

Re-expressed on the unified ``LTXRuntime``: conditioning is a mixin over
``LTX2Pipeline`` mirroring ``LTXI2VConditioningMixin``. The anchor pinning runs
through the ``_step_video_latents_i2v`` hook (which the shared step adapter calls
whenever a conditioning mask is present), so no shared runtime file is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import PIL.Image
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from . import ltx2_latents as latent_ops
from .ltx2_guidance import euler_step_from_velocity

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    from .ltx2_conditioning import LTXPromptContext
    from .ltx2_denoise import LTXDenoiseContext, LTXForwardContext
    from .ltx2_request import LTXRequestInputs


@dataclass
class LTX2VideoCondition:
    """A single frame-conditioning item for LTX-2 / LTX-2.3 video generation.

    Ports the Diffusers ``LTX2VideoCondition`` dataclass. Used by
    ``LTX2ConditionPipeline`` for FLF2V (first-last-frame) and FMLF
    (first-middle-last-frame) workflows.

    Attributes:
        frames: The image (or video) to condition on. PIL, list-of-PIL, numpy
            array, or ``torch.Tensor`` -- anything ``VideoProcessor`` handles.
        index: The frame index at which the condition is applied. May be
            negative (``-1`` = last frame).
        strength: Conditioning strength in ``[0, 1]``. ``1.0`` = fully applied.
    """

    frames: PIL.Image.Image | list[PIL.Image.Image] | np.ndarray | torch.Tensor
    index: int = 0
    strength: float = 1.0


@dataclass
class _KeyframeExtras:
    """Appended keyframe tokens for non-first-frame anchors (``latent_idx > 0``)."""

    tokens: torch.Tensor  # (B, num_keyframe_tokens, C)
    coords: torch.Tensor  # (B, 3, num_keyframe_tokens, 2) -- to append to video_coords
    mask: torch.Tensor  # (B, num_keyframe_tokens) -- conditioning strength per token


def preprocess_conditions(
    conditions: list[LTX2VideoCondition],
    video_processor: VideoProcessor,
    height: int,
    width: int,
    latent_num_frames: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[float], list[int], list[int]]:
    """Validate and preprocess a list of :class:`LTX2VideoCondition` items.

    Returns parallel lists ``(frames_tensors, strengths, indices, pixel_frames)``.
    ``pixel_frames`` is the number of pixel-space frames per condition (1 for a
    single image), used to compute keyframe RoPE coordinates. Negative ``index``
    values are resolved against ``latent_num_frames``.

    Raises ``ValueError`` for empty input, out-of-range indices, or strengths
    outside ``[0, 1]``.
    """
    if not conditions:
        raise ValueError("LTX2ConditionPipeline requires at least one condition.")

    frames_tensors: list[torch.Tensor] = []
    strengths: list[float] = []
    indices: list[int] = []
    pixel_frames: list[int] = []
    for cond in conditions:
        if not 0.0 <= cond.strength <= 1.0:
            raise ValueError(f"Condition strength must be in [0, 1], got {cond.strength}.")
        resolved = cond.index if cond.index >= 0 else latent_num_frames + cond.index
        if not 0 <= resolved < latent_num_frames:
            raise ValueError(
                f"Condition index {cond.index} resolves to {resolved}, "
                f"outside the valid range [0, {latent_num_frames})."
            )
        frames_tensor = video_processor.preprocess_video(cond.frames, height=height, width=width).to(
            device=device, dtype=dtype
        )
        frames_tensors.append(frames_tensor)
        strengths.append(float(cond.strength))
        indices.append(resolved)
        pixel_frames.append(int(frames_tensor.shape[2]))
    return frames_tensors, strengths, indices, pixel_frames


def encode_condition_latents(
    pipeline: Any,
    condition_frames: list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """VAE-encode, normalize each condition tensor; return ``(latents_5d, latents_packed)``.

    The unpacked 5D latents are kept for keyframe coordinate dimensions; the
    packed 3D latents feed both first-frame replacement and keyframe append.
    """
    latents_5d: list[torch.Tensor] = []
    latents_packed: list[torch.Tensor] = []
    for condition_tensor in condition_frames:
        condition_latent = retrieve_latents(
            pipeline.vae.encode(condition_tensor.to(dtype=pipeline.vae.dtype)),
            generator=generator,
            sample_mode="argmax",
        )
        condition_latent = latent_ops.normalize_latents(
            condition_latent, pipeline.vae.latents_mean, pipeline.vae.latents_std
        ).to(device=device, dtype=dtype)
        packed = latent_ops.pack_latents(
            condition_latent,
            pipeline.transformer_spatial_patch_size,
            pipeline.transformer_temporal_patch_size,
        )
        latents_5d.append(condition_latent)
        latents_packed.append(packed)
    return latents_5d, latents_packed


def apply_first_frame_conditioning(
    latents: torch.Tensor,
    conditioning_mask: torch.Tensor,
    condition_latents: list[torch.Tensor],
    condition_strengths: list[float],
    condition_indices: list[int],
    *,
    latent_height: int,
    latent_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Overwrite ``latents`` / ``conditioning_mask`` in place at the first-frame
    positions and build the parallel ``clean_latents`` tensor.

    Only conditions with ``latent_idx == 0`` are applied here (matching Diffusers'
    ``apply_first_frame_conditioning``); non-first-frame anchors are appended as
    keyframe tokens instead and are skipped. ``conditioning_mask`` is ``(B, N)``.
    """
    clean_latents = torch.zeros_like(latents)
    for cond, strength, latent_idx in zip(condition_latents, condition_strengths, condition_indices, strict=True):
        if latent_idx != 0:
            continue
        num_cond_tokens = cond.size(1)
        start_token_idx = latent_idx * latent_height * latent_width
        end_token_idx = start_token_idx + num_cond_tokens
        latents[:, start_token_idx:end_token_idx] = cond
        conditioning_mask[:, start_token_idx:end_token_idx] = strength
        clean_latents[:, start_token_idx:end_token_idx] = cond
    return latents, conditioning_mask, clean_latents


class LTXMultiAnchorConditioningMixin:
    """Multi-anchor frame conditioning (FLF2V / FMLF) for LTX-2 / LTX-2.3.

    Sibling to ``LTXI2VConditioningMixin``: whereas I2V pins only the first
    frame, this mixin pins an arbitrary set of anchor frame indices, each with
    its own strength. Applied via MRO on top of ``LTX2Pipeline`` so it reuses the
    prompt connector, x0-space CFG, and audio branch unchanged.

    First-frame anchors overwrite the base timeline; non-first anchors are
    appended as keyframe tokens (with their own RoPE coordinates) for the denoise
    loop and trimmed before VAE decode. The anchor pinning runs through
    ``_step_video_latents_i2v`` (the conditioning-step hook the shared step
    adapter calls whenever a conditioning mask is present) as an x0-space blend
    toward the clean anchor latents. Transient per-request state is stashed on
    ``self`` for the duration of ``forward`` and cleared in its ``finally``. When
    no conditions are present the pipeline degrades to the pure T2V path.
    """

    support_image_input = True
    unified_text_image_entry = True
    # Conditions are resolved from the first request only; disable request batching
    # so a batch cannot silently apply one item's anchors to every output.
    supports_request_batch = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio,
            resample="bilinear",
        )

    @staticmethod
    def _resolve_conditions_from_request(
        req: DiffusionRequestBatch,
    ) -> list[LTX2VideoCondition] | None:
        """Read condition anchors from the request's ``multi_modal_data``.

        The OpenAI-compatible serving layer stashes a list of objects exposing
        ``.data`` (PIL image), ``.index`` (int), and ``.strength`` (float) under
        ``prompt["multi_modal_data"]["conditions"]``. Duck-typed here so the
        pipeline does not depend on serving types.
        """
        if not req.prompts:
            return None
        first = req.prompts[0]
        if isinstance(first, str):
            return None
        multi_modal_data = first.get("multi_modal_data") or {}
        raw_conditions = multi_modal_data.get("conditions")
        if not raw_conditions:
            return None
        resolved: list[LTX2VideoCondition] = []
        for anchor in raw_conditions:
            if isinstance(anchor, LTX2VideoCondition):
                resolved.append(anchor)
                continue
            resolved.append(
                LTX2VideoCondition(
                    frames=anchor.data,
                    index=int(anchor.index),
                    strength=float(anchor.strength),
                )
            )
        return resolved or None

    def _prepare_keyframe_coords(
        self,
        keyframe_latent_num_frames: int,
        keyframe_latent_height: int,
        keyframe_latent_width: int,
        pixel_frame_idx: int,
        num_pixel_frames: int,
        fps: float,
        device: torch.device,
    ) -> torch.Tensor:
        """RoPE coordinates for a keyframe condition appended as extra tokens.

        Ports Diffusers' ``_prepare_keyframe_coords``: latent coords are scaled to
        pixel space *without* the first-frame causal fix, the temporal axis is
        offset by ``pixel_frame_idx``, single-frame keyframes are clamped to a
        one-timestep extent, and time is divided by ``fps``. Output shape
        ``[1, 3, num_patches, 2]`` matches the base ``prepare_video_coords``
        layout so it can be concatenated onto it.
        """
        patch_size = self.transformer_spatial_patch_size
        patch_size_t = self.transformer_temporal_patch_size
        scale_factors = (
            self.vae_temporal_compression_ratio,
            self.vae_spatial_compression_ratio,
            self.vae_spatial_compression_ratio,
        )

        grid_f = torch.arange(
            start=0, end=keyframe_latent_num_frames, step=patch_size_t, dtype=torch.float32, device=device
        )
        grid_h = torch.arange(start=0, end=keyframe_latent_height, step=patch_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(start=0, end=keyframe_latent_width, step=patch_size, dtype=torch.float32, device=device)
        grid = torch.stack(torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij"), dim=0)

        patch_size_delta = torch.tensor((patch_size_t, patch_size, patch_size), dtype=grid.dtype, device=device)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        latent_coords = torch.stack([grid, patch_ends], dim=-1).flatten(1, 3).unsqueeze(0)  # [1, 3, num_patches, 2]

        scale_tensor = torch.tensor(scale_factors, device=device, dtype=latent_coords.dtype)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)

        # No causal fix: place the keyframe at `pixel_frame_idx` directly.
        pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] + pixel_frame_idx
        if num_pixel_frames == 1:
            # Single-pixel-frame keyframe: clamp temporal extent to [idx, idx + 1).
            pixel_coords[:, 0, :, 1:] = pixel_coords[:, 0, :, :1] + 1
        pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps
        return pixel_coords

    def _prepare_condition_latents(
        self,
        conditions: list[LTX2VideoCondition],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        noise_scale: float,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, _KeyframeExtras | None]:
        """Build the base ``(latents, conditioning_mask, clean_latents)`` plus the
        appended keyframe extras (or ``None`` if all anchors are first-frame).

        ``conditioning_mask`` is packed ``(B, N)``. First-frame anchors are
        overwritten in place; non-first anchors become keyframe tokens with their
        own coordinates. Caller-provided ``latents`` are renoised SDEdit-style
        (deferred two-stage Stage 2); otherwise the base state is Gaussian noise
        at non-anchor tokens.
        """
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
        mask_shape = (batch_size, 1, latent_num_frames, latent_height, latent_width)

        latents_supplied = latents is not None
        if latents_supplied:
            latents = latent_ops.normalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
        else:
            latents = torch.zeros(shape, device=device, dtype=dtype)

        conditioning_mask = latents.new_zeros(mask_shape)
        latents = latent_ops.pack_latents(
            latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )
        conditioning_mask = latent_ops.pack_latents(
            conditioning_mask, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        ).squeeze(-1)  # (B, N)

        if isinstance(generator, list):
            generator = generator[0]

        condition_frames, strengths, indices, pixel_frames = preprocess_conditions(
            conditions, self.video_processor, height, width, latent_num_frames, device=device, dtype=dtype
        )
        latents_5d, latents_packed = encode_condition_latents(
            self, condition_frames, device=device, dtype=dtype, generator=generator
        )

        # First-frame anchors (latent_idx == 0): in-place overwrite.
        latents, conditioning_mask, clean_latents = apply_first_frame_conditioning(
            latents,
            conditioning_mask,
            latents_packed,
            strengths,
            indices,
            latent_height=latent_height,
            latent_width=latent_width,
        )

        # Non-first-frame anchors (latent_idx > 0): build appended keyframe tokens.
        frame_scale_factor = self.vae_temporal_compression_ratio
        kf_tokens, kf_coords, kf_mask = [], [], []
        for cond_5d, cond_packed, strength, latent_idx, num_pixel_frames in zip(
            latents_5d, latents_packed, strengths, indices, pixel_frames, strict=True
        ):
            if latent_idx == 0:
                continue
            _, _, kf_latent_frames, kf_latent_height, kf_latent_width = cond_5d.shape
            pixel_frame_idx = (latent_idx - 1) * frame_scale_factor + 1
            coords = self._prepare_keyframe_coords(
                kf_latent_frames,
                kf_latent_height,
                kf_latent_width,
                pixel_frame_idx,
                num_pixel_frames,
                frame_rate,
                device,
            )
            kf_tokens.append(cond_packed)
            kf_coords.append(coords)
            kf_mask.append(cond_packed.new_full((cond_packed.shape[0], cond_packed.shape[1]), float(strength)))

        keyframe: _KeyframeExtras | None = None
        if kf_tokens:
            keyframe = _KeyframeExtras(
                tokens=torch.cat(kf_tokens, dim=1),
                coords=torch.cat(kf_coords, dim=2),
                mask=torch.cat(kf_mask, dim=1),
            )

        noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        effective_scale = noise_scale if latents_supplied else 1.0
        base_scaled = (effective_scale * (1.0 - conditioning_mask)).unsqueeze(-1)
        latents = base_scaled * noise + (1.0 - base_scaled) * latents

        # Renoise keyframe tokens with the same formula (strength=1 keeps them clean).
        if keyframe is not None:
            kf_scaled = (effective_scale * (1.0 - keyframe.mask)).unsqueeze(-1)
            kf_noise = randn_tensor(
                keyframe.tokens.shape, generator=generator, device=keyframe.tokens.device, dtype=keyframe.tokens.dtype
            )
            keyframe.tokens = kf_scaled * kf_noise + (1.0 - kf_scaled) * keyframe.tokens

        return latents, conditioning_mask, clean_latents, keyframe

    def _prepare_video_latents_stage(
        self,
        request_inputs: LTXRequestInputs,
        prompt_context: LTXPromptContext,
        *,
        device: torch.device,
        noise_scale: float,
        image: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        conditions = getattr(self, "_pending_conditions", None)
        if not conditions:
            return super()._prepare_video_latents_stage(
                request_inputs, prompt_context, device=device, noise_scale=noise_scale, image=image
            )

        latents, conditioning_mask, clean_latents, keyframe = self._prepare_condition_latents(
            conditions,
            prompt_context.batch_size * request_inputs.num_videos_per_prompt,
            self.transformer.config.in_channels,
            request_inputs.height,
            request_inputs.width,
            request_inputs.num_frames,
            request_inputs.frame_rate,
            noise_scale,
            prompt_context.positive_connector_prompt_embeds.dtype,
            device,
            request_inputs.generator,
            request_inputs.latents,
        )

        # The step hook operates on the *extended* sequence (base + keyframe
        # tokens); build the full mask + clean latents for it here.
        if keyframe is not None:
            full_mask = torch.cat([conditioning_mask, keyframe.mask], dim=1)
            full_clean = torch.cat([clean_latents, keyframe.tokens], dim=1)
        else:
            full_mask, full_clean = conditioning_mask, clean_latents

        self._pending_keyframe = keyframe
        self._pending_num_keyframe_tokens = keyframe.tokens.shape[1] if keyframe is not None else 0
        self._pending_conditioning_mask = full_mask
        self._pending_clean_latents = full_clean
        return latents, conditioning_mask

    def _step_video_latents_i2v(
        self,
        noise_pred_video: torch.Tensor,
        latents: torch.Tensor,
        step_index: int,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        # Pin every conditioned token (first-frame slots + appended keyframe
        # tokens) via an x0-space blend, then take the Euler step. Operates on the
        # packed sequence directly so it is agnostic to the appended keyframes.
        mask = getattr(self, "_pending_conditioning_mask", None)
        clean = getattr(self, "_pending_clean_latents", None)
        if mask is None or clean is None:
            return super()._step_video_latents_i2v(
                noise_pred_video, latents, step_index, latent_num_frames, latent_height, latent_width
            )
        bsz = latents.size(0)
        if mask.shape[0] != bsz:
            mask = mask[:bsz]
            clean = clean[:bsz]
        sigma = self.scheduler.sigmas[step_index].to(latents.dtype)
        mask_3d = mask.unsqueeze(-1)
        x0 = latents - noise_pred_video * sigma
        x0_blended = x0 * (1 - mask_3d) + clean * mask_3d
        velocity_corr = (latents - x0_blended) / sigma
        return euler_step_from_velocity(latents, velocity_corr, self.scheduler.sigmas, step_index)

    def _prepare_denoise_context_for_cfg(
        self,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext:
        # Extend the sequence with keyframe tokens/coords/mask *before* the base
        # CFG prep duplicates video_coords, so the extras get duplicated too.
        keyframe = getattr(self, "_pending_keyframe", None)
        if keyframe is not None:
            denoise_ctx.latents = torch.cat([denoise_ctx.latents, keyframe.tokens], dim=1)
            denoise_ctx.video_coords = torch.cat([denoise_ctx.video_coords, keyframe.coords], dim=2)
            if denoise_ctx.conditioning_mask is not None:
                denoise_ctx.conditioning_mask = torch.cat([denoise_ctx.conditioning_mask, keyframe.mask], dim=1)

        denoise_ctx = super()._prepare_denoise_context_for_cfg(forward_ctx, denoise_ctx)
        if denoise_ctx.conditioning_mask is None:
            return denoise_ctx

        mask_batch = denoise_ctx.conditioning_mask.shape[0]
        model_batch = denoise_ctx.video_coords.shape[0]
        if model_batch % mask_batch != 0:
            raise ValueError(
                f"Condition mask batch must divide the Transformer input batch, but got {mask_batch} and {model_batch}."
            )
        repeats = model_batch // mask_batch
        denoise_ctx.conditioning_mask_for_model = (
            denoise_ctx.conditioning_mask if repeats == 1 else torch.cat([denoise_ctx.conditioning_mask] * repeats)
        )
        return denoise_ctx

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
        *,
        video_token_count: int,
        audio_token_count: int,
    ) -> dict[str, torch.Tensor]:
        kwargs = super()._denoise_timestep_kwargs(
            ts, forward_ctx, denoise_ctx, video_token_count=video_token_count, audio_token_count=audio_token_count
        )
        conditioning_mask = (
            denoise_ctx.conditioning_mask if forward_ctx.cfg_parallel_ready else denoise_ctx.conditioning_mask_for_model
        )
        if conditioning_mask is None:
            return kwargs
        kwargs.update(timestep=ts.reshape(-1, 1) * (1 - conditioning_mask))
        return kwargs

    def _video_guidance_model_sigma(
        self,
        sigma: torch.Tensor,
        denoise_ctx: LTXDenoiseContext,
    ) -> torch.Tensor:
        if denoise_ctx.conditioning_mask is None:
            return super()._video_guidance_model_sigma(sigma, denoise_ctx)
        # Conditioned tokens are evaluated at timestep zero, so velocity-to-x0
        # guidance must use that same per-token sigma.
        return sigma.reshape(-1, 1, 1) * (1 - denoise_ctx.conditioning_mask).unsqueeze(-1)

    def _unpack_and_denormalize_stage(
        self,
        forward_ctx: LTXForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Trim appended keyframe tokens before the base grid is unpacked.
        num_keyframe_tokens = getattr(self, "_pending_num_keyframe_tokens", 0)
        if num_keyframe_tokens:
            latents = latents[:, : latents.shape[1] - num_keyframe_tokens]
        return super()._unpack_and_denormalize_stage(forward_ctx, latents, audio_latents)

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        conditions: list[LTX2VideoCondition] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Multi-anchor conditional forward.

        ``conditions=None`` triggers the request-side resolver (serving path); an
        explicit empty list opts out. Falls back to pure T2V when no conditions
        resolve.
        """
        if conditions is None:
            conditions = self._resolve_conditions_from_request(req)
        try:
            self._pending_conditions = conditions or None
            self._pending_keyframe = None
            self._pending_num_keyframe_tokens = 0
            self._pending_conditioning_mask = None
            self._pending_clean_latents = None
            return super().forward(req, **kwargs)
        finally:
            self._pending_conditions = None
            self._pending_keyframe = None
            self._pending_num_keyframe_tokens = 0
            self._pending_conditioning_mask = None
            self._pending_clean_latents = None

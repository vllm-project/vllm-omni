# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Generated standalone Kandinsky 6 vLLM-Omni pipeline.

Unlike the Diffusers port (one fully self-contained file, shipped into an
external package), this file is one member of a normal multi-file
vLLM-Omni model package — it imports its sibling generated modules
(``modeling_kandinsky6``, ``modeling_kandinsky6_vae``,
``modeling_kandinsky6_audio``, ``scheduling_kandinsky6``) via relative
imports, exactly like vLLM-Omni's other native pipelines (e.g.
``sana_video/pipeline_sana_video.py`` importing from
``.transformer_sana_video``).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import torch
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers import AutoProcessor as QwenAutoProcessor

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportAudioOutput, SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.utils import _load_json
from vllm_omni.diffusion.offloader.config import offload_enabled
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .kandinsky6_transformer import Kandinsky6Transformer3DModel
from .modeling_kandinsky6_audio import Kandinsky6AudioVAE
from .modeling_kandinsky6_vae import AutoencoderKLHunyuanVideo
from .scheduling_kandinsky6 import KandinskyFlowMatchScheduler


class TextEmbeds(TypedDict):
    """Output of TextEmbedder.encode() — packed across batch."""

    text_embeds: Tensor  # (total_seq_len, text_dim)  Qwen hidden states
    pooled_embed: Tensor  # (total_seq_len, clip_dim)  CLIP pooled; packed per-sample


@dataclass
class LatentBundle:
    """Packed latent state flowing through the denoising loop.

    Both video and audio tensors use varlen layout: segments from different
    batch items are concatenated along dim-0, with cu_seqlens marking
    boundaries.  audio is None for T2V modes.
    """

    video: Tensor | None  # (sum_T, H, W, C)  e.g. sum_T = bs*duration
    audio: Tensor | None  # (sum_A, audio_dim); None for T2V
    video_cu_seqlens: Tensor | None  # (bs+1,) int32
    audio_cu_seqlens: Tensor | None  # (bs+1,) int32; None for T2V


@dataclass
class Kandinsky6PipelineOutput:
    frames: Tensor  # (bs, 3, T, H, W) uint8
    audio: list[np.ndarray] | None = None  # list[bs] of (samples,) int16; None for T2V
    path: str | None = None  # set when save_path was passed to the pipeline


def _build_video_input(
    video: Tensor,
    has_visual_cond: bool,
    first_frames: Tensor | None,
    video_cu_seqlens: Tensor | None,
    visual_cond_scheme: str,
) -> Tensor:
    """Appends visual-conditioning channels [cond, mask] to the video latent.

    Schemes (K5 parity):
      - ``pretrain``: inject first_frames into cond channels at sequence starts
      - ``i2v``: overwrite latents at sequence starts; cond unused
      - ``tail_cond_first_frame``: overwrite / mask the last (reference) frame
    """
    if not has_visual_cond:
        return video

    cond = torch.zeros_like(video)
    mask = torch.zeros([*video.shape[:-1], 1], dtype=video.dtype, device=video.device)

    if first_frames is not None:
        ff = first_frames.to(device=video.device, dtype=video.dtype)
        if video_cu_seqlens is None:
            raise ValueError(f"{visual_cond_scheme} requires video_cu_seqlens")
        starts = video_cu_seqlens[:-1]
        tail_cond = visual_cond_scheme == "tail_cond_first_frame"
        inject = video_cu_seqlens[1:] - 1 if tail_cond else starts

        if visual_cond_scheme == "i2v":
            video[starts] = ff
        elif tail_cond:
            video[inject] = ff
        elif visual_cond_scheme == "pretrain":
            cond[starts] = ff
        else:
            raise ValueError(f"unknown visual_cond_scheme={visual_cond_scheme!r}")
        mask[inject] = 1

    return torch.cat([video, cond, mask], dim=-1)


def _resolve_null_embeds(
    null_text_embeds: TextEmbeds | list[TextEmbeds | None],
    null_text_rope: Tensor | list[Tensor | None],
) -> tuple[Tensor, Tensor, Tensor | list[Tensor]]:
    """Returns (text_embed, pooled_embed, text_rope) for the uncond DiT pass.

    When null_text_embeds is a list of two items (T2VA per-modality nulls),
    returns list arguments so the DiT can apply different null text per modality.
    """
    if isinstance(null_text_embeds, list):
        # null_text_embeds[0] = video null, null_text_embeds[1] = audio null (may be None)
        null_v = null_text_embeds[0]
        null_a = null_text_embeds[1]
        null_rope_v = null_text_rope[0] if isinstance(null_text_rope, list) else null_text_rope
        null_rope_a = null_text_rope[1] if isinstance(null_text_rope, list) else None

        if null_a is not None and null_rope_a is not None:
            # Different null text per modality — pass as list to DiT
            text_embed = [null_v["text_embeds"], null_a["text_embeds"]]
            pooled_embed = [null_v["pooled_embed"], null_a["pooled_embed"]]
            rope = [null_rope_v, null_rope_a]
        else:
            text_embed = null_v["text_embeds"]
            pooled_embed = null_v["pooled_embed"]
            rope = null_rope_v
    else:
        text_embed = null_text_embeds["text_embeds"]
        pooled_embed = null_text_embeds["pooled_embed"]
        rope = null_text_rope

    return text_embed, pooled_embed, rope


def _raw_dit(dit: nn.Module) -> nn.Module:
    inner = getattr(dit, "module", None)
    return inner if inner is not None and hasattr(dit, "set_cache") else dit


def _rebuild_text_rope(dit: nn.Module, template: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
    """Rebuild text RoPE using the standalone Diffusers transformer names."""
    raw = getattr(dit, "module", None)
    raw = raw if raw is not None and hasattr(raw, "set_cache") else dit
    if isinstance(template, list):
        if getattr(raw, "is_multimodal", False):
            modules = [raw.video_text_rope_embeddings, raw.audio_text_rope_embeddings]
        else:
            modules = [raw.text_rope_embeddings] * len(template)
        return [compute_rope1d(module, int(item.shape[0])) for module, item in zip(modules, template, strict=True)]
    module = raw.video_text_rope_embeddings if getattr(raw, "is_multimodal", False) else raw.text_rope_embeddings
    return compute_rope1d(module, int(template.shape[0]))


def apply_cfg(cond: Tensor, uncond: Tensor, guidance_weight: float) -> Tensor:
    """Classifier-free guidance: uncond + w * (cond - uncond)."""
    return uncond + guidance_weight * (cond - uncond)


def prepare_video_latents(
    bs: int,
    duration: int,
    H_lat: int,
    W_lat: int,
    C: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> LatentBundle:
    """Sample random video noise latent and compute cu_seqlens.

    Returns a LatentBundle with video=(bs*duration, H_lat, W_lat, C) and
    uniform video_cu_seqlens=(bs+1,) int32.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    video = torch.randn(bs * duration, H_lat, W_lat, C, device=device, dtype=dtype, generator=g)
    cu = duration * torch.arange(bs + 1, dtype=torch.int32, device=device)
    return LatentBundle(video=video, audio=None, video_cu_seqlens=cu, audio_cu_seqlens=None)


def audio_latent_duration(
    video_latent_frames: int,
    *,
    fps: float = 24.0,
    audio_fps: int = 44100,
    downsample_factor: int = 1024,
) -> int:
    """Audio latent length matching K5 T2VA: ceil(sample_frames/fps * audio_fps / downsample)."""
    sample_frames = (video_latent_frames - 1) * 4 + 1
    return int(math.ceil(sample_frames / fps * audio_fps / downsample_factor))


def prepare_audio_latents(
    bundle: LatentBundle,
    audio_duration: int,
    audio_dim: int,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> LatentBundle:
    """Attach random audio noise latent to an existing LatentBundle.

    audio_duration: number of latent audio frames per batch item.
    """
    bs = bundle.video_cu_seqlens.shape[0] - 1 if bundle.video_cu_seqlens is not None else 1
    g = torch.Generator(device=device)
    g.manual_seed(seed + 1)  # offset from video seed for independence
    audio = torch.randn(bs * audio_duration, audio_dim, device=device, dtype=dtype, generator=g)
    cu = audio_duration * torch.arange(bs + 1, dtype=torch.int32, device=device)
    return LatentBundle(
        video=bundle.video,
        audio=audio,
        video_cu_seqlens=bundle.video_cu_seqlens,
        audio_cu_seqlens=cu,
    )


def compute_rope1d(rope: nn.Module, length: int, device: torch.device | None = None) -> Tensor:
    """Build 1-D RoPE for positions ``0..length-1``."""
    if length < 1:
        raise ValueError(f"rope length must be positive, got {length}")
    # Index the table where it lives (the DiT may still be CPU-resident under
    # sequential offload) and move the small result to the requested device.
    table_device = next(rope.buffers()).device
    out = rope(torch.arange(length, device=table_device))
    return out if device is None or out.device == torch.device(device) else out.to(device)


def compute_visual_rope(
    rope: nn.Module,
    shape: tuple[int, int, int],
    scale_factor: tuple[float, float, float],
    device: torch.device | None = None,
) -> Tensor:
    """Build 3-D RoPE grid for ``shape=(T,H,W)`` with prefix arange positions."""
    T, H, W = (int(shape[0]), int(shape[1]), int(shape[2]))
    if T < 1 or H < 1 or W < 1:
        raise ValueError(f"visual rope shape must be positive, got {shape}")
    table_device = next(rope.buffers()).device
    pos = [
        torch.arange(T, device=table_device),
        torch.arange(H, device=table_device),
        torch.arange(W, device=table_device),
    ]
    scale = (float(scale_factor[0]), float(scale_factor[1]), float(scale_factor[2]))
    out = rope((T, H, W), pos, scale)
    return out if device is None or out.device == torch.device(device) else out.to(device)


@torch.no_grad()
def postprocess_audio(
    bundle: LatentBundle,
    audio_vae,
    normalization_mode: str = "normalize",
) -> list[np.ndarray] | None:
    """Decode audio latents → list of (samples,) int16 numpy arrays.

    Returns None when bundle.audio is None (T2V mode).

    ``normalize`` (default, matches the k6_video production pipeline)
    peak-normalizes each waveform before converting it to int16.
    ``clip`` preserves the decoded amplitude and saturates it to [-1, 1].
    """
    audio = bundle.audio
    if audio is None:
        return None

    cu = bundle.audio_cu_seqlens
    assert cu is not None
    bs = cu.shape[0] - 1

    # Reverse audio VAE normalization
    audio_scaled = audio / getattr(audio_vae, "scaling_factor", 1.0)
    audio_scaled = audio_scaled + getattr(audio_vae, "mean_value", 0.0)

    result: list[np.ndarray] = []
    for i in range(bs):
        segment = audio_scaled[cu[i].item() : cu[i + 1].item()]  # (A, audio_dim)
        waveform = (
            audio_vae.wrapped_decode(segment.transpose(1, 0).unsqueeze(0).to(audio_vae.device))
            .squeeze()
            .cpu()
            .float()
            .numpy()
        )
        if normalization_mode == "normalize":
            peak = np.max(np.abs(waveform))
            if peak > 0:
                waveform = waveform / peak * 32767
        elif normalization_mode == "clip":
            waveform = np.clip(waveform, -1.0, 1.0) * 32767
        else:
            raise ValueError(f"unknown audio normalization_mode={normalization_mode}")

        result.append(waveform.astype(np.int16))

    return result


@torch.no_grad()
def postprocess_video(
    bundle: LatentBundle,
    vae,
    bs: int,
) -> Tensor:
    """Decode video latents → (bs, 3, T, H, W) uint8 in [0, 255].

    Input layout: (sum_T, H_lat, W_lat, C) — packed over batch*duration.
    """
    video = bundle.video
    assert video is not None

    # (sum_T, H, W, C) → (bs, T, H, W, C)
    frames = video.reshape(bs, -1, video.shape[-3], video.shape[-2], video.shape[-1])
    # (bs, T, H, W, C) → (bs, C, T, H, W) for VAE input
    frames = (frames / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
    # Hunyuan VAE loads as fp16; DiT latents are bf16 — match weight dtype.
    vae_dtype = next(vae.parameters()).dtype
    frames = vae.decode(frames.to(dtype=vae_dtype)).sample

    return ((frames.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)


@torch.no_grad()
def resize_image(
    image: Tensor,
    max_area: int,
    divisibility: int = 16,
    world_size: int = 1,
) -> tuple[Tensor, float]:
    """Aspect-preserving resize to fit ``max_area`` with sides divisible by ``divisibility``.

    K5 ``i2v_pipeline.resize_image`` parity. ``image`` is ``(B, C, H, W)``.
    Returns ``(resized, scale_k)``.
    """
    from math import sqrt

    h, w = image.shape[2:]
    area = h * w
    div = divisibility
    if div == 16:
        if world_size in (2, 4):
            div *= 2
        elif world_size == 8:
            div *= 4
    k = sqrt(max_area / area) / div
    new_h = int(round(h * k) * div)
    new_w = int(round(w * k) * div)
    import torchvision.transforms.functional as TF

    return TF.resize(image, (new_h, new_w)), k


def _load_pil_rgb(image: str | object):
    from PIL import Image

    if isinstance(image, str):
        try:
            pil_image = Image.open(image).convert("RGB")
            pil_image.load()
        except Exception as exc:
            raise ValueError(f"Cannot decode i2va input image {image!r}: {exc}") from exc
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"i2va image must be a path or PIL image, got {type(image).__name__}")
    return pil_image


@torch.no_grad()
def encode_i2va_first_frame(
    image: str | object,
    vae,
    device: torch.device | str,
    height: int | None = None,
    width: int | None = None,
    *,
    max_area: int | None = None,
    divisibility: int = 16,
    world_size: int = 1,
) -> tuple[Tensor, int, int]:
    """VAE-encode a reference image to one latent frame ``(1, H_lat, W_lat, C)``.

    Two modes (K5 parity):
      - ``max_area`` set (and height/width omitted): aspect-preserving resize like
        K5 ``resize_image`` / ``get_first_frame_from_image`` — output HxW from input.
      - ``height`` + ``width`` set: cover-resize + center-crop to that canvas
        (K5 ``encode_i2va_first_frame``).

    Returns ``(latent, pixel_height, pixel_width)``.
    """
    import torchvision.transforms.functional as TF

    pil_image = _load_pil_rgb(image)
    tensor = TF.pil_to_tensor(pil_image).unsqueeze(0)

    if max_area is not None and (height is None or width is None):
        tensor, _ = resize_image(tensor, max_area=max_area, divisibility=divisibility, world_size=world_size)
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    else:
        if height is None or width is None:
            raise ValueError("encode_i2va_first_frame needs height/width or max_area")
        src_h, src_w = tensor.shape[-2:]
        scale = min(src_h / height, src_w / width)
        tensor = TF.resize(tensor, (int(src_h / scale), int(src_w / scale)))
        cur_h, cur_w = tensor.shape[-2:]
        tensor = TF.crop(tensor, (cur_h - height) // 2, (cur_w - width) // 2, height, width)

    tensor = tensor / 127.5 - 1.0
    vae_dtype = next(vae.parameters()).dtype
    # (1, 3, H, W) → (1, 3, 1, H, W) video layout for Hunyuan VAE
    tensor = tensor.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
    latent = vae.encode(tensor, opt_tiling=False).latent_dist.sample()
    latent = latent.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor
    return latent, int(height), int(width)


def append_i2va_tail_condition(
    latent_visual: Tensor,
    first_frames: Tensor,
    batch_size: int,
    video_duration: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Append one clean first-frame latent and build generated/reference token ids.

    Returns ``(latent, token_types, generated_mask)`` where ``token_types`` is
    0=generated / 1=reference and ``generated_mask`` selects frames to decode.
    """
    _, height, width, dim = latent_visual.shape
    first_frames = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
    expected = (batch_size, height, width, dim)
    if tuple(first_frames.shape) != expected:
        raise ValueError(f"first-frame latent shape mismatch: expected {expected}, got {tuple(first_frames.shape)}")
    if latent_visual.shape[0] != batch_size * video_duration:
        raise ValueError(
            "generated visual latent length mismatch: expected "
            f"{batch_size * video_duration}, got {latent_visual.shape[0]}"
        )

    latent_visual = torch.cat(
        [
            latent_visual.reshape(batch_size, video_duration, height, width, dim),
            first_frames[:, None],
        ],
        dim=1,
    ).reshape(batch_size * (video_duration + 1), height, width, dim)

    token_types = torch.cat(
        [
            torch.zeros(video_duration, dtype=torch.long, device=latent_visual.device),
            torch.ones(1, dtype=torch.long, device=latent_visual.device),
        ]
    ).repeat(batch_size)
    return latent_visual, token_types, token_types == 0


@torch.no_grad()
def denoise_loop(  # noqa: PLR0912, PLR0913, PLR0915
    bundle: LatentBundle,
    dit: nn.Module,
    cfg_model: CFGParallelMixin,
    text_embeds: TextEmbeds,
    null_text_embeds: TextEmbeds | list[TextEmbeds | None],
    visual_rope: Tensor | None,
    audio_rope: Tensor | None,
    text_rope: Tensor | list[Tensor],
    null_text_rope: Tensor | list[Tensor | None],
    num_steps: int,
    guidance_weight: float,
    scheduler: Any,
    first_frames: Tensor | None = None,
    visual_cond_scheme: str = "pretrain",
    sparse_params: dict | None = None,
    sample_video: bool = True,
    sample_audio: bool = True,
    *,
    attention_mask: Tensor | None = None,
    null_attention_mask: Tensor | None = None,
    visual_token_type_ids: Tensor | None = None,
    recompute_ropes_each_step: bool = False,
    scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    progress_callback=None,
) -> LatentBundle:
    """Single-device denoising loop for T2V(A) driven by a vLLM-Omni scheduler.

    RoPE tensors are normally precomputed by the caller (pipeline). Set
    ``recompute_ropes_each_step=True`` to rebuild them every DiT forward (A/B
    vs pipeline cache; not for production).
    """
    video = bundle.video  # (sum_T, H, W, C) or (sum_T, H, W, C+extra) for instruct
    audio = bundle.audio  # (sum_A, audio_dim) or None
    is_multimodal = video is not None and audio is not None

    device = video.device if video is not None else audio.device
    guidance_epsilon = 1e-6
    scheduler.set_timesteps(num_steps, device=device)
    step_timesteps = scheduler.timesteps
    if step_timesteps.shape[0] != num_steps:
        raise ValueError(
            "vLLM-Omni scheduler returned an unexpected number of timesteps: "
            f"expected {num_steps}, got {step_timesteps.shape[0]}"
        )

    bs = (
        bundle.video_cu_seqlens.shape[0] - 1
        if bundle.video_cu_seqlens is not None
        else bundle.audio_cu_seqlens.shape[0] - 1
    )

    null_te, null_pe, null_rope = _resolve_null_embeds(null_text_embeds, null_text_rope)

    out_c: int | None = None  # determined after first step, used to strip instruct channels
    raw = _raw_dit(dit)
    vis_shape = (
        (int(visual_rope.shape[0]), int(visual_rope.shape[1]), int(visual_rope.shape[2]))
        if visual_rope is not None
        else None
    )
    scale = (float(scale_factor[0]), float(scale_factor[1]), float(scale_factor[2]))
    tail_cond = visual_cond_scheme == "tail_cond_first_frame"
    ref_positions = bundle.video_cu_seqlens[1:] - 1 if tail_cond and bundle.video_cu_seqlens is not None else None

    def _cache_scope(name: str):
        cache_context = getattr(dit, "cache_context", None)
        return cache_context(name) if callable(cache_context) else nullcontext()

    for step_index, t in enumerate(step_timesteps):
        # The scheduler's `timesteps` are already model-scale (sigma * 1000).
        t_step: Tensor | list[Tensor] = t.unsqueeze(0).expand(bs)  # (bs,)

        model_input_v = (
            _build_video_input(
                video,
                dit.visual_cond,
                first_frames,
                bundle.video_cu_seqlens,
                visual_cond_scheme,
            )
            if video is not None
            else None
        )

        # Freeze one modality at t=0 for partial sampling (T2VA only)
        if is_multimodal:
            t_frozen = scheduler.sigmas[-1].unsqueeze(0).expand(bs) * 1000
            if not sample_audio:
                t_step = [t_step, t_frozen]
            elif not sample_video:
                t_step = [t_frozen, t_step]

        def _forward(
            te,
            pe,
            rope,
            attn_mask,
            *,
            _model_input_v=model_input_v,
            _audio=audio,
            _t_step=t_step,
        ):
            vr, ar, tr = visual_rope, audio_rope, rope
            if recompute_ropes_each_step:
                if vis_shape is not None:
                    vr = compute_visual_rope(raw.visual_rope_embeddings, vis_shape, scale)
                if audio_rope is not None:
                    ar = compute_rope1d(raw.audio_rope_embeddings, int(audio_rope.shape[0]))
                tr = _rebuild_text_rope(raw, rope)
            return dit(
                x_video=_model_input_v,
                x_audio=_audio,
                text_embed=te,
                pooled_text_embed=pe,
                time=_t_step,
                visual_rope=vr,
                audio_rope=ar,
                text_rope=tr,
                sparse_params=sparse_params,
                attention_mask=attn_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        cache = getattr(raw, "_k6_step_cache", None)
        if cache is not None and cache.should_skip(step_index):
            vel_cond, vel_uncond = cache.last_cond, cache.last_uncond
        else:
            use_cfg = abs(guidance_weight - 1.0) > guidance_epsilon
            guided = cfg_model.predict_noise_maybe_with_cfg(
                do_true_cfg=use_cfg,
                true_cfg_scale=guidance_weight,
                positive_kwargs={
                    "forward": _forward,
                    "text_embeds": text_embeds["text_embeds"],
                    "pooled": text_embeds["pooled_embed"],
                    "rope": text_rope,
                    "attn_mask": attention_mask,
                    "cache_scope": _cache_scope("cond"),
                },
                negative_kwargs={
                    "forward": _forward,
                    "text_embeds": null_te,
                    "pooled": null_pe,
                    "rope": null_rope,
                    "attn_mask": null_attention_mask,
                    "cache_scope": _cache_scope("uncond"),
                },
                cfg_normalize=False,
            )
            # The mixin already applied CFG. Storing the guided velocity on
            # both slots keeps a later apply_cfg call as the identity.
            vel_cond = guided
            vel_uncond = guided
            if cache is not None:
                cache.store(vel_cond, vel_uncond)

        # Step per modality. A single scheduler step is consumed for each
        # denoising iteration; when both modalities are sampled, the same
        # scalar step size (derived from `scheduler.sigmas`, not a second
        # stateful `.step()` call) is applied to the second modality so the
        # scheduler's internal step-index bookkeeping stays in sync with the
        # loop counter.
        if isinstance(vel_cond, tuple):
            vel_v, vel_a = vel_cond
            uvel_v, uvel_a = vel_uncond if vel_uncond is not vel_cond else (vel_v, vel_a)
            if out_c is None:
                out_c = vel_v.shape[-1]
            if video is not None and sample_video:
                guided_v = apply_cfg(vel_v, uvel_v, guidance_weight)
                video = scheduler.step(guided_v, t, video, return_dict=False)[0]
                if tail_cond and first_frames is not None and ref_positions is not None:
                    video[ref_positions] = first_frames.to(device=device, dtype=video.dtype)
            if audio is not None and sample_audio:
                guided_a = apply_cfg(vel_a, uvel_a, guidance_weight)
                if not (video is not None and sample_video):
                    audio = scheduler.step(guided_a, t, audio, return_dict=False)[0]
                else:
                    step_size = scheduler.sigmas[step_index + 1] - scheduler.sigmas[step_index]
                    audio += step_size.to(device=audio.device, dtype=audio.dtype) * guided_a
        else:
            if out_c is None:
                out_c = vel_cond.shape[-1]
            vel = apply_cfg(vel_cond, vel_uncond, guidance_weight)
            if video is not None and sample_video:
                video = scheduler.step(vel, t, video, return_dict=False)[0]
                if tail_cond and first_frames is not None and ref_positions is not None:
                    video[ref_positions] = first_frames.to(device=device, dtype=video.dtype)
            elif audio is not None and sample_audio:
                audio = scheduler.step(vel, t, audio, return_dict=False)[0]

        if progress_callback is not None:
            progress_callback()

    # I2V / I2VA: keep injected reference frames unchanged
    if first_frames is not None and video is not None and bundle.video_cu_seqlens is not None:
        ff = first_frames.to(device=device, dtype=video.dtype)
        if visual_cond_scheme == "i2v":
            video[bundle.video_cu_seqlens[:-1]] = ff
        elif tail_cond and ref_positions is not None:
            video[ref_positions] = ff

    # Strip instruct extra-channel padding from the video latent
    if out_c is not None and video is not None:
        video = video[..., :out_c]

    return LatentBundle(
        video=video,
        audio=audio,
        video_cu_seqlens=bundle.video_cu_seqlens,
        audio_cu_seqlens=bundle.audio_cu_seqlens,
    )


"""vLLM-Omni native pipeline for Kandinsky 6 TI2VA.

The port assembler extracts this module's classes/functions into the
generated vLLM-Omni pipeline (``inline_module``). Unlike the Diffusers port
(whose pipeline class only needs to satisfy ``DiffusionPipeline`` and can
rely on ``DiffusionPipeline.from_pretrained()`` for generic component
loading), a vLLM-Omni native pipeline is constructed directly by
``DiffusionModelRegistry`` from an ``OmniDiffusionConfig`` and must own its
own component loading (``_load_components``), request-batch parsing
(``forward``), and ``load_weights``. This mirrors the shape of
vLLM-Omni's other native pipelines (e.g. SanaVideoPipeline,
MiniMaxH3Pipeline): ``encode_prompt`` / ``prepare_latents`` / ``diffuse`` /
``forward`` methods, ``SupportsComponentDiscovery`` for offload/sharding,
and ``SupportAudioOutput`` for the joint video+audio output path.

Scope note: weights are loaded per Hub component after construction.
``weights_sources`` points at ``transformer/``, ``vae/``, ``text_encoder/``,
``text_encoder_2/``, and ``audio_vae/`` in
``kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers``. ``load_weights`` maps those
prefixes onto the pipeline modules. Tokenizers and the scheduler are not
weight tensors; they are still read from their subfolders here.
"""

# The assembler inlines these functions/classes from core/algo, core/types,
# and the vLLM overrides package into the same generated module. They are
# intentionally resolved there.
# ruff: noqa: F821


_PROMPT_TEMPLATE = "\n".join(
    [
        "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
        "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
        "Describe the location of the video, main characters or objects and their action.",
        "Describe the dynamism of the video and presented actions.",
        "Name the visual style of the video: whether it is a professional footage, user generated content, "
        "some kind of animation, video game or scren content.",
        "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
        "Pay attention to the order of key actions shown in the scene.<|im_end|>",
        "<|im_start|>user\n{}<|im_end|>",
    ]
)
_QWEN_CROP_START = 129
_CLIP_MAX_LENGTH = 77
_DEFAULT_NEGATIVE_PROMPT = (
    "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
    "worst quality, low quality, ugly, deformed, walking backwards"
)


_TRANSFORMER_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"


_WEIGHT_SUBFOLDERS = (
    ("transformer", "transformer."),
    ("vae", "vae."),
    ("text_encoder", "text_encoder."),
    ("text_encoder_2", "text_encoder_2."),
    ("audio_vae", "audio_vae."),
)


def _resolve_bundle_root(model: str) -> str:
    """Return a local Diffusers bundle directory for a path or Hub repo id."""
    if os.path.isdir(model):
        return model
    from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

    return download_weights_from_hf_specific(
        model_name_or_path=model,
        cache_dir=None,
        allow_patterns=["*"],
        require_all=True,
    )


def _component_sources(model_root: str, *, include_audio_vae: bool) -> list:
    """One loader source per weight folder in the Diffusers repo."""
    sources = []
    for subfolder, prefix in _WEIGHT_SUBFOLDERS:
        if subfolder == "audio_vae" and not include_audio_vae:
            continue
        if not os.path.isdir(os.path.join(model_root, subfolder)):
            continue
        sources.append(
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_root,
                subfolder=subfolder,
                revision=None,
                prefix=prefix,
                fall_back_to_pt=True,
            )
        )
    return sources


def _adapt_k6_weight_name(name: str) -> str:
    """Map Hub component keys onto this pipeline's module tree.

    ``text_encoder/`` is Qwen2.5-VL saved with the language stack at
    ``model.layers`` and the vision tower at ``visual.*``. This transformers
    build nests those under ``model.language_model`` and ``model.visual``.
    ``audio_vae/`` stores MMAudio keys at the root (``vae.*``, ``vocoder.*``,
    ``mel_converter.*``). ``vae`` and ``vocoder`` live under ``native.tod``;
    ``mel_converter`` lives on ``native``.
    """
    if name.startswith("transformer."):
        # Current Pro-5s snapshots use Diffusers names: encoder ``attn``,
        # ``FeedForward.net``, and ``TimestepEmbedding.linear_*``. Older
        # snapshots still use ``videoT`` / ``audioT`` and ``in_layer``.
        rest = name[len("transformer.") :]
        rest = rest.replace(".videoT.", ".video_dec_block.").replace(".audioT.", ".audio_dec_block.")
        rest = rest.replace(".attn.", ".self_attention.")
        rest = rest.replace(".feed_forward.net.0.proj.", ".feed_forward.in_layer.")
        rest = rest.replace(".feed_forward.net.2.", ".feed_forward.out_layer.")
        rest = rest.replace(".timestep_embedder.linear_1.", ".in_layer.")
        rest = rest.replace(".timestep_embedder.linear_2.", ".out_layer.")
        return "transformer." + rest
    if name.startswith("text_encoder."):
        rest = name[len("text_encoder.") :]
        if rest.startswith("visual."):
            return "text_encoder.model." + rest
        if rest.startswith("model.") and not rest.startswith(("model.language_model.", "model.visual.")):
            return "text_encoder.model.language_model." + rest[len("model.") :]
        return name
    if not name.startswith("audio_vae."):
        return name
    rest = name[len("audio_vae.") :]
    if rest.startswith("native."):
        return name
    if rest.startswith(("vae.", "vocoder.")):
        rest = f"tod.{rest}"
    return f"audio_vae.native.{rest}"


def _build_audio_vae(audio_vae_dir: str) -> nn.Module:
    with open(os.path.join(audio_vae_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    return Kandinsky6AudioVAE(
        vocoder_config=config.get("vocoder_config"),
        mode=str(config.get("mode", "44k")),
        need_vae_encoder=bool(config.get("need_vae_encoder", True)),
        need_vae_decoder=bool(config.get("need_vae_decoder", True)),
        scaling_factor=float(config.get("scaling_factor", 1.0)),
    )


def _shard_loaded_weight(param: nn.Parameter, loaded_weight: Tensor) -> Tensor:
    """Slice a full checkpoint tensor the way ``ColumnParallelLinear`` /
    ``RowParallelLinear.weight_loader`` would, without requiring the
    parameter storage to already be allocated.

    Column weights/biases carry ``output_dim``; row weights carry
    ``input_dim``. Row biases are replicated (shapes already match).
    """
    if getattr(param, "is_sharded_weight", False):
        return loaded_weight
    loader = getattr(param, "weight_loader", None)
    owner = getattr(loader, "__self__", None)
    tp_rank = int(getattr(owner, "tp_rank", 0) or 0)
    input_dim = getattr(param, "input_dim", None)
    output_dim = getattr(param, "output_dim", None)
    if (
        input_dim is not None
        and loaded_weight.ndim > input_dim
        and loaded_weight.shape[input_dim] != param.shape[input_dim]
    ):
        shard = param.shape[input_dim]
        loaded_weight = loaded_weight.narrow(input_dim, tp_rank * shard, shard)
    elif (
        output_dim is not None
        and loaded_weight.ndim > output_dim
        and loaded_weight.shape[output_dim] != param.shape[output_dim]
    ):
        shard = param.shape[output_dim]
        loaded_weight = loaded_weight.narrow(output_dim, tp_rank * shard, shard)
    if not loaded_weight.is_contiguous():
        loaded_weight = loaded_weight.contiguous()
    return loaded_weight


def get_kandinsky6_post_process_func(od_config: OmniDiffusionConfig):
    """Factory returning the post-process function registered in
    ``vllm_omni/diffusion/registry.py``'s ``_DIFFUSION_POST_PROCESS_FUNCS``.

    Unpacks the ``(video, audio)`` pair set on ``DiffusionOutput.output`` by
    ``Kandinsky6TI2VAPipeline.forward`` into the flat ``{"video": ...,
    "audio": ..., "audio_sample_rate": ..., "fps": ...}`` payload the
    framework's shared audio-output plumbing (``io_support.py`` ->
    ``output_formatter.py`` -> ``media_utils.py`` PyAV muxing) already
    expects — the same shape MiniMax H3's post-process function produces.
    ``output_formatter.normalize_diffusion_postprocess_output`` lifts
    ``audio_sample_rate``/``fps`` from this flat form into metadata; wrapping
    it in a ``{"payload", "metadata"}`` envelope instead would silently drop
    the sample rate (the envelope path only reads
    ``metadata["audio"]["sample_rate"]``) and the MP4 would be muxed at the
    24 kHz default, i.e. ~1.8x too slow.

    ``postprocess_audio`` emits int16 PCM; the muxer (and the OpenAI video
    route) consume float32 in ``[-1, 1]``, so the waveform is rescaled here.
    """
    model_config = dict(getattr(od_config, "model_config", None) or {})
    fps = float(model_config.get("sample_fps", 24.0))

    def post_process_func(output, output_type: str = "np", sampling_params=None):
        video, audio = output["video"], output["audio"]
        payload: dict[str, object] = {"video": video, "fps": fps}
        if audio is not None:
            waveform = audio[0] if isinstance(audio, list) else audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()
            waveform = np.asarray(waveform)
            if np.issubdtype(waveform.dtype, np.integer):
                waveform = waveform.astype(np.float32) / float(np.iinfo(waveform.dtype).max)
            payload["audio"] = waveform.astype(np.float32, copy=False)
            audio_sample_rate = output.get("audio_sample_rate")
            if audio_sample_rate is not None:
                payload["audio_sample_rate"] = int(audio_sample_rate)
        return payload

    return post_process_func


def get_kandinsky6_pre_process_func(od_config: OmniDiffusionConfig):
    """Factory for the request pre-process hook (image-conditioning input)."""

    def pre_process_func(request):
        return request

    return pre_process_func


class Kandinsky6TI2VAPipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
    SupportAudioOutput,
    SupportImageInput,
):
    """vLLM-Omni pipeline for text/image-to-video-and-audio generation with Kandinsky 6.

    Args:
        transformer: Multimodal K6 transformer used to denoise video and
            audio latents (``Kandinsky6Transformer3DModel``).
        vae: Video VAE used to decode generated video latents
            (``AutoencoderKLHunyuanVideo``).
        text_encoder: Qwen2.5-VL text encoder for token-level embeddings.
        audio_vae: Audio VAE used to decode generated audio latents
            (``Kandinsky6AudioVAE``). May be ``None`` only when
            ``sample_audio=False`` is the caller's own default.
        scheduler: Flow-matching scheduler used by the denoising loop
            (``KandinskyFlowMatchScheduler``).
        tokenizer: Qwen2.5-VL processor.
        text_encoder_2: CLIP text encoder for pooled embeddings.
        tokenizer_2: CLIP tokenizer.
    """

    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder", "text_encoder_2"]
    _vae_modules = ["vae", "audio_vae"]
    supports_step_execution = True
    support_audio_output = True
    support_image_input = True
    # Startup warmup is a 512px image-to-video probe. Skip it for the 28B DiT.
    dummy_run_num_frames = 0
    default_num_inference_steps = 50

    def __init__(
        self,
        transformer: nn.Module | None = None,
        vae: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        audio_vae: nn.Module | None = None,
        scheduler: object | None = None,
        tokenizer: object | None = None,
        text_encoder_2: nn.Module | None = None,
        tokenizer_2: object | None = None,
        *,
        scale_factor: tuple[float, float, float] = (1.0, 2.0, 2.0),
        sample_fps: float = 24.0,
        audio_sample_rate: int = 44100,
        audio_downsample_factor: int = 1024,
        max_sequence_length: int = 1024,
        text_token_padding: bool = False,
        od_config: OmniDiffusionConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        if od_config is not None:
            model_config = dict(od_config.model_config or {})
            scale_factor = tuple(float(v) for v in model_config.get("scale_factor", scale_factor))
            sample_fps = float(model_config.get("sample_fps", sample_fps))
            audio_sample_rate = int(model_config.get("audio_sample_rate", audio_sample_rate))
            audio_downsample_factor = int(model_config.get("audio_downsample_factor", audio_downsample_factor))
            max_sequence_length = int(model_config.get("max_sequence_length", max_sequence_length))
            text_token_padding = bool(model_config.get("text_token_padding", text_token_padding))
            (
                transformer,
                vae,
                text_encoder,
                audio_vae,
                scheduler,
                tokenizer,
                text_encoder_2,
                tokenizer_2,
            ) = self._load_components(od_config, prefix)

        if transformer is None or vae is None or text_encoder is None or scheduler is None:
            raise ValueError("Kandinsky6TI2VAPipeline requires transformer, vae, text_encoder, and scheduler.")

        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.audio_vae = audio_vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2

        self.scale_factor = scale_factor
        self.sample_fps = sample_fps
        self.audio_sample_rate = audio_sample_rate
        self.audio_downsample_factor = audio_downsample_factor
        self.max_sequence_length = max_sequence_length
        self.text_token_padding = text_token_padding

        # Keep ``self.device`` on the accelerator even when components were
        # loaded to CPU for the offload backend: latents/RoPE live on the GPU
        # and the sequential-offload hooks move the DiT there on first forward.
        first_parameter = next(self.transformer.parameters(), None)
        if first_parameter is not None and first_parameter.device.type != "cpu":
            self.device = first_parameter.device

        # Layerwise offload streams the DiT block-by-block (~20 GiB footprint
        # instead of the 56 GiB resident bf16 DiT). The block-list names depend
        # on the checkpoint flavour (multimodal TI2VA vs. video-only), so the
        # plan is declared per instance from what the transformer actually has.
        block_attrs = tuple(
            name
            for name in (
                "text_transformer_blocks",
                "video_text_transformer_blocks",
                "audio_text_transformer_blocks",
                "visual_transformer_blocks",
            )
            if isinstance(getattr(_raw_dit(self.transformer), name, None), nn.ModuleList)
        )
        if block_attrs:
            self._offload_plan = OffloadPlan(
                block_attrs={"transformer": block_attrs},
                resident_dit_paths=frozenset({"transformer"}),
            )

        if od_config is not None:
            self.setup_diffusion_pipeline_profiler(
                enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
                profiler_targets=["forward"],
            )

    # ------------------------------------------------------------------
    # Modules are constructed empty. ``weights_sources`` then loads each
    # Hub folder (transformer, vae, text_encoder, text_encoder_2, audio_vae)
    # on its own.
    # ------------------------------------------------------------------

    def _load_components(
        self,
        od_config: OmniDiffusionConfig,
        prefix: str,
    ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module | None, object, object, nn.Module | None, object | None]:
        del prefix
        from diffusers.models.modeling_utils import no_init_weights

        model_root = _resolve_bundle_root(od_config.model)
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model_config = dict(od_config.model_config or {})

        # Keep the empty modules on CPU when offload will own GPU placement.
        # Allocating the bf16 DiT (~56 GiB) plus Qwen (~17 GiB) on the GPU
        # before weights arrive does not fit an 80 GB device.
        load_device = torch.device("cpu") if offload_enabled(od_config) else self.device

        transformer_dir = os.path.join(model_root, "transformer")
        with open(os.path.join(transformer_dir, "config.json"), encoding="utf-8") as handle:
            transformer_config = json.load(handle)
        with torch.device(load_device), no_init_weights():
            transformer = Kandinsky6Transformer3DModel.from_diffusers_config(
                transformer_config,
                quant_config=getattr(od_config, "quantization_config", None),
            )

        vae_dir = os.path.join(model_root, "vae")
        with open(os.path.join(vae_dir, "config.json"), encoding="utf-8") as handle:
            vae_config = json.load(handle)
        with torch.device(load_device), no_init_weights():
            vae = AutoencoderKLHunyuanVideo.from_config(vae_config)
        vae.to(dtype=torch.float16)

        text_encoder_dir = os.path.join(model_root, "text_encoder")
        text_encoder_config = AutoConfig.from_pretrained(text_encoder_dir)
        with torch.device(load_device), no_init_weights():
            text_encoder = Qwen2_5_VLForConditionalGeneration(text_encoder_config)
        text_encoder.to(dtype=dtype)

        # Tokenizer/processor files live in the sibling `tokenizer/` directory
        # (standard Diffusers multi-component layout), not inside
        # `text_encoder/` itself — that dir only has the model weights/config.
        tokenizer = QwenAutoProcessor.from_pretrained(os.path.join(model_root, "tokenizer"))
        # encode_prompt() crops embeddings at a fixed offset (_QWEN_CROP_START)
        # assuming the fixed system-prompt prefix leads and any padding trails
        # — don't trust the checkpoint's own tokenizer_config.json for this;
        # pin it explicitly like every sibling pipeline that does fixed-offset
        # crops does (sana_video, sana_wm, boogu_image).
        tokenizer.tokenizer.padding_side = "right"

        clip_dir = os.path.join(model_root, "text_encoder_2")
        clip_config = AutoConfig.from_pretrained(clip_dir)
        with torch.device(load_device), no_init_weights():
            text_encoder_2 = CLIPTextModel(clip_config)
        text_encoder_2.to(dtype=dtype)
        tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(model_root, "tokenizer_2"))

        audio_vae = None
        audio_vae_dir = os.path.join(model_root, "audio_vae")
        include_audio_vae = bool(model_config.get("sample_audio", True)) and os.path.isdir(audio_vae_dir)
        if include_audio_vae:
            with torch.device(load_device), no_init_weights():
                audio_vae = _build_audio_vae(audio_vae_dir)
            audio_vae.to(dtype=dtype)

        self.weights_sources = _component_sources(model_root, include_audio_vae=include_audio_vae)

        # model_config never actually carries a "scheduler_scale" key in
        # practice (nothing populates it from the checkpoint), so this always
        # fell back to a hardcoded 3.0 — wrong for every known K6 checkpoint,
        # which persist their real shift as scheduler/scheduler_config.json's
        # "shift" key (same convention lingbot_world's pipeline already reads
        # at vllm_omni/diffusion/models/lingbot_world/pipeline.py:504).
        checkpoint_scheduler_scale = 5.0  # matches every known K6 generation config and kandinsky-5's own CLI default
        scheduler_config_path = os.path.join(model_root, "scheduler", "scheduler_config.json")
        if os.path.isfile(scheduler_config_path):
            checkpoint_scheduler_scale = float(
                _load_json(model_root, "scheduler/scheduler_config.json").get("shift", checkpoint_scheduler_scale)
            )
        scheduler_scale = float(model_config.get("scheduler_scale", checkpoint_scheduler_scale))
        scheduler = KandinskyFlowMatchScheduler(scheduler_scale=scheduler_scale, device=self.device)

        return transformer, vae, text_encoder, audio_vae, scheduler, tokenizer, text_encoder_2, tokenizer_2

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Load each Hub folder into its module.

        The framework yields ``transformer.*``, ``vae.*``, ``text_encoder.*``,
        ``text_encoder_2.*``, and ``audio_vae.*`` from ``weights_sources``.
        """
        from vllm.model_executor.models.utils import AutoWeightsLoader, is_pp_missing_parameter

        quant_config = getattr(self.od_config, "quantization_config", None)

        def adapted():
            for name, tensor in weights:
                if name.startswith("transformer."):
                    key = name[len("transformer.") :]
                    if is_pp_missing_parameter(key, self.transformer):
                        continue
                    if quant_config is not None and tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
                        tensor = tensor.to(torch.bfloat16)
                name = _adapt_k6_weight_name(name)
                yield name, tensor

        return AutoWeightsLoader(self).load_weights(adapted())

    # ------------------------------------------------------------------
    # Prompt encoding — same Qwen2.5-VL + CLIP dual-encoder logic as the
    # native/Diffusers ports (framework-agnostic; no vLLM-specific change).
    # ------------------------------------------------------------------

    @staticmethod
    def _as_text_embeds(value) -> TextEmbeds:
        if not isinstance(value, dict):
            raise TypeError("prompt embeddings must be a mapping with text_embeds and pooled_embed")
        missing = {"text_embeds", "pooled_embed"} - set(value)
        if missing:
            raise ValueError(f"prompt embeddings are missing: {sorted(missing)}")
        return value

    def encode_prompt(self, text: str) -> tuple[TextEmbeds, Tensor, Tensor | None]:
        full_text = _PROMPT_TEMPLATE.format(text)
        # Inputs go to the execution device, not the encoder's current one:
        # under sequential offload the encoders rest on CPU between requests
        # and are swapped onto the GPU by the hook when their forward runs.
        qwen_device = self.device
        inputs = self.tokenizer(
            text=[full_text],
            images=None,
            videos=None,
            max_length=self.max_sequence_length + _QWEN_CROP_START,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(qwen_device)
        qwen_output = self.text_encoder(
            input_ids=inputs["input_ids"],
            return_dict=True,
            output_hidden_states=True,
        )
        embeds = qwen_output["hidden_states"][-1][:, _QWEN_CROP_START:]
        attention = inputs["attention_mask"][:, _QWEN_CROP_START:].to(dtype=torch.bool)
        if self.text_token_padding:
            qwen_embeds = embeds[0]
            qwen_attention = attention[0]
            cu_seqlens = torch.tensor([0, qwen_embeds.shape[0]], dtype=torch.int32, device=qwen_embeds.device)
        else:
            qwen_embeds = embeds[attention]
            qwen_attention = None
            cu_seqlens = torch.tensor([0, int(attention.sum().item())], dtype=torch.int32, device=embeds.device)

        clip_device = self.device
        clip_inputs = self.tokenizer_2(
            [text],
            max_length=_CLIP_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(clip_device)
        pooled_embed = self.text_encoder_2(**clip_inputs)["pooler_output"]
        return {"text_embeds": qwen_embeds, "pooled_embed": pooled_embed}, cu_seqlens, qwen_attention

    def _encode_prompt_pair(
        self, prompt: str, negative_prompt: str
    ) -> tuple[TextEmbeds, Tensor, Tensor | None, TextEmbeds, Tensor, Tensor | None]:
        prompt_embeds, prompt_cu, prompt_mask = self.encode_prompt(prompt)
        negative_embeds, negative_cu, negative_mask = self.encode_prompt(negative_prompt)
        return prompt_embeds, prompt_cu, prompt_mask, negative_embeds, negative_cu, negative_mask

    @staticmethod
    def _move_text_embeds(
        embeds: TextEmbeds,
        cu_seqlens: Tensor,
        attention_mask: Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[TextEmbeds, Tensor, Tensor | None]:
        moved = {key: value.to(device=device, dtype=dtype) for key, value in embeds.items()}
        mask = attention_mask.to(device=device) if attention_mask is not None else None
        return moved, cu_seqlens.to(device=device), mask

    def _text_ropes(
        self,
        raw_dit: nn.Module,
        *,
        text_length: int,
        negative_text_length: int,
        audio_length: int | None,
        device: torch.device,
    ) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor], Tensor | None]:
        if getattr(raw_dit, "is_multimodal", False):
            text_rope = [
                compute_rope1d(raw_dit.video_text_rope_embeddings, text_length, device=device),
                compute_rope1d(raw_dit.audio_text_rope_embeddings, text_length, device=device),
            ]
            negative_text_rope = [
                compute_rope1d(raw_dit.video_text_rope_embeddings, negative_text_length, device=device),
                compute_rope1d(raw_dit.audio_text_rope_embeddings, negative_text_length, device=device),
            ]
        else:
            text_rope = compute_rope1d(raw_dit.text_rope_embeddings, text_length, device=device)
            negative_text_rope = compute_rope1d(raw_dit.text_rope_embeddings, negative_text_length, device=device)
        audio_rope = (
            compute_rope1d(raw_dit.audio_rope_embeddings, audio_length, device=device)
            if audio_length is not None
            else None
        )
        return text_rope, negative_text_rope, audio_rope

    # ------------------------------------------------------------------
    # Latent preparation
    # ------------------------------------------------------------------

    def prepare_latents(
        self,
        *,
        latent_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        sample_audio: bool,
    ) -> LatentBundle:
        raw_dit = self.transformer
        channels = int(getattr(raw_dit, "in_visual_dim", 16))
        bundle = prepare_video_latents(
            bs=1,
            duration=latent_frames,
            H_lat=height // 8,
            W_lat=width // 8,
            C=channels,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        if not sample_audio:
            return bundle
        if self.audio_vae is None:
            raise ValueError("sample_audio=True requires an audio_vae")

        audio_dim = int(getattr(raw_dit, "in_audio_dim", 20))
        downsample_factor = int(getattr(self.audio_vae, "downsample_factor", self.audio_downsample_factor))
        audio_duration = audio_latent_duration(
            latent_frames,
            fps=self.sample_fps,
            audio_fps=self.audio_sample_rate,
            downsample_factor=downsample_factor,
        )
        return prepare_audio_latents(
            bundle,
            audio_duration=audio_duration,
            audio_dim=audio_dim,
            seed=seed,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Denoise
    # ------------------------------------------------------------------

    def predict_noise(
        self,
        *,
        forward,
        text_embeds,
        pooled,
        rope,
        attn_mask,
        cache_scope,
    ):
        """One CFG branch. ``forward`` is the denoise loop's DiT call."""
        with cache_scope:
            return forward(text_embeds, pooled, rope, attn_mask)

    def diffuse(
        self,
        *,
        bundle: LatentBundle,
        positive: TextEmbeds,
        negative: TextEmbeds,
        positive_cu: Tensor,
        negative_cu: Tensor,
        positive_mask: Tensor | None,
        negative_mask: Tensor | None,
        visual_rope: Tensor,
        audio_rope: Tensor | None,
        num_inference_steps: int,
        guidance_scale: float,
        first_frames: Tensor | None,
        visual_cond_scheme: str,
        sample_audio: bool,
        visual_token_type_ids: Tensor | None,
        device: torch.device,
    ) -> LatentBundle:
        text_rope, negative_text_rope, resolved_audio_rope = self._text_ropes(
            self.transformer,
            text_length=int(positive_cu[-1].item()),
            negative_text_length=int(negative_cu[-1].item()),
            audio_length=int(bundle.audio_cu_seqlens[-1].item())
            if bundle.audio is not None and bundle.audio_cu_seqlens is not None
            else None,
            device=device,
        )
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            return denoise_loop(
                bundle=bundle,
                dit=self.transformer,
                cfg_model=self,
                text_embeds=positive,
                null_text_embeds=negative,
                visual_rope=visual_rope,
                audio_rope=resolved_audio_rope,
                text_rope=text_rope,
                null_text_rope=negative_text_rope,
                num_steps=num_inference_steps,
                guidance_weight=guidance_scale,
                scheduler=self.scheduler,
                first_frames=first_frames,
                visual_cond_scheme=visual_cond_scheme,
                sample_video=True,
                sample_audio=sample_audio,
                attention_mask=positive_mask,
                null_attention_mask=negative_mask,
                visual_token_type_ids=visual_token_type_ids,
                scale_factor=self.scale_factor,
                progress_callback=progress_bar.update,
            )

    # ------------------------------------------------------------------
    # Request entrypoint
    # ------------------------------------------------------------------

    def __call__(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        return self.forward(req)

    def _guided_velocity(self, state: Any) -> Tensor | tuple:
        """One CFG-combined DiT velocity. Audio, when present, is the second element."""
        extra = state.extra
        step_index = int(state.step_index)
        scheduler = state.scheduler
        t = state.current_timestep
        video = state.latents
        audio = extra["audio"]
        bundle = extra["bundle"]
        dit = self.transformer
        raw = _raw_dit(dit)
        bs = int(extra["bs"])
        guidance_weight = float(extra["guidance_scale"])
        sample_audio = bool(extra["sample_audio"])
        is_multimodal = video is not None and audio is not None
        t_step: Tensor | list[Tensor] = t.unsqueeze(0).expand(bs)
        model_input_v = (
            _build_video_input(
                video,
                dit.visual_cond,
                extra["first_frames"],
                bundle.video_cu_seqlens,
                extra["visual_cond_scheme"],
            )
            if video is not None
            else None
        )
        if is_multimodal:
            t_frozen = scheduler.sigmas[-1].unsqueeze(0).expand(bs) * 1000
            if not sample_audio:
                t_step = [t_step, t_frozen]
            elif not extra["sample_video"]:
                t_step = [t_frozen, t_step]

        def _cache_scope(name: str):
            cache_context = getattr(dit, "cache_context", None)
            return cache_context(name) if callable(cache_context) else nullcontext()

        def _forward(te, pe, rope, attn_mask, *, _model_input_v=model_input_v, _audio=audio, _t_step=t_step):
            return dit(
                x_video=_model_input_v,
                x_audio=_audio,
                text_embed=te,
                pooled_text_embed=pe,
                time=_t_step,
                visual_rope=extra["visual_rope"],
                audio_rope=extra["audio_rope"],
                text_rope=rope,
                sparse_params=extra["sparse_params"],
                attention_mask=attn_mask,
                visual_token_type_ids=extra["visual_token_type_ids"],
            )

        cache = getattr(raw, "_k6_step_cache", None)
        if cache is not None and cache.should_skip(step_index):
            return cache.last_cond
        positive = extra["positive"]
        guided = self.predict_noise_maybe_with_cfg(
            do_true_cfg=bool(state.do_true_cfg),
            true_cfg_scale=guidance_weight,
            positive_kwargs={
                "forward": _forward,
                "text_embeds": positive["text_embeds"],
                "pooled": positive["pooled_embed"],
                "rope": extra["text_rope"],
                "attn_mask": extra["positive_mask"],
                "cache_scope": _cache_scope("cond"),
            },
            negative_kwargs={
                "forward": _forward,
                "text_embeds": extra["null_te"],
                "pooled": extra["null_pe"],
                "rope": extra["null_rope"],
                "attn_mask": extra["negative_mask"],
                "cache_scope": _cache_scope("uncond"),
            },
            cfg_normalize=False,
        )
        if cache is not None:
            cache.store(guided, guided)
        return guided

    def prepare_encode(self, state: Any, **kwargs: Any) -> Any:
        """Encode one request and store everything the denoise step needs."""
        del kwargs
        if state.prompt is None:
            raise ValueError("Prompt is required for Kandinsky 6 generation.")
        prompt_obj = state.prompt
        prompt = prompt_obj if isinstance(prompt_obj, str) else (prompt_obj.get("prompt") or "")
        negative_prompt = "" if isinstance(prompt_obj, str) else (prompt_obj.get("negative_prompt") or "")
        image = None
        if not isinstance(prompt_obj, str):
            image = (prompt_obj.get("multi_modal_data") or {}).get("image")
            if image is None:
                image = prompt_obj.get("image")
            if isinstance(image, (list, tuple)):
                image = image[0] if image else None
        sampling = state.sampling
        _raw_dit(self.transformer).clear_text_proj_cache()
        height = sampling.height or 512
        width = sampling.width or 768
        num_frames = sampling.num_frames or 121
        num_inference_steps = sampling.num_inference_steps or self.default_num_inference_steps
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 5.0
        extra_args = sampling.extra_args or {}
        sample_audio = bool(extra_args.get("sample_audio", True))
        # The Diffusers TI2VA pipeline always appends a reference image as a masked tail frame.
        visual_cond_scheme = "tail_cond_first_frame" if image else "pretrain"
        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)
        seed = (
            int(torch.randint(0, 2**31, (1,), generator=generator, device=generator.device).item())
            if generator is not None
            else int(torch.randint(0, 2**31, (1,)).item())
        )
        device = self.device
        dtype = getattr(self.transformer, "dtype", torch.bfloat16)
        if not isinstance(dtype, torch.dtype) or dtype.itemsize < 2:
            dtype = torch.bfloat16
        raw_dit = _raw_dit(self.transformer)
        patch_size = tuple(int(v) for v in getattr(raw_dit, "patch_size", (1, 2, 2)))
        first_frames = None
        if image is not None:
            first_frames, height, width = encode_i2va_first_frame(image, self.vae, device, height=height, width=width)
        negative_prompt = negative_prompt or _DEFAULT_NEGATIVE_PROMPT
        positive, positive_cu, positive_mask, negative, negative_cu, negative_mask = self._encode_prompt_pair(
            prompt, negative_prompt
        )
        positive, positive_cu, positive_mask = self._move_text_embeds(
            positive, positive_cu, positive_mask, device=device, dtype=dtype
        )
        negative, negative_cu, negative_mask = self._move_text_embeds(
            negative, negative_cu, negative_mask, device=device, dtype=dtype
        )
        latent_frames = (num_frames - 1) // 4 + 1
        bundle = self.prepare_latents(
            latent_frames=latent_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            seed=seed,
            sample_audio=sample_audio,
        )
        visual_token_type_ids = None
        generated_visual_mask = None
        if image is not None and visual_cond_scheme == "tail_cond_first_frame":
            video, visual_token_type_ids, generated_visual_mask = append_i2va_tail_condition(
                bundle.video, first_frames, batch_size=1, video_duration=latent_frames
            )
            bundle = LatentBundle(
                video=video,
                audio=bundle.audio,
                video_cu_seqlens=torch.tensor([0, latent_frames + 1], dtype=torch.int32, device=device),
                audio_cu_seqlens=bundle.audio_cu_seqlens,
            )
        visual_shape = (
            latent_frames // patch_size[0],
            (height // 8) // patch_size[1],
            (width // 8) // patch_size[2],
        )
        visual_rope = compute_visual_rope(
            raw_dit.visual_rope_embeddings, visual_shape, self.scale_factor, device=device
        )
        if generated_visual_mask is not None:
            visual_rope = torch.cat([visual_rope, visual_rope[:1]], dim=0)
        scheduler = KandinskyFlowMatchScheduler(
            scheduler_scale=self.scheduler.scheduler_scale,
            device=device,
        )
        scheduler.set_timesteps(num_inference_steps, device=device)
        text_rope, null_rope, audio_rope = self._text_ropes(
            raw_dit,
            text_length=int(positive_cu[-1].item()),
            negative_text_length=int(negative_cu[-1].item()),
            audio_length=int(bundle.audio_cu_seqlens[-1].item())
            if bundle.audio is not None and bundle.audio_cu_seqlens is not None
            else None,
            device=device,
        )
        null_te, null_pe, null_rope = _resolve_null_embeds(negative, null_rope)
        state.latents = bundle.video if bundle.video is not None else bundle.audio
        state.timesteps = scheduler.timesteps
        state.step_index = 0
        state.scheduler = scheduler
        state.do_true_cfg = abs(guidance_scale - 1.0) > 1e-6
        state.extra = {
            "bundle": bundle,
            "audio": bundle.audio,
            "positive": positive,
            "positive_mask": positive_mask,
            "negative_mask": negative_mask,
            "null_te": null_te,
            "null_pe": null_pe,
            "null_rope": null_rope,
            "text_rope": text_rope,
            "visual_rope": visual_rope,
            "audio_rope": audio_rope,
            "sparse_params": None,
            "first_frames": first_frames,
            "visual_cond_scheme": visual_cond_scheme,
            "visual_token_type_ids": visual_token_type_ids,
            "generated_visual_mask": generated_visual_mask,
            "sample_audio": sample_audio,
            "sample_video": True,
            "guidance_scale": guidance_scale,
            "bs": 1,
            "latent_frames": latent_frames,
            "audio_normalization": str(extra_args.get("audio_normalization", "normalize")),
            "audio_velocity": None,
            "out_c": None,
        }
        return state

    def denoise_step(self, input_batch: Any, **kwargs: Any) -> Tensor | None:
        """One CFG-combined video velocity. Audio velocity is stashed on the request."""
        del input_batch
        states = kwargs.get("states") or ()
        if len(states) != 1:
            raise ValueError("Kandinsky 6 step execution supports one request at a time.")
        state = states[0]
        guided = self._guided_velocity(state)
        if isinstance(guided, tuple):
            video_velocity, audio_velocity = guided
            state.extra["audio_velocity"] = audio_velocity
            state.extra["out_c"] = int(video_velocity.shape[-1])
            return video_velocity
        state.extra["out_c"] = int(guided.shape[-1])
        return guided

    def step_scheduler(self, state: Any, noise_pred: Tensor | None, **kwargs: Any) -> None:
        """Apply one scheduler update to video and, when present, audio."""
        del kwargs
        if noise_pred is None:
            state.step_index += 1
            return
        extra = state.extra
        scheduler = state.scheduler
        t = state.current_timestep
        step_index = int(state.step_index)
        video = state.latents
        scheme = extra["visual_cond_scheme"]
        tail_cond = scheme == "tail_cond_first_frame"
        bundle = extra["bundle"]
        ref_positions = bundle.video_cu_seqlens[1:] - 1 if tail_cond and bundle.video_cu_seqlens is not None else None
        first_frames = extra["first_frames"]
        if extra["sample_video"] and video is not None and video.ndim >= 4:
            video = scheduler.step(noise_pred, t, video, return_dict=False)[0]
            if tail_cond and first_frames is not None and ref_positions is not None:
                video[ref_positions] = first_frames.to(device=video.device, dtype=video.dtype)
            state.latents = video
        audio = extra["audio"]
        audio_velocity = extra.get("audio_velocity")
        if audio is not None and audio_velocity is not None and extra["sample_audio"]:
            if not (video is not None and extra["sample_video"] and video.ndim >= 4):
                extra["audio"] = scheduler.step(audio_velocity, t, audio, return_dict=False)[0]
            else:
                step_size = scheduler.sigmas[step_index + 1] - scheduler.sigmas[step_index]
                extra["audio"] = audio + step_size.to(device=audio.device, dtype=audio.dtype) * audio_velocity
        state.step_index += 1

    def post_decode(self, state: Any, **kwargs: Any) -> DiffusionOutput:
        """Decode the finished video and audio latents."""
        del kwargs
        extra = state.extra
        video = state.latents
        out_c = extra.get("out_c")
        if out_c is not None and video is not None and video.ndim >= 4:
            video = video[..., :out_c]
        mask = extra.get("generated_visual_mask")
        if mask is not None and video is not None:
            video = video[mask]
        device = video.device if video is not None else self.device
        bundle = LatentBundle(
            video=video,
            audio=extra["audio"],
            video_cu_seqlens=torch.tensor([0, extra["latent_frames"]], dtype=torch.int32, device=device),
            audio_cu_seqlens=extra["bundle"].audio_cu_seqlens,
        )
        if state.sampling.output_type == "latent":
            video_out: Tensor | np.ndarray = video.unsqueeze(0)
        else:
            decoded = postprocess_video(bundle, self.vae, bs=1)
            video_out = decoded.permute(0, 2, 3, 4, 1).cpu().numpy()
        audio_out = (
            postprocess_audio(bundle, self.audio_vae, normalization_mode=extra["audio_normalization"])
            if extra["sample_audio"]
            else None
        )
        audio_sample_rate = self.audio_sample_rate if audio_out is not None else None
        return DiffusionOutput(output={"video": video_out, "audio": audio_out, "audio_sample_rate": audio_sample_rate})

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("Kandinsky6TI2VAPipeline currently supports exactly one prompt per request.")
        prompt_obj = req.prompts[0]
        prompt = prompt_obj if isinstance(prompt_obj, str) else (prompt_obj.get("prompt") or "")
        negative_prompt = "" if isinstance(prompt_obj, str) else (prompt_obj.get("negative_prompt") or "")
        image = None
        if not isinstance(prompt_obj, str):
            # The framework's request envelope carries reference images under
            # ``multi_modal_data.image`` (see sana_video / wan2_2 I2V); keep the
            # bare ``image`` key for direct/legacy callers.
            image = (prompt_obj.get("multi_modal_data") or {}).get("image")
            if image is None:
                image = prompt_obj.get("image")
            if isinstance(image, (list, tuple)):
                if len(image) > 1:
                    raise ValueError("Kandinsky6TI2VAPipeline accepts at most one reference image.")
                image = image[0] if image else None
        if not prompt:
            raise ValueError("Prompt is required for Kandinsky 6 generation.")

        # The transformer's text-projection cache is keyed by tensor.data_ptr(),
        # not a value/content key, and is documented as scoped "within a
        # generation" — but nothing ever called clear_text_proj_cache() before
        # this fix, so it persisted for the whole process lifetime. The pooled
        # CLIP embedding's shape never varies, so a freed tensor's CUDA address
        # getting reused by the allocator (e.g. after the startup dummy-run
        # warmup, or the previous request) silently returns stale conditioning
        # from an unrelated prompt. Clear it at the start of every generation.
        _raw_dit(self.transformer).clear_text_proj_cache()

        sampling = req.sampling_params
        height = sampling.height or 512
        width = sampling.width or 768
        num_frames = sampling.num_frames or 121
        num_inference_steps = sampling.num_inference_steps or self.default_num_inference_steps
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 5.0
        extra_args = sampling.extra_args or {}
        sample_audio = bool(extra_args.get("sample_audio", True))
        # ``normalize`` (peak-normalize each waveform) matches the k6_video
        # production pipeline; ``clip`` keeps the raw decoded amplitude.
        audio_normalization = str(extra_args.get("audio_normalization", "normalize"))
        # The Diffusers TI2VA pipeline always appends a reference image as a masked tail frame.
        visual_cond_scheme = "tail_cond_first_frame" if image else "pretrain"

        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)
        seed = (
            int(torch.randint(0, 2**31, (1,), generator=generator, device=generator.device).item())
            if generator is not None
            else int(torch.randint(0, 2**31, (1,)).item())
        )

        device = self.device
        dtype = getattr(self.transformer, "dtype", torch.bfloat16)
        if not isinstance(dtype, torch.dtype) or dtype.itemsize < 2:
            # FP8 weights must not become the latent/activation dtype.
            dtype = torch.bfloat16

        raw_dit = self.transformer
        patch_size = tuple(int(v) for v in getattr(raw_dit, "patch_size", (1, 2, 2)))

        first_frames = None
        if image is not None:
            first_frames, height, width = encode_i2va_first_frame(image, self.vae, device, height=height, width=width)

        negative_prompt = negative_prompt or _DEFAULT_NEGATIVE_PROMPT
        positive, positive_cu, positive_mask, negative, negative_cu, negative_mask = self._encode_prompt_pair(
            prompt, negative_prompt
        )
        positive, positive_cu, positive_mask = self._move_text_embeds(
            positive, positive_cu, positive_mask, device=device, dtype=dtype
        )
        negative, negative_cu, negative_mask = self._move_text_embeds(
            negative, negative_cu, negative_mask, device=device, dtype=dtype
        )

        latent_frames = (num_frames - 1) // 4 + 1
        bundle = self.prepare_latents(
            latent_frames=latent_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            seed=seed,
            sample_audio=sample_audio,
        )

        visual_token_type_ids = None
        generated_visual_mask = None
        if image is not None and visual_cond_scheme == "tail_cond_first_frame":
            video, visual_token_type_ids, generated_visual_mask = append_i2va_tail_condition(
                bundle.video, first_frames, batch_size=1, video_duration=latent_frames
            )
            bundle = LatentBundle(
                video=video,
                audio=bundle.audio,
                video_cu_seqlens=torch.tensor([0, latent_frames + 1], dtype=torch.int32, device=device),
                audio_cu_seqlens=bundle.audio_cu_seqlens,
            )

        visual_shape = (
            latent_frames // patch_size[0],
            (height // 8) // patch_size[1],
            (width // 8) // patch_size[2],
        )
        if visual_shape[0] < 1:
            raise ValueError(f"invalid visual latent shape {visual_shape}")
        visual_rope = compute_visual_rope(
            raw_dit.visual_rope_embeddings, visual_shape, self.scale_factor, device=device
        )
        if generated_visual_mask is not None:
            visual_rope = torch.cat([visual_rope, visual_rope[:1]], dim=0)

        result = self.diffuse(
            bundle=bundle,
            positive=positive,
            negative=negative,
            positive_cu=positive_cu,
            negative_cu=negative_cu,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
            visual_rope=visual_rope,
            audio_rope=None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            first_frames=first_frames,
            visual_cond_scheme=visual_cond_scheme,
            sample_audio=sample_audio,
            visual_token_type_ids=visual_token_type_ids,
            device=device,
        )

        if generated_visual_mask is not None and result.video is not None:
            result = LatentBundle(
                video=result.video[generated_visual_mask],
                audio=result.audio,
                video_cu_seqlens=torch.tensor([0, latent_frames], dtype=torch.int32, device=device),
                audio_cu_seqlens=result.audio_cu_seqlens,
            )

        if sampling.output_type == "latent":
            video_out: Tensor | np.ndarray = result.video.unsqueeze(0)
        else:
            decoded = postprocess_video(result, self.vae, bs=1)
            video_out = decoded.permute(0, 2, 3, 4, 1).cpu().numpy()

        audio_out = (
            postprocess_audio(result, self.audio_vae, normalization_mode=audio_normalization) if sample_audio else None
        )
        audio_sample_rate = self.audio_sample_rate if audio_out is not None else None
        return DiffusionOutput(output={"video": video_out, "audio": audio_out, "audio_sample_rate": audio_sample_rate})


__all__ = [
    "Kandinsky6TI2VAPipeline",
    "get_kandinsky6_post_process_func",
    "get_kandinsky6_pre_process_func",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native MAGI-2 Preview classifier-free-guidance sampling.

The CFG and denoising math is adapted from SandAI's Apache-2.0 MAGI-2
Preview inference implementation.  Model placement is deliberately not
managed here: vLLM-Omni's loader/offloader (including DLO) owns placement,
while this sampler only prepares tensors and invokes the already placed model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler

from .preview_data_proxy import Magi2DataProxy, ModelInput


def _pad_or_trim(
    tensor: torch.Tensor,
    target_size: int,
    dim: int,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, int]:
    current_size = tensor.size(dim)
    if current_size < target_size:
        padding_amount = target_size - current_size
        padding = [0] * (2 * tensor.dim())
        padding_dim_index = tensor.dim() - 1 - dim
        padding[2 * padding_dim_index + 1] = padding_amount
        return F.pad(tensor, tuple(padding), "constant", pad_value), current_size
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(0, target_size)
    return tensor[tuple(slices)], current_size


@dataclass(frozen=True)
class CFGConfig:
    """Preview CFG controls, defaulted to the released MAGI-2 recipe."""

    use_cfg_trick: bool = False
    cfg_trick_start_frame: int = 13
    cfg_trick_value: float = 2.0
    use_dynamic_cfg: bool = False
    dynamic_cfg_start_t: int = 500
    dynamic_cfg_cutoff_value: float = 2.0
    video_txt_guidance_scale: float = 5.0
    audio_txt_guidance_scale: float = 7.0
    # The released I2V config keeps image context in both CFG branches, so CFG
    # changes text conditioning without pushing away from the input image.
    use_ref_for_uncond: bool = True
    use_skimmed_cfg_linear: bool = False
    skimmed_cfg_scale: float = 3.0
    cfg_rescale: float = 0.0

    def __post_init__(self) -> None:
        finite_values = {
            "cfg_trick_value": self.cfg_trick_value,
            "dynamic_cfg_cutoff_value": self.dynamic_cfg_cutoff_value,
            "video_txt_guidance_scale": self.video_txt_guidance_scale,
            "audio_txt_guidance_scale": self.audio_txt_guidance_scale,
            "skimmed_cfg_scale": self.skimmed_cfg_scale,
            "cfg_rescale": self.cfg_rescale,
        }
        for name, value in finite_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.cfg_trick_start_frame < 0:
            raise ValueError("cfg_trick_start_frame must be non-negative")
        if self.skimmed_cfg_scale < 0 or self.cfg_rescale < 0:
            raise ValueError("skimmed_cfg_scale and cfg_rescale must be non-negative")


@dataclass
class SamplerInput:
    video_t_list: Sequence[torch.Tensor]
    audio_t_list: Sequence[torch.Tensor]
    latent: torch.Tensor  # [B,C,T,H,W]
    audio_latent: torch.Tensor  # [B,L,C]
    txt_feat: torch.Tensor  # [B,L,D]
    null_txt_feat: torch.Tensor  # [B,L,D]
    ref_audio_feat: torch.Tensor | None
    ref_video_feat: torch.Tensor | None
    video_scheduler: SchedulerMixin
    audio_scheduler: SchedulerMixin
    cfg_config: CFGConfig
    ref_image_feat: torch.Tensor | None = None  # [B,M,C,1,H,W]
    ref_image_feat_len: torch.Tensor | None = None  # [B,M,2]
    ref_image_special_token_embedding: torch.Tensor | None = None  # [B,M,D]


def build_magi2_preview_schedulers(
    num_inference_steps: int,
    *,
    device: torch.device | str,
    shift: float = 7.0,
) -> tuple[FlowUniPCMultistepScheduler, FlowUniPCMultistepScheduler]:
    """Create independent, identically configured video/audio schedulers."""

    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(f"shift must be positive and finite, got {shift}")
    video_scheduler = FlowUniPCMultistepScheduler()
    audio_scheduler = FlowUniPCMultistepScheduler()
    video_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
    audio_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
    return video_scheduler, audio_scheduler


class Magi2PreviewSampler(CFGParallelMixin):
    """Run MAGI-2 Preview's joint video/audio denoising loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        data_proxy: Magi2DataProxy | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.data_proxy = data_proxy or Magi2DataProxy()
        self.device = torch.device(device) if device is not None else None
        # Retained for the reference-compatible constructor.  Model placement
        # and parameter dtype are owned by the generic loader/offloader.
        self.dtype = dtype

    @torch.inference_mode()
    def forward(self, model_input: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        packed_input = self.data_proxy.process_input(model_input)
        model_output = self.model(*packed_input.model_args)
        if not isinstance(model_output, torch.Tensor):
            raise TypeError(f"MAGI-2 preview transformer must return a tensor, got {type(model_output)!r}")
        return self.data_proxy.process_output(model_output, packed_input.output_layout)

    def predict_noise(self, model_input: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one conditional or unconditional branch for shared CFG dispatch."""

        return self.forward(model_input)

    @torch.inference_mode()
    def sample(self, sampler_input: SamplerInput) -> tuple[torch.Tensor, torch.Tensor]:
        video_timesteps = list(sampler_input.video_t_list)
        audio_timesteps = list(sampler_input.audio_t_list)
        if not video_timesteps:
            raise ValueError("MAGI-2 sampling needs at least one timestep")
        if len(video_timesteps) != len(audio_timesteps):
            raise ValueError("video and audio timestep schedules must have equal lengths")

        video_cfgs, audio_cfgs = self.precalculate_cfg(
            video_timesteps,
            sampler_input.latent.shape[2],
            sampler_input.cfg_config,
            device=sampler_input.latent.device,
        )
        latent = sampler_input.latent.clone()
        audio_latent = sampler_input.audio_latent.clone()

        for timestep, video_cfg, audio_cfg in zip(
            video_timesteps,
            video_cfgs,
            audio_cfgs,
            strict=True,
        ):
            model_input = self.prepare_model_input(
                latent=latent,
                audio_latent=audio_latent,
                txt_feat=sampler_input.txt_feat,
                null_txt_feat=sampler_input.null_txt_feat,
                ref_audio_feat=sampler_input.ref_audio_feat,
                ref_video_feat=sampler_input.ref_video_feat,
                ref_image_feat=sampler_input.ref_image_feat,
                ref_image_feat_len=sampler_input.ref_image_feat_len,
                ref_image_special_token_embedding=(sampler_input.ref_image_special_token_embedding),
                t=timestep,
                cfg_config=sampler_input.cfg_config,
            )
            if get_classifier_free_guidance_world_size() > 1:
                positive_input, negative_input = self._split_cfg_model_input(model_input)
                guided = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=True,
                    true_cfg_scale=1.0,
                    positive_kwargs={"model_input": positive_input},
                    negative_kwargs={"model_input": negative_input},
                    cfg_normalize=False,
                    kwargs={
                        "video_txt_guidance_scale": video_cfg,
                        "audio_txt_guidance_scale": audio_cfg,
                        "cfg_config": sampler_input.cfg_config,
                        "latent": latent,
                        "audio_latent": audio_latent,
                    },
                )
                if not isinstance(guided, tuple) or len(guided) != 2:
                    raise RuntimeError("MAGI-2 CFG parallel combine must return video and audio predictions")
                latent, audio_latent = self._step_guided(
                    guided,
                    latent,
                    audio_latent,
                    sampler_input.video_scheduler,
                    sampler_input.audio_scheduler,
                    timestep,
                )
            else:
                model_pred = self.forward(model_input)
                latent, audio_latent, _, _ = self.step(
                    model_pred,
                    latent,
                    audio_latent,
                    video_cfg,
                    audio_cfg,
                    sampler_input.video_scheduler,
                    sampler_input.audio_scheduler,
                    timestep,
                    cfg_config=sampler_input.cfg_config,
                )

        return latent, audio_latent

    @staticmethod
    def _validate_prepare_inputs(
        latent: torch.Tensor,
        audio_latent: torch.Tensor,
        txt_feat: torch.Tensor,
        null_txt_feat: torch.Tensor,
    ) -> None:
        if latent.ndim != 5:
            raise ValueError("latent must be [B,C,T,H,W]")
        if audio_latent.ndim != 3 or txt_feat.ndim != 3 or null_txt_feat.ndim != 3:
            raise ValueError("audio and text features must be [B,L,C]")
        batch_size = latent.shape[0]
        if not (audio_latent.shape[0] == txt_feat.shape[0] == null_txt_feat.shape[0] == batch_size):
            raise ValueError("video, audio, positive-text, and negative-text batch sizes must match")

    def prepare_model_input(
        self,
        latent: torch.Tensor,
        audio_latent: torch.Tensor,
        txt_feat: torch.Tensor,
        null_txt_feat: torch.Tensor,
        ref_audio_feat: torch.Tensor | None = None,
        ref_video_feat: torch.Tensor | None = None,
        ref_image_feat: torch.Tensor | None = None,
        ref_image_feat_len: torch.Tensor | None = None,
        ref_image_special_token_embedding: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
        cfg_config: CFGConfig | None = None,
    ) -> ModelInput:
        self._validate_prepare_inputs(latent, audio_latent, txt_feat, null_txt_feat)
        batch_size = latent.shape[0]
        audio_latent_len = audio_latent.shape[1]
        txt_feat_len = txt_feat.shape[1]
        null_txt_feat_len = null_txt_feat.shape[1]
        target_length = max(txt_feat_len, null_txt_feat_len)
        txt_feat, _ = _pad_or_trim(txt_feat, target_length, 1)
        null_txt_feat, _ = _pad_or_trim(null_txt_feat, target_length, 1)

        if ref_audio_feat is None:
            ref_audio_feat = audio_latent.new_empty(batch_size, 0, audio_latent.shape[-1])
        if ref_audio_feat.ndim != 3 or ref_audio_feat.shape[0] != batch_size:
            raise ValueError("ref_audio_feat must be [B,L,C] with the same batch size")
        ref_audio_feat_len = ref_audio_feat.shape[1]

        if ref_video_feat is None:
            # Preserve the 5-D ABI without allocating or reading a full-size
            # placeholder that the zero reference length excludes downstream.
            ref_video_feat = latent[:, :, :0]
            ref_video_feat_len = 0
        else:
            if ref_video_feat.ndim != 5 or ref_video_feat.shape[0] != batch_size:
                raise ValueError("ref_video_feat must be [B,C,T,H,W] with the same batch size")
            ref_video_feat_len = (ref_video_feat.shape[3] * ref_video_feat.shape[4]) // 4

        ref_image_feat_cfg, ref_image_feat_len_cfg, ref_image_special_tokens_cfg = self._prepare_ref_image_cfg(
            ref_image_feat,
            ref_image_feat_len,
            ref_image_special_token_embedding,
            cfg_config,
        )

        if t is None:
            t_value = torch.tensor(0.0, device=latent.device, dtype=latent.dtype)
        elif isinstance(t, torch.Tensor):
            t_value = t.reshape(-1)[0].to(device=latent.device, dtype=latent.dtype)
        else:
            t_value = torch.tensor(float(t), device=latent.device, dtype=latent.dtype)
        t_normalized = t_value / 1000.0
        cfg_batch = batch_size * 2
        t_batch = t_normalized.expand(cfg_batch)

        _, _, video_t, video_h, video_w = latent.shape
        audio_t = audio_latent.shape[1]
        # Keep zero-stride views here. Patchification/materialization happens
        # once in the data proxy, so cloning the full token grids every denoise
        # step only increases peak memory and allocator pressure.
        per_token_video_t = t_batch.view(-1, 1, 1, 1, 1).expand(cfg_batch, 1, video_t, video_h, video_w)
        per_token_audio_t = t_batch.view(-1, 1, 1).expand(cfg_batch, audio_t, 1)

        audio_lengths = torch.full(
            (cfg_batch,),
            audio_latent_len,
            device=latent.device,
            dtype=torch.long,
        )
        text_lengths = torch.tensor(
            [txt_feat_len] * batch_size + [null_txt_feat_len] * batch_size,
            device=latent.device,
            dtype=torch.long,
        )
        ref_audio_lengths = torch.full(
            (cfg_batch,),
            ref_audio_feat_len,
            device=latent.device,
            dtype=torch.long,
        )
        ref_video_lengths = torch.full(
            (cfg_batch,),
            ref_video_feat_len,
            device=latent.device,
            dtype=torch.long,
        )

        return ModelInput(
            x_t=torch.cat((latent, latent), dim=0),
            audio_x_t=torch.cat((audio_latent, audio_latent), dim=0),
            audio_feat_len=audio_lengths,
            txt_feat=torch.cat((txt_feat, null_txt_feat), dim=0),
            txt_feat_len=text_lengths,
            t=t_batch,
            per_token_video_t=per_token_video_t,
            per_token_audio_t=per_token_audio_t,
            ref_audio_feat=torch.cat((ref_audio_feat, torch.zeros_like(ref_audio_feat)), dim=0),
            ref_audio_feat_len=ref_audio_lengths,
            ref_video_feat=torch.cat((ref_video_feat, torch.zeros_like(ref_video_feat)), dim=0),
            ref_video_feat_len=ref_video_lengths,
            ref_image_feat=ref_image_feat_cfg,
            ref_image_feat_len=ref_image_feat_len_cfg,
            ref_image_special_token_embedding=ref_image_special_tokens_cfg,
        )

    @staticmethod
    def _split_cfg_model_input(model_input: ModelInput) -> tuple[ModelInput, ModelInput]:
        """Split the packed positive/negative batch for CFG rank dispatch."""

        batch = model_input.x_t.shape[0]
        if batch % 2:
            raise ValueError("MAGI-2 CFG model input must contain equal positive and negative halves")
        half = batch // 2

        def split(value: torch.Tensor | Sequence[int] | None) -> tuple[torch.Tensor | Sequence[int] | None, ...]:
            if value is None:
                return None, None
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or value.shape[0] != batch:
                    raise ValueError("MAGI-2 CFG tensor fields must use the packed CFG batch dimension")
                return value[:half], value[half:]
            if len(value) != batch:
                raise ValueError("MAGI-2 CFG sequence fields must use the packed CFG batch dimension")
            return value[:half], value[half:]

        field_pairs = {
            name: split(getattr(model_input, name))
            for name in (
                "x_t",
                "audio_x_t",
                "audio_feat_len",
                "txt_feat",
                "txt_feat_len",
                "t",
                "ref_audio_feat",
                "ref_audio_feat_len",
                "ref_video_feat",
                "ref_video_feat_len",
                "per_token_video_t",
                "per_token_audio_t",
                "ref_image_feat",
                "ref_image_feat_len",
                "ref_image_special_token_embedding",
            )
        }
        branches = []
        for branch in range(2):
            branches.append(ModelInput(**{name: pair[branch] for name, pair in field_pairs.items()}))
        return branches[0], branches[1]

    @staticmethod
    def _prepare_ref_image_cfg(
        ref_image_feat: torch.Tensor | None,
        ref_image_feat_len: torch.Tensor | None,
        ref_image_special_token_embedding: torch.Tensor | None,
        cfg_config: CFGConfig | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if ref_image_feat is None:
            if ref_image_feat_len is not None or ref_image_special_token_embedding is not None:
                raise ValueError("reference-image features, lengths, and special tokens must be provided together")
            return None, None, None
        if ref_image_feat_len is None or ref_image_special_token_embedding is None:
            raise ValueError("reference-image features, lengths, and special tokens must be provided together")

        keep_ref = bool(cfg_config is not None and cfg_config.use_ref_for_uncond)
        uncond_feature = ref_image_feat if keep_ref else torch.zeros_like(ref_image_feat)
        return (
            torch.cat((ref_image_feat, uncond_feature), dim=0),
            torch.cat((ref_image_feat_len, ref_image_feat_len), dim=0),
            torch.cat(
                (ref_image_special_token_embedding, ref_image_special_token_embedding),
                dim=0,
            ),
        )

    def precalculate_cfg(
        self,
        t_list: Sequence[torch.Tensor],
        latent_length: int,
        cfg_config: CFGConfig,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[list[torch.Tensor], list[float]]:
        if latent_length < 0:
            raise ValueError("latent_length must be non-negative")
        if device is None:
            if self.device is not None:
                device = self.device
            elif t_list:
                device = t_list[0].device
            else:
                device = torch.device("cpu")

        if cfg_config.use_cfg_trick:
            video_cfg_base = (
                torch.tensor(cfg_config.video_txt_guidance_scale, device=device)
                .expand(1, 1, latent_length, 1, 1)
                .clone()
            )
            video_cfg_base[:, :, : cfg_config.cfg_trick_start_frame] = min(
                cfg_config.cfg_trick_value,
                cfg_config.video_txt_guidance_scale,
            )
        else:
            video_cfg_base = torch.tensor([cfg_config.video_txt_guidance_scale], device=device)

        all_video_cfgs: list[torch.Tensor] = []
        all_audio_cfgs: list[float] = []
        for timestep in t_list:
            video_cfg = video_cfg_base.clone()
            if cfg_config.use_dynamic_cfg:
                timestep_value = float(timestep.detach().reshape(-1)[0].item())
                if timestep_value < cfg_config.dynamic_cfg_start_t:
                    video_cfg[video_cfg > cfg_config.dynamic_cfg_cutoff_value] = cfg_config.dynamic_cfg_cutoff_value
            all_video_cfgs.append(video_cfg)
            all_audio_cfgs.append(cfg_config.audio_txt_guidance_scale)
        return all_video_cfgs, all_audio_cfgs

    @staticmethod
    def _cfg_scale_tensor(
        guidance_scale: torch.Tensor | float,
        like: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(guidance_scale, torch.Tensor):
            return guidance_scale.to(device=like.device, dtype=like.dtype)
        return torch.tensor(float(guidance_scale), device=like.device, dtype=like.dtype)

    @classmethod
    def _get_skimming_mask(
        cls,
        x_orig: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        guidance_scale = cls._cfg_scale_tensor(guidance_scale, cond)
        x_orig = x_orig.to(device=cond.device, dtype=cond.dtype)
        denoised = x_orig - ((x_orig - uncond) + guidance_scale * ((x_orig - cond) - (x_orig - uncond)))
        matching_pred_signs = (cond - uncond).sign() == cond.sign()
        matching_diff_after = cond.sign() == (cond * guidance_scale - uncond * (guidance_scale - 1)).sign()
        deviation_influence = denoised.sign() == (denoised - x_orig).sign()
        return matching_pred_signs & matching_diff_after & deviation_influence

    @classmethod
    def _apply_skimmed_cfg_linear(
        cls,
        x_orig: torch.Tensor | None,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guidance_scale: torch.Tensor | float,
        cfg_config: CFGConfig | None,
    ) -> torch.Tensor:
        if (
            cfg_config is None
            or not cfg_config.use_skimmed_cfg_linear
            or x_orig is None
            or not torch.any(uncond).item()
        ):
            return uncond

        guidance_scale = cls._cfg_scale_tensor(guidance_scale, cond)
        valid_scale = guidance_scale > 1
        if not torch.any(valid_scale).item():
            return uncond
        scale_delta = torch.where(
            valid_scale,
            guidance_scale - 1,
            torch.ones_like(guidance_scale),
        )
        fallback_weight = (cfg_config.skimmed_cfg_scale - 1) / scale_delta
        target_uncond = cond * (1 - fallback_weight) + uncond * fallback_weight
        skim_mask = cls._get_skimming_mask(x_orig, cond, uncond, guidance_scale)
        uncond = torch.where(skim_mask & valid_scale, target_uncond, uncond)

        target_uncond = cond * (1 - fallback_weight) + uncond * fallback_weight
        skim_mask = cls._get_skimming_mask(x_orig, uncond, cond, guidance_scale)
        return torch.where(skim_mask & valid_scale, target_uncond, uncond)

    @staticmethod
    def _cfg_rescale_dims(tensor: torch.Tensor) -> tuple[int, ...]:
        if tensor.ndim <= 1:
            return ()
        if tensor.ndim in (2, 3):
            return (1,)
        return tuple(range(2, tensor.ndim))

    @classmethod
    def _apply_cfg_rescale(
        cls,
        pos: torch.Tensor,
        cfg: torch.Tensor,
        cfg_config: CFGConfig | None,
    ) -> torch.Tensor:
        if cfg_config is None or cfg_config.cfg_rescale <= 0 or cfg.numel() == 0:
            return cfg
        spatial_dims = cls._cfg_rescale_dims(cfg)
        if not spatial_dims:
            return cfg
        pos_std = pos.float().std(dim=spatial_dims, keepdim=True, unbiased=False)
        cfg_std = cfg.float().std(dim=spatial_dims, keepdim=True, unbiased=False).clamp_min(1e-6)
        factor = cfg_config.cfg_rescale * (pos_std / cfg_std) + (1 - cfg_config.cfg_rescale)
        return cfg * factor.to(device=cfg.device, dtype=cfg.dtype)

    def cfg_velocity(
        self,
        model_output: tuple[torch.Tensor, torch.Tensor],
        video_txt_guidance_scale: torch.Tensor | float,
        audio_txt_guidance_scale: torch.Tensor | float,
        cfg_config: CFGConfig | None = None,
        latent: torch.Tensor | None = None,
        audio_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cond_uncond_video, cond_uncond_audio = model_output
        if cond_uncond_video.shape[0] % 2 or cond_uncond_audio.shape[0] % 2:
            raise ValueError("MAGI-2 CFG outputs must contain equal conditional/unconditional halves")
        video_batch = cond_uncond_video.shape[0] // 2
        audio_batch = cond_uncond_audio.shape[0] // 2
        if video_batch != audio_batch:
            raise ValueError("video and audio CFG output batch sizes differ")

        if cond_uncond_video.numel() > 0:
            cond_video = cond_uncond_video[:video_batch]
            uncond_video = cond_uncond_video[video_batch:]
            uncond_video = self._apply_skimmed_cfg_linear(
                latent,
                cond_video,
                uncond_video,
                video_txt_guidance_scale,
                cfg_config,
            )
            cfg_video = uncond_video + video_txt_guidance_scale * (cond_video - uncond_video)
            cfg_video = self._apply_cfg_rescale(cond_video, cfg_video, cfg_config)
        else:
            cfg_video = None

        if cond_uncond_audio.numel() > 0:
            cond_audio = cond_uncond_audio[:audio_batch]
            uncond_audio = cond_uncond_audio[audio_batch:]
            uncond_audio = self._apply_skimmed_cfg_linear(
                audio_latent,
                cond_audio,
                uncond_audio,
                audio_txt_guidance_scale,
                cfg_config,
            )
            cfg_audio = uncond_audio + audio_txt_guidance_scale * (cond_audio - uncond_audio)
            cfg_audio = self._apply_cfg_rescale(cond_audio, cfg_audio, cfg_config)
        else:
            cfg_audio = None
        return cfg_video, cfg_audio

    def combine_cfg_noise(
        self,
        positive_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        negative_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        true_cfg_scale: float,
        cfg_normalize: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve MAGI-2's dual-modality guidance under shared CFG dispatch."""

        del true_cfg_scale, cfg_normalize
        if not isinstance(positive_noise_pred, tuple) or not isinstance(negative_noise_pred, tuple):
            raise TypeError("MAGI-2 CFG predictions must contain video and audio tensors")
        if len(positive_noise_pred) != 2 or len(negative_noise_pred) != 2:
            raise ValueError("MAGI-2 CFG predictions must contain exactly two tensors")
        if kwargs is None:
            raise ValueError("MAGI-2 CFG combine requires guidance and latent context")
        cfg_video, cfg_audio = self.cfg_velocity(
            (
                torch.cat((positive_noise_pred[0], negative_noise_pred[0]), dim=0),
                torch.cat((positive_noise_pred[1], negative_noise_pred[1]), dim=0),
            ),
            kwargs["video_txt_guidance_scale"],
            kwargs["audio_txt_guidance_scale"],
            kwargs.get("cfg_config"),
            kwargs.get("latent"),
            kwargs.get("audio_latent"),
        )
        if cfg_video is None or cfg_audio is None:
            raise RuntimeError("MAGI-2 Preview must produce both video and audio CFG predictions")
        return cfg_video, cfg_audio

    @staticmethod
    def _step_guided(
        guided_output: tuple[torch.Tensor, torch.Tensor],
        latent: torch.Tensor,
        audio_latent: torch.Tensor,
        video_scheduler: SchedulerMixin,
        audio_scheduler: SchedulerMixin,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg_video, cfg_audio = guided_output
        latent = video_scheduler.step(cfg_video, t, latent, return_dict=False)[0]
        audio_latent = audio_scheduler.step(cfg_audio, t, audio_latent, return_dict=False)[0]
        return latent, audio_latent

    def step(
        self,
        model_output: tuple[torch.Tensor, torch.Tensor],
        latent: torch.Tensor,
        audio_latent: torch.Tensor,
        video_txt_guidance_scale: torch.Tensor | float,
        audio_txt_guidance_scale: torch.Tensor | float,
        video_scheduler: SchedulerMixin,
        audio_scheduler: SchedulerMixin,
        t: torch.Tensor,
        cfg_config: CFGConfig | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        cfg_video, cfg_audio = self.cfg_velocity(
            model_output,
            video_txt_guidance_scale,
            audio_txt_guidance_scale,
            cfg_config,
            latent,
            audio_latent,
        )
        if cfg_video is not None:
            latent = video_scheduler.step(cfg_video, t, latent, return_dict=False)[0]
        if cfg_audio is not None:
            audio_latent = audio_scheduler.step(
                cfg_audio,
                t,
                audio_latent,
                return_dict=False,
            )[0]
        return latent, audio_latent, cfg_video, cfg_audio


__all__ = [
    "CFGConfig",
    "Magi2PreviewSampler",
    "SamplerInput",
    "build_magi2_preview_schedulers",
]

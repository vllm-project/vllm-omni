# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from contextlib import nullcontext
from inspect import signature
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportAudioOutput,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.nava.audio_vae import NAVAAudioVAE
from vllm_omni.diffusion.models.nava.config import (
    DEFAULT_NAVA_AUDIO_NEGATIVE_PROMPT,
    DEFAULT_NAVA_MODEL_INDEX,
    DEFAULT_NAVA_VIDEO_NEGATIVE_PROMPT,
    NAVA_CONFIG_ALIAS_MAP,
    NAVAConfig,
    NAVARequestContext,
    NAVASpeakerCondition,
    inject_speaker_sentinel,
    parse_speech_spans,
)
from vllm_omni.diffusion.models.nava.nava_transformer import NAVATransformer
from vllm_omni.diffusion.models.nava.scheduler import NAVAFlowMatchScheduler
from vllm_omni.diffusion.models.nava.speaker import NAVASpeakerEncoder
from vllm_omni.diffusion.models.nava.text_encoder import NAVAWanTextEncoder as _NAVATextEncoder
from vllm_omni.diffusion.models.nava.utils import as_bool, image_to_tensor, resolve_num_frames
from vllm_omni.diffusion.models.nava.video_vae import NAVAVideoVAE
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


def get_nava_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(output: dict[str, Any] | Any, output_type: str = "np", sampling_params=None):
        if output_type == "latent":
            return output
        if isinstance(output, dict):
            video = output.get("video")
            audio = output.get("audio")
            audio_sample_rate = output.get("audio_sample_rate", NAVAPipeline.audio_sample_rate)
            fps = output.get("fps", _resolve_output_fps(sampling_params, NAVAPipeline.fps))
        else:
            video = output
            audio = None
            audio_sample_rate = NAVAPipeline.audio_sample_rate
            fps = _resolve_output_fps(sampling_params, NAVAPipeline.fps)
        return {
            "video": _normalize_video_output(video, output_type),
            "audio": audio,
            "audio_sample_rate": int(audio_sample_rate),
            "fps": fps,
        }

    return post_process_func


class NAVAPipeline(
    nn.Module,
    SupportImageInput,
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    support_image_input: ClassVar[bool] = True
    support_audio_input: ClassVar[bool] = True
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 16000
    fps: ClassVar[int] = 24
    dummy_run_num_frames: ClassVar[int] = 0

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "speaker_encoder"]
    _vae_modules: ClassVar[list[str]] = ["video_vae", "audio_vae"]
    _resident_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        self.model_root = self._resolve_model_root()
        self.nava_config = self._load_nava_config(od_config)
        self.audio_sample_rate = self.nava_config.audio_sample_rate
        self.fps = self.nava_config.fps
        self.scheduler = NAVAFlowMatchScheduler(shift=5.0)
        self.scheduler_audio = NAVAFlowMatchScheduler(shift=5.0)
        init_seed = self._custom_pipeline_arg("nava_init_seed")
        if init_seed is not None:
            self._set_seed(int(init_seed))
        self._rng_state_after_init: dict[str, Any] | None = None
        self._validate_runtime_features()
        self._init_native_components()
        if as_bool(self._custom_pipeline_arg("nava_restore_init_cuda_rng_before_sample", False)):
            self._rng_state_after_init = self._capture_rng_state()
        self._init_weight_sources()
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = self.transformer.load_weights(weights)
        return {f"transformer.{name}" for name in loaded}

    def forward(self, request: OmniDiffusionRequest, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        ctx = self._parse_request(request)
        generator = self._make_generator(ctx.seed)

        if ctx.speaker_condition is None:
            text_embeds = self._encode_text(ctx.prompt)
            speaker_positions = None
        else:
            text_embeds, speaker_positions = self._encode_text_with_speaker_positions(ctx.prompt)
        negative_video_text_embeds, negative_audio_text_embeds = self._encode_negative_texts(ctx, text_embeds)
        image_embeds = self._encode_image(ctx)
        speaker_embeds = self._encode_speakers(ctx)
        if speaker_embeds is not None and not speaker_positions:
            raise ValueError("NAVA speaker references require <S>...<E> spans with <extra_id_2> speaker markers.")
        if speaker_embeds is not None and len(speaker_positions[0]) != speaker_embeds.shape[0]:
            raise ValueError(
                "NAVA speaker embedding count must match tokenizer speaker marker count: "
                f"got {speaker_embeds.shape[0]} embedding(s) and {len(speaker_positions[0])} marker(s)."
            )
        latents = self._prepare_latents(ctx, generator)

        video_latents, audio_latents = self._denoise(
            ctx=ctx,
            video_latents=latents["video"],
            audio_latents=latents["audio"],
            text_embeds=text_embeds,
            negative_video_text_embeds=negative_video_text_embeds,
            negative_audio_text_embeds=negative_audio_text_embeds,
            image_embeds=image_embeds,
            speaker_embeds=speaker_embeds,
            speaker_positions=speaker_positions,
        )
        video = self._decode_video(video_latents, ctx)
        audio = self._decode_audio(audio_latents)
        return DiffusionOutput(
            output={
                "video": video,
                "audio": audio,
                "audio_sample_rate": self.audio_sample_rate,
                "fps": ctx.fps,
            }
        )

    def _validate_runtime_features(self) -> None:
        if not self._is_cuda_device():
            raise ValueError("NAVAPipeline is currently verified only on CUDA/NVIDIA GPUs.")
        if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload:
            raise ValueError("NAVAPipeline CPU and layerwise offload are not verified yet.")
        pc = self.od_config.parallel_config
        enabled = {
            "tensor_parallel_size": pc.tensor_parallel_size,
            "sequence_parallel_size": pc.sequence_parallel_size,
            "cfg_parallel_size": pc.cfg_parallel_size,
            "vae_patch_parallel_size": pc.vae_patch_parallel_size,
            "pipeline_parallel_size": pc.pipeline_parallel_size,
            "data_parallel_size": pc.data_parallel_size,
        }
        enabled = {key: value for key, value in enabled.items() if int(value) != 1}
        if enabled or pc.use_hsdp or pc.enable_expert_parallel:
            raise ValueError(
                "NAVAPipeline native parallel and sharding modes are not verified yet. "
                f"Unsupported settings: {enabled}, use_hsdp={pc.use_hsdp}, "
                f"enable_expert_parallel={pc.enable_expert_parallel}."
            )

    def _is_cuda_device(self) -> bool:
        return self.device.type == "cuda"

    def _init_native_components(self) -> None:
        text_compile = as_bool(self._custom_pipeline_arg("nava_text_encoder_compile", False))
        if self._custom_pipeline_arg("disable_text_encoder_compile") is not None:
            text_compile = not as_bool(self._custom_pipeline_arg("disable_text_encoder_compile"))
        self.text_encoder = _NAVATextEncoder(
            self.model_root,
            self.nava_config,
            self.device,
            compile_model=text_compile,
        )
        self.video_vae = NAVAVideoVAE(self.model_root, self.nava_config, self.device)
        self.audio_vae = NAVAAudioVAE(self.model_root, self.nava_config)
        self.speaker_encoder = NAVASpeakerEncoder(self.model_root, self.nava_config)
        self.transformer = NAVATransformer(self.nava_config)
        self.to(self.device)

    def _resolve_model_root(self) -> str:
        model = str(self.od_config.model or "")
        if not model:
            raise ValueError("NAVAPipeline requires a local model directory or Hugging Face repo id.")
        if os.path.isdir(model):
            return model
        return download_weights_from_hf_specific(model, None, ["*"], revision=self.od_config.revision)

    def _init_weight_sources(self) -> None:
        model = self.model_root
        if not model:
            return
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder=None,
                revision=self.od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
                allow_patterns_overrides=[self.nava_config.ckpt_name],
            )
        ]

    def _load_nava_config(self, od_config: OmniDiffusionConfig) -> NAVAConfig:
        config_data: dict[str, Any] = dict(DEFAULT_NAVA_MODEL_INDEX)
        explicit = dict(od_config.model_config or {})
        explicit_config_name = explicit.get("config_name", explicit.get("config"))
        if self.model_root and os.path.isdir(self.model_root):
            index_path = os.path.join(self.model_root, "model_index.json")
            if os.path.exists(index_path):
                with open(index_path, encoding="utf-8") as f:
                    index_data = json.load(f)
                if isinstance(index_data, dict):
                    config_data.update(index_data)
            config_name = str(explicit_config_name or config_data.get("config") or DEFAULT_NAVA_MODEL_INDEX["config"])
            config_path = os.path.join(self.model_root, config_name)
            if not os.path.exists(config_path) and config_name == "configs/nava.yaml":
                legacy_path = os.path.join(self.model_root, "nava.yaml")
                if os.path.exists(legacy_path):
                    config_path = legacy_path
                    config_data["config"] = "nava.yaml"
            if os.path.exists(config_path):
                config_data.update(_load_yaml(config_path))
            joint_config_path = os.path.join(self.model_root, "config.json")
            if os.path.exists(joint_config_path):
                with open(joint_config_path, encoding="utf-8") as f:
                    joint_config = json.load(f)
                if isinstance(joint_config, dict):
                    config_data.setdefault("model", {})["joint_config_data"] = joint_config

        for old_key, new_key in NAVA_CONFIG_ALIAS_MAP.items():
            if old_key in explicit:
                explicit[new_key] = explicit[old_key]
        explicit_keys = set(explicit)
        config_data.update(explicit)
        additional = dict(od_config.additional_config or {})
        explicit_keys.update(additional)
        config_data.update(additional)
        config_data["_explicit_keys"] = explicit_keys
        return NAVAConfig.from_dict(config_data)

    def _parse_request(self, request: OmniDiffusionRequest) -> NAVARequestContext:
        prompts = getattr(request, "prompts", None)
        if prompts is None:
            prompts = [request.prompt]
        if len(prompts) != 1:
            raise ValueError("NAVAPipeline currently supports one prompt per request. Use request-level batching.")
        prompt_data = prompts[0]
        prompt = prompt_data if isinstance(prompt_data, str) else str(prompt_data.get("prompt", ""))
        if not prompt:
            raise ValueError("NAVAPipeline requires a non-empty prompt.")
        multi_modal_data = {} if isinstance(prompt_data, str) else (prompt_data.get("multi_modal_data") or {})
        sp = request.sampling_params
        extra = sp.extra_args or {}
        speaker_condition = self._parse_speaker_condition(prompt, multi_modal_data, extra)
        image = multi_modal_data.get("image")

        output_frames = max(1, int(resolve_num_frames(sp.num_frames, extra, self.nava_config.frames)))
        latent_frames = self.nava_config.video_latent_frames(output_frames)
        return NAVARequestContext(
            prompt=prompt,
            negative_prompt=str(extra.get("negative_prompt", self.nava_config.negative_prompt)),
            audio_negative_prompt=str(extra.get("audio_negative_prompt", self.nava_config.audio_negative_prompt)),
            video_negative_prompt=str(extra.get("video_negative_prompt", self.nava_config.video_negative_prompt)),
            image=image,
            speaker_condition=speaker_condition,
            height=int(sp.height or extra.get("height") or self.nava_config.log_height),
            width=int(sp.width or extra.get("width") or self.nava_config.log_width),
            frames=latent_frames,
            fps=float(sp.resolved_frame_rate or extra.get("fps") or self.nava_config.fps),
            seed=int(sp.seed if sp.seed is not None else extra.get("seed", 100)),
            num_steps=int(
                sp.num_inference_steps
                or extra.get("num_inference_steps")
                or extra.get("num_steps")
                or self.nava_config.num_steps
            ),
            video_guidance_scale=float(extra.get("video_guidance_scale", self.nava_config.video_guidance_scale)),
            audio_guidance_scale=float(extra.get("audio_guidance_scale", self.nava_config.audio_guidance_scale)),
            video_align_guidance_scale=float(
                extra.get("video_align_guidance_scale", self.nava_config.video_align_guidance_scale)
            ),
            audio_align_guidance_scale=float(
                extra.get("audio_align_guidance_scale", self.nava_config.audio_align_guidance_scale)
            ),
            timbre_align_guidance_scale=float(
                extra.get("timbre_align_guidance_scale", self.nava_config.timbre_align_guidance_scale)
            ),
            align_3d_cfg=as_bool(extra.get("align_3d_cfg", self.nava_config.align_3d_cfg)),
            timbre_cfg=self._resolve_timbre_cfg(extra, speaker_condition),
            negative_prompt_mode=as_bool(extra.get("negative_prompt_mode", self.nava_config.negative_prompt_mode)),
        )

    def _parse_speaker_condition(
        self,
        prompt: str,
        multi_modal_data: dict[str, Any],
        extra: dict[str, Any],
    ) -> NAVASpeakerCondition | None:
        wavs = multi_modal_data.get("spk_wavs", multi_modal_data.get("audio", extra.get("spk_wavs")))
        if wavs is None:
            return None
        if not isinstance(wavs, list):
            wavs = [wavs]
        spans = parse_speech_spans(prompt)
        if len(wavs) != len(spans):
            raise ValueError(
                "NAVA speaker reference count must match <S>...<E> speech span count: "
                f"got {len(wavs)} reference wav(s) and {len(spans)} span(s)."
            )
        return NAVASpeakerCondition(wavs=list(wavs), spans=spans)

    def _resolve_timbre_cfg(self, extra: dict[str, Any], speaker_condition: NAVASpeakerCondition | None) -> bool:
        if "timbre_cfg" in extra:
            requested = as_bool(extra["timbre_cfg"])
            if requested and speaker_condition is None:
                raise ValueError("NAVA timbre_cfg requires reference speaker WAVs aligned to <S>...<E> spans.")
            return requested
        return bool(self.nava_config.timbre_cfg and speaker_condition is not None)

    def _encode_text(self, prompt: str) -> torch.Tensor:
        return self._run_text_encoder(prompt, return_speaker_positions=False)[0]

    def _encode_texts(self, prompts: list[str]) -> torch.Tensor:
        return self._run_text_encoder(prompts, return_speaker_positions=False)[0]

    def _encode_text_with_speaker_positions(self, prompt: str) -> tuple[torch.Tensor, list[list[int]] | None]:
        return self._run_text_encoder(prompt, return_speaker_positions=True)

    def _encode_negative_texts(
        self,
        ctx: NAVARequestContext,
        positive_text_embeds: torch.Tensor | list[torch.Tensor],
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        if not ctx.negative_prompt_mode:
            if isinstance(positive_text_embeds, list):
                zeros = [torch.zeros_like(item) for item in positive_text_embeds]
                return [zeros[0]], [zeros[0]]
            zeros = torch.zeros_like(positive_text_embeds)
            return zeros, zeros

        if ctx.negative_prompt:
            negative_prompts = [ctx.negative_prompt, ctx.negative_prompt]
        else:
            negative_prompts = [
                ctx.video_negative_prompt or DEFAULT_NAVA_VIDEO_NEGATIVE_PROMPT,
                ctx.audio_negative_prompt or DEFAULT_NAVA_AUDIO_NEGATIVE_PROMPT,
            ]
        embeds = self._encode_texts(negative_prompts)
        if isinstance(embeds, list):
            return [embeds[0]], [embeds[1]]
        return embeds[0:1], embeds[1:2]

    def _run_text_encoder(
        self,
        prompt: str | list[str],
        *,
        return_speaker_positions: bool,
    ) -> tuple[torch.Tensor | list[torch.Tensor], list[list[int]] | None]:
        captions = [prompt] if isinstance(prompt, str) else list(prompt)
        captions = [inject_speaker_sentinel(item) for item in captions]
        encoder = self.text_encoder
        if hasattr(encoder, "encode"):
            # Text embedding: [batch, text_tokens, text_dim], shared by video
            # and audio denoising branches.
            kwargs: dict[str, Any] = {}
            if "return_speaker_positions" in signature(encoder.encode).parameters:
                kwargs["return_speaker_positions"] = return_speaker_positions
            target_dtype = self.nava_config.target_dtype
            autocast_enabled = self.device.type == "cuda" and target_dtype in (torch.bfloat16, torch.float16)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=target_dtype) if autocast_enabled else nullcontext()
            )
            with autocast_context:
                result = encoder.encode(captions, device=self.device, dtype=target_dtype, **kwargs)
            if isinstance(result, tuple):
                return result
            return result, None
        raise TypeError("NAVA text_encoder must expose encode(texts, device, dtype).")

    def _encode_image(self, ctx: NAVARequestContext) -> torch.Tensor | None:
        if ctx.image is None:
            return None
        image = image_to_tensor(ctx.image, ctx.height, ctx.width).to(self.device)
        if not hasattr(self.video_vae, "encode_first_frame"):
            raise TypeError("NAVA video_vae must expose encode_first_frame(image).")
        # Image embedding: first-frame pixels become a conditioning latent for
        # the video denoising branch.
        image_embeds = self.video_vae.encode_first_frame(image)
        return image_embeds

    def _encode_speakers(self, ctx: NAVARequestContext) -> torch.Tensor | None:
        if ctx.speaker_condition is None:
            return None
        if not hasattr(self.speaker_encoder, "encode"):
            raise TypeError("NAVA speaker_encoder must expose encode(wavs, device, dtype).")
        # Speaker embedding: reference WAVs are ordered to match <S>...<E>
        # spans and only condition the timbre branch.
        speaker_embeds = self.speaker_encoder.encode(
            ctx.speaker_condition.wavs,
            device=self.device,
            dtype=torch.float32,
        )
        return speaker_embeds

    def _custom_pipeline_arg(self, key: str, default: Any = None) -> Any:
        return (self.od_config.custom_pipeline_args or {}).get(key, default)

    def _prepare_latents(
        self,
        ctx: NAVARequestContext,
        generator: torch.Generator | None,
    ) -> dict[str, torch.Tensor]:
        latent_h, latent_w = self.nava_config.video_latent_hw(ctx.height, ctx.width)
        video_tokens = ctx.frames * latent_h * latent_w
        audio_tokens = self.nava_config.audio_latent_length(ctx.frames, fps=ctx.fps)
        # Video/audio latent initialization fixes the generated duration and
        # resolution before denoising starts. Upstream initializes and keeps
        # sampling latents in float32; the transformer casts internally where
        # BF16 math is expected.
        latents = {
            "video": torch.randn(
                1,
                video_tokens,
                self.nava_config.video_latent_ch,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            ),
            "audio": torch.randn(
                1,
                audio_tokens,
                self.nava_config.audio_latent_ch,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            ),
        }
        return latents

    def _denoise(
        self,
        *,
        ctx: NAVARequestContext,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        text_embeds: torch.Tensor,
        negative_video_text_embeds: torch.Tensor | list[torch.Tensor],
        negative_audio_text_embeds: torch.Tensor | list[torch.Tensor],
        image_embeds: torch.Tensor | None,
        speaker_embeds: torch.Tensor | None,
        speaker_positions: list[list[int]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_latents.ndim == 3 and video_latents.shape[0] == 1:
            video_latents = video_latents[0]
        if audio_latents.ndim == 3 and audio_latents.shape[0] == 1:
            audio_latents = audio_latents[0]
        timesteps = self.scheduler.set_timesteps(ctx.num_steps, device=self.device)
        self.scheduler_audio.set_timesteps(ctx.num_steps, device=self.device)
        video_grid = self._video_grid(ctx)
        needs_negative_cfg = ctx.video_guidance_scale != 1.0 or ctx.audio_guidance_scale != 1.0
        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_index, timestep in enumerate(timesteps):
                timestep_tensor = timestep.reshape(1)
                positive = self.transformer(
                    video_latents=video_latents,
                    audio_latents=audio_latents,
                    timestep=timestep_tensor,
                    text_embeds=text_embeds,
                    image_embeds=image_embeds,
                    speaker_embeds=speaker_embeds,
                    speaker_positions=speaker_positions,
                    video_grid=video_grid,
                    step_index=step_index,
                )
                negative = None
                if needs_negative_cfg:
                    negative = self.transformer(
                        video_latents=video_latents,
                        audio_latents=audio_latents,
                        timestep=timestep_tensor,
                        text_embeds=negative_video_text_embeds,
                        audio_text_embeds=negative_audio_text_embeds,
                        image_embeds=image_embeds,
                        speaker_embeds=None,
                        speaker_positions=None,
                        video_grid=video_grid,
                        step_index=step_index,
                    )
                align = None
                if ctx.align_3d_cfg:
                    align = self.transformer(
                        video_latents=video_latents,
                        audio_latents=audio_latents,
                        timestep=timestep_tensor,
                        text_embeds=text_embeds,
                        image_embeds=image_embeds,
                        speaker_embeds=speaker_embeds,
                        speaker_positions=None,
                        masking_modality=True,
                        video_grid=video_grid,
                        step_index=step_index,
                    )
                timbre = None
                if ctx.timbre_cfg and speaker_embeds is not None:
                    timbre = self.transformer(
                        video_latents=video_latents,
                        audio_latents=audio_latents,
                        timestep=timestep_tensor,
                        text_embeds=text_embeds,
                        image_embeds=image_embeds,
                        speaker_embeds=None,
                        speaker_positions=speaker_positions,
                        video_grid=video_grid,
                        step_index=step_index,
                    )
                noise = self._combine_guidance(ctx, positive, negative, align=align, timbre=timbre)
                # Denoise step: FlowMatch updates video/audio latents with matching
                # timesteps so generated duration stays synchronized.
                video_latents = self.scheduler.step(
                    noise["video"].to(torch.float32), timestep, video_latents.to(torch.float32)
                )
                audio_latents = self.scheduler_audio.step(
                    noise["audio"].to(torch.float32), timestep, audio_latents.to(torch.float32)
                )
                pbar.update()
        return video_latents, audio_latents

    def _video_grid(self, ctx: NAVARequestContext) -> tuple[int, int, int]:
        latent_h, latent_w = self.nava_config.video_latent_hw(ctx.height, ctx.width)
        return ctx.frames, latent_h, latent_w

    def _combine_guidance(
        self,
        ctx: NAVARequestContext,
        positive: dict[str, torch.Tensor],
        negative: dict[str, torch.Tensor] | None,
        *,
        align: dict[str, torch.Tensor] | None = None,
        timbre: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if negative is None:
            negative = positive
        # CFG branch combine mirrors NAVA's AV guidance: regular negative
        # prompt CFG, optional modality-mask alignment, and optional timbre CFG.
        if align is None:
            video = negative["video"] + ctx.video_guidance_scale * (positive["video"] - negative["video"])
            audio = negative["audio"] + ctx.audio_guidance_scale * (positive["audio"] - negative["audio"])
        else:
            video = (
                positive["video"]
                + ctx.video_guidance_scale * (positive["video"] - negative["video"])
                + ctx.video_align_guidance_scale * (positive["video"] - align["video"])
            )
            audio = (
                positive["audio"]
                + ctx.audio_guidance_scale * (positive["audio"] - negative["audio"])
                + ctx.audio_align_guidance_scale * (positive["audio"] - align["audio"])
            )
        if timbre is not None:
            audio = audio + ctx.timbre_align_guidance_scale * (positive["audio"] - timbre["audio"])
        return {
            "video": video,
            "audio": audio,
        }

    def _decode_video(self, video_latents: torch.Tensor, ctx: NAVARequestContext) -> torch.Tensor:
        if not hasattr(self.video_vae, "decode"):
            raise TypeError("NAVA video_vae must expose decode(video_latents, height, width, frames).")
        # Video decode maps denoised video tokens back to [B, C, T, H, W].
        output_frames = self.nava_config.video_output_frames(ctx.frames)
        if video_latents.ndim == 2:
            video_latents = video_latents.unsqueeze(0)
        return self.video_vae.decode(video_latents, height=ctx.height, width=ctx.width, frames=output_frames)

    def _decode_audio(self, audio_latents: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.audio_vae, "decode"):
            raise TypeError("NAVA audio_vae must expose decode(audio_latents).")
        # Audio decode maps denoised audio tokens to waveform samples.
        if audio_latents.ndim == 2:
            audio_latents = audio_latents.unsqueeze(0)
        return self.audio_vae.decode(audio_latents)

    def _make_generator(self, seed: int) -> torch.Generator | None:
        if self._rng_state_after_init is not None:
            self._restore_rng_state(self._rng_state_after_init)
            return None
        if as_bool(self._custom_pipeline_arg("skip_request_seed", False)):
            return None
        self._set_seed(seed)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return generator

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _capture_rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if self._is_cuda_device():
            state["torch_cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and self._is_cuda_device():
            torch.cuda.set_rng_state(cuda_state, self.device)


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("NAVAPipeline requires PyYAML to read configs/nava.yaml.") from exc
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"NAVA config must be a mapping: {path}")
    return data


def _resolve_output_fps(sampling_params, default: int) -> int | float:
    value = getattr(sampling_params, "resolved_frame_rate", None)
    if value is None:
        value = getattr(sampling_params, "fps", None)
    fps = float(value if value is not None else default)
    return int(fps) if fps.is_integer() else fps


def _normalize_video_output(video: Any, output_type: str) -> Any:
    if not isinstance(video, torch.Tensor):
        return video
    video = video.detach().cpu()
    if output_type == "pt":
        return video
    video = video.float()
    if video.ndim == 5 and video.shape[1] in (1, 3):
        return [sample.permute(1, 0, 2, 3).contiguous().numpy() for sample in video]
    return video.numpy()

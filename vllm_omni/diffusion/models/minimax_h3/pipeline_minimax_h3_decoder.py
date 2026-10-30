# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioOutput, SupportsComponentDiscovery
from vllm_omni.diffusion.offloader import OffloadPlan
from vllm_omni.diffusion.offloader.config import OffloadStrategy, offload_streams_blocks, resolve_offload_strategy
from vllm_omni.diffusion.offloader.module_residency import BoundedAllocatorCache
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.minimax_h3.checkpoint import resolve_minimax_h3_partition
from vllm_omni.platforms import current_omni_platform

from .latent_cache import trim_decoded_overlap
from .pipeline_minimax_h3 import (
    MINIMAX_H3_AUDIO_SAMPLE_RATE,
    MINIMAX_H3_FPS,
    MiniMaxH3Pipeline,
    _MiniMaxH3EncodedOutput,
    _prepare_minimax_h3_video_output,
    _register_dlo_component_cache,
    get_minimax_h3_post_process_func,
)
from .vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE


def resolve_minimax_h3_decoder_model_path(
    model: str,
    revision: str | None,
    task_type: str | None,
) -> str:
    partition = resolve_minimax_h3_partition(model, task_type, auto_partition="fl2va")
    subdir = "Ref2VA" if partition == "ref2va" else "FL2VA"
    path = Path(model)
    if path.is_dir():
        if path.name in {"FL2VA", "Ref2VA"}:
            path = path.parent
        return str(path / subdir)
    # Decoder-only deployment downloads native VAE assets, without DiT or text weights.
    snapshot = download_weights_from_hf_specific(
        model_name_or_path=model,
        cache_dir=None,
        allow_patterns=[f"{subdir}/video_vae/**", f"{subdir}/audio_vae/**"],
        revision=revision,
        require_all=True,
    )
    return str(Path(snapshot) / subdir)


class MiniMaxH3DecoderPipeline(
    nn.Module, DiffusionPipelineProfilerMixin, SupportAudioOutput, SupportsComponentDiscovery
):
    supports_step_execution: ClassVar[bool] = False
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["video_vae", "audio_vae"]
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(on_demand_component_paths=frozenset({"video_vae", "audio_vae"}))
    _PROFILER_TARGETS: ClassVar[list[str]] = ["decode", "video_vae.decode_latent", "audio_vae.decode_latent"]

    # Reuse decoding behavior to preserve regular and chunked MP4 output contracts.
    decode = MiniMaxH3Pipeline.decode
    decode_to_mp4 = MiniMaxH3Pipeline.decode_to_mp4
    _component_on_device = MiniMaxH3Pipeline._component_on_device
    _uses_manual_component_offload = MiniMaxH3Pipeline._uses_manual_component_offload
    _offload_model_cpu_stage_output = MiniMaxH3Pipeline._offload_model_cpu_stage_output
    _release_stage_cache = MiniMaxH3Pipeline._release_stage_cache
    _is_output_owner_rank = staticmethod(MiniMaxH3Pipeline._is_output_owner_rank)

    def _decoder_load_device(self) -> torch.device:
        # Match the combined H3 pipeline's compatibility policy. Explicit
        # dit/text_encoder selection must not implicitly offload native VAEs.
        legacy_staging = self.od_config.diffusion_offload_config is None and offload_streams_blocks(self.od_config)
        return torch.device("cpu") if legacy_staging else self.device

    def _setup_decoder_component_cache(self) -> None:
        self._dlo_component_cache = None
        if (
            self._decoder_load_device().type == "cpu"
            and resolve_offload_strategy(self.od_config) is OffloadStrategy.DISTRIBUTED_LAYER_WISE
        ):
            self._dlo_component_cache = BoundedAllocatorCache(self.device)
            _register_dlo_component_cache(
                self._dlo_component_cache,
                *(getattr(self, name, None) for name in self._vae_modules),
            )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()
        model_path = Path(
            resolve_minimax_h3_decoder_model_path(str(od_config.model), od_config.revision, od_config.task_type)
        )
        self.video_vae = MiniMaxH3VideoVAE(
            str(model_path / "video_vae"),
            device=self.device,
            load_device=self._decoder_load_device(),
            pin_memory=od_config.pin_cpu_memory,
            decode_only=True,
            trust_remote_code=od_config.trust_remote_code,
        )
        self.audio_vae = MiniMaxH3AudioVAE(
            str(model_path / "audio_vae"),
            device=self.device,
            load_device=self._decoder_load_device(),
            pin_memory=od_config.pin_cpu_memory,
            decode_only=True,
            trust_remote_code=od_config.trust_remote_code,
        )
        self.vae = self.video_vae
        self._setup_decoder_component_cache()
        self.weights_sources = []
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Both VAEs are loaded from native checkpoints in the constructor.
        del weights
        return {name for name, _ in self.named_parameters()}

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise OmniClientError("MiniMax H3 decoder supports one request at a time")
        payload = request.prompts[0]["additional_information"]["minimax_h3_decode"]
        videos, audios = [], []
        frame_counts = []
        for video_latent, audio_latent in zip(payload["video_latents"], payload["audio_latents"], strict=True):
            video_latent = video_latent.to(self.device)
            audio_latent = audio_latent.to(self.device)
            if payload["preencode_mp4"]:
                videos.append(
                    self.decode_to_mp4(
                        video_latent,
                        audio_latent,
                        height=payload["height"],
                        width=payload["width"],
                        video_codec_options=payload["video_codec_options"],
                        batch_frames=payload["preencode_batch_frames"],
                        frame_counts=frame_counts,
                        trim=payload.get("trim"),
                    )
                )
                audios.append(None)
            else:
                video, audio = self.decode(video_latent, audio_latent, height=payload["height"], width=payload["width"])
                if payload.get("trim") is not None:
                    # A continuation's hidden overlap was decoded only as causal context.
                    video, audio = trim_decoded_overlap(video, audio, **payload["trim"])
                video = self._offload_model_cpu_stage_output(_prepare_minimax_h3_video_output(video))
                videos.append(video)
                audios.append(audio)
                del video
                self._release_stage_cache()
            del video_latent, audio_latent
        if isinstance(videos[0], bytes):
            video = videos[0] if len(videos) == 1 else videos
            audio = None
        else:
            video = videos[0] if len(videos) == 1 else torch.cat(videos, dim=0)
            audio = audios[0] if len(audios) == 1 else torch.cat(audios, dim=0)
        return DiffusionOutput(
            output=(
                _MiniMaxH3EncodedOutput(video=video, frame_counts=frame_counts) if frame_counts else (video, audio)
            ),
            post_process_func=get_minimax_h3_post_process_func(self.od_config),
            stage_durations=(self.stage_durations if hasattr(self, "_stage_durations") else {}),
        )


def _video_post_process(output):
    """Post-process the output of the video-only decoder stage."""
    if isinstance(output, (bytes, bytearray, memoryview)):
        output = [bytes(output)]
    if isinstance(output, list) and all(isinstance(item, (bytes, bytearray, memoryview)) for item in output):
        return {"video": [bytes(item) for item in output], "fps": MINIMAX_H3_FPS}
    if not isinstance(output, torch.Tensor) or output.dtype != torch.uint8 or output.ndim != 5:
        raise ValueError(f"MiniMax-H3 video decoder expects uint8 (B,T,H,W,C), got {type(output)!r}")
    return {"video": [sample.detach().cpu().numpy() for sample in output], "fps": MINIMAX_H3_FPS}


def _audio_post_process(output):
    """Post-process the output of the audio-only decoder stage."""
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"MiniMax-H3 audio decoder expects a tensor, got {type(output)!r}")
    return {"audio": output.detach().float().cpu().numpy(), "audio_sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE}


def get_minimax_h3_video_decoder_post_process_func(od_config: OmniDiffusionConfig):
    del od_config
    return _video_post_process


def get_minimax_h3_audio_decoder_post_process_func(od_config: OmniDiffusionConfig):
    del od_config
    return _audio_post_process


class _MiniMaxH3ModalityDecoderPipeline(MiniMaxH3DecoderPipeline):
    """Shared implementation for an isolated Video or Audio VAE stage."""

    modality: ClassVar[str]
    _vae_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        nn.Module.__init__(self)
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()
        model_path = Path(
            resolve_minimax_h3_decoder_model_path(str(od_config.model), od_config.revision, od_config.task_type)
        )
        if self.modality == "video":
            self.video_vae = MiniMaxH3VideoVAE(
                str(model_path / "video_vae"),
                device=self.device,
                load_device=self._decoder_load_device(),
                pin_memory=od_config.pin_cpu_memory,
                decode_only=True,
                trust_remote_code=od_config.trust_remote_code,
            )
            self.vae = self.video_vae
        else:
            # The native audio codec has no distributed decode. Reserve the
            # configured audio group, but load and execute only on its leader.
            # Additional audio devices do not imply single-output acceleration.
            self.audio_vae = (
                MiniMaxH3AudioVAE(
                    str(model_path / "audio_vae"),
                    device=self.device,
                    load_device=self._decoder_load_device(),
                    pin_memory=od_config.pin_cpu_memory,
                    decode_only=True,
                    trust_remote_code=od_config.trust_remote_code,
                )
                if self._is_output_owner_rank()
                else None
            )
            self.vae = self.audio_vae
            if self.audio_vae is None:
                self._vae_modules = []
        self._setup_decoder_component_cache()
        self.weights_sources = []
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=[] if self.modality == "audio" and self.audio_vae is None else None,
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise OmniClientError("MiniMax H3 modality decoder supports one request at a time")
        payload = request.prompts[0]["additional_information"]["minimax_h3_decode"]
        if payload.get("preencode_mp4", False):
            raise OmniClientError("parallel H3 decoding requires preencode_mp4=false; use serial for incremental MP4")
        values = []
        latent_key = f"{self.modality}_latents"
        if not payload.get(latent_key):
            raise OmniClientError(f"MiniMax H3 decoder received no {self.modality} latents")
        if self.modality == "audio" and self.audio_vae is None:
            return DiffusionOutput(output=None)
        self._release_stage_cache()
        for latent in payload[latent_key]:
            latent = latent.to(self.device)
            if self.modality == "video":
                with self._component_on_device(self.video_vae.decoder_component):
                    with current_omni_platform.create_autocast_context(
                        device_type=self.device.type,
                        dtype=self.video_vae.decode_dtype,
                        enabled=True,
                    ):
                        value = self.video_vae.decode_latent(latent)
                # All ranks participate in tiled decode, but only the reply
                # rank needs the full video in transfer layout.
                if not self._is_output_owner_rank():
                    del value, latent
                    self._release_stage_cache()
                    continue
                value = value[..., : int(payload["height"]), : int(payload["width"])].contiguous()
                value = self._offload_model_cpu_stage_output(_prepare_minimax_h3_video_output(value))
            else:
                with self._component_on_device(self.audio_vae):
                    value = self.audio_vae.decode_latent(latent)
                value = self._offload_model_cpu_stage_output(value)
            values.append(value)
            del latent
            self._release_stage_cache()
        if not values:
            return DiffusionOutput(output=None)
        value = values[0] if len(values) == 1 else torch.cat(values, dim=0)
        return DiffusionOutput(
            output=value,
            post_process_func=_video_post_process if self.modality == "video" else _audio_post_process,
            stage_durations=(self.stage_durations if hasattr(self, "_stage_durations") else {}),
        )


class MiniMaxH3VideoDecoderPipeline(_MiniMaxH3ModalityDecoderPipeline):
    modality = "video"
    support_audio_output: ClassVar[bool] = False
    _vae_modules: ClassVar[list[str]] = ["video_vae"]
    _PROFILER_TARGETS: ClassVar[list[str]] = ["video_vae.decode_latent"]


class MiniMaxH3AudioDecoderPipeline(_MiniMaxH3ModalityDecoderPipeline):
    modality = "audio"
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = MINIMAX_H3_AUDIO_SAMPLE_RATE
    _vae_modules: ClassVar[list[str]] = ["audio_vae"]
    _PROFILER_TARGETS: ClassVar[list[str]] = ["audio_vae.decode_latent"]

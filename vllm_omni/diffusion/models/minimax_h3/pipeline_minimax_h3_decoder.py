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
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.minimax_h3.checkpoint import resolve_minimax_h3_partition

from .pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    _prepare_minimax_h3_video_output,
    get_minimax_h3_post_process_func,
)
from .vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE


def _resolve_decoder_model_path(od_config: OmniDiffusionConfig) -> Path:
    model = str(od_config.model)
    partition = resolve_minimax_h3_partition(model, od_config.task_type, auto_partition="fl2va")
    subdir = "Ref2VA" if partition == "ref2va" else "FL2VA"
    path = Path(model)
    if path.is_dir():
        if path.name in {"FL2VA", "Ref2VA"}:
            path = path.parent
        return path / subdir
    # Decoder-only deployment downloads native VAE assets, without DiT or text weights.
    snapshot = download_weights_from_hf_specific(
        model_name_or_path=model,
        cache_dir=None,
        allow_patterns=[f"{subdir}/video_vae/**", f"{subdir}/audio_vae/**"],
        revision=od_config.revision,
        require_all=True,
    )
    return Path(snapshot) / subdir


class MiniMaxH3DecoderPipeline(
    nn.Module, DiffusionPipelineProfilerMixin, SupportAudioOutput, SupportsComponentDiscovery
):
    supports_step_execution: ClassVar[bool] = False
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["video_vae", "audio_vae"]
    _PROFILER_TARGETS: ClassVar[list[str]] = ["decode", "video_vae.decode_latent", "audio_vae.decode_latent"]

    # Reuse decoding behavior to preserve regular and chunked MP4 output contracts.
    decode = MiniMaxH3Pipeline.decode
    decode_to_mp4 = MiniMaxH3Pipeline.decode_to_mp4
    _component_on_device = MiniMaxH3Pipeline._component_on_device
    _uses_manual_component_offload = MiniMaxH3Pipeline._uses_manual_component_offload
    _offload_model_cpu_stage_output = MiniMaxH3Pipeline._offload_model_cpu_stage_output
    _release_stage_cache = MiniMaxH3Pipeline._release_stage_cache
    _is_output_owner_rank = staticmethod(MiniMaxH3Pipeline._is_output_owner_rank)

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()
        model_path = _resolve_decoder_model_path(od_config)
        self.video_vae = MiniMaxH3VideoVAE(
            str(model_path / "video_vae"),
            device=self.device,
            decode_only=True,
            trust_remote_code=od_config.trust_remote_code,
        )
        self.audio_vae = MiniMaxH3AudioVAE(
            str(model_path / "audio_vae"),
            device=self.device,
            decode_only=True,
            trust_remote_code=od_config.trust_remote_code,
        )
        self.vae = self.video_vae
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
                    )
                )
                audios.append(None)
            else:
                video, audio = self.decode(video_latent, audio_latent, height=payload["height"], width=payload["width"])
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
            output=(video, audio),
            post_process_func=get_minimax_h3_post_process_func(self.od_config),
            stage_durations=(self.stage_durations if hasattr(self, "_stage_durations") else {}),
        )

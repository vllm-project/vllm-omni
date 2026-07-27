"""SoulX-Singer SVC (voice conversion) pipeline implementation."""

import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.soulx_singer.modules import (
    CFMDecoder,
    WhisperEncoder,
)
from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_base import (
    FlowMatchingAudioPipeline,
    convert_soulx_audio_output_to_numpy,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
    SOULX_PREPROCESSED_KEY,
    get_soulx_preprocessed_payload,
    has_precomputed,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.pre_process import (
    attach_preprocess_for_diffusion_request,
    is_warmup_request,
    resolve_preprocess_audio,
)
from vllm_omni.diffusion.models.soulx_singer.utils import (
    f0_to_coarse,
    load_config,
    resolve_pitch_shift,
    validate_soulx_extra_args,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)


def get_soulxsinger_svc_pre_process_func(od_config: OmniDiffusionConfig):
    hf_config = load_config(os.path.join(od_config.model, "config.yaml"))
    sample_rate = hf_config.audio.sample_rate
    device = get_local_device()
    _pipeline = None

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        nonlocal _pipeline

        extra_args = validate_soulx_extra_args(
            "svc",
            dict(getattr(request.sampling_params, "extra_args", None) or {}),
        )

        # Inline build: when no warmup/no precomputed paths/no IPC payload,
        # build the preprocess payload directly from audio file paths.
        if not (is_warmup_request(request) or has_precomputed(extra_args, "svc")):
            prompt = request.prompt
            if not isinstance(prompt, str) and not get_soulx_preprocessed_payload(prompt):  # type: ignore[arg-type]
                prompt_audio, target_audio = resolve_preprocess_audio(prompt, extra_args)  # type: ignore[arg-type]
                if prompt_audio is not None and target_audio is not None:
                    if _pipeline is None:
                        from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.pipeline import (
                            SoulXPreprocessPipeline,
                        )

                        _pipeline = SoulXPreprocessPipeline(
                            od_config=od_config,
                            vocal_sep=bool(extra_args.get("vocal_sep", False)),
                            verbose=bool(extra_args.get("preprocess_verbose", False)),
                            extra_args=extra_args,
                        )
                    payload = _pipeline.build_svc_payload_from_audio(
                        prompt_audio=prompt_audio,
                        target_audio=target_audio,
                        sample_rate=sample_rate,
                        device=device,
                        vocal_sep=extra_args.get("vocal_sep"),
                    )
                    payload.setdefault("kind", "svc")
                    prompt.setdefault("additional_information", {})[SOULX_PREPROCESSED_KEY] = payload  # type: ignore[union-attr, assignment]

        return attach_preprocess_for_diffusion_request(
            request,
            kind="svc",
            sample_rate=sample_rate,
            device=device,
        )

    return pre_process_func


def get_soulxsinger_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(audio: torch.Tensor):
        return convert_soulx_audio_output_to_numpy(audio)

    return post_process_func


class PipelineSoulXSingerSVC(FlowMatchingAudioPipeline):
    """SVC pipeline for the SoulX-Singer model."""

    _encoder_modules: ClassVar[list[str]] = [
        "whisper_encoder",
        "mel",
        "f0_encoder",
    ]

    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "prompt_wav_path",
            "target_wav_path",
            "prompt_f0_path",
            "target_f0_path",
            "prompt_audio",
            "target_audio",
            "vocal_sep",
            "preprocess_weights_dir",
            "preprocess_verbose",
            "auto_shift",
            "pitch_shift",
        }
    )

    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset({"pitch_shift"})

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)

        self.f0_encoder = nn.Embedding(self.f0_bin, self.f0_dim)

        self.cfm_decoder = CFMDecoder(self.flow_matching_config)

        self.mel, self.vocoder = self._build_fp32_audio_modules(self.audio_config)
        with set_default_torch_dtype(torch.float32):
            self.whisper_encoder = WhisperEncoder(device=self.device)

        self._setup_soulx_profiler()

    def _encode_condition(self, *, whisper_features: torch.Tensor, f0_coarse: torch.Tensor) -> torch.Tensor:
        cond = whisper_features + self.f0_encoder(f0_coarse)
        return self._to_trunk_dtype(cond)[0]

    def _encode_prompt_whisper_feature(self, prompt_wav: torch.Tensor) -> torch.Tensor:
        trunk_dtype = self.f0_encoder.weight.dtype
        with self._stage_timer("whisper"):
            return self.whisper_encoder.encode(
                prompt_wav,
                sr=self.audio_config.sample_rate,
                output_dtype=trunk_dtype,
            )

    _encode_target_whisper_feature = _encode_prompt_whisper_feature

    def _infer_segment(
        self,
        *,
        prompt_mel: torch.Tensor,
        prompt_wav: torch.Tensor,
        target_wav: torch.Tensor,
        prompt_f0: torch.Tensor,
        target_f0: torch.Tensor,
        pitch_shift: int,
        num_inference_steps: int,
        guidance_scale: float,
        prompt_feature: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Single-segment SVC inference aligned with ``SoulXSingerSVC.infer_segment``."""
        len_prompt_mel = prompt_mel.shape[1]
        prompt_f0 = F.pad(prompt_f0, (0, 0, 0, max(0, len_prompt_mel - prompt_f0.shape[1])))[:, :len_prompt_mel]

        f0_coarse_prompt = f0_to_coarse(prompt_f0)
        f0_coarse_target = f0_to_coarse(target_f0, f0_shift=int(pitch_shift * 5))
        f0_coarse = torch.cat([f0_coarse_prompt, f0_coarse_target], dim=1)

        trunk_dtype = self.f0_encoder.weight.dtype
        if prompt_feature is None:
            prompt_feature = self._encode_prompt_whisper_feature(prompt_wav)
        with self._stage_timer("whisper"):
            target_feature = self.whisper_encoder.encode(
                target_wav,
                sr=self.audio_config.sample_rate,
                output_dtype=trunk_dtype,
            )
        prompt_feature = F.pad(
            prompt_feature,
            (0, 0, 0, max(0, f0_coarse_prompt.shape[1] - prompt_feature.shape[1])),
        )[:, : f0_coarse_prompt.shape[1], :]
        target_feature = F.pad(
            target_feature,
            (0, 0, 0, max(0, f0_coarse_target.shape[1] - target_feature.shape[1])),
        )[:, : f0_coarse_target.shape[1], :]

        whisper_features = torch.cat([prompt_feature, target_feature], dim=1)
        with self._stage_timer("cond_encode"):
            cond = self._encode_condition(whisper_features=whisper_features, f0_coarse=f0_coarse)

        with self._stage_timer("cfm"):
            generated_mel = self._run_flow_matching_loop(
                prompt=prompt_mel,
                cond=cond,
                n_timesteps=num_inference_steps,
                cfg=guidance_scale,
                generator=generator,
            )

        with self._stage_timer("vocoder"):
            generated_audio = self._mel_to_audio(generated_mel, squeeze=True)
        target_len = target_wav.shape[-1]
        if generated_audio.shape[-1] > target_len:
            generated_audio = generated_audio[:target_len]
        elif generated_audio.shape[-1] < target_len:
            generated_audio = F.pad(generated_audio, (0, target_len - generated_audio.shape[-1]))
        return generated_audio

    def _build_vocal_segments(
        self,
        f0: torch.Tensor,
        *,
        hop_size: int,
        sample_rate: int,
        uv_frames_th: int = 10,
        min_duration_sec: float = 15.0,
        max_duration_sec: float = 30.0,
        ignore_silent: bool = True,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Build vocal segments based on F0 contour.

        Mirrors upstream ``SoulXSingerSVC.build_vocal_segments``.
        Returns (overlap_segments, segments) tuples of (start_sec, end_sec).
        """
        f0_np = (
            f0.detach().float().cpu().numpy().squeeze() if isinstance(f0, torch.Tensor) else np.asarray(f0).squeeze()
        )
        total_frames = int(f0_np.shape[0])
        if total_frames == 0:
            return [], []

        f0_rate = sample_rate // hop_size
        min_frames = max(1, int(round(min_duration_sec * f0_rate)))
        max_frames = max(1, int(round(max_duration_sec * f0_rate)))

        split_points = [0]

        def append_split_point(point: int) -> None:
            point = int(max(0, min(point, total_frames)))
            while point - split_points[-1] > max_frames:
                split_points.append(split_points[-1] + max_frames)
            if point > split_points[-1]:
                split_points.append(point)

        idx = 0
        while idx < total_frames:
            if f0_np[idx] == 0:
                run_start = idx
                while idx < total_frames and f0_np[idx] == 0:
                    idx += 1
                run_end = idx
                if (run_end - run_start) >= uv_frames_th:
                    split_point = max(run_end - 5, (run_start + run_end) // 2)
                    append_split_point(split_point)
            else:
                idx += 1
        append_split_point(total_frames)

        segments: list[tuple[int, int]] = []
        overlap_segments: list[tuple[int, int]] = []
        num_overlaps = 1

        def append_segment(start_idx: int, end_idx: int, overlaps: int = num_overlaps) -> None:
            segments.append((split_points[start_idx], split_points[end_idx]))
            overlap_start_idx = start_idx
            if start_idx > 0 and (split_points[end_idx] - split_points[start_idx - overlaps]) <= max_frames:
                overlap_start_idx = start_idx - overlaps
            overlap_segments.append((split_points[overlap_start_idx], split_points[end_idx]))

        seg_start, seg_end = 0, 1
        while seg_start < len(split_points) - 1:
            while seg_end < len(split_points) and (split_points[seg_end] - split_points[seg_start]) < min_frames:
                seg_end += 1
            if seg_end >= len(split_points):
                append_segment(seg_start, len(split_points) - 1)
                break
            append_segment(seg_start, seg_end)
            seg_start = seg_end
            seg_end = seg_start + 1

        if ignore_silent:
            filtered_idx = []
            for i, (ov_start, ov_end) in enumerate(overlap_segments):
                voice_ratio = np.sum(f0_np[ov_start:ov_end] > 0) / max(1, ov_end - ov_start)
                voiced_frames = np.sum(f0_np[ov_start:ov_end] > 0)
                if voice_ratio > 0.05 and voiced_frames >= 10:
                    filtered_idx.append(i)
            overlap_segments = [overlap_segments[i] for i in filtered_idx]
            segments = [segments[i] for i in filtered_idx]

        # convert indices to seconds
        overlap_sec = [(s / f0_rate, e / f0_rate) for s, e in overlap_segments]
        seg_sec = [(s / f0_rate, e / f0_rate) for s, e in segments]
        return overlap_sec, seg_sec

    def infer_svc_batch(
        self,
        payload: dict[str, Any],
        *,
        extra_args: dict[str, Any],
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Batch SVC with chunked inference for long audio (matches upstream)."""
        prompt_wav = payload["prompt_wav"]
        target_wav = payload["target_wav"]
        prompt_f0 = payload["prompt_f0"]
        target_f0 = payload["target_f0"]

        pitch_shift = resolve_pitch_shift(
            auto_shift=bool(extra_args.get("auto_shift", True)),
            manual_shift=int(extra_args.get("pitch_shift", 0)),
            prompt_f0=prompt_f0,
            target_f0=target_f0,
        )

        prompt_mel = self._mel_from_wav(prompt_wav)

        # Long audio: chunk by vocal segments (same as upstream)
        max_sec = 30.0 * self.audio_config.sample_rate
        if target_wav.shape[-1] < max_sec:
            # Short audio: one-shot
            generated_audio = self._infer_segment(
                prompt_mel=prompt_mel,
                prompt_wav=prompt_wav,
                target_wav=target_wav,
                prompt_f0=prompt_f0,
                target_f0=target_f0,
                pitch_shift=pitch_shift,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            return generated_audio.unsqueeze(0), pitch_shift

        # Build vocal segments
        overlap_segments, segments = self._build_vocal_segments(
            target_f0,
            hop_size=self.audio_config.hop_size,
            sample_rate=self.audio_config.sample_rate,
        )
        if len(segments) == 0:
            segments = [(0.0, target_wav.shape[-1] / self.audio_config.sample_rate)]
            overlap_segments = [(0.0, target_wav.shape[-1] / self.audio_config.sample_rate)]

        f0_rate = self.audio_config.sample_rate // self.audio_config.hop_size
        generated_audio = torch.zeros_like(target_wav)

        for idx in range(len(segments)):
            overlap_start_sec, overlap_end_sec = overlap_segments[idx]
            seg_start_sec, seg_end_sec = segments[idx]

            wav_start = int(round(overlap_start_sec * self.audio_config.sample_rate))
            wav_end = int(round(overlap_end_sec * self.audio_config.sample_rate))
            f0_start = int(round(overlap_start_sec * f0_rate))
            f0_end = int(round(overlap_end_sec * f0_rate))

            wav_start = max(0, min(wav_start, target_wav.shape[-1]))
            wav_end = max(wav_start, min(wav_end, target_wav.shape[-1]))
            f0_start = max(0, min(f0_start, target_f0.shape[-1]))
            f0_end = max(f0_start, min(f0_end, target_f0.shape[-1]))

            segment_gt_wav = target_wav[:, wav_start:wav_end]
            segment_gt_f0 = target_f0[:, f0_start:f0_end]

            segment_generated = self._infer_segment(
                prompt_mel=prompt_mel,
                prompt_wav=prompt_wav,
                target_wav=segment_gt_wav,
                prompt_f0=prompt_f0,
                target_f0=segment_gt_f0,
                pitch_shift=pitch_shift,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            seg_start = int(round(seg_start_sec * self.audio_config.sample_rate))
            seg_end = int(round(seg_end_sec * self.audio_config.sample_rate))
            segment_generated = segment_generated[seg_start - wav_start : seg_end - wav_start]
            generated_audio[:, seg_start:seg_end] = segment_generated

        return generated_audio.unsqueeze(0), pitch_shift

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        return self._forward_batch_from_request(
            req,
            kind="svc",
            metadata_key="pitch_shift",
            infer_batch_fn=self.infer_svc_batch,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        self._load_soulx_checkpoint("model-svc.pt")
        return {name for name, _ in self.named_parameters()}

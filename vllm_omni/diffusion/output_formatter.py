# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeGuard

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.io_support import supports_audio_output
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniPromptType
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.output_metadata import (
    DiffusionMetadata,
    DiffusionMultimodalOutput,
    DiffusionOutputEnvelope,
    DiffusionPayload,
    DiffusionPayloadValue,
    DiffusionPostprocessRawOutput,
    DiffusionTrajectoryPayload,
    strip_internal_metadata,
    validate_diffusion_metadata,
    validate_public_diffusion_metadata,
)


@dataclass(frozen=True)
class DiffusionPostprocessOutput:
    outputs: DiffusionPayload
    metadata: DiffusionMetadata = field(default_factory=dict)
    primary_key: str | None = None


def _is_output_envelope(outputs: DiffusionPostprocessRawOutput) -> TypeGuard[DiffusionOutputEnvelope]:
    return isinstance(outputs, dict) and isinstance(outputs.get("payload"), dict)


def normalize_diffusion_postprocess_output(
    outputs: DiffusionPostprocessRawOutput,
) -> DiffusionPostprocessOutput:
    """Normalize diffusion postprocess output into payload and metadata."""

    if _is_output_envelope(outputs):
        payload: DiffusionPayload = outputs.get("payload") or {}
        metadata = outputs.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        validate_diffusion_metadata(metadata)

        public_metadata = strip_internal_metadata(metadata)
        if "text" in payload and "text" not in public_metadata:
            public_metadata = {**public_metadata, "text": {}}
        validate_public_diffusion_metadata(public_metadata)

        return DiffusionPostprocessOutput(
            outputs=payload,
            metadata=public_metadata,
            primary_key=_infer_primary_payload_key(payload),
        )

    if isinstance(outputs, dict):
        payload = {key: value for key, value in outputs.items() if key not in {"audio_sample_rate", "fps"}}
        metadata = _metadata_from_legacy_payload(outputs)
        if "text" in payload and "text" not in metadata:
            metadata["text"] = {}
        if metadata:
            validate_public_diffusion_metadata(metadata)
        return DiffusionPostprocessOutput(
            outputs=payload,
            metadata=metadata,
            primary_key=_infer_primary_payload_key(payload),
        )

    return DiffusionPostprocessOutput(
        outputs={"output": outputs},
        primary_key="output",
    )


def _metadata_from_legacy_payload(payload: dict[str, DiffusionPayloadValue]) -> DiffusionMetadata:
    metadata: DiffusionMetadata = {}
    audio_sample_rate = payload.get("audio_sample_rate")
    if audio_sample_rate is not None:
        metadata["audio"] = {"sample_rate": audio_sample_rate}
    fps = payload.get("fps")
    if fps is not None:
        metadata["video"] = {"fps": fps}
    return metadata


def _infer_primary_payload_key(payload: DiffusionPayload) -> str | None:
    for key in ("video", "image", "text", "audio", "output"):
        if key in payload:
            return key
    if set(payload).issubset({"actions", "trajectory"}):
        return None
    if "actions" in payload:
        return None
    return next(iter(payload), None)


def format_empty_diffusion_outputs(
    request: OmniDiffusionRequest,
    *,
    finished: bool = True,
) -> list[OmniRequestOutput]:
    return [
        OmniRequestOutput.from_diffusion(
            request_id=request.request_id,
            images=[],
            prompt=request.prompt,
            metrics={},
            latents=None,
            finished=finished,
        )
    ]


def format_diffusion_outputs(
    *,
    request: OmniDiffusionRequest,
    od_config: OmniDiffusionConfig,
    diffusion_output: DiffusionOutput,
    output_data: DiffusionPostprocessRawOutput,
    postprocess_output: DiffusionPostprocessOutput,
) -> list[OmniRequestOutput]:
    """Convert a finished diffusion model output into API-facing outputs."""

    primary_payload = _primary_payload(postprocess_output)
    outputs = _ensure_list(primary_payload)
    metrics = {
        "image_num": int(request.sampling_params.num_outputs_per_prompt),
        "resolution": (
            int(request.sampling_params.resolution) if request.sampling_params.resolution is not None else None
        ),
        "width": request.sampling_params.width,
        "height": request.sampling_params.height,
    }

    # Only the primary payload determines the response type. Some image models
    # include reasoning text in metadata, but their final output is still an image.
    is_text_output = postprocess_output.primary_key == "text"

    is_audio_output = supports_audio_output(od_config.model_class_name)
    audio_sample_rate = _metadata_audio_sample_rate(postprocess_output.metadata)
    if is_audio_output and audio_sample_rate is None:
        model_cls = DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
        audio_sample_rate = getattr(model_cls, "audio_sample_rate", None)

    return _format_single_prompt_output(
        request=request,
        prompt=request.prompt,
        diffusion_output=diffusion_output,
        outputs=outputs,
        metrics=metrics,
        postprocess_output=postprocess_output,
        is_text_output=is_text_output,
        is_audio_output=is_audio_output,
        audio_sample_rate=audio_sample_rate,
        finished=diffusion_output.finished,
    )


def _ensure_list(outputs: DiffusionPayloadValue) -> list[DiffusionPayloadValue]:
    if isinstance(outputs, list):
        return outputs
    return [outputs] if outputs is not None else []


def _primary_payload(postprocess_output: DiffusionPostprocessOutput) -> DiffusionPayloadValue | None:
    if postprocess_output.primary_key is None:
        return []
    return postprocess_output.outputs.get(postprocess_output.primary_key)


def _metadata_audio_sample_rate(metadata: DiffusionMetadata) -> int | None:
    audio_metadata = metadata.get("audio")
    if not isinstance(audio_metadata, dict):
        return None
    sample_rate = audio_metadata.get("sample_rate")
    return sample_rate if isinstance(sample_rate, int) else None


def _metadata_video_fps(metadata: DiffusionMetadata) -> float | None:
    video_metadata = metadata.get("video")
    if not isinstance(video_metadata, dict):
        return None
    fps = video_metadata.get("fps")
    return fps if isinstance(fps, (int, float)) and not isinstance(fps, bool) else None


def _format_audio_multimodal_output(
    payload: DiffusionPayloadValue,
    audio_sample_rate: int | None,
    metadata: DiffusionMetadata | None = None,
) -> DiffusionMultimodalOutput:
    mm_output: DiffusionMultimodalOutput = {"audio": payload}
    if metadata:
        mm_output["metadata"] = metadata
    if audio_sample_rate is not None:
        mm_output["audio_sample_rate"] = audio_sample_rate
    return mm_output


def _has_non_audio_postprocess_payload(postprocess_output: DiffusionPostprocessOutput) -> bool:
    return (
        "video" in postprocess_output.outputs
        or "image" in postprocess_output.outputs
        or "text" in postprocess_output.metadata
        or "actions" in postprocess_output.outputs
        or "trajectory" in postprocess_output.outputs
        or _metadata_video_fps(postprocess_output.metadata) is not None
    )


def _build_multimodal_output(
    postprocess_output: DiffusionPostprocessOutput,
    audio_sample_rate: int | None,
) -> DiffusionMultimodalOutput:
    mm_output: DiffusionMultimodalOutput = {}
    if postprocess_output.metadata:
        mm_output["metadata"] = postprocess_output.metadata
    for key, value in postprocess_output.outputs.items():
        if key in {"audio", "actions", "trajectory"}:
            mm_output[key] = value
    if audio_sample_rate is not None:
        mm_output["audio_sample_rate"] = audio_sample_rate
    if (fps := _metadata_video_fps(postprocess_output.metadata)) is not None:
        mm_output["fps"] = fps
    return mm_output


def _format_single_prompt_output(
    *,
    request: OmniDiffusionRequest,
    prompt: OmniPromptType,
    diffusion_output: DiffusionOutput,
    outputs: list[DiffusionPayloadValue],
    metrics: dict[str, object],
    postprocess_output: DiffusionPostprocessOutput,
    is_text_output: bool,
    is_audio_output: bool,
    audio_sample_rate: int | None,
    finished: bool = True,
) -> list[OmniRequestOutput]:
    request_id = request.request_id
    mm_output = _build_multimodal_output(postprocess_output, audio_sample_rate)
    if is_text_output:
        mm_output["text"] = outputs[0] if len(outputs) == 1 else outputs
    trajectory_payload = _trajectory_payload(postprocess_output, diffusion_output)
    trajectory_latents = trajectory_payload.get("latents")
    trajectory_timesteps = trajectory_payload.get("timesteps")
    trajectory_log_probs = trajectory_payload.get("log_probs")
    trajectory_decoded = trajectory_payload.get("decoded")

    if is_text_output:
        return [
            OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=[],
                prompt=prompt,
                metrics=metrics,
                multimodal_output=mm_output,
                final_output_type="text",
                stage_durations=diffusion_output.stage_durations,
                peak_memory_mb=diffusion_output.peak_memory_mb,
                finished=finished,
            ),
        ]

    if is_audio_output and not _has_non_audio_postprocess_payload(postprocess_output):
        request_audio_payload = postprocess_output.outputs.get("audio")
        if request_audio_payload is None:
            request_audio_payload = outputs[0] if len(outputs) == 1 else outputs
        return [
            OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=[],
                prompt=prompt,
                metrics=metrics,
                latents=trajectory_latents,
                trajectory_latents=trajectory_latents,
                trajectory_timesteps=trajectory_timesteps,
                trajectory_log_probs=trajectory_log_probs,
                trajectory_decoded=trajectory_decoded,
                multimodal_output=_format_audio_multimodal_output(
                    request_audio_payload,
                    audio_sample_rate,
                    postprocess_output.metadata,
                ),
                final_output_type="audio",
                stage_durations=diffusion_output.stage_durations,
                peak_memory_mb=diffusion_output.peak_memory_mb,
                finished=finished,
            ),
        ]

    return [
        OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=outputs,
            prompt=prompt,
            metrics=metrics,
            latents=trajectory_latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            trajectory_log_probs=trajectory_log_probs,
            trajectory_decoded=trajectory_decoded,
            multimodal_output=mm_output,
            stage_durations=diffusion_output.stage_durations,
            peak_memory_mb=diffusion_output.peak_memory_mb,
            finished=finished,
        ),
    ]


def _trajectory_payload(
    postprocess_output: DiffusionPostprocessOutput,
    diffusion_output: DiffusionOutput,
) -> DiffusionTrajectoryPayload:
    trajectory: DiffusionTrajectoryPayload = {}
    payload = postprocess_output.outputs.get("trajectory")
    if isinstance(payload, Mapping):
        for source_key, target_key in (
            ("latents", "latents"),
            ("timesteps", "timesteps"),
            ("log_probs", "log_probs"),
            ("decoded", "decoded"),
        ):
            value = payload.get(source_key)
            if value is not None:
                trajectory[target_key] = value

    fallback_fields = (
        ("latents", diffusion_output.trajectory_latents),
        ("timesteps", diffusion_output.trajectory_timesteps),
        ("log_probs", diffusion_output.trajectory_log_probs),
        ("decoded", diffusion_output.trajectory_decoded),
    )
    for key, value in fallback_fields:
        if key not in trajectory and value is not None:
            trajectory[key] = value
    return trajectory

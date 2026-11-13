# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeGuard, cast

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.io_support import get_diffusion_output_type, supports_audio_output
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
from vllm_omni.sample_rate import SAMPLE_RATE_KEY


@dataclass(frozen=True)
class DiffusionPostprocessOutput:
    outputs: DiffusionPayload
    metadata: DiffusionMetadata = field(default_factory=dict)
    primary_key: str | None = None


def _is_output_envelope(outputs: DiffusionPostprocessRawOutput) -> TypeGuard[DiffusionOutputEnvelope]:
    return isinstance(outputs, dict) and isinstance(outputs.get("payload"), dict)


# Flat media keys a pipeline may leave on a payload instead of using the
# metadata groups; normalize_* lifts them into ``metadata`` and strips them
# from the payload either way.
_FLAT_MEDIA_METADATA_KEYS = frozenset({"audio_sample_rate", SAMPLE_RATE_KEY, "fps"})


def _lift_flat_media_metadata(payload: DiffusionPayload, metadata: DiffusionMetadata) -> DiffusionMetadata:
    """Fold flat media rates left on an envelope payload into the metadata groups.

    Envelope metadata wins: a flat key only fills a field the envelope did not
    provide. Without this lift, a pipeline returning an envelope with a flat
    rate on the payload (kandinsky6) would ship unlabeled audio — muxers would
    then fall back to their default rate.
    """
    rate = payload.get(SAMPLE_RATE_KEY)
    if rate is None:
        rate = payload.get("audio_sample_rate")
    if rate is not None:
        audio_metadata = metadata.get("audio")
        audio_metadata = dict(audio_metadata) if isinstance(audio_metadata, dict) else {}
        if "sample_rate" not in audio_metadata:
            audio_metadata["sample_rate"] = rate
            metadata = {**metadata, "audio": audio_metadata}
    fps = payload.get("fps")
    if fps is not None:
        video_metadata = metadata.get("video")
        video_metadata = dict(video_metadata) if isinstance(video_metadata, dict) else {}
        if "fps" not in video_metadata:
            video_metadata["fps"] = fps
            metadata = {**metadata, "video": video_metadata}
    return metadata


def normalize_diffusion_postprocess_output(
    outputs: DiffusionPostprocessRawOutput,
) -> DiffusionPostprocessOutput:
    """Normalize diffusion postprocess output into payload and metadata."""

    if _is_output_envelope(outputs):
        payload: DiffusionPayload = outputs.get("payload") or {}
        metadata = outputs.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = _lift_flat_media_metadata(payload, metadata)
        payload = {key: value for key, value in payload.items() if key not in _FLAT_MEDIA_METADATA_KEYS}
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
        payload = {key: value for key, value in outputs.items() if key not in _FLAT_MEDIA_METADATA_KEYS}
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
    audio_sample_rate = payload.get(SAMPLE_RATE_KEY)
    if audio_sample_rate is None:
        audio_sample_rate = payload.get("audio_sample_rate")
    if audio_sample_rate is not None:
        metadata["audio"] = {"sample_rate": audio_sample_rate}
    fps = payload.get("fps")
    if fps is not None:
        metadata["video"] = {"fps": fps}
    return metadata


# Structured (non-media) payloads travel in ``multimodal_output`` under their own key.
STRUCTURED_PAYLOAD_KEYS = frozenset({"actions", "trajectory", "segmentation", "video_segmentation", "video_depth"})


def _infer_primary_payload_key(payload: DiffusionPayload) -> str | None:
    for key in ("video", "image", "text", "audio", "output"):
        if key in payload:
            return key
    if set(payload).issubset(STRUCTURED_PAYLOAD_KEYS):
        return None
    if "actions" in payload:
        return None
    return next(iter(payload), None)


def format_empty_diffusion_outputs(
    request: OmniDiffusionRequest,
    *,
    finished: bool = True,
    custom_output: dict[str, Any] | None = None,
) -> list[OmniRequestOutput]:
    return [
        OmniRequestOutput.from_diffusion(
            request_id=request.request_id,
            images=[],
            prompt=request.prompt,
            metrics={},
            latents=None,
            custom_output=custom_output,
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
        "num_inference_steps": request.sampling_params.num_inference_steps,
    }

    # Only the primary payload determines the response type. Some image models
    # include reasoning text in metadata, but their final output is still an image.
    is_text_output = postprocess_output.primary_key == "text"

    is_audio_output = supports_audio_output(od_config.model_class_name)
    final_output_type = (
        postprocess_output.primary_key
        if postprocess_output.primary_key in {"image", "video"}
        else get_diffusion_output_type(od_config.model_class_name)
    )
    audio_sample_rate = _metadata_audio_sample_rate(postprocess_output.metadata)
    if is_audio_output and audio_sample_rate is None:
        model_cls = DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
        audio_sample_rate = getattr(model_cls, "audio_sample_rate", None)

    stream_metadata = _stream_metadata_from_diffusion_output(diffusion_output)
    if stream_metadata:
        existing_stream = postprocess_output.metadata.get("stream")
        merged_stream = dict(existing_stream) if isinstance(existing_stream, Mapping) else {}
        merged_stream.update(stream_metadata)
        postprocess_output.metadata["stream"] = merged_stream

    return _format_single_prompt_output(
        request=request,
        prompt=request.prompt,
        diffusion_output=diffusion_output,
        outputs=outputs,
        metrics=metrics,
        postprocess_output=postprocess_output,
        is_text_output=is_text_output,
        is_audio_output=is_audio_output,
        final_output_type=final_output_type,
        audio_sample_rate=audio_sample_rate,
        finished=diffusion_output.finished,
    )


def _stream_metadata_from_diffusion_output(diffusion_output: DiffusionOutput) -> DiffusionMetadata | None:
    has_stream_data = (
        diffusion_output.total_chunks > 1
        or not diffusion_output.finished
        or bool(diffusion_output.started_event_ids)
        or bool(diffusion_output.active_event_ids)
        or bool(diffusion_output.completed_event_ids)
        or bool(diffusion_output.coalesced_event_ids)
        or bool(diffusion_output.streaming_metrics)
    )
    if not has_stream_data:
        return None
    return cast(
        DiffusionMetadata,
        {
            "generation_chunk_index": diffusion_output.chunk_index,
            **(
                {"output_chunk_index": diffusion_output.chunk_index, "pacing": diffusion_output.streaming_metrics}
                if diffusion_output.streaming_metrics
                else {}
            ),
            "started_event_ids": diffusion_output.started_event_ids,
            "active_event_ids": diffusion_output.active_event_ids,
            "completed_event_ids": diffusion_output.completed_event_ids,
            **(
                {"coalesced_event_ids": diffusion_output.coalesced_event_ids}
                if diffusion_output.coalesced_event_ids
                else {}
            ),
        },
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
        mm_output[SAMPLE_RATE_KEY] = audio_sample_rate
    return mm_output


def _has_non_audio_postprocess_payload(postprocess_output: DiffusionPostprocessOutput) -> bool:
    return (
        "video" in postprocess_output.outputs
        or "image" in postprocess_output.outputs
        or "text" in postprocess_output.metadata
        or any(key in postprocess_output.outputs for key in STRUCTURED_PAYLOAD_KEYS)
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
        if key == "audio" or key in STRUCTURED_PAYLOAD_KEYS:
            mm_output[key] = value
    if audio_sample_rate is not None:
        mm_output[SAMPLE_RATE_KEY] = audio_sample_rate
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
    final_output_type: str,
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
            final_output_type=final_output_type,
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

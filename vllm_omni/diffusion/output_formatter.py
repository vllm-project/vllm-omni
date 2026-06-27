# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.io_support import supports_audio_output
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.outputs import OmniRequestOutput


@dataclass(frozen=True)
class DiffusionStepTimings:
    preprocess_time_s: float
    exec_time_s: float
    postprocess_time_s: float
    total_time_ms: float
    step_start_ts: float = 0.0


@dataclass(frozen=True)
class DiffusionPostprocessOutput:
    outputs: Any
    custom_output: dict[str, Any]
    audio_payload: Any | None = None
    action_payload: Any | None = None
    audio_sample_rate: int | None = None
    fps: float | None = None
    has_video_payload: bool = False


def normalize_diffusion_postprocess_output(
    outputs: Any,
    custom_output: dict[str, Any],
) -> DiffusionPostprocessOutput:
    """Normalize the legacy postprocess dict shape used by diffusion models.

    The returned envelope owns a shallow merged copy of ``custom_output`` so
    postprocess values keep legacy override precedence without mutating the
    caller-owned dict.
    """

    audio_payload = None
    action_payload = None
    audio_sample_rate = None
    fps = None
    has_video_payload = False
    merged_custom_output = dict(custom_output)

    if isinstance(outputs, dict):
        has_video_payload = "video" in outputs
        audio_payload = outputs.get("audio")
        action_payload = outputs.get("actions")
        merged_custom_output.update(outputs.get("custom_output") or {})
        audio_sample_rate = outputs.get("audio_sample_rate")
        fps = outputs.get("fps")
        outputs = outputs.get("video", outputs)

    return DiffusionPostprocessOutput(
        outputs=outputs,
        custom_output=merged_custom_output,
        audio_payload=audio_payload,
        action_payload=action_payload,
        audio_sample_rate=audio_sample_rate,
        fps=fps,
        has_video_payload=has_video_payload,
    )


def format_empty_diffusion_outputs(
    request: OmniDiffusionRequest,
    *,
    finished: bool = True,
) -> list[OmniRequestOutput]:
    return [
        OmniRequestOutput.from_diffusion(
            request_id=request.request_id,
            images=[],
            prompt=prompt,
            metrics={},
            latents=None,
            finished=finished,
        )
        for prompt in request.prompts
    ]


def format_diffusion_outputs(
    *,
    request: OmniDiffusionRequest,
    od_config: OmniDiffusionConfig,
    diffusion_output: DiffusionOutput,
    output_data: Any,
    postprocess_output: DiffusionPostprocessOutput,
    timings: DiffusionStepTimings,
) -> list[OmniRequestOutput]:
    """Convert a finished diffusion model output into API-facing outputs."""

    outputs = _ensure_list(postprocess_output.outputs)
    metrics = {
        "preprocess_time_ms": timings.preprocess_time_s * 1000,
        "diffusion_engine_exec_time_ms": timings.exec_time_s * 1000,
        "diffusion_engine_total_time_ms": timings.total_time_ms,
        "image_num": int(request.sampling_params.num_outputs_per_prompt),
        "resolution": int(request.sampling_params.resolution),
        "postprocess_time_ms": timings.postprocess_time_s * 1000,
        "step_start_ts": timings.step_start_ts,
    }

    # Detect text output: when the pipeline returns a string (e.g.,
    # SenseNova-U1 / BAGEL single-stage img2text / text2text), wrap it
    # as a text-type response instead of an image.
    is_text_output = isinstance(output_data, str) and postprocess_output.custom_output.get("text_output") is not None

    is_audio_output = supports_audio_output(od_config.model_class_name)
    audio_sample_rate = postprocess_output.audio_sample_rate
    if is_audio_output and audio_sample_rate is None:
        model_cls = DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
        audio_sample_rate = getattr(model_cls, "audio_sample_rate", None)

    if len(request.prompts) == 1:
        return _format_single_prompt_output(
            request=request,
            diffusion_output=diffusion_output,
            outputs=outputs,
            metrics=metrics,
            postprocess_output=postprocess_output,
            is_text_output=is_text_output,
            is_audio_output=is_audio_output,
            audio_sample_rate=audio_sample_rate,
            finished=diffusion_output.finished,
        )

    return _format_multi_prompt_outputs(
        request=request,
        diffusion_output=diffusion_output,
        outputs=outputs,
        metrics=metrics,
        postprocess_output=postprocess_output,
        is_audio_output=is_audio_output,
        audio_sample_rate=audio_sample_rate,
        finished=diffusion_output.finished,
    )


def _ensure_list(outputs: Any) -> list[Any]:
    if isinstance(outputs, list):
        return outputs
    return [outputs] if outputs is not None else []


def _format_audio_multimodal_output(payload: Any, audio_sample_rate: int | None) -> dict[str, Any]:
    mm_output: dict[str, Any] = {"audio": payload}
    if audio_sample_rate is not None:
        mm_output["audio_sample_rate"] = audio_sample_rate
    return mm_output


def _has_non_audio_postprocess_payload(postprocess_output: DiffusionPostprocessOutput) -> bool:
    return (
        postprocess_output.has_video_payload
        or postprocess_output.action_payload is not None
        or postprocess_output.fps is not None
    )


def _build_multimodal_output(
    postprocess_output: DiffusionPostprocessOutput,
    audio_sample_rate: int | None,
) -> dict[str, Any]:
    mm_output: dict[str, Any] = {}
    if postprocess_output.audio_payload is not None:
        mm_output["audio"] = postprocess_output.audio_payload
    if audio_sample_rate is not None:
        mm_output["audio_sample_rate"] = audio_sample_rate
    if postprocess_output.fps is not None:
        mm_output["fps"] = postprocess_output.fps
    if postprocess_output.action_payload is not None:
        mm_output["actions"] = postprocess_output.action_payload
    return mm_output


def _format_single_prompt_output(
    *,
    request: OmniDiffusionRequest,
    diffusion_output: DiffusionOutput,
    outputs: list[Any],
    metrics: dict[str, Any],
    postprocess_output: DiffusionPostprocessOutput,
    is_text_output: bool,
    is_audio_output: bool,
    audio_sample_rate: int | None,
    finished: bool = True,
) -> list[OmniRequestOutput]:
    prompt = request.prompts[0]
    request_id = request.request_id
    mm_output = _build_multimodal_output(postprocess_output, audio_sample_rate)

    if is_text_output:
        return [
            OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=[],
                prompt=prompt,
                metrics=metrics,
                custom_output=postprocess_output.custom_output,
                multimodal_output=mm_output,
                final_output_type="text",
                stage_durations=diffusion_output.stage_durations,
                peak_memory_mb=diffusion_output.peak_memory_mb,
                finished=finished,
            ),
        ]

    if is_audio_output and not _has_non_audio_postprocess_payload(postprocess_output):
        request_audio_payload = postprocess_output.audio_payload
        if request_audio_payload is None:
            request_audio_payload = outputs[0] if len(outputs) == 1 else outputs
        return [
            OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=[],
                prompt=prompt,
                metrics=metrics,
                latents=diffusion_output.trajectory_latents,
                trajectory_latents=diffusion_output.trajectory_latents,
                trajectory_timesteps=diffusion_output.trajectory_timesteps,
                trajectory_log_probs=diffusion_output.trajectory_log_probs,
                trajectory_decoded=diffusion_output.trajectory_decoded,
                multimodal_output=_format_audio_multimodal_output(
                    request_audio_payload,
                    audio_sample_rate,
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
            latents=diffusion_output.trajectory_latents,
            trajectory_latents=diffusion_output.trajectory_latents,
            trajectory_timesteps=diffusion_output.trajectory_timesteps,
            trajectory_log_probs=diffusion_output.trajectory_log_probs,
            trajectory_decoded=diffusion_output.trajectory_decoded,
            custom_output=postprocess_output.custom_output,
            multimodal_output=mm_output,
            stage_durations=diffusion_output.stage_durations,
            peak_memory_mb=diffusion_output.peak_memory_mb,
            finished=finished,
        ),
    ]


def _format_multi_prompt_outputs(
    *,
    request: OmniDiffusionRequest,
    diffusion_output: DiffusionOutput,
    outputs: list[Any],
    metrics: dict[str, Any],
    postprocess_output: DiffusionPostprocessOutput,
    is_audio_output: bool,
    audio_sample_rate: int | None,
    finished: bool = True,
) -> list[OmniRequestOutput]:
    results = []
    output_idx = 0
    request_id = request.request_id

    for prompt in request.prompts:
        num_outputs = request.sampling_params.num_outputs_per_prompt
        start_idx = output_idx
        end_idx = start_idx + num_outputs
        request_outputs = outputs[start_idx:end_idx] if output_idx < len(outputs) else []
        output_idx = end_idx

        if is_audio_output and not _has_non_audio_postprocess_payload(postprocess_output):
            if postprocess_output.audio_payload is not None:
                request_audio_payload = _slice_batch_payload(
                    postprocess_output.audio_payload,
                    start_idx,
                    end_idx,
                    num_outputs,
                )
            else:
                request_audio_payload = request_outputs[0] if len(request_outputs) == 1 else request_outputs
            results.append(
                OmniRequestOutput.from_diffusion(
                    request_id=request_id,
                    images=[],
                    prompt=prompt,
                    metrics=metrics,
                    latents=diffusion_output.trajectory_latents,
                    trajectory_latents=diffusion_output.trajectory_latents,
                    trajectory_timesteps=diffusion_output.trajectory_timesteps,
                    trajectory_log_probs=diffusion_output.trajectory_log_probs,
                    trajectory_decoded=diffusion_output.trajectory_decoded,
                    multimodal_output=_format_audio_multimodal_output(
                        request_audio_payload,
                        audio_sample_rate,
                    ),
                    final_output_type="audio",
                    stage_durations=diffusion_output.stage_durations,
                    peak_memory_mb=diffusion_output.peak_memory_mb,
                    finished=finished,
                ),
            )
            continue

        mm_output: dict[str, Any] = {}
        if postprocess_output.audio_payload is not None:
            mm_output["audio"] = _slice_batch_payload(
                postprocess_output.audio_payload,
                start_idx,
                end_idx,
                num_outputs,
            )
        if audio_sample_rate is not None:
            mm_output["audio_sample_rate"] = audio_sample_rate
        if postprocess_output.fps is not None:
            mm_output["fps"] = postprocess_output.fps
        if postprocess_output.action_payload is not None:
            mm_output["actions"] = _slice_batch_payload(
                postprocess_output.action_payload,
                start_idx,
                end_idx,
                num_outputs,
            )

        results.append(
            OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=request_outputs,
                prompt=prompt,
                metrics=metrics,
                latents=diffusion_output.trajectory_latents,
                trajectory_latents=diffusion_output.trajectory_latents,
                trajectory_timesteps=diffusion_output.trajectory_timesteps,
                trajectory_log_probs=diffusion_output.trajectory_log_probs,
                trajectory_decoded=diffusion_output.trajectory_decoded,
                custom_output=postprocess_output.custom_output,
                multimodal_output=mm_output,
                stage_durations=diffusion_output.stage_durations,
                peak_memory_mb=diffusion_output.peak_memory_mb,
                finished=finished,
            ),
        )

    return results


def _slice_batch_payload(payload: Any, start_idx: int, end_idx: int, num_outputs: int) -> Any:
    sliced = payload
    if isinstance(payload, (list, tuple)):
        sliced = payload[start_idx:end_idx]
        if len(sliced) == 1:
            sliced = sliced[0]
    elif hasattr(payload, "shape") and getattr(payload, "shape", None) is not None:
        if len(payload.shape) > 0 and payload.shape[0] >= end_idx:
            sliced = payload[start_idx:end_idx]
            if num_outputs == 1:
                sliced = sliced[0]
    return sliced

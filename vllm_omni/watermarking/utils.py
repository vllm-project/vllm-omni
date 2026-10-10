# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from vllm.logger import init_logger
from vllm.outputs import RequestOutput

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput, MultimodalPayload
from vllm_omni.outputs.output_modality import OutputModalityNames
from vllm_omni.watermarking.base import Watermarker, WatermarkFailureError
from vllm_omni.watermarking.converters import (
    MEDIA_CONVERTERS,
    media_to_tensor,
    restore_media,
)

logger = init_logger(__name__)

OutputPayload = MultimodalPayload | dict[str, object]


def watermark_media(
    request_id: str,
    modality: OutputModalityNames,
    watermarker: Watermarker,
    data: object,
    metadata: Mapping[str, object],
) -> object:
    """Watermark one media value and restore its output type.

    This is needed since current watermarkers assume torch tensors as inputs.
    """
    try:
        # NOTE: This is to avoid double watermarking on cumulative semantics.
        if modality == OutputModalityNames.AUDIO and isinstance(data, list):
            raise TypeError("audio chunk lists must be watermarked per chunk before accumulation")
        converter = MEDIA_CONVERTERS[modality]
        tensor = media_to_tensor(data, converter.to_tensor)
        watermarked = watermarker.watermark_output(request_id, tensor, metadata)
        return restore_media(watermarked, data, converter.restore)
    except (RuntimeError, TypeError, ValueError) as error:
        raise WatermarkFailureError(f"invalid {modality.value} output") from error


def watermark_payload(
    request_id: str,
    modality: OutputModalityNames,
    watermarker: Watermarker,
    payload: OutputPayload,
) -> None:
    """Watermark one modality payload in place."""
    modality_key = modality.value
    data = payload.get(modality_key)
    if data is None or isinstance(data, list) and not data:
        return
    metadata: Mapping[str, object] = payload
    if modality == OutputModalityNames.AUDIO and payload.get("sr") is None:
        metadata = {"sr": payload.get("audio_sample_rate")}
    result = watermark_media(request_id, modality, watermarker, data, metadata)
    if not isinstance(payload, MultimodalPayload):
        payload[modality_key] = result
    elif modality_key not in payload.tensors:
        payload.metadata[modality_key] = result
    elif isinstance(result, torch.Tensor):
        payload.tensors[modality_key] = result
    else:
        raise WatermarkFailureError("tensor payload must remain a tensor")


def _watermark_core_output(
    output: OmniEngineCoreOutput,
    watermarkers: Mapping[str, Watermarker],
) -> None:
    """watermark engine core outputs."""
    if output.multimodal_output is None:
        return
    for modality_key, watermarker in watermarkers.items():
        modality = OutputModalityNames(modality_key)
        payload = MultimodalPayload.from_raw(output.multimodal_output, modality.value)
        if payload is None:
            continue
        watermark_payload(output.request_id, modality, watermarker, payload)
        output.multimodal_output = payload  # type: ignore[assignment]


def _watermark_request_output(
    output: RequestOutput,
    watermarkers: Mapping[str, Watermarker],
) -> None:
    payloads: list[OutputPayload] = [
        completion.multimodal_output
        for completion in output.outputs
        if isinstance(completion, MultimodalCompletionOutput) and completion.multimodal_output is not None
    ]

    if isinstance(output, OmniRequestOutput) and not output.outputs:
        if isinstance(output.multimodal_output, dict):
            payloads.append(output.multimodal_output)

    for payload in payloads:
        for modality_key, watermarker in watermarkers.items():
            watermark_payload(
                output.request_id,
                OutputModalityNames(modality_key),
                watermarker,
                payload,
            )

    if output.finished:
        for watermarker in watermarkers.values():
            watermarker.discard_request_state(output.request_id)


def _watermark_output(
    output: object,
    watermarkers: Mapping[str, Watermarker],
) -> None:
    """Apply watermark to either core or request outputs."""
    if isinstance(output, OmniEngineCoreOutput):
        _watermark_core_output(output, watermarkers)
    elif isinstance(output, RequestOutput):
        _watermark_request_output(output, watermarkers)
    else:
        # Output type is unhandled for watermarking; this should not happen
        raise TypeError(f"[watermark] unsupported output type: {type(output).__name__}")


def _handle_watermark_failure(request_id: str, watermarkers: Mapping[str, Watermarker]) -> None:
    """Log a watermark failure and discard any active state for this request.

    NOTE: Whether it's strict / not strict doesn't matter here, since we don't handle it in utils.
    """
    for watermarker in watermarkers.values():
        watermarker.discard_request_state(request_id)
    logger.exception("Failed to watermark %s output for request %s", ", ".join(watermarkers), request_id)


def watermark_outputs(
    outputs: Sequence[OmniEngineCoreOutput | RequestOutput],
    watermarkers: Mapping[str, Watermarker],
) -> set[str]:
    """Watermark outputs in place; returns the request ids whose outputs failed watermarking."""
    watermark_failed_request_ids: set[str] = set()
    for output in outputs:
        try:
            # TODO: Ensure failure behavior is correct for when we are handling multiple modalities
            _watermark_output(output, watermarkers)
        except WatermarkFailureError:
            _handle_watermark_failure(output.request_id, watermarkers)
            watermark_failed_request_ids.add(output.request_id)
    return watermark_failed_request_ids

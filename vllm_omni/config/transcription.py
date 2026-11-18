# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve ASR model capabilities and transcription pipeline compatibility."""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

ASR_ALIGNER_INPUT_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.forced_aligner.asr2aligner"


def supports_transcription_model(model_config: "ModelConfig | None") -> bool:
    """Check the resolved model's vLLM transcription contract, independent of topology.

    This declares a capability; callers still construct transcription requests
    with task_type="transcribe" and retain the corresponding audio waveform.
    Registry inspection errors propagate so explicit aligner startup fails.
    """
    if model_config is None or model_config.runner_type != "generate":
        return False
    info, _ = model_config.registry.inspect_model_cls(model_config.architectures, model_config)
    return info.supports_transcription and not info.is_pooling_model


def validate_asr_aligner_stages(stage_configs: Sequence[Any], stage_vllm_configs: Sequence[Any]) -> None:
    """Check the source model's transcription capability for each ASR aligner.

    The injected aligner consumes the immediately preceding stage's output.
    """
    for index, config in enumerate(stage_configs[1:], start=1):
        if getattr(config, "custom_process_input_func", None) != ASR_ALIGNER_INPUT_PROCESSOR:
            continue
        source_config = getattr(stage_vllm_configs[index - 1], "model_config", None)
        if not supports_transcription_model(source_config):
            raise ValueError(
                f"ASR + ForcedAligner requires a transcription-capable, non-pooling source model; "
                f"stage {stage_configs[index - 1].stage_id} does not support transcription."
            )


def supports_transcription_pipeline(
    stage_configs: Sequence[Any], stage_metadata: Sequence[Any], stage_vllm_configs: Sequence[Any]
) -> bool:
    """Check compatibility with the transcription HTTP handler, not ASR identity.

    Plain transcription requests select only the entry model's text output.
    Other output modalities do not affect whether this endpoint is available.
    """
    if not stage_configs:
        return False
    # AsyncOmni exposes the first comprehension stage's config to serving.
    # It must describe the same stage that receives the transcription prompt.
    if not getattr(stage_configs[0], "is_comprehension", False) and any(
        getattr(config, "is_comprehension", False) and runtime is not None
        for config, runtime in zip(stage_configs[1:], stage_vllm_configs[1:])
    ):
        return False

    # Requests select modalities rather than arbitrary output-stage IDs. A
    # second text terminal would be indistinguishable from the transcript.
    text_stages = [i for i, meta in enumerate(stage_metadata) if meta.final_output and meta.final_output_type == "text"]
    if text_stages != [0]:
        return False

    model_config = getattr(stage_vllm_configs[0], "model_config", None)
    try:
        return supports_transcription_model(model_config)
    except Exception:
        logger.warning(
            "Could not inspect ASR capability for stage 0; transcription endpoint disabled.",
            exc_info=True,
        )
        return False


def has_transcription_aligner(
    stage_configs: Sequence[Any], stage_metadata: Sequence[Any], stage_vllm_configs: Sequence[Any]
) -> bool:
    """Can a transcription request select timestamps for the uploaded audio?"""
    # The request selects a modality, so another timestamps output is ambiguous.
    timestamp_stages = [
        i for i, meta in enumerate(stage_metadata) if meta.final_output and meta.final_output_type == "timestamps"
    ]
    if timestamp_stages != [1]:
        return False
    aligner = stage_configs[1]
    model_config = getattr(stage_vllm_configs[1], "model_config", None)
    return (
        aligner.custom_process_input_func == ASR_ALIGNER_INPUT_PROCESSOR
        and tuple(aligner.input_sources) == (0,)
        and getattr(model_config, "runner_type", None) == "pooling"
    )

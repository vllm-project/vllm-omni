# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for the Qwen3-ASR forced-aligner stage.

Builds the aligner's input from two sources: the transcript stage 0 just
produced, and the audio the request arrived with. The aligner needs both -- it
aligns known words against known audio -- and the input audio is not any
stage's *output*, so it has to be carried across from the original prompt.

That is the same shape as ``aura_omni``'s ``asr2aura``, which pairs stage 0's
transcript with the request's original video payload. The difference is where
the audio comes from. ``aura_omni`` is driven by ``/v1/chat/completions``,
whose prompts still carry raw ``multi_modal_data`` when the stage sees them.
``/v1/audio/transcriptions`` renders its prompts up front, so by this point the
audio survives only as processed features (``mm_kwargs``) and the raw waveform
is gone. The serving layer therefore attaches the waveform it already decoded
under ``additional_information[ALIGNER_AUDIO_KEY]``, which keeps the pipeline
to a single decode per request.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from vllm.logger import init_logger

from vllm_omni.utils import qwen3_force_align_processor as _processor

logger = init_logger(__name__)

#: The Qwen3 forced aligner consumes 16 kHz mono audio.
ALIGNER_SAMPLE_RATE = 16000

#: ``additional_information`` key carrying the decoded input waveform from the
#: serving layer to this stage. Shared so the two ends cannot drift apart.
ALIGNER_AUDIO_KEY = "aligner_audio"

#: ``model_stage`` of the aligner stage, as declared in the pipeline topology.
#: The serving layer matches on it to detect that alignment is available.
ALIGNER_STAGE_NAME = "forced_aligner"


def attach_aligner_audio(engine_input: Any, waveform: np.ndarray, sample_rate: int) -> None:
    """Put the decoded waveform where this stage's processor will look for it.

    Tolerates both mapping-shaped and attribute-shaped engine inputs, since the
    concrete prompt type depends on how upstream rendered the request.
    """
    payload = (waveform, int(sample_rate))
    if isinstance(engine_input, dict):
        additional = engine_input.get("additional_information")
        if not isinstance(additional, dict):
            additional = {}
            engine_input["additional_information"] = additional
    else:
        additional = getattr(engine_input, "additional_information", None)
        if not isinstance(additional, dict):
            additional = {}
            engine_input.additional_information = additional
    additional[ALIGNER_AUDIO_KEY] = payload


def _source_prompt_by_request_id(source_outputs: list[Any], prompt: Any) -> dict[str, Any]:
    """Index the originating prompts by request id, mirroring aura_omni."""
    prompts = prompt if isinstance(prompt, list) else [prompt]
    by_id: dict[str, Any] = {}
    for idx, p in enumerate(prompts):
        if not isinstance(p, dict):
            continue
        rid = str(p.get("request_id", idx))
        by_id[rid] = p
    if len(by_id) == 1 and len(source_outputs) == 1:
        # Single request: index may not line up, but there is no ambiguity.
        return {str(getattr(source_outputs[0], "request_id", 0)): next(iter(by_id.values()))}
    return by_id


def _extract_text(source_output: Any) -> str:
    """Stage 0's transcript, post-processed the way the endpoint returns it.

    Qwen3-ASR generates ``"language {lang}<asr_text>{transcription}"``. The
    serving layer strips that scaffolding before returning the transcript, so
    the aligner has to as well -- segmenting the raw text yields phantom words
    ("language", "English", "asr", "text") that are not in the audio, and every
    timestamp after them is attributed to the wrong word.
    """
    outputs = getattr(source_output, "outputs", None) or []
    for out in outputs:
        text = getattr(out, "text", None)
        if text:
            from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration

            return str(Qwen3ASRForConditionalGeneration.post_process_output(str(text)))
    return ""


def _unwrap(value: Any) -> Any:
    """Undo the per-request list wrapping applied to additional_information."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _audio_from_prompt(src_prompt: dict[str, Any]) -> tuple[np.ndarray, int] | None:
    """Recover the request's waveform without decoding the upload again."""
    additional = src_prompt.get("additional_information") or {}
    if isinstance(additional, dict):
        carried = _unwrap(additional.get(ALIGNER_AUDIO_KEY))
        if carried is not None:
            waveform, sample_rate = carried
            return np.asarray(waveform), int(sample_rate)

    # Pipelines that submit unrendered prompts still carry the raw payload.
    mm = src_prompt.get("multi_modal_data") or {}
    audio = mm.get("audio") if isinstance(mm, dict) else None
    if audio is None:
        return None
    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        waveform, sample_rate = audio
        return np.asarray(waveform), int(sample_rate)
    return np.asarray(audio), ALIGNER_SAMPLE_RATE


def asr2aligner(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = True,
) -> list[dict[str, Any]]:
    """Build forced-aligner inputs from ASR transcripts plus the source audio.

    A request whose transcript is empty, or whose audio did not survive, yields
    no aligner input: there is nothing to align, and a stage error would fail a
    request whose transcript is already correct.
    """
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[dict[str, Any]] = []

    for idx, source_output in enumerate(source_outputs):
        rid = str(getattr(source_output, "request_id", idx))
        src_prompt = prompt_by_request_id.get(rid, {})

        transcript = _extract_text(source_output).strip()
        if not transcript:
            logger.debug("Aligner stage: empty transcript for %s, skipping", rid)
            continue

        audio = _audio_from_prompt(src_prompt)
        if audio is None:
            logger.warning("Aligner stage: no audio on the source prompt for %s, skipping", rid)
            continue
        waveform, sample_rate = audio

        # Segment once: the word units in the prompt and in the timestamp
        # decode MUST match, or the markers drift out of sync.
        words = _processor.segment_words(transcript, src_prompt.get("language"))
        if not words:
            continue

        next_inputs.append(
            {
                "prompt": _processor.build_prompt(words),
                "multi_modal_data": {"audio": (waveform, sample_rate)},
                # Carried so the output side can decode timestamps against the
                # same word units, without re-segmenting and risking a mismatch.
                "additional_information": {"aligner_words": words},
            }
        )

    return next_inputs

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


#: The aligner checkpoint, whose tokenizer knows the ``<timestamp>`` marker that
#: the ASR tokenizer does not carry.
_ALIGNER_TOKENIZER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

_tokenizer: Any = None
_logged_prompt = False


def _aligner_tokenizer() -> Any:
    """Load the aligner tokenizer once, lazily.

    Only needed for the tail of the prompt; the audio portion is copied from
    stage 0's token ids rather than re-rendered.
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(_ALIGNER_TOKENIZER_MODEL, trust_remote_code=True)
    return _tokenizer


def _audio_span_end(src_prompt: dict[str, Any]) -> int | None:
    """Index just past the source prompt's audio placeholder tokens.

    The two models share a feature extractor *and* a prompt prefix
    (``<|im_start|>user\\n`` then the audio), so slicing here yields a prefix
    whose audio tokens sit at exactly the offsets the forwarded ``mm_features``
    already describe. That is what makes reusing them safe: no offset is
    recomputed, so none can drift.
    """
    placeholders = src_prompt.get("mm_placeholders")
    if not placeholders:
        return None
    ranges = placeholders.get("audio") if isinstance(placeholders, dict) else None
    if not ranges:
        return None
    first = ranges[0]
    offset = getattr(first, "offset", None)
    length = getattr(first, "length", None)
    if offset is None and isinstance(first, dict):
        offset, length = first.get("offset"), first.get("length")
    if offset is None or length is None:
        return None
    return int(offset) + int(length)


def _tokens_input(src_prompt: dict[str, Any], words: list[str]) -> dict[str, Any] | None:
    """Build the aligner prompt as token ids, reusing stage 0's encoded audio.

    Returns ``None`` when the source prompt is not in the rendered form this
    depends on, so the caller can fall back to handing over a raw waveform.
    """
    src_ids = src_prompt.get("prompt_token_ids")
    end = _audio_span_end(src_prompt)
    if not src_ids or end is None or end > len(src_ids):
        return None

    tokenizer = _aligner_tokenizer()
    # The placeholder range covers the expanded audio_pad tokens; the closing
    # <|audio_end|> marker may sit just past it. Absorb it if so, because the
    # tail below starts after that marker.
    audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")
    window = list(src_ids[end : end + 3])
    if audio_end_id is not None and audio_end_id in window:
        end += window.index(audio_end_id) + 1

    body = _processor.build_prompt(words)
    tail = body.split(_processor.AUDIO_PLACEHOLDER, 1)[-1]
    tail_ids = tokenizer.encode(tail, add_special_tokens=False)
    token_ids = list(src_ids[:end]) + list(tail_ids)

    global _logged_prompt
    if not _logged_prompt:
        _logged_prompt = True
        # Cheap one-time sanity check that the reused prefix and the freshly
        # tokenized tail actually join into the prompt the aligner expects.
        logger.info(
            "Aligner stage prompt: %d audio-prefix tokens + %d tail tokens; tail=%r",
            end,
            len(tail_ids),
            tokenizer.decode(tail_ids)[:160],
        )

    return {
        "prompt_token_ids": token_ids,
        "additional_information": {"aligner_words": words},
    }


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

        # Segment once: the word units in the prompt and in the timestamp
        # decode MUST match, or the markers drift out of sync.
        words = _processor.segment_words(transcript, src_prompt.get("language"))
        if not words:
            continue

        # Preferred path: hand over token ids that reuse stage 0's encoded
        # audio, so the orchestrator can forward the mm_features it already
        # produced and the mel pass happens once for the whole pipeline.
        tokens_input = _tokens_input(src_prompt, words)
        if tokens_input is not None:
            next_inputs.append(tokens_input)
            continue

        # Fallback: re-extract features from the waveform. Still one file
        # decode per request, but a second encoder pass.
        audio = _audio_from_prompt(src_prompt)
        if audio is None:
            logger.warning("Aligner stage: no audio on the source prompt for %s, skipping", rid)
            continue
        waveform, sample_rate = audio
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

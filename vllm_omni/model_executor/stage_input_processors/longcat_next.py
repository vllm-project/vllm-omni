"""Stage input processors bridging LongCat-Next thinker -> decoders.

Visual (image) generation still uses the thinker's flattened output *token*
stream, which interleaves text ids with multimodal code rows (8 consecutive
per position, one id per codebook level, each id carrying its level offset)
delimited by marker tokens:

    <longcat_img_start> [8 ids] ... <longcat_img_newline> ... <longcat_img_end>

Extraction there is segment-aware rather than range-based: the audio and
visual flat-id ranges overlap (audio level 2+ ids exceed the visual base
offset), so only the surrounding markers disambiguate the modality -- see
extract_visual_codes()/infer_visual_grid() in longcat_next_utils.py.

Audio (speech synthesis) generation does NOT go through the visible token
stream at all: <longcat_audiogen_start>/<longcat_audiogen_end> only bracket
a fixed audio_pad_token_id placeholder per step in output_ids (see
compute_logits() in modeling_longcat_next.py), while the real 8-value
per-frame codes are produced by talker_mtp (the audio_head depth-transformer
loop) and surface via the finished RequestOutput's
``outputs[0].multimodal_output["codes"]["audio"]`` -- a [T, 8] tensor,
offset-carrying (same convention the audio decoder / extract_audio_codes' old
flat-stream path used). This mirrors how Qwen3-TTS's
talker2code2wav_token_only consumes talker_mtp output (see
stage_input_processors/qwen3_tts.py).

The [T, 8] is assembled by the *output processor*, not by the runner's
model_intermediate_buffer: talker_mtp returns one frame per decode step,
make_omni_output puts that single frame on the step's OmniOutput, and the
accumulation strategy for the thinker stage's ``latent`` modality
(CONCAT_DIM0) concatenates the per-step rows. The buffer entry itself only
ever holds the newest frame -- reading it directly would yield one row, not
the generation.

These are ``sync_process_input_func`` hooks, invoked by the stage engine as
``func(source_outputs, prompt, requires_multimodal_data)`` where each source
output carries ``prompt_token_ids``, ``finished`` and ``outputs[0]`` with the
generated ids (see cosyvoice3.text2flow_token_only for the reference shape).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    extract_visual_codes,
    infer_visual_grid,
)


def _ensure_list(ids: Any) -> list[int]:
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        return list(ids.tolist())
    return list(ids)


def _generated_ids(source_output: Any) -> list[int]:
    """Generated ids for one finished request, prompt prefix stripped."""
    output = source_output.outputs[0]
    output_ids = _ensure_list(getattr(output, "cumulative_token_ids", None)) or _ensure_list(
        getattr(output, "token_ids", None)
    )
    prefix = _ensure_list(getattr(source_output, "prompt_token_ids", None))
    if prefix and output_ids[: len(prefix)] == prefix:
        output_ids = output_ids[len(prefix):]
    return output_ids


def thinker2image_decoder_token_only(
    source_outputs: Sequence[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = True,
) -> list[OmniTokensPrompt]:
    del prompt
    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        output_ids = _generated_ids(source_output)
        additional_information: dict[str, Any] = {
            "visual_token_ids": extract_visual_codes(output_ids),
        }
        grid = infer_visual_grid(output_ids)
        if grid is not None:
            additional_information["token_h"] = grid[0]
            additional_information["token_w"] = grid[1]
        engine_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return engine_inputs


def _extract_audio_codes_from_output(source_output: Any) -> list[list[int]]:
    """Pull talker_mtp's accumulated [T, 8] audio codes off a finished
    RequestOutput. Codes are offset-carrying (each level already has its
    AUDIO_LEVEL_OFFSETS[level] added, matching the audio decoder's expected
    input), same convention the old flat-token-stream extraction used."""
    output = source_output.outputs[0]
    mm = getattr(output, "multimodal_output", None)
    if not isinstance(mm, Mapping):
        return []
    audio = mm.get("codes", {})
    audio = audio.get("audio") if isinstance(audio, Mapping) else None
    if audio is None:
        return []
    if isinstance(audio, torch.Tensor):
        if audio.numel() == 0:
            return []
        rows = audio.to(torch.long).tolist()
    else:
        rows = list(audio)
    # talker_mtp marks discarded/terminal frames with an all -1 row (frames
    # generated before audio_start, or the chunk-end sentinel). Keep only real
    # kept frames.
    return [row for row in rows if row and all(c >= 0 for c in row)]


def thinker2audio_decoder_token_only(
    source_outputs: Sequence[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = True,
) -> list[OmniTokensPrompt]:
    del prompt
    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        engine_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information={
                    "audio_token_ids": _extract_audio_codes_from_output(source_output),
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return engine_inputs

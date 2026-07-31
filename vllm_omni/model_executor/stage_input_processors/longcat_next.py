"""Stage input processors bridging LongCat-Next thinker -> decoders.

Neither visual (image) nor audio (speech) generation goes through the
visible output *token* stream for their real per-frame codes.
<longcat_img_start>/<longcat_img_newline>/<longcat_img_end> and
<longcat_audiogen_start>/<longcat_audiogen_end> only bracket a fixed
IMG_PAD/AUDIOTEXT_PAD placeholder per step in the visible stream (forced in
compute_logits(), modeling_longcat_next.py) -- confirmed against the
official reference's own state machine (output_processor.py: GEN_IMAGE_STAGE
overwrites text_ids to IMAGE_PAD/IMAGE_NEWLINE every step, GEN_AUDIO_STAGE
overwrites to AUDIOTEXT_PAD/START). The real 8-value per-frame codes for
BOTH modalities are produced by talker_mtp (the audio_head/visual_head
depth-transformer loop, same checkpoint class for both) and surface via the
finished RequestOutput's ``outputs[0].multimodal_output["codes"]["audio"]``
or ``["codes"]["visual"]`` -- a [T, 8] tensor of RAW per-level codebook
indices (0..codebook_size-1), not offset-carrying: neither decoder adds/
subtracts LEVEL_OFFSETS (confirmed by reading both
modeling_longcat_next_audio_decoder.py and the reference's own
``lazy_decode_and_save``, which indexes each level's codebook directly via
``embed[data[..., idx]]``). talker_mtp only adds the offsets transiently,
for its own embedding-feedback lookup during generation (see
``_sample_depth_head``'s caller in modeling_longcat_next.py) -- the values
actually stored in ``all_codes``/emitted here are the pre-offset samples.
This mirrors
how Qwen3-TTS's talker2code2wav_token_only consumes talker_mtp output (see
stage_input_processors/qwen3_tts.py).

An EARLIER version of this file's image path assumed visual codes rode the
visible token stream directly (8 consecutive ids per grid position) and
extracted them from output_ids via extract_visual_codes() -- that was wrong:
lm_head/logits_processor is sized to text_vocab_size only (131125), which
does not cover the visual offset range (150581+), so those ids could never
actually be sampled as ordinary vocab tokens. infer_visual_grid() 's row-width
math shared the same wrong assumption (it divided by NUM_CODEBOOKS, expecting
8 visible ids per grid position) and has ALSO been corrected: the visible
stream carries exactly one IMG_PAD placeholder per real-pixel step (forced in
compute_logits), not the real per-level codes.

The [T, 8] for both modalities is assembled by the *output processor*, not
by the runner's model_intermediate_buffer: talker_mtp returns one frame per
decode step, make_omni_output puts that single frame on the step's
OmniOutput, and the accumulation strategy for the thinker stage's
``latent`` modality (CONCAT_DIM0) concatenates the per-step rows. The
buffer entry itself only ever holds the newest frame -- reading it directly
would yield one row, not the generation.

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


def _extract_codes_from_output(source_output: Any, modality_key: str) -> list[list[int]]:
    """Pull talker_mtp's accumulated [T, 8] codes off a finished
    RequestOutput, for either modality (``"audio"`` or ``"visual"``). Codes
    are RAW per-level codebook indices (0..codebook_size-1) -- see the
    module docstring for why this is not offset-carrying."""
    output = source_output.outputs[0]
    mm = getattr(output, "multimodal_output", None)
    if not isinstance(mm, Mapping):
        return []
    codes = mm.get("codes", {})
    codes = codes.get(modality_key) if isinstance(codes, Mapping) else None
    if codes is None:
        return []
    if isinstance(codes, torch.Tensor):
        if codes.numel() == 0:
            return []
        rows = codes.to(torch.long).tolist()
    else:
        rows = list(codes)
    # talker_mtp marks discarded/terminal frames with an all -1 row (frames
    # generated before audio_start / at an image row boundary, or the
    # chunk/image-end sentinel). Keep only real kept frames.
    return [row for row in rows if row and all(c >= 0 for c in row)]


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
        additional_information: dict[str, Any] = {
            "visual_token_ids": _extract_codes_from_output(source_output, "visual"),
        }
        # infer_visual_grid counts visible IMG_PAD placeholders per row (one
        # per real-pixel step, forced in compute_logits) -- it was ALSO
        # fixed alongside extract_visual_codes: an earlier version divided
        # by NUM_CODEBOOKS, matching the same wrong assumption that the real
        # per-level codes rode this stream directly.
        grid = infer_visual_grid(_generated_ids(source_output))
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
                    "audio_token_ids": _extract_codes_from_output(source_output, "audio"),
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return engine_inputs


def thinker2multi_decoder_token_only(
    source_outputs: Sequence[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = True,
) -> list[OmniTokensPrompt]:
    """Feeds LongcatNextMultiDecoder (thinker -> multi_decoder, 2-stage
    pipeline). Extracts both codes.visual and codes.audio from the SAME
    finished thinker output -- unlike the split image_decoder/audio_decoder
    pipeline, there's no risk of stage 2 reading stage 1's output here,
    since this is the only downstream stage. Exactly one of the two will be
    non-empty per request per the reference's own mutually-exclusive
    GEN_IMAGE_STAGE/GEN_AUDIO_STAGE state machine; the decoder module itself
    dispatches on whichever is present."""
    del prompt
    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        additional_information: dict[str, Any] = {
            "visual_token_ids": _extract_codes_from_output(source_output, "visual"),
            "audio_token_ids": _extract_codes_from_output(source_output, "audio"),
        }
        grid = infer_visual_grid(_generated_ids(source_output))
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

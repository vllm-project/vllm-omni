# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    AUDIO_LEVEL_OFFSETS,
    AUDIOGEN_END_TOKEN_ID,
    AUDIOGEN_START_TOKEN_ID,
    IMG_END_TOKEN_ID,
    IMG_NEWLINE_TOKEN_ID,
    IMG_PAD_TOKEN_ID,
    IMG_START_TOKEN_ID,
    NUM_CODEBOOKS,
    VISUAL_LEVEL_OFFSETS,
    extract_audio_codes,
    extract_visual_codes,
    infer_visual_grid,
)
from vllm_omni.model_executor.stage_input_processors.longcat_next import (
    thinker2audio_decoder_token_only,
    thinker2image_decoder_token_only,
)


def _visual_row(codes: list[int]) -> list[int]:
    assert len(codes) == NUM_CODEBOOKS
    return [code + VISUAL_LEVEL_OFFSETS[level] for level, code in enumerate(codes)]


def _audio_row(codes: list[int]) -> list[int]:
    assert len(codes) == NUM_CODEBOOKS
    return [code + AUDIO_LEVEL_OFFSETS[level] for level, code in enumerate(codes)]


def _source_output(
    out_ids: list[int],
    prompt_ids: list[int] | None = None,
    finished: bool = True,
):
    return SimpleNamespace(
        request_id="req-0",
        prompt_token_ids=prompt_ids or [],
        finished=finished,
        outputs=[SimpleNamespace(token_ids=out_ids, cumulative_token_ids=out_ids, multimodal_output=None)],
    )


def test_extract_visual_codes_roundtrip():
    row_a = list(range(8))
    row_b = [100 + i for i in range(8)]
    stream = (
        [10, 20, IMG_START_TOKEN_ID]
        + _visual_row(row_a)
        + [IMG_NEWLINE_TOKEN_ID]
        + _visual_row(row_b)
        + [IMG_END_TOKEN_ID, 30]
    )
    assert extract_visual_codes(stream) == [row_a, row_b]


def test_extract_visual_codes_ignores_text_outside_segments():
    stream = [1, 2, 3, 150581, 150582]  # visual-range ids without markers
    assert extract_visual_codes(stream) == []


def test_extract_audio_codes_roundtrip():
    row = [7] * 8
    end_row = [16384] + [0] * 7  # level-0 chunk-end marker row
    stream = (
        [5, AUDIOGEN_START_TOKEN_ID]
        + _audio_row(row)
        + _audio_row(end_row)
        + [AUDIOGEN_END_TOKEN_ID]
    )
    assert extract_audio_codes(stream) == [row, end_row]


def test_audio_visual_ranges_do_not_cross_contaminate():
    # Audio level-2 ids overlap the visual level-0 range numerically; only the
    # segment markers may decide the modality.
    audio_row = [50] * 8
    stream = [AUDIOGEN_START_TOKEN_ID] + _audio_row(audio_row) + [AUDIOGEN_END_TOKEN_ID]
    assert extract_visual_codes(stream) == []
    assert extract_audio_codes(stream) == [audio_row]


def test_infer_visual_grid_from_newlines():
    # One IMG_PAD placeholder per real-pixel step, not NUM_CODEBOOKS ids --
    # the visible stream never carries the real per-level codes (see
    # infer_visual_grid's docstring).
    stream = [IMG_START_TOKEN_ID]
    for i in range(6):
        stream.append(IMG_PAD_TOKEN_ID)
        if i % 3 == 2:  # newline after every 3 positions -> w=3
            stream.append(IMG_NEWLINE_TOKEN_ID)
    stream.append(IMG_END_TOKEN_ID)
    assert infer_visual_grid(stream) == (2, 3)


def test_thinker2image_decoder_token_only():
    # Real per-frame codes now come from multimodal_output (talker_mtp's
    # visual_head output), de-offset raw indices -- NOT extracted from the
    # visible stream (lm_head can't sample the 150581+ visual vocab range at
    # all, so those ids never actually appear there). The visible stream
    # still only needs to carry the marker structure for infer_visual_grid.
    row = [9] * 8
    # One IMG_PAD real-pixel placeholder before the newline -> width=1.
    marker_stream = [IMG_START_TOKEN_ID, IMG_PAD_TOKEN_ID, IMG_NEWLINE_TOKEN_ID, IMG_END_TOKEN_ID]
    prompts = thinker2image_decoder_token_only(
        [_source_output_with_mm({"codes": {"visual": [row]}}, out_ids=marker_stream)]
    )
    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert info["visual_token_ids"] == [row]
    assert info["token_h"] == 1
    assert info["token_w"] == 1


def test_thinker2image_decoder_empty_when_no_codes():
    prompts = thinker2image_decoder_token_only([_source_output_with_mm(None)])
    assert len(prompts) == 1
    assert prompts[0]["additional_information"]["visual_token_ids"] == []


def test_thinker2image_decoder_skips_unfinished():
    prompts = thinker2image_decoder_token_only(
        [_source_output_with_mm({"codes": {"visual": [[1] * 8]}}, finished=False)]
    )
    assert prompts == []


def _source_output_with_mm(
    mm_output: dict | None, finished: bool = True, out_ids: list[int] | None = None,
):
    """Like _source_output, but with a real multimodal_output payload --
    matching what talker_mtp's accumulated codes look like on a finished
    RequestOutput (see stage_input_processors/longcat_next.py). ``out_ids``
    carries the VISIBLE token stream (markers only, e.g. img_start/newline/
    end) separately from the real per-frame codes in mm_output -- the two
    are decoupled now that codes ride multimodal_output, not the visible
    stream (infer_visual_grid still needs the visible marker structure)."""
    ids = out_ids or []
    return SimpleNamespace(
        request_id="req-0",
        prompt_token_ids=[],
        finished=finished,
        outputs=[SimpleNamespace(token_ids=ids, cumulative_token_ids=ids, multimodal_output=mm_output)],
    )


def test_thinker2audio_decoder_token_only():
    # talker_mtp's codes are de-offset raw indices, not offset-carrying --
    # see modeling_longcat_next.py::talker_mtp's docstring for why.
    codes = [[3] * 8, [5] * 8]
    prompts = thinker2audio_decoder_token_only(
        [_source_output_with_mm({"codes": {"audio": codes}})]
    )
    assert len(prompts) == 1
    assert prompts[0]["additional_information"]["audio_token_ids"] == codes


def test_thinker2audio_decoder_token_only_empty_when_no_codes():
    prompts = thinker2audio_decoder_token_only([_source_output_with_mm(None)])
    assert len(prompts) == 1
    assert prompts[0]["additional_information"]["audio_token_ids"] == []


def test_thinker2audio_decoder_skips_unfinished():
    prompts = thinker2audio_decoder_token_only(
        [_source_output_with_mm({"codes": {"audio": [[1] * 8]}}, finished=False)]
    )
    assert prompts == []


def test_truncated_segment_is_handled():
    # A partial row (fewer than 8 ids before the end marker) is dropped
    # instead of corrupting alignment.
    good = [4] * 8
    stream = (
        [IMG_START_TOKEN_ID]
        + _visual_row(good)
        + _visual_row(good)[:5]
        + [IMG_END_TOKEN_ID]
    )
    assert extract_visual_codes(stream) == [good]


def test_unterminated_segment_yields_nothing():
    stream = [IMG_START_TOKEN_ID] + _visual_row([1] * 8)
    assert extract_visual_codes(stream) == []

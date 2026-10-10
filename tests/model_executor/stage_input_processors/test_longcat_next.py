# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for the LongCat-Next stage-input bridges.

Pins the two ``sync_process_input_func``/``prompt_expand_func`` hooks the
2-stage pipeline hangs on: the visual-CFG prompt expansion and the
thinker->multi_decoder token-only handoff. Both run on every stage-0 request
but were previously exercised only by the 4x-GPU e2e.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    IMG_END_TOKEN_ID,
    IMG_NEWLINE_TOKEN_ID,
    IMG_PAD_TOKEN_ID,
    IMG_START_TOKEN_ID,
)
from vllm_omni.model_executor.stage_input_processors.longcat_next import (
    CFG_VISUAL_SUFFIX,
    expand_longcat_cfg_prompts,
    thinker2multi_decoder_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


IMAGE_PROMPT = (
    "<longcat_system>You are a helpful assistant.<longcat_end>"
    "<longcat_img_token_size>2 3</longcat_img_token_size><longcat_img_start>"
    "<longcat_user>draw a cat<|longcat_end|>"
    "<longcat_assistant>"
)


class TestExpandLongcatCfgPrompts:
    def test_image_prompt_yields_one_visual_uncond_twin(self):
        expanded = expand_longcat_cfg_prompts(IMAGE_PROMPT, sampling_params=None)
        assert len(expanded) == 1
        twin = expanded[0]
        assert twin.role == "visual_uncond"
        assert twin.request_id_suffix == CFG_VISUAL_SUFFIX == "__cfg_visual"
        # System turn, anyres prefix and img entry marker survive; only the
        # user instruction is blanked (to a single space).
        assert "<longcat_system>" in twin.prompt
        assert "<longcat_img_token_size>2 3</longcat_img_token_size>" in twin.prompt
        assert "<longcat_img_start>" in twin.prompt
        assert "draw a cat" not in twin.prompt
        assert "<longcat_user> <longcat_assistant>" in twin.prompt

    def test_non_image_prompt_expands_to_nothing(self):
        text_prompt = "<longcat_user>hello there<|longcat_end|><longcat_assistant>"
        assert expand_longcat_cfg_prompts(text_prompt, sampling_params=None) == []

    def test_non_string_prompt_without_prompt_field_expands_to_nothing(self):
        assert expand_longcat_cfg_prompts(SimpleNamespace(foo="bar"), sampling_params=None) == []
        assert expand_longcat_cfg_prompts({"kind": "tokens"}, sampling_params=None) == []

    def test_empty_user_turn_expands_to_nothing(self):
        # An empty (whitespace-only) user body leaves the prompt unchanged, so
        # an identical twin would no-op CFG -- the expansion must be skipped.
        prompt = IMAGE_PROMPT.replace("draw a cat", "").replace("<|longcat_end|>", "")
        assert expand_longcat_cfg_prompts(prompt, sampling_params=None) == []

    def test_unparsable_user_turn_expands_to_nothing(self):
        # A <longcat_user> marker with no closing <longcat_assistant> cannot
        # be blanked, so the twin would equal the parent.
        prompt = IMAGE_PROMPT.replace("<longcat_assistant>", "")
        assert expand_longcat_cfg_prompts(prompt, sampling_params=None) == []


def _source_output(
    *,
    finished: bool = True,
    mm_codes: dict[str, torch.Tensor] | None = None,
    generated_ids: list[int] | None = None,
    prompt_token_ids: list[int] | None = None,
) -> SimpleNamespace:
    output = SimpleNamespace(multimodal_output={"codes": mm_codes} if mm_codes is not None else {})
    if generated_ids is not None:
        output.token_ids = generated_ids
    return SimpleNamespace(
        finished=finished,
        prompt_token_ids=prompt_token_ids or [],
        outputs=[output],
    )


class TestThinker2MultiDecoderTokenOnly:
    def test_unfinished_outputs_are_skipped(self):
        result = thinker2multi_decoder_token_only([_source_output(finished=False)])
        assert result == []

    def test_finished_text_only_output_yields_empty_modality_keys(self):
        result = thinker2multi_decoder_token_only([_source_output()])
        assert len(result) == 1
        info = result[0]["additional_information"]
        assert info["visual_token_ids"] == []
        assert info["audio_token_ids"] == []

    def test_codes_with_discard_rows_are_stripped_per_modality(self):
        # talker_mtp marks discarded/terminal frames with an all -1 row; the
        # handoff must deliver only the real kept frames, per modality.
        visual_codes = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [-1] * 8, [9, 10, 11, 12, 13, 14, 15, 16]])
        result = thinker2multi_decoder_token_only([_source_output(mm_codes={"visual": visual_codes})])
        assert len(result) == 1
        info = result[0]["additional_information"]
        assert info["visual_token_ids"] == [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
        assert info["audio_token_ids"] == []

    def test_audio_codes_route_to_audio_key(self):
        audio_codes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
        result = thinker2multi_decoder_token_only([_source_output(mm_codes={"audio": audio_codes})])
        info = result[0]["additional_information"]
        assert info["audio_token_ids"] == audio_codes.tolist()
        assert info["visual_token_ids"] == []

    def test_image_stream_sets_token_grid(self):
        # 2 rows x 3 cols of IMG_PAD placeholders, newline-terminated rows:
        # the handoff must surface the inferred (token_h, token_w) grid.
        stream = [IMG_START_TOKEN_ID]
        for i in range(6):
            stream.append(IMG_PAD_TOKEN_ID)
            if i % 3 == 2:
                stream.append(IMG_NEWLINE_TOKEN_ID)
        stream.append(IMG_END_TOKEN_ID)
        result = thinker2multi_decoder_token_only([_source_output(generated_ids=stream)])
        info = result[0]["additional_information"]
        assert (info["token_h"], info["token_w"]) == (2, 3)

    def test_envelope_is_single_placeholder_token(self):
        # Stage 1 consumes a token-batch prompt; the codes ride
        # additional_information, so the envelope is exactly [0].
        result = thinker2multi_decoder_token_only([_source_output()])
        assert len(result) == 1
        assert set(result[0]) >= {"prompt_token_ids", "additional_information"}
        assert result[0]["prompt_token_ids"] == [0]

    def test_mixed_batch_preserves_order_and_count(self):
        outputs = [
            _source_output(finished=False),
            _source_output(mm_codes={"audio": torch.tensor([[1] * 8])}),
            _source_output(),
        ]
        result = thinker2multi_decoder_token_only(outputs)
        assert len(result) == 2
        assert result[0]["additional_information"]["audio_token_ids"] == [[1] * 8]
        assert result[1]["additional_information"]["audio_token_ids"] == []

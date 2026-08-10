# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Omni-Diffusion utility functions, special tokens, and tokenizer base data."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_AUDIO_CODEBOOK_SIZE,
    OMNI_DIFFUSION_AUDIO_START_TOKEN,
    OMNI_DIFFUSION_IMAGE_CODEBOOK_SIZE,
    OMNI_DIFFUSION_IMAGE_START_TOKEN,
    OMNI_DIFFUSION_INPUT_SAMPLE_RATE,
    OMNI_DIFFUSION_OUTPUT_SAMPLE_RATE,
    OmniDiffusionModelSpecialTokens,
    OmniDiffusionTokenizerBaseData,
    get_single_token_id,
    get_single_token_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# OmniDiffusionModelSpecialTokens
# ---------------------------------------------------------------------------


class TestOmniDiffusionModelSpecialTokens:
    def test_all_tokens_are_non_empty_strings(self) -> None:
        for token in OmniDiffusionModelSpecialTokens:
            assert isinstance(token.value, str)
            assert len(token.value) > 0

    def test_audio_tag_is_placeholder(self) -> None:
        assert OmniDiffusionModelSpecialTokens.AUD_TAG.value == "<|audio|>"

    def test_image_tag_is_placeholder(self) -> None:
        assert OmniDiffusionModelSpecialTokens.IMG_TAG.value == "<|image|>"

    def test_audio_tokens_form_a_set_of_four(self) -> None:
        audio_tokens = {
            OmniDiffusionModelSpecialTokens.AUD_TAG,
            OmniDiffusionModelSpecialTokens.AUD_CONTEXT,
            OmniDiffusionModelSpecialTokens.AUD_START,
            OmniDiffusionModelSpecialTokens.AUD_END,
        }
        assert len(audio_tokens) == 4

    def test_image_tokens_form_a_set_of_three(self) -> None:
        image_tokens = {
            OmniDiffusionModelSpecialTokens.IMG_TAG,
            OmniDiffusionModelSpecialTokens.IMG_START,
            OmniDiffusionModelSpecialTokens.IMG_END,
        }
        assert len(image_tokens) == 3

    def test_audio_and_image_tokens_do_not_overlap(self) -> None:
        audio_values = {t.value for t in OmniDiffusionModelSpecialTokens if "audio" in t.value}
        image_values = {t.value for t in OmniDiffusionModelSpecialTokens if "image" in t.value}
        assert audio_values.isdisjoint(image_values)


# ---------------------------------------------------------------------------
# get_single_token_id / get_single_token_ids
# ---------------------------------------------------------------------------


class TestGetSingleTokenIds:
    @pytest.fixture
    def tokenizer(self) -> MagicMock:
        tok = MagicMock()
        token_ids: dict[str, int] = {}

        def _encode(tokens, add_special_tokens):
            del add_special_tokens
            return SimpleNamespace(input_ids=[[token_ids.setdefault(token, len(token_ids) + 1000)] for token in tokens])

        tok.side_effect = _encode
        return tok

    def test_returns_single_id_list(self, tokenizer: MagicMock) -> None:
        ids = get_single_token_ids(tokenizer, ["hello"])
        assert len(ids) == 1
        assert isinstance(ids[0], int)

    def test_returns_multiple_ids(self, tokenizer: MagicMock) -> None:
        ids = get_single_token_ids(tokenizer, ["hello", "world"])
        assert len(ids) == 2
        assert all(isinstance(tid, int) for tid in ids)

    def test_get_single_token_id_is_equivalent(self, tokenizer: MagicMock) -> None:
        tid = get_single_token_id(tokenizer, "hello")
        assert tid == get_single_token_ids(tokenizer, ["hello"])[0]

    def test_raises_when_token_maps_to_multiple_ids(self) -> None:
        tok = MagicMock()
        tok.side_effect = lambda tokens, add_special_tokens: SimpleNamespace(input_ids=[[100, 200]])

        with pytest.raises(ValueError, match="exactly one token ID"):
            get_single_token_ids(tok, ["bad_token"])

    def test_raises_when_token_count_mismatches(self) -> None:
        tok = MagicMock()
        tok.side_effect = lambda tokens, add_special_tokens: SimpleNamespace(input_ids=[[100], [200], [300]])

        with pytest.raises(ValueError, match="Expected 2 encoded token sequences"):
            get_single_token_ids(tok, ["a", "b"])


# ---------------------------------------------------------------------------
# OmniDiffusionTokenizerBaseData
# ---------------------------------------------------------------------------


class TestOmniDiffusionTokenizerBaseData:
    @pytest.fixture
    def tokenizer(self) -> MagicMock:
        tok = MagicMock()

        # Assign a unique, stable token ID per special token.
        def _encode(tokens, add_special_tokens):
            del add_special_tokens
            ids = []
            for t in tokens:
                for index, special in enumerate(OmniDiffusionModelSpecialTokens):
                    if special.value == t:
                        ids.append([10000 + index])
                        break
                else:
                    ids.append([len(t)])
            return SimpleNamespace(input_ids=ids)

        tok.side_effect = _encode
        return tok

    def test_initializes_all_special_token_ids(self, tokenizer: MagicMock) -> None:
        base_data = OmniDiffusionTokenizerBaseData(tokenizer)
        for token in OmniDiffusionModelSpecialTokens:
            tid = base_data.get_token_id(token)
            assert isinstance(tid, int)
            assert tid >= 0

    def test_different_tokens_have_different_ids(self, tokenizer: MagicMock) -> None:
        base_data = OmniDiffusionTokenizerBaseData(tokenizer)
        ids = {base_data.get_token_id(t) for t in OmniDiffusionModelSpecialTokens}
        assert len(ids) == len(OmniDiffusionModelSpecialTokens)

    def test_get_token_id_returns_consistent_value(self, tokenizer: MagicMock) -> None:
        base_data = OmniDiffusionTokenizerBaseData(tokenizer)
        for token in OmniDiffusionModelSpecialTokens:
            first = base_data.get_token_id(token)
            second = base_data.get_token_id(token)
            assert first == second


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestOmniDiffusionConstants:
    def test_input_sample_rate_is_16khz(self) -> None:
        assert OMNI_DIFFUSION_INPUT_SAMPLE_RATE == 16000

    def test_output_sample_rate_is_22050(self) -> None:
        assert OMNI_DIFFUSION_OUTPUT_SAMPLE_RATE == 22050

    def test_audio_codebook_size_is_16384(self) -> None:
        assert OMNI_DIFFUSION_AUDIO_CODEBOOK_SIZE == 16384

    def test_image_codebook_size_is_8192(self) -> None:
        assert OMNI_DIFFUSION_IMAGE_CODEBOOK_SIZE == 8192

    def test_audio_start_token_format(self) -> None:
        assert OMNI_DIFFUSION_AUDIO_START_TOKEN == "<|audio_0|>"

    def test_image_start_token_format(self) -> None:
        assert OMNI_DIFFUSION_IMAGE_START_TOKEN == "<|image_0|>"

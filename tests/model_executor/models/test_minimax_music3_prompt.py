# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniMax Music 3 prompt assembly and window arithmetic.

The prompt transformations are not cosmetic: they decide token positions, and
token positions decide the audio. These tests pin the documented behaviours
that a well-meaning cleanup would otherwise "fix".
"""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.minimax_music3.chunking import (
    chunk_windows,
    crop_sample_bounds,
    overlap_mel_length,
)
from vllm_omni.model_executor.models.minimax_music3.constants import (
    AR_CHUNK_FRAMES,
    AR_CHUNK_HOP_FRAMES,
    BLEND_MEL_FRAMES,
    DAV_HOP_SAMPLES,
    HOP_MEL_FRAMES,
)
from vllm_omni.model_executor.models.minimax_music3.prompt import (
    SPECIAL_TOKEN_IDS,
    build_cfg_null_token_ids,
    build_prompt,
    clean_caption,
    normalize_lyrics,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestNormalizeLyrics:
    def test_prepends_start_tag(self) -> None:
        assert normalize_lyrics("[Verse]\nhello").startswith("[start]\n")

    def test_lowercases_structure_tags(self) -> None:
        assert "[verse]" in normalize_lyrics("[Verse]\nhello")
        assert "[Verse]" not in normalize_lyrics("[Verse]\nhello")

    def test_text_after_a_leading_tag_is_dropped(self) -> None:
        """Documented contract: a tag must sit on its own line.

        Lyrics written next to a tag are silently lost, which is why the
        serving docs tell callers to put tags on their own line.
        """
        out = normalize_lyrics("[Verse] Walking down the street")
        assert "Walking" not in out
        assert "[verse]" in out

    def test_text_on_its_own_line_survives(self) -> None:
        out = normalize_lyrics("[Verse]\nWalking down the street")
        assert "Walking down the street" in out

    def test_bracket_space_becomes_newline(self) -> None:
        assert normalize_lyrics("a [chorus]") == "[start]\na\n[chorus]"


class TestCleanCaption:
    def test_special_token_form_is_rewritten_not_passed_through(self) -> None:
        """A caption must not be able to inject a real special token."""
        assert clean_caption("<|bpm 92|>") == "bpm is 92"
        assert "<|" not in clean_caption("<|bpm 92|>")

    def test_lone_tag_keeps_its_inner_text(self) -> None:
        assert clean_caption("<|instrumental|>") == "instrumental"

    def test_markdown_is_stripped(self) -> None:
        assert clean_caption("## Heading") == "Heading"
        assert clean_caption("**bold**") == "bold"
        assert clean_caption("- item") == "item"

    def test_blank_lines_collapse_at_two_not_three(self) -> None:
        """The source collapses runs of two or more, not three or more.

        This shifts blank-line token positions and therefore the audio, so it
        is pinned deliberately.
        """
        assert clean_caption("a\n\nb") == "a\nb"
        assert clean_caption("a\n\n\n\nb") == "a\nb"


class TestBuildPrompt:
    def test_framing_tokens_are_present_and_ordered(self) -> None:
        prompt = build_prompt("a bright pop song", "[Verse]\nhello")
        assert prompt.startswith("<|im_start|><|caption_start|>")
        assert prompt.endswith("<|im_end|><|audio_start|>")
        assert prompt.index("<|caption_end|>") < prompt.index("<|lyrics_start|>")


class TestBuildCfgNullTokenIds:
    @staticmethod
    def _prompt_ids(body_len: int) -> list[int]:
        return (
            [SPECIAL_TOKEN_IDS["<|im_start|>"]]
            + list(range(1000, 1000 + body_len))
            + [SPECIAL_TOKEN_IDS["<|im_end|>"], SPECIAL_TOKEN_IDS["<|audio_start|>"]]
        )

    def test_length_is_preserved(self) -> None:
        """Equal length is what lets the two rows decode in lockstep."""
        ids = self._prompt_ids(12)
        assert len(build_cfg_null_token_ids(ids)) == len(ids)

    def test_body_is_replaced_and_framing_survives(self) -> None:
        ids = self._prompt_ids(12)
        null = build_cfg_null_token_ids(ids)
        assert null[0] == SPECIAL_TOKEN_IDS["<|im_start|>"]
        assert null[-2] == SPECIAL_TOKEN_IDS["<|im_end|>"]
        assert null[-1] == SPECIAL_TOKEN_IDS["<|audio_start|>"]
        assert set(null[1:-2]) == {SPECIAL_TOKEN_IDS["<|audio_cfg|>"]}

    def test_input_is_not_mutated(self) -> None:
        ids = self._prompt_ids(6)
        before = list(ids)
        build_cfg_null_token_ids(ids)
        assert ids == before

    def test_minimal_prompt_with_empty_body(self) -> None:
        ids = [
            SPECIAL_TOKEN_IDS["<|im_start|>"],
            SPECIAL_TOKEN_IDS["<|audio_cfg|>"],
            SPECIAL_TOKEN_IDS["<|im_end|>"],
            SPECIAL_TOKEN_IDS["<|audio_start|>"],
        ]
        assert build_cfg_null_token_ids(ids) == ids

    def test_too_short_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            build_cfg_null_token_ids([1, 2, 3])

    def test_wrong_framing_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="framing"):
            build_cfg_null_token_ids([9, 9, 9, 9, 9])


class TestChunkWindows:
    def test_no_frames_yields_nothing(self) -> None:
        assert chunk_windows(0) == []

    def test_short_song_is_one_final_window(self) -> None:
        windows = chunk_windows(50)
        assert len(windows) == 1
        assert (windows[0].start, windows[0].end) == (0, 50)
        assert windows[0].is_first and windows[0].is_last

    def test_exactly_one_window(self) -> None:
        windows = chunk_windows(AR_CHUNK_FRAMES)
        assert len(windows) == 1
        assert windows[0].is_last

    def test_windows_advance_by_the_hop(self) -> None:
        windows = chunk_windows(450)
        starts = [w.start for w in windows]
        assert starts == list(range(0, starts[-1] + 1, AR_CHUNK_HOP_FRAMES))
        assert windows[-1].is_last
        assert not any(w.is_last for w in windows[:-1])

    def test_windows_cover_every_frame(self) -> None:
        frames = 437
        windows = chunk_windows(frames)
        covered: set[int] = set()
        for window in windows:
            covered.update(range(window.start, window.end))
        assert covered == set(range(frames))

    def test_final_window_is_clamped_to_the_song(self) -> None:
        windows = chunk_windows(437)
        assert windows[-1].end == 437

    def test_only_the_first_window_is_first(self) -> None:
        windows = chunk_windows(600)
        assert windows[0].is_first
        assert not any(w.is_first for w in windows[1:])


class TestCropBounds:
    def test_single_window_song_is_not_cropped(self) -> None:
        window = chunk_windows(50)[0]
        assert crop_sample_bounds(window) == (0, 0)

    def test_first_window_keeps_its_head(self) -> None:
        left, right = crop_sample_bounds(chunk_windows(600)[0])
        assert left == 0
        assert right == (HOP_MEL_FRAMES - BLEND_MEL_FRAMES) * DAV_HOP_SAMPLES

    def test_last_window_keeps_its_tail(self) -> None:
        left, right = crop_sample_bounds(chunk_windows(600)[-1])
        assert left == BLEND_MEL_FRAMES * DAV_HOP_SAMPLES
        assert right == 0

    def test_middle_window_is_cropped_on_both_sides(self) -> None:
        windows = chunk_windows(900)
        middle = windows[len(windows) // 2]
        left, right = crop_sample_bounds(middle)
        assert left > 0 and right > 0

    def test_overlap_is_half_a_window(self) -> None:
        assert overlap_mel_length() == HOP_MEL_FRAMES // 2

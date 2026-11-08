# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MiniMax Music 3 row-to-request binding.

vLLM compacts and reorders the persistent batch whenever a request finishes or
is preempted, so a batch row number is only meaningful within the step that
produced it. Every failure pinned here was a real defect: state keyed by row
survived into a step where the row belonged to someone else, or state keyed by
one spelling of the request id was released under another and leaked.

These exercise the pure helpers and the binding arithmetic directly, so they
need no GPU and no engine.
"""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.minimax_music3.constants import (
    DEFAULT_MAX_AUDIO_FRAMES,
)
from vllm_omni.model_executor.models.minimax_music3.talker import (
    _CFG_UNCOND_SUFFIX,
    _first_int,
    _pair_id_from_request,
    _request_key,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestRequestKey:
    def test_engine_list_form_is_unwrapped(self) -> None:
        """The engine injects global_request_id as a single-element list.

        Keying state on the list's ``str()`` would produce "['req-1']", which
        never matches the bare id that ``on_requests_finished`` delivers, so
        every finished song would leak its frame buffer.
        """
        assert _request_key(["req-1"]) == "req-1"

    def test_bare_string_is_unchanged(self) -> None:
        assert _request_key("req-1") == "req-1"

    def test_list_and_bare_forms_agree(self) -> None:
        """Allocation and release must land on the same key."""
        assert _request_key(["req-1"]) == _request_key("req-1")

    def test_empty_and_missing_are_stable(self) -> None:
        assert _request_key(None) == ""
        assert _request_key([]) == ""

    def test_tuple_form(self) -> None:
        assert _request_key(("req-9",)) == "req-9"


class TestPairIdDerivation:
    def test_conditioned_request_is_its_own_pair(self) -> None:
        assert _pair_id_from_request("req-1") == "req-1"

    def test_companion_maps_to_its_partner(self) -> None:
        assert _pair_id_from_request(f"req-1{_CFG_UNCOND_SUFFIX}") == "req-1"

    def test_pair_members_agree_on_the_pair_id(self) -> None:
        cond = "req-42"
        uncond = f"req-42{_CFG_UNCOND_SUFFIX}"
        assert _pair_id_from_request(cond) == _pair_id_from_request(uncond)

    def test_a_request_ending_in_a_similar_string_is_not_confused(self) -> None:
        assert _pair_id_from_request("req-cfg_uncond") == "req-cfg_uncond"


class TestFirstInt:
    def test_serving_layer_list_wrapping(self) -> None:
        """tts_params values arrive wrapped in a single-element list."""
        assert _first_int([750], DEFAULT_MAX_AUDIO_FRAMES) == 750

    def test_bare_value(self) -> None:
        assert _first_int(750, DEFAULT_MAX_AUDIO_FRAMES) == 750

    def test_missing_falls_back(self) -> None:
        assert _first_int(None, DEFAULT_MAX_AUDIO_FRAMES) == DEFAULT_MAX_AUDIO_FRAMES
        assert _first_int([], DEFAULT_MAX_AUDIO_FRAMES) == DEFAULT_MAX_AUDIO_FRAMES

    def test_nonsense_falls_back_rather_than_raising(self) -> None:
        assert _first_int("abc", DEFAULT_MAX_AUDIO_FRAMES) == DEFAULT_MAX_AUDIO_FRAMES
        assert _first_int(0, DEFAULT_MAX_AUDIO_FRAMES) == DEFAULT_MAX_AUDIO_FRAMES
        assert _first_int(-5, DEFAULT_MAX_AUDIO_FRAMES) == DEFAULT_MAX_AUDIO_FRAMES


def _feedback_plan(
    request_ids: list[str],
    computed: list[int],
    scheduled: list[int],
    prompt_tokens: dict[str, int],
    has_codes: set[str],
) -> list[int]:
    """Mirror of the talker's feedback row selection.

    Returns the flat token offsets whose embedding gets replaced by an audio
    frame. Kept as a standalone function so the arithmetic can be checked
    without constructing a model.
    """
    offsets: list[int] = []
    running = 0
    for count in scheduled:
        offsets.append(running)
        running += count

    rows: list[int] = []
    for row, request_id in enumerate(request_ids):
        if request_id not in has_codes:
            continue
        if scheduled[row] != 1 or computed[row] < prompt_tokens.get(request_id, 0):
            continue
        rows.append(offsets[row])
    return rows


class TestFeedbackRowSelection:
    """The guard that decides which token embeddings become audio frames."""

    def test_decode_only_step_replaces_every_row(self) -> None:
        ids = ["a", "a__cfg_uncond", "b", "b__cfg_uncond"]
        rows = _feedback_plan(
            ids,
            computed=[120, 120, 90, 90],
            scheduled=[1, 1, 1, 1],
            prompt_tokens={i: 50 for i in ids},
            has_codes=set(ids),
        )
        assert rows == [0, 1, 2, 3]

    def test_prefill_row_is_never_overwritten(self) -> None:
        """A prompt token must keep its real embedding.

        The earlier guard compared the flat token count against a row budget,
        so a short prompt slipped through and had its first embeddings
        replaced with a previous request's audio frame.
        """
        rows = _feedback_plan(
            ["a", "b"],
            computed=[0, 300],
            scheduled=[8, 1],
            prompt_tokens={"a": 8, "b": 40},
            has_codes={"b"},
        )
        # Row 0 occupies flat tokens 0..7 and must be untouched; row 1's single
        # decode token sits at flat offset 8.
        assert rows == [8]

    def test_short_prefill_below_the_row_budget_is_still_safe(self) -> None:
        """The exact shape that defeated the old token-count guard."""
        rows = _feedback_plan(
            ["a"],
            computed=[0],
            scheduled=[10],
            prompt_tokens={"a": 10},
            has_codes={"a"},
        )
        assert rows == []

    def test_chunked_prefill_midway_is_not_fed_back(self) -> None:
        rows = _feedback_plan(
            ["a"],
            computed=[2048],
            scheduled=[2048],
            prompt_tokens={"a": 5000},
            has_codes={"a"},
        )
        assert rows == []

    def test_final_prefill_chunk_is_not_fed_back(self) -> None:
        """The step that completes a prompt still carries prompt tokens."""
        rows = _feedback_plan(
            ["a"],
            computed=[4096],
            scheduled=[904],
            prompt_tokens={"a": 5000},
            has_codes={"a"},
        )
        assert rows == []

    def test_mixed_batch_offsets_follow_the_flat_layout(self) -> None:
        """Tokens are a flat concatenation, so offsets are a running sum."""
        ids = ["p", "d1", "d2"]
        rows = _feedback_plan(
            ids,
            computed=[0, 500, 700],
            scheduled=[64, 1, 1],
            prompt_tokens={"p": 64, "d1": 40, "d2": 40},
            has_codes={"d1", "d2"},
        )
        assert rows == [64, 65]

    def test_request_without_codes_yet_is_skipped(self) -> None:
        """The first decode step after prefill has no previous frame."""
        rows = _feedback_plan(
            ["a"],
            computed=[50],
            scheduled=[1],
            prompt_tokens={"a": 50},
            has_codes=set(),
        )
        assert rows == []

    def test_reordered_batch_binds_by_request_not_position(self) -> None:
        """The same two requests in swapped rows get swapped offsets.

        This is the compaction case: holding codes in a row-indexed buffer
        would have handed each request the other's frame.
        """
        prompt_tokens = {"a": 50, "b": 50}
        first = _feedback_plan(["a", "b"], [100, 200], [1, 1], prompt_tokens, {"a"})
        swapped = _feedback_plan(["b", "a"], [200, 100], [1, 1], prompt_tokens, {"a"})
        assert first == [0]
        assert swapped == [1]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``nemotron_voicechat`` stage-input-processor pipeline.

Covers the two public entrypoints used by the orchestrator:

* :func:`eartts_prewarm_input` — arms stage 1 (EarTTS) with the
  user-supplied ``speaker_latent`` from stage 0's prompt.
* :func:`nemotron2eartts_async_chunk` — translates stage-0
  ``StreamingInput`` chunks into the per-step EarTTS payload over the
  chunk-transfer connector.

Also exercises the small ``_ensure_list`` / ``_coerce_speaker_latent`` /
``_get_async_chunk_state`` helpers through the public surface (and a
couple of direct calls so we can pin their contracts).
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.nemotron_voicechat import (
    _coerce_speaker_latent,
    _ensure_list,
    _get_async_chunk_state,
    eartts_prewarm_input,
    nemotron2eartts_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_request(
    request_id: str,
    *,
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
    speaker_latent: torch.Tensor | None = None,
) -> SimpleNamespace:
    """Build a minimal stage-0 request with the attrs the hook accesses."""
    add_info = None
    if speaker_latent is not None:
        add_info = {"speaker_latent": speaker_latent}
    return SimpleNamespace(
        external_req_id=request_id,
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [],
        output_token_ids=output_token_ids or [],
        additional_information=add_info,
    )


def _make_transfer_manager(*, put_chunk: dict[str, int] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        put_req_chunk=defaultdict(int, put_chunk or {}),
    )


# ---------------------------------------------------------------------------
# _ensure_list
# ---------------------------------------------------------------------------


class TestEnsureList:
    def test_list_input(self) -> None:
        assert _ensure_list([1, 2, 3]) == [1, 2, 3]

    def test_tensor_input(self) -> None:
        assert _ensure_list(torch.tensor([4, 5, 6])) == [4, 5, 6]

    def test_constant_list_like_input(self) -> None:
        """vLLM's ``ConstantList`` exposes data via the ``_x`` attribute."""
        wrapper = SimpleNamespace(_x=[7, 8, 9])
        assert _ensure_list(wrapper) == [7, 8, 9]

    def test_tuple_input(self) -> None:
        assert _ensure_list((1, 2)) == [1, 2]


# ---------------------------------------------------------------------------
# _coerce_speaker_latent
# ---------------------------------------------------------------------------


class TestCoerceSpeakerLatent:
    def test_returns_none_for_none(self) -> None:
        assert _coerce_speaker_latent(None) is None

    def test_picks_latent_from_flat_dict(self) -> None:
        latent = torch.randn(3, 4)
        out = _coerce_speaker_latent({"speaker_latent": latent})
        assert isinstance(out, torch.Tensor)
        assert torch.equal(out, latent)
        # Must be contiguous and on CPU.
        assert out.is_contiguous()
        assert out.device.type == "cpu"

    def test_picks_latent_from_nested_dict(self) -> None:
        """Prompt dicts wrap ``additional_information`` under a key."""
        latent = torch.randn(2, 3)
        out = _coerce_speaker_latent(
            {"additional_information": {"speaker_latent": latent}}
        )
        assert torch.equal(out, latent)

    def test_returns_none_for_missing_key(self) -> None:
        assert _coerce_speaker_latent({"other": 1}) is None

    def test_returns_none_for_empty_tensor(self) -> None:
        assert _coerce_speaker_latent({"speaker_latent": torch.zeros(0)}) is None

    def test_returns_none_for_zero_dim_tensor(self) -> None:
        assert _coerce_speaker_latent({"speaker_latent": torch.tensor(1.0)}) is None


# ---------------------------------------------------------------------------
# eartts_prewarm_input
# ---------------------------------------------------------------------------


class TestEarTTSPrewarmInput:
    def test_returns_placeholder_with_speaker_latent(self) -> None:
        latent = torch.randn(7, 8)
        prompt = {"additional_information": {"speaker_latent": latent}}

        result = eartts_prewarm_input(
            stage_id=1,
            stage0_request=None,
            original_prompt=prompt,
        )

        assert result is not None
        # Placeholder length matches Tref so the prefill chunk-count
        # contract on stage 1 stays consistent.
        assert result["prompt_token_ids"] == [0] * 7
        assert torch.equal(result["additional_information"]["speaker_latent"], latent)
        assert result["multi_modal_data"] is None
        assert result["mm_processor_kwargs"] is None

    def test_returns_none_when_no_speaker_latent(self) -> None:
        """Hook falls back to the default placeholder when nothing is supplied."""
        result = eartts_prewarm_input(
            stage_id=1,
            stage0_request=None,
            original_prompt={"additional_information": {}},
        )
        assert result is None

    def test_returns_none_when_prompt_is_none(self) -> None:
        result = eartts_prewarm_input(
            stage_id=1,
            stage0_request=None,
            original_prompt=None,
        )
        assert result is None


# ---------------------------------------------------------------------------
# _get_async_chunk_state
# ---------------------------------------------------------------------------


class TestGetAsyncChunkState:
    def test_creates_and_reuses_cache_on_transfer_manager(self) -> None:
        tm = SimpleNamespace()
        first = _get_async_chunk_state(tm)
        first["foo"] = 1
        second = _get_async_chunk_state(tm)
        # Cache survives across calls on the same transfer manager.
        assert second is first
        assert second["foo"] == 1


# ---------------------------------------------------------------------------
# nemotron2eartts_async_chunk — chunk 0 (prefill)
# ---------------------------------------------------------------------------


class TestChunkZero:
    def test_chunk0_emits_speaker_latent(self) -> None:
        latent = torch.randn(5, 8)
        tm = _make_transfer_manager()
        request = _make_request(
            "rid-0",
            prompt_token_ids=[0] * 5,
            speaker_latent=latent,
        )

        payload = nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=request,
            is_finished=False,
        )

        assert payload is not None
        assert torch.equal(payload["speaker_latent"], latent)
        assert payload["finished"].dtype == torch.bool
        assert payload["finished"].item() is False
        # State cache captures the prefill length for the race-detection
        # invariant on later decode chunks.
        state = _get_async_chunk_state(tm)["rid-0"]
        assert state["t_prefill"] == 5
        assert torch.equal(state["speaker_latent"], latent.detach().cpu().contiguous())

    def test_chunk0_returns_none_if_speaker_latent_missing(self) -> None:
        tm = _make_transfer_manager()
        request = _make_request("rid-bad", prompt_token_ids=[0] * 3)

        payload = nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=request,
            is_finished=False,
        )
        assert payload is None

    def test_chunk0_uses_state_cached_latent_after_first_call(self) -> None:
        """Once cached, the latent survives even if the live request loses it."""
        latent = torch.randn(4, 6)
        tm = _make_transfer_manager()
        request_first = _make_request("rid-c", prompt_token_ids=[0] * 4, speaker_latent=latent)
        nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=request_first,
            is_finished=False,
        )

        # Subsequent chunk-0 call (e.g. a retry) with the latent stripped
        # from the live request still finds the cached copy.
        request_retry = _make_request("rid-c", prompt_token_ids=[0] * 4)
        payload = nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=request_retry,
            is_finished=False,
        )
        assert payload is not None
        assert torch.equal(payload["speaker_latent"], latent)

    def test_returns_none_when_request_has_no_id(self) -> None:
        tm = _make_transfer_manager()
        request = SimpleNamespace(
            external_req_id=None,
            request_id=None,
            prompt_token_ids=[],
            output_token_ids=[],
            additional_information=None,
        )
        payload = nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=request,
        )
        assert payload is None


# ---------------------------------------------------------------------------
# nemotron2eartts_async_chunk — decode chunks (k >= 1)
# ---------------------------------------------------------------------------


class TestDecodeChunks:
    def test_decode_chunk_emits_latest_token(self) -> None:
        """chunk k extracts ``prompt_token_ids[-1]`` (= the new t_{k-1} token)."""
        latent = torch.randn(3, 4)
        tm = _make_transfer_manager()
        rid = "rid-d"
        # Arm the per-request state with a chunk-0 send first.
        nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=_make_request(rid, prompt_token_ids=[0] * 3, speaker_latent=latent),
            is_finished=False,
        )

        # Decode chunk 1: live session now has [0]*3 + [42].
        tm.put_req_chunk[rid] = 1
        request_decode = _make_request(rid, prompt_token_ids=[0, 0, 0, 42])

        payload = nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=request_decode,
            is_finished=False,
        )

        assert payload is not None
        assert payload["input_text_tokens"] == [42]
        assert payload["finished"].item() is False

    def test_decode_chunk_returns_none_on_empty_prompt(self) -> None:
        tm = _make_transfer_manager()
        rid = "rid-empty"
        tm.put_req_chunk[rid] = 1
        request = _make_request(rid, prompt_token_ids=[])
        payload = nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=request,
            is_finished=False,
        )
        assert payload is None

    def test_decode_chunk_race_detection_assert_triggers(self) -> None:
        """A mismatched ``len(prompt) != T_PREFILL + chunk_id`` flags a race."""
        latent = torch.randn(3, 4)
        tm = _make_transfer_manager()
        rid = "rid-race"
        # Arm state.
        nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=_make_request(rid, prompt_token_ids=[0] * 3, speaker_latent=latent),
        )

        # Decode chunk 2 but the live prompt is short by one (race).
        tm.put_req_chunk[rid] = 2
        request_decode = _make_request(rid, prompt_token_ids=[0, 0, 0, 42])

        with pytest.raises(AssertionError, match="save_async.*save_loop race"):
            nemotron2eartts_async_chunk(
                transfer_manager=tm,
                pooling_output=None,
                request=request_decode,
            )

    def test_terminal_chunk_forwards_final_sampled_token(self) -> None:
        """Terminal chunk appends ``output_token_ids[-1]`` for the closing frame."""
        latent = torch.randn(3, 4)
        tm = _make_transfer_manager()
        rid = "rid-finished"
        nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=_make_request(rid, prompt_token_ids=[0] * 3, speaker_latent=latent),
        )

        tm.put_req_chunk[rid] = 1
        request_term = _make_request(
            rid,
            prompt_token_ids=[0, 0, 0, 99],
            output_token_ids=[123],
        )

        payload = nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=request_term,
            is_finished=True,
        )
        assert payload is not None
        assert payload["input_text_tokens"] == [99, 123]
        assert payload["finished"].item() is True
        # State cache must be released on terminal chunk so the
        # transfer manager doesn't leak per-request memory.
        assert rid not in _get_async_chunk_state(tm)

    def test_terminal_chunk_with_empty_outputs_only_emits_prev_token(self) -> None:
        latent = torch.randn(3, 4)
        tm = _make_transfer_manager()
        rid = "rid-finished-empty"
        nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=_make_request(rid, prompt_token_ids=[0] * 3, speaker_latent=latent),
        )

        tm.put_req_chunk[rid] = 1
        request_term = _make_request(
            rid,
            prompt_token_ids=[0, 0, 0, 99],
            output_token_ids=[],
        )
        payload = nemotron2eartts_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=request_term,
            is_finished=True,
        )
        # Only the previous-step token survives — no final sampled token
        # is available to forward.
        assert payload["input_text_tokens"] == [99]
        assert payload["finished"].item() is True

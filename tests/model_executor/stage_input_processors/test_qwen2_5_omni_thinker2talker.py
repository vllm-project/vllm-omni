# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Light coverage for qwen2_5_omni.thinker2talker_full_payload.

Covers the finish-reason-aware stop-row trim contract: when the request
status is FINISHED_STOPPED, the builder must drop one row from the
accumulated hidden states (vLLM v1 appends the sampled stop token to
output_token_ids before check_stop, so the trailing hidden-state row
corresponds to the stop emission and must not reach the talker).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
    thinker2talker_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_request(
    prompt_token_ids,
    output_token_ids,
    status_name: str | None = "FINISHED_STOPPED",
):
    status = SimpleNamespace(name=status_name) if status_name else None
    return SimpleNamespace(
        request_id="r1",
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        all_token_ids=list(prompt_token_ids) + list(output_token_ids),
        status=status,
        sampling_params=None,
    )


def test_finished_stopped_trims_one_decode_row():
    """FINISHED_STOPPED: drop trailing hidden-state row so talker does not
    consume the stop-emission row.
    """
    prompt = [1, 2, 3]
    output = [10, 11, 12]
    request = _make_request(prompt, output, status_name="FINISHED_STOPPED")
    # 6 prompt+output rows + 1 stop-emission row = 7 hidden rows total.
    hidden = torch.arange(7 * 4, dtype=torch.float32).reshape(7, 4)
    pooling = {"hidden": hidden}

    payload = thinker2talker_full_payload(transfer_manager=None, pooling_output=pooling, request=request)

    assert payload is not None
    # ids.output had one trailing stop row dropped: 3 - 1 = 2 remaining.
    assert payload["ids"]["output"] == output[:-1]
    # embed.prefill must cover only the prompt rows.
    assert payload["embed"]["prefill"].shape[0] == len(prompt)
    # hidden_states.output covers the decode rows minus the dropped stop row.
    assert payload["hidden_states"]["output"].shape[0] == len(output) - 1


def test_finished_length_capped_keeps_all_rows():
    """FINISHED_LENGTH_CAPPED: no row drop; hidden_states.output covers
    all decode rows.
    """
    prompt = [1, 2, 3]
    output = [10, 11, 12]
    request = _make_request(prompt, output, status_name="FINISHED_LENGTH_CAPPED")
    hidden = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    pooling = {"hidden": hidden}

    payload = thinker2talker_full_payload(transfer_manager=None, pooling_output=pooling, request=request)

    assert payload is not None
    assert payload["ids"]["output"] == output
    assert payload["embed"]["prefill"].shape[0] == len(prompt)
    assert payload["hidden_states"]["output"].shape[0] == len(output)


def test_missing_hidden_returns_none():
    """Defensive: pooling_output without "hidden" returns None."""
    request = _make_request([1, 2], [3], status_name="FINISHED_STOPPED")
    assert thinker2talker_full_payload(transfer_manager=None, pooling_output={}, request=request) is None


def _make_request_with_raw_token_ids(prompt, output, all_ids, status_name=None):
    """Build a request without the ``_make_request`` ``list(...)`` coercion so a
    ``None`` token-id field survives to the builder unchanged.
    """
    status = SimpleNamespace(name=status_name) if status_name else None
    return SimpleNamespace(
        request_id="r1",
        prompt_token_ids=prompt,
        output_token_ids=output,
        all_token_ids=all_ids,
        status=status,
        sampling_params=None,
    )


def test_none_prompt_and_output_token_ids_do_not_crash():
    """Regression (linyueqian #4290791768): ``None`` token-id fields must be
    normalised to ``[]`` by ``ensure_list`` instead of reaching ``len(None)`` /
    ``list(None)`` and raising ``TypeError``.
    """
    request = _make_request_with_raw_token_ids(None, None, None)
    hidden = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)

    # No prompt and no output => target_rows == 0 => graceful None (no crash).
    payload = thinker2talker_full_payload(transfer_manager=None, pooling_output={"hidden": hidden}, request=request)
    assert payload is None


def test_none_output_token_ids_with_prompt_builds_payload():
    """``output_token_ids=None`` (a fresh, pre-decode request) must be treated
    as an empty list: prompt-only payload is still shipped.
    """
    request = _make_request_with_raw_token_ids((1, 2, 3), None, None)
    # Tuple prompt is normalised to list; 3 hidden rows match the prompt span.
    hidden = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)

    payload = thinker2talker_full_payload(transfer_manager=None, pooling_output={"hidden": hidden}, request=request)

    assert payload is not None
    assert payload["ids"]["prompt"] == [1, 2, 3]
    assert payload["ids"]["output"] == []
    assert payload["embed"]["prefill"].shape[0] == 3


def test_none_all_token_ids_falls_back_to_prompt_plus_output():
    """``all_token_ids=None`` must fall back to ``prompt + output`` rather than
    raising on ``list(None)``.
    """
    request = _make_request_with_raw_token_ids([1, 2], [10, 11], None)
    hidden = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)

    payload = thinker2talker_full_payload(transfer_manager=None, pooling_output={"hidden": hidden}, request=request)

    assert payload is not None
    assert payload["ids"]["prompt"] == [1, 2]
    assert payload["ids"]["output"] == [10, 11]
    assert payload["embed"]["prefill"].shape[0] == 2
    assert payload["hidden_states"]["output"].shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

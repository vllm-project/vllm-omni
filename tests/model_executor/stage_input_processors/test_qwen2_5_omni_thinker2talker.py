# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Light coverage for qwen2_5_omni.thinker2talker_full_payload.

The final sampled token closes the request before another decode forward can
produce its hidden state. The payload builder must therefore remove a sampled
stop token, or a length-capped trailing token, from the IDs sent to the talker.
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


def test_finished_stopped_trims_unrepresented_stop_token():
    """FINISHED_STOPPED: remove the stop token that has no decode row."""
    prompt = [1, 2, 3]
    output = [10, 11, 12]
    request = _make_request(prompt, output, status_name="FINISHED_STOPPED")
    # Prefill has three rows. The first two output tokens have decode rows;
    # output[2] is the sampled stop token and has no subsequent forward pass.
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    pooling = {"hidden": hidden}

    payload = thinker2talker_full_payload(transfer_manager=None, pooling_output=pooling, request=request)

    assert payload is not None
    # ids.output had one trailing stop token dropped: 3 - 1 = 2 remaining.
    assert payload["ids"]["output"] == output[:-1]
    # embed.prefill must cover only the prompt rows.
    assert payload["embed"]["prefill"].shape[0] == len(prompt)
    # hidden_states.output covers the two represented output tokens.
    assert payload["hidden_states"]["output"].shape[0] == len(output) - 1


def test_finished_length_capped_trims_unrepresented_final_token():
    """FINISHED_LENGTH_CAPPED: the final sampled token has no decode row."""
    prompt = [1, 2, 3]
    output = [10, 11, 12]
    request = _make_request(prompt, output, status_name="FINISHED_LENGTH_CAPPED")
    # Prefill produces all prompt rows and samples output[0]. Each subsequent
    # decode row samples the next token, so a three-token length cap has only
    # two represented output rows.
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    pooling = {"hidden": hidden}

    payload = thinker2talker_full_payload(transfer_manager=None, pooling_output=pooling, request=request)

    assert payload is not None
    assert payload["ids"]["output"] == output[:-1]
    assert payload["embed"]["prefill"].shape[0] == len(prompt)
    assert payload["hidden_states"]["output"].shape[0] == len(output) - 1


def test_cached_request_state_without_status_uses_stop_token_set():
    """The production worker request has no status; detect its sampled EOS."""
    prompt = [1, 2, 3]
    output = [10, 11, 12]
    request = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=prompt,
        output_token_ids=output,
        all_token_ids=[*prompt, *output],
        sampling_params=SimpleNamespace(
            ignore_eos=False,
            stop_token_ids=[],
            eos_token_id=12,
            _eos_token_id=None,
            all_stop_token_ids=[],
            _all_stop_token_ids=[],
        ),
    )
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)

    payload = thinker2talker_full_payload(
        transfer_manager=None,
        pooling_output={"hidden": hidden},
        request=request,
    )

    assert payload is not None
    assert payload["ids"]["output"] == output[:-1]
    assert payload["hidden_states"]["output"].shape[0] == len(output) - 1


def test_cached_request_state_without_status_reconciles_length_cap():
    """Without a stop token, captured decode rows still bound shipped IDs."""
    prompt = [1, 2, 3]
    output = [10, 11, 12]
    request = SimpleNamespace(
        request_id="r1",
        prompt_token_ids=prompt,
        output_token_ids=output,
        all_token_ids=[*prompt, *output],
        sampling_params=SimpleNamespace(
            ignore_eos=True,
            stop_token_ids=[],
        ),
    )
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)

    payload = thinker2talker_full_payload(
        transfer_manager=None,
        pooling_output={"hidden": hidden},
        request=request,
    )

    assert payload is not None
    assert payload["ids"]["output"] == output[:-1]
    assert payload["hidden_states"]["output"].shape[0] == len(output) - 1


def test_missing_hidden_returns_none():
    """Defensive: pooling_output without "hidden" returns None."""
    request = _make_request([1, 2], [3], status_name="FINISHED_STOPPED")
    assert thinker2talker_full_payload(transfer_manager=None, pooling_output={}, request=request) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

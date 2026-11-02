# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests for speculative payload token alignment."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_payload_adapter_uses_committed_tokens_and_drops_dflash_mask_rows() -> None:
    """DFlash query/mask rows must not become Talker condition rows."""

    class _Model:
        def embed_input_ids(self, input_ids):
            return input_ids.to(dtype=torch.float32).view(-1, 1)

    runner = object.__new__(GPUARModelRunner)
    runner.model = _Model()
    runner.device = torch.device("cpu")
    payload = runner._build_omni_pooler_payload(
        rid="r1",
        idx=0,
        start=0,
        end=3,
        hidden_states_cpu=torch.ones(3, 2),
        req_hidden_states_cpu={"r1": torch.full((3, 2), 9.0)},
        combined_hidden_states=None,
        combined_multimodal_outputs=None,
        mm_cpu={
            "hidden_states.layer_0": torch.full((3, 2), 8.0),
            "hidden_states.layer_24": torch.full((3, 2), 7.0),
        },
        audio_sparse_output=False,
        sparse_mm_index={},
        hidden_seq_len=3,
        scheduled_seq_len=3,
        committed_token_ids=[101, 102],
    )

    assert torch.equal(payload["hidden_states.layer_0"], torch.tensor([[101.0], [102.0]]))
    assert "hidden_states.layer_24" not in payload
    assert "hidden" not in payload


@pytest.mark.parametrize(
    ("use_dflash", "sampled", "expected"),
    [
        (True, [10, 11], [10, 11]),
        (True, [], []),
        (False, [10, 11], None),
    ],
)
def test_speculative_token_selection_tracks_committed_rows(use_dflash, sampled, expected) -> None:
    runner = object.__new__(GPUARModelRunner)
    runner.speculative_config = SimpleNamespace(use_dflash=lambda: use_dflash)
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={"r1": [1, 2, 3]})

    assert runner._get_speculative_committed_token_ids("r1", 0, scheduler_output, [sampled]) == expected


def test_first_speculative_delta_seeds_prefill_sampled_token() -> None:
    runner = object.__new__(GPUARModelRunner)
    runner.speculative_config = SimpleNamespace(use_dflash=lambda: True)
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner._pending_full_payload_send = {
        "r1": ({}, {}, {"hidden_states.layer_0": 3}, object())
    }
    runner.requests = {
        "r1": SimpleNamespace(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11])
    }
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={"r1": [4, 5, 6]})

    assert runner._get_speculative_committed_token_ids("r1", 0, scheduler_output, [[11, 12]]) == [10, 11, 12]

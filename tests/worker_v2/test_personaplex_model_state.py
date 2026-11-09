# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.personaplex.model_state import PersonaPlexModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_post_sample_keeps_request_slot_and_token_rows_aligned():
    state = object.__new__(PersonaPlexModelState)
    hook = Mock(return_value=torch.tensor([[11] * 8, [22] * 8]))
    state.model = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(session_mode="duplex")),
        post_sample_talker_mtp=hook,
        num_active_codebooks=8,
    )
    state._turn_codes = None
    state.intermediate_buffer = SimpleNamespace(
        buffers=[
            {"duplex": {"data_plane": True}, "slot": 0},
            {"duplex": {"data_plane": True}, "slot": 1},
            {"duplex": {"data_plane": True}, "slot": 2},
        ]
    )
    batch = SimpleNamespace(
        num_reqs=3,
        num_tokens=5,
        req_ids=["b", "a", "c"],
        idx_mapping_np=np.array([1, 0, 2]),
        num_computed_tokens_np=np.array([4, 0, 2]),
        num_scheduled_tokens=np.array([1, 3, 1]),
        prefill_len_np=np.array([5, 6, 3]),
        query_start_loc=torch.tensor([0, 1, 4, 5]),
    )
    hidden = torch.arange(20).reshape(5, 4).float()
    _, output = state.postprocess_model_output(hidden, batch, None)
    state.post_sample(batch, hidden, torch.tensor([[7], [8], [9]]), output)
    args = hook.call_args.kwargs
    assert args["req_ids"] == ["b", "c"]
    assert [info["slot"] for info in args["req_infos"]] == [1, 2]
    assert args["input_ids"].tolist() == [7, 9]
    assert torch.equal(args["hidden_states"], hidden[[0, 4]])
    assert output["codes"]["audio"][:, 0].tolist() == [11, 0, 0, 0, 22]


def test_post_sample_rejects_multi_token_rows():
    state = object.__new__(PersonaPlexModelState)
    state.model = SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(session_mode="duplex")))
    with pytest.raises(ValueError, match="one sampled text token"):
        state.post_sample(None, None, torch.zeros(1, 2), {})


@pytest.mark.parametrize("turn", [False, True])
def test_preprocess_reuses_batch_encode_and_preserves_turn_order(turn):
    state = object.__new__(PersonaPlexModelState)
    updates = []
    order = []

    def batch_preprocess(**kwargs):
        order.append("batch")
        assert kwargs["req_ids"] == ["r"]
        if not turn:
            kwargs["inputs_embeds"].fill_(3)
            return {"r"}
        return set()

    def preprocess(ids, embeds, **info):
        order.append("preprocess")
        assert info["request_id"] == "r"
        assert info["_omni_is_prefill"] is False
        return ids, torch.ones_like(embeds), {"mtp_inputs": (torch.ones(1, 4), torch.ones(1, 4))}

    def mtp(ids, embeds, hidden, step):
        order.append("mtp")
        return embeds, torch.full((1, 8), 7, dtype=torch.long)

    state.model = SimpleNamespace(
        preprocess_batch=batch_preprocess,
        preprocess=preprocess,
        talker_mtp=mtp,
        gpu_resident_buffer_keys=set(),
    )
    state.intermediate_buffer = SimpleNamespace(
        buffers=[{"req_id": "r"}],
        update=lambda *args, **kwargs: updates.append(args),
    )
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r"],
        idx_mapping_np=np.array([0]),
        query_start_loc_np=np.array([0, 1]),
        num_scheduled_tokens=np.array([1]),
        num_computed_tokens_np=np.array([5]),
        prefill_len_np=np.array([5]),
    )
    model_inputs = {"input_ids": torch.tensor([1]), "inputs_embeds": torch.zeros(1, 4)}
    state.run_preprocess(batch, model_inputs)
    assert order == (["batch", "preprocess", "mtp"] if turn else ["batch"])
    if turn:
        assert updates[-1][1]["codes"]["audio"].tolist() == [[7] * 8]
    else:
        assert model_inputs["inputs_embeds"].tolist() == [[3] * 4]

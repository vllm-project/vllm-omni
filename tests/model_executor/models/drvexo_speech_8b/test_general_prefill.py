# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Chunked prefill must score only its request's final prompt token."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.drvexo_speech_8b.general_input import BranchTokenizer
from vllm_omni.model_executor.models.drvexo_speech_8b.general_prefill import GeneralPrefill

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chunk_offsets_own_scoring_and_do_not_repeat_on_decode():
    seen = []

    def score(hidden, task):
        seen.append((hidden.item(), task))
        return hidden.clone()

    runtime = GeneralPrefill(SimpleNamespace(score=score))
    state = {"drvexo_general_decision": {"prompt_length": 5, "task": "choice"}}
    first = runtime.process(torch.tensor([[1.0], [2.0], [3.0]]), [state], [(0, 3)], [0])
    assert first["drvexo_general_status"][0].item() == 0
    second = runtime.process(torch.tensor([[4.0], [5.0], [99.0]]), [state, {}], [(0, 2), (2, 3)], [3, 0])
    assert seen == [(5.0, "choice")]
    assert [s.item() for s in second["drvexo_general_status"]] == [1, 0]
    runtime.process(torch.tensor([[6.0]]), [state], [(0, 1)], [5])
    assert len(seen) == 1
    state["drvexo_general_decision"]["task"] = "score"
    assert runtime.process(torch.tensor([[7.0]]), [state], [(0, 1)], [6])["drvexo_general_status"][0].item() == -1


def test_branch_budgets_keep_closing_chat_boundary():
    branch = BranchTokenizer(SimpleNamespace(encode=lambda text, **kw: list(text.encode())), 1536, 384)
    row = {"system_prompt": "", "instruction": "i" * 4000, "state_json": "s" * 4000}
    assert len(branch.prefix(row)) == 1536
    suffix = branch.candidate("a" * 5000)
    assert len(suffix) == 384
    assert suffix[-len(branch.close_user) :] == branch.close_user
    assert branch.prefix(row) == branch.prefix(row)

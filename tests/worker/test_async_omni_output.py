# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.worker import async_omni_output
from vllm_omni.worker.async_omni_output import AsyncOmniOutputRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("scheduled_tokens", [2, 8])
def test_prefix_cache_copies_only_available_scheduled_hidden(scheduled_tokens):
    runner = AsyncOmniOutputRunnerMixin()
    runner._maybe_get_combined_prefix_cache_tensors = Mock(return_value=(None, None))
    hidden = torch.arange(12.0).reshape(3, 4).requires_grad_()
    scheduler = SimpleNamespace(
        total_num_scheduled_tokens=scheduled_tokens, num_scheduled_tokens={"r1": scheduled_tokens}
    )

    cpu, combined, mm = runner._prepare_prefix_cache_pooler_payload_sources(
        hidden_states=hidden,
        staged_hidden_states_cpu=None,
        multimodal_outputs={},
        scheduler_output=scheduler,
        needs_scheduled_hidden_payload=True,
    )

    assert torch.equal(cpu, hidden[:scheduled_tokens])
    assert cpu.device.type == "cpu"
    assert cpu.is_contiguous()
    assert not cpu.requires_grad
    assert combined is None and mm is None
    assert runner._maybe_get_combined_prefix_cache_tensors.call_args.args[1] is cpu


@pytest.mark.parametrize("needs_hidden", [False, True])
def test_prefix_cache_reuses_staged_hidden_without_copy(monkeypatch, needs_hidden):
    runner = AsyncOmniOutputRunnerMixin()
    merged_hidden: dict[str, torch.Tensor] = {"r1": torch.ones(1, 4)}
    merged_mm: dict[str, object] = {"codes": {}}
    runner._maybe_get_combined_prefix_cache_tensors = Mock(return_value=(merged_hidden, merged_mm))
    copy = Mock(side_effect=AssertionError("unexpected hidden-state copy"))
    monkeypatch.setattr(async_omni_output, "_to_cpu_contiguous", copy)
    staged = torch.ones(2, 4) if needs_hidden else None

    cpu, combined, mm = runner._prepare_prefix_cache_pooler_payload_sources(
        hidden_states=torch.zeros(3, 4),
        staged_hidden_states_cpu=staged,
        multimodal_outputs={},
        scheduler_output=SimpleNamespace(total_num_scheduled_tokens=2, num_scheduled_tokens={"r1": 2}),
        needs_scheduled_hidden_payload=needs_hidden,
    )

    assert cpu is staged
    assert combined is merged_hidden
    assert mm is merged_mm
    copy.assert_not_called()

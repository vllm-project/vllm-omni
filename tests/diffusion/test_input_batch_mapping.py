# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.worker.input_batch import InputBatch, _select_states, scatter_latents
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_state(request_id: str, latent_value: float) -> StepRequestState:
    return StepRequestState(
        request_id=request_id,
        sampling=OmniDiffusionSamplingParams(),
        latents=torch.tensor([[latent_value]]),
        timesteps=torch.tensor([1.0]),
    )


def test_select_states_explicit_mapping_preserves_device_and_order():
    states = [_make_state(f"req-{index}", float(index)) for index in range(3)]
    selected, idx_mapping, idx_mapping_np = _select_states(
        states,
        torch.tensor([2, 0], dtype=torch.int64),
    )

    assert selected[0] is states[2]
    assert selected[1] is states[0]
    assert idx_mapping.dtype == torch.int32
    assert idx_mapping.device.type == "cpu"
    np.testing.assert_array_equal(idx_mapping_np, np.array([2, 0], dtype=np.int32))


def test_identity_mapping_reuses_cached_device_and_host_mappings(mocker):
    states = [_make_state(f"req-{index}", float(index)) for index in range(2)]
    arange = mocker.spy(torch, "arange")
    batch = InputBatch.make_batch(states)
    cached_idx_mapping = batch.idx_mapping
    cached_idx_mapping_np = batch.idx_mapping_np

    refreshed = InputBatch.make_batch(states, cached_batch=batch)

    assert refreshed is batch
    assert arange.call_count == 1
    assert refreshed.idx_mapping is cached_idx_mapping
    assert refreshed.idx_mapping_np is cached_idx_mapping_np
    assert refreshed.states[0] is states[0]
    assert refreshed.states[1] is states[1]


def test_identity_mapping_does_not_reuse_cached_explicit_reordering():
    states = [_make_state(f"req-{index}", float(index)) for index in range(2)]
    batch = InputBatch.make_batch(
        states,
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
    )
    reordered_idx_mapping = batch.idx_mapping

    rebuilt = InputBatch.make_batch(states, cached_batch=batch)

    assert rebuilt is batch
    assert rebuilt.idx_mapping is not reordered_idx_mapping
    assert rebuilt.request_ids == ["req-0", "req-1"]
    torch.testing.assert_close(rebuilt.idx_mapping, torch.tensor([0, 1], dtype=torch.int32))
    np.testing.assert_array_equal(rebuilt.idx_mapping_np, np.array([0, 1], dtype=np.int32))


def test_identity_mapping_rebuilds_when_batch_size_changes():
    states = [_make_state(f"req-{index}", float(index)) for index in range(2)]
    batch = InputBatch.make_batch(states)
    cached_idx_mapping = batch.idx_mapping
    states.append(_make_state("req-2", 2.0))

    rebuilt = InputBatch.make_batch(states, cached_batch=batch)

    assert rebuilt is batch
    assert rebuilt.idx_mapping is not cached_idx_mapping
    torch.testing.assert_close(rebuilt.idx_mapping, torch.tensor([0, 1, 2], dtype=torch.int32))
    np.testing.assert_array_equal(rebuilt.idx_mapping_np, np.array([0, 1, 2], dtype=np.int32))


def test_identity_mapping_takes_ownership_from_explicit_identity_mapping():
    states = [_make_state(f"req-{index}", float(index)) for index in range(2)]
    explicit_mapping = torch.tensor([0, 1], dtype=torch.int32)
    batch = InputBatch.make_batch(states, idx_mapping=explicit_mapping)

    refreshed = InputBatch.make_batch(states, cached_batch=batch)

    assert refreshed is batch
    assert refreshed.idx_mapping is not explicit_mapping
    explicit_mapping.fill_(1)
    torch.testing.assert_close(refreshed.idx_mapping, torch.tensor([0, 1], dtype=torch.int32))


def test_cached_identity_mapping_handles_request_reordering_and_scatter():
    first, second = _make_state("first", 1.0), _make_state("second", 2.0)
    batch = InputBatch.make_batch([first, second])
    mapping = batch.idx_mapping
    reordered_states = [second, first]

    refreshed = InputBatch.make_batch(reordered_states, cached_batch=batch)

    assert refreshed is batch
    assert refreshed.idx_mapping is mapping
    assert refreshed.request_ids == ["second", "first"]
    torch.testing.assert_close(refreshed.latents, torch.tensor([[2.0], [1.0]]))
    refreshed.latents = torch.tensor([[20.0], [10.0]])
    scatter_latents(reordered_states, refreshed)
    torch.testing.assert_close(first.latents, torch.tensor([[10.0]]))
    torch.testing.assert_close(second.latents, torch.tensor([[20.0]]))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract coverage for NPU routing capture and the native data plane."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import AuxOutputConfig
from vllm.distributed.aux_output_connector import worker
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxOutputSchedulerConnector,
    PackedBlockHashes,
)
from vllm.model_executor.layers.fused_moe import routed_experts_capturer
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_omni.platforms.npu.worker.aux_output import NPUAuxOutputMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Router:
    def _select_experts(self, logical_ids):
        return torch.ones_like(logical_ids, dtype=torch.float32), logical_ids


class _AscendMoE(torch.nn.Module):
    __module__ = "vllm_ascend.ops.fused_moe"

    def __init__(self):
        super().__init__()
        self.layer_id = 0
        self.router = _Router()


class _Runner(NPUAuxOutputMixin):
    def _sync_device(self):
        self.synchronized += 1


@pytest.fixture
def runner(monkeypatch):
    # Exercise the real capturer, block store, worker and scheduler connectors
    # with CPU tensors; importing vllm-ascend or owning NPU hardware is unnecessary.
    monkeypatch.setattr(routed_experts_capturer, "current_platform", SimpleNamespace(device_type="cpu"))
    monkeypatch.setattr(routed_experts_capturer, "get_forward_context", lambda: SimpleNamespace(dp_metadata=None))
    monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace(is_first_rank=True))
    runner = _Runner()
    runner.model = torch.nn.Sequential(_AscendMoE())
    runner.vllm_config = SimpleNamespace(
        aux_output_config=AuxOutputConfig(enable_return_routed_experts=True),
        model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: 1,
            get_num_experts=lambda: 16,
            get_num_experts_per_tok=lambda: 2,
            hf_text_config=SimpleNamespace(model_type="test_moe"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        parallel_config=SimpleNamespace(data_parallel_rank=0, tensor_parallel_size=1, decode_context_parallel_size=1),
        cache_config=SimpleNamespace(block_size=4, prefix_match_unit=None),
        max_concurrent_batches=2,
    )
    runner.input_batch = SimpleNamespace(req_ids=["r"], num_computed_tokens_cpu=np.array([0]))
    runner.query_start_loc = SimpleNamespace(np=np.array([0, 4]))
    runner.synchronized = 0
    runner._init_omni_aux_output(KVCacheConfig(8, [], []))
    try:
        yield runner
    finally:
        runner._close_omni_aux_output()


def _step(runner, *, emit_start=0, hashes=None, finished=()):
    scheduler_output = SimpleNamespace(
        aux_output_connector_metadata=AuxOutputConnectorMetadata(
            generation=0,
            requests={} if finished else {"r": emit_start},
            block_hashes=hashes or {},
            finished_requests=finished,
        ),
        scheduled_spec_decode_tokens={},
    )
    runner._begin_omni_aux_output_step(scheduler_output)
    return scheduler_output


def _capture(runner, ids):
    logical_ids = torch.tensor(ids, dtype=torch.int32)
    weights, returned = runner.model[0].router._select_experts(logical_ids)
    assert torch.equal(returned, logical_ids)
    assert weights.shape == logical_ids.shape
    return logical_ids.numpy()[:, None, :].astype(np.uint8)


def test_routing_reaches_scheduler_and_owns_legacy_batch_snapshot(runner):
    step = _step(runner)
    expected = _capture(runner, [[0, 1], [1, 2], [2, 3], [3, 4]])
    pending = runner._prepare_omni_aux_output()
    # Simulate bookkeeping and the next batch reusing the runner's buffers.
    runner.input_batch.num_computed_tokens_cpu[:] = 99
    runner.query_start_loc.np[:] = 99
    _capture(runner, [[7, 8]] * 4)
    output = runner._finish_omni_aux_output(pending, step, torch.tensor([[5]]))
    request = SimpleNamespace(request_id="r", num_tokens=5)
    rows = AuxOutputSchedulerConnector().take_output(request, output)
    np.testing.assert_array_equal(rows, expected)
    assert output["r"].token_start == 0
    assert runner.synchronized == 1


def test_speculative_rejection_drops_unaccepted_routing_suffix(runner):
    step = _step(runner)
    step.scheduled_spec_decode_tokens = {"r": [7, 8, 9]}
    expected = _capture(runner, [[0, 1], [1, 2], [2, 3], [3, 4]])
    pending = runner._prepare_omni_aux_output()
    output = runner._finish_omni_aux_output(pending, step, torch.tensor([[5, 6, -1, -1]]))
    np.testing.assert_array_equal(output["r"].rows, expected[:2])


def test_partial_prefill_accumulates_rows_and_no_forward_step_tears_down(runner):
    step = _step(runner)
    prefix = _capture(runner, [[0, 1], [1, 2], [2, 3], [3, 4]])
    pending = runner._prepare_omni_aux_output()
    assert runner._finish_omni_aux_output(pending, step, torch.tensor([[5]]), [0]) == {}

    # Both completed four-token blocks need their scheduler-provided hashes.
    step = _step(runner, hashes={"r": PackedBlockHashes(b"h0h1", 2)})
    runner.input_batch.num_computed_tokens_cpu[:] = 4
    suffix = _capture(runner, [[4, 5], [5, 6], [6, 7], [7, 8]])
    pending = runner._prepare_omni_aux_output()
    output = runner._finish_omni_aux_output(pending, step, torch.tensor([[5]]))
    np.testing.assert_array_equal(output["r"].rows, np.concatenate((prefix, suffix)))
    _step(runner, finished=("r",))
    assert not runner.aux_output_connector._requests
    store = runner.aux_output_connector._store
    runner._close_omni_aux_output()
    assert not store._thread.is_alive()
    assert runner.model[0].capture_fn is None
    runner._close_omni_aux_output()  # shutdown is idempotent


def test_reinitialization_does_not_wrap_router_twice(runner):
    selection = runner.model[0].router._select_experts
    runner._init_omni_aux_output(KVCacheConfig(8, [], []))
    assert runner.model[0].router._select_experts is selection
    step = _step(runner)
    expected = _capture(runner, [[0, 1]] * 4)
    output = runner._finish_omni_aux_output(runner._prepare_omni_aux_output(), step, torch.tensor([[5]]))
    np.testing.assert_array_equal(output["r"].rows, expected)


def test_disabled_capture_preserves_normal_runner_path(runner):
    runner._close_omni_aux_output()
    runner.vllm_config.aux_output_config.enable_return_routed_experts = False
    runner._init_omni_aux_output(KVCacheConfig(8, [], []))
    runner._begin_omni_aux_output_step(SimpleNamespace())
    assert runner._prepare_omni_aux_output() is None
    assert runner._finish_omni_aux_output(None, SimpleNamespace()) is None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 duplex admission, row ownership and partial-prefill policy."""

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45DuplexSampler
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sampler(mocker, infos, slots, prefill, computed=None):
    model = mocker.Mock(
        spec=MiniCPMO45OmniForConditionalGeneration,
        _mrv2_duplex_infos=infos,
        _minicpmo45_native_duplex_token_ids=lambda: {},
        _minicpmo45_chunk_terminator_token_ids=lambda _: {7},
        _sample_minicpmo45_native_duplex_rows=mocker.Mock(return_value=[7]),
        _sample_minicpmo45_native_duplex_rows_deferred=mocker.Mock(return_value=None),
    )
    batch = mocker.Mock(
        spec=InputBatch,
        num_reqs=len(infos),
        idx_mapping_np=np.array(slots),
        num_computed_prefill_tokens_np=np.array(computed if computed is not None else [0] * len(infos)),
        num_scheduled_tokens=np.full(len(infos), 2),
        prefill_len_np=np.array(prefill),
    )
    return MiniCPMO45DuplexSampler(mocker.Mock(spec=Sampler, return_value="sampled"), model), model, batch


def test_sampler_skips_partial_prefill_and_resets_history_on_new_segment(mocker):
    params = SamplingParams(temperature=0.0, top_k=100, top_p=0.8, seed=42, max_tokens=20)
    info = {
        "req_id": "r",
        "sampling_params": params,
        "duplex": {"data_plane": True, "session_id": "s", "seq": 0, "payload": {}},
    }
    sampler, model, batch = _sampler(mocker, [info], [5], [4])
    assert sampler(torch.zeros(1, 10), batch) == "sampled"
    model.prepare_duplex_sampling.assert_not_called()
    batch.num_computed_prefill_tokens_np[0] = 2
    sampler(torch.zeros(1, 10), batch)
    model._sample_minicpmo45_native_duplex_rows.assert_called_once()
    # A lookahead after the terminator does not mutate the session policy.
    sampler(torch.zeros(1, 10), batch)
    model._sample_minicpmo45_native_duplex_rows.assert_called_once()
    info["duplex"]["seq"] = 1
    sampler(torch.zeros(1, 10), batch)
    assert model._sample_minicpmo45_native_duplex_rows.call_count == 2
    assert model.prepare_duplex_sampling.call_args.args[2][0].row_idx == 0
    assert sampler._requests["r"][0] == 1
    generator = sampler._requests["r"][2]
    assert generator is sampler._generators["r"]
    # Preemption and slot reassignment preserve accepted history and RNG.
    torch.rand(1, generator=generator)
    rng_state = generator.get_state()
    history = sampler._requests["r"][1]
    batch.idx_mapping_np[0] = 2
    sampler.add_request(2, params)
    sampler(torch.zeros(1, 10), batch)
    assert sampler._requests["r"][2] is generator
    assert sampler._requests["r"][1] is history
    assert model._sample_minicpmo45_native_duplex_rows.call_count == 2
    assert torch.equal(generator.get_state(), rng_state)
    sampler.on_requests_finished({"r"})
    assert not sampler._requests and not sampler._generators


@pytest.mark.parametrize("shared_session", [False, True])
def test_mixed_batch_preserves_sampling_params_and_shared_session_order(mocker, shared_session):
    params = [SamplingParams(temperature=t, top_k=k, top_p=p, seed=42) for t, k, p in ((0.0, 100, 0.8), (0.7, 17, 0.9))]
    infos = [
        {
            "req_id": f"r-{i}",
            "sampling_params": p,
            "duplex": {"data_plane": True, "seq": 0, "session_id": "s" if shared_session else f"s-{i}"},
        }
        for i, p in enumerate(params)
    ]
    policies = []

    def sample(_logits, metadata, *, row_idxs, token_ids, row_params):
        policies.append((list(row_idxs), metadata.all_greedy, row_params))
        return [7] * len(row_idxs)

    sampler, model, batch = _sampler(mocker, infos, [5, 2], [2, 2])
    model._sample_minicpmo45_native_duplex_rows.side_effect = sample
    sampler(torch.zeros(2, 10), batch)
    # vLLM normalizes greedy parameters before the model sees them.
    expected_params = [(0.0, 0, 1.0), (0.7, 17, 0.9)]
    if shared_session:
        assert policies == [([0], False, expected_params), ([1], False, expected_params)]
        assert model._sample_minicpmo45_native_duplex_rows.call_count == 2
    else:
        assert policies == [([0, 1], False, expected_params)]
        model._sample_minicpmo45_native_duplex_rows.assert_called_once()
        call = model._sample_minicpmo45_native_duplex_rows.call_args
        assert call.kwargs["row_idxs"] == [0, 1]
        torch.testing.assert_close(call.args[1].temperature, torch.tensor([0.0, 0.7]))
    prepared = model.prepare_duplex_sampling.call_args.args[2]
    assert [(r.temperature, r.top_k, r.top_p) for r in prepared] == expected_params
    assert sampler._requests["r-0"][0] == 0
    assert sampler._requests["r-1"][0] == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for staged block-table writes")
@torch.inference_mode()
def test_reanchor_reads_real_mrv2_staged_block_tables_after_overwrite(mocker):
    from vllm.v1.worker.gpu.block_table import BlockTables

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex import window_kv

    device = torch.device("cuda")
    blocks = BlockTables([16, 16], 3, 32, [4, 4], device, [16, 16])
    # Slot 2 belongs to this request even when it is the only scheduled row.
    blocks.append_block_ids(2, ([3, 5, 9], [1, 6, 8]), overwrite=True)
    blocks.apply_staged_writes()
    blocks.append_block_ids(2, ([3, 9], [1, 8]), overwrite=True)
    blocks.apply_staged_writes()
    original = torch.arange(10 * 2 * 16 * 8, device=device, dtype=torch.float32).reshape(10, 2, 16, 8) / 1000
    kv_caches = [original.clone(), original.clone()]
    command = {"moved_from": 32, "delta": 16, "sink_blocks": 1, "old_computed_tokens": 48}
    intermediate = OmniIntermediateBuffer(3)
    intermediate.buffers[2] = {"duplex": {"stage0_reanchor": command}}
    runner = mocker.Mock(
        spec=OmniGPUModelRunner,
        req_states=mocker.Mock(
            spec=RequestState, req_id_to_index={"r": 2}, num_computed_tokens_np=np.array([0, 0, 32])
        ),
        block_tables=blocks,
        model_state=mocker.Mock(spec=OmniModelState, intermediate_buffer=intermediate),
        device=device,
        cache_config=mocker.Mock(block_size=16),
        kv_caches=kv_caches,
        kv_cache_group_ids=[0, 1],
        model=object(),
        _duplex_inv_freq=torch.tensor([1.0, 0.01], device=device),
    )
    schedule = mocker.Mock(spec=SchedulerOutput, num_scheduled_tokens={"r": 1}, scheduled_new_reqs=[])
    for group, expected_ids in enumerate(([3, 9], [1, 8])):
        row = window_kv.MiniCPMO45DuplexWorkerHelper.resolve_group_block_ids(runner, "r", 0, group)
        torch.testing.assert_close(row, torch.tensor(expected_ids, device=device, dtype=torch.int32))
        assert row.data_ptr() == blocks.block_tables[group].gpu[2].data_ptr()
    window_kv.MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner, schedule)
    for cache, retained in zip(kv_caches, (9, 8), strict=True):
        expected = original.clone()
        expected[retained, :, :, :4] = window_kv.rotate_keys(
            original[retained, :, :, :4].transpose(0, 1), 16, runner._duplex_inv_freq
        ).transpose(0, 1)
        torch.testing.assert_close(cache, expected)
    assert runner.req_states.num_computed_tokens_np[2] == 32
    assert "stage0_reanchor" not in intermediate.buffers[2]["duplex"]
    snapshots = [cache.clone() for cache in kv_caches]
    window_kv.MiniCPMO45DuplexWorkerHelper.maybe_apply_reanchor(runner, schedule)
    for actual, expected in zip(kv_caches, snapshots, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

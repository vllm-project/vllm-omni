# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniGenerationModelRunner contracts: empty-step lifecycle, output partition,
CPU-sync vs CUDA-async dispatch, and async-chunk slot recycling."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner, check_exact_input_shape

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStagedField:
    def __init__(self, data):
        self.np = data
        self._staged = []

    def stage_write_elem(self, idx, value):
        self._staged.append((idx, value))

    def apply_write(self):
        for idx, value in self._staged:
            self.np[idx] = value
        self._staged.clear()


def _make_runner(model_output, num_reqs=1, prompt_len=10):
    runner = object.__new__(OmniGenerationModelRunner)
    runner.device = torch.device("cpu")
    runner.model_config = MagicMock(hf_text_config=None)
    del runner.model_config.eos_token_id
    runner.postprocess = lambda *a, **kw: None
    runner._gen_model_output = model_output
    runner._gen_input_batch = SimpleNamespace(
        num_reqs=num_reqs, req_ids=[f"req-{i}" for i in range(num_reqs)], idx_mapping_np=np.arange(num_reqs)
    )
    runner.execute_model_state = SimpleNamespace(finished_req_ids={"finished"}, ec_connector_output=None)
    runner.kv_connector = SimpleNamespace(post_forward=MagicMock(return_value=None))
    runner.check_ep_fault = False
    runner.req_states = MagicMock(
        prompt_len=SimpleNamespace(np=np.full(num_reqs, prompt_len, dtype=np.int32)),
        num_computed_tokens=_FakeStagedField(np.zeros(num_reqs, dtype=np.int32)),
    )
    return runner


def test_control_only_step_keeps_lifecycle_and_skips_input_construction():
    runner = object.__new__(OmniGenerationModelRunner)
    order = []
    for name in (
        "_prepare_native_data_plane",
        "finish_requests",
        "free_states",
        "_apply_block_table_staged_writes_if_available",
    ):
        setattr(runner, name, lambda *args, name=name: order.append(name))
    for name in ("_handle_async_chunk_updates", "add_requests", "update_requests", "_sync_native_data_plane_payloads"):
        setattr(runner, name, MagicMock(side_effect=AssertionError("control step built model inputs")))
    output = object()
    runner.kv_connector = SimpleNamespace(no_forward=lambda _s: output)
    runner._merge_ec_connector_no_forward = lambda _s, value: value
    runner._attach_native_data_plane_signals = lambda value: value
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0, scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=[])
    )
    assert runner.execute_model(scheduler_output) is output
    assert order == [
        "_prepare_native_data_plane",
        "finish_requests",
        "free_states",
        "_apply_block_table_staged_writes_if_available",
    ]


def test_released_chunk_reuses_scheduler_output_and_slot_recycle_clears_state():
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_omni.core.sched.output import OmniCachedRequestData

    runner = object.__new__(OmniGenerationModelRunner)
    runner.model = SimpleNamespace(logits_processor=None)
    runner._remove_request = MagicMock()
    req_id_to_index = MagicMock()
    req_id_to_index.get.return_value = None
    runner.req_states = SimpleNamespace(
        add_request=MagicMock(), req_id_to_index=req_id_to_index, apply_staged_writes=MagicMock()
    )
    runner.pooling_runner = None
    runner.encoder_cache = None
    runner.model_state = SimpleNamespace(
        add_request=MagicMock(),
        apply_staged_writes=MagicMock(),
        remove_request=MagicMock(),
        intermediate_buffer=SimpleNamespace(remove_request=MagicMock()),
    )
    runner.block_tables = SimpleNamespace(append_block_ids=MagicMock())
    runner.lora_state = SimpleNamespace(add_request=MagicMock())
    runner.is_last_pp_rank = False
    runner.adaptive_verification = None
    runner.sampler = None
    cached = OmniCachedRequestData(
        req_ids=["req"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={"req": [1]},
        new_block_ids=[()],
        num_computed_tokens=[0],
        num_output_tokens=[0],
        prompt_token_ids={"req": [1]},
        additional_information={"req": None},
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    runner._handle_async_chunk_updates(scheduler_output)
    added = runner.model_state.add_request.call_args.args[1]
    assert (added.req_id, added.prompt_token_ids) == ("req", [1])


def test_sample_tokens_cpu_sync_owns_outputs_and_runs_connector_last():
    waveform = torch.arange(4, dtype=torch.float32)
    output = OmniOutput(text_hidden_states=torch.empty(0), multimodal_outputs={"codes": {"audio": [waveform]}})
    runner = _make_runner(output, num_reqs=1)
    runner._reserve_native_data_plane_outputs = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock(side_effect=lambda value: value)

    def post_forward(_finished):
        assert runner.req_states.num_computed_tokens.np.tolist() == [10]

    runner.kv_connector.post_forward.side_effect = post_forward
    result = OmniGenerationModelRunner.sample_tokens(runner)
    assert isinstance(result, OmniModelRunnerOutput)
    # Reserve precedes finalize; connector runs after token state is applied.
    runner._reserve_native_data_plane_outputs.assert_called_once_with(["req-0"])
    runner.kv_connector.post_forward.assert_called_once_with({"finished"})
    waveform.fill_(-1)
    assert torch.equal(result.multimodal_outputs[0]["codes.audio"], torch.arange(4, dtype=torch.float32))


def test_sample_tokens_uses_async_output_for_cuda_and_snapshots_req_ids(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = OmniOutput(text_hidden_states=torch.zeros(1), multimodal_outputs={"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    runner.device = SimpleNamespace(type="cuda")
    runner.main_stream = object()
    runner.output_copy_stream = object()
    runner.model_config.async_chunk = True
    runner._release_generation_slots = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock()
    monkeypatch.setattr(generation_runner, "_contains_cuda_tensor", lambda _: True)
    captured = {}

    class _FakeAsyncOutput:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(generation_runner, "OmniGenerationAsyncOutput", _FakeAsyncOutput)
    input_batch = runner._gen_input_batch
    result = generation_runner.OmniGenerationModelRunner.sample_tokens(runner)

    assert isinstance(result, _FakeAsyncOutput)
    assert captured["finalize_output"] is runner._finalize_native_data_plane_output
    assert captured["model_runner_output"].sampled_token_ids == [[]]
    runner._release_generation_slots.assert_called_once()
    output_req_ids = captured["model_runner_output"].req_ids
    input_batch.req_ids[0] = "reused"  # snapshot must insulate the published output
    assert output_req_ids == ["req-0"]


@pytest.mark.parametrize("exact", [True, False])
def test_exact_input_shape_models_reject_padded_batches(exact):
    # Qwen3-Omni Code2Wav splits input_ids by seq_token_counts (#6712).
    model = SimpleNamespace(requires_exact_input_shape=exact)
    check_exact_input_shape(model, torch.zeros(32, dtype=torch.long), 32)
    check_exact_input_shape(model, None, 16)
    if exact:
        with pytest.raises(RuntimeError, match="exact input shape"):
            check_exact_input_shape(model, torch.zeros(48, dtype=torch.long), 32)
    else:
        check_exact_input_shape(model, torch.zeros(48, dtype=torch.long), 32)

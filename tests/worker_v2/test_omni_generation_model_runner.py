"""Tests for OmniGenerationModelRunner.sample_tokens (V2).

Covers the core multimodal_outputs construction paths via _build_pooler_output:
  - OmniOutput with batched tensor multimodal_outputs → per-request slicing
  - OmniOutput with list multimodal_outputs → direct mapping (including None)
  - OmniOutput with dict scalar values → broadcast to all requests
  - None model output → returns None
  - Non-dict multimodal_outputs → [{}] * num_reqs
  - sampled_token_ids always emits empty lists per request (no token sampling)
  - req_states.num_computed_tokens updated to prompt_len after sample_tokens
"""

import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_execute_model_propagates_make_omni_output_failure(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    runner = object.__new__(generation_runner.OmniGenerationModelRunner)
    runner._prepare_native_data_plane = MagicMock()
    runner.finish_requests = MagicMock()
    runner.free_states = MagicMock()
    runner._handle_async_chunk_updates = MagicMock()
    runner.add_requests = MagicMock()
    runner.update_requests = MagicMock()
    runner._sync_native_data_plane_payloads = MagicMock()
    runner._apply_block_table_staged_writes_if_available = MagicMock()
    runner._dispatch_batch_descriptor = MagicMock(
        return_value=(
            SimpleNamespace(
                num_tokens=1,
                num_active_loras=0,
                cg_mode=None,
            ),
            None,
        )
    )
    input_batch = SimpleNamespace(
        positions=torch.zeros(1, dtype=torch.long),
        num_tokens=1,
        num_tokens_after_padding=1,
        is_padding=False,
    )
    runner.prepare_inputs = MagicMock(return_value=input_batch)
    runner.gather_batch_req_state = MagicMock(return_value=(SimpleNamespace(num_tokens=1), None))
    runner._prepare_mm_inputs = MagicMock(return_value=(torch.zeros(1, dtype=torch.long), None, None))
    runner._add_legacy_forward_inputs = MagicMock()
    runner.model_state = SimpleNamespace(
        prepare_inputs=lambda *_args: {},
        intermediate_buffer=SimpleNamespace(gather=lambda _batch: [{}]),
    )
    runner.req_states = object()
    runner.lora_config = None
    runner.vllm_config = object()
    runner._dummy_hidden = torch.zeros(1)
    runner.kv_connector = SimpleNamespace(
        pre_forward=lambda _output: None,
        post_forward=lambda _finished: None,
    )
    runner.model = MagicMock(return_value=torch.zeros(1))
    runner.model.requires_native_model_intermediate_buffer = True
    runner.model.make_omni_output.side_effect = RuntimeError("broken Code2Wav output")
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
    )
    monkeypatch.setattr(generation_runner, "set_forward_context", lambda *_args, **_kwargs: nullcontext())

    with pytest.raises(RuntimeError, match="broken Code2Wav output"):
        generation_runner.OmniGenerationModelRunner.execute_model(runner, scheduler_output)


def test_execute_model_does_not_reference_removed_perf_hook():
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    source = inspect.getsource(OmniGenerationModelRunner.execute_model)
    assert "_record_execution_batch" not in source


def test_execute_model_uses_shared_vllm_025_mm_input_contract():
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    source = inspect.getsource(OmniGenerationModelRunner.execute_model)
    assert "self._prepare_mm_inputs(" in source
    assert "self.model_state.get_mm_embeddings(" not in source
    assert '"input_ids": input_ids' in source


class _FakeInputBatch:
    """Minimal input batch for sample_tokens."""

    def __init__(self, num_reqs: int = 1, req_ids: list[str] | None = None):
        self.num_reqs = num_reqs
        self.req_ids = req_ids or [f"req-{i}" for i in range(num_reqs)]
        self.idx_mapping_np = np.arange(num_reqs, dtype=np.int32)


class _FakeStagedField:
    """Minimal mock for req_states fields that support staged writes."""

    def __init__(self, data: np.ndarray):
        self.np = data
        self._staged: list[tuple[int, int]] = []

    def stage_write_elem(self, idx: int, value: int) -> None:
        self._staged.append((idx, value))

    def apply_write(self) -> None:
        for idx, value in self._staged:
            self.np[idx] = value
        self._staged.clear()


class _FakeNpField:
    """Minimal mock for req_states fields with .np attribute."""

    def __init__(self, data: np.ndarray):
        self.np = data


def _make_omni_output(multimodal_outputs: dict | None = None) -> OmniOutput:
    """Create an OmniOutput with given multimodal_outputs."""
    return OmniOutput(
        text_hidden_states=torch.zeros(1),
        multimodal_outputs=multimodal_outputs,
    )


def _make_runner(
    model_output,
    num_reqs: int = 1,
    prompt_len: int = 10,
):
    """Build a minimal OmniGenerationModelRunner for sample_tokens testing."""
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    runner = object.__new__(OmniGenerationModelRunner)
    runner.device = torch.device("cpu")

    mc = MagicMock()
    del mc.eos_token_id
    mc.hf_text_config = None
    runner.model_config = mc

    runner.postprocess = lambda *a, **kw: None

    input_batch = _FakeInputBatch(num_reqs)
    runner._gen_model_output = model_output
    runner._gen_input_batch = input_batch
    runner.execute_model_state = SimpleNamespace(finished_req_ids={"finished"}, ec_connector_output=None)
    runner.kv_connector = SimpleNamespace(post_forward=MagicMock(return_value=None))
    runner.check_ep_fault = False

    req_states = MagicMock()
    req_states.prompt_len = _FakeNpField(
        np.full(num_reqs, prompt_len, dtype=np.int32),
    )
    req_states.num_computed_tokens = _FakeStagedField(
        np.zeros(num_reqs, dtype=np.int32),
    )
    runner.req_states = req_states

    return runner


class TestSampleTokensTensorOutput(unittest.TestCase):
    def test_single_request(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": torch.randn(1, 4, 8)})
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert isinstance(result, OmniModelRunnerOutput)
        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 1
        assert result.multimodal_outputs[0]["model_outputs"].shape == (4, 8)

    def test_multi_request(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": torch.randn(3, 2, 5)})
        runner = _make_runner(output, num_reqs=3)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 3
        for i in range(3):
            assert result.multimodal_outputs[i]["model_outputs"].shape == (2, 5)


class TestSampleTokensListOutput(unittest.TestCase):
    def test_list_of_tensors(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": [torch.randn(3, 2)]})
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 1
        assert result.multimodal_outputs[0]["model_outputs"].shape == (3, 2)

    def test_list_with_none(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": [None]})
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert result.multimodal_outputs == [{}]


class TestSampleTokensDictOutput(unittest.TestCase):
    def test_dict_with_batched_tensor(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"audio": torch.randn(2, 16000), "sr": 24000})
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 2
        assert result.multimodal_outputs[0]["audio"].shape == (16000,)
        assert result.multimodal_outputs[1]["audio"].shape == (16000,)
        assert torch.is_tensor(result.multimodal_outputs[0]["sr"])
        assert result.multimodal_outputs[0]["sr"].item() == 24000

    def test_dict_with_list_values(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"chunks": [torch.randn(10), torch.randn(20)]})
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert len(result.multimodal_outputs) == 2
        assert result.multimodal_outputs[0]["chunks"].shape == (10,)
        assert result.multimodal_outputs[1]["chunks"].shape == (20,)


class TestSampleTokensNoneOutput(unittest.TestCase):
    def test_none_model_output(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = _make_runner(None, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)
        assert result is None


class TestNonDictMultimodalOutputs(unittest.TestCase):
    """When multimodal_outputs is None or non-dict, per-request output is empty."""

    def test_none_multimodal_outputs(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output(multimodal_outputs=None)
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert result.pooler_output is None
        assert result.multimodal_outputs == [{}, {}]


class TestSampledTokenIds(unittest.TestCase):
    def test_empty_sampled_token_ids_per_request(self):
        """Generation models emit empty sampled_token_ids (no token sampling)."""
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": torch.randn(3, 2)})
        runner = _make_runner(output, num_reqs=3)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.sampled_token_ids) == 3
        for ids in result.sampled_token_ids:
            assert ids == []


class TestReqStatesUpdate(unittest.TestCase):
    """Verify that sample_tokens marks all tokens as computed."""

    def test_num_computed_tokens_set_to_prompt_len(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        prompt_len = 15
        output = _make_omni_output({"model_outputs": torch.randn(2, 4)})
        runner = _make_runner(output, num_reqs=2, prompt_len=prompt_len)

        OmniGenerationModelRunner.sample_tokens(runner)

        for i in range(2):
            assert runner.req_states.num_computed_tokens.np[i] == prompt_len


def test_sample_tokens_uses_async_output_for_cuda(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = _make_omni_output({"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    runner.device = SimpleNamespace(type="cuda")
    runner.main_stream = object()
    runner.output_copy_stream = object()
    runner.model_config.async_chunk = True
    runner._release_generation_slots = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock()

    captured = {}

    class _FakeAsyncOutput:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(generation_runner, "OmniGenerationAsyncOutput", _FakeAsyncOutput)

    result = generation_runner.OmniGenerationModelRunner.sample_tokens(runner)

    assert isinstance(result, _FakeAsyncOutput)
    assert captured["multimodal_outputs"] is output.multimodal_outputs
    assert captured["num_reqs"] == 1
    assert captured["main_stream"] is runner.main_stream
    assert captured["copy_stream"] is runner.output_copy_stream
    assert captured["finalize_output"] is runner._finalize_native_data_plane_output
    assert captured["model_runner_output"].sampled_token_ids == [[]]
    runner._release_generation_slots.assert_called_once()


def test_sample_tokens_snapshots_request_ids_before_async_finalize(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = _make_omni_output({"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    runner.device = SimpleNamespace(type="cuda")
    runner.main_stream = object()
    runner.output_copy_stream = object()
    runner.model_config.async_chunk = True
    runner._release_generation_slots = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock()

    captured = {}

    class _FakeAsyncOutput:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(generation_runner, "OmniGenerationAsyncOutput", _FakeAsyncOutput)

    input_batch = runner._gen_input_batch
    generation_runner.OmniGenerationModelRunner.sample_tokens(runner)
    output_req_ids = captured["model_runner_output"].req_ids
    assert output_req_ids == ["req-0"]

    # The scheduler can reuse and mutate the input-batch list while the async
    # output is still waiting for its D2H event.
    input_batch.req_ids[0] = "reused-request"
    assert output_req_ids == ["req-0"]


def test_sample_tokens_keeps_sync_output_for_cpu(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = _make_omni_output({"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    result = generation_runner.OmniGenerationModelRunner.sample_tokens(runner)

    assert isinstance(result, OmniModelRunnerOutput)
    assert result.multimodal_outputs[0]["model_outputs"].device.type == "cpu"


def test_sample_tokens_reserves_native_output_before_sync_finalize(monkeypatch):
    from vllm_omni.worker_v2 import omni_generation_model_runner as generation_runner

    output = _make_omni_output({"model_outputs": [torch.randn(4)]})
    runner = _make_runner(output, num_reqs=1)
    runner._reserve_native_data_plane_outputs = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock(side_effect=lambda value: value)
    generation_runner.OmniGenerationModelRunner.sample_tokens(runner)

    runner._reserve_native_data_plane_outputs.assert_called_once_with(["req-0"])


class TestMultimodalOutputsPassthrough(unittest.TestCase):
    """multimodal_outputs is a per-request list (tensor-only) on OmniModelRunnerOutput.

    OmniGenerationScheduler indexes it as mm_outputs[req_index], so it must be a
    list (not the raw dict). Each entry mirrors the per-request pooler payload.
    """

    def test_multimodal_outputs_on_result(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        mm = {"audio": [torch.randn(10)]}
        output = _make_omni_output(mm)
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert isinstance(result.multimodal_outputs, list)
        assert len(result.multimodal_outputs) == 1
        assert "audio" in result.multimodal_outputs[0]
        assert torch.is_tensor(result.multimodal_outputs[0]["audio"])

    def test_none_multimodal_outputs_becomes_empty_dict(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output(multimodal_outputs=None)
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        # No multimodal data -> one empty dict per request.
        assert result.multimodal_outputs == [{}]


class TestBlockTableWrites(unittest.TestCase):
    def test_skips_no_kv_block_table_without_fused_writer(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = object.__new__(OmniGenerationModelRunner)
        block_tables = MagicMock()
        block_tables.fused_writer = None
        runner.block_tables = block_tables

        runner._apply_block_table_staged_writes_if_available()

        block_tables.apply_staged_writes.assert_not_called()

    def test_applies_block_table_writes_when_writer_exists(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = object.__new__(OmniGenerationModelRunner)
        block_tables = MagicMock()
        block_tables.fused_writer = object()
        runner.block_tables = block_tables

        runner._apply_block_table_staged_writes_if_available()

        block_tables.apply_staged_writes.assert_called_once_with()


def test_async_chunk_slot_recycle_notifies_model_state_plugins():
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    runner = object.__new__(OmniGenerationModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"req": 0},
        prompt_len=SimpleNamespace(np=np.zeros(1, dtype=np.int32)),
        prefill_len=SimpleNamespace(np=np.zeros(1, dtype=np.int32)),
        total_len=MagicMock(),
        all_token_ids=MagicMock(),
        num_computed_tokens=MagicMock(),
        num_computed_prefill_tokens=np.zeros(1, dtype=np.int32),
        apply_staged_writes=MagicMock(),
    )
    runner.model_state = SimpleNamespace(
        remove_request=MagicMock(),
        intermediate_buffer=SimpleNamespace(remove_request=MagicMock()),
    )
    cached = SimpleNamespace(
        req_ids=["req"],
        prompt_token_ids={"req": [7]},
        new_block_ids=[()],
        additional_information={},
    )

    with patch(
        "vllm_omni.worker_v2.omni_generation_model_runner.OmniCachedRequestData",
        type(cached),
    ):
        runner._handle_async_chunk_updates(SimpleNamespace(scheduled_cached_reqs=cached))

    runner.model_state.remove_request.assert_called_once_with(0)
    runner.model_state.intermediate_buffer.remove_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()

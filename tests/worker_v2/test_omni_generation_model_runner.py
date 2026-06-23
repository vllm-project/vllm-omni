"""Tests for OmniGenerationModelRunner.sample_tokens (V2).

Covers the core pooler_output construction paths via _build_pooler_output:
  - OmniOutput with batched tensor multimodal_outputs → per-request slicing
  - OmniOutput with list multimodal_outputs → direct mapping (including None)
  - OmniOutput with dict scalar values → broadcast to all requests
  - None model output → returns None
  - Non-dict multimodal_outputs → [None] * num_reqs
  - sampled_token_ids always emits empty lists per request (no token sampling)
  - req_states.num_computed_tokens updated to prompt_len after sample_tokens
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = []


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
    runner._gen_kv_connector_output = None
    runner.execute_model_state = None

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
        assert len(result.pooler_output) == 1
        assert result.pooler_output[0]["model_outputs"].shape == (4, 8)

    def test_multi_request(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": torch.randn(3, 2, 5)})
        runner = _make_runner(output, num_reqs=3)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 3
        for i in range(3):
            assert result.pooler_output[i]["model_outputs"].shape == (2, 5)


class TestSampleTokensListOutput(unittest.TestCase):
    def test_list_of_tensors(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": [torch.randn(3, 2)]})
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 1
        assert result.pooler_output[0]["model_outputs"].shape == (3, 2)

    def test_list_with_none(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"model_outputs": [None]})
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 1
        assert result.pooler_output[0]["model_outputs"] is None


class TestSampleTokensDictOutput(unittest.TestCase):
    def test_dict_with_batched_tensor(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"audio": torch.randn(2, 16000), "sr": 24000})
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 2
        assert result.pooler_output[0]["audio"].shape == (16000,)
        assert result.pooler_output[1]["audio"].shape == (16000,)
        assert result.pooler_output[0]["sr"] == 24000

    def test_dict_with_list_values(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output({"chunks": [torch.randn(10), torch.randn(20)]})
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 2
        assert result.pooler_output[0]["chunks"].shape == (10,)
        assert result.pooler_output[1]["chunks"].shape == (20,)


class TestSampleTokensNoneOutput(unittest.TestCase):
    def test_none_model_output(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = _make_runner(None, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)
        assert result is None


class TestNonDictMultimodalOutputs(unittest.TestCase):
    """When multimodal_outputs is None or non-dict, pooler_output is [None]*num_reqs."""

    def test_none_multimodal_outputs(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = _make_omni_output(multimodal_outputs=None)
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 2
        assert result.pooler_output[0] is None
        assert result.pooler_output[1] is None


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


if __name__ == "__main__":
    unittest.main()

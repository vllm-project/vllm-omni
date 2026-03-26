"""Tests for OmniGenerationModelRunner.sample_tokens (V2).

Covers the core pooler_output construction paths:
  - tensor output → per-request slicing
  - list output → direct mapping (including None elements)
  - dict output → per-request key extraction
  - None output → None pooler_output list
  - eos_id fallback chain (model_config → hf_text_config → 0)
  - sampled_token_ids always emits [[eos_id]] per request
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = []


class _FakeInputBatch:
    """Minimal input batch for sample_tokens."""

    def __init__(self, num_reqs: int = 1, req_ids: list[str] | None = None):
        self.num_reqs = num_reqs
        self.req_ids = req_ids or [f"req-{i}" for i in range(num_reqs)]


def _make_runner(
    model_output,
    num_reqs: int = 1,
    eos_on_model_config: int | None = None,
    eos_on_hf_text_config: int | None = None,
):
    """Build a minimal OmniGenerationModelRunner for sample_tokens testing."""
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationModelRunner,
    )

    runner = object.__new__(OmniGenerationModelRunner)
    runner.device = torch.device("cpu")

    # model_config mock
    mc = MagicMock()
    if eos_on_model_config is not None:
        mc.eos_token_id = eos_on_model_config
    else:
        del mc.eos_token_id  # getattr returns default

    if eos_on_hf_text_config is not None:
        mc.hf_text_config = MagicMock()
        mc.hf_text_config.eos_token_id = eos_on_hf_text_config
    else:
        mc.hf_text_config = None

    runner.model_config = mc

    # Stub postprocess as no-op (avoids needing full sampler state)
    runner.postprocess = lambda *a, **kw: None

    # Set stored state
    input_batch = _FakeInputBatch(num_reqs)
    runner._gen_model_output = model_output
    runner._gen_input_batch = input_batch
    runner._gen_kv_connector_output = None
    runner.execute_model_state = None

    return runner


class TestSampleTokensTensorOutput(unittest.TestCase):
    def test_single_request(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = torch.randn(1, 4, 8)
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert isinstance(result, OmniModelRunnerOutput)
        assert len(result.pooler_output) == 1
        assert result.pooler_output[0]["model_outputs"].shape == (4, 8)

    def test_multi_request(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = torch.randn(3, 2, 5)
        runner = _make_runner(output, num_reqs=3)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 3
        for i in range(3):
            assert result.pooler_output[i]["model_outputs"].shape == (2, 5)


class TestSampleTokensListOutput(unittest.TestCase):
    def test_list_of_tensors(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = [torch.randn(3, 2)]
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 1
        assert result.pooler_output[0]["model_outputs"].shape == (3, 2)

    def test_list_with_none(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = [None]
        runner = _make_runner(output, num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 1
        assert result.pooler_output[0]["model_outputs"] is None


class TestSampleTokensDictOutput(unittest.TestCase):
    def test_dict_with_batched_tensor(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = {"audio": torch.randn(2, 16000), "sr": 24000}
        runner = _make_runner(output, num_reqs=2)
        result = OmniGenerationModelRunner.sample_tokens(runner)

        assert len(result.pooler_output) == 2
        assert result.pooler_output[0]["audio"].shape == (16000,)
        assert result.pooler_output[1]["audio"].shape == (16000,)
        assert result.pooler_output[0]["sr"] == 24000

    def test_dict_with_list_values(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        output = {"chunks": [torch.randn(10), torch.randn(20)]}
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


class TestEosIdFallback(unittest.TestCase):
    def test_eos_from_model_config(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = _make_runner(torch.randn(1, 2), num_reqs=1, eos_on_model_config=42)
        result = OmniGenerationModelRunner.sample_tokens(runner)
        assert result.sampled_token_ids == [[42]]

    def test_eos_from_hf_text_config(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = _make_runner(torch.randn(1, 2), num_reqs=1, eos_on_hf_text_config=99)
        result = OmniGenerationModelRunner.sample_tokens(runner)
        assert result.sampled_token_ids == [[99]]

    def test_eos_fallback_to_zero(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = _make_runner(torch.randn(1, 2), num_reqs=1)
        result = OmniGenerationModelRunner.sample_tokens(runner)
        assert result.sampled_token_ids == [[0]]


class TestSampledTokenIds(unittest.TestCase):
    def test_one_eos_per_request(self):
        from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

        runner = _make_runner(torch.randn(3, 2), num_reqs=3)
        result = OmniGenerationModelRunner.sample_tokens(runner)
        assert len(result.sampled_token_ids) == 3
        for ids in result.sampled_token_ids:
            assert len(ids) == 1


if __name__ == "__main__":
    unittest.main()

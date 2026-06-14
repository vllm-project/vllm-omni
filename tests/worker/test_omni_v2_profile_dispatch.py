# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.compilation.cuda_graph import CUDAGraphMode

from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner


def test_eager_dispatch_uses_eager_without_cudagraph_manager():
    runner = object.__new__(OmniGPUModelRunner)
    runner.cudagraph_manager = None
    runner.dp_size = 1
    runner.dp_rank = 0

    batch_desc, num_tokens_across_dp = runner._dispatch_batch_descriptor(
        num_reqs=1,
        num_toks=8,
        uniform_tok_count=8,
        use_eager=True,
    )

    assert batch_desc.cg_mode == CUDAGraphMode.NONE
    assert batch_desc.num_tokens == 8
    assert batch_desc.num_reqs == 1
    assert num_tokens_across_dp is None


def test_eager_dispatch_syncs_dp_padding(monkeypatch):
    runner = object.__new__(OmniGPUModelRunner)
    runner.cudagraph_manager = None
    runner.dp_size = 2
    runner.dp_rank = 1
    expected_tokens_across_dp = object()
    calls = []

    def sync_dp(cudagraph_manager, batch_desc, num_tokens, num_reqs, uniform_token_count, dp_size, dp_rank):
        calls.append((cudagraph_manager, batch_desc, num_tokens, num_reqs, uniform_token_count, dp_size, dp_rank))
        return batch_desc, expected_tokens_across_dp

    import vllm.v1.worker.gpu.dp_utils as dp_utils

    monkeypatch.setattr(dp_utils, "sync_cudagraph_and_dp_padding", sync_dp)

    batch_desc, num_tokens_across_dp = runner._dispatch_batch_descriptor(
        num_reqs=1,
        num_toks=8,
        uniform_tok_count=8,
        use_eager=True,
    )

    assert batch_desc.cg_mode == CUDAGraphMode.NONE
    assert batch_desc.num_tokens == 8
    assert batch_desc.num_reqs == 1
    assert num_tokens_across_dp is expected_tokens_across_dp
    assert calls == [(None, batch_desc, 8, 1, 8, 2, 1)]


def test_non_eager_dispatch_uses_cudagraph_manager():
    runner = object.__new__(OmniGPUModelRunner)
    expected = SimpleNamespace(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=8, num_reqs=1)
    calls = []
    runner.cudagraph_manager = SimpleNamespace(dispatch=lambda *args: calls.append(args) or expected)
    runner.dp_size = 1
    runner.dp_rank = 0

    batch_desc, num_tokens_across_dp = runner._dispatch_batch_descriptor(
        num_reqs=1,
        num_toks=8,
        uniform_tok_count=8,
        use_eager=False,
    )

    assert batch_desc is expected
    assert calls == [(1, 8, 8)]
    assert num_tokens_across_dp is None


if __name__ == "__main__":
    test_eager_dispatch_uses_eager_without_cudagraph_manager()
    test_non_eager_dispatch_uses_cudagraph_manager()


def test_skip_compiled_uses_eager_dispatch_without_overwriting_batch_descriptor():
    source = __import__("inspect").getsource(OmniGPUModelRunner.execute_model)

    assert "use_eager=is_profile or skip_compiled" in source
    assert "if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs" not in source


def test_fullgraph_requires_cudagraph_manager_contract():
    source = __import__("inspect").getsource(OmniGPUModelRunner.execute_model)

    assert "assert self.cudagraph_manager is not None" in source

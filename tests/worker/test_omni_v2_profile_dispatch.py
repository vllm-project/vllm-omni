# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm.compilation.cuda_graph import CUDAGraphMode

from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dp_size", [1, 2])
def test_dispatch_eager_and_manager_paths(monkeypatch, dp_size):
    runner = object.__new__(OmniGPUModelRunner)
    runner.cudagraph_manager = None
    runner.dp_size = dp_size
    runner.dp_rank = 1

    import vllm.v1.worker.gpu.dp_utils as dp_utils

    expected_tokens = object()
    sync_calls = []

    def sync_padding(*args, **kwargs):
        sync_calls.append((args, kwargs))
        return args[1], expected_tokens

    monkeypatch.setattr(
        dp_utils,
        "sync_cudagraph_and_dp_padding",
        sync_padding,
    )

    batch_desc, num_tokens_across_dp = runner._dispatch_batch_descriptor(
        num_reqs=1, num_toks=8, uniform_tok_count=8, num_active_loras=0, use_eager=True, max_query_len=8
    )
    assert batch_desc.cg_mode == CUDAGraphMode.NONE
    if dp_size == 1:
        assert num_tokens_across_dp is None
        assert sync_calls == []
    else:
        assert num_tokens_across_dp is expected_tokens

    # Non-eager dispatch goes through the cudagraph manager.
    expected = SimpleNamespace(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=8, num_reqs=1)
    dispatch_calls = []

    def dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        return expected

    runner.cudagraph_manager = SimpleNamespace(dispatch=dispatch)
    runner.dp_size = 1
    batch_desc, _ = runner._dispatch_batch_descriptor(
        num_reqs=1, num_toks=8, uniform_tok_count=8, num_active_loras=0, use_eager=False, max_query_len=8
    )
    assert batch_desc is expected
    assert dispatch_calls == [((1, 8, 8), {"num_active_loras": 0, "max_query_len": 8})]

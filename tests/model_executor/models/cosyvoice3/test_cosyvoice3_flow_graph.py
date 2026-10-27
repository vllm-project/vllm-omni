# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_full_batch import (
    PACKED_BACKENDS,
    MelOutput,
    _items,
    require_packed_backend,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("backend", PACKED_BACKENDS)
@pytest.mark.parametrize("streaming", [False, True])
@torch.inference_mode()
def test_flow_graph_replay_matches_eager_solve(monkeypatch, streaming, backend):
    require_packed_backend(monkeypatch, backend)
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", "1")
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0" if streaming else "1")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1" if streaming else "0")
    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
    # Random weights and inputs: seed them so a failure reproduces.
    torch.manual_seed(0)
    config = CosyVoice3Config()
    config.flow["pre_lookahead_layer"]["channels"] = 32
    # 256 channels in 16 conv groups keep the packed convolution (and graphs).
    config.flow["decoder"]["estimator"].update(dim=256, depth=2, heads=4, dim_head=64)
    model = module.CosyVoice3Code2Wav(config).cuda().bfloat16().eval()
    if streaming:
        monkeypatch.setattr(
            model, "_stream_hift_from_feat", lambda mel, cache_state, finalize: (mel, None if finalize else cache_state)
        )
    else:
        model.hift.upsample_rates = [1]
        model.hift.istft_params = {"hop_len": 1}

    def run(items, graph):
        estimator = getattr(model, "_packed_estimator", None)
        runner = None if estimator is None else estimator.graph_runner
        if estimator is not None and not graph:
            estimator.graph_runner = None
        torch.manual_seed(3)
        try:
            if streaming:
                return [audio for audio, _ in model.forward_streaming_batch(items, n_timesteps=3)]
            return model.forward_batch(items, n_timesteps=3)
        finally:
            if estimator is not None:
                estimator.graph_runner = runner

    extra = dict(token_offset_tokens=2, finalize=False, cache_state=None) if streaming else {}
    # The second batch reuses the first one's padded layout with new lengths.
    batches = [_items([(7, 60), (13, 41), (3, 97)], **extra), _items([(5, 71), (9, 30), (11, 88)], **extra)]
    # The graph pads the packed length to a multiple of 128, so Inductor runs
    # kernels specialized for 16-divisible sizes that the eager layout does not
    # hit; their reductions round a few elements one bf16 ULP apart, and CFG
    # amplifies that over the Euler steps to ~3e-2. A misread KV row is O(1).
    tolerance = 5e-2
    run(batches[0], graph=False)  # compile
    for items in batches:
        actual = run(items, graph=True)
        expected = run(items, graph=False)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=tolerance, atol=tolerance)
            assert (output - reference).float().norm() <= 1e-2 * reference.float().norm()
    runner = model._packed_estimator.graph_runner
    assert len(runner.graphs) == 1

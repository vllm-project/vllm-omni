# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_full_batch import MelOutput, _items
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("streaming", [False, True])
@torch.inference_mode()
def test_flow_graph_replay_matches_eager_solve(monkeypatch, streaming):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the opt-in packed backend requires Hopper FA3")
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", "1")
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0" if streaming else "1")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1" if streaming else "0")
    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
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
    run(batches[0], graph=False)  # compile
    for items in batches:
        actual = run(items, graph=True)
        expected = run(items, graph=False)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
    runner = model._packed_estimator.graph_runner
    assert len(runner.graphs) == 1


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_incremental_graph_reuses_bucket_with_request_caches(monkeypatch):
    from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.incremental import solve_flow_euler_incremental
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import PackedDiT
    from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_packed_streaming_enabled

    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    if not cosyvoice3_packed_streaming_enabled():
        pytest.skip("packed FlashAttention is unavailable on this GPU/build")
    torch.manual_seed(71)
    dit = (
        DiT(dim=256, depth=2, heads=4, dim_head=64, mel_dim=80, mu_dim=80, spk_dim=80, dropout=0, static_chunk_size=50)
        .cuda()
        .bfloat16()
        .eval()
    )
    estimator = PackedDiT(dit)
    data = [[torch.randn(1, 180, 80, device="cuda", dtype=torch.bfloat16) for _ in range(3)] for _ in range(3)]
    speakers = [torch.randn(1, 80, device="cuda", dtype=torch.bfloat16) for _ in data]
    span = torch.linspace(0, 1, 4, device="cuda", dtype=torch.bfloat16)
    caches, graph_caches = [None] * 3, [None] * 3
    recorded = None
    # Three requests occupy four slots; the final two calls reuse one bucket
    # with different prefix/suffix distributions, including a partial block.
    for lengths in ((30, 50, 80), (50, 80, 100), (80, 100, 130), (100, 125, 140)):
        args = (
            [d[0][:, :n] for d, n in zip(data, lengths)],
            span,
            [d[1][:, :n] for d, n in zip(data, lengths)],
            speakers,
            [d[2][:, :n] for d, n in zip(data, lengths)],
        )
        monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", "0")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected, caches = solve_flow_euler_incremental(
                estimator, *args, cfg_rate=0.7, schedule=(3,), caches=caches
            )
            monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", "1")
            actual, graph_caches = solve_flow_euler_incremental(
                estimator, *args, cfg_rate=0.7, schedule=(3,), caches=graph_caches
            )
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=3e-2, atol=3e-2)
        runner = estimator.incremental_runner
        current = next(reversed(runner.graphs.values()))
        if lengths == (80, 100, 130):
            recorded = current
        elif lengths == (100, 125, 140):
            assert current is recorded
        for cache, reference in zip(graph_caches, caches):
            assert cache.frames == reference.frames
            for step, ref_step in zip(cache.steps, reference.steps):
                torch.testing.assert_close(step.conv1_cache, ref_step.conv1_cache, rtol=3e-2, atol=3e-2)
                torch.testing.assert_close(step.conv2_cache, ref_step.conv2_cache, rtol=3e-2, atol=3e-2)
                for kv, ref_kv in zip(step.kv, ref_step.kv):
                    for value, ref_value in zip(kv, ref_kv):
                        torch.testing.assert_close(value, ref_value, rtol=3e-2, atol=3e-2)

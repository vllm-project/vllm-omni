# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

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


def _tiny_packed_streaming(monkeypatch, *, prefix_gib: str, graphs: str):
    """A small bf16 packed streaming code2wav whose HiFT stand-in returns the
    new mel and keeps an empty streaming state."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the opt-in packed backend requires Hopper FA3")
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav as module
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH", graphs)
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")
    monkeypatch.setenv("COSYVOICE3_FLOW_PREFIX_CACHE_GIB", prefix_gib)
    monkeypatch.setattr(module, "CausalHiFTGenerator", MelOutput)
    config = CosyVoice3Config()
    config.flow["pre_lookahead_layer"]["channels"] = 32
    config.flow["decoder"]["estimator"].update(dim=256, depth=2, heads=4, dim_head=64)
    torch.manual_seed(0)  # the same weights on every call
    model = module.CosyVoice3Code2Wav(config).cuda().bfloat16().eval()
    monkeypatch.setattr(
        model, "_stream_hift_from_feat", lambda mel, cache_state, finalize: (mel, None if finalize else {})
    )
    return model


def _stream(model, requests):
    """Two non-final hops (20 and 40 tokens), then the final hop; every hop's mel."""
    states, emitted, mels = [None] * len(requests), [0] * len(requests), []
    for hop in (20, 40, None):
        items = []
        for request, state, offset in zip(requests, states, emitted):
            token = request["token"] if hop is None else request["token"][:, :hop]
            items.append(
                dict(request, token=token, token_offset_tokens=offset, finalize=hop is None, cache_state=state)
            )
        values = model.forward_streaming_batch(items, n_timesteps=3)
        mels += [mel for mel, _ in values]
        states = [state for _, state in values]
        emitted = [0 if hop is None else hop - 3] * len(requests)
    return mels


# The second request's second hop resumes from 50 cached frames (13 prompt
# tokens and 20 generated cover one attention chunk); the others recompute.
_REQUESTS = [(7, 60), (13, 41), (3, 97)]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_compiled_prefix_hops_read_only_their_own_cache(monkeypatch):
    """Compiled prefix-cache hops match full recompute although the pool starts
    full of NaN, as reused pages hold other requests' state: a hop reads only
    what this request's earlier hops stored, the first hop's included."""
    torch.manual_seed(5)
    requests = _items(_REQUESTS)
    expected = _stream(_tiny_packed_streaming(monkeypatch, prefix_gib="0", graphs="0"), requests)
    model = _tiny_packed_streaming(monkeypatch, prefix_gib="0.25", graphs="0")
    cache = model._get_flow_prefix_cache(3)
    cache.kv.fill_(float("nan"))
    cache.conv_input.fill_(float("nan"))
    solved = []
    solve = cache.solve
    monkeypatch.setattr(
        cache, "solve", lambda queries, *a, **k: solved.extend(q.start for q in queries) or solve(queries, *a, **k)
    )
    actual = _stream(model, requests)
    assert max(solved) > 0
    # The two paths pack rows differently, so bf16 rounding differs by an ulp
    # or two; a stale read moves the mel by 0.3 or more, or makes it NaN.
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=5e-2)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@torch.inference_mode()
def test_captured_flow_graphs_serve_a_stream_without_recording(monkeypatch):
    """Graphs captured at startup serve prefix-cache hops and final hops of
    short streams: nothing records on demand and the mel matches eager Flow."""
    model = _tiny_packed_streaming(monkeypatch, prefix_gib="0.25", graphs="1")
    # The HiFT stand-in has no FFT plans to warm.
    monkeypatch.setattr(model, "_get_hift_batcher", lambda: SimpleNamespace(warm_fft_plans=lambda frames: 0))
    # Three requests, 128 frames a row at most: one row slot of 4, up to 512 frames.
    model.capture_graphs(n_timesteps=3, row_frames=128)
    runner = model._packed_estimator.graph_runner
    prefix = model._flow_prefix_cache
    assert len(runner.graphs) == len(prefix.graphs) == 1 + 2 + 4 + 8
    torch.manual_seed(5)
    requests = _items(_REQUESTS)
    graphs = (runner, prefix.graphs)
    model._packed_estimator.graph_runner, prefix.graphs = None, None
    expected = _stream(model, requests)
    model._packed_estimator.graph_runner, prefix.graphs = graphs
    counts = (len(runner.graphs), len(prefix.graphs))
    actual = _stream(model, requests)
    assert (len(runner.graphs), len(prefix.graphs)) == counts
    assert prefix.used_pages == 0
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

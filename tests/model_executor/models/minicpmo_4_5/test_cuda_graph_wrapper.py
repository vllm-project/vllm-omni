# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from vllm.platforms import current_platform

import vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper as wrapper_module
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    HiFTGenerator,
)
from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
    CFMGraphWrapper,
    HiFTGraphWrapper,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


class _F0Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv1d(80, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(1).abs()


class _DeterministicSineGen(nn.Module):
    """Remove source RNG so eager and replay compare only execution paths."""

    def __init__(self, num_harmonics: int) -> None:
        super().__init__()
        self.num_harmonics = num_harmonics

    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (*f0.shape[:-1], self.num_harmonics)
        sine = f0.new_zeros(shape)
        uv = f0.new_ones((*f0.shape[:-1], 1))
        return sine, uv, sine


def _small_hift() -> HiFTGenerator:
    hift = HiFTGenerator(
        base_channels=32,
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
        f0_predictor=_F0Predictor(),
    )
    hift.m_source.l_sin_gen = _DeterministicSineGen(hift.nb_harmonics + 1)
    return hift.eval().cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hift_graph_replay_matches_eager_for_uncached_and_cached_shapes() -> None:
    torch.manual_seed(0)
    hift = _small_hift()
    token2wav = SimpleNamespace(
        hift=hift,
        flow=SimpleNamespace(
            encoder=SimpleNamespace(pre_lookahead_layer=SimpleNamespace(pre_lookahead_len=3)),
            token_mel_ratio=2,
        ),
        mel_cache_len=2,
        source_cache_len=960,
    )
    wrapper = HiFTGraphWrapper(
        token2wav,
        connector_config={"codec_chunk_frames": 2, "codec_left_context_frames": 3},
        capture_batch_sizes=[1],
    )
    wrapper.capture()

    cases = (
        (torch.randn(1, 80, 4, device="cuda"), torch.zeros(1, 1, 0, device="cuda")),
        (torch.randn(1, 80, 6, device="cuda"), torch.randn(1, 1, 960, device="cuda")),
    )
    with torch.inference_mode():
        for speech_feat, cache_source in cases:
            expected_speech, expected_source = hift.inference(speech_feat, cache_source)
            actual_speech, actual_source = wrapper.replay(speech_feat, cache_source)
            torch.testing.assert_close(actual_speech, expected_speech, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(actual_source, expected_source, rtol=1e-4, atol=1e-5)


class _FakeGraph:
    def replay(self) -> None:
        return None


def _fake_wrapper(monkeypatch: pytest.MonkeyPatch) -> HiFTGraphWrapper:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = object.__new__(HiFTGraphWrapper)
    wrapper.capture_batch_sizes = [1]
    wrapper.graph = {}
    wrapper.static_speech_inputs = {}
    wrapper.static_cache_source_inputs = {}
    wrapper.static_magnitude_outputs = {}
    wrapper.static_phase_outputs = {}
    wrapper.static_cache_source_outputs = {}
    wrapper.lazy_graph_count = 0
    wrapper.max_lazy_graphs = 1
    wrapper.decode_fn = Mock(return_value=(torch.tensor([[99.0]]), torch.tensor([[[98.0]]])))
    wrapper.finalize_fn = lambda magnitude, phase: magnitude + phase

    def capture(batch_size: int, num_frames: int, cache_len: int) -> None:
        key = (batch_size, num_frames, cache_len)
        wrapper.graph[key] = _FakeGraph()
        wrapper.static_speech_inputs[key] = torch.zeros(batch_size, 80, num_frames)
        wrapper.static_cache_source_inputs[key] = torch.zeros(batch_size, 1, cache_len)
        wrapper.static_magnitude_outputs[key] = torch.ones(batch_size, 1, num_frames)
        wrapper.static_phase_outputs[key] = torch.ones(batch_size, 1, num_frames)
        wrapper.static_cache_source_outputs[key] = torch.ones(batch_size, 1, num_frames)

    wrapper._capture = Mock(side_effect=capture)
    return wrapper


def test_unseen_shape_is_lazily_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _fake_wrapper(monkeypatch)
    speech, source = wrapper.replay(torch.randn(1, 80, 7), torch.zeros(1, 1, 0))

    wrapper._capture.assert_called_once_with(1, 7, 0)
    assert wrapper.lazy_graph_count == 1
    assert speech.shape == (1, 1, 7)
    assert source.shape == (1, 1, 7)
    wrapper.decode_fn.assert_not_called()


def test_lazy_capture_limit_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _fake_wrapper(monkeypatch)
    wrapper.lazy_graph_count = wrapper.max_lazy_graphs
    speech_feat = torch.randn(1, 80, 9)
    cache_source = torch.zeros(1, 1, 0)

    result = wrapper.replay(speech_feat, cache_source)

    wrapper._capture.assert_not_called()
    wrapper.decode_fn.assert_called_once_with(speech_feat, cache_source)
    assert result is wrapper.decode_fn.return_value


# ---------------------------------------------------------------------------
# CFMGraphWrapper tests
# ---------------------------------------------------------------------------


class _MiniDiT(nn.Module):
    """Minimal DiT-like module with blocks_forward_chunk for CFM graph testing.

    Mimics the upstream cosyvoice2 DiT's cache semantics:
    - CausalConv1d.forward_chunk: cat([cnn_cache, x], dim=time) when cache is not None
    - Attention.forward_chunk: cat([k, k_cache], dim=seq) when att_cache is not None
    """

    def __init__(self, in_dim: int = 16, hidden: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.final_layer = nn.Linear(hidden, in_dim)

    def blocks_forward_chunk(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None,
        cnn_cache: torch.Tensor | None = None,
        att_cache: torch.Tensor | None = None,
        cnn_cache_buffer: torch.Tensor | None = None,
        att_cache_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        for b_idx in range(len(self.blocks)):
            # Simulate CausalConv1d: cnn_cache contributes to the first frames.
            # Upstream: if cnn_cache[b_idx] is None, creates zeros internally (same as zeros).
            cnn_b = cnn_cache[b_idx] if cnn_cache is not None else None
            if cnn_b is not None:
                x[:, : cnn_b.shape[2], :] += cnn_b.transpose(1, 2)
            # Simulate Attention: att_cache contributes a bias when non-empty.
            # Upstream: if att_cache[b_idx] is None, skips cat entirely (different path).
            att_b = att_cache[b_idx] if att_cache is not None else None
            if att_b is not None and att_b.shape[3] > 0:
                x += att_b.sum(dim=(1, 2), keepdim=False).unsqueeze(1)
            x = self.blocks[b_idx](x)
            x = x + t
            cnn_cache_buffer[b_idx] = x[:, -2:, :].transpose(1, 2).contiguous()
            dt = x.shape[1]
            att_cache_buffer[b_idx][:, :, :dt, :] = x.unsqueeze(1)
        x = self.final_layer(x)
        x = x.transpose(1, 2)
        return x


def _cfm_inputs(
    batch_size: int, chunk_size: int, old_att_len: int, *, device: str = "cuda"
) -> tuple[torch.Tensor, ...]:
    depth = 2
    hidden = 8
    estimator_input = torch.randn(batch_size, 16, chunk_size, device=device)
    time_emb = torch.randn(batch_size, 1, hidden, device=device)
    cnn_cache = torch.randn(depth, batch_size, hidden, 2, device=device)
    att_cache = torch.randn(depth, batch_size, 1, old_att_len, hidden, device=device)
    cnn_out = torch.empty(depth, batch_size, hidden, 2, device=device)
    att_out = torch.empty(depth, batch_size, 1, old_att_len + chunk_size, hidden, device=device)
    return estimator_input, time_emb, cnn_cache, att_cache, cnn_out, att_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cfm_graph_replay_matches_eager_for_uncached_and_cached_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _MiniDiT().eval().cuda()
    wrapper = CFMGraphWrapper(graph_fn=estimator.blocks_forward_chunk, max_graphs=32)

    with torch.inference_mode():
        for _, chunk_size, old_att_len in ((2, 10, 0), (2, 10, 5)):
            inputs = _cfm_inputs(2, chunk_size, old_att_len)

            eager_inputs = tuple(v.clone() for v in inputs)
            with torch.no_grad():
                eager_result = estimator.blocks_forward_chunk(
                    eager_inputs[0],
                    eager_inputs[1],
                    None,
                    eager_inputs[2],
                    eager_inputs[3],
                    eager_inputs[4],
                    eager_inputs[5],
                )

            wrapper.replay(*inputs)
            replay_inputs = tuple(v.clone() for v in inputs)
            graph_result, graph_cnn, graph_att = wrapper.replay(*replay_inputs)

            torch.testing.assert_close(graph_result, eager_result, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(graph_cnn, eager_inputs[4], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(graph_att, eager_inputs[5], rtol=1e-4, atol=1e-5)

    wrapper._flush()


def test_importing_wrapper_does_not_resolve_platform() -> None:
    """Importing this module must not build the OmniPlatform singleton.

    The NPU platform's ``__init__`` patches Code2Wav, which imports
    ``batched_token2wav`` -> ``cuda_graph_wrapper``. Resolving
    ``current_omni_platform`` at module scope re-enters
    ``platforms.__getattr__`` while the singleton is still under construction
    and the import graph deadlocks. The module reaches for nothing on
    ``vllm_omni.platforms`` today; this keeps it that way.
    """
    import importlib

    import vllm_omni.platforms as platforms_module

    name = "vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper"
    importlib.import_module(name)
    saved = platforms_module._current_omni_platform
    try:
        del sys.modules[name]
        platforms_module._current_omni_platform = None
        importlib.import_module(name)
        assert platforms_module._current_omni_platform is None
    finally:
        platforms_module._current_omni_platform = saved


def _cfm_mock_wrapper(monkeypatch: pytest.MonkeyPatch, *, max_graphs: int = 1) -> CFMGraphWrapper:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = object.__new__(CFMGraphWrapper)
    wrapper.max_graphs = max_graphs
    wrapper.enabled = True
    wrapper.graph_fn = Mock(return_value=torch.tensor([42.0]))
    wrapper.device = torch.device("cuda")
    wrapper._cache = {}
    wrapper._unsupported = set()
    wrapper._stats = {"calls": 0, "hits": 0, "captures": 0, "flushes": 0, "eager": 0}
    wrapper._capture = Mock(return_value=None)
    return wrapper


def test_cfm_unseen_shape_is_lazily_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _cfm_mock_wrapper(monkeypatch)

    inputs = _cfm_inputs(2, 10, 0)
    wrapper.replay(*inputs)

    wrapper._capture.assert_called_once()
    wrapper.graph_fn.assert_called_once()


def test_cfm_returning_no_entry_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture that yields no entry must still serve the request eagerly.

    ``_capture`` is mocked here, so this says nothing about ``_disable``; see
    ``test_cfm_capture_failure_disables_further_capture`` for that.
    """
    wrapper = _cfm_mock_wrapper(monkeypatch)

    inputs = _cfm_inputs(2, 10, 0)
    result = wrapper.replay(*inputs)

    wrapper.graph_fn.assert_called_once()
    assert result[0] is wrapper.graph_fn.return_value
    assert wrapper._stats["eager"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cfm_capture_failure_disables_further_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real capture failure must stop the wrapper capturing for good.

    A failed capture can leave the capture stream current and the allocator
    still routing into the graph pool, so the next shape must not try again.
    """
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _MiniDiT().eval().cuda()
    wrapper = CFMGraphWrapper(graph_fn=estimator.blocks_forward_chunk, max_graphs=4)

    def _explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("capture failed")

    monkeypatch.setattr(torch.cuda, "graph", _explode)

    with torch.inference_mode():
        wrapper.replay(*_cfm_inputs(2, 10, 0))
        assert wrapper.enabled is False
        assert wrapper._cache == {}

        captures_after_failure = wrapper._stats["captures"]
        wrapper.replay(*_cfm_inputs(2, 12, 0))

    assert wrapper._stats["captures"] == captures_after_failure
    assert wrapper._stats["eager"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cfm_unsupported_dtype_eagers_only_that_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key that cannot round-trip is a property of one shape, not of the GPU.

    It must not disable the wrapper or retire the generation the way a capture
    failure does.
    """
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _MiniDiT().eval().cuda()
    wrapper = CFMGraphWrapper(graph_fn=estimator.blocks_forward_chunk, max_graphs=4)

    unbuildable_width = 10
    real_tensors_from_key = wrapper_module._tensors_from_key

    def _reject_one_shape(key: tuple) -> tuple:
        if key[1][0][2] == unbuildable_width:
            raise KeyError("torch.int64")
        return real_tensors_from_key(key)

    monkeypatch.setattr(wrapper_module, "_tensors_from_key", _reject_one_shape)

    with torch.inference_mode():
        wrapper.replay(*_cfm_inputs(2, unbuildable_width, 0))
        assert wrapper.enabled is True
        assert wrapper._stats["captures"] == 0

        # a capturable shape still gets a graph
        wrapper.replay(*_cfm_inputs(2, 12, 0))
        assert wrapper._stats["captures"] == 1

        # and the rejected shape stays eager without retrying the capture
        wrapper.replay(*_cfm_inputs(2, unbuildable_width, 0))
        assert wrapper._stats["captures"] == 1
        assert wrapper._stats["eager"] == 2

    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hift_replay_survives_a_cfm_generation_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retiring a CFM generation must not disturb the vocoder.

    Both wrappers capture into ``get_global_graph_pool()``, so the HiFT graphs
    are exactly the live graphs a CFM flush could strand.
    """
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    hift = _small_hift()
    token2wav = SimpleNamespace(
        hift=hift,
        flow=SimpleNamespace(
            encoder=SimpleNamespace(pre_lookahead_layer=SimpleNamespace(pre_lookahead_len=3)),
            token_mel_ratio=2,
        ),
        mel_cache_len=2,
        source_cache_len=960,
    )
    hift_wrapper = HiFTGraphWrapper(
        token2wav,
        connector_config={"codec_chunk_frames": 2, "codec_left_context_frames": 3},
        capture_batch_sizes=[1],
    )
    hift_wrapper.capture()

    estimator = _MiniDiT().eval().cuda()
    cfm = CFMGraphWrapper(graph_fn=estimator.blocks_forward_chunk, max_graphs=2)

    speech_feat = torch.randn(1, 80, 4, device="cuda")
    cache_source = torch.zeros(1, 1, 0, device="cuda")

    with torch.inference_mode():
        expected_speech, expected_source = hift.inference(speech_feat, cache_source)

        for width in (10, 12, 14, 16):
            cfm.replay(*_cfm_inputs(2, width, 0))
        assert cfm._stats["flushes"] >= 1, "cache never overflowed; the test proves nothing"

        actual_speech, actual_source = hift_wrapper.replay(speech_feat, cache_source)

    torch.testing.assert_close(actual_speech, expected_speech, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual_source, expected_source, rtol=1e-4, atol=1e-5)

    cfm._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cfm_cache_flushes_whole_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full cache is retired all at once, never one graph at a time.

    Destroying a single graph while its peers stay live strands them, so the
    cache must never hold a graph that outlived one of its generation-mates.
    """
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _MiniDiT().eval().cuda()
    wrapper = CFMGraphWrapper(graph_fn=estimator.blocks_forward_chunk, max_graphs=2)

    with torch.inference_mode():
        inputs_a = _cfm_inputs(2, 10, 0)
        wrapper.replay(*inputs_a)
        assert len(wrapper._cache) == 1

        inputs_b = _cfm_inputs(2, 12, 0)
        wrapper.replay(*inputs_b)
        assert len(wrapper._cache) == 2

        # A hit must not grow the cache or trigger a flush.
        wrapper.replay(*inputs_a)
        assert len(wrapper._cache) == 2
        assert wrapper._stats["flushes"] == 0
        assert wrapper._stats["hits"] == 1

        # The third distinct shape flushes the generation, then captures alone.
        inputs_c = _cfm_inputs(2, 14, 0)
        wrapper.replay(*inputs_c)
        assert wrapper._stats["flushes"] == 1
        assert len(wrapper._cache) == 1

        # The flushed shapes are gone, so they capture again rather than hit.
        hits_before = wrapper._stats["hits"]
        wrapper.replay(*inputs_a)
        assert wrapper._stats["hits"] == hits_before
        wrapper.replay(*inputs_a)
        assert wrapper._stats["hits"] == hits_before + 1

    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cfm_none_cache_parity_between_graph_and_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that None→zeros (graph path) matches [None]*depth (eager path).

    setup_batch calls _decode_cfm with cnn_cache=None on every request's
    prompt pass. The graph path replaces None with zeros; the eager path
    passes [None]*depth. Both must produce identical outputs.
    """
    torch.accelerator.synchronize()
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _MiniDiT().eval().cuda()
    wrapper = CFMGraphWrapper(graph_fn=estimator.blocks_forward_chunk, max_graphs=32)

    depth = len(estimator.blocks)
    hidden = 8
    batch_size = 2
    chunk_size = 20

    estimator_input = torch.randn(batch_size, 16, chunk_size, device="cuda")
    time_emb = torch.randn(batch_size, 1, hidden, device="cuda")
    cnn_out = torch.empty(depth, batch_size, hidden, 2, device="cuda")
    att_out = torch.empty(depth, batch_size, 1, chunk_size, hidden, device="cuda")

    # Eager path: cnn_cache=[None]*depth, att_cache=[None]*depth
    eager_input = estimator_input.clone()
    eager_time = time_emb.clone()
    eager_cnn_out = torch.empty_like(cnn_out)
    eager_att_out = torch.empty_like(att_out)
    with torch.no_grad():
        eager_result = estimator.blocks_forward_chunk(
            eager_input,
            eager_time,
            None,
            [None] * depth,
            [None] * depth,
            eager_cnn_out,
            eager_att_out,
        )

    # Graph path: cnn_cache=zeros, att_cache=zero-length (simulating _estimator_step's None→zeros)
    zero_cnn = torch.zeros_like(cnn_out)
    zero_att = estimator_input.new_zeros(att_out.shape[:3] + (0,) + att_out.shape[4:])
    wrapper.replay(estimator_input, time_emb, zero_cnn, zero_att, cnn_out, att_out)
    graph_input = estimator_input.clone()
    graph_time = time_emb.clone()
    graph_cnn_out = torch.empty_like(cnn_out)
    graph_att_out = torch.empty_like(att_out)
    graph_zero_cnn = torch.zeros_like(cnn_out)
    graph_zero_att = estimator_input.new_zeros(att_out.shape[:3] + (0,) + att_out.shape[4:])
    graph_result, graph_cnn, graph_att = wrapper.replay(
        graph_input,
        graph_time,
        graph_zero_cnn,
        graph_zero_att,
        graph_cnn_out,
        graph_att_out,
    )

    torch.testing.assert_close(graph_result, eager_result, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_cnn, eager_cnn_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_att, eager_att_out, rtol=1e-4, atol=1e-5)

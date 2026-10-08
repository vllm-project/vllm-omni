# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.platforms import current_platform

import vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper as wrapper_module
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    HiFTGenerator,
)
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
    CFMGraphWrapper,
    HiFTGraphWrapper,
    WholeEulerCFMGraphWrapper,
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
    wrapper._legit_shapes = {(7, 0), (9, 0)}
    wrapper.graph = {}
    wrapper.static_speech_inputs = {}
    wrapper.static_cache_source_inputs = {}
    wrapper.static_magnitude_outputs = {}
    wrapper.static_phase_outputs = {}
    wrapper.static_cache_source_outputs = {}
    wrapper.lazy_graph_count = 0
    wrapper.max_lazy_graphs = 1
    wrapper.max_serial_batch = 8
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


def test_multibatch_serial_replay_with_batch1_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _fake_wrapper(monkeypatch)
    # 3-batch input with only batch size 1 supported
    speech_feat = torch.randn(3, 80, 7)
    cache_source = torch.zeros(3, 1, 0)
    speech, source = wrapper.replay(speech_feat, cache_source)

    # Lazily captured once for shape (1, 7, 0)
    wrapper._capture.assert_called_once_with(1, 7, 0)
    assert speech.shape == (3, 1, 7)
    assert source.shape == (3, 1, 7)
    wrapper.decode_fn.assert_not_called()


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
        assert cnn_cache_buffer is not None
        assert att_cache_buffer is not None
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
            if att_b is not None:
                att_cache_buffer[b_idx][:, :, dt:, :] = att_b
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
    # The fake estimator only fills ``att_out[:, :, :, :chunk_size]``. The tail
    # is the previous cache slot and is not written. ``empty`` leaves it
    # uninitialized, so eager and replay compare different garbage and CI
    # fails when that garbage is NaN.
    cnn_out = torch.zeros(depth, batch_size, hidden, 2, device=device)
    att_out = torch.zeros(depth, batch_size, 1, old_att_len + chunk_size, hidden, device=device)
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
            # A cache output must overwrite every row, including the old tail.
            inputs[4].fill_(float("nan"))
            inputs[5].fill_(float("nan"))

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
            assert torch.isfinite(graph_att).all()
            torch.testing.assert_close(graph_att[:, :, :, chunk_size:, :], inputs[3])

    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cfm_capture_keeps_the_real_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Static buffers must be built from the real inputs, mask included.

    Capturing from zero-filled placeholders would bake the "nothing is masked"
    branch into the graph, while replay copies the real mask into those very
    buffers -- so the captured branch has to come from the real value.
    """
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _MiniDiT().eval().cuda()
    wrapper = CFMGraphWrapper(graph_fn=estimator.blocks_forward_chunk, max_graphs=4)

    inputs = _cfm_inputs(2, 12, 0)
    frames = int(inputs[0].shape[2])
    mask = torch.ones(2, frames, frames, dtype=torch.bool, device="cuda")
    mask[:, :, -2:] = False  # two padded keys are masked out

    with torch.inference_mode():
        wrapper.replay(*inputs, mask)
        assert wrapper._stats["captures"] == 1

    static_inputs, _, _ = next(iter(wrapper._cache.values()))
    static_mask = static_inputs[6]
    assert static_mask is not None
    assert torch.equal(static_mask, mask)
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
    wrapper.max_serial_batch = 8
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
    failure does. `_tensors_from_key` is the fallback used when no real inputs
    are available, so the probe drives `_capture` without them.
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
        key = ("estimator_step",) + tuple(
            wrapper_module._tensor_signature(t) for t in _cfm_inputs(2, unbuildable_width, 0)
        )
        assert wrapper._capture(key, None) is None
        assert wrapper.enabled is True
        assert wrapper._stats["captures"] == 0

        # a capturable shape still gets a graph
        wrapper.replay(*_cfm_inputs(2, 12, 0))
        assert wrapper._stats["captures"] == 1

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


# ---------------------------------------------------------------------------
# WholeEulerCFMGraphWrapper tests
# ---------------------------------------------------------------------------


class _WholeEulerDiT(nn.Module):
    """DiT module with t_embedder and buffer dimensions for Whole-Euler testing."""

    def __init__(self, x_dim: int = 4, hidden: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.in_proj = nn.Linear(x_dim * 4, hidden)
        self.blocks = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.final_layer = nn.Linear(hidden, x_dim)
        for b in self.blocks:
            b.conv = SimpleNamespace(
                in_channels=4,
                out_channels=4,
                block=[None, SimpleNamespace(causal_padding=[2])],
            )
            b.attn = SimpleNamespace(num_heads=2, head_dim=4)

    def t_embedder(self, t: torch.Tensor) -> torch.Tensor:
        return t[:, None].expand(-1, 8)

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
        assert cnn_cache_buffer is not None
        assert att_cache_buffer is not None
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        for b_idx in range(len(self.blocks)):
            cnn_b = cnn_cache[b_idx] if cnn_cache is not None else None
            if cnn_b is not None:
                x[:, : cnn_b.shape[2], :] += cnn_b.transpose(1, 2)
            att_b = att_cache[b_idx] if att_cache is not None else None
            old_len = 0
            if att_b is not None and att_b.shape[2] > 0:
                old_len = att_b.shape[2]
                x += att_b.sum(dim=(1, 2), keepdim=False).unsqueeze(1)
                att_cache_buffer[b_idx][:, :, :old_len, :] = att_b
            x = self.blocks[b_idx](x)
            x = x + t
            cnn_cache_buffer[b_idx] = x[:, -2:, :].transpose(1, 2).contiguous()
            dt = x.shape[1]
            att_cache_buffer[b_idx][:, :, old_len : old_len + dt, :] = x.unsqueeze(1)
        x = self.final_layer(x)
        return x.transpose(1, 2)


def _eager_solve_euler(
    estimator: nn.Module,
    x: torch.Tensor,
    mu_cfg: torch.Tensor,
    speakers_cfg: torch.Tensor,
    cond_cfg: torch.Tensor,
    cnn_cache: torch.Tensor | None,
    att_cache: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    timeline: torch.Tensor,
    *,
    n_timesteps: int = 10,
    inference_cfg_rate: float = 0.7,
    mel_frames: int | None = None,
    pad_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur_x = x.clone()
    batch_size = int(x.shape[0])
    width = int(mu_cfg.shape[2])
    if mel_frames is None:
        mel_frames = width - pad_frames
    speaker_features = speakers_cfg.unsqueeze(-1).expand(-1, -1, width)
    if pad_frames > 0:
        speaker_features = speaker_features.clone()
        speaker_features[..., mel_frames:] = 0.0

    depth = len(estimator.blocks)
    block0 = estimator.blocks[0]
    cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
    cnn_width = int(block0.conv.block[1].causal_padding[0])
    heads = int(block0.attn.num_heads)
    att_width = int(block0.attn.head_dim * 2)
    offset = int(att_cache.shape[4]) if att_cache is not None else 0

    next_cnns = []
    next_atts = []

    for step in range(n_timesteps):
        t_val = timeline[step].expand(2 * batch_size)
        dt = timeline[step + 1] - timeline[step]
        time_embedding = estimator.t_embedder(t_val).unsqueeze(1)
        x_cfg = torch.cat((cur_x, cur_x), dim=0)
        estimator_input = torch.cat((x_cfg, mu_cfg, speaker_features, cond_cfg), dim=1)

        c_out = torch.empty(depth, 2 * batch_size, cnn_channels, cnn_width, device=x.device, dtype=x.dtype)
        a_out = torch.empty(depth, 2 * batch_size, heads, offset + width, att_width, device=x.device, dtype=x.dtype)
        old_c = cnn_cache[step] if cnn_cache is not None else [None] * depth
        old_a = att_cache[step] if att_cache is not None else [None] * depth

        est = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            attn_mask,
            old_c,
            old_a,
            c_out,
            a_out,
        )

        if pad_frames > 0:
            wrapper_module._zero_padded_cnn_cache(c_out, estimator, pad_frames)
            a_out[..., mel_frames : mel_frames + pad_frames, :] = 0.0

        conditional, unconditional = est.split(batch_size, dim=0)
        velocity = (1.0 + inference_cfg_rate) * conditional - inference_cfg_rate * unconditional
        cur_x = cur_x + dt * velocity
        if pad_frames > 0:
            cur_x[..., mel_frames:] = 0.0

        next_cnns.append(c_out)
        next_atts.append(a_out)

    return cur_x[:, :, :mel_frames], torch.stack(next_cnns), torch.stack(next_atts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_whole_euler_graph_replay_matches_eager_for_uncached_and_cached_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=32)

    batch_size = 2
    chunk_size = 10
    x = torch.randn(batch_size, 4, chunk_size, device="cuda")
    mu = torch.randn(batch_size, 4, chunk_size, device="cuda")
    speakers = torch.randn(batch_size, 4, device="cuda")
    cond = torch.randn(batch_size, 4, chunk_size, device="cuda")

    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
    speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)

    # 1. Uncached case
    eager_x, eager_cnn, eager_att = _eager_solve_euler(
        estimator, x, mu_cfg, speakers_cfg, cond_cfg, None, None, None, wrapper.timeline
    )
    graph_res = wrapper.replay(
        x=x.clone(),
        mu_cfg=mu_cfg.clone(),
        speakers_cfg=speakers_cfg.clone(),
        cond_cfg=cond_cfg.clone(),
        cnn_cache=None,
        att_cache=None,
        attn_mask=None,
    )
    assert graph_res is not None
    graph_x, graph_cnn, graph_att = graph_res

    torch.testing.assert_close(graph_x, eager_x, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_cnn, eager_cnn, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_att, eager_att, rtol=1e-4, atol=1e-5)

    # 2. Cached case (carry forward previous chunk outputs)
    x2 = torch.randn(batch_size, 4, chunk_size, device="cuda")
    eager_x2, eager_cnn2, eager_att2 = _eager_solve_euler(
        estimator, x2, mu_cfg, speakers_cfg, cond_cfg, eager_cnn, eager_att, None, wrapper.timeline
    )
    graph_res2 = wrapper.replay(
        x=x2.clone(),
        mu_cfg=mu_cfg.clone(),
        speakers_cfg=speakers_cfg.clone(),
        cond_cfg=cond_cfg.clone(),
        cnn_cache=graph_cnn,
        att_cache=graph_att,
        attn_mask=None,
    )
    assert graph_res2 is not None
    graph_x2, graph_cnn2, graph_att2 = graph_res2

    torch.testing.assert_close(graph_x2, eager_x2, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_cnn2, eager_cnn2, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_att2, eager_att2, rtol=1e-4, atol=1e-5)

    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_whole_euler_graph_with_padding_matches_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(42)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=16)

    batch_size = 2
    mel_frames = 8
    pad_frames = 2
    chunk_size = mel_frames + pad_frames

    x = torch.randn(batch_size, 4, chunk_size, device="cuda")
    x[:, :, mel_frames:] = 0.0
    mu = torch.randn(batch_size, 4, chunk_size, device="cuda")
    speakers = torch.randn(batch_size, 4, device="cuda")
    cond = torch.randn(batch_size, 4, chunk_size, device="cuda")

    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
    speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)

    attn_mask = torch.ones(2 * batch_size, chunk_size, chunk_size, dtype=torch.bool, device="cuda")
    attn_mask[:, :, mel_frames : mel_frames + pad_frames] = False

    eager_x, eager_cnn, eager_att = _eager_solve_euler(
        estimator,
        x,
        mu_cfg,
        speakers_cfg,
        cond_cfg,
        None,
        None,
        attn_mask,
        wrapper.timeline,
        mel_frames=mel_frames,
        pad_frames=pad_frames,
    )

    graph_res = wrapper.replay(
        x=x.clone(),
        mu_cfg=mu_cfg.clone(),
        speakers_cfg=speakers_cfg.clone(),
        cond_cfg=cond_cfg.clone(),
        cnn_cache=None,
        att_cache=None,
        attn_mask=attn_mask.clone(),
        mel_frames=mel_frames,
        pad_frames=pad_frames,
    )
    assert graph_res is not None
    graph_x, graph_cnn, graph_att = graph_res

    assert graph_x.shape[2] == mel_frames
    assert eager_x.shape[2] == mel_frames
    torch.testing.assert_close(graph_x, eager_x, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_cnn, eager_cnn, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_att, eager_att, rtol=1e-4, atol=1e-5)

    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_whole_euler_graph_boundary_enforces_budget_and_falls_back_to_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: torch.cuda.graph_pool_handle())

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=2)

    def _call(w: int):
        x = torch.randn(1, 4, w, device="cuda")
        mu_cfg = torch.randn(2, 4, w, device="cuda")
        spk_cfg = torch.randn(2, 4, device="cuda")
        cond_cfg = torch.randn(2, 4, w, device="cuda")
        return wrapper.replay(
            x=x,
            mu_cfg=mu_cfg,
            speakers_cfg=spk_cfg,
            cond_cfg=cond_cfg,
            cnn_cache=None,
            att_cache=None,
        )

    res10 = _call(10)
    assert res10 is not None
    assert len(wrapper._cache) == 1
    assert wrapper._stats["captures"] == 1

    # Same shape hits
    _call(10)
    assert wrapper._stats["hits"] == 1

    # Second distinct shape fills cache to max_graphs=2
    res12 = _call(12)
    assert res12 is not None
    assert len(wrapper._cache) == 2
    assert wrapper._stats["captures"] == 2
    assert wrapper._stats["flushes"] == 0

    # A third distinct shape exceeds max_graphs=2: falls back to eager (replay returns None) without expanding budget
    res14 = _call(14)
    assert res14 is None
    assert wrapper._stats["flushes"] == 0
    assert len(wrapper._cache) == 2
    assert wrapper.max_graphs == 2
    assert wrapper._stats["captures"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_whole_euler_precapture_enforces_budget_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: torch.cuda.graph_pool_handle())

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    # Configure max_graphs=1 with query_bucket_frames=16
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=1,
        max_graph_batch=2,
        micro_batch_size=2,
        query_bucket_frames=16,
    )

    count = wrapper.precapture(
        offsets=[0, 16, 32],
        steady=32,
        channels=4,
        spk_dim=4,
    )
    # Must capture at most 1 graph and strictly respect max_graphs=1
    assert count <= 1
    assert wrapper.max_graphs == 1
    assert len(wrapper._cache) <= 1

    # An uncached shape falls back to eager (None) because max_graphs=1 is already exhausted
    x_uncached = torch.randn(1, 4, 10, device="cuda")
    mu_uncached = torch.randn(2, 4, 10, device="cuda")
    spk_uncached = torch.randn(2, 4, device="cuda")
    cond_uncached = torch.randn(2, 4, 10, device="cuda")
    res_uncached = wrapper.replay(
        x=x_uncached,
        mu_cfg=mu_uncached,
        speakers_cfg=spk_uncached,
        cond_cfg=cond_uncached,
        cnn_cache=None,
        att_cache=None,
    )
    assert res_uncached is None  # Eager fallback
    assert wrapper.max_graphs == 1
    assert len(wrapper._cache) == 1
    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_whole_euler_slot_entry_enforces_budget_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: torch.cuda.graph_pool_handle())
    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=2,
        att_slots=2,
        ragged_body=Mock(),
        modulation_fn=lambda t: t,
        query_bucket_frames=16,
    )
    # `_record` returns (statics, x, cnn, att, graph); `_retire` calls entry[4].reset().
    monkeypatch.setattr(wrapper, "_record", lambda key, *args, **kwargs: (Mock(), Mock(), Mock(), Mock(), Mock()))
    pool = wrapper._ensure_slot_pool((0, 4))
    assert pool is not None

    fill = wrapper._precapture_fill
    # Capture 1 slot graph
    entry1 = wrapper._slot_entry(graph_batch=1, query_cap=10, channels=4, spk_dim=4, fill=fill)
    assert entry1 is not None
    assert len(wrapper._slot_graphs) == 1
    assert wrapper.stats_snapshot()["cache_size"] == 1

    # Capture 1 arena graph
    x = torch.empty((1, 4, 1), device="cuda", dtype=torch.float32)
    entry2 = wrapper._entry(graph_batch=1, query_cap=12, offset=0, x=x, spk_dim=4, fill=fill)
    assert entry2 is not None
    assert len(wrapper._cache) == 1
    assert wrapper.stats_snapshot()["cache_size"] == 2

    # A 3rd graph (slot or arena) exceeds max_graphs=2: falls back to eager (returns None)
    entry3_slot = wrapper._slot_entry(graph_batch=1, query_cap=14, channels=4, spk_dim=4, fill=fill)
    assert entry3_slot is None

    entry3_arena = wrapper._entry(graph_batch=1, query_cap=16, offset=0, x=x, spk_dim=4, fill=fill)
    assert entry3_arena is None

    assert wrapper.stats_snapshot()["cache_size"] == 2
    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_whole_euler_arena_cleaned_when_first_capture_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=2)

    # Force failure during capture
    def mock_fail(*args, **kwargs):
        raise RuntimeError("Simulated first capture failure")

    monkeypatch.setattr(wrapper, "_run_euler_loop", mock_fail)

    x = torch.randn(1, 4, 10, device="cuda")
    mu_cfg = torch.randn(2, 4, 10, device="cuda")
    spk_cfg = torch.randn(2, 4, device="cuda")
    cond_cfg = torch.randn(2, 4, 10, device="cuda")

    res = wrapper.replay(
        x=x,
        mu_cfg=mu_cfg,
        speakers_cfg=spk_cfg,
        cond_cfg=cond_cfg,
        cnn_cache=None,
        att_cache=None,
    )
    assert res is None
    assert wrapper.enabled is False
    assert len(wrapper._cache) == 0
    # Arena must be cleared even when cache has zero entries
    assert not wrapper.arena._buffers


@pytest.mark.parametrize(
    ("micro", "pad_max_rows", "plans"),
    [
        (4, 1, {1: [(1, 1)], 2: [(2, 2)], 3: [(4, 3)], 6: [(4, 4), (2, 2)], 7: [(4, 4), (4, 3)]}),
        (4, 0, {3: [(2, 2), (1, 1)], 7: [(4, 4), (2, 2), (1, 1)]}),
        (
            8,
            2,
            {
                5: [(4, 4), (1, 1)],
                6: [(8, 6)],
                12: [(8, 8), (4, 4)],
                13: [(8, 8), (4, 4), (1, 1)],
                16: [(8, 8), (8, 8)],
            },
        ),
    ],
)
def test_whole_euler_plan_groups_uses_power_of_two_graph_batches(
    micro: int,
    pad_max_rows: int,
    plans: dict[int, list[tuple[int, int]]],
) -> None:
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=_WholeEulerDiT(),
        n_timesteps=10,
        max_graphs=8,
        max_graph_batch=16,
        micro_batch_size=micro,
        pad_max_rows=pad_max_rows,
    )
    for batch_size, plan in plans.items():
        assert wrapper._plan_groups(batch_size) == plan
        assert sum(rows for _, rows in plan) == batch_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
@pytest.mark.parametrize(("pad_max_rows", "graph_batches"), [(1, {4}), (0, {1, 2})])
def test_whole_euler_multibatch_remainder_replay(
    monkeypatch: pytest.MonkeyPatch,
    pad_max_rows: int,
    graph_batches: set[int],
) -> None:
    """B=3 runs one padded B=4 replay by default, or B=2 + B=1 replays; both match eager."""
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=4, pad_max_rows=pad_max_rows)

    batch_size = 3
    w = 8
    x = torch.randn(batch_size, 4, w, device="cuda")
    mu_cfg = torch.randn(2 * batch_size, 4, w, device="cuda")
    spk_cfg = torch.randn(2 * batch_size, 4, device="cuda")
    cond_cfg = torch.randn(2 * batch_size, 4, w, device="cuda")

    out_mel, out_cnn, out_att = wrapper.replay(
        x=x.clone(),
        mu_cfg=mu_cfg,
        speakers_cfg=spk_cfg,
        cond_cfg=cond_cfg,
        cnn_cache=None,
        att_cache=None,
    )
    assert out_mel.shape == (batch_size, 4, w)
    assert out_cnn.shape[2] == 2 * batch_size
    assert out_att.shape[2] == 2 * batch_size
    assert {key[1] for key in wrapper._cache} == graph_batches

    eager_x, eager_cnn, eager_att = _eager_solve_euler(
        estimator, x, mu_cfg, spk_cfg, cond_cfg, None, None, None, wrapper.timeline
    )
    torch.testing.assert_close(out_mel, eager_x, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out_cnn, eager_cnn, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out_att, eager_att, rtol=1e-4, atol=1e-5)

    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_hierarchical_microbatch_b4_and_b8(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=8, max_graph_batch=8)

    w = 8
    # 1. Native B=4 capture
    batch_size = 4
    x4 = torch.randn(batch_size, 4, w, device="cuda")
    mu4 = torch.randn(2 * batch_size, 4, w, device="cuda")
    spk4 = torch.randn(2 * batch_size, 4, device="cuda")
    cond4 = torch.randn(2 * batch_size, 4, w, device="cuda")

    out_mel4, out_cnn4, out_att4 = wrapper.replay(
        x=x4,
        mu_cfg=mu4,
        speakers_cfg=spk4,
        cond_cfg=cond4,
        cnn_cache=None,
        att_cache=None,
    )
    assert out_mel4 is not None
    assert out_mel4.shape == (4, 4, w)
    assert out_cnn4.shape[2] == 8
    assert out_att4.shape[2] == 8
    assert any(key[1] == 4 for key in wrapper._cache)

    # 2. B=8 partitions into [4, 4], reusing the native B=4 graph
    batch_size = 8
    x8 = torch.randn(batch_size, 4, w, device="cuda")
    mu8 = torch.randn(2 * batch_size, 4, w, device="cuda")
    spk8 = torch.randn(2 * batch_size, 4, device="cuda")
    cond8 = torch.randn(2 * batch_size, 4, w, device="cuda")

    out_mel8, out_cnn8, out_att8 = wrapper.replay(
        x=x8,
        mu_cfg=mu8,
        speakers_cfg=spk8,
        cond_cfg=cond8,
        cnn_cache=None,
        att_cache=None,
    )
    assert out_mel8 is not None
    assert out_mel8.shape == (8, 4, w)
    assert out_cnn8.shape[2] == 16
    assert out_att8.shape[2] == 16

    # 3. B=5 partitions into [4, 1], capturing B=1 graph as well
    batch_size = 5
    x5 = torch.randn(batch_size, 4, w, device="cuda")
    mu5 = torch.randn(2 * batch_size, 4, w, device="cuda")
    spk5 = torch.randn(2 * batch_size, 4, device="cuda")
    cond5 = torch.randn(2 * batch_size, 4, w, device="cuda")

    out_mel5, out_cnn5, out_att5 = wrapper.replay(
        x=x5,
        mu_cfg=mu5,
        speakers_cfg=spk5,
        cond_cfg=cond5,
        cnn_cache=None,
        att_cache=None,
    )
    assert out_mel5 is not None
    assert out_mel5.shape == (5, 4, w)
    assert out_cnn5.shape[2] == 10
    assert out_att5.shape[2] == 10
    cached_batches = {key[1] for key in wrapper._cache}
    assert 4 in cached_batches
    assert 1 in cached_batches

    # 4. B=16 partitions into [4, 4, 4, 4] with max_graph_batch=16
    wrapper_16 = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=8, max_graph_batch=16)
    batch_size = 16
    x16 = torch.randn(batch_size, 4, w, device="cuda")
    mu16 = torch.randn(2 * batch_size, 4, w, device="cuda")
    spk16 = torch.randn(2 * batch_size, 4, device="cuda")
    cond16 = torch.randn(2 * batch_size, 4, w, device="cuda")

    out_mel16, out_cnn16, out_att16 = wrapper_16.replay(
        x=x16,
        mu_cfg=mu16,
        speakers_cfg=spk16,
        cond_cfg=cond16,
        cnn_cache=None,
        att_cache=None,
    )
    assert out_mel16 is not None
    assert out_mel16.shape == (16, 4, w)
    assert out_cnn16.shape[2] == 32
    assert out_att16.shape[2] == 32

    # B > 16 returns None (eager fallback)
    x17 = torch.randn(17, 4, w, device="cuda")
    mu17 = torch.randn(34, 4, w, device="cuda")
    spk17 = torch.randn(34, 4, device="cuda")
    cond17 = torch.randn(34, 4, w, device="cuda")
    assert (
        wrapper_16.replay(
            x=x17,
            mu_cfg=mu17,
            speakers_cfg=spk17,
            cond_cfg=cond17,
            cnn_cache=None,
            att_cache=None,
        )
        is None
    )

    wrapper._flush()
    wrapper_16._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_same_bucket_different_padding_hits_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=4)

    batch_size = 1
    mel_width = 16

    # Call 1: mel_frames=12, pad_frames=4 -> captures graph 1
    x1 = torch.randn(batch_size, 4, mel_width, device="cuda")
    mu1 = torch.randn(2 * batch_size, 4, mel_width, device="cuda")
    spk1 = torch.randn(2 * batch_size, 4, device="cuda")
    cond1 = torch.randn(2 * batch_size, 4, mel_width, device="cuda")
    mask1 = torch.ones(2 * batch_size, mel_width, mel_width, dtype=torch.bool, device="cuda")
    mask1[:, :, 12:] = False

    res1 = wrapper.replay(
        x=x1,
        mu_cfg=mu1,
        speakers_cfg=spk1,
        cond_cfg=cond1,
        cnn_cache=None,
        att_cache=None,
        attn_mask=mask1,
        mel_frames=12,
        pad_frames=4,
    )
    assert res1 is not None
    assert res1[0].shape[-1] == 12
    assert wrapper._stats["captures"] == 1
    assert wrapper._stats["hits"] == 0

    # Call 2: mel_frames=10, pad_frames=6 (different unpadded length, same bucket mel_width)
    x2 = torch.randn(batch_size, 4, mel_width, device="cuda")
    mu2 = torch.randn(2 * batch_size, 4, mel_width, device="cuda")
    spk2 = torch.randn(2 * batch_size, 4, device="cuda")
    cond2 = torch.randn(2 * batch_size, 4, mel_width, device="cuda")
    mask2 = torch.ones(2 * batch_size, mel_width, mel_width, dtype=torch.bool, device="cuda")
    mask2[:, :, 10:] = False

    res2 = wrapper.replay(
        x=x2,
        mu_cfg=mu2,
        speakers_cfg=spk2,
        cond_cfg=cond2,
        cnn_cache=None,
        att_cache=None,
        attn_mask=mask2,
        mel_frames=10,
        pad_frames=6,
    )
    assert res2 is not None
    assert res2[0].shape[-1] == 10
    # Must be a cache hit!
    assert wrapper._stats["captures"] == 1
    assert wrapper._stats["hits"] == 1
    assert len(wrapper._cache) == 1

    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_mask_presence_does_not_split_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """#7416 fills static buffers every replay; mask presence is data, not a key."""
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=4)

    batch_size = 1
    w = 16
    x = torch.randn(batch_size, 4, w, device="cuda")
    mu = torch.randn(2 * batch_size, 4, w, device="cuda")
    spk = torch.randn(2 * batch_size, 4, device="cuda")
    cond = torch.randn(2 * batch_size, 4, w, device="cuda")
    mask = torch.ones(2 * batch_size, w, w, dtype=torch.bool, device="cuda")
    mask[:, :, 12:] = False

    assert (
        wrapper.replay(
            x=x,
            mu_cfg=mu,
            speakers_cfg=spk,
            cond_cfg=cond,
            cnn_cache=None,
            att_cache=None,
            attn_mask=None,
        )
        is not None
    )
    assert wrapper._stats["captures"] == 1

    assert (
        wrapper.replay(
            x=x,
            mu_cfg=mu,
            speakers_cfg=spk,
            cond_cfg=cond,
            cnn_cache=None,
            att_cache=None,
            attn_mask=mask,
        )
        is not None
    )
    assert wrapper._stats["captures"] == 1
    assert wrapper._stats["hits"] == 1
    assert len(wrapper._cache) == 1
    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_query_capture_bucket_reuses_decode_widths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same capture grid as _cfm_pad_frames, but 64 so 16/32/48/64 share one graph."""
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=4,
        query_bucket_frames=64,
    )

    def _call(width: int):
        x = torch.randn(1, 4, width, device="cuda")
        mu = torch.randn(2, 4, width, device="cuda")
        spk = torch.randn(2, 4, device="cuda")
        cond = torch.randn(2, 4, width, device="cuda")
        return wrapper.replay(
            x=x,
            mu_cfg=mu,
            speakers_cfg=spk,
            cond_cfg=cond,
            cnn_cache=None,
            att_cache=None,
        )

    # The wide offset-0 prompt solve comes first, as in serving; it reserves
    # the storage the decode widths then share.
    assert _call(304) is not None
    assert _call(16) is not None
    assert _call(48) is not None
    assert _call(64) is not None
    assert wrapper._stats["captures"] == 2
    assert wrapper._stats["hits"] == 2
    assert wrapper._stats["flushes"] == 0
    query_caps = {key[2] for key in wrapper._cache}
    assert query_caps == {64, 320}
    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_max_serial_batch_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    # Configure threshold = 2
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=4,
        max_serial_batch=2,
    )

    w = 8
    # Batch = 2 <= max_serial_batch: uses CUDA graph serial replay
    x_small = torch.randn(2, 4, w, device="cuda")
    mu_small = torch.randn(4, 4, w, device="cuda")
    spk_small = torch.randn(4, 4, device="cuda")
    cond_small = torch.randn(4, 4, w, device="cuda")

    res_small = wrapper.replay(
        x=x_small,
        mu_cfg=mu_small,
        speakers_cfg=spk_small,
        cond_cfg=cond_small,
        cnn_cache=None,
        att_cache=None,
    )
    assert res_small is not None
    assert wrapper._stats["captures"] == 1

    # Batch = 4 > max_serial_batch: returns None to fall back to batched eager
    x_large = torch.randn(4, 4, w, device="cuda")
    mu_large = torch.randn(8, 4, w, device="cuda")
    spk_large = torch.randn(8, 4, device="cuda")
    cond_large = torch.randn(8, 4, w, device="cuda")

    res_large = wrapper.replay(
        x=x_large,
        mu_cfg=mu_large,
        speakers_cfg=spk_large,
        cond_cfg=cond_large,
        cnn_cache=None,
        att_cache=None,
    )
    assert res_large is None
    # No new graph captures triggered for large batch
    assert wrapper._stats["captures"] == 1

    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_cfm_step_graph_captures_distinct_batch_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Step-level CFM keys by tensor signature (upstream #7416): a larger CFG batch is another shape."""
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _MiniDiT().eval().cuda()
    wrapper = CFMGraphWrapper(graph_fn=estimator.blocks_forward_chunk, max_graphs=4)

    wrapper.replay(*_cfm_inputs(batch_size=4, chunk_size=8, old_att_len=0))
    assert wrapper._stats["captures"] == 1
    assert wrapper._stats["eager"] == 0

    wrapper.replay(*_cfm_inputs(batch_size=8, chunk_size=8, old_att_len=0))
    assert wrapper._stats["captures"] == 2
    assert wrapper._stats["eager"] == 0

    wrapper._flush()


def test_zero_padded_cnn_cache_clears_every_timestep_of_a_whole_euler_cache() -> None:
    # Whole-Euler keeps every timestep's CNN cache as (n_timesteps, depth, 2B, C, W)
    # and hands the helper a block-first view of it.
    estimator = _WholeEulerDiT()
    width = int(estimator.blocks[0].conv.block[1].causal_padding[0])
    cache = torch.randn(10, len(estimator.blocks), 4, 8, width)
    expected = cache.clone()
    expected[..., width - 1 :] = 0.0

    wrapper_module._zero_padded_cnn_cache(cache.transpose(0, 1), estimator, 1)

    torch.testing.assert_close(cache, expected)


def test_euler_step_applies_cfg_guidance() -> None:
    torch.manual_seed(42)
    B, C, T = 2, 8, 16
    cur_x = torch.randn(B, C, T)
    estimate = torch.randn(2 * B, C, T)
    dt = 0.1
    cfg = 0.7

    cond, uncond = estimate.split(B, dim=0)
    v = (1.0 + cfg) * cond - cfg * uncond
    expected = cur_x + dt * v

    actual = wrapper_module._euler_step(cur_x.clone(), estimate, dt, cfg, B)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_execution_arena_buffer_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=4)

    w = 10
    x1 = torch.randn(1, 4, w, device="cuda")
    mu1 = torch.randn(2, 4, w, device="cuda")
    spk1 = torch.randn(2, 4, device="cuda")
    cond1 = torch.randn(2, 4, w, device="cuda")

    # First call captures shape 1 (offset=0); att_keep reserves both offsets.
    res1 = wrapper.replay(
        x=x1,
        mu_cfg=mu1,
        speakers_cfg=spk1,
        cond_cfg=cond1,
        cnn_cache=None,
        att_cache=None,
        att_keep=(w, w),
    )
    assert res1 is not None
    assert wrapper._stats["captures"] == 1
    arena = wrapper.arena
    shared = {key: buf for key, buf in arena._buffers.items() if key[0] in ("time_embeddings", "cnn_in", "cnn_out")}
    assert sorted(key[0] for key in shared) == ["cnn_in", "cnn_out", "time_embeddings"]

    # Second call with cache (offset=10)
    res2 = wrapper.replay(
        x=x1,
        mu_cfg=mu1,
        speakers_cfg=spk1,
        cond_cfg=cond1,
        cnn_cache=res1[1],
        att_cache=res1[2],
    )
    assert res2 is not None
    assert wrapper._stats["captures"] == 2

    # Verify that the shared buffers are the exact same tensor instances (no duplicate allocation)
    assert all(arena._buffers[key] is buf for key, buf in shared.items())

    wrapper._flush()
    assert not arena._buffers


def _whole_euler_chunk(batch_size: int, width: int) -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(batch_size, 4, width, device="cuda"),
        "mu_cfg": torch.randn(2 * batch_size, 4, width, device="cuda"),
        "speakers_cfg": torch.randn(2 * batch_size, 4, device="cuda"),
        "cond_cfg": torch.randn(2 * batch_size, 4, width, device="cuda"),
    }


def _split_cfg_rows(stacked: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [
        torch.cat((stacked[:, :, row : row + 1], stacked[:, :, batch_size + row : batch_size + row + 1]), dim=2)
        for row in range(batch_size)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_graphs_share_one_attention_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every offset and batch size reads and writes prefix views of one storage, not a copy each."""
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=8)

    w = 8
    cnn = att = None
    eager_cnn = eager_att = None
    for _ in range(3):
        chunk = _whole_euler_chunk(1, w)
        # att_keep sums to the third chunk's length: it reserves that much, and trims nothing.
        graph_x, cnn, att = wrapper.replay(**chunk, cnn_cache=cnn, att_cache=att, att_keep=(w, 2 * w))
        eager_x, eager_cnn, eager_att = _eager_solve_euler(
            estimator,
            chunk["x"],
            chunk["mu_cfg"],
            chunk["speakers_cfg"],
            chunk["cond_cfg"],
            eager_cnn,
            eager_att,
            None,
            wrapper.timeline,
        )
        torch.testing.assert_close(graph_x, eager_x, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(att, eager_att, rtol=1e-4, atol=1e-5)

    # A B=2 stream at a known offset maps onto the same storage.
    _, pair_cnn, pair_att = wrapper.replay(**_whole_euler_chunk(2, w), cnn_cache=None, att_cache=None)
    wrapper.replay(**_whole_euler_chunk(2, w), cnn_cache=pair_cnn, att_cache=pair_att)

    assert wrapper._stats["captures"] == 5
    assert {(key[1], key[3]) for key in wrapper._cache} == {(1, 0), (1, w), (1, 2 * w), (2, 0), (2, w)}
    storage = wrapper.arena._att
    assert storage is not None
    assert int(storage.shape[2]) == 2 * wrapper.micro_batch_size
    for statics, _final_x, _out_cnn, out_att, _graph in wrapper._cache.values():
        assert statics.att_cache.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
        assert out_att.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    wrapper._flush()
    assert wrapper.arena._att is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_storage_growth_retires_the_graphs_on_the_old_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture that outgrows the attention storage must not leave the old one alive beside it."""
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=8)

    w = 8
    first = _whole_euler_chunk(1, w)
    # att_keep reserves 2 * w frames: the next w-wide chunk fits, a 2w-wide one does not.
    _, cnn, att = wrapper.replay(**first, cnn_cache=None, att_cache=None, att_keep=(w, w))
    _, eager_cnn, eager_att = _eager_solve_euler(
        estimator,
        first["x"],
        first["mu_cfg"],
        first["speakers_cfg"],
        first["cond_cfg"],
        None,
        None,
        None,
        wrapper.timeline,
    )
    wrapper.replay(**_whole_euler_chunk(1, w), cnn_cache=cnn, att_cache=att)
    assert wrapper._stats["flushes"] == 0
    assert len(wrapper._cache) == 2

    wide = _whole_euler_chunk(1, 2 * w)
    graph_x, _, graph_att = wrapper.replay(**wide, cnn_cache=cnn, att_cache=att)

    assert wrapper._stats["flushes"] == 1
    assert len(wrapper._cache) == 1
    storage = wrapper.arena._att
    assert storage is not None and int(storage.shape[4]) >= 3 * w
    for statics, _final_x, _out_cnn, out_att, _graph in wrapper._cache.values():
        assert statics.att_cache.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
        assert out_att.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    eager_x, _, eager_wide_att = _eager_solve_euler(
        estimator,
        wide["x"],
        wide["mu_cfg"],
        wide["speakers_cfg"],
        wide["cond_cfg"],
        eager_cnn,
        eager_att,
        None,
        wrapper.timeline,
    )
    torch.testing.assert_close(graph_x, eager_x, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_att, eager_wide_att, rtol=1e-4, atol=1e-5)
    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_per_request_rows_match_stacked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-request cache in/out (with the streaming trim) equals the stacked path, row for row."""
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(0)
    estimator = _WholeEulerDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=estimator, n_timesteps=10, max_graphs=8, max_graph_batch=8)

    batch_size = 6  # one B=4 replay plus one B=2 replay
    w = 8
    _, cnn, att = wrapper.replay(**_whole_euler_chunk(batch_size, w), cnn_cache=None, att_cache=None)
    chunk = _whole_euler_chunk(batch_size, w)
    stacked_x, stacked_cnn, stacked_att = wrapper.replay(**chunk, cnn_cache=cnn, att_cache=att)
    keep = (4, 6)
    rows_x, rows_cnn, rows_att = wrapper.replay(
        **chunk,
        cnn_cache=cnn,
        att_cache=_split_cfg_rows(att, batch_size),
        att_keep=keep,
    )

    torch.testing.assert_close(rows_x, stacked_x)
    torch.testing.assert_close(rows_cnn, stacked_cnn)
    assert isinstance(rows_att, list) and len(rows_att) == batch_size
    kept = wrapper_module._att_keep_ranges(int(stacked_att.shape[4]), keep)
    assert sum(length for _, length in kept) == 10
    for row, expected in enumerate(_split_cfg_rows(stacked_att, batch_size)):
        trimmed = torch.cat([expected[..., start : start + length, :] for start, length in kept], dim=-2)
        torch.testing.assert_close(rows_att[row], trimmed)
    wrapper._flush()


class _RealCausalDiTBlock(nn.Module):
    """Causal convolution and self-attention block for multi-chunk numerical verification."""

    def __init__(self, channels: int = 4, hidden: int = 8, causal_padding: int = 2, num_heads: int = 2) -> None:
        super().__init__()
        self.causal_padding = causal_padding
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.conv1d = nn.Conv1d(hidden, hidden, kernel_size=causal_padding + 1, padding=0)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.conv = SimpleNamespace(
            in_channels=channels,
            out_channels=channels,
            block=[None, SimpleNamespace(causal_padding=[causal_padding])],
        )
        self.attn = SimpleNamespace(num_heads=num_heads, head_dim=self.head_dim)

    def forward_chunk(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        cnn_buf: torch.Tensor,
        att_buf: torch.Tensor,
    ) -> torch.Tensor:
        B, C, T = x.shape
        # 1. Causal Conv1D
        if cnn_cache is not None and cnn_cache.shape[-1] > 0:
            conv_in = torch.cat([cnn_cache, x], dim=-1)
        else:
            conv_in = F.pad(x, (self.causal_padding, 0))
        cnn_buf.copy_(conv_in[..., -self.causal_padding :])
        h_conv = self.conv1d(conv_in)

        # 2. Multi-head self-attention with attn_mask
        x_time = x.transpose(1, 2)
        Q = self.q_proj(x_time).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_time).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_time).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if att_cache is not None and att_cache.shape[2] > 0:
            past_k = att_cache[..., : self.head_dim]
            past_v = att_cache[..., self.head_dim :]
            K_all = torch.cat([past_k, K], dim=2)
            V_all = torch.cat([past_v, V], dim=2)
        else:
            K_all = K
            V_all = V

        att_buf.copy_(torch.cat([K_all, V_all], dim=-1))

        scores = torch.matmul(Q, K_all.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        probs = F.softmax(scores, dim=-1)
        h_attn = torch.matmul(probs, V_all).transpose(1, 2).contiguous().view(B, T, C).transpose(1, 2)
        h_attn = self.out_proj(h_attn.transpose(1, 2)).transpose(1, 2)

        x = x + self.norm1((x + h_attn).transpose(1, 2)).transpose(1, 2)
        x = x + self.norm2((x + h_conv).transpose(1, 2)).transpose(1, 2)
        return x


class _RealAttentionCausalConvDiT(nn.Module):
    """Estimator with real attention consuming attn_mask and causal convolutions for cross-chunk parity testing."""

    def __init__(self, x_dim: int = 4, hidden: int = 8, depth: int = 2, causal_pad: int = 2) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.in_proj = nn.Linear(x_dim * 4, hidden)
        self.blocks = nn.ModuleList(
            [_RealCausalDiTBlock(channels=x_dim, hidden=hidden, causal_padding=causal_pad) for _ in range(depth)]
        )
        self.final_layer = nn.Linear(hidden, x_dim)

    def t_embedder(self, t: torch.Tensor) -> torch.Tensor:
        return t[:, None].expand(-1, 8)

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
        assert cnn_cache_buffer is not None and att_cache_buffer is not None
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        t_feat = t.transpose(1, 2)
        for b_idx, block in enumerate(self.blocks):
            cnn_b = cnn_cache[b_idx] if cnn_cache is not None else None
            att_b = att_cache[b_idx] if att_cache is not None else None
            x = block.forward_chunk(x, mask, cnn_b, att_b, cnn_cache_buffer[b_idx], att_cache_buffer[b_idx])
            x = x + t_feat
        x = self.final_layer(x.transpose(1, 2)).transpose(1, 2)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_real_attention_causal_conv_parity_across_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(42)
    dit = _RealAttentionCausalConvDiT().eval().cuda()
    wrapper = WholeEulerCFMGraphWrapper(estimator=dit, n_timesteps=10, max_graphs=4)

    B = 1
    W = 16

    # Call 1 (bucket=16, mel_frames=12, pad_frames=4, offset=0)
    x1 = torch.randn(B, 4, W, device="cuda")
    mu1 = torch.randn(2 * B, 4, W, device="cuda")
    spk1 = torch.randn(2 * B, 4, device="cuda")
    cond1 = torch.randn(2 * B, 4, W, device="cuda")
    mask1 = torch.ones(2 * B, W, W, dtype=torch.bool, device="cuda")
    mask1[:, :, 12:] = False

    g_x1, g_cnn1, g_att1 = wrapper.replay(
        x=x1,
        mu_cfg=mu1,
        speakers_cfg=spk1,
        cond_cfg=cond1,
        cnn_cache=None,
        att_cache=None,
        attn_mask=mask1,
        mel_frames=12,
        pad_frames=4,
        # Reserves chunk 2's 12 + 16 frames up front, as serving does; trims nothing.
        att_keep=(12, 16),
    )
    e_x1, e_cnn1, e_att1 = _eager_solve_euler(
        dit,
        x1,
        mu1,
        spk1,
        cond1,
        None,
        None,
        mask1,
        wrapper.timeline,
        mel_frames=12,
        pad_frames=4,
    )

    torch.testing.assert_close(g_x1, e_x1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_cnn1, e_cnn1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_att1[..., :12, :], e_att1[..., :12, :], rtol=1e-5, atol=1e-5)

    # Chunk 2 consuming Chunk 1 caches (offset=12, mel_frames=16, pad_frames=0)
    x1_c2 = torch.randn(B, 4, W, device="cuda")
    mu1_c2 = torch.randn(2 * B, 4, W, device="cuda")
    spk1_c2 = torch.randn(2 * B, 4, device="cuda")
    cond1_c2 = torch.randn(2 * B, 4, W, device="cuda")
    mask1_c2 = torch.ones(2 * B, W, W + 12, dtype=torch.bool, device="cuda")

    g_x1_c2, g_cnn1_c2, g_att1_c2 = wrapper.replay(
        x=x1_c2,
        mu_cfg=mu1_c2,
        speakers_cfg=spk1_c2,
        cond_cfg=cond1_c2,
        cnn_cache=g_cnn1,
        att_cache=g_att1[..., :12, :],
        attn_mask=mask1_c2,
        mel_frames=16,
        pad_frames=0,
    )
    e_x1_c2, e_cnn1_c2, e_att1_c2 = _eager_solve_euler(
        dit,
        x1_c2,
        mu1_c2,
        spk1_c2,
        cond1_c2,
        e_cnn1,
        e_att1[..., :12, :],
        mask1_c2,
        wrapper.timeline,
        mel_frames=16,
        pad_frames=0,
    )

    torch.testing.assert_close(g_x1_c2, e_x1_c2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_cnn1_c2, e_cnn1_c2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_att1_c2, e_att1_c2, rtol=1e-5, atol=1e-5)

    # Call 2 on same bucket (bucket=16, mel_frames=15, pad_frames=1, offset=0)
    x2 = torch.randn(B, 4, W, device="cuda")
    mu2 = torch.randn(2 * B, 4, W, device="cuda")
    spk2 = torch.randn(2 * B, 4, device="cuda")
    cond2 = torch.randn(2 * B, 4, W, device="cuda")
    mask2 = torch.ones(2 * B, W, W, dtype=torch.bool, device="cuda")
    mask2[:, :, 15:] = False

    g_x2, g_cnn2, g_att2 = wrapper.replay(
        x=x2,
        mu_cfg=mu2,
        speakers_cfg=spk2,
        cond_cfg=cond2,
        cnn_cache=None,
        att_cache=None,
        attn_mask=mask2,
        mel_frames=15,
        pad_frames=1,
    )
    e_x2, e_cnn2, e_att2 = _eager_solve_euler(
        dit,
        x2,
        mu2,
        spk2,
        cond2,
        None,
        None,
        mask2,
        wrapper.timeline,
        mel_frames=15,
        pad_frames=1,
    )

    torch.testing.assert_close(g_x2, e_x2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_cnn2, e_cnn2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_att2[..., :15, :], e_att2[..., :15, :], rtol=1e-5, atol=1e-5)

    # Subsequent chunk consuming Call 2 caches (offset=15, mel_frames=16, pad_frames=0)
    mask2_c2 = torch.ones(2 * B, W, W + 15, dtype=torch.bool, device="cuda")
    g_x2_c2, g_cnn2_c2, g_att2_c2 = wrapper.replay(
        x=x1_c2,
        mu_cfg=mu1_c2,
        speakers_cfg=spk1_c2,
        cond_cfg=cond1_c2,
        cnn_cache=g_cnn2,
        att_cache=g_att2[..., :15, :],
        attn_mask=mask2_c2,
        mel_frames=16,
        pad_frames=0,
    )
    e_x2_c2, e_cnn2_c2, e_att2_c2 = _eager_solve_euler(
        dit,
        x1_c2,
        mu1_c2,
        spk1_c2,
        cond1_c2,
        e_cnn2,
        e_att2[..., :15, :],
        mask2_c2,
        wrapper.timeline,
        mel_frames=16,
        pad_frames=0,
    )
    torch.testing.assert_close(g_x2_c2, e_x2_c2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_cnn2_c2, e_cnn2_c2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_att2_c2, e_att2_c2, rtol=1e-5, atol=1e-5)

    # Cache hit check: Call 2 must hit the graph captured in Call 1
    assert wrapper._stats["hits"] >= 1
    wrapper._flush()


class _CurrentFirstCausalDiTBlock(_RealCausalDiTBlock):
    """``_RealCausalDiTBlock`` with stepaudio2's cache layout and masked SDPA.

    stepaudio2 ``Attention.forward_chunk`` concatenates ``[k, k_cache]``: the
    current chunk comes first and the cache after it. Query capture padding
    has to move the cache block, which an old-first fake cannot catch.
    """

    def forward_chunk(self, x, mask, cnn_cache, att_cache, cnn_buf, att_buf):
        B, C, T = x.shape
        if cnn_cache is not None and cnn_cache.shape[-1] > 0:
            conv_in = torch.cat([cnn_cache, x], dim=-1)
        else:
            conv_in = F.pad(x, (self.causal_padding, 0))
        cnn_buf.copy_(conv_in[..., -self.causal_padding :])
        h_conv = self.conv1d(conv_in)

        x_time = x.transpose(1, 2)
        Q = self.q_proj(x_time).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_time).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_time).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if att_cache is not None and att_cache.shape[2] > 0:
            K = torch.cat([K, att_cache[..., : self.head_dim]], dim=2)
            V = torch.cat([V, att_cache[..., self.head_dim :]], dim=2)
        att_buf.copy_(torch.cat([K, V], dim=-1))
        h_attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=None if mask is None else mask.unsqueeze(1))
        h_attn = self.out_proj(h_attn.transpose(1, 2).reshape(B, T, C)).transpose(1, 2)

        x = x + self.norm1((x + h_attn).transpose(1, 2)).transpose(1, 2)
        x = x + self.norm2((x + h_conv).transpose(1, 2)).transpose(1, 2)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_query_bucket_keeps_current_first_cache_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A narrow padded chunk replayed on a wider capture keeps the cache aligned.

    The capture mask has to put the cache block after the full capture width,
    and the returned cache has to drop the capture padding -- otherwise the
    chunk attends to padding, loses its newest cache frames, and hands a
    shifted cache to the next chunk.
    """
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)

    torch.manual_seed(7)
    dit = _RealAttentionCausalConvDiT().eval().cuda()
    dit.blocks = nn.ModuleList([_CurrentFirstCausalDiTBlock() for _ in range(2)]).eval().cuda()
    exact = WholeEulerCFMGraphWrapper(estimator=dit, n_timesteps=10, max_graphs=8, query_bucket_frames=0)
    bucketed = WholeEulerCFMGraphWrapper(estimator=dit, n_timesteps=10, max_graphs=8, query_bucket_frames=16)

    def _mask(width: int, valid: int, offset: int) -> torch.Tensor:
        mask = torch.ones(2, width, width + offset, dtype=torch.bool, device="cuda")
        mask[:, :, valid:width] = False
        return mask

    chunks = [
        (_whole_euler_chunk(1, 16), 16, 0),  # full-width chunk builds a cache
        (_whole_euler_chunk(1, 8), 6, 2),  # narrow padded chunk: capture pads 8 -> 16
        (_whole_euler_chunk(1, 16), 16, 0),  # consumes the narrow chunk's cache
    ]
    caches: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "exact": (None, None),
        "bucketed": (None, None),
    }
    for chunk, mel_frames, pad_frames in chunks:
        chunk["x"][:, :, mel_frames:] = 0.0
        results = {}
        for name, wrapper in (("exact", exact), ("bucketed", bucketed)):
            cnn, att = caches[name]
            offset = 0 if att is None else int(att.shape[4])
            width = int(chunk["mu_cfg"].shape[2])
            results[name] = wrapper.replay(
                **{key: value.clone() for key, value in chunk.items()},
                cnn_cache=cnn,
                att_cache=att,
                attn_mask=_mask(width, mel_frames, offset) if pad_frames else None,
                mel_frames=mel_frames,
                pad_frames=pad_frames,
                # Reserves all three chunks (16 + 6 + 16 frames) up front; trims nothing.
                att_keep=(16, 48),
            )
            assert results[name] is not None
            assert all(torch.isfinite(t).all() for t in results[name])
            caches[name] = (results[name][1], results[name][2])
        for exact_value, bucketed_value in zip(results["exact"], results["bucketed"], strict=True):
            torch.testing.assert_close(bucketed_value, exact_value, rtol=1e-4, atol=1e-5)

    assert {key[2] for key in bucketed._cache} == {16}
    assert {key[2] for key in exact._cache} == {8, 16}
    exact._flush()
    bucketed._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_skipped_when_trt_stepper_configured() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
        BatchedToken2Wav,
    )

    mock_stepper = Mock()
    mock_stepper.step.side_effect = lambda x, mu, t, spks, cond, cnn_cache, att_cache: (
        x,
        torch.zeros(2, x.shape[0], 4, 2, device=x.device),
        torch.zeros(2, x.shape[0], 2, x.shape[2], 4, device=x.device),
    )

    class _MockDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.estimator = _WholeEulerDiT().eval().cuda()
            self.inference_cfg_rate = 0.7
            self.register_buffer("rand_noise", torch.zeros(1, 4, 100, device="cuda"), persistent=False)

    class _MockFlow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = _MockDecoder()
            self.spk_embed_affine_layer = nn.Identity()

    class _MockToken2Wav:
        def __init__(self) -> None:
            self.flow = _MockFlow()
            self.hift = nn.Module()
            self.float16 = False
            self.n_timesteps = 10
            self.mel_cache_len = 1
            self.source_cache_len = 2
            self.speech_window = torch.hamming_window(4, periodic=False)

    # Both TRT stepper and enable_cfm_graph (with enable_whole_euler=True) configured
    adapter = BatchedToken2Wav(
        _MockToken2Wav(),
        trt_stepper=mock_stepper,
        cfm_graph_config={"enabled": True, "enable_whole_euler": True, "bucket_frames": 16},
    )

    # 1. Whole-Euler graph wrapper must not be initialized
    assert adapter._whole_euler_graph_wrapper is None

    # 2. Replay/decode must route to TRT stepper rather than Whole-Euler
    mu = torch.randn(1, 4, 16, device="cuda")
    spk = torch.randn(1, 4, device="cuda")
    cond = torch.randn(1, 4, 16, device="cuda")
    adapter._decode_cfm(
        mu=mu,
        speakers=spk,
        cond=cond,
        cnn_cache=None,
        att_cache=None,
    )
    assert mock_stepper.step.called
    assert mock_stepper.step.call_count == adapter.n_timesteps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_disabled_via_serving_config() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
        BatchedToken2Wav,
    )

    class _MockDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.estimator = _WholeEulerDiT().eval().cuda()
            self.inference_cfg_rate = 0.7
            self.register_buffer("rand_noise", torch.zeros(1, 4, 100, device="cuda"), persistent=False)

    class _MockFlow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = _MockDecoder()
            self.spk_embed_affine_layer = nn.Identity()

    class _MockToken2Wav:
        def __init__(self) -> None:
            self.flow = _MockFlow()
            self.hift = nn.Module()
            self.float16 = False
            self.n_timesteps = 10
            self.mel_cache_len = 1
            self.source_cache_len = 2
            self.speech_window = torch.hamming_window(4, periodic=False)

    adapter_disabled = BatchedToken2Wav(
        _MockToken2Wav(),
        cfm_graph_config={"enabled": True, "enable_whole_euler": False},
    )
    assert adapter_disabled._whole_euler_graph_wrapper is None
    assert adapter_disabled._cfm_graph_wrapper is not None

    adapter_enabled = BatchedToken2Wav(
        _MockToken2Wav(),
        cfm_graph_config={"enabled": True, "enable_whole_euler": True},
    )
    assert adapter_enabled._whole_euler_graph_wrapper is not None
    assert adapter_enabled._cfm_graph_wrapper is not None

    # Unconfigured, the capture width is the steady chunk's mel width.
    adapter_derived = BatchedToken2Wav(
        _MockToken2Wav(),
        connector_config={"codec_chunk_frames": 25, "codec_left_context_frames": 3},
        cfm_graph_config={"enabled": True, "enable_whole_euler": True},
    )
    assert adapter_derived._whole_euler_graph_wrapper.query_bucket_frames == 50


def _tiny_upstream_dit() -> nn.Module:
    """The shipped DiT architecture at toy width, so ``_blocks_forward_chunk_ragged`` runs as in serving."""
    for name in ("cosyvoice2.flow.decoder_dit", "stepaudio2.cosyvoice2.flow.decoder_dit"):
        try:
            import importlib

            decoder_dit = importlib.import_module(name)
            break
        except ImportError:
            pass
    else:
        decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    torch.manual_seed(0)
    estimator = decoder_dit.DiT(in_channels=16, out_channels=4, depth=2, num_heads=2, head_dim=8, hidden_size=16)
    with torch.no_grad():
        # The adaLN-Zero init makes every block an identity; any weights will do here.
        for parameter in estimator.parameters():
            parameter.normal_(0.0, 0.1)
    return estimator.eval().cuda()


def _ragged_mask(lengths: list[int], width: int, offset: int) -> torch.Tensor:
    """The attention mask ``_decode_cfm`` builds for ``valid_lengths``."""
    cfg_lengths = torch.tensor((*lengths, *lengths), device="cuda")
    valid = torch.arange(width, device="cuda").unsqueeze(0) < cfg_lengths.unsqueeze(1)
    current = valid.unsqueeze(1).expand(-1, width, -1)
    old = torch.ones((2 * len(lengths), width, offset), dtype=torch.bool, device="cuda")
    return valid.unsqueeze(2) & torch.cat((current, old), dim=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_ragged_rows_match_per_row_exact_solves(monkeypatch: pytest.MonkeyPatch) -> None:
    """One graph solves rows of different lengths exactly as each row's own unpadded solve.

    Chunk 1 runs through a 16-frame query bucket at width 8: its CNN cache must
    be the unpadded solve's, not zeros. Chunk 2 is ragged (8/5/3 valid frames)
    with garbage in every padded column.
    """
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)
    estimator = _tiny_upstream_dit()
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=8,
        query_bucket_frames=16,
        ragged_body=BatchedToken2Wav._blocks_forward_chunk_ragged,
    )
    batch_size, width = 3, 8

    first = _whole_euler_chunk(batch_size, width)
    graph_x, graph_cnn, graph_att = wrapper.replay(**first, cnn_cache=None, att_cache=None)
    eager_x, eager_cnn, eager_att = _eager_solve_euler(
        estimator,
        first["x"],
        first["mu_cfg"],
        first["speakers_cfg"],
        first["cond_cfg"],
        None,
        None,
        None,
        wrapper.timeline,
    )
    torch.testing.assert_close(graph_x, eager_x, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(graph_cnn, eager_cnn, rtol=1e-4, atol=1e-5)
    assert eager_cnn.abs().amax() > 0
    torch.testing.assert_close(graph_att, eager_att, rtol=1e-4, atol=1e-5)

    lengths = [8, 5, 3]
    second = _whole_euler_chunk(batch_size, width)
    rows_att = _split_cfg_rows(eager_att, batch_size)
    out_x, out_cnn, out_att = wrapper.replay(
        **second,
        cnn_cache=eager_cnn,
        att_cache=rows_att,
        attn_mask=_ragged_mask(lengths, width, width),
        valid_lengths=lengths,
    )
    assert isinstance(out_att, list) and len(out_att) == batch_size
    cnn_rows = _split_cfg_rows(out_cnn, batch_size)
    eager_cnn_rows = _split_cfg_rows(eager_cnn, batch_size)
    for row, length in enumerate(lengths):
        cfg = [row, batch_size + row]
        ref_x, ref_cnn, ref_att = _eager_solve_euler(
            estimator,
            second["x"][row : row + 1, :, :length],
            second["mu_cfg"][cfg, :, :length],
            second["speakers_cfg"][cfg],
            second["cond_cfg"][cfg, :, :length],
            eager_cnn_rows[row],
            rows_att[row],
            None,
            wrapper.timeline,
        )
        torch.testing.assert_close(out_x[row : row + 1, :, :length], ref_x, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(cnn_rows[row], ref_cnn, rtol=1e-4, atol=1e-5)
        assert out_att[row].shape[4] == length + width
        torch.testing.assert_close(out_att[row], ref_att, rtol=1e-4, atol=1e-5)
    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_padded_ragged_replay_after_a_wider_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """A padded ragged replay is exact after a wider replay of the same graph batch.

    Every graph of one batch size shares the lengths buffer. The 16-frame
    replay leaves 16 in the padded row, past the 8-frame history the ragged
    body gathers from.
    """
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)
    estimator = _tiny_upstream_dit()
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=estimator,
        n_timesteps=10,
        max_graphs=8,
        micro_batch_size=4,
        ragged_body=BatchedToken2Wav._blocks_forward_chunk_ragged,
    )
    assert wrapper.replay(**_whole_euler_chunk(4, 16), cnn_cache=None, att_cache=None) is not None

    lengths, width = [8, 5, 3], 8
    assert wrapper._plan_groups(len(lengths)) == [(4, 3)]
    chunk = _whole_euler_chunk(len(lengths), width)
    out_x, out_cnn, out_att = wrapper.replay(
        **chunk,
        cnn_cache=None,
        att_cache=None,
        attn_mask=_ragged_mask(lengths, width, 0),
        valid_lengths=lengths,
    )
    cnn_rows = _split_cfg_rows(out_cnn, len(lengths))
    for row, length in enumerate(lengths):
        cfg = [row, len(lengths) + row]
        ref_x, ref_cnn, ref_att = _eager_solve_euler(
            estimator,
            chunk["x"][row : row + 1, :, :length],
            chunk["mu_cfg"][cfg, :, :length],
            chunk["speakers_cfg"][cfg],
            chunk["cond_cfg"][cfg, :, :length],
            None,
            None,
            None,
            wrapper.timeline,
        )
        torch.testing.assert_close(out_x[row : row + 1, :, :length], ref_x, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(cnn_rows[row], ref_cnn, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out_att[row], ref_att, rtol=1e-4, atol=1e-5)
    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_whole_euler_request_caches_grow_and_update_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request's cache is allocated once, with room for the steady length, then updated in place.

    Both requests start from one shared prompt cache, which is never written.
    Each chunk must match a solve whose input caches cannot be updated in place.
    """
    pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: pool)
    wrapper = WholeEulerCFMGraphWrapper(
        estimator=_tiny_upstream_dit(),
        n_timesteps=10,
        max_graphs=8,
        query_bucket_frames=8,
        ragged_body=BatchedToken2Wav._blocks_forward_chunk_ragged,
    )
    batch_size, width, keep = 2, 8, (8, 16)
    _, _, prompt_att = wrapper.replay(**_whole_euler_chunk(1, width), cnn_cache=None, att_cache=None, att_keep=keep)
    (shared,) = _split_cfg_rows(prompt_att, 1)
    shared_before = shared.clone()

    def fixed_copy(cache: torch.Tensor) -> torch.Tensor:
        # Same values in another memory layout, which is never resized in place.
        return cache.transpose(0, 1).contiguous().transpose(0, 1)

    rows = [shared, shared]
    storages: list[int] | None = None
    for frames in (16, 24, 24):
        chunk = _whole_euler_chunk(batch_size, width)
        _, _, expected = wrapper.replay(
            **chunk, cnn_cache=None, att_cache=[fixed_copy(row) for row in rows], att_keep=keep
        )
        _, _, rows = wrapper.replay(**chunk, cnn_cache=None, att_cache=rows, att_keep=keep)
        for row, reference in zip(rows, expected, strict=True):
            assert row.shape[4] == frames
            assert torch.equal(row, reference)
            assert row.untyped_storage().nbytes() == row[..., :1, :].numel() * sum(keep) * row.element_size()
        if storages is None:
            storages = [row.data_ptr() for row in rows]
            assert shared.data_ptr() not in storages and len(set(storages)) == batch_size
        else:
            assert [row.data_ptr() for row in rows] == storages
    assert torch.equal(shared, shared_before)
    wrapper._flush()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_whole_euler_groups_are_acquired_again_after_a_flush() -> None:
    """Capturing a later group can retire an earlier one; no group replays until all are held at once."""
    wrapper = WholeEulerCFMGraphWrapper(estimator=_tiny_upstream_dit(), n_timesteps=10, max_graphs=8)
    calls: list[int] = []

    def entry(*, graph_batch: int, fill, **_: object) -> tuple:
        calls.append(graph_batch)
        if len(calls) == 2 or flush_always:
            wrapper._stats["flushes"] += 1
        return ("entry", graph_batch, len(calls))

    flush_always = False
    groups, fills = [(16, 16), (1, 1)], [None, None]
    assert wrapper._group_entries(entry, groups, fills) == [("entry", 16, 3), ("entry", 1, 4)]
    assert calls == [16, 1, 16, 1]
    flush_always = True
    assert wrapper._group_entries(entry, groups, fills) is None


@pytest.mark.parametrize("fail", [False, True])
def test_vocoder_restores_tf32_policy(fail):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav

    previous = torch.backends.cuda.matmul.allow_tf32

    def forward(*args, **kwargs):
        assert torch.backends.cuda.matmul.allow_tf32
        if fail:
            raise RuntimeError("injected")
        return "ok"

    model = SimpleNamespace(_extra_config=lambda: {"token2wav_allow_tf32": True}, _forward_impl=forward)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            MiniCPMO45Code2Wav.forward(model)
    else:
        assert MiniCPMO45Code2Wav.forward(model) == "ok"
    assert torch.backends.cuda.matmul.allow_tf32 == previous

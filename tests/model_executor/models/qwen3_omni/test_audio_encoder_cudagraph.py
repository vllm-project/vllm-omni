# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio chunk planning and captured full-tower numerical equivalence."""

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.qwen3_omni.test_vision_encoder_cudagraph_cuda import _parallel_state  # noqa: F401
from vllm_omni.model_executor.models.qwen3_omni.audio_encoder_cudagraph import (
    Qwen3OmniAudioEncoderCudaGraphs,
    audio_chunk_metadata,
    replay_audio_inputs,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"})]


def test_audio_chunk_metadata_preserves_clip_boundaries():
    chunks, indices, boundaries, lengths = audio_chunk_metadata([100, 101, 205], 100, 800)
    assert chunks == [100, 100, 1, 100, 100, 5]
    assert lengths == [13, 14, 27]
    assert boundaries == [0, 13, 27, 54]
    assert indices == list(range(27)) + list(range(39, 66))


def test_default_audio_capture_is_one_full_width_graph_per_chunk_count():
    graphs = Qwen3OmniAudioEncoderCudaGraphs(type("Tower", (), {"n_window": 50})())
    assert graphs.capture_shapes == [(count, count * 13) for count in range(1, 65)]
    # One exact-width graph per single clip shorter than the 100-frame window.
    assert graphs.short_widths == tuple(range(1, 100))
    assert graphs.num_graphs_to_capture() == 64 + 99


def test_explicit_budgets_do_not_capture_short_widths():
    graphs = Qwen3OmniAudioEncoderCudaGraphs(type("Tower", (), {"n_window": 50})(), budgets=(1, 2))
    assert graphs.short_widths == ()


def test_shared_features_and_metadata_refresh(monkeypatch):
    from types import SimpleNamespace

    import vllm_omni.model_executor.models.qwen3_omni.audio_encoder_cudagraph as module

    tower = SimpleNamespace(
        n_window=50,
        n_window_infer=800,
        num_mel_bins=2,
        conv2d1=SimpleNamespace(weight=torch.zeros(1)),
    )
    graphs = Qwen3OmniAudioEncoderCudaGraphs(tower, budgets=(1, 2))
    monkeypatch.setattr(module, "audio_forward_prepared", lambda *args: torch.zeros(args[2].numel(), 2))
    monkeypatch.setattr(graphs, "_capture_graph", lambda forward, pool, key: (None, forward()))
    monkeypatch.setattr(graphs, "_replay_graph", lambda *args: None)
    uploads = []

    def upload(data, *, out):
        tensor = torch.tensor(data, dtype=out.dtype)
        uploads.append(tensor.clone())
        return out.copy_(tensor)

    monkeypatch.setattr(module, "async_tensor_h2d", upload)
    plans = []
    prepare = graphs._prepare_replay

    def prepare_replay(lengths):
        plans.append(tuple(lengths))
        return prepare(lengths)

    monkeypatch.setattr(graphs, "_prepare_replay", prepare_replay)
    graphs.capture(None)
    assert graphs.graphs[1, 13].features.data_ptr() == graphs.graphs[2, 26].features.data_ptr()
    # Same graph and token count, different clip boundaries must refresh.
    for lengths in ([100, 100], [100, 100], [200], [100], [200]):
        graphs.execute(torch.ones(2, sum(lengths)), lengths)
    assert len(uploads) == 6
    assert graphs.graphs[2, 26].cu_seqlens[:2].tolist() == [0, 26]
    assert len(plans) == 4
    # Full-width tail replays also reuse the host plan, but changed tails must
    # refresh both gather indices and clip boundaries even at the same count.
    for lengths in ([101], [101], [199], [101]):
        graphs.execute(torch.ones(2, sum(lengths)), lengths)
        chunks, indices, boundaries, _ = audio_chunk_metadata(lengths, 100, 800)
        planned = replay_audio_inputs(chunks, indices, boundaries, 13, graphs.graphs[2, 26].cu_seqlens.numel(), 104)
        assert graphs.graphs[2, 26].indices.tolist() == planned[0]
        assert graphs.graphs[2, 26].cu_seqlens.tolist() == planned[1]
    assert len(plans) == 7
    assert len(uploads) == 12
    graphs.clear()
    assert graphs._last_replay is None
    graphs.capture(None)
    graphs.execute(torch.ones(2, 101), [101])
    assert len(plans) == 8
    assert len(uploads) == 14


def test_tail_rows_replay_full_width_graph_without_mixing_padding():
    chunks, indices, boundaries, _ = audio_chunk_metadata([100, 101, 205], 100, 800)
    cnn_width = 13
    full_tokens = len(chunks) * cnn_width
    planned = replay_audio_inputs(chunks, indices, boundaries, cnn_width, len(chunks) + 2, 104)
    assert planned is not None
    replay_indices, replay_boundaries, live_tokens = planned
    assert live_tokens == len(indices) == 54
    assert replay_indices[:live_tokens] == indices
    assert sorted(replay_indices) == list(range(full_tokens))
    assert replay_boundaries[: len(boundaries)] == boundaries
    assert replay_boundaries[len(boundaries)] == full_tokens
    assert replay_boundaries[-1] == full_tokens
    assert len(replay_boundaries) == len(chunks) + 2
    lengths = [right - left for left, right in zip(replay_boundaries, replay_boundaries[1:])]
    assert max(lengths) <= 104

    chunks, indices, boundaries, _ = audio_chunk_metadata([100] + [1] * 40, 100, 800)
    full_tokens = len(chunks) * cnn_width
    pad_steps = (full_tokens - len(indices) + 103) // 104
    planned = replay_audio_inputs(chunks, indices, boundaries, cnn_width, len(chunks) + 1 + pad_steps, 104)
    assert planned is not None
    replay_indices, replay_boundaries, live_tokens = planned
    assert replay_indices[:live_tokens] == indices
    assert sorted(replay_indices) == list(range(full_tokens))
    lengths = [right - left for left, right in zip(replay_boundaries, replay_boundaries[1:])]
    assert max(lengths) <= 104


def _init_tower_weights(tower):
    # Weights are allocated uninitialized; exact comparisons need finite outputs.
    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        for name, param in tower.named_parameters():
            if "norm" in name or "ln_post" in name:
                param.fill_(1 if name.endswith("weight") else 0)
            else:
                param.copy_(torch.randn(param.shape, generator=generator, device="cuda") * 0.02)
    return generator


@pytest.mark.parametrize("input_stride", [1, 2])
@pytest.mark.parametrize("backend", [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.usefixtures("_parallel_state")
def test_audio_graph_matches_eager_and_preserves_cached_outputs(monkeypatch, backend, input_stride):
    import vllm.model_executor.models.vision as vision_module
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.distributed.parallel_state import graph_capture
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder

    monkeypatch.setattr(
        vision_module, "get_multimodal_config", lambda: MultiModalConfig(mm_encoder_attn_backend=backend)
    )
    config = Qwen3OmniMoeAudioEncoderConfig(
        d_model=128,
        encoder_attention_heads=2,
        encoder_layers=2,
        encoder_ffn_dim=256,
        downsample_hidden_size=8,
        output_dim=64,
        num_mel_bins=80,
        max_source_positions=1500,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=2,
    )
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        tower = Qwen3OmniMoeAudioEncoder(config)
    generator = _init_tower_weights(tower)
    graphs = Qwen3OmniAudioEncoderCudaGraphs(
        tower, budgets=tuple(range(1, 17)), extra_shapes=((5, 47), (9, 105), (10, 118), (15, 195))
    )
    # Sub-window inputs fall back, so one chunk only needs the full-width graph.
    assert [shape for shape in graphs.capture_shapes if shape[0] == 1] == [(1, 13)]
    retained = []
    with torch.inference_mode(), graph_capture(device=torch.device("cuda")):
        set_cudagraph_capturing_enabled(True)
        try:
            graphs.capture(torch.cuda.graph_pool_handle())
        finally:
            set_cudagraph_capturing_enabled(False)
    with torch.inference_mode():
        for lengths in (
            [100],
            [200, 100],
            [100, 100, 100],
            [100, 99, 99, 99],
            [100] + [99] * 7,
            [100] + [99] * 15,
            [101],
            [101],  # Same host plan, new feature values.
            [199],
            [101],  # Restore the gather indices after a different tail.
            [201],
            [150, 205],
            [205, 150],
            [801],
            [1500],
            [801, 100],
            [1601],
        ):
            features = torch.randn(
                80, sum(lengths) * input_stride, device="cuda", dtype=torch.bfloat16, generator=generator
            )[:, ::input_stride]
            lens = torch.tensor(lengths, device="cuda")
            output_lens = audio_chunk_metadata(list(lengths), 100, 800)[-1]
            expected = tower(features, lens, torch.tensor(output_lens, device="cuda")).split(output_lens)
            actual = graphs.execute(features, list(lengths))
            if sum((n + 99) // 100 for n in lengths) > 16:
                assert actual is None
                continue
            assert [x.shape for x in actual] == [x.shape for x in expected]
            for output, reference in zip(actual, expected):
                # Full-width replay pads the GEMM rows, so the GPU may pick a
                # different kernel and round differently from eager.
                torch.testing.assert_close(output, reference, rtol=1.6e-2, atol=2e-3)
                retained.append((output, output.clone()))
        assert graphs.execute(torch.zeros(80, 99, device="cuda", dtype=torch.bfloat16), [99]) is None
        for output, saved in retained:
            torch.testing.assert_close(output, saved, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.usefixtures("_parallel_state")
def test_short_clip_graph_matches_eager(monkeypatch):
    """A single sub-window clip replays an exact-width graph, not eager."""
    import vllm.model_executor.models.vision as vision_module
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.distributed.parallel_state import graph_capture
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder

    monkeypatch.setattr(
        vision_module,
        "get_multimodal_config",
        lambda: MultiModalConfig(mm_encoder_attn_backend=AttentionBackendEnum.FLASH_ATTN),
    )
    config = Qwen3OmniMoeAudioEncoderConfig(
        d_model=128,
        encoder_attention_heads=2,
        encoder_layers=2,
        encoder_ffn_dim=256,
        downsample_hidden_size=8,
        output_dim=64,
        num_mel_bins=80,
        max_source_positions=1500,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=2,
    )
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        tower = Qwen3OmniMoeAudioEncoder(config)
    _init_tower_weights(tower)
    graphs = Qwen3OmniAudioEncoderCudaGraphs(tower, budgets=(1,), short_widths=(13, 50, 80, 99))
    with torch.inference_mode(), graph_capture(device=torch.device("cuda")):
        set_cudagraph_capturing_enabled(True)
        try:
            graphs.capture(torch.cuda.graph_pool_handle())
        finally:
            set_cudagraph_capturing_enabled(False)
    generator = torch.Generator(device="cuda").manual_seed(7)
    with torch.inference_mode():
        for frames in (13, 50, 80, 99):
            features = torch.randn(80, frames, device="cuda", dtype=torch.bfloat16, generator=generator)
            output_lens = audio_chunk_metadata([frames], 100, 800)[-1]
            expected = tower(features, torch.tensor([frames], device="cuda"), torch.tensor(output_lens, device="cuda"))
            actual = graphs.execute(features, [frames])
            assert actual is not None
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
        # Two short clips are not one captured CNN width.
        assert graphs.execute(torch.randn(80, 100, device="cuda", dtype=torch.bfloat16), [50, 50]) is None

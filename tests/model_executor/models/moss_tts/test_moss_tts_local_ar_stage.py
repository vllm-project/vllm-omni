# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.moss_tts.moss_tts_local_ar_stage import (
    _LOCAL_CUDAGRAPH_DEFAULT_BATCH_SIZES,
    MossTTSARStageModel,
    MossTTSLocalKVCache,
    MossTTSLocalRequestState,
    MossTTSNativeLocalTransformer,
    _apply_top_p_filter,
    _parse_local_cudagraph_batch_sizes,
)
from vllm_omni.model_executor.models.moss_tts.moss_tts_local_cuda_graph import (
    MossTTSLocalCUDAGraphManager,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_top_p_filter_masks_tokens_outside_nucleus():
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0]])

    filtered = _apply_top_p_filter(logits.clone(), 0.8)

    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_parse_local_cudagraph_batch_sizes_dedupes_and_ignores_blanks():
    assert _parse_local_cudagraph_batch_sizes("1, 2,,4,2") == (1, 2, 4)


def test_parse_local_cudagraph_batch_sizes_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive integers"):
        _parse_local_cudagraph_batch_sizes("1,0")


def test_local_cudagraph_default_batch_sizes_match_merge_safe_plan():
    assert _parse_local_cudagraph_batch_sizes(
        _LOCAL_CUDAGRAPH_DEFAULT_BATCH_SIZES
    ) == (1, 2, 4, 8, 16, 32)


class _TinyLocalTransformer(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.supports_kv_cache = False

    def forward(self, inputs_embeds, past_key_values=None, use_cache=False):
        del past_key_values, use_cache
        return torch.tanh(self.proj(inputs_embeds)), None


class _TinyCfg:
    local_hidden_size = 4


def _make_tiny_ar_stage(device: torch.device) -> MossTTSARStageModel:
    torch.manual_seed(0)
    model = object.__new__(MossTTSARStageModel)
    nn.Module.__init__(model)
    model.config = _TinyCfg()
    model.channels = 3
    model.n_vq = 2
    model.audio_pad_code = 5
    model.local_transformer = _TinyLocalTransformer(4).to(device).eval()
    model.speech_embedding_to_local_mlp = nn.Linear(4, 4, bias=False).to(device)
    model.local_to_speech_embedding_mlps = nn.ModuleList(
        [nn.Linear(4, 4, bias=False) for _ in range(model.channels)]
    ).to(device)
    model.layer_norm_before_lm_heads = nn.ModuleList(
        [nn.LayerNorm(4) for _ in range(model.channels)]
    ).to(device)
    model.lm_heads = nn.ModuleList(
        [nn.Linear(4, 8, bias=False) for _ in range(model.channels)]
    ).to(device)
    model.embedding_list = nn.ModuleList(
        [nn.Embedding(8, 4) for _ in range(model.channels)]
    ).to(device)
    model._local_cudagraphs = None
    for module in (
        model.speech_embedding_to_local_mlp,
        model.local_to_speech_embedding_mlps,
        model.layer_norm_before_lm_heads,
        model.lm_heads,
        model.embedding_list,
    ):
        module.eval()
    return model


def test_local_cudagraph_manager_cpu_replays_fallback_to_eager():
    model = _make_tiny_ar_stage(torch.device("cpu"))
    manager = MossTTSLocalCUDAGraphManager(model, batch_sizes=(1,), warmups=1)
    current_proj = torch.randn(1, 4)
    local_ctx = torch.zeros(1, 0, 4)

    result = manager.replay_channel(
        channel=0,
        current_proj=current_proj,
        local_ctx=local_ctx,
        logits_dim=8,
    )

    assert result is None


def test_local_cudagraph_manager_selects_smallest_usable_bucket():
    model = _make_tiny_ar_stage(torch.device("cpu"))
    manager = MossTTSLocalCUDAGraphManager(model, batch_sizes=(4, 1, 8), warmups=1)

    assert manager._select_bucket(1) == 1
    assert manager._select_bucket(2) == 4
    assert manager._select_bucket(4) == 4
    assert manager._select_bucket(5) == 8
    assert manager._select_bucket(9) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_local_cudagraph_replay_matches_eager_logits():
    device = torch.device("cuda:0")
    model = _make_tiny_ar_stage(device)
    manager = MossTTSLocalCUDAGraphManager(model, batch_sizes=(1,), warmups=1)
    current_proj = torch.randn(1, 4, device=device)
    local_ctx = torch.zeros(1, 0, 4, device=device)

    eager_logits, eager_ctx = model._local_channel_logits_eager(
        ch=0,
        current_proj=current_proj,
        local_ctx=local_ctx,
    )
    graph_result = manager.replay_channel(
        channel=0,
        current_proj=current_proj,
        local_ctx=local_ctx,
        logits_dim=8,
    )

    assert graph_result is not None
    graph_logits, graph_ctx = graph_result
    torch.testing.assert_close(graph_ctx, eager_ctx)
    torch.testing.assert_close(graph_logits, eager_logits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_local_cudagraph_replay_with_larger_bucket_matches_eager_logits():
    device = torch.device("cuda:0")
    model = _make_tiny_ar_stage(device)
    manager = MossTTSLocalCUDAGraphManager(model, batch_sizes=(4,), warmups=1)
    current_proj = torch.randn(2, 4, device=device)
    local_ctx = torch.zeros(2, 0, 4, device=device)

    eager_logits, eager_ctx = model._local_channel_logits_eager(
        ch=0,
        current_proj=current_proj,
        local_ctx=local_ctx,
    )
    graph_result = manager.replay_channel(
        channel=0,
        current_proj=current_proj,
        local_ctx=local_ctx,
        logits_dim=8,
    )

    assert graph_result is not None
    graph_logits, graph_ctx = graph_result
    assert graph_logits.shape == eager_logits.shape
    torch.testing.assert_close(graph_ctx, eager_ctx)
    torch.testing.assert_close(graph_logits, eager_logits)


def test_compute_logits_forces_audio_start_when_fsm_state_missing():
    model = object.__new__(MossTTSARStageModel)
    model.audio_start_token_id = 4
    model._last_request_ids = []
    model.lm_heads = [torch.nn.Linear(2, 8, bias=False)]
    with torch.no_grad():
        model.lm_heads[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                    [5.0, 0.0],
                    [6.0, 0.0],
                    [7.0, 0.0],
                    [8.0, 0.0],
                ]
            )
        )

    logits = MossTTSARStageModel.compute_logits(model, torch.ones((1, 2)))

    assert torch.isfinite(logits[0, 4])
    masked = torch.cat([logits[0, :4], logits[0, 5:]])
    assert torch.isneginf(masked).all()


def test_native_local_transformer_recompute_forward_shape():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
    )
    model = MossTTSNativeLocalTransformer(config)

    hidden, past = model(torch.randn(2, 3, 16))

    assert hidden.shape == (2, 3, 16)
    assert past is None


def test_native_local_transformer_rejects_non_incremental_cache_path():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
    )
    model = MossTTSNativeLocalTransformer(config)

    with pytest.raises(RuntimeError, match="single-token incremental"):
        model(torch.randn(1, 2, 16), use_cache=True)


def test_native_local_transformer_rejects_cache_without_use_cache():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
    )
    model = MossTTSNativeLocalTransformer(config)

    with pytest.raises(RuntimeError, match="use_cache=False"):
        model(
            torch.randn(1, 1, 16),
            past_key_values=MossTTSLocalKVCache(config.num_hidden_layers),
            use_cache=False,
        )


def test_native_local_transformer_incremental_cache_matches_last_token_recompute():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
    )
    model = MossTTSNativeLocalTransformer(config)
    inputs = torch.randn(2, 4, 16)

    full_hidden, _ = model(inputs)

    cache = None
    step_hidden = None
    for pos in range(inputs.shape[1]):
        step_hidden, cache = model(
            inputs[:, pos : pos + 1, :],
            past_key_values=cache,
            use_cache=True,
        )

    assert cache is not None
    assert cache.get_seq_length() == inputs.shape[1]
    assert step_hidden is not None
    torch.testing.assert_close(step_hidden[:, -1, :], full_hidden[:, -1, :])


def test_request_state_keeps_pending_audio_row_on_input_device():
    state = MossTTSLocalRequestState(n_vq=4, audio_pad_code=1024)
    row = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    state.store_next_audio_row(row)

    assert state.pending_audio_row.device == row.device
    assert state.pending_audio_row.tolist() == [1, 2, 3, 4]


def test_outer_capture_local_forward_guard_stays_disabled():
    model = object.__new__(MossTTSARStageModel)

    assert model._should_run_local_forward_during_outer_capture() is False

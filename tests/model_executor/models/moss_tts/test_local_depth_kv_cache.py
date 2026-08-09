# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config

from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
    MossTTSLocalConfig,
)
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
    MossTTSLocalDepthTransformer,
)

_HIDDEN_SIZE = 64
_N_HEAD = 4
_N_INNER = 128
_N_VQ = 12
_AUDIO_VOCAB_SIZE = 32


def _make_local(*, use_static_local_kv_cache: bool = False) -> MossTTSLocalDepthTransformer:
    config = GPT2Config(
        n_embd=_HIDDEN_SIZE,
        n_head=_N_HEAD,
        n_inner=_N_INNER,
        n_layer=1,
        layer_norm_epsilon=1e-6,
    )
    config.rope_base = 1_000_000.0
    return MossTTSLocalDepthTransformer(
        config,
        hidden_size=_HIDDEN_SIZE,
        max_positions=_N_VQ,
        use_static_local_kv_cache=use_static_local_kv_cache,
    ).eval()


def _reference_greedy_frame(
    local: MossTTSLocalDepthTransformer,
    backbone_hidden: torch.Tensor,
    audio_heads: nn.ModuleList,
    audio_embeddings: nn.ModuleList,
    text_head: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = backbone_hidden.shape[0]
    embeds = backbone_hidden.new_zeros(batch_size, _N_VQ, _HIDDEN_SIZE)
    embeds[:, 0] = backbone_hidden
    local_hidden = local._forward_prefix(embeds[:, :1])[:, -1]
    should_continue = text_head(local_hidden).argmax(dim=-1).eq(0)
    codes = torch.zeros(batch_size, _N_VQ, dtype=torch.long)
    for channel in range(_N_VQ):
        code = audio_heads[channel](local_hidden).argmax(dim=-1)
        codes[:, channel] = code
        if channel + 1 < _N_VQ:
            embeds[:, channel + 1] = audio_embeddings[channel](code)
            local_hidden = local._forward_prefix(embeds[:, : channel + 2])[:, -1]
    return should_continue, codes


@pytest.mark.parametrize(
    ("config_kwargs", "expected"),
    [
        ({}, False),
        ({"use_static_local_kv_cache": True}, True),
    ],
)
def test_local_config_preserves_static_kv_cache_opt_in(
    config_kwargs: dict[str, bool],
    expected: bool,
) -> None:
    config = MossTTSLocalConfig(**config_kwargs)
    assert config.use_static_local_kv_cache is expected


def test_generate_frame_without_static_cache_uses_prefix_without_workspace() -> None:
    torch.manual_seed(4)
    local = _make_local(use_static_local_kv_cache=False)
    audio_heads = nn.ModuleList([nn.Linear(_HIDDEN_SIZE, _AUDIO_VOCAB_SIZE, bias=False) for _ in range(_N_VQ)])
    audio_embeddings = nn.ModuleList([nn.Embedding(_AUDIO_VOCAB_SIZE, _HIDDEN_SIZE) for _ in range(_N_VQ)])
    text_head = nn.Linear(_HIDDEN_SIZE, 2, bias=False)
    backbone_hidden = torch.randn(2, _HIDDEN_SIZE)

    expected_continue, expected_codes = _reference_greedy_frame(
        local,
        backbone_hidden,
        audio_heads,
        audio_embeddings,
        text_head,
    )
    actual_continue, actual_codes = local.generate_frame(
        backbone_hidden,
        audio_heads,
        audio_embeddings,
        text_head,
        n_vq=_N_VQ,
        do_sample=False,
    )

    torch.testing.assert_close(actual_continue, expected_continue)
    torch.testing.assert_close(actual_codes, expected_codes)
    local.prepare_kv_cache(4)
    assert local._kv_cache == []


def test_generate_frame_with_static_cache_uses_fixed_workspace() -> None:
    torch.manual_seed(5)
    local = _make_local(use_static_local_kv_cache=True)
    audio_heads = nn.ModuleList([nn.Linear(_HIDDEN_SIZE, _AUDIO_VOCAB_SIZE, bias=False) for _ in range(_N_VQ)])
    audio_embeddings = nn.ModuleList([nn.Embedding(_AUDIO_VOCAB_SIZE, _HIDDEN_SIZE) for _ in range(_N_VQ)])
    text_head = nn.Linear(_HIDDEN_SIZE, 2, bias=False)
    backbone_hidden = torch.randn(2, _HIDDEN_SIZE)

    expected_continue, expected_codes = _reference_greedy_frame(
        local,
        backbone_hidden,
        audio_heads,
        audio_embeddings,
        text_head,
    )
    actual_continue, actual_codes = local.generate_frame(
        backbone_hidden,
        audio_heads,
        audio_embeddings,
        text_head,
        n_vq=_N_VQ,
        do_sample=False,
    )

    torch.testing.assert_close(actual_continue, expected_continue)
    torch.testing.assert_close(actual_codes, expected_codes)
    assert local._kv_capacity == backbone_hidden.shape[0]
    assert len(local._kv_cache) == len(local.h)


def test_fixed_seed_sampling_matches_prefix_path() -> None:
    torch.manual_seed(6)
    prefix = _make_local(use_static_local_kv_cache=False)
    cached = _make_local(use_static_local_kv_cache=True)
    cached.load_state_dict(prefix.state_dict())
    audio_heads = nn.ModuleList([nn.Linear(_HIDDEN_SIZE, _AUDIO_VOCAB_SIZE, bias=False) for _ in range(_N_VQ)])
    audio_embeddings = nn.ModuleList([nn.Embedding(_AUDIO_VOCAB_SIZE, _HIDDEN_SIZE) for _ in range(_N_VQ)])
    text_head = nn.Linear(_HIDDEN_SIZE, 2, bias=False)
    backbone_hidden = torch.randn(3, _HIDDEN_SIZE)

    generation_kwargs = {
        "n_vq": _N_VQ,
        "temperature": 0.8,
        "top_k": 16,
        "top_p": 0.9,
        "text_temperature": 0.7,
        "text_top_k": 2,
        "text_top_p": 0.95,
    }
    prefix_output = prefix.generate_frame(
        backbone_hidden,
        audio_heads,
        audio_embeddings,
        text_head,
        generator=torch.Generator().manual_seed(1234),
        **generation_kwargs,
    )
    cached_output = cached.generate_frame(
        backbone_hidden,
        audio_heads,
        audio_embeddings,
        text_head,
        generator=torch.Generator().manual_seed(1234),
        **generation_kwargs,
    )

    torch.testing.assert_close(cached_output[0], prefix_output[0])
    torch.testing.assert_close(cached_output[1], prefix_output[1])


def test_setup_compile_selects_configured_path(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled_functions = []

    def fake_compile(function, *, dynamic, options):
        assert dynamic is False
        assert options == {"epilogue_fusion": False}
        compiled_functions.append(function)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth."
        "current_omni_platform.supports_torch_inductor",
        lambda: True,
    )

    prefix = _make_local(use_static_local_kv_cache=False)
    prefix.setup_compile()
    assert compiled_functions[-1].__func__ is prefix._forward_prefix.__func__
    assert prefix._compiled_forward_token is None

    cached = _make_local(use_static_local_kv_cache=True)
    cached.setup_compile()
    assert compiled_functions[-1].__func__ is cached._forward_token_static.__func__
    assert cached._compiled_forward_prefix is None


def test_incremental_hidden_and_logits_match_full_prefix() -> None:
    torch.manual_seed(0)
    local = _make_local(use_static_local_kv_cache=True)
    inputs = torch.randn(3, _N_VQ, _HIDDEN_SIZE)
    text_head = nn.Linear(_HIDDEN_SIZE, 2, bias=False)
    audio_heads = nn.ModuleList([nn.Linear(_HIDDEN_SIZE, _AUDIO_VOCAB_SIZE, bias=False) for _ in range(_N_VQ)])

    for position in range(_N_VQ):
        expected = local._forward_prefix(inputs[:, : position + 1])[:, -1]
        actual = local._run_token(inputs[:, position], position)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        head = text_head if position == 0 else audio_heads[position]
        torch.testing.assert_close(
            head(actual),
            head(expected),
            rtol=1e-4,
            atol=1e-5,
        )


def test_incremental_cache_does_not_leak_across_frames_or_batch_sizes() -> None:
    torch.manual_seed(1)
    local = _make_local(use_static_local_kv_cache=True)
    first_frame = torch.randn(4, _N_VQ, _HIDDEN_SIZE)
    second_frame = torch.randn(2, _N_VQ, _HIDDEN_SIZE)

    for position in range(_N_VQ):
        local._run_token(first_frame[:, position], position)
    for position in range(_N_VQ):
        actual = local._run_token(second_frame[:, position], position)
        expected = local._forward_prefix(second_frame[:, : position + 1])[:, -1]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_greedy_frame_codes_match_full_prefix() -> None:
    torch.manual_seed(2)
    local = _make_local(use_static_local_kv_cache=True)
    audio_heads = nn.ModuleList([nn.Linear(_HIDDEN_SIZE, _AUDIO_VOCAB_SIZE, bias=False) for _ in range(_N_VQ)])
    audio_embeddings = nn.ModuleList([nn.Embedding(_AUDIO_VOCAB_SIZE, _HIDDEN_SIZE) for _ in range(_N_VQ)])
    text_head = nn.Linear(_HIDDEN_SIZE, 2, bias=False)
    backbone_hidden = torch.randn(3, _HIDDEN_SIZE)

    expected_continue, expected_codes = _reference_greedy_frame(
        local,
        backbone_hidden,
        audio_heads,
        audio_embeddings,
        text_head,
    )
    actual_continue, actual_codes = local.generate_frame(
        backbone_hidden,
        audio_heads,
        audio_embeddings,
        text_head,
        n_vq=_N_VQ,
        do_sample=False,
    )

    torch.testing.assert_close(actual_continue, expected_continue)
    torch.testing.assert_close(actual_codes, expected_codes)


def test_prepare_kv_cache_freezes_capacity_and_addresses() -> None:
    local = _make_local(use_static_local_kv_cache=True)
    local._run_token(torch.randn(2, _HIDDEN_SIZE), 0)
    local.prepare_kv_cache(4)
    pointers = [(key.data_ptr(), value.data_ptr()) for key, value in local._kv_cache]

    local._run_token(torch.randn(1, _HIDDEN_SIZE), 0)
    assert pointers == [(key.data_ptr(), value.data_ptr()) for key, value in local._kv_cache]
    with pytest.raises(RuntimeError, match="frozen"):
        local._run_token(torch.randn(5, _HIDDEN_SIZE), 0)


def test_rejects_out_of_range_position() -> None:
    local = _make_local(use_static_local_kv_cache=True)
    with pytest.raises(ValueError, match="out of range"):
        local._run_token(torch.randn(1, _HIDDEN_SIZE), _N_VQ)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_graph_replay_updates_fixed_kv_workspace() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    local = _make_local(use_static_local_kv_cache=True).to(device=device, dtype=torch.bfloat16)
    local.prepare_kv_cache(2)
    local.setup_compile()
    static_inputs = torch.randn(
        2,
        _N_VQ,
        _HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )

    def run_frame() -> torch.Tensor:
        return torch.stack(
            [local._run_token(static_inputs[:, position], position) for position in range(_N_VQ)],
            dim=1,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            run_frame()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.accelerator.synchronize()

    pointers = [(key.data_ptr(), value.data_ptr()) for key, value in local._kv_cache]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run_frame()

    static_inputs.copy_(torch.randn_like(static_inputs))
    graph.replay()
    replay_output = graph_output.clone()
    eager_output = run_frame()

    torch.testing.assert_close(replay_output, eager_output, rtol=2e-2, atol=2e-2)
    assert pointers == [(key.data_ptr(), value.data_ptr()) for key, value in local._kv_cache]

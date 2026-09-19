# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for Personaplex depformer static KV and CUDA-graph wrapper control flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.personaplex.duplex._depformer_testing import (
    clone_depformer,
    frame,
    make_depformer,
)
from vllm_omni.model_executor.models.personaplex.configuration_personaplex import PersonaPlexConfig
from vllm_omni.model_executor.models.personaplex.personaplex_depformer_cudagraph import (
    CUDAGraphDepformerWrapper,
    resolve_depformer_graph_settings,
)
from vllm_omni.model_executor.models.personaplex.personaplex_talker import PersonaPlexTalkerForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_padding_rows_do_not_affect_live_rows() -> None:
    live = make_depformer(seed=3)
    padded = clone_depformer(live)
    text1, hidden1, tokens1, provided1 = frame(batch=1, seed=4)
    text2, hidden2, tokens2, provided2 = frame(batch=2, seed=5)
    text2[0] = text1[0]
    hidden2[0] = hidden1[0]
    tokens2[0] = tokens1[0]
    provided2[0] = provided1[0]

    out1 = live(text1, hidden1, audio_tokens=tokens1, audio_provided=provided1)
    out2 = padded(text2, hidden2, audio_tokens=tokens2, audio_provided=provided2)
    torch.testing.assert_close(out2[0], out1[0], rtol=0, atol=0)


def test_kv_reset_isolates_successive_frames() -> None:
    streaming = make_depformer(seed=6)
    isolated = clone_depformer(streaming)
    first = frame(batch=1, seed=7)
    second = frame(batch=1, seed=8)
    streaming(*first[:2], audio_tokens=first[2], audio_provided=first[3])
    streamed_second = streaming(*second[:2], audio_tokens=second[2], audio_provided=second[3])
    only_second = isolated(*second[:2], audio_tokens=second[2], audio_provided=second[3])
    torch.testing.assert_close(streamed_second, only_second, rtol=0, atol=0)


def test_wrapper_selects_smallest_capture_size_that_fits() -> None:
    model = make_depformer()
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1, 2, 4, 8], enabled=False)
    assert wrapper._select_padded_b(1) == 1
    assert wrapper._select_padded_b(3) == 4
    assert wrapper._select_padded_b(9) is None


def test_disabled_wrapper_stays_eager() -> None:
    model = make_depformer(seed=9)
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1, 2], enabled=False)
    wrapper.warmup(torch.device("cpu"))
    text, hidden, tokens, provided = frame(batch=1, seed=10)
    out_wrap = wrapper(text, hidden, audio_tokens=tokens, audio_provided=provided)
    out_eager = model(text, hidden, audio_tokens=tokens, audio_provided=provided)
    torch.testing.assert_close(out_wrap, out_eager, rtol=0, atol=0)
    stats = wrapper.stats_snapshot()
    assert stats["calls"] == 1
    assert stats["eager"] == 1
    assert stats["replays"] == 0
    assert stats["num_graphs"] == 0


def test_cpu_warmup_does_not_capture() -> None:
    model = make_depformer()
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1, 2], enabled=True)
    wrapper.warmup(torch.device("cpu"))
    assert not wrapper.is_ready
    text, hidden, tokens, provided = frame(batch=2, seed=11)
    wrapper(text, hidden, audio_tokens=tokens, audio_provided=provided)
    assert wrapper.stats.eager == 1
    assert wrapper.stats.replays == 0


def test_resolve_depformer_graph_settings_uses_compilation_and_max_seqs() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4], cudagraph_num_of_warmups=1),
        scheduler_config=SimpleNamespace(max_num_seqs=10),
    )
    enabled, sizes, max_batch, warmup = resolve_depformer_graph_settings(vllm_config, enabled=True)
    assert enabled is True
    assert sizes == (1, 2, 4)
    assert max_batch == 10
    assert warmup == 1


def test_personaplex_config_depformer_cuda_graphs_default_off() -> None:
    assert PersonaPlexConfig().depformer_cuda_graphs is False


def _talker_with_depformer(depformer, recorded: list) -> SimpleNamespace:
    model = SimpleNamespace(
        _dtype=torch.float32,
        depformer=depformer,
        _depformer_graphs_enabled=False,
        _depformer_graph=None,
        _duplex_stage0_runtime=lambda: SimpleNamespace(
            record_sample=lambda *, request_id, text_token, agent_codes: recorded.append(
                (request_id, text_token.clone(), agent_codes.clone())
            )
        ),
    )
    model._maybe_init_depformer_graphs = PersonaPlexTalkerForConditionalGeneration._maybe_init_depformer_graphs.__get__(
        model
    )
    model._run_depformer = PersonaPlexTalkerForConditionalGeneration._run_depformer.__get__(model)
    return model


def test_run_depformer_uses_eager_module_when_graphs_disabled() -> None:
    model = make_depformer(seed=40)
    recorded: list = []
    talker = _talker_with_depformer(model, recorded)
    text, hidden, tokens, provided = frame(batch=1, seed=41)
    got = PersonaPlexTalkerForConditionalGeneration._run_depformer(
        talker, text, hidden, audio_tokens=tokens, audio_provided=provided
    )
    want = model(text, hidden, audio_tokens=tokens, audio_provided=provided)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_talk_run_depformer_dispatches_to_wrapper() -> None:
    model = make_depformer(seed=42)
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1], enabled=False)
    recorded: list = []
    talker = _talker_with_depformer(model, recorded)
    talker._depformer_graphs_enabled = True
    talker._depformer_graph = wrapper
    talker._maybe_init_depformer_graphs = lambda: None
    text, hidden, tokens, provided = frame(batch=1, seed=43)
    got = PersonaPlexTalkerForConditionalGeneration._run_depformer(talker, text, hidden, tokens, provided)
    want = model(text, hidden, audio_tokens=tokens, audio_provided=provided)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert wrapper.stats.calls == 1
    assert wrapper.stats.eager == 1

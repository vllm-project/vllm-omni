# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend hardware validation for MiniCPM-o Code2Wav NPUGraph replay."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
)
from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner
from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import (
    _cfm_loop_constants,
    _graphable_cfm_loop,
    _graphable_estimator_step,
    prepare_code2wav_graph_runtime,
)
from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
    npu_token2wav_sdpa_context,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.omni,
    *hardware_marks(res={"npu": "A3"}, num_cards=1),
]


class _Flow(nn.Module):
    def __init__(self, estimator: nn.Module):
        super().__init__()
        self.decoder = SimpleNamespace(
            estimator=estimator,
            inference_cfg_rate=0.7,
            rand_noise=torch.zeros((1, 2, 512), device="npu"),
        )


class _Token2Wav:
    def __init__(self, estimator: nn.Module, *, steps: int = 2):
        self.flow = _Flow(estimator).eval()
        self.hift = nn.Identity().eval()
        self.float16 = False
        self.n_timesteps = steps
        self.mel_cache_len = 1
        self.source_cache_len = 2
        self.speech_window = torch.ones(4, device="npu")


@pytest.mark.parametrize("flow_fp16", [False, True], ids=["fp32", "flow-fp16"])
def test_minicpmo_cfm_estimator_npugraph_matches_eager(flow_fp16):
    decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()

    estimator = (
        decoder_dit.DiT(
            in_channels=6,
            out_channels=2,
            depth=2,
            num_heads=2,
            head_dim=8,
            hidden_size=16,
            mlp_ratio=2.0,
        )
        .npu()
        .eval()
    )
    graph_runner = NPUExactGraphRunner(max_graphs=4)
    adapter = BatchedToken2Wav(
        _Token2Wav(estimator),
        npu_flow_float16=flow_fp16,
    )

    def inputs(seed: int):
        def make(shape: tuple[int, ...], offset: float):
            values = torch.arange(prod(shape), device="npu", dtype=torch.float32)
            return torch.sin(values * 0.17 + seed + offset).reshape(shape)

        return (
            make((2, 2, 6), 0.0),
            make((2, 2, 6), 0.5),
            make((2, 1, 16), 1.0),
            make((2, 1), 1.5),
            make((2, 1, 6), 2.0),
        )

    def compute(*values, caches=None):
        cnn_cache, att_cache = (None, None) if caches is None else caches
        return _graphable_estimator_step(
            adapter,
            estimator,
            x=values[0],
            mu=values[1],
            time_embedding=values[2],
            speakers=values[3],
            cond=values[4],
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )

    with torch.inference_mode(), npu_token2wav_sdpa_context(require_math=True):
        warm_inputs = inputs(11)
        warm_outputs = graph_runner.run(
            "cfm_estimator",
            warm_inputs,
            (False, "float16" if flow_fp16 else "float32"),
            lambda *values: compute(*values),
        )
        replay_inputs = inputs(13)
        expected = compute(*replay_inputs)
        actual = graph_runner.run(
            "cfm_estimator",
            replay_inputs,
            (False, "float16" if flow_fp16 else "float32"),
            lambda *values: compute(*values),
        )
        for eager, replayed in zip(expected, actual, strict=True):
            assert torch.isfinite(replayed).all()
            torch.testing.assert_close(replayed, eager, rtol=1e-3, atol=1e-3)

        cached_inputs = (*inputs(17), warm_outputs[1], warm_outputs[2])
        graph_runner.run(
            "cfm_estimator",
            cached_inputs,
            (True, "float16" if flow_fp16 else "float32"),
            lambda *values: compute(*values[:5], caches=(values[5], values[6])),
        )
        replay_cached_inputs = (
            *inputs(19),
            warm_outputs[1] + 0.1,
            warm_outputs[2] + 0.1,
        )
        expected_cached = compute(
            *replay_cached_inputs[:5],
            caches=(replay_cached_inputs[5], replay_cached_inputs[6]),
        )
        actual_cached = graph_runner.run(
            "cfm_estimator",
            replay_cached_inputs,
            (True, "float16" if flow_fp16 else "float32"),
            lambda *values: compute(*values[:5], caches=(values[5], values[6])),
        )
        for eager, replayed in zip(expected_cached, actual_cached, strict=True):
            assert torch.isfinite(replayed).all()
            torch.testing.assert_close(replayed, eager, rtol=1e-3, atol=1e-3)
        assert graph_runner.stats == {"captures": 2, "failed": 0, "hits": 2}
        expected_dtype = "float16" if flow_fp16 else "float32"
        assert adapter.precision_telemetry()["effective_dtype"] == expected_dtype


@pytest.mark.parametrize("steps", [10, 8, 6])
@pytest.mark.parametrize("flow_fp16", [False, True], ids=["fp32", "flow-fp16"])
@pytest.mark.parametrize("with_cache", [False, True], ids=["uncached", "cached"])
def test_minicpmo_whole_cfm_loop_npugraph_matches_eager(steps, flow_fp16, with_cache):
    decoder_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
    prepare_code2wav_graph_runtime()
    estimator = (
        decoder_dit.DiT(
            in_channels=6,
            out_channels=2,
            depth=2,
            num_heads=2,
            head_dim=8,
            hidden_size=16,
            mlp_ratio=2.0,
        )
        .npu()
        .eval()
    )
    adapter = BatchedToken2Wav(
        _Token2Wav(estimator, steps=steps),
        npu_flow_float16=flow_fp16,
    )
    graph_runner = NPUExactGraphRunner(max_graphs=8)

    def inputs(seed: int):
        def make(shape: tuple[int, ...], offset: float):
            values = torch.arange(prod(shape), device="npu", dtype=torch.float32)
            return torch.sin(values * 0.13 + seed + offset).reshape(shape)

        return (
            make((1, 2, 6), 0.0),
            make((1, 1), 0.5),
            make((1, 1, 6), 1.0),
        )

    def compute(*values):
        mu, speakers, cond = values[:3]
        cnn_cache, att_cache = (None, None) if len(values) == 3 else values[3:5]
        constants = _cfm_loop_constants(adapter, estimator, mu, 1)
        return _graphable_cfm_loop(
            adapter,
            estimator,
            constants,
            mu=mu,
            speakers=speakers,
            cond=cond,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )

    precision = "float16" if flow_fp16 else "float32"
    with torch.inference_mode(), npu_token2wav_sdpa_context(require_math=True):
        warm = inputs(23)
        if with_cache:
            _, warm_cnn, warm_att = compute(*warm)
            warm = (*warm, warm_cnn, warm_att)
        graph_runner.run("cfm_loop:first", warm, (with_cache, steps, precision), compute)

        replay = inputs(29)
        if with_cache:
            replay = (*replay, warm[3] + 0.01, warm[4] + 0.01)
        expected = compute(*replay)
        actual = graph_runner.run(
            "cfm_loop:first",
            replay,
            (with_cache, steps, precision),
            compute,
        )

    for eager, replayed in zip(expected, actual, strict=True):
        assert torch.isfinite(replayed).all()
        torch.testing.assert_close(replayed, eager, rtol=1e-3, atol=1e-3)
    assert graph_runner.stats == {"captures": 1, "failed": 0, "hits": 1}

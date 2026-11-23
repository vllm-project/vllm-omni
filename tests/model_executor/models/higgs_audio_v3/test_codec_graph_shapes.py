# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.higgs_audio_v3.full_sample_graph import (
    _STATE,
    _graph_key,
    _replay_captured,
    run_dense_sample,
)
from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import (
    HiggsAudioV3Code2WavForConditionalGeneration as Codec,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_matched_streaming_graphs_cover_every_exact_batch():
    model = SimpleNamespace(
        config=SimpleNamespace(
            codec_graph_single_max_frames=33,
            codec_graph_batch_sizes=list(range(1, 65)),
            codec_graph_frame_sizes=[8, 33],
        )
    )
    shapes = Codec._decode_graph_shapes(model)
    assert len(shapes) == 159
    assert all((b, f) in shapes for b in range(1, 65) for f in (8, 33))
    assert all((1, f) in shapes for f in range(1, 34))
    assert (2, 32) not in shapes


def test_default_graph_shapes_are_preserved():
    shapes = Codec._decode_graph_shapes(SimpleNamespace(config=SimpleNamespace()))
    assert len(shapes) == 162
    assert (1, 150) in shapes and (16, 54) in shapes


def test_invalid_graph_shape_rejected():
    with pytest.raises(ValueError):
        Codec._decode_graph_shapes(SimpleNamespace(config=SimpleNamespace(codec_graph_batch_sizes=[0])))


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("sampler", [None, False, True])
@pytest.mark.parametrize("codec", [None, False, True])
def test_graph_defaults_follow_runner_and_preserve_explicit_overrides(use_v2, sampler, codec):
    from vllm_omni.transformers_utils.configs.higgs_audio_v3 import HiggsAudioV3Config

    config = HiggsAudioV3Config(audio_full_sample_graph=sampler, codec_cuda_graph=codec)
    assert config.resolve_graph_defaults(use_v2_model_runner=use_v2) == (
        use_v2 if sampler is None else sampler,
        use_v2 if codec is None else codec,
    )
    # Resolving a stage must not bake its defaults into shared model config.
    assert config.audio_full_sample_graph is sampler
    assert config.codec_cuda_graph is codec


def test_sampler_cache_miss_does_not_capture_or_advance_rng():
    model = SimpleNamespace(_dense_sample_graphs={})
    torch.manual_seed(7)
    before = torch.random.get_rng_state()
    assert run_dense_sample(model, torch.zeros(2, 4), None, None) is None
    assert torch.equal(before, torch.random.get_rng_state())
    assert not model._dense_sample_graphs


def test_sampler_replay_returns_owned_live_rows_and_refreshes_controls(mocker):
    from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import HiggsAudioSamplingParams

    hidden = torch.ones(3, 2)
    controls = HiggsAudioSamplingParams(torch.ones(3), torch.full((3,), -1), torch.ones(3), False, {})
    static_controls = HiggsAudioSamplingParams(torch.ones(4), torch.full((4,), -1), torch.ones(4), False, {})
    output = torch.arange(4, dtype=torch.int32)[:, None]
    model = SimpleNamespace(
        config=SimpleNamespace(audio_mrv2_sample_graph_buckets=True),
        num_codebooks=2,
        _ensure_decode_state_capacity=lambda *args: None,
        _finish_audio_staging=mocker.Mock(),
        **{name: torch.zeros(4) for name in _STATE},
    )
    # A CPU replay stand-in changes graph-owned storage on each invocation.
    graph = SimpleNamespace(replay=lambda: output.add_(1))
    key = _graph_key(model, 4, hidden, True, controls)
    model._dense_sample_graphs = {
        key: (
            graph,
            torch.zeros(4, 2),
            torch.zeros(4),
            torch.zeros(8, 3),
            static_controls,
            output,
            torch.zeros(4, 4),
            [],
        )
    }

    first = _replay_captured(model, hidden, controls, True)
    controls = HiggsAudioSamplingParams(torch.tensor([0.4, 0.5, 0.6]), controls.top_k, controls.top_p, False, {})
    second = _replay_captured(model, hidden, controls, True)

    assert first.tolist() == [[1], [2], [3]]
    assert second.tolist() == [[2], [3], [4]]
    assert first.data_ptr() != output.data_ptr() and second.data_ptr() != output.data_ptr()
    torch.testing.assert_close(static_controls.temperature[:3], controls.temperature)
    assert model._finish_audio_staging.call_count == 2
    assert model._finish_audio_staging.call_args.args[0].shape == (3, 4)

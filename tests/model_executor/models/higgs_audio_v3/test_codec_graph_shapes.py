# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.higgs_audio_v3.full_sample_graph import run_dense_sample
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

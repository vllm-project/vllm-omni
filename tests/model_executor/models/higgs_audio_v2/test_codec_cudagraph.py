# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the exact-shape codec decode CUDA graphs of higgs-audio v2
(``HiggsAudioV2Code2Wav``, ported from v3's ``_capture_decode_graphs``).

Graph replay must be bit-exact against the eager decode (exact shapes, never
padded — convolution boundary conditions must stay exact), consume fresh
inputs on every replay, fall back to eager for uncovered shapes, and keep the
code-range validator firing on the serving path while staying capture-legal.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
    HiggsAudioV2Config,
)
from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_code2wav import (
    HiggsAudioV2Code2WavForConditionalGeneration as Codec,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = pytest.mark.core_model

_requires_cuda = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="needs CUDA")

NUM_CODEBOOKS = 8


# ---------------------------------------------------------------------------
# Shape table and config resolution (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_default_graph_shapes_are_batch_one_singles():
    """The v2 runtime decodes per request, so the default table is batch-1 only."""
    shapes = Codec._decode_graph_shapes(SimpleNamespace(config=SimpleNamespace()))
    assert len(shapes) == 150
    assert (1, 1) in shapes and (1, 150) in shapes
    assert all(b == 1 for b, _ in shapes)
    assert (2, 54) not in shapes


@pytest.mark.cpu
def test_configured_batch_tables_add_multi_request_shapes():
    shapes = Codec._decode_graph_shapes(
        SimpleNamespace(
            config=SimpleNamespace(
                codec_graph_single_max_frames=33,
                codec_graph_batch_sizes=[2, 4],
                codec_graph_frame_sizes=[8, 33],
            )
        )
    )
    assert (2, 8) in shapes and (4, 33) in shapes
    assert all((1, f) in shapes for f in range(1, 34))
    assert (2, 32) not in shapes


@pytest.mark.cpu
def test_invalid_graph_shape_rejected():
    with pytest.raises(ValueError):
        Codec._decode_graph_shapes(SimpleNamespace(config=SimpleNamespace(codec_graph_single_max_frames=0)))
    with pytest.raises(ValueError):
        Codec._decode_graph_shapes(SimpleNamespace(config=SimpleNamespace(codec_graph_batch_sizes=[0])))


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("codec", [None, False, True])
@pytest.mark.cpu
def test_graph_defaults_follow_runner_and_preserve_explicit_overrides(use_v2, codec):
    config = HiggsAudioV2Config(codec_cuda_graph=codec)
    assert config.resolve_graph_defaults(use_v2_model_runner=use_v2) == (use_v2 if codec is None else codec)
    # Resolving a stage must not bake its defaults into shared model config.
    assert config.codec_cuda_graph is codec


@pytest.mark.cpu
def test_validator_skips_range_check_during_capture(monkeypatch):
    """The .max()/.min() .item() pair is illegal inside graph capture."""
    model = Codec(HiggsAudioV2Config())
    model._loaded = True
    bad = torch.full((1, NUM_CODEBOOKS, 4), 999_999, dtype=torch.long)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    model._validate_codes(bad)  # must not raise mid-capture
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(ValueError, match="out-of-range"):
        model._validate_codes(bad)


@pytest.mark.cpu
def test_unloaded_model_rejects_decode():
    with pytest.raises(RuntimeError, match="not loaded"):
        Codec(HiggsAudioV2Config()).decode_codes(torch.zeros(1, NUM_CODEBOOKS, 4, dtype=torch.long))


# ---------------------------------------------------------------------------
# Graph capture / replay against the real codec (CUDA)
# ---------------------------------------------------------------------------


def _build_model(frames_cap=150):
    """Random-weight codec in the real HiggsAudioV2Code2Wav shell.

    Mirrors the geometry of ``load_higgs_audio_codec`` (RVQ hidden 1024 ->
    fc2 256 -> BosonDacDecoder); timing and bit-exactness do not depend on
    weight values.
    """
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import (
        HiggsAudioRVQ,
        adjust_conv_transpose_output_padding,
        build_boson_dac_decoder,
    )

    config = HiggsAudioV2Config(codec_cuda_graph=True, codec_graph_single_max_frames=frames_cap)
    model = Codec(config)
    torch.manual_seed(1234)
    quantizer = HiggsAudioRVQ(num_quantizers=NUM_CODEBOOKS, codebook_size=1024, codebook_dim=64, hidden_size=1024)
    import torch.nn as nn

    fc2 = nn.Linear(1024, 256)
    decoder = build_boson_dac_decoder(torch.device("cuda"))
    adjust_conv_transpose_output_padding(decoder)
    model.quantizer = quantizer.to("cuda").eval()
    model.fc2 = fc2.to("cuda").eval()
    model.acoustic_decoder = decoder.to("cuda").eval()
    model._loaded = True
    return model


@pytest.fixture(scope="module")
def graph_model():
    return _build_model()


def _codes(frames, seed=42, batch=1):
    torch.manual_seed(seed)
    return torch.randint(0, 1024, (batch, NUM_CODEBOOKS, frames), device="cuda", dtype=torch.long)


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_capture_populates_every_single_shape(graph_model):
    graph_model.capture_auxiliary_graphs()
    assert graph_model._decode_graphs_initialized
    assert set(graph_model._decode_graphs) == {(1, n) for n in range(1, 151)}


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("frames", [1, 5, 29, 54, 150])
def test_graph_replay_is_bit_exact_vs_eager(graph_model, frames):
    codes = _codes(frames)
    with torch.inference_mode():
        graph_model._codec_graph_enabled = False
        eager = graph_model.decode_codes(codes.clone())
        graph_model._codec_graph_enabled = True
        replayed = graph_model.decode_codes(codes.clone())
    assert torch.equal(eager, replayed)


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_uncovered_shape_falls_back_to_eager(graph_model):
    """frames > single_max stays eager — never padded onto a captured shape."""
    codes = _codes(151, seed=7)
    with torch.inference_mode():
        out = graph_model.decode_codes(codes)
        ref = graph_model._decode_codes_impl(codes)
    assert (1, 151) not in graph_model._decode_graphs
    assert torch.equal(out, ref)


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_graph_replay_consumes_new_inputs(graph_model):
    codes_a = _codes(29, seed=1)
    codes_b = _codes(29, seed=2)
    with torch.inference_mode():
        out_a = graph_model.decode_codes(codes_a)
        out_b = graph_model.decode_codes(codes_b)
    assert not torch.equal(out_a, out_b)
    with torch.inference_mode():
        ref_b = graph_model._decode_codes_impl(codes_b)
    assert torch.equal(out_b, ref_b)


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_runtime_forward_matches_eager(graph_model):
    """The engine path (flat codebook-major ids, CUDA) decodes identically
    with graphs on vs off, and out-of-range codes are skipped there."""
    frames = 50
    flat = _codes(frames, seed=11).reshape(-1)
    outputs = []
    for enabled in (False, True):
        graph_model._codec_graph_enabled = enabled
        out = graph_model.forward(input_ids=flat)
        pcm = out.multimodal_outputs["model_outputs"][0]
        assert pcm.numel() > 0
        outputs.append(pcm)
    assert torch.equal(outputs[0], outputs[1])

    # The runtime forward catches the validator's ValueError and emits an
    # empty wav for that request instead of failing the batch.
    bad_flat = flat.clone()
    bad_flat[0] = 999_999
    out = graph_model.forward(input_ids=bad_flat)
    assert out.multimodal_outputs["model_outputs"][0].numel() == 0

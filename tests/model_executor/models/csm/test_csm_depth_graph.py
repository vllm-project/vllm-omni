# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the CSM depth decoder's CUDA graphs.

The CPU tests run the padded fixed-buffer path that a replay reads and writes;
the graph itself needs a GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import CsmForConditionalGeneration
from vllm.config.compilation import CUDAGraphMode
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_test
from tests.model_executor.models.csm.test_csm_depth_decoder import BACKBONE_HIDDEN, _config, _inputs, _new_decoder
from vllm_omni.model_executor.models.csm.csm_depth_decoder import CsmDepthDecoder, draw_frame_uniforms
from vllm_omni.model_executor.models.csm.csm_depth_graph import CsmDepthDecoderGraphs, depth_graph_buckets

pytestmark = [pytest.mark.core_model, pytest.mark.tts]

SAMPLING = {"do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0}


@pytest.fixture(scope="module")
def decoder() -> CsmDepthDecoder:
    torch.manual_seed(0)
    return _new_decoder(CsmForConditionalGeneration(_config()).eval())


def _stage(*, max_num_seqs: int, capture_sizes: list[int] | None, enforce_eager: bool = False, mode=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE if mode is None else mode,
            cudagraph_capture_sizes=capture_sizes,
        ),
    )


def _uniforms(decoder: CsmDepthDecoder, rows: int, seed: int, device: str = "cpu") -> torch.Tensor:
    generators = [torch.Generator(device=device).manual_seed(seed + row) for row in range(rows)]
    return draw_frame_uniforms(rows, decoder.num_steps, decoder.vocab_size, device=device, generators=generators)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("max_num_seqs", "capture_sizes", "expected"),
    [
        (8, [1, 2, 4, 8, 16], [1, 2, 4, 8]),
        (6, [1, 2, 4, 8, 16], [1, 2, 4, 6]),
        (5, None, [1, 2, 4, 5]),
        (8, [], [1, 2, 4, 8]),
        (1, [1, 2], [1]),
    ],
)
def test_buckets_follow_the_capture_sizes_and_cover_every_batch(max_num_seqs, capture_sizes, expected) -> None:
    assert depth_graph_buckets(_stage(max_num_seqs=max_num_seqs, capture_sizes=capture_sizes)) == expected


@pytest.mark.cpu
def test_an_eager_stage_has_no_buckets() -> None:
    assert depth_graph_buckets(_stage(max_num_seqs=8, capture_sizes=[1, 2], enforce_eager=True)) == []
    assert depth_graph_buckets(_stage(max_num_seqs=8, capture_sizes=[], mode=CUDAGraphMode.NONE)) == []


@pytest.mark.cpu
def test_padded_greedy_frame_matches_the_decoder(decoder: CsmDepthDecoder) -> None:
    graphs = CsmDepthDecoderGraphs(decoder, buckets=[1, 2, 4])
    with torch.no_grad():
        # Leaves other rows' inputs in the buffers that pad the next frame.
        graphs.generate_frame(*_inputs(batch=4, seed=21))
        hidden, first_codes = _inputs(batch=3, seed=22)
        codes, frame_embeds = graphs.generate_frame(hidden, first_codes)
        expected_codes, expected_embeds = decoder.generate_frame(hidden, first_codes)
    assert codes.shape == (3, decoder.num_codebooks)
    assert torch.equal(codes, expected_codes)
    torch.testing.assert_close(frame_embeds, expected_embeds, rtol=0, atol=1e-5)


@pytest.mark.cpu
def test_padded_sampled_frame_matches_the_decoder_given_the_same_noise(decoder: CsmDepthDecoder) -> None:
    graphs = CsmDepthDecoderGraphs(decoder, buckets=[1, 2, 4], **SAMPLING)
    hidden, first_codes = _inputs(batch=3, seed=23)
    uniforms = _uniforms(decoder, 3, seed=5)
    with torch.no_grad():
        codes, _ = graphs.generate_frame(hidden, first_codes, uniforms=uniforms, **SAMPLING)
        expected, _ = decoder.generate_frame(hidden, first_codes, uniforms=uniforms, **SAMPLING)
        greedy, _ = decoder.generate_frame(hidden, first_codes)
    assert torch.equal(codes, expected)
    assert not torch.equal(codes, greedy)


@pytest.mark.cpu
def test_noise_keeps_float32_when_built_under_the_stage_dtype(decoder: CsmDepthDecoder) -> None:
    """vLLM loads weights, and so builds the graphs, with the stage dtype as torch's default."""
    with set_default_torch_dtype(torch.bfloat16):
        graphs = CsmDepthDecoderGraphs(decoder, buckets=[2], **SAMPLING)
    hidden, first_codes = _inputs(batch=2, seed=29)
    uniforms = _uniforms(decoder, 2, seed=7)
    # Noise this close to 1 becomes exactly 1 in bf16, which makes its token win whatever its score.
    uniforms[:, :, 0] = 1.0 - 1e-6
    with torch.no_grad():
        codes, _ = graphs.generate_frame(hidden, first_codes, uniforms=uniforms, **SAMPLING)
        expected, _ = decoder.generate_frame(hidden, first_codes, uniforms=uniforms, **SAMPLING)
    assert torch.equal(codes, expected)


@pytest.mark.cpu
def test_a_frame_is_not_changed_by_the_next_one(decoder: CsmDepthDecoder) -> None:
    graphs = CsmDepthDecoderGraphs(decoder, buckets=[2])
    with torch.no_grad():
        codes, frame_embeds = graphs.generate_frame(*_inputs(batch=2, seed=24))
        kept_codes, kept_embeds = codes.clone(), frame_embeds.clone()
        graphs.generate_frame(*_inputs(batch=2, seed=25))
    assert torch.equal(codes, kept_codes)
    assert torch.equal(frame_embeds, kept_embeds)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("batch", "sampling", "with_noise"),
    [
        pytest.param(3, SAMPLING, True, id="more-rows-than-the-largest-bucket"),
        pytest.param(2, {**SAMPLING, "temperature": 0.5}, True, id="another-temperature"),
        pytest.param(2, {"do_sample": False}, False, id="greedy-on-sampling-graphs"),
        pytest.param(2, SAMPLING, False, id="no-noise-given"),
    ],
)
def test_frames_the_graphs_did_not_record_go_to_the_decoder(
    decoder: CsmDepthDecoder, mocker, batch: int, sampling: dict, with_noise: bool
) -> None:
    graphs = CsmDepthDecoderGraphs(decoder, buckets=[1, 2], **SAMPLING)
    hidden, first_codes = _inputs(batch=batch, seed=26)
    uniforms = _uniforms(decoder, batch, seed=6) if with_noise else None
    spy = mocker.spy(decoder, "generate_frame")
    with torch.no_grad():
        codes, _ = graphs.generate_frame(hidden, first_codes, uniforms=uniforms, **sampling)

    # Called with the caller's tensors, not the padded buffers.
    assert spy.call_count == 1
    assert spy.call_args.args[0] is hidden and spy.call_args.args[1] is first_codes
    assert spy.call_args.kwargs["uniforms"] is uniforms
    assert codes.shape == (batch, decoder.num_codebooks)


@pytest.mark.cpu
def test_greedy_graphs_ignore_the_unused_sampling_values(decoder: CsmDepthDecoder, mocker) -> None:
    graphs = CsmDepthDecoderGraphs(decoder, buckets=[2], do_sample=False, temperature=0.9, top_k=50)
    hidden, first_codes = _inputs(batch=1, seed=27)
    spy = mocker.spy(decoder, "generate_frame")
    with torch.no_grad():
        graphs.generate_frame(hidden, first_codes, do_sample=False, temperature=0.7, top_k=5)
    assert spy.call_args.args[0].shape[0] == 2


@pytest.mark.cpu
def test_nothing_is_captured_off_cuda(decoder: CsmDepthDecoder) -> None:
    graphs = CsmDepthDecoderGraphs(decoder, buckets=[1, 2])
    assert graphs.capture() is False
    hidden, first_codes = _inputs(batch=2, seed=28)
    with torch.no_grad():
        codes, _ = graphs.generate_frame(hidden, first_codes)
        expected, _ = decoder.generate_frame(hidden, first_codes)
    assert torch.equal(codes, expected)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sampling", [{"do_sample": False}, SAMPLING], ids=["greedy", "sampled"])
@torch.inference_mode()
def test_replayed_frames_match_eager_frames(sampling: dict) -> None:
    torch.manual_seed(0)
    decoder = _new_decoder(CsmForConditionalGeneration(_config()).eval()).to("cuda")
    graphs = CsmDepthDecoderGraphs(decoder, buckets=[1, 2, 4], **sampling)
    assert graphs.capture() is True

    for seed, rows in enumerate((4, 1, 3, 2, 3)):
        generator = torch.Generator().manual_seed(seed)
        hidden = torch.randn(rows, BACKBONE_HIDDEN, generator=generator).to("cuda")
        first_codes = torch.randint(0, decoder.vocab_size, (rows,), generator=generator).to("cuda")
        uniforms = _uniforms(decoder, rows, seed=seed, device="cuda") if sampling["do_sample"] else None
        codes, frame_embeds = graphs.generate_frame(hidden, first_codes, uniforms=uniforms, **sampling)
        expected_codes, expected_embeds = decoder.generate_frame(hidden, first_codes, uniforms=uniforms, **sampling)
        assert torch.equal(codes, expected_codes)
        torch.testing.assert_close(frame_embeds, expected_embeds, rtol=0, atol=1e-5)

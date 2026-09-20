# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import copy

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts import audio_tokenizer_v2 as codec_module
from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MHAState,
    MossAudioTokenizerModel,
    TransformerState,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer_v2 import MossAudioTokenizerConfig

pytestmark = [pytest.mark.core_model]


def make_codec(device="cpu"):
    block = dict(
        module_type="Transformer",
        d_model=128,
        num_heads=2,
        num_layers=3,
        dim_feedforward=128,
        causal=True,
        norm="layer_norm",
        positional_embedding="rope",
        gating="none",
        context_duration=5.0,
    )
    config = MossAudioTokenizerConfig(
        sampling_rate=8,
        downsample_rate=8,
        number_channels=1,
        encoder_kwargs=[dict(module_type="PatchedPretransform", patch_size=8)],
        decoder_kwargs=[
            dict(block, input_dimension=8, output_dimension=16),
            dict(module_type="PatchedPretransform", patch_size=2),
            dict(block, input_dimension=8, output_dimension=4),
            dict(module_type="PatchedPretransform", patch_size=4),
        ],
        quantizer_kwargs=dict(input_dim=8, rvq_dim=8, output_dim=8, num_quantizers=2, codebook_size=16, codebook_dim=4),
    )
    model = MossAudioTokenizerModel(config).eval().to(device)
    if device != "cpu":
        model.decoder.to(dtype=torch.bfloat16)
    return model


def pair(device):
    torch.manual_seed(17)
    baseline = make_codec(device)
    candidate = copy.deepcopy(baseline)
    baseline.shared_decoder_kv = False
    assert candidate.shared_decoder_kv
    baseline.initialize_decoder_state_pool(4, 4)
    candidate.initialize_decoder_state_pool(4, 4)
    return baseline, candidate


def check_states(baseline, candidate):
    for left, right in zip(baseline._streaming_modules, candidate._streaming_modules):
        a, b = left._streaming_state, right._streaming_state
        if isinstance(a, MHAState):
            torch.testing.assert_close(a.offset[:4], b.offset[:4], rtol=0, atol=0)
            torch.testing.assert_close(a.kv_cache.cache[:, :4], b.kv_cache.cache[:, :4], rtol=0, atol=0)
            assert b.kv_cache.cache[:, 4].count_nonzero() == 0
        elif isinstance(a, TransformerState):
            torch.testing.assert_close(a.offsets[:4], b.offsets[:4], rtol=0, atol=0)
    assert candidate._decoder_slot_offsets[:, 4].count_nonzero() == 0


@pytest.mark.cpu
@pytest.mark.parametrize("legacy_env", [None, "0", "1"])
def test_shared_decoder_pool_is_default(monkeypatch, legacy_env):
    if legacy_env is None:
        monkeypatch.delenv("VLLM_OMNI_MOSS_CODEC_SHARED_KV", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_MOSS_CODEC_SHARED_KV", legacy_env)
    model = make_codec()
    assert model.shared_decoder_kv
    model.initialize_decoder_state_pool(4, 128)
    assert model._decoder_state_capacity == 5
    assert model._decoder_slot_offsets.shape == (2, 5)
    model.close_decoder_state_pool()


@pytest.mark.cpu
def test_pool_size_metadata_sharing_and_legacy_lifecycle():
    baseline, candidate = pair("cpu")
    assert baseline._decoder_slot_offsets.shape == (14, 8)
    assert candidate._decoder_slot_offsets.shape == (2, 5)
    for group in (candidate.decoder[0].transformer, candidate.decoder[2].transformer):
        for layer in group.layers:
            state = layer.self_attn._streaming_state
            assert state.offset is group._streaming_state.offsets
            assert state.kv_cache.end_offset is state.offset
    candidate.close_decoder_state_pool()
    # Fixed-width reference contexts must remain independent of the opt-in pool.
    with candidate.decoder_streaming(1):
        assert not candidate.decoder[0].transformer._shared_kv_metadata
    candidate.initialize_decoder_state_pool(2, 128)
    assert candidate._decoder_state_capacity == 3


@pytest.mark.cpu
def test_full_codec_wrap_padding_reorder_reset_and_oversized_chunks(mocker):
    baseline, candidate = pair("cpu")
    prepare = mocker.spy(codec_module, "prepare_streaming_attention_metadata")
    for step, frames in enumerate([1, 3, 3, 7, 1, 3, 7, 3]):
        codes = torch.randint(0, 16, (2, 4, frames))
        valid = torch.tensor([True, True, False, False]) if step != 5 else torch.zeros(4, dtype=torch.bool)
        slots = torch.tensor([1, 0, 6, 7]) if step % 2 else torch.tensor([0, 1, 6, 7])
        if step == 5:
            slots = torch.arange(4, 8)
        lengths = torch.full((4,), frames) * valid
        with torch.inference_mode():
            ref = baseline.decode_streaming_tensors(codes, lengths, slots, valid)
            prepare.reset_mock()
            out = candidate.decode_streaming_tensors(codes, lengths, slots, valid)
            assert prepare.call_count == 2  # Once per resolution, not six layers.
        torch.testing.assert_close(out[0], ref[0], rtol=0, atol=0)
        torch.testing.assert_close(out[1], ref[1], rtol=0, atol=0)
        check_states(baseline, candidate)
        if step == 3:
            for model in (baseline, candidate):
                model.reset_decoder_state_slots(torch.tensor([0]))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_null_slot_and_state_parity():
    baseline, candidate = pair("cuda")
    codes = torch.randint(0, 16, (2, 4, 3), device="cuda")
    lengths = torch.full((4,), 3, device="cuda", dtype=torch.long)
    slots = torch.tensor([0, 1, 6, 7], device="cuda")
    valid = torch.tensor([True, True, False, False], device="cuda")
    with torch.inference_mode():
        # Compile both valid and masked store paths before capture.
        for _ in range(2):
            candidate.decode_streaming_tensors(codes, lengths, slots, valid)
        candidate.reset_decoder_state_slots(torch.arange(4, device="cuda"))
        for module in candidate._streaming_modules:
            state = module._streaming_state
            if isinstance(state, MHAState):
                state.kv_cache.cache.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = candidate.decode_streaming_tensors(codes, lengths, slots, valid)
        for step in range(8):
            codes.random_(0, 16)
            slots[:2] = torch.tensor([step % 2, 1 - step % 2], device="cuda")
            if step == 5:
                slots[:2] = torch.tensor([4, 5], device="cuda")
            valid.fill_(step != 5)
            valid[2:] = False
            ref = baseline.decode_streaming_tensors(codes, lengths, slots, valid)
            graph.replay()
            torch.testing.assert_close(out[0], ref[0], rtol=0, atol=0)
            check_states(baseline, candidate)
            if step == 3:
                for model in (baseline, candidate):
                    model.reset_decoder_state_slots(torch.tensor([0], device="cuda"))


@pytest.mark.cuda
@pytest.mark.parametrize("frames", [1, 15, 240, 480])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_masked_commit_preserves_reference_scatter(frames):
    from vllm_omni.model_executor.models.moss_tts.codec_kv_state import commit_cache, commit_offsets

    pool = torch.randn(2, 5, 2, 400, 64, device="cuda", dtype=torch.bfloat16)
    before = pool.clone()
    slots = torch.tensor([2, 0, 4, 4], device="cuda")
    valid = torch.tensor([True, True, False, False], device="cuda")
    offsets = torch.tensor([397, 1234, 0, 0], device="cuda")
    rows = pool.index_select(1, slots)
    indexes = ((offsets[:, None] + torch.arange(frames, device="cuda")) % 400)[:, None, :, None]
    for kv in (0, 1):
        values = torch.randn(4, 2, frames, 64, device="cuda", dtype=torch.bfloat16)
        rows[kv].scatter_(2, indexes.expand_as(values), values)
    # Compare against the SAME scatter output, not two nondeterministic scatters.
    expected = before.clone()
    expected.index_copy_(1, slots[:2], rows[:, :2])
    commit_cache(rows, pool, slots, valid, offsets, frames)
    torch.testing.assert_close(pool, expected, rtol=0, atol=0)
    state = torch.zeros(5, device="cuda", dtype=torch.long)
    commit_offsets(offsets + frames, state, slots, valid)
    assert state[4] == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_strict_dynamic_compilation_and_wrapper_padding():
    from vllm_omni.model_executor.models.moss_tts.cuda_graph_streaming_decoder_wrapper import (
        CUDAGraphStreamingDecoderWrapper,
    )

    baseline, candidate = pair("cuda")
    compiled = torch.compile(candidate.decode_streaming_tensors, fullgraph=True, dynamic=True)
    with torch.inference_mode():
        for batch, frames in [(4, 3), (2, 1), (3, 4)]:
            codes = torch.randint(0, 16, (2, batch, frames), device="cuda")
            lengths = torch.full((batch,), frames, device="cuda", dtype=torch.long)
            slots = torch.arange(batch, device="cuda")
            valid = torch.ones(batch, device="cuda", dtype=torch.bool)
            for tensor, dims in [(codes, [1, 2]), (lengths, [0]), (slots, [0]), (valid, [0])]:
                for dim in dims:
                    torch._dynamo.mark_dynamic(tensor, dim)
            ref = baseline.decode_streaming_tensors(codes, lengths, slots, valid)
            out = compiled(codes, lengths, slots, valid)
            torch.testing.assert_close(out[0], ref[0], rtol=0.02, atol=0.002)
            torch.testing.assert_close(
                candidate._decoder_slot_offsets[0, :4], baseline.decoder[0].transformer._streaming_state.offsets[:4]
            )

        # Use the real graph input staging/capture code, without constructing
        # an unrelated vLLM engine. The decode callable above is already compiled.
        wrapper = CUDAGraphStreamingDecoderWrapper.__new__(CUDAGraphStreamingDecoderWrapper)
        wrapper.codec = candidate
        wrapper.state_capacity = 4
        wrapper.batch_sizes = [4]
        wrapper.frame_sizes = [3]
        wrapper.num_quantizers = 2
        wrapper._shared_kv = True
        wrapper._pool = None
        wrapper.graphs = {}
        wrapper._capture_with_decode(4, 3, torch.device("cuda"), compiled)
        assert wrapper.scratch_capacity == 1
        for _ in range(3):
            codes = torch.randint(0, 16, (2, 2, 3), device="cuda")
            result = wrapper.decode(codes, torch.tensor([0, 1], device="cuda"))
            assert result is not None and result[2] == 2
            assert wrapper.graphs[(4, 3)].static_state_slot_ids.tolist() == [0, 1, 4, 4]
            assert candidate._decoder_slot_offsets[:, 4].count_nonzero() == 0
            for module in candidate._streaming_modules:
                if isinstance(module._streaming_state, MHAState):
                    assert module._streaming_state.kv_cache.cache[:, 4].count_nonzero() == 0

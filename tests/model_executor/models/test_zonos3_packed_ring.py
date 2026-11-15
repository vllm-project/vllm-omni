# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed attention against a stateless whole-utterance Zonos3 oracle."""

import pytest
import torch

from tests.model_executor.models.test_zonos3_codec import small_config, spec, whole_audio, whole_sequence, whole_trunk
from vllm_omni.model_executor.models.zonos3.codec import (
    ZACDecoderNetwork,
    ZACMossLayer,
    ZACPackedCodec,
    ZACPackedRingState,
    zac_packed_ring_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("scale", [1, 2])
@torch.inference_mode()
def test_packed_ring_variable_chunks_wrap_and_padding(batch, scale):
    torch.manual_seed(91)
    layer = ZACMossLayer(spec(context=7)).eval()
    max_chunk = 5 * scale
    ring = layer.context - 1 + max_chunk
    padding = 3
    storage = batch * ring + batch * max_chunk + padding
    state = ZACPackedRingState(
        torch.zeros(storage, layer.heads, layer.head_dim),
        torch.zeros(storage, layer.heads, layer.head_dim),
        batch,
        ring,
        max_chunk,
    )
    inputs = [torch.randn(1, 64 * scale, 16) for _ in range(batch)]
    expected = [whole_sequence(layer, x)[0] for x in inputs]
    cumulative = torch.zeros(batch, dtype=torch.long)
    progress = [0] * batch
    projection_rows = []
    hook = layer.qkv.register_forward_pre_hook(lambda _, args: projection_rows.append(args[0].shape[0]))
    try:
        for step in range(12):
            order = list(reversed(range(batch))) if step % 2 else list(range(batch))
            lengths = [((i + step) % 5 + 1) if not (step % 3 == 0 and i % 2) else 0 for i in order]
            offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
            pieces = [inputs[i][0, progress[i] * scale : (progress[i] + n) * scale] for i, n in zip(order, lengths)]
            packed = torch.cat([*pieces, torch.randn(padding, 16)])
            before_k, before_v = state.keys[: batch * ring].clone(), state.values[: batch * ring].clone()
            actual = zac_packed_ring_attention(
                layer,
                packed,
                torch.tensor(order),
                offsets,
                cumulative,
                state,
                scale=scale,
            )
            # Change only padding values at identical GEMM shape: live KV must
            # be bitwise unchanged, including the slots adjacent to scratch.
            clean = ZACPackedRingState(
                torch.cat((before_k, torch.zeros_like(state.keys[batch * ring :]))),
                torch.cat((before_v, torch.zeros_like(state.values[batch * ring :]))),
                batch,
                ring,
                max_chunk,
            )
            changed_padding = packed.clone()
            changed_padding[-padding:] = torch.randn_like(changed_padding[-padding:]) * 100
            zac_packed_ring_attention(
                layer, changed_padding, torch.tensor(order), offsets, cumulative, clean, scale=scale
            )
            torch.testing.assert_close(state.keys[: batch * ring], clean.keys[: batch * ring], atol=0, rtol=0)
            torch.testing.assert_close(state.values[: batch * ring], clean.values[: batch * ring], atol=0, rtol=0)
            cursor = 0
            for i, n in zip(order, lengths):
                count = n * scale
                torch.testing.assert_close(
                    actual[cursor : cursor + count],
                    expected[i][progress[i] * scale : (progress[i] + n) * scale],
                    atol=2e-6,
                    rtol=2e-5,
                )
                if n == 0:
                    torch.testing.assert_close(
                        state.keys[i * ring : (i + 1) * ring], before_k[i * ring : (i + 1) * ring]
                    )
                    torch.testing.assert_close(
                        state.values[i * ring : (i + 1) * ring], before_v[i * ring : (i + 1) * ring]
                    )
                progress[i] += n
                cursor += count
            assert not actual[cursor:].any()
            assert projection_rows[-2] == packed.shape[0]
            cumulative.copy_(torch.tensor(progress))
        assert len(projection_rows) == 24
        assert max(progress) * scale > 2 * ring
    finally:
        hook.remove()


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_packed_neural_variable_lengths_matches_whole_trunk(batch):
    torch.manual_seed(93)
    model = ZACDecoderNetwork(small_config()).eval()
    state = model.allocate_packed_state(batch, 5, batch * 5 + 3)
    codes = [torch.randint(0, 32, (40, 10)) for _ in range(batch)]
    for sequence in codes:
        sequence[:3, 4:] = -1
    expected = [whole_trunk(model, sequence[None]) for sequence in codes]
    progress = [0] * batch
    for step in range(10):
        order = list(reversed(range(batch)))
        lengths = [(i + step) % 6 for i in order]
        offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
        packed = torch.cat(
            [
                *(codes[i][progress[i] : progress[i] + n] for i, n in zip(order, lengths)),
                torch.zeros(3, 10, dtype=torch.long),
            ]
        )
        result = model.forward_packed(packed, torch.tensor(order), offsets, state)
        cursor = 0
        for i, n in zip(order, lengths):
            for actual, oracle in zip(result, expected[i]):
                torch.testing.assert_close(
                    actual[cursor * 2 : (cursor + n) * 2],
                    oracle[0, progress[i] * 2 : (progress[i] + n) * 2],
                    atol=3e-6,
                    rtol=3e-5,
                )
            progress[i] += n
            cursor += n
        assert state.frames.tolist() == progress


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_complete_packed_pcm_matches_whole_audio(batch):
    torch.manual_seed(97)
    network = ZACDecoderNetwork(small_config()).eval()
    codec = ZACPackedCodec(network, batch, 5, batch * 5 + 3)
    codes = [torch.randint(0, 32, (17 + i % 4, 10)) for i in range(batch)]
    for sequence in codes:
        sequence[:2, 4:] = -1
    expected = [whole_audio(network, sequence) for sequence in codes]
    progress = [0] * batch
    outputs = [[] for _ in range(batch)]
    for step in range(16):
        lengths = [min((i + step) % 6, len(codes[i]) - progress[i]) for i in range(batch)]
        offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
        packed = torch.cat(
            [
                *(codes[i][progress[i] : progress[i] + n] for i, n in enumerate(lengths)),
                torch.zeros(batch * 5 + 3 - sum(lengths), 10, dtype=torch.long),
            ]
        )
        final = torch.tensor(
            [step == 15 or (i % 2 == 0 and progress[i] + lengths[i] == len(codes[i])) for i in range(batch)]
        )
        audio, pcm_offsets, _ = codec(packed, torch.arange(batch), offsets, final)
        for i, n in enumerate(lengths):
            outputs[i].append(audio[pcm_offsets[i] : pcm_offsets[i + 1]].clone())
            progress[i] += n
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(torch.cat(actual), reference, atol=2e-7, rtol=3e-4)


@torch.inference_mode()
def test_packed_manager_lifecycle_crop_empty_final_cancel_reuse_and_ownership():
    from vllm_omni.model_executor.models.zonos3.codec import CodecManager, ZACCodecChunk

    torch.manual_seed(71)
    network = ZACDecoderNetwork(small_config()).eval()
    packed = CodecManager(network, 2, max_chunk=4)
    codes = torch.randint(0, network.config.codebook_size, (8, 10))
    held = []
    outputs = {"b": [], "c": []}

    def decode(chunks):
        result = packed.decode(chunks)
        assert [r.request_id for r in result] == [c.request_id for c in chunks]
        for out in result:
            held.append((out.audio, out.audio.clone()))
            if out.request_id in outputs:
                outputs[out.request_id].append(out.audio)

    decode([ZACCodecChunk("a", 0, 0, codes[:2], skip_samples=4), ZACCodecChunk("b", 0, 0, codes[:1])])
    slot = packed._streams["a"].slot
    packed.release("a")
    decode([ZACCodecChunk("c", 0, 0, codes[2:6]), ZACCodecChunk("b", 1, 1, codes[1:3])])
    assert packed._streams["c"].slot == slot
    decode([ZACCodecChunk("c", 1, 4, codes[:0], True)])
    decode([ZACCodecChunk("b", 2, 3, codes[:0], True)])
    for name, data in (("b", codes[:3]), ("c", codes[2:6])):
        torch.testing.assert_close(torch.cat(outputs[name]), whole_audio(network, data), atol=1e-6, rtol=1e-3)
        packed.release(name)
    assert not packed._streams and not packed._finished and len(packed._free) == 2
    final = packed.decode([ZACCodecChunk("a", 0, 0, codes[:2], True, 4)])[0]
    torch.testing.assert_close(final.audio, whole_audio(network, codes[:2])[4:], atol=1e-6, rtol=1e-3)
    for audio, snapshot in held:
        torch.testing.assert_close(audio, snapshot, atol=0, rtol=0)

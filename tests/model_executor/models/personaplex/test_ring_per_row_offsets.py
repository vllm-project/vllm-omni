# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture

from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
    PersonaPlexMimiCodec,
    _MimiStreamingTransformer,
    _StreamConv1d,
    _StreamConvTr1d,
)
from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
    PersonaPlexTemporalStreaming,
    _RingKV,
)

pytestmark = pytest.mark.core_model

SEED = 1234
CUDA_DEVICE = torch.device("cuda")


def _mask(*rows: bool, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bool, device=device)


def _inputs(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(device)


def _make_mimi_transformer(
    batch_size: int,
    device: torch.device,
    context: int = 6,
) -> _MimiStreamingTransformer:
    torch.manual_seed(SEED)
    transformer = _MimiStreamingTransformer(
        num_layers=2,
        dim=32,
        num_heads=4,
        context=context,
    ).to(device)
    for parameter in transformer.parameters():
        if parameter.dtype.is_floating_point:
            nn.init.normal_(parameter, std=0.1)
    transformer.streaming_init(batch_size)
    return transformer


def _make_temporal(
    batch_size: int,
    device: torch.device,
    context: int = 6,
) -> PersonaPlexTemporalStreaming:
    torch.manual_seed(SEED)
    temporal = PersonaPlexTemporalStreaming(
        dim=16,
        num_layers=2,
        num_heads=4,
        hidden=32,
        context=context,
        text_card=11,
    ).to(device)
    for parameter in temporal.parameters():
        if parameter.dtype.is_floating_point:
            nn.init.normal_(parameter, std=0.1)
    temporal.streaming_init(batch_size)
    return temporal


def _assert_valid_ring_row_matches(
    batched: _RingKV,
    reference: _RingKV,
    batched_positions: torch.Tensor,
    reference_positions: torch.Tensor,
    row: int,
) -> None:
    assert torch.equal(batched_positions[row], reference_positions)
    assert torch.equal(batched.end_offset[row], reference.end_offset[0])

    # The active mask covers the physical write as well as offset advancement,
    # so the complete per-row ring state remains singleton-identical.
    assert torch.equal(batched.cache[:, row], reference.cache[:, 0])


def _stream_carry(stream: _StreamConv1d | _StreamConvTr1d) -> torch.Tensor:
    if isinstance(stream, _StreamConv1d):
        assert stream.prev is not None
        return stream.prev
    assert stream.partial is not None
    return stream.partial


def _make_passthrough_stage(mocker: MockerFixture):
    stage = mocker.Mock(spec=["reset", "reset_slot"])
    stage.side_effect = lambda x, active: x
    return stage


def _make_passthrough_transformer(mocker: MockerFixture):
    transformer = mocker.Mock(spec=["streaming_init", "reset_streaming", "reset_slot", "step"])
    transformer.step.side_effect = lambda x, active: x
    return transformer


def _active_args(call_args_list) -> torch.Tensor:
    return torch.stack([call.args[1] for call in call_args_list])


def _make_codec_stub(mocker: MockerFixture) -> PersonaPlexMimiCodec:
    codec = PersonaPlexMimiCodec.__new__(PersonaPlexMimiCodec)
    nn.Module.__init__(codec)

    quantizer = mocker.Mock(spec=["encode", "decode"])
    quantizer.encode.side_effect = lambda x: torch.zeros(8, x.shape[0], x.shape[-1], dtype=torch.long)
    quantizer.decode.side_effect = lambda codes: torch.zeros(codes.shape[0], 1, codes.shape[-1])

    codec.device = torch.device("cpu")
    codec.dtype = torch.float32
    codec.model = SimpleNamespace(quantizer=quantizer)
    codec._enc_stages = []
    codec._dec_stages = []
    codec._downsample = _make_passthrough_stage(mocker)
    codec._upsample = _make_passthrough_stage(mocker)
    codec.encoder_transformer = _make_passthrough_transformer(mocker)
    codec.decoder_transformer = _make_passthrough_transformer(mocker)
    codec.streaming_init(batch_size=2)
    return codec


@pytest.mark.cpu
def test_codec_entrypoints_forward_active_mask(mocker: MockerFixture) -> None:
    codec = _make_codec_stub(mocker)
    active = _mask(True, False)
    all_active = torch.ones_like(active)

    codec.encode_frame(torch.zeros(2, 1920), active)
    codec.decode_frame(torch.zeros(2, 8), active)
    codec.decode_frames(torch.zeros(2, 8, 3), active)
    codec.decode_frames(torch.zeros(2, 8, 3), active=None)

    assert torch.equal(_active_args(codec._downsample.call_args_list), active[None])
    assert torch.equal(_active_args(codec._upsample.call_args_list), torch.stack((active, active, all_active)))
    assert torch.equal(_active_args(codec.encoder_transformer.step.call_args_list), active[None])
    assert torch.equal(
        _active_args(codec.decoder_transformer.step.call_args_list),
        torch.stack((active, active, all_active)),
    )


@pytest.mark.cpu
def test_ring_mixed_offsets_match_singleton_streams() -> None:
    batch_size, heads, head_dim, capacity, tokens = 3, 2, 3, 6, 2
    batched = _RingKV(batch_size, heads, head_dim, capacity, torch.device("cpu"), torch.float32)
    singletons = [_RingKV(1, heads, head_dim, capacity, torch.device("cpu"), torch.float32) for _ in range(batch_size)]
    reference_positions: list[torch.Tensor | None] = [None] * batch_size
    active_schedule = [
        _mask(True, True, True),
        _mask(True, False, False),
        _mask(False, False, True),
        _mask(True, True, False),
        _mask(False, True, True),
        _mask(True, False, True),
        _mask(True, True, True),
    ] * 2

    torch.manual_seed(SEED)
    for active in active_schedule:
        keys = torch.randn(batch_size, heads, tokens, head_dim)
        values = torch.randn_like(keys)
        _, _, positions = batched.complete(keys, values, active)

        for row, is_active in enumerate(active.tolist()):
            if is_active:
                _, _, singleton_positions = singletons[row].complete(
                    keys[row : row + 1],
                    values[row : row + 1],
                    _mask(True),
                )
                reference_positions[row] = singleton_positions[0].clone()

            assert reference_positions[row] is not None
            _assert_valid_ring_row_matches(
                batched,
                singletons[row],
                positions,
                reference_positions[row],
                row,
            )

    assert any(int(offset) > capacity for offset in batched.end_offset)
    assert not torch.equal(batched.end_offset[0], batched.end_offset[1])


@pytest.mark.cpu
def test_ring_slot_recycle_masks_old_history() -> None:
    ring = _RingKV(2, 1, 1, 8, torch.device("cpu"), torch.float32)
    for _ in range(4):
        ring.complete(torch.randn(2, 1, 2, 1), torch.randn(2, 1, 2, 1), _mask(True, True))

    old_end = ring.end_offset.clone()
    ring.reset_slot(1)
    _, _, positions = ring.complete(
        torch.randn(2, 1, 2, 1),
        torch.randn(2, 1, 2, 1),
        _mask(False, True),
    )
    visible = positions[1] >= 0
    assert torch.all(positions[1][visible] >= old_end[1])

    ring.reset_slot(0)
    ring.bump_slot_start(0)
    _, _, positions = ring.complete(
        torch.randn(2, 1, 2, 1),
        torch.randn(2, 1, 2, 1),
        _mask(True, False),
    )
    visible = positions[0] >= 0
    assert torch.all(positions[0][visible] >= old_end[0] + 1)


@pytest.mark.cpu
def test_temporal_streaming_rejects_invalid_active_shape() -> None:
    temporal = _make_temporal(2, torch.device("cpu"))

    with pytest.raises(ValueError, match=r"active must have shape \(2,\)"):
        temporal.step(torch.zeros(2, 1, 16), torch.ones(1))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mimi_transformer_mixed_offsets_match_singletons() -> None:
    batch_size, tokens, dim = 3, 2, 32
    batched = _make_mimi_transformer(batch_size, CUDA_DEVICE)
    singletons = [_make_mimi_transformer(1, CUDA_DEVICE) for _ in range(batch_size)]
    active_schedule = [
        _mask(True, True, True, device=CUDA_DEVICE),
        _mask(True, False, False, device=CUDA_DEVICE),
        _mask(False, False, True, device=CUDA_DEVICE),
        _mask(True, True, False, device=CUDA_DEVICE),
        _mask(False, True, True, device=CUDA_DEVICE),
        _mask(True, False, True, device=CUDA_DEVICE),
        _mask(True, True, True, device=CUDA_DEVICE),
    ] * 2

    for step, active in enumerate(active_schedule):
        inputs = _inputs((batch_size, tokens, dim), SEED + step, CUDA_DEVICE)
        offsets_before = batched._offset.clone()
        cache_before = [kv.cache.clone() for kv in batched._kv]
        output = batched.step(inputs, active)

        for row, is_active in enumerate(active.tolist()):
            if is_active:
                expected = singletons[row].step(
                    inputs[row : row + 1],
                    _mask(True, device=CUDA_DEVICE),
                )
                torch.testing.assert_close(output[row : row + 1], expected, rtol=0.0, atol=0.0)
                assert torch.equal(batched._offset[row], singletons[row]._offset[0])
                for batched_kv, singleton_kv in zip(batched._kv, singletons[row]._kv):
                    assert torch.equal(batched_kv.end_offset[row], singleton_kv.end_offset[0])
                    torch.testing.assert_close(batched_kv.cache[:, row], singleton_kv.cache[:, 0], rtol=0.0, atol=0.0)
            else:
                assert torch.equal(batched._offset[row], offsets_before[row])
                for kv, previous_cache in zip(batched._kv, cache_before):
                    assert torch.equal(kv.end_offset[row], offsets_before[row])
                    assert torch.equal(kv.cache[:, row], previous_cache[:, row])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_temporal_streaming_mixed_offsets_match_isolated_rows() -> None:
    # Keep the reference shape at B=2 so exact comparisons use the same CUDA
    # kernel shape while the target row runs independently.
    batched = _make_temporal(2, CUDA_DEVICE)
    references = [_make_temporal(2, CUDA_DEVICE) for _ in range(2)]
    active_schedule = [
        _mask(True, True, device=CUDA_DEVICE),
        _mask(True, False, device=CUDA_DEVICE),
        _mask(False, False, device=CUDA_DEVICE),
        _mask(False, True, device=CUDA_DEVICE),
        _mask(True, True, device=CUDA_DEVICE),
        _mask(True, False, device=CUDA_DEVICE),
        _mask(False, True, device=CUDA_DEVICE),
        _mask(True, True, device=CUDA_DEVICE),
    ] * 2

    for step, active in enumerate(active_schedule):
        inputs = _inputs((2, 1, 16), SEED + 100 + step, CUDA_DEVICE)
        offsets_before = batched._offset.clone()
        cache_before = [kv.cache.clone() for kv in batched._kv]
        output, logits = batched.step(inputs, active)

        for row, (reference, is_active) in enumerate(zip(references, active.tolist())):
            if is_active:
                reference_inputs = torch.zeros_like(inputs)
                reference_inputs[row] = inputs[row]
                reference_active = torch.zeros_like(active)
                reference_active[row] = True
                expected_output, expected_logits = reference.step(reference_inputs, reference_active)
                torch.testing.assert_close(output[row : row + 1], expected_output[row : row + 1], rtol=0.0, atol=0.0)
                torch.testing.assert_close(logits[row : row + 1], expected_logits[row : row + 1], rtol=0.0, atol=0.0)
                assert torch.equal(batched._offset[row], reference._offset[row])
                for batched_kv, reference_kv in zip(batched._kv, reference._kv):
                    assert torch.equal(batched_kv.end_offset[row], reference_kv.end_offset[row])
                    torch.testing.assert_close(batched_kv.cache[:, row], reference_kv.cache[:, row], rtol=0.0, atol=0.0)
            else:
                assert torch.equal(batched._offset[row], offsets_before[row])
                for kv, previous_cache in zip(batched._kv, cache_before):
                    assert torch.equal(kv.end_offset[row], offsets_before[row])
                    assert torch.equal(kv.cache[:, row], previous_cache[:, row])


@pytest.mark.parametrize("kind", ["conv", "convtr"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mimi_conv_carries_preserve_inactive_rows(kind: str) -> None:
    if kind == "conv":
        conv = nn.Conv1d(1, 2, kernel_size=3, stride=2, device=CUDA_DEVICE)
        batched = _StreamConv1d(conv, pad_mode="replicate")
        row0 = _StreamConv1d(conv, pad_mode="replicate")
        row1 = _StreamConv1d(conv, pad_mode="replicate")
        samples = 4
    else:
        conv = nn.ConvTranspose1d(1, 2, kernel_size=4, stride=2, device=CUDA_DEVICE)
        batched = _StreamConvTr1d(conv)
        row0 = _StreamConvTr1d(conv)
        row1 = _StreamConvTr1d(conv)
        samples = 2

    for stream, batch_size in ((batched, 2), (row0, 1), (row1, 1)):
        stream.reset(batch_size, CUDA_DEVICE, torch.float32)

    singletons = [row0, row1]
    active_schedule = [
        _mask(False, False, device=CUDA_DEVICE),
        _mask(True, False, device=CUDA_DEVICE),
        _mask(True, True, device=CUDA_DEVICE),
        _mask(False, True, device=CUDA_DEVICE),
        _mask(True, True, device=CUDA_DEVICE),
    ]

    for step, active in enumerate(active_schedule):
        inputs = _inputs((2, 1, samples), SEED + 200 + step, CUDA_DEVICE)
        carry_before = _stream_carry(batched).clone()
        fresh_before = batched._fresh.clone()
        output = PersonaPlexMimiCodec._run_stages(inputs, [(kind, batched)], active)

        for row, is_active in enumerate(active.tolist()):
            if is_active:
                expected = PersonaPlexMimiCodec._run_stages(
                    inputs[row : row + 1],
                    [(kind, singletons[row])],
                    _mask(True, device=CUDA_DEVICE),
                )
                torch.testing.assert_close(output[row : row + 1], expected, rtol=0.0, atol=0.0)
                torch.testing.assert_close(
                    _stream_carry(batched)[row], _stream_carry(singletons[row])[0], rtol=0.0, atol=0.0
                )
                assert torch.equal(batched._fresh[row], singletons[row]._fresh[0])
            else:
                assert torch.equal(_stream_carry(batched)[row], carry_before[row])
                assert torch.equal(batched._fresh[row], fresh_before[row])

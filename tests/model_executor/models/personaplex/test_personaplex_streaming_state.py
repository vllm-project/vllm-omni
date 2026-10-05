# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture

from vllm_omni.model_executor.models.moss_tts.cuda_graph_streaming_decoder_wrapper import (
    CUDAGraphStreamingDecoderWrapper,
)
from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
    FRAME_SIZE,
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
    stage = mocker.Mock(spec=["reset", "reset_all", "reset_slot", "reset_slots"])
    stage.side_effect = lambda x, active: x
    return stage


def _make_passthrough_transformer(mocker: MockerFixture):
    transformer = mocker.Mock(spec=["streaming_init", "reset_streaming", "reset_slot", "reset_slots", "step"])
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
    codec._batch_size = None
    codec._downsample = _make_passthrough_stage(mocker)
    codec._upsample = _make_passthrough_stage(mocker)
    codec.encoder_transformer = _make_passthrough_transformer(mocker)
    codec.decoder_transformer = _make_passthrough_transformer(mocker)
    codec.streaming_init(batch_size=2)
    return codec


class _SyntheticStreamingGraphCodec(nn.Module):
    def __init__(self, state_capacity: int, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(state_capacity, device=device))

    def reset_decoder_state_slots(self, state_slot_ids: torch.Tensor) -> None:
        self.state.index_fill_(0, state_slot_ids, 0)

    def decode_streaming_tensors(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.state.index_add_(0, state_slot_ids, valid_rows.to(self.state.dtype))
        audio = codes[0].to(torch.float32).repeat_interleave(4, dim=1)
        return audio, codes_lengths * 4


@pytest.mark.cpu
def test_mimi_codec_entrypoints_forward_active_mask(mocker: MockerFixture) -> None:
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


def test_mimi_streaming_tensor_decode_pads_lengths_and_masks_invalid_rows(mocker: MockerFixture) -> None:
    codec = _make_codec_stub(mocker)
    codec.streaming_init(batch_size=3)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def decode_frame(
        codes: torch.Tensor,
        active: torch.Tensor,
        state_slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        del codes
        calls.append((active.clone(), state_slot_ids.clone()))
        values = (state_slot_ids + 1).to(torch.float32).view(-1, 1)
        return values.expand(-1, FRAME_SIZE)

    mocker.patch.object(codec, "decode_frame", side_effect=decode_frame)
    codes = torch.zeros(8, 3, 3, dtype=torch.long)
    codes_lengths = torch.tensor([1, 3, 2], dtype=torch.long)
    state_slot_ids = torch.tensor([2, 0, 1], dtype=torch.long)
    valid_rows = torch.tensor([True, False, True], dtype=torch.bool)

    audio, audio_lengths = codec.decode_streaming_tensors(
        codes,
        codes_lengths,
        state_slot_ids,
        valid_rows,
    )

    assert audio.shape == (3, 3 * FRAME_SIZE)
    expected = torch.zeros_like(audio)
    expected[0, :FRAME_SIZE] = 3
    expected[2, : 2 * FRAME_SIZE] = 2
    torch.testing.assert_close(audio, expected)
    assert torch.equal(audio_lengths, torch.tensor([1920, 0, 3840]))
    assert [active.tolist() for active, _ in calls] == [
        [True, False, True],
        [False, False, True],
        [False, False, False],
    ]
    assert all(torch.equal(slots, state_slot_ids) for _, slots in calls)


@pytest.mark.cpu
def test_mimi_codec_separates_physical_and_logical_state_capacity(mocker: MockerFixture) -> None:
    codec = _make_codec_stub(mocker)

    codec.streaming_init(batch_size=4, state_capacity=2)

    assert codec._batch_size == 4
    assert codec._state_capacity == 2
    assert codec._all_active.shape == (4,)
    scratch_slots = torch.tensor([2, 3], dtype=torch.long)
    codec.reset_decoder_state_slots(scratch_slots)
    codec.encoder_transformer.reset_slots.assert_called_once_with(scratch_slots)
    codec.decoder_transformer.reset_slots.assert_called_once_with(scratch_slots)

    with pytest.raises(ValueError, match=r"state_slot_ids must be in \[0, 4\)"):
        codec.reset_decoder_state_slots(torch.tensor([4], dtype=torch.long))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streaming_graph_variable_lengths_and_scratch_rows_cuda() -> None:
    device = CUDA_DEVICE
    codec = _SyntheticStreamingGraphCodec(state_capacity=4, device=device)
    wrapper = CUDAGraphStreamingDecoderWrapper.__new__(CUDAGraphStreamingDecoderWrapper)
    wrapper.codec = codec
    wrapper.state_capacity = 2
    wrapper.batch_sizes = [2]
    wrapper.frame_sizes = [5]
    wrapper.num_quantizers = 2
    wrapper.graphs = {}
    wrapper._pool = torch.cuda.graph_pool_handle()
    wrapper._warmed_up = False

    wrapper._capture_with_decode(2, 5, device, codec.decode_streaming_tensors)

    codes = torch.arange(6, device=device, dtype=torch.long).view(2, 1, 3)
    lengths = torch.tensor([2], device=device, dtype=torch.long)
    state_slots = torch.tensor([0], device=device, dtype=torch.long)
    valid_rows = torch.ones(1, device=device, dtype=torch.bool)
    graph_decode = wrapper.decode(
        codes,
        state_slots,
        codes_lengths=lengths,
        valid_rows=valid_rows,
        allow_frame_padding=True,
    )
    assert graph_decode is not None
    graph_audio, graph_lengths, actual_batch = graph_decode
    assert actual_batch == 1
    assert torch.equal(graph_lengths[:1], lengths * 4)

    codec.reset_decoder_state_slots(torch.arange(4, device=device, dtype=torch.long))
    eager_audio, eager_lengths = codec.decode_streaming_tensors(
        codes,
        lengths,
        state_slots,
        valid_rows,
    )
    torch.testing.assert_close(graph_audio[:1, : eager_audio.shape[1]], eager_audio, rtol=0.0, atol=0.0)
    assert torch.equal(graph_lengths[:1], eager_lengths)
    assert torch.equal(codec.state, torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))


@pytest.mark.cpu
def test_mimi_codec_same_batch_streaming_init_reuses_state(mocker: MockerFixture) -> None:
    codec = _make_codec_stub(mocker)
    codec._enc_stages = [
        (
            "conv",
            _StreamConv1d(nn.Conv1d(1, 2, kernel_size=3, stride=2), pad_mode="replicate"),
        )
    ]
    codec._dec_stages = [
        (
            "convtr",
            _StreamConvTr1d(nn.ConvTranspose1d(1, 2, kernel_size=4, stride=2)),
        )
    ]
    codec._downsample = _StreamConv1d(nn.Conv1d(1, 2, kernel_size=3, stride=2), pad_mode="replicate")
    codec._upsample = _StreamConvTr1d(nn.ConvTranspose1d(1, 2, kernel_size=4, stride=2))
    codec._batch_size = None

    codec.streaming_init(2)
    states = list(codec._conv_states())
    pointers = tuple(
        pointer
        for state in states
        for pointer in (
            _stream_carry(state).data_ptr(),
            state._fresh.data_ptr(),
        )
    )
    all_active_pointer = codec._all_active.data_ptr()
    for state in states:
        _stream_carry(state).fill_(1.0)
        state._fresh.fill_(False)

    codec.streaming_init(2)

    assert codec._all_active.data_ptr() == all_active_pointer
    assert (
        tuple(pointer for state in states for pointer in (_stream_carry(state).data_ptr(), state._fresh.data_ptr()))
        == pointers
    )
    for state in states:
        assert torch.count_nonzero(_stream_carry(state)) == 0
        assert torch.all(state._fresh)


@pytest.mark.cpu
def test_mimi_transformer_same_batch_streaming_init_reuses_ring_state() -> None:
    transformer = _make_mimi_transformer(2, torch.device("cpu"))
    transformer.step(torch.randn(2, 2, 32), _mask(True, True))
    offset_pointer = transformer._offset.data_ptr()
    ring_pointers = tuple(
        pointer
        for kv in transformer._kv
        for pointer in (kv.cache.data_ptr(), kv.end_offset.data_ptr(), kv.start_offset.data_ptr())
    )

    transformer.streaming_init(2)

    assert transformer._offset.data_ptr() == offset_pointer
    assert (
        tuple(
            pointer
            for kv in transformer._kv
            for pointer in (kv.cache.data_ptr(), kv.end_offset.data_ptr(), kv.start_offset.data_ptr())
        )
        == ring_pointers
    )
    assert torch.count_nonzero(transformer._offset) == 0
    for kv in transformer._kv:
        assert torch.count_nonzero(kv.end_offset) == 0
        assert torch.count_nonzero(kv.start_offset) == 0


def test_mimi_transformer_state_slots_match_singleton_streams() -> None:
    device = torch.device("cpu")
    pooled = _make_mimi_transformer(3, device)
    references = [_make_mimi_transformer(1, device) for _ in range(3)]

    schedules = [
        (torch.tensor([2, 0]), _inputs((2, 1, 32), 10, device), _mask(True, True)),
        (torch.tensor([1, 2]), _inputs((2, 1, 32), 11, device), _mask(True, False)),
    ]
    for slots, inputs, active in schedules:
        output = pooled.step(inputs, active, state_slot_ids=slots)
        for row, (slot, is_active) in enumerate(zip(slots.tolist(), active.tolist())):
            if is_active:
                expected = references[slot].step(inputs[row : row + 1], _mask(True))
                torch.testing.assert_close(output[row : row + 1], expected, rtol=1e-5, atol=1e-6)
            else:
                expected_offset = references[slot]._offset.clone()
                torch.testing.assert_close(pooled._offset[slot], expected_offset[0], rtol=1e-5, atol=1e-6)

    for slot, reference in enumerate(references):
        assert torch.equal(pooled._offset[slot], reference._offset[0])
        for pooled_kv, reference_kv in zip(pooled._kv, reference._kv):
            torch.testing.assert_close(
                pooled_kv.cache[:, slot],
                reference_kv.cache[:, 0],
                rtol=1e-5,
                atol=1e-6,
            )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mimi_transformer_state_slots_match_singletons_cuda() -> None:
    pooled = _make_mimi_transformer(3, CUDA_DEVICE)
    references = [_make_mimi_transformer(2, CUDA_DEVICE) for _ in range(3)]
    reference_active = _mask(True, False, device=CUDA_DEVICE)
    schedules = [
        (
            torch.tensor([2, 0], device=CUDA_DEVICE),
            _inputs((2, 1, 32), 10, CUDA_DEVICE),
            _mask(True, True, device=CUDA_DEVICE),
        ),
        (
            torch.tensor([1, 2], device=CUDA_DEVICE),
            _inputs((2, 1, 32), 11, CUDA_DEVICE),
            _mask(True, False, device=CUDA_DEVICE),
        ),
    ]

    for slots, inputs, active in schedules:
        output = pooled.step(inputs, active, state_slot_ids=slots)
        for row, (slot, is_active) in enumerate(zip(slots.tolist(), active.tolist())):
            if is_active:
                reference_inputs = torch.zeros_like(inputs)
                reference_inputs[0].copy_(inputs[row])
                expected = references[slot].step(reference_inputs, reference_active)
                torch.testing.assert_close(output[row : row + 1], expected[:1], rtol=0.0, atol=0.0)
            else:
                assert torch.equal(pooled._offset[slot], references[slot]._offset[0])

    for slot, reference in enumerate(references):
        assert torch.equal(pooled._offset[slot], reference._offset[0])
        for pooled_kv, reference_kv in zip(pooled._kv, reference._kv):
            assert torch.equal(pooled_kv.cache[:, slot], reference_kv.cache[:, 0])


@pytest.mark.cpu
def test_mimi_codec_active_requires_bool_contiguous_mask(mocker: MockerFixture) -> None:
    codec = _make_codec_stub(mocker)

    with pytest.raises(TypeError, match="active must have dtype torch.bool"):
        codec.encode_frame(torch.zeros(2, 1920), torch.ones(2))

    non_contiguous = torch.ones((2, 2), dtype=torch.bool)[:, 0]
    assert not non_contiguous.is_contiguous()
    with pytest.raises(ValueError, match="active must be contiguous"):
        codec.encode_frame(torch.zeros(2, 1920), non_contiguous)


@pytest.mark.cpu
def test_mimi_codec_single_frame_decode_matches_multiframe_regression(mocker: MockerFixture) -> None:
    codec = _make_codec_stub(mocker)
    codec.model.quantizer.decode.side_effect = lambda codes: codes.to(torch.float32).sum(dim=1, keepdim=True)
    codes = torch.arange(16, dtype=torch.long).view(2, 8)

    single_frame = codec.decode_frame(codes)
    one_frame_batch = codec.decode_frames(codes.unsqueeze(-1))

    torch.testing.assert_close(single_frame, one_frame_batch, rtol=0.0, atol=0.0)


@pytest.mark.cpu
def test_ring_kv_mixed_offsets_match_singleton_streams() -> None:
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
def test_ring_kv_slot_recycle_masks_old_history() -> None:
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
def test_personaplex_temporal_rejects_invalid_active_shape() -> None:
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
def test_personaplex_temporal_mixed_offsets_match_isolated_rows() -> None:
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


def _assert_mimi_conv_carries_preserve_inactive_rows(kind: str, device: torch.device) -> None:
    if kind == "conv":
        conv = nn.Conv1d(1, 2, kernel_size=3, stride=2, device=device)
        batched = _StreamConv1d(conv, pad_mode="replicate")
        row0 = _StreamConv1d(conv, pad_mode="replicate")
        row1 = _StreamConv1d(conv, pad_mode="replicate")
        samples = 4
    else:
        conv = nn.ConvTranspose1d(1, 2, kernel_size=4, stride=2, device=device)
        batched = _StreamConvTr1d(conv)
        row0 = _StreamConvTr1d(conv)
        row1 = _StreamConvTr1d(conv)
        samples = 2

    for stream, batch_size in ((batched, 2), (row0, 1), (row1, 1)):
        stream.reset(batch_size, device, torch.float32)

    carry_pointer = _stream_carry(batched).data_ptr()
    fresh_pointer = batched._fresh.data_ptr()
    singletons = [row0, row1]
    active_schedule = [
        _mask(False, False, device=device),
        _mask(True, False, device=device),
        _mask(True, True, device=device),
        _mask(False, True, device=device),
        _mask(True, True, device=device),
    ]

    for step, active in enumerate(active_schedule):
        inputs = _inputs((2, 1, samples), SEED + 200 + step, device)
        carry_before = _stream_carry(batched).clone()
        fresh_before = batched._fresh.clone()
        output = PersonaPlexMimiCodec._run_stages(inputs, [(kind, batched)], active)

        assert _stream_carry(batched).data_ptr() == carry_pointer
        assert batched._fresh.data_ptr() == fresh_pointer
        for row, is_active in enumerate(active.tolist()):
            if is_active:
                expected = PersonaPlexMimiCodec._run_stages(
                    inputs[row : row + 1],
                    [(kind, singletons[row])],
                    _mask(True, device=device),
                )
                tolerance = 0.0 if device.type == "cuda" else 1e-6
                torch.testing.assert_close(output[row : row + 1], expected, rtol=0.0, atol=tolerance)
                torch.testing.assert_close(
                    _stream_carry(batched)[row],
                    _stream_carry(singletons[row])[0],
                    rtol=0.0,
                    atol=tolerance,
                )
                assert torch.equal(batched._fresh[row], singletons[row]._fresh[0])
            else:
                assert torch.equal(_stream_carry(batched)[row], carry_before[row])
                assert torch.equal(batched._fresh[row], fresh_before[row])


@pytest.mark.parametrize("kind", ["conv", "convtr"])
@pytest.mark.cpu
def test_mimi_conv_carries_preserve_inactive_rows_cpu(kind: str) -> None:
    _assert_mimi_conv_carries_preserve_inactive_rows(kind, torch.device("cpu"))


@pytest.mark.parametrize("kind", ["conv", "convtr"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mimi_conv_carries_preserve_inactive_rows_cuda(kind: str) -> None:
    _assert_mimi_conv_carries_preserve_inactive_rows(kind, CUDA_DEVICE)


@pytest.mark.core_model
@pytest.mark.cpu
def test_mimi_transformer_recycled_row_matches_fresh_stream():
    from vllm_omni.model_executor.models.personaplex.personaplex_mimi import _MimiStreamingTransformer

    torch.manual_seed(0)
    used = _MimiStreamingTransformer(num_layers=2, dim=16, num_heads=2, context=8)
    for param in used.parameters():
        torch.nn.init.normal_(param, std=0.1)
    fresh = _MimiStreamingTransformer(num_layers=2, dim=16, num_heads=2, context=8)
    fresh.load_state_dict(used.state_dict())
    used.streaming_init(2)
    fresh.streaming_init(2)
    both = torch.tensor([True, True])

    # Wrap row 0's ring (6 frames x 2 positions > context 8) before recycling it.
    for _ in range(6):
        used.step(torch.randn(2, 2, 16), both)
    used.reset_slot(0)

    frames = [torch.randn(2, 2, 16) for _ in range(5)]
    for x in frames:
        recycled = used.step(x, both)
        expected = fresh.step(x, both)
        torch.testing.assert_close(recycled[0], expected[0], rtol=0, atol=0)

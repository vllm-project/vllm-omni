# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.personaplex import (
    personaplex_code2wav,
    personaplex_mimi,
)
from vllm_omni.model_executor.models.personaplex.personaplex_code2wav import (
    PersonaPlexCode2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStreamingMimi(nn.Module):
    def __init__(self, stream_value: int = 0) -> None:
        super().__init__()
        self.stream_value = stream_value
        self.decode_frame_calls = 0
        self.reset_calls = 0

    def decode(self, codes: torch.Tensor, return_dict: bool = True):
        assert return_dict
        frames = int(codes.shape[-1])
        return SimpleNamespace(audio_values=torch.arange(frames * 4, dtype=torch.float32).reshape(1, 1, -1))

    def streaming_init(self, batch_size: int) -> None:
        assert batch_size == 1

    def decode_frame(self, codes: torch.Tensor) -> torch.Tensor:
        assert codes.shape == (1, 2)
        self.decode_frame_calls += 1
        value = self.stream_value + self.decode_frame_calls
        return torch.full((1, 4), float(value), dtype=torch.float32)

    def reset_streaming(self) -> None:
        self.reset_calls += 1


class _FakeChunkedMimi(_FakeStreamingMimi):
    def __init__(self) -> None:
        super().__init__()
        self.decode_frames_calls = 0
        self.decode_frames_shapes: list[tuple[int, ...]] = []

    def decode_frames(self, codes: torch.Tensor) -> torch.Tensor:
        self.decode_frames_calls += 1
        self.decode_frames_shapes.append(tuple(codes.shape))
        return torch.arange(codes.shape[-1] * 4, dtype=torch.float32).reshape(1, -1)


class _FakeMimiModel(nn.Module):
    def __init__(self, _config) -> None:
        super().__init__()
        self.encoder_transformer = nn.Linear(1, 1, bias=False)
        self.decoder_transformer = nn.Linear(1, 1, bias=False)
        self.quantizer = nn.Linear(1, 1, bias=False)
        self.encoder = SimpleNamespace(layers=[])
        self.downsample = SimpleNamespace(conv=nn.Conv1d(1, 1, 1))
        self.upsample = SimpleNamespace(conv=nn.ConvTranspose1d(1, 1, 1))
        self.decoder = SimpleNamespace(layers=[])

    def load_state_dict(self, _state_dict, strict: bool = True):
        assert not strict
        return SimpleNamespace(
            missing_keys=[
                "encoder_transformer.weight",
                "decoder_transformer.weight",
            ],
            unexpected_keys=[],
        )


class _FakeMimiTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def load_weights(self, _state_dict, _prefix: str) -> int:
        return 80


def test_moshi_checkpoint_mapping_drops_replaced_transformer_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapper = getattr(personaplex_mimi, "_map_moshi_codec_weights", None)
    assert callable(mapper), "PersonaPlex Mimi must load its bundled codec weights without Hugging Face"

    source = {
        "encoder.model.3.conv.conv.weight": torch.ones(1),
        "downsample.conv.conv.conv.weight": torch.ones(1),
        "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum": torch.ones(1),
        "quantizer.rvq_rest.vq.layers.3._codebook._initialized": torch.ones(1),
        "encoder_transformer.transformer.layers.0.norm1.weight": torch.ones(1),
    }

    mapped = mapper(source)

    assert set(mapped) == {
        "encoder.layers.3.conv.weight",
        "downsample.conv.weight",
        "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
        "quantizer.acoustic_residual_vector_quantizer.layers.3.codebook.initialized",
    }
    assert mapped["encoder.layers.3.conv.weight"] is source["encoder.model.3.conv.conv.weight"]
    monkeypatch.setattr("transformers.MimiConfig", lambda: object())
    monkeypatch.setattr("transformers.MimiModel", _FakeMimiModel)
    monkeypatch.setattr("safetensors.torch.load_file", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(personaplex_mimi, "_MimiStreamingTransformer", _FakeMimiTransformer)

    codec = personaplex_mimi.PersonaPlexMimiCodec(
        checkpoint="/unused",
        device="cpu",
    )

    assert not hasattr(codec.model, "encoder_transformer")
    assert not hasattr(codec.model, "decoder_transformer")
    assert not any(
        name.startswith(("model.encoder_transformer.", "model.decoder_transformer."))
        for name, _ in codec.named_parameters()
    )


def test_mimi_full_stream_reset_reuses_all_state_storage_and_clears_offsets() -> None:
    codec = personaplex_mimi.PersonaPlexMimiCodec.__new__(personaplex_mimi.PersonaPlexMimiCodec)
    nn.Module.__init__(codec)
    codec.device = torch.device("cpu")
    codec.dtype = torch.float32
    codec._enc_stages = [("conv", personaplex_mimi._StreamConv1d(nn.Conv1d(1, 1, 3)))]
    codec._dec_stages = []
    codec._downsample = personaplex_mimi._StreamConv1d(nn.Conv1d(1, 1, 3))
    codec._upsample = personaplex_mimi._StreamConvTr1d(nn.ConvTranspose1d(1, 1, 3))
    codec.encoder_transformer = personaplex_mimi._MimiStreamingTransformer(
        num_layers=1,
        dim=4,
        num_heads=1,
        context=3,
    )
    codec.decoder_transformer = personaplex_mimi._MimiStreamingTransformer(
        num_layers=1,
        dim=4,
        num_heads=1,
        context=3,
    )
    codec.streaming_init(batch_size=2)

    conv_states = list(codec._conv_states())
    conv_buffers = [
        state.prev if isinstance(state, personaplex_mimi._StreamConv1d) else state.partial for state in conv_states
    ]
    transformers = [codec.encoder_transformer, codec.decoder_transformer]
    kv_states = [kv for transformer in transformers for kv in transformer._kv]
    state_tensors = [
        *conv_buffers,
        *(kv.cache for kv in kv_states),
        *(kv.end_offset for kv in kv_states),
        *(kv.start_offset for kv in kv_states),
        *(transformer._offset for transformer in transformers),
    ]
    for tensor in state_tensors:
        tensor.fill_(1)
    for state in conv_states:
        state._fresh.zero_()
    storage = [tensor.data_ptr() for tensor in state_tensors]

    codec.reset_streaming()

    assert storage == [tensor.data_ptr() for tensor in state_tensors]
    cleared = [
        *conv_buffers,
        *(kv.end_offset for kv in kv_states),
        *(kv.start_offset for kv in kv_states),
        *(transformer._offset for transformer in transformers),
    ]
    assert all(not tensor.any() for tensor in cleared)
    assert all(state._fresh.all() for state in conv_states)


def _model(
    *,
    max_sessions: int = 1,
    async_chunk: bool = True,
    shared_decoder: bool = False,
) -> tuple[PersonaPlexCode2Wav, _FakeStreamingMimi]:
    mimi_config = SimpleNamespace(num_codebooks=2, sample_rate=24000, samples_per_frame=4, mimi_name=None)
    config = SimpleNamespace(mimi_config=mimi_config, mimi_name=None, mimi_shared_decoder=shared_decoder)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/unused",
            hf_config=config,
            duplex_max_sessions=max_sessions,
            async_chunk=async_chunk,
        ),
        device_config=SimpleNamespace(device="cpu"),
    )
    model = PersonaPlexCode2Wav(vllm_config=vllm_config)
    mimi = _FakeStreamingMimi()
    model.mimi = mimi
    model._mimi_device = torch.device("cpu")
    return model, mimi


def _codes(frames: int, *, start: int = 0) -> torch.Tensor:
    return torch.stack([torch.arange(start + offset, start + offset + frames) for offset in (0, 100)]).reshape(-1)


def _audio(output) -> torch.Tensor:
    return output.multimodal_outputs["model_outputs"][0]


def test_delta_codes_skip_cpu_history(mocker) -> None:
    model, _ = _model()
    chunks = [_codes(1, start=frame) for frame in range(1000)]
    decode = mocker.patch.object(
        model,
        "_decode_streaming_frames",
        return_value=torch.empty(0),
    )
    try:
        for chunk in chunks:
            model(input_ids=chunk, request_ids=["req"])
    finally:
        model.on_requests_finished({"req"})

    assert decode.call_count == 1000
    assert all(call.args[1].shape == (2, 1) for call in decode.call_args_list)


def test_identical_consecutive_delta_frames_are_both_decoded() -> None:
    model, mimi = _model()
    chunk = _codes(1)

    first = model(input_ids=chunk, request_ids=["req"])
    second = model(input_ids=chunk, request_ids=["req"])

    assert _audio(first).numel() == 4
    assert _audio(second).numel() == 4
    assert mimi.decode_frame_calls == 2


def test_full_payload_is_consumed_once_across_forwards() -> None:
    model, mimi = _model(async_chunk=False)
    runtime_info = [{"codes": {"audio": _codes(2)}}]

    first = model(
        input_ids=torch.zeros(2, dtype=torch.long),
        request_ids=["req"],
        runtime_additional_information=runtime_info,
    )
    second = model(
        input_ids=torch.zeros(2, dtype=torch.long),
        request_ids=["req"],
        runtime_additional_information=runtime_info,
    )

    assert _audio(first).numel() == 8
    assert _audio(second).numel() == 0
    assert mimi.decode_frame_calls == 2

    model.on_requests_finished({"req"})
    reused = model(
        input_ids=torch.zeros(2, dtype=torch.long),
        request_ids=["req"],
        runtime_additional_information=runtime_info,
    )

    assert _audio(reused).numel() == 8
    assert mimi.decode_frame_calls == 4


def test_resumable_delta_codes_emit_only_new_pcm() -> None:
    model, mimi = _model()

    first = model(input_ids=_codes(2), request_ids=["req"])
    second = model(input_ids=_codes(1, start=100), request_ids=["req"])

    assert _audio(first).numel() == 8
    assert _audio(second).numel() == 4
    assert mimi.decode_frame_calls == 3


@pytest.mark.parametrize(
    ("frames", "expected_shapes"),
    [
        (5, [(1, 2, 5)]),
        (12, [(1, 2, 5), (1, 2, 5), (1, 2, 2)]),
    ],
)
def test_streaming_decode_uses_bounded_multi_frame_calls(
    frames: int,
    expected_shapes: list[tuple[int, ...]],
) -> None:
    model, _ = _model()
    mimi = _FakeChunkedMimi()
    model.mimi = mimi

    output = model(input_ids=_codes(frames), request_ids=["req"])

    assert _audio(output).numel() == frames * 4
    assert mimi.decode_frames_shapes == expected_shapes
    assert mimi.decode_frame_calls == 0


def test_profile_inputs_skip_decode_while_malformed_online_input_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, mimi = _model()
    runtime_info = [{"meta": {"personaplex_dummy_profile": True}}]
    warnings: list[tuple] = []
    monkeypatch.setattr(personaplex_code2wav.logger, "warning", lambda *args: warnings.append(args))

    output = model(
        input_ids=torch.arange(3),
        runtime_additional_information=runtime_info,
    )

    assert _audio(output).numel() == 0
    assert mimi.decode_frame_calls == 0
    assert warnings == []

    output = model(input_ids=torch.arange(3))

    assert _audio(output).numel() == 0
    assert mimi.decode_frame_calls == 0
    assert len(warnings) == 1
    assert "not divisible by" in warnings[0][0]
    assert model.get_dummy_runtime_additional_information(2) == [
        {"meta": {"personaplex_dummy_profile": True}},
        {"meta": {"personaplex_dummy_profile": True}},
    ]


def test_request_id_falls_back_to_runtime_information() -> None:
    model, _ = _model()
    info = [{"request_id": "runtime-req"}]

    model(input_ids=_codes(2), runtime_additional_information=info)
    second = model(input_ids=_codes(1, start=100), runtime_additional_information=info)

    assert _audio(second).numel() == 4
    assert model._request_codec_slots == {"runtime-req": 0}


def test_decoder_slot_lifecycle_isolated_capacity_and_reuse() -> None:
    model, _ = _model(max_sessions=2)
    first = _FakeStreamingMimi(stream_value=10)
    second = _FakeStreamingMimi(stream_value=20)
    model._set_mimi_codecs([first, second])
    assert model._max_codec_sessions == 2

    initial = model(
        input_ids=torch.cat([_codes(1), _codes(1)]),
        request_ids=["first", "second"],
        seq_token_counts=[2, 2],
    )
    continued = model(
        input_ids=torch.cat([_codes(1, start=10), _codes(1, start=20)]),
        request_ids=["first", "second"],
        seq_token_counts=[2, 2],
    )

    initial_audio = initial.multimodal_outputs["model_outputs"]
    continued_audio = continued.multimodal_outputs["model_outputs"]
    assert initial_audio[0].tolist() == [11.0] * 4
    assert initial_audio[1].tolist() == [21.0] * 4
    assert continued_audio[0].tolist() == [12.0] * 4
    assert continued_audio[1].tolist() == [22.0] * 4
    assert model._request_codec_slots == {"first": 0, "second": 1}
    with pytest.raises(RuntimeError, match="decoder capacity 2 is exhausted"):
        model._codec_for_request("overflow")

    model.on_requests_finished(["first"])
    assert first.reset_calls == 1
    assert second.reset_calls == 0
    replacement = model(input_ids=_codes(1), request_ids=["replacement"])
    assert _audio(replacement).tolist() == [13.0] * 4


class _GraphStatsMimi(_FakeStreamingMimi):
    """A leased decoder that reports invented graph/eager decode counters."""

    def __init__(self) -> None:
        super().__init__()
        self.pretend_graph = False
        self.decode_graph = 0
        self.decode_eager = 0

    def decode_frame(self, codes: torch.Tensor) -> torch.Tensor:
        if self.pretend_graph:
            self.decode_graph += 1
        else:
            self.decode_eager += 1
        return super().decode_frame(codes)

    def graph_replay_stats(self) -> dict[str, int]:
        return {
            "encode_graph": 0,
            "encode_eager": 0,
            "decode_graph": self.decode_graph,
            "decode_eager": self.decode_eager,
        }


def test_graph_stats_logged_periodically_with_hit_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.model_executor.models.personaplex import personaplex_code2wav

    monkeypatch.setattr(personaplex_code2wav, "_MIMI_GRAPH_STATS_LOG_INTERVAL", 3)
    infos: list[tuple] = []
    monkeypatch.setattr(personaplex_code2wav.logger, "info", lambda *args, **kwargs: infos.append(args))
    model, _ = _model()
    mimi = _GraphStatsMimi()
    model.mimi = mimi
    model._mimi_device = torch.device("cpu")
    try:
        for start in range(2):
            model(input_ids=_codes(1, start=start), request_ids=["req"])
        assert not infos  # below the interval: nothing is logged yet

        for start in range(2, 6):
            model(input_ids=_codes(1, start=start), request_ids=["req"])
        mimi.pretend_graph = True
        for start in range(6, 9):
            model(input_ids=_codes(1, start=start), request_ids=["req"])
    finally:
        model.on_requests_finished({"req"})

    assert len(infos) == 3  # at 3, 6 and 9 decoded requests
    rendered = [args[0] % args[1:] for args in infos]
    assert "graph_replays=0" in rendered[0] and "eager_decodes=3" in rendered[0]
    assert "hit_rate=0.0%" in rendered[0]
    assert "graph_replays=0" in rendered[1] and "eager_decodes=6" in rendered[1]
    assert "graph_replays=3" in rendered[2] and "eager_decodes=6" in rendered[2]
    assert "hit_rate=33.3%" in rendered[2]


def test_graph_stats_log_fires_when_multi_request_steps_cross_the_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """A duplex step decodes several requests at once, so the request counter
    jumps past interval boundaries instead of landing on them exactly."""
    from vllm_omni.model_executor.models.personaplex import personaplex_code2wav

    monkeypatch.setattr(personaplex_code2wav, "_MIMI_GRAPH_STATS_LOG_INTERVAL", 3)
    infos: list[tuple] = []
    monkeypatch.setattr(personaplex_code2wav.logger, "info", lambda *args, **kwargs: infos.append(args))
    model, _ = _model(max_sessions=2)
    first, second = _GraphStatsMimi(), _GraphStatsMimi()
    model._set_mimi_codecs([first, second])
    model._mimi_device = torch.device("cpu")
    try:
        # Three two-request steps: the counter goes 2 -> 4 -> 6, crossing the
        # interval multiples 3 and 6 without ever equaling 3.
        for start in range(3):
            model(
                input_ids=torch.cat([_codes(1, start=start), _codes(1, start=10 + start)]),
                request_ids=["first", "second"],
                seq_token_counts=[2, 2],
            )
    finally:
        model.on_requests_finished({"first", "second"})

    assert len(infos) == 2
    rendered = [args[0] % args[1:] for args in infos]
    assert "decoded_requests=4" in rendered[0]  # first crossing of 3
    assert "decoded_requests=6" in rendered[1]  # crossing of 6


def test_graph_stats_log_skips_codecs_without_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.model_executor.models.personaplex import personaplex_code2wav

    monkeypatch.setattr(personaplex_code2wav, "_MIMI_GRAPH_STATS_LOG_INTERVAL", 1)
    infos: list[tuple] = []
    monkeypatch.setattr(personaplex_code2wav.logger, "info", lambda *args, **kwargs: infos.append(args))
    model, mimi = _model()  # _FakeStreamingMimi has no graph_replay_stats

    model(input_ids=_codes(1), request_ids=["req"])
    model.on_requests_finished({"req"})

    assert mimi.decode_frame_calls == 1
    assert not infos


class _FakeSharedMimi(nn.Module):
    """A width-N shared decoder: per-row state, one decode call per frame-count group."""

    def __init__(self, samples_per_frame: int = 4) -> None:
        super().__init__()
        self.samples_per_frame = samples_per_frame
        self.batch_size: int | None = None
        self.calls: list[tuple[tuple[int, ...], list[bool]]] = []
        self.reset_rows: list[int] = []
        self.row_state: list[float] = []

    def streaming_init(self, batch_size: int) -> None:
        self.batch_size = batch_size
        if len(self.row_state) < batch_size:
            self.row_state = [0.0] * batch_size

    def reset_slot(self, row: int) -> None:
        self.reset_rows.append(row)
        self.row_state[row] = 0.0

    def decode_frames(self, codes: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        n_frames = codes.shape[-1]
        self.calls.append((tuple(codes.shape), active.tolist()))
        out = torch.zeros(codes.shape[0], n_frames * self.samples_per_frame)
        for row in range(codes.shape[0]):
            if not active[row]:
                continue
            value = float(codes[row, :, 0].sum()) + self.row_state[row]
            self.row_state[row] += 1.0
            out[row, :] = value
        return out


def _shared_model(max_sessions: int) -> tuple[PersonaPlexCode2Wav, _FakeSharedMimi]:
    model, _ = _model(max_sessions=max_sessions, shared_decoder=True)
    codec = _FakeSharedMimi()
    codec.streaming_init(max_sessions)  # load_weights initializes the shared codec at full width
    model._set_mimi_codecs([codec])
    model._mimi_device = torch.device("cpu")
    return model, codec


def test_shared_decoder_groups_requests_by_frame_count_and_slices_rows() -> None:
    model, codec = _shared_model(max_sessions=3)

    out = model(
        input_ids=torch.cat([_codes(1, start=0), _codes(2, start=10)]),
        request_ids=["a", "b"],
        seq_token_counts=[2, 4],
    )

    # One full-width call per distinct frame count; only leased rows are active.
    assert [shape for shape, _ in codec.calls] == [(3, 2, 1), (3, 2, 2)]
    assert [active for _, active in codec.calls] == [
        [True, False, False],
        [False, True, False],
    ]
    # Each request receives its own row's samples, and inactive row state is untouched.
    audios = out.multimodal_outputs["model_outputs"]
    assert audios[0].tolist() == [100.0] * 4
    assert audios[1].tolist() == [120.0] * 8
    assert codec.row_state == [1.0, 1.0, 0.0]
    assert model._request_codec_slots == {"a": 0, "b": 1}


def test_shared_decoder_leases_rows_recycles_and_enforces_capacity() -> None:
    model, codec = _shared_model(max_sessions=2)
    model(
        input_ids=torch.cat([_codes(1, start=0), _codes(1, start=10)]),
        request_ids=["a", "b"],
        seq_token_counts=[2, 2],
    )
    assert model._request_codec_slots == {"a": 0, "b": 1}

    with pytest.raises(RuntimeError, match=r"decoder capacity 2 is exhausted"):
        model(input_ids=_codes(1, start=20), request_ids=["c"])

    model.on_requests_finished(["a"])
    # [0, 1] are the initial leases of a and b; the release re-primes row 0.
    assert codec.reset_rows == [0, 1, 0]
    model(input_ids=_codes(1, start=20), request_ids=["c"])
    assert model._request_codec_slots == {"b": 1, "c": 0}
    assert codec.reset_rows == [0, 1, 0, 0]  # release reset + lease reset

    # An anonymous request borrows a free row without claiming a slot.
    model.on_requests_finished(["b"])
    model(input_ids=_codes(1, start=30))
    # ... then the release of b resets row 1, and the anonymous lease borrows it.
    assert codec.reset_rows == [0, 1, 0, 0, 1, 1]
    assert model._request_codec_slots == {"c": 0}


def test_shared_decoder_chunks_long_payloads_in_order() -> None:
    model, codec = _shared_model(max_sessions=2)

    out = model(input_ids=_codes(7, start=0), request_ids=["long"])

    # F=7 splits into segments of 5 then 2, decoded in order on one row. The
    # fake's per-call value is the segment's first frame (100, then 110) plus
    # the row state it has accumulated (0, then 1).
    assert [shape for shape, _ in codec.calls] == [(2, 2, 5), (2, 2, 2)]
    assert [active for _, active in codec.calls] == [[True, False], [True, False]]
    audio = out.multimodal_outputs["model_outputs"][0]
    assert audio.numel() == 7 * 4
    assert audio.tolist() == [100.0] * (5 * 4) + [111.0] * (2 * 4)
    assert codec.row_state == [2.0, 0.0]


def test_mimi_shared_decoder_config_flag_defaults_off() -> None:
    from vllm_omni.model_executor.models.personaplex.configuration_personaplex import PersonaPlexConfig

    assert PersonaPlexConfig().mimi_shared_decoder is False

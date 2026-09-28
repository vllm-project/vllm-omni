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


class _FakeBatchedMimi(nn.Module):
    """Shared streaming decoder: row ``r``'s ``n``-th decoded frame is 4 samples of ``100 * r + n``."""

    def __init__(self) -> None:
        super().__init__()
        self.frames: torch.Tensor | None = None
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.reset_rows: list[int] = []

    def streaming_init(self, batch_size: int) -> None:
        self.frames = torch.zeros(batch_size)

    def decode_frame(self, codes: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        assert self.frames is not None
        assert codes.shape == (self.frames.shape[0], 2)
        assert active.shape == self.frames.shape
        self.calls.append((codes.clone(), active.clone()))
        self.frames += active
        rows = torch.arange(self.frames.shape[0], dtype=torch.float32)
        return (100 * rows + self.frames)[:, None].expand(-1, 4).clone()

    def reset_slot(self, row: int) -> None:
        assert self.frames is not None
        self.frames[row] = 0
        self.reset_rows.append(row)


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
    *, max_sessions: int = 1, install: bool = True, cuda_graphs: bool = False, async_chunk: bool = True
) -> tuple[PersonaPlexCode2Wav, _FakeBatchedMimi]:
    mimi_config = SimpleNamespace(num_codebooks=2, sample_rate=24000, samples_per_frame=4, mimi_name=None)
    config = SimpleNamespace(mimi_config=mimi_config, mimi_name=None, mimi_cuda_graphs=cuda_graphs)
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
    mimi = _FakeBatchedMimi()
    if install:
        model._install_mimi(mimi, torch.device("cpu"))
    return model, mimi


def _codes(frames: int, *, start: int = 0) -> torch.Tensor:
    return torch.stack([torch.arange(start + offset, start + offset + frames) for offset in (0, 100)]).reshape(-1)


def _audio(output) -> torch.Tensor:
    return output.multimodal_outputs["model_outputs"][0]


def _audios(output) -> list[list[float]]:
    return [audio.tolist() for audio in output.multimodal_outputs["model_outputs"]]


def _pcm(*values: float) -> list[float]:
    """Fake PCM of consecutive decoded frames (4 samples each)."""
    return [float(value) for value in values for _ in range(4)]


def _actives(mimi: _FakeBatchedMimi) -> list[list[bool]]:
    return [active.tolist() for _, active in mimi.calls]


@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_load_weights_builds_one_shared_decoder_with_a_row_per_session(
    monkeypatch: pytest.MonkeyPatch,
    cuda_graphs: bool,
) -> None:
    built: list[_FakeBatchedMimi] = []

    class _Codec(_FakeBatchedMimi):
        def __init__(self, checkpoint: str | None, device: str) -> None:
            super().__init__()
            self.model = SimpleNamespace(config=SimpleNamespace(sampling_rate=24000))
            self.captured_rows: list[int] = []
            built.append(self)

        def capture_decode_graph(self) -> bool:
            # The graph is captured over the streaming rows, so they must exist.
            assert self.frames is not None
            self.captured_rows.append(self.frames.shape[0])
            return True

    monkeypatch.setattr(personaplex_mimi, "PersonaPlexMimiCodec", _Codec)
    model, _ = _model(max_sessions=3, install=False, cuda_graphs=cuda_graphs)

    model.load_weights(iter([("unused.weight", torch.zeros(1))]))

    # Three session rows plus the scratch row, allocated before the first request.
    assert built[0].frames is not None and built[0].frames.shape == (4,)
    # Only mimi_cuda_graphs decides the decode graph; Stage 1 stays enforce_eager.
    assert built[0].captured_rows == ([4] if cuda_graphs else [])


def test_delta_codes_skip_cpu_history(mocker) -> None:
    model, _ = _model()
    decode = mocker.patch.object(model, "_decode_pending", return_value=[])
    cat = mocker.spy(personaplex_code2wav.torch, "cat")
    equal = mocker.spy(personaplex_code2wav.torch, "equal")

    for frame in range(1000):
        model(input_ids=_codes(1, start=frame), request_ids=["req"])

    assert decode.call_count == 1000
    assert all([codes.shape for _, _, codes in call.args[0]] == [(2, 1)] for call in decode.call_args_list)
    assert cat.call_count == 0
    assert equal.call_count == 0


def test_resumable_delta_codes_emit_only_new_pcm() -> None:
    model, mimi = _model()

    first = model(input_ids=_codes(2), request_ids=["req"])
    second = model(input_ids=_codes(1, start=100), request_ids=["req"])

    assert _audio(first).tolist() == _pcm(1, 2)
    assert _audio(second).tolist() == _pcm(3)
    assert len(mimi.calls) == 3


def test_identical_consecutive_delta_frames_are_both_decoded() -> None:
    model, mimi = _model()
    chunk = _codes(1)

    first = model(input_ids=chunk, request_ids=["req"])
    second = model(input_ids=chunk, request_ids=["req"])

    assert _audio(first).tolist() == _pcm(1)
    assert _audio(second).tolist() == _pcm(2)
    assert len(mimi.calls) == 2


def test_full_payload_is_consumed_once_across_forwards() -> None:
    model, mimi = _model(async_chunk=False)
    runtime_info = [{"codes": {"audio": _codes(2)}}]

    def forward():
        return model(
            input_ids=torch.zeros(2, dtype=torch.long),
            request_ids=["req"],
            runtime_additional_information=runtime_info,
        )

    first = forward()
    second = forward()

    assert _audio(first).tolist() == _pcm(1, 2)
    assert _audio(second).numel() == 0
    assert len(mimi.calls) == 2

    model.on_requests_finished({"req"})
    reused = forward()

    assert _audio(reused).tolist() == _pcm(1, 2)
    assert len(mimi.calls) == 4


@pytest.mark.parametrize("frames", [5, 12])
def test_each_frame_is_one_decoder_call_across_all_rows(frames: int) -> None:
    model, mimi = _model()

    output = model(input_ids=_codes(frames), request_ids=["req"])

    assert _audio(output).tolist() == _pcm(*range(1, frames + 1))
    assert len(mimi.calls) == frames
    for frame, (codes, active) in enumerate(mimi.calls):
        # Row 0 is the request's leased row; row 1 is the idle scratch row.
        assert codes[0].tolist() == [frame, 100 + frame]
        assert active.tolist() == [True, False]


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
    assert mimi.calls == []
    assert warnings == []

    output = model(input_ids=torch.arange(3))

    assert _audio(output).numel() == 0
    assert mimi.calls == []
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

    assert _audio(second).tolist() == _pcm(3)
    assert model._request_rows == {"runtime-req": 0}


def test_rows_are_leased_per_request_isolated_and_recycled() -> None:
    model, mimi = _model(max_sessions=2)

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

    assert _audios(initial) == [_pcm(1), _pcm(101)]
    assert _audios(continued) == [_pcm(2), _pcm(102)]
    assert model._request_rows == {"first": 0, "second": 1}
    with pytest.raises(RuntimeError, match="decoder capacity 2 is exhausted"):
        model(input_ids=_codes(1), request_ids=["overflow"])

    model.on_requests_finished(["first"])
    assert mimi.reset_rows == [0]

    replacement = model(
        input_ids=torch.cat([_codes(1), _codes(1, start=30)]),
        request_ids=["replacement", "second"],
        seq_token_counts=[2, 2],
    )
    # The recycled row restarts its stream; the other session carries on.
    assert _audios(replacement) == [_pcm(1), _pcm(103)]
    assert model._request_rows == {"second": 1, "replacement": 0}


def test_requests_without_id_use_the_scratch_row_one_pass_each() -> None:
    model, _ = _model(max_sessions=1)

    output = model(
        input_ids=torch.cat([_codes(1), _codes(2), _codes(1)]),
        runtime_additional_information=[{"request_id": "leased"}, {}, {}],
        seq_token_counts=[2, 4, 2],
    )

    # Row 1 is scratch and is reset after every pass, so each id-less request
    # decodes as a fresh stream and never touches the leased row.
    assert _audios(output) == [_pcm(1), _pcm(101, 102), _pcm(101)]
    assert model._request_rows == {"leased": 0}

    continued = model(input_ids=_codes(1, start=10), request_ids=["leased"])

    assert _audio(continued).tolist() == _pcm(2)


def test_mixed_frame_counts_route_pcm_and_advance_only_active_rows() -> None:
    model, mimi = _model(max_sessions=2)

    output = model(
        input_ids=torch.cat([_codes(3), _codes(1, start=50)]),
        request_ids=["long", "short"],
        seq_token_counts=[6, 2],
    )

    long_audio, short_audio = output.multimodal_outputs["model_outputs"]
    assert long_audio.tolist() == _pcm(1, 2, 3)
    assert short_audio.tolist() == _pcm(101)
    assert all(audio.ndim == 1 and audio.is_contiguous() for audio in (long_audio, short_audio))
    assert _actives(mimi) == [[True, True, False], [True, False, False], [True, False, False]]
    assert mimi.calls[0][0][:2].tolist() == [[0, 100], [50, 150]]
    assert mimi.frames is not None and mimi.frames.tolist() == [3.0, 1.0, 0.0]

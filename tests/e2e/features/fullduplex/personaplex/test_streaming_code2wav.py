# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


def _model(*, max_sessions: int = 1) -> tuple[PersonaPlexCode2Wav, _FakeStreamingMimi]:
    mimi_config = SimpleNamespace(num_codebooks=2, sample_rate=24000, samples_per_frame=4, mimi_name=None)
    config = SimpleNamespace(mimi_config=mimi_config, mimi_name=None)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/unused",
            hf_config=config,
            duplex_max_sessions=max_sessions,
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


@pytest.mark.parametrize("second_codes", [_codes(3), _codes(1, start=100)])
def test_resumable_codes_emit_only_new_pcm(second_codes: torch.Tensor) -> None:
    model, mimi = _model()

    first = model(input_ids=_codes(2), request_ids=["req"])
    second = model(input_ids=second_codes, request_ids=["req"])

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
    second = model(input_ids=_codes(3), runtime_additional_information=info)

    assert _audio(second).numel() == 4


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

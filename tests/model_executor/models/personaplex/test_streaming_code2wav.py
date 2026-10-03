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

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.frames: torch.Tensor | None = None
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.reset_rows: list[int] = []
        self.encodes: bool | None = None

    def streaming_init(self, batch_size: int, *, encode: bool = True) -> None:
        self.frames = torch.zeros(batch_size, device=self.device)
        self.encodes = encode

    def decode_frame(self, codes: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        assert self.frames is not None
        assert codes.shape == (self.frames.shape[0], 2)
        assert active.shape == self.frames.shape
        self.calls.append((codes.clone(), active.clone()))
        self.frames += active
        rows = torch.arange(self.frames.shape[0], dtype=torch.float32, device=self.device)
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
    codec.device, codec.dtype = torch.device("cpu"), torch.float32
    codec._enc_stages = [("conv", personaplex_mimi._StreamConv1d(nn.Conv1d(1, 1, 3)))]
    codec._dec_stages = []
    codec._downsample = personaplex_mimi._StreamConv1d(nn.Conv1d(1, 1, 3))
    codec._upsample = personaplex_mimi._StreamConvTr1d(nn.ConvTranspose1d(1, 1, 3))
    codec.encoder_transformer = personaplex_mimi._MimiStreamingTransformer(1, 4, 1, 3)
    codec.decoder_transformer = personaplex_mimi._MimiStreamingTransformer(1, 4, 1, 3)
    codec.streaming_init(batch_size=2)

    conv_states = list(codec._conv_states())
    conv_buffers = [s.prev if isinstance(s, personaplex_mimi._StreamConv1d) else s.partial for s in conv_states]
    transformers = [codec.encoder_transformer, codec.decoder_transformer]
    kv_states = [kv for t in transformers for kv in t._kv]
    state_tensors = [*conv_buffers, *(kv.cache for kv in kv_states), *(kv.end_offset for kv in kv_states), *(t._offset for t in transformers)]
    for tensor in state_tensors:
        tensor.fill_(1)
    storage = [t.data_ptr() for t in state_tensors]

    codec.reset_streaming()
    assert storage == [t.data_ptr() for t in state_tensors]
    assert all(not (t != 0).any() for t in [*conv_buffers, *(kv.end_offset for kv in kv_states)])


def _model(
    *,
    max_sessions: int = 1,
    install: bool = True,
    cuda_graphs: bool = False,
    async_chunk: bool = True,
    device: str = "cpu",
    runner_device: str = "cpu",
    decode_tf32: bool | None = None,
) -> tuple[PersonaPlexCode2Wav, _FakeBatchedMimi]:
    mimi_config = SimpleNamespace(num_codebooks=2, sample_rate=24000, samples_per_frame=4, mimi_name=None)
    config = SimpleNamespace(mimi_config=mimi_config, mimi_name=None, mimi_cuda_graphs=cuda_graphs)
    if decode_tf32 is not None:
        config.mimi_decode_tf32 = decode_tf32
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/unused", hf_config=config, duplex_max_sessions=max_sessions, async_chunk=async_chunk,
        ),
        device_config=SimpleNamespace(device=runner_device),
    )
    model = PersonaPlexCode2Wav(vllm_config=vllm_config)
    mimi = _FakeBatchedMimi(device)
    if install:
        model._install_mimi(mimi, torch.device(device))
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

    # Three session rows plus the scratch row of the decoder half only, allocated before the first request.
    assert built[0].frames is not None and built[0].frames.shape == (4,)
    assert built[0].encodes is False
    # Only mimi_cuda_graphs decides the decode graph; Stage 1 stays enforce_eager.
    assert built[0].captured_rows == ([4] if cuda_graphs else [])


class _LoadCodec(_FakeBatchedMimi):
    def __init__(self, checkpoint: str | None, device: str) -> None:
        super().__init__()
        self.model = SimpleNamespace(config=SimpleNamespace(sampling_rate=24000))


@pytest.mark.parametrize(
    ("runner_device", "decode_tf32", "expect_tf32"),
    [("cuda", None, True), ("cuda", False, False), ("cpu", None, False)],
)
def test_load_weights_tf32_follows_device_and_config(
    monkeypatch: pytest.MonkeyPatch,
    runner_device: str,
    decode_tf32: bool | None,
    expect_tf32: bool,
) -> None:
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, False
        torch.set_float32_matmul_precision("highest")
        monkeypatch.setattr(personaplex_mimi, "PersonaPlexMimiCodec", _LoadCodec)
        model, _ = _model(install=False, runner_device=runner_device, decode_tf32=decode_tf32)
        model.load_weights(iter([("unused.weight", torch.zeros(1))]))
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == (expect_tf32, expect_tf32)
        assert torch.get_float32_matmul_precision() == ("high" if expect_tf32 else "highest")
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_matmul, old_cudnn
        torch.set_float32_matmul_precision(old_precision)


def test_delta_codes_skip_history_and_emit_only_new_pcm(mocker) -> None:
    model, mimi = _model()
    decode = mocker.patch.object(model, "_decode_pending", return_value=[])
    cat = mocker.spy(personaplex_code2wav.torch, "cat")

    for frame in range(10):
        model(input_ids=_codes(1, start=frame), request_ids=["req"])
    assert decode.call_count == 10
    assert cat.call_count == 0

    first = model(input_ids=_codes(2), request_ids=["req2"])
    second = model(input_ids=_codes(1, start=100), request_ids=["req2"])
    assert _audio(first).tolist() == _pcm(1, 2)
    assert _audio(second).tolist() == _pcm(3)


def test_full_payload_is_consumed_once_across_forwards() -> None:
    model, mimi = _model(async_chunk=False)
    runtime_info = [{"codes": {"audio": _codes(2)}}]

    def forward():
        return model(
            input_ids=torch.zeros(2, dtype=torch.long),
            request_ids=["req"],
            runtime_additional_information=runtime_info,
        )

    first, second = forward(), forward()
    assert _audio(first).tolist() == _pcm(1, 2)
    assert _audio(second).numel() == 0

    model.on_requests_finished({"req"})
    reused = forward()
    assert _audio(reused).tolist() == _pcm(1, 2)


def test_profile_inputs_skip_decode_while_malformed_online_input_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, mimi = _model()
    warnings: list[tuple] = []
    monkeypatch.setattr(personaplex_code2wav.logger, "warning", lambda *args: warnings.append(args))

    out_prof = model(
        input_ids=torch.arange(3), runtime_additional_information=[{"meta": {"personaplex_dummy_profile": True}}]
    )
    out_mal = model(input_ids=torch.arange(3))
    assert _audio(out_prof).numel() == 0 and _audio(out_mal).numel() == 0
    assert len(warnings) == 1 and "not divisible by" in warnings[0][0]


def test_rows_leased_recycled_and_scratch_fallback() -> None:
    model, mimi = _model(max_sessions=2)
    initial = model(
        input_ids=torch.cat([_codes(1), _codes(1)]), request_ids=["first", "second"], seq_token_counts=[2, 2]
    )
    continued = model(
        input_ids=torch.cat([_codes(1, start=10), _codes(1, start=20)]),
        request_ids=["first", "second"],
        seq_token_counts=[2, 2],
    )

    assert _audios(initial) == [_pcm(1), _pcm(101)]
    assert _audios(continued) == [_pcm(2), _pcm(102)]
    assert model._request_rows == {"first": 0, "second": 1}

    model.on_requests_finished(["first"])
    assert mimi.reset_rows == [0]

    # Scratch row for id-less request
    scratch = model(
        input_ids=torch.cat([_codes(1), _codes(1)]),
        runtime_additional_information=[{"request_id": "second"}, {}],
        seq_token_counts=[2, 2],
    )
    assert _audios(scratch) == [_pcm(103), _pcm(101)]


def _step(model, requests: dict[str, int], *, step: int = 0, infos=None):
    """One runner step: each request's ``frames`` new frames, as the runner lays them out."""
    ids = torch.cat([_codes(frames, start=10 * step + 1000 * index) for index, frames in enumerate(requests.values())])
    return model(
        input_ids=ids,
        request_ids=list(requests),
        seq_token_counts=[2 * frames for frames in requests.values()],
        runtime_additional_information=infos,
    )


def test_a_step_decodes_its_request_spans_exactly_like_one_request_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batched, batched_mimi = _model(max_sessions=4)
    per_request, per_request_mimi = _model(max_sessions=4)
    monkeypatch.setattr(per_request, "_decode_input_id_spans", lambda *args: None)
    warnings: list[tuple] = []
    monkeypatch.setattr(personaplex_code2wav.logger, "warning", lambda *args: warnings.append(args))
    steps = [{"a": 1, "b": 3, "c": 2}, {"b": 1, "a": 2}, {"c": 3, "d": 1, "a": 1}]
    for step, requests in enumerate(steps):
        outputs = [_audios(_step(model, requests, step=step)) for model in (batched, per_request)]
        assert outputs[0] == outputs[1]
        if step == 1:
            for model in (batched, per_request):
                model.on_requests_finished(["b"])
    assert batched._request_rows == per_request._request_rows


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_cuda_step_matches_the_cpu_step() -> None:
    cuda_model, _ = _model(max_sessions=4, device="cuda")
    cpu_model, _ = _model(max_sessions=4)
    for step, requests in enumerate([{"a": 1, "b": 3}, {"b": 2, "c": 5, "a": 1}]):
        cuda_output = _step(cuda_model, requests, step=step)
        assert _audios(cuda_output) == _audios(_step(cpu_model, requests, step=step))

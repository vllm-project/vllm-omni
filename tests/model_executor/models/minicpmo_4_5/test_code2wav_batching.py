from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
    BatchedToken2WavState,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    _CFM_STEPS_ENV,
    _FLOW_FP16_ENV,
    MiniCPMO45Code2Wav,
    _parse_token2wav_float16,
    _parse_token2wav_n_timesteps,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []
        self.last_chunk_calls: list[bool] = []

    def forward_chunk(self, xs, last_chunk=False, cnn_cache=None, att_cache=None):
        batch, length, _ = xs.shape
        self.calls.append(batch)
        self.last_chunk_calls.append(last_chunk)
        old_length = 0 if att_cache is None else att_cache.shape[3]
        output = xs[:, : max(1, length - 1)]
        cnn = xs[:, :1, :].transpose(1, 2).contiguous()
        marker = xs[:, 0, 0].reshape(1, batch, 1, 1, 1)
        att = marker.expand(1, batch, 1, old_length + output.shape[1], 1).clone()
        return output, cnn, att


class _FakeBlock:
    def __init__(self):
        conv1 = SimpleNamespace(causal_padding=(1, 0))
        self.conv = SimpleNamespace(
            in_channels=1,
            out_channels=1,
            block=[None, conv1],
        )
        self.attn = SimpleNamespace(num_heads=1, head_dim=1)


class _FakeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [_FakeBlock()]
        self.cfg_batches: list[int] = []
        self.speaker_order: list[list[float]] = []
        self.register_buffer("att_cache_buffer", torch.ones(1), persistent=False)
        self.register_buffer("cnn_cache_buffer", torch.ones(1), persistent=False)

    def t_embedder(self, time):
        return time[:, None]

    def blocks_forward_chunk(
        self,
        inputs,
        time,
        mask,
        cnn_cache,
        att_cache,
        cnn_out,
        att_out,
    ):
        del time, mask, cnn_cache, att_cache
        self.cfg_batches.append(inputs.shape[0])
        self.speaker_order.append(inputs[:, 2, 0].tolist())
        marker = inputs[:, 1, 0]
        cnn_out.copy_(marker.reshape(1, -1, 1, 1).expand_as(cnn_out))
        att_out.copy_(marker.reshape(1, -1, 1, 1, 1).expand_as(att_out))
        return inputs[:, 1:2]


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = _FakeEstimator()
        self.inference_cfg_rate = 0.7
        self.register_buffer("rand_noise", torch.zeros(1, 1, 100), persistent=False)
        self.register_buffer("att_cache_buffer", torch.ones(1), persistent=False)
        self.register_buffer("cnn_cache_buffer", torch.ones(1), persistent=False)


class _FakeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.encoder_proj = nn.Identity()
        self.decoder = _FakeDecoder()
        self.spk_embed_affine_layer = nn.Identity()

    def input_embedding(self, tokens):
        return tokens.to(torch.float32).unsqueeze(-1)


class _FakeHiFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []

    def forward(self, mel, source):
        del source
        self.calls.append(mel.shape[0])
        speech = mel[:, 0].repeat_interleave(3, dim=1)
        generated_source = speech[:, None]
        return speech, generated_source


class _FakeToken2Wav:
    def __init__(self):
        self.flow = _FakeFlow()
        self.hift = _FakeHiFT()
        self.float16 = False
        self.n_timesteps = 2
        self.mel_cache_len = 1
        self.source_cache_len = 2
        self.speech_window = torch.hamming_window(4, periodic=False)
        self.prompt_calls = 0

    def _prepare_prompt(self, prompt_wav):
        del prompt_wav
        self.prompt_calls += 1
        return (
            torch.tensor([[5, 6]], dtype=torch.long),
            torch.tensor([2], dtype=torch.int32),
            torch.ones(1, 1),
            torch.ones(1, 4, 1),
            torch.tensor([4], dtype=torch.int32),
        )

    def stream(self, *args, **kwargs):
        raise AssertionError("sequential stream fallback must never be called")

    def __call__(self, *args, **kwargs):
        raise AssertionError("sequential __call__ fallback must never be called")


def _config(minimum: int = 1, **extra):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="/fake/model",
            stage_connector_config={
                "extra": {
                    "code2wav_min_batch_size": minimum,
                    "prompt_cache_id": "shared",
                    "prompt_wav": "/fake/prompt.wav",
                    **extra,
                }
            },
        )
    )


def _model():
    token2wav = _FakeToken2Wav()
    backend = BatchedToken2Wav(token2wav)
    model = MiniCPMO45Code2Wav(vllm_config=_config())
    model.backend = backend
    return model, token2wav


def test_adapter_releases_unused_upstream_streaming_buffers():
    token2wav = _FakeToken2Wav()
    decoder = token2wav.flow.decoder
    modules = (decoder, decoder.estimator)

    assert all(
        getattr(module, name) is not None for module in modules for name in ("att_cache_buffer", "cnn_cache_buffer")
    )

    BatchedToken2Wav(token2wav)

    assert all(
        name in module._buffers and getattr(module, name) is None
        for module in modules
        for name in ("att_cache_buffer", "cnn_cache_buffer")
    )


def test_code2wav_resolves_hf_model_id_for_assets(mocker, tmp_path):
    resolved_root = tmp_path / "snapshot"
    resolved_root.mkdir()
    config = _config()
    config.model_config.model = "openbmb/MiniCPM-o-4_5"
    config.model_config.revision = "test-revision"
    config.model_config.stage_connector_config["extra"].pop("prompt_wav")
    config.load_config = SimpleNamespace(download_dir="/model-cache")
    model = MiniCPMO45Code2Wav(vllm_config=config)
    mock_download = mocker.patch(
        "vllm_omni.model_executor.model_loader.weight_utils.download_weights_from_hf_specific",
        return_value=str(resolved_root),
    )

    assert model._resolve_model_root() == resolved_root
    assert model.model_path == str(resolved_root)
    assert model._default_prompt_wav == str(resolved_root / "assets" / "HT_ref_audio.wav")
    mock_download.assert_called_once_with(
        "openbmb/MiniCPM-o-4_5",
        "/model-cache",
        allow_patterns=[
            "assets/HT_ref_audio.wav",
            "assets/token2wav/*",
        ],
        revision="test-revision",
        require_all=True,
    )


def _info(
    request_id: str,
    chunk_seq: int,
    codes: list[int],
    *,
    last_chunk: bool = False,
    cache_epoch: int = 0,
):
    return {
        "codes": {"audio": torch.tensor(codes, dtype=torch.long)},
        "meta": {
            "request_id": request_id,
            "chunk_seq": chunk_seq,
            "cache_epoch": cache_epoch,
            "last_chunk": last_chunk,
            "prompt_cache_id": "shared",
        },
    }


def _forward(model, infos, placeholder_counts=None, request_ids=None):
    placeholder_counts = placeholder_counts or [1] * len(infos)
    input_ids = torch.zeros(sum(placeholder_counts), dtype=torch.long)
    return model(
        input_ids=input_ids,
        seq_token_counts=placeholder_counts,
        runtime_additional_information=infos,
        request_ids=request_ids,
    )


def test_adapter_runs_true_batch_cfg_and_splits_request_caches():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    audios, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert token2wav.prompt_calls == 1
    assert token2wav.flow.encoder.calls == [2, 2]
    assert token2wav.flow.decoder.estimator.cfg_batches == [4, 4, 4, 4]
    assert all(order == [1.0, 1.0, 0.0, 0.0] for order in token2wav.flow.decoder.estimator.speaker_order)
    assert token2wav.hift.calls == [2]
    assert len(audios) == 2
    cache0 = states[0].flow_cache["estimator_cnn_cache"]
    cache1 = states[1].flow_cache["estimator_cnn_cache"]
    assert cache0.data_ptr() != cache1.data_ptr()
    assert cache0[0, 0, 0, 0, 0].item() == 10
    assert cache1[0, 0, 0, 0, 0].item() == 20


@pytest.mark.parametrize("steps", [10, 8, 6])
def test_token2wav_n_timesteps_accepts_bounded_quality_grid(steps):
    assert _parse_token2wav_n_timesteps(steps) == steps
    model = MiniCPMO45Code2Wav(vllm_config=_config(token2wav_n_timesteps=steps))
    assert model._token2wav_n_timesteps == steps
    assert model._token2wav_n_timesteps_source == "config"


@pytest.mark.parametrize("value", [True, False, 10.0, 8.0, "10", "8", 0, 1, 7, 9, 11, None])
def test_token2wav_n_timesteps_rejects_non_whitelisted_config_values(value):
    with pytest.raises(ValueError, match="token2wav_n_timesteps must be an integer"):
        MiniCPMO45Code2Wav(vllm_config=_config(token2wav_n_timesteps=value))


def test_numerical_lab_defaults_to_ten_step_fp32(monkeypatch):
    monkeypatch.delenv(_CFM_STEPS_ENV, raising=False)
    monkeypatch.delenv(_FLOW_FP16_ENV, raising=False)

    model = MiniCPMO45Code2Wav(vllm_config=_config())

    assert model._token2wav_n_timesteps == 10
    assert model._token2wav_n_timesteps_source == "default"
    assert model._token2wav_float16 is False
    assert model._token2wav_float16_source == "default"


def test_numerical_lab_environment_switches_and_config_priority(monkeypatch):
    monkeypatch.setenv(_CFM_STEPS_ENV, "6")
    monkeypatch.setenv(_FLOW_FP16_ENV, "yes")

    environment_model = MiniCPMO45Code2Wav(vllm_config=_config())
    config_model = MiniCPMO45Code2Wav(
        vllm_config=_config(
            token2wav_n_timesteps=8,
            token2wav_float16=False,
        )
    )

    assert environment_model._token2wav_n_timesteps == 6
    assert environment_model._token2wav_n_timesteps_source == "environment"
    assert environment_model._token2wav_float16 is True
    assert environment_model._token2wav_float16_source == "environment"
    assert config_model._token2wav_n_timesteps == 8
    assert config_model._token2wav_n_timesteps_source == "config"
    assert config_model._token2wav_float16 is False
    assert config_model._token2wav_float16_source == "config"


@pytest.mark.parametrize("env_name", [_CFM_STEPS_ENV, _FLOW_FP16_ENV])
def test_numerical_lab_rejects_invalid_environment_values(monkeypatch, env_name):
    monkeypatch.delenv(_CFM_STEPS_ENV, raising=False)
    monkeypatch.delenv(_FLOW_FP16_ENV, raising=False)
    monkeypatch.setenv(env_name, "invalid")

    with pytest.raises(ValueError, match=env_name):
        MiniCPMO45Code2Wav(vllm_config=_config())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        (" yes ", True),
        ("on", True),
        ("false", False),
        (" no ", False),
        ("off", False),
    ],
)
def test_token2wav_float16_parses_explicit_boolean_values(value, expected):
    assert _parse_token2wav_float16(value) is expected


@pytest.mark.parametrize("value", [2, -1, 0.0, 1.0, None, "", "enabled", [], {}])
def test_token2wav_float16_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="token2wav_float16 must be"):
        _parse_token2wav_float16(value)


@pytest.mark.parametrize("steps", [10, 8, 6])
def test_cfm_steps_execute_exact_count_and_return_finite_fp32_audio(steps):
    token2wav = _FakeToken2Wav()
    token2wav.n_timesteps = steps
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")

    states = adapter.setup_batch(prompt, 1)
    calls_after_setup = len(token2wav.flow.decoder.estimator.cfg_batches)
    audios, _ = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=True,
    )

    assert calls_after_setup == steps
    assert len(token2wav.flow.decoder.estimator.cfg_batches) == 2 * steps
    assert len(audios) == 1
    assert audios[0].dtype == torch.float32
    assert audios[0].numel() > 0
    assert torch.isfinite(audios[0]).all()


class _AutocastSpy:
    def __init__(self, events, *, enter_error=None):
        self.events = events
        self.enter_error = enter_error

    def __enter__(self):
        self.events.append("autocast-enter")
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.events.append("autocast-exit")
        return False


def _npu_autocast_adapter(monkeypatch, *, enter_error=None):
    adapter = BatchedToken2Wav(_FakeToken2Wav(), npu_flow_float16=True)
    events = []
    monkeypatch.setattr(adapter, "_is_npu_device", lambda _device: True)
    monkeypatch.setattr(
        adapter,
        "_make_npu_autocast",
        lambda: _AutocastSpy(events, enter_error=enter_error),
    )
    return adapter, events


def test_npu_flow_autocast_tracks_effective_fp16(monkeypatch):
    adapter, events = _npu_autocast_adapter(monkeypatch)

    with adapter._npu_flow_autocast(torch.device("cpu")):
        events.append("flow")

    assert events == ["autocast-enter", "flow", "autocast-exit"]
    assert adapter.precision_telemetry() == {
        "requested_dtype": "float16",
        "effective_dtype": "float16",
        "fallback_count": 0,
        "fallback_reason": None,
        "fallback_error_type": None,
    }


def test_npu_flow_autocast_falls_back_once_when_context_entry_is_unsupported(monkeypatch):
    adapter, events = _npu_autocast_adapter(
        monkeypatch,
        enter_error=RuntimeError("npu autocast is not registered"),
    )
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav._autocast_disabled",
        lambda _device: _AutocastSpy(events),
    )

    with adapter._npu_flow_autocast(torch.device("cpu")):
        events.append("first-flow")
    with adapter._npu_flow_autocast(torch.device("cpu")):
        events.append("second-flow")

    assert adapter.precision_telemetry() == {
        "requested_dtype": "float16",
        "effective_dtype": "float32",
        "fallback_count": 1,
        "fallback_reason": "npu_autocast_unavailable",
        "fallback_error_type": "RuntimeError",
    }


def test_npu_flow_autocast_entry_failure_after_fp16_execution_is_fatal(monkeypatch):
    adapter, events = _npu_autocast_adapter(monkeypatch)

    with adapter._npu_flow_autocast(torch.device("cpu")):
        events.append("first-flow")
    monkeypatch.setattr(
        adapter,
        "_make_npu_autocast",
        lambda: _AutocastSpy(events, enter_error=RuntimeError("autocast lost")),
    )

    with pytest.raises(RuntimeError, match="restart the Stage 2 process"):
        with adapter._npu_flow_autocast(torch.device("cpu")):
            pytest.fail("failed autocast entry must not execute Flow")

    assert adapter.precision_telemetry() == {
        "requested_dtype": "float16",
        "effective_dtype": "float16",
        "fallback_count": 0,
        "fallback_reason": None,
        "fallback_error_type": None,
    }


def test_npu_flow_operator_error_propagates_without_precision_fallback(monkeypatch):
    adapter, events = _npu_autocast_adapter(monkeypatch)

    with pytest.raises(RuntimeError, match="flow failed"):
        with adapter._npu_flow_autocast(torch.device("cpu")):
            raise RuntimeError("flow failed")

    assert events == ["autocast-enter", "autocast-exit"]
    assert adapter.precision_telemetry()["fallback_count"] == 0


def test_default_fp32_flow_never_enters_npu_autocast(monkeypatch):
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    monkeypatch.setattr(
        adapter,
        "_npu_flow_autocast",
        lambda _device: pytest.fail("default FP32 path entered NPU autocast"),
    )
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")

    states = adapter.setup_batch(prompt, 1)
    audios, _ = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=True,
    )

    assert audios[0].dtype == torch.float32


def test_npu_flow_fp16_keeps_hift_fp32_and_output_fp32(monkeypatch):
    adapter, events = _npu_autocast_adapter(monkeypatch)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav._autocast_disabled",
        lambda _device: _AutocastSpy(events),
    )
    original_hift = adapter.hift.forward

    def hift_spy(mel, source):
        events.append(("hift", mel.dtype, source.dtype))
        return original_hift(mel, source)

    monkeypatch.setattr(adapter.hift, "forward", hift_spy)
    states = adapter.setup_batch(prompt, 1)
    audios, _ = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        prompt,
        states,
        last_chunk=True,
    )

    hift_events = [event for event in events if isinstance(event, tuple) and event[0] == "hift"]
    assert hift_events == [("hift", torch.float32, torch.float32)]
    assert audios[0].dtype == torch.float32
    assert torch.isfinite(audios[0]).all()


def test_adapter_timeline_context_emits_cfm_and_hift_boundaries(monkeypatch):
    events = []
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav.emit_ultra_timeline_event",
        lambda event, **metadata: events.append((event, metadata)),
    )
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")

    with adapter.timeline_context(["request-a"]):
        states = adapter.setup_batch(prompt, 1)
        adapter.decode_batch(torch.tensor([[10, 11]]), prompt, states, last_chunk=False)

    assert [event for event, _ in events] == [
        "cfm_setup_begin",
        "cfm_setup_end",
        "cfm_begin",
        "cfm_end",
        "hift_begin",
        "hift_end",
    ]
    assert all(metadata["request_id"] == "request-a" for _, metadata in events)
    assert all(metadata["stage"] == 2 for _, metadata in events)


def test_fade_in_out_limits_overlap_to_available_previous_audio():
    speech = torch.arange(6, dtype=torch.float32).reshape(1, -1)
    previous = torch.full((1, 3), 2.0)
    window = torch.hamming_window(8, periodic=False)

    actual = BatchedToken2Wav._fade_in_out(speech, previous, window)

    expected = speech.clone()
    expected[..., :3] = speech[..., :3] * window[:3] + previous * window[-3:]
    torch.testing.assert_close(actual, expected)


def test_estimator_cache_stack_split_round_trip_preserves_cfg_rows():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    _, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    stacked = adapter._stack_flow_cache(states)
    assert stacked["estimator_cnn_cache"].shape[2] == 4
    assert stacked["estimator_att_cache"].shape[2] == 4
    restored = adapter._split_flow_cache(stacked, 2)
    for original, round_tripped in zip(states, restored, strict=True):
        torch.testing.assert_close(
            round_tripped["estimator_cnn_cache"],
            original.flow_cache["estimator_cnn_cache"],
        )
        torch.testing.assert_close(
            round_tripped["estimator_att_cache"],
            original.flow_cache["estimator_att_cache"],
        )


def test_batch1_flow_cache_handoff_reuses_request_owned_storage():
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    cache = {
        "conformer_cnn_cache": torch.randn(1, 2, 3),
        "conformer_att_cache": torch.randn(2, 1, 3, 4),
        "estimator_cnn_cache": torch.randn(2, 3, 2, 4, 5),
        "estimator_att_cache": torch.randn(2, 3, 2, 4, 5, 6),
    }

    request_cache = adapter._split_flow_cache(cache, 1)[0]
    state = BatchedToken2WavState(
        flow_cache=request_cache,
        hift_cache={},
    )
    restacked = adapter._stack_flow_cache([state])

    assert request_cache is not cache
    assert restacked is not request_cache
    for name, original in cache.items():
        assert request_cache[name].data_ptr() == original.data_ptr()
        assert restacked[name].data_ptr() == original.data_ptr()


def test_batch1_decode_preserves_previous_state_and_isolates_next_state():
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    previous = adapter.setup_batch(prompt, 1)[0]
    flow_before = {name: value.clone() for name, value in previous.flow_cache.items()}
    hift_before = {name: value.clone() for name, value in previous.hift_cache.items()}
    tokens = torch.tensor([[10, 11]])

    audios, next_states = adapter.decode_batch(
        tokens,
        prompt,
        [previous],
        last_chunk=False,
    )
    next_state = next_states[0]

    for name, expected in flow_before.items():
        torch.testing.assert_close(previous.flow_cache[name], expected)
        assert next_state.flow_cache[name].data_ptr() != previous.flow_cache[name].data_ptr()
    for name, expected in hift_before.items():
        torch.testing.assert_close(previous.hift_cache[name], expected)
    for cache in next_state.hift_cache.values():
        assert cache.untyped_storage().data_ptr() != audios[0].untyped_storage().data_ptr()
    torch.testing.assert_close(tokens, torch.tensor([[10, 11]]))


def test_batch1_low_copy_matches_generic_batched_row():
    single = BatchedToken2Wav(_FakeToken2Wav())
    batched = BatchedToken2Wav(_FakeToken2Wav())
    single_prompt = single.prepare_prompt("shared", "/fake/prompt.wav")
    batched_prompt = batched.prepare_prompt("shared", "/fake/prompt.wav")

    single_audio, single_states = single.decode_batch(
        torch.tensor([[10, 11]]),
        single_prompt,
        single.setup_batch(single_prompt, 1),
        last_chunk=False,
    )
    batched_audio, batched_states = batched.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        batched_prompt,
        batched.setup_batch(batched_prompt, 2),
        last_chunk=False,
    )

    torch.testing.assert_close(single_audio[0], batched_audio[0])
    for name, value in single_states[0].flow_cache.items():
        torch.testing.assert_close(value, batched_states[0].flow_cache[name])
    for name, value in single_states[0].hift_cache.items():
        torch.testing.assert_close(value, batched_states[0].hift_cache[name])


def test_singleton_segment_and_codec_batch_are_views(monkeypatch):
    flat = torch.tensor([10, 11, 12])
    monkeypatch.setattr(
        torch,
        "split",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("singleton split must be skipped")),
    )
    monkeypatch.setattr(
        torch,
        "stack",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("singleton stack must be skipped")),
    )

    segment = MiniCPMO45Code2Wav._split_segments(flat, [3])[0]
    item = SimpleNamespace(tokens=segment)
    batched = MiniCPMO45Code2Wav._batch_codec_tokens([item])

    assert segment.data_ptr() == flat.data_ptr()
    assert batched.data_ptr() == flat.data_ptr()
    assert tuple(batched.shape) == (1, 3)


def test_model_preserves_output_slots_and_prefers_runtime_codes():
    model, token2wav = _model()
    output = _forward(
        model,
        [_info("a", 0, [10, 11]), _info("b", 0, [20, 21])],
        placeholder_counts=[3, 1],
    )

    audios = output.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert len(output.multimodal_outputs["sr"]) == 2
    assert all(sr.item() == 24000 for sr in output.multimodal_outputs["sr"])
    assert all(audio.dtype == torch.float32 for audio in audios)
    # Fake CFM uses two Euler steps whose deltas sum to one. Its conditional
    # row is mu and its unconditional row is zero, so CFG produces 1.7 * mu.
    torch.testing.assert_close(audios[0][0], torch.tensor(1.7 * 10))
    torch.testing.assert_close(audios[1][0], torch.tensor(1.7 * 20))
    assert token2wav.flow.encoder.calls[-1] == 2


def test_code2wav_timeline_marks_first_pcm_and_last_audio(monkeypatch):
    events = []
    model, _ = _model()
    model._ultra_timeline_enabled = True
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav.emit_ultra_timeline_event",
        lambda event, **metadata: events.append((event, metadata)),
    )

    _forward(model, [_info("a", 0, [10, 11])])
    _forward(model, [_info("a", 1, [12, 13], last_chunk=True)])

    assert [event for event, _ in events] == ["first_pcm_ready", "pcm_ready", "last_audio_ready"]
    assert events[0][1]["chunk_id"] == 0
    assert events[-1][1]["chunk_id"] == 1
    assert events[-1][1]["details"]["cache_epoch"] == 0


def test_code2wav_projects_duplex_metadata_to_final_audio_output():
    model, token2wav = _model()
    segment = _info("duplex", 0, [10, 11])
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    segment["meta"].update(
        {
            "duplex_epoch": 3,
            "duplex_turn_id": 7,
            "llm_output_text_utf8": segment_text_utf8,
            "tts_is_last_chunk": True,
            "turn_end": False,
        }
    )

    segment_output = _forward(model, [segment])

    assert segment_output.multimodal_outputs["meta.turn_end"][0].item() is False
    # A Talker unit boundary only drains pending codec tokens. The official
    # streaming path keeps Token2wav open until the assistant turn ends.
    assert token2wav.flow.encoder.last_chunk_calls[-1] is False
    assert "duplex" in model._states

    final = _info("duplex", 1, [12, 13], last_chunk=True)
    final["meta"].update(segment["meta"])
    final["meta"]["chunk_seq"] = 1
    final["meta"]["last_chunk"] = True
    final["meta"]["turn_end"] = True
    output = _forward(model, [final])

    payload = output.multimodal_outputs
    assert "meta" not in payload
    assert payload["meta.duplex_epoch"][0].item() == 3
    assert payload["meta.duplex_turn_id"][0].item() == 7
    torch.testing.assert_close(
        payload["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert payload["meta.tts_is_last_chunk"][0].item() is True
    assert payload["meta.turn_end"][0].item() is True
    assert token2wav.flow.encoder.last_chunk_calls[-1] is True
    assert "duplex" not in model._states


def test_initial_empty_segment_marker_initializes_stream_without_audio():
    model, token2wav = _model()
    boundary = _info("duplex", 0, [])
    boundary["meta"].update(
        {
            "code_flat_numel": 0,
            "tts_is_last_chunk": True,
            "turn_end": False,
        }
    )

    output = _forward(model, [boundary])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert "duplex" in model._states
    assert token2wav.hift.calls == []

    resumed = _info(
        "duplex",
        1,
        [4218, 4218, 4218, 10, 11, 12, 13, 14],
    )
    output = _forward(model, [resumed])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert "duplex" in model._states


def test_init_only_prepares_state_without_audio_or_codec_progress():
    model, token2wav = _model()
    init = _info("early", 0, [])
    init["meta"].update(
        {
            "code_flat_numel": 0,
            "init_only": True,
        }
    )

    init_output = _forward(model, [init])

    assert init_output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert init_output.multimodal_outputs["meta.init_only"][0].item() is True
    assert token2wav.hift.calls == []
    assert model._states["early"].chunk_seq == -1
    setup_calls = list(token2wav.flow.encoder.calls)

    audio_output = _forward(model, [_info("early", 0, [10, 11])])

    assert audio_output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert audio_output.multimodal_outputs["meta.init_only"][0].item() is False
    assert model._states["early"].chunk_seq == 0
    assert token2wav.flow.encoder.calls == [*setup_calls, 1]


def test_duplicate_init_only_is_rejected_and_cleanup_releases_state():
    model, _ = _model()
    init = _info("early", 0, [])
    init["meta"].update({"code_flat_numel": 0, "init_only": True})
    _forward(model, [init])

    with pytest.raises(RuntimeError, match="duplicate_init_only"):
        _forward(model, [init])

    model.on_requests_finished(["early"])
    assert "early" not in model._states


def test_init_only_runtime_reference_is_released_on_abort(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    init = _info("voice", 0, [])
    init["codes"]["ref"] = torch.tensor([0.0, 0.25, -0.25, 0.0])
    init["meta"].update(
        {
            "code_flat_numel": 0,
            "init_only": True,
            "ref_audio_sr": 16000,
        }
    )
    init["meta"].pop("prompt_cache_id")

    _forward(model, [init], request_ids=["internal-voice"])

    prompt_key = model._request_prompt_keys["internal-voice"]
    prompt_path = Path(model._runtime_prompts[prompt_key].path)
    assert prompt_path.is_file()

    model.on_requests_finished(["internal-voice"])

    assert "internal-voice" not in model._states
    assert prompt_key not in model._runtime_prompts
    assert not prompt_path.exists()


def test_init_only_rejects_terminal_or_nonempty_payload():
    model, _ = _model()
    terminal = _info("early", 0, [], last_chunk=True)
    terminal["meta"].update({"code_flat_numel": 0, "init_only": True})
    nonempty = _info("other", 0, [10])
    nonempty["meta"]["init_only"] = True

    with pytest.raises(RuntimeError, match="invalid_init_only_payload"):
        _forward(model, [terminal])
    with pytest.raises(RuntimeError, match="invalid_init_only_payload"):
        _forward(model, [nonempty])


def test_shared_runtime_prompt_recreates_missing_file_before_second_owner(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    first = _info("voice-a", 0, [10, 11])
    first["codes"]["ref"] = reference
    first["meta"]["ref_audio_sr"] = 16000
    first["meta"].pop("prompt_cache_id")
    _forward(model, [first], request_ids=["internal-a"])

    prompt_key = model._request_prompt_keys["internal-a"]
    prompt_path = Path(model._runtime_prompts[prompt_key].path)
    prompt_path.unlink()

    second = _info("voice-b", 0, [12, 13])
    second["codes"]["ref"] = reference
    second["meta"]["ref_audio_sr"] = 16000
    second["meta"].pop("prompt_cache_id")
    _forward(model, [second], request_ids=["internal-b"])

    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"internal-a", "internal-b"}

    model.on_requests_finished(["internal-a"])
    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"internal-b"}

    model.on_requests_finished(["internal-b"])
    assert not prompt_path.exists()
    assert prompt_key not in model._runtime_prompts


def test_runtime_prompt_write_failure_does_not_publish_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def fail_after_partial_write(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav.sf.write",
        fail_after_partial_write,
    )

    with pytest.raises(OSError, match="simulated write failure"):
        model._materialize_runtime_prompt(reference, 16000)

    assert len(model._runtime_prompts) == 1
    entry = next(iter(model._runtime_prompts.values()))
    assert not Path(entry.path).exists()
    assert list(Path(entry.path).parent.iterdir()) == []


def test_runtime_prompt_files_are_isolated_between_model_instances(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    first_model, _ = _model()
    second_model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def runtime_ref_info(request_id: str):
        info = _info(request_id, 0, [10, 11])
        info["codes"]["ref"] = reference
        info["meta"]["ref_audio_sr"] = 16000
        info["meta"].pop("prompt_cache_id")
        return info

    _forward(first_model, [runtime_ref_info("voice-a")], request_ids=["internal-a"])
    _forward(second_model, [runtime_ref_info("voice-b")], request_ids=["internal-b"])

    first_key = first_model._request_prompt_keys["internal-a"]
    second_key = second_model._request_prompt_keys["internal-b"]
    first_path = Path(first_model._runtime_prompts[first_key].path)
    second_path = Path(second_model._runtime_prompts[second_key].path)
    assert first_key == second_key
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()

    first_model.on_requests_finished(["internal-a"])
    assert not first_path.exists()
    assert second_path.is_file()

    second_model.on_requests_finished(["internal-b"])
    assert not second_path.exists()


def test_mixed_final_exact_buckets_keep_order_and_release_only_final_states():
    model, _ = _model()
    _forward(
        model,
        [_info(name, 0, [index + 1, index + 2]) for index, name in enumerate(("a", "b", "c", "d"))],
    )
    output = _forward(
        model,
        [
            _info("a", 1, [11, 12]),
            _info("c", 1, [31, 32, 33], last_chunk=True),
            _info("b", 1, [21, 22]),
            _info("d", 1, [41, 42, 43], last_chunk=True),
        ],
    )

    audios = output.multimodal_outputs["model_outputs"]
    window = torch.hamming_window(4, periodic=False)
    overlap_scale = 1.7 * (window[0] + window[2])
    expected = torch.tensor([1, 3, 2, 4], dtype=torch.float32) * overlap_scale
    actual = torch.stack([audio[0] for audio in audios])
    torch.testing.assert_close(actual, expected)
    assert set(model._states) == {"a", "b"}


def test_empty_final_sentinel_emits_empty_and_releases_state_without_compute():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    hift_calls = list(token2wav.hift.calls)
    output = _forward(
        model,
        [
            _info("a", 1, [], last_chunk=True),
            _info("b", 1, [], last_chunk=True),
        ],
    )

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}
    assert token2wav.hift.calls == hift_calls


def test_empty_final_ignores_generation_scheduler_placeholder_token():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    infos = [_info("a", 1, [], last_chunk=True), _info("b", 1, [], last_chunk=True)]
    for info in infos:
        info.pop("codes")
        info["meta"]["code_flat_numel"] = 0

    output = _forward(model, infos, placeholder_counts=[1, 1])

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}


@pytest.mark.parametrize(
    "info",
    [
        # The runner injects the engine request id on every step (GPU
        # _preprocess, NPU _gather_runtime_additional_information)...
        {"request_id": "a", "meta": {"request_id": "a"}},
        # ...but a pre-warm step can also reach the model with nothing at all.
        {},
    ],
)
def test_prewarm_placeholder_step_emits_silence_without_touching_state(info):
    # async-chunk pre-warm submits Stage 2 with a reserved placeholder prompt.
    # If it gets scheduled before the first codec window lands, those reserved
    # tokens must neither be vocoded nor held to the codec payload contract.
    model, token2wav = _model()

    output = _forward(model, [info], request_ids=["a"])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert model._states == {}
    assert token2wav.hift.calls == []


def test_metadata_only_payload_still_decodes_codec_from_prompt_tokens():
    # The connector strips 1-D codec tensors out of additional_information and
    # leaves them in the prompt tokens, so a real chunk reaches the model as
    # producer metadata plus input ids. It must still be vocoded.
    model, _ = _model()
    info = {
        "request_id": "a",
        "meta": {
            "request_id": "a",
            "chunk_seq": 0,
            "code_flat_numel": 2,
            "prompt_cache_id": "shared",
        },
    }

    output = _forward(model, [info], placeholder_counts=[2])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert set(model._states) == {"a"}


def test_non_final_chunk_shorter_than_lookahead_window_is_rejected():
    token2wav = _FakeToken2Wav()
    token2wav.flow.encoder.pre_lookahead_layer = SimpleNamespace(pre_lookahead_len=3)
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)

    with pytest.raises(RuntimeError, match="chunk_below_lookahead_window"):
        adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=False)

    # The final chunk is zero-padded by the encoder, so it stays decodable.
    audios, _ = adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=True)
    assert len(audios) == 1


def test_forward_builds_backend_when_weight_loading_was_skipped(monkeypatch):
    # load_format=dummy never calls load_weights(), so Stage 2 would otherwise
    # reach its first request with no Token2wav assets at all.
    model = MiniCPMO45Code2Wav(vllm_config=_config())
    token2wav = _FakeToken2Wav()
    builds = 0

    def build_backend():
        nonlocal builds
        builds += 1
        model.backend = BatchedToken2Wav(token2wav)

    monkeypatch.setattr(model, "_build_backend", build_backend)

    output = _forward(model, [_info("a", 0, [10, 11])])
    _forward(model, [_info("a", 1, [12, 13])])

    assert builds == 1
    assert output.multimodal_outputs["model_outputs"][0].numel() > 0


@pytest.mark.parametrize(
    ("info", "reason"),
    [
        (_info("a", 0, [1, 2], cache_epoch=-1), "negative_stream_position"),
        (_info("a", 0, [1, 2]), "stale_or_reordered_chunk"),
        (_info("a", 2, [1, 2]), "stale_or_reordered_chunk"),
    ],
)
def test_stale_epoch_and_reordered_chunks_are_rejected(info, reason):
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])

    with pytest.raises(RuntimeError, match=reason):
        _forward(model, [info, _info("b", 1, [3, 4])])


def test_singleton_and_mixed_shape_buckets_use_same_batched_backend_without_fallback():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    output = _forward(model, [_info("a", 1, [5, 6]), _info("b", 1, [7, 8, 9])])

    assert len(output.multimodal_outputs["model_outputs"]) == 2
    # Exact-shape buckets execute independently but both use the same vectorized
    # adapter; there is no Token2wav.stream/__call__ fallback.
    assert token2wav.hift.calls[-2:] == [1, 1]


def test_backend_failure_does_not_commit_any_request_state(monkeypatch):
    model, _ = _model()
    _forward(
        model,
        [_info(name, 0, [index + 1, index + 2]) for index, name in enumerate(("a", "b", "c", "d"))],
    )
    before = dict(model._states)
    original = model.backend.decode_batch
    call_count = 0

    def fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backend, "decode_batch", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        _forward(
            model,
            [
                _info("a", 1, [5, 6]),
                _info("b", 1, [7, 8]),
                _info("c", 1, [9, 10, 11]),
                _info("d", 1, [12, 13, 14]),
            ],
        )
    assert call_count == 2
    assert model._states == before


def test_cleanup_and_profile_output_are_aligned():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    model.on_requests_finished(["a"])
    assert set(model._states) == {"b"}

    profile = model(
        input_ids=torch.zeros(5, dtype=torch.long),
        seq_token_counts=[2, 3],
    )
    assert [audio.numel() for audio in profile.multimodal_outputs["model_outputs"]] == [0, 0]
    assert set(model._states) == {"b"}


def test_cleanup_uses_generation_runner_internal_request_ids():
    model, _ = _model()
    _forward(
        model,
        [_info("external-a", 0, [1, 2]), _info("external-b", 0, [3, 4])],
        request_ids=["internal-a", "internal-b"],
    )

    model.on_requests_finished(["internal-a"])

    assert set(model._states) == {"internal-b"}


def test_reference_voice_and_duplex_metadata_follow_request_lifecycle():
    model, _ = _model()
    first = _info("voice-a", 0, [1, 2])
    first["codes"]["ref"] = torch.linspace(-0.1, 0.1, 160)
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    first["meta"].update(
        ref_audio_sr=16000,
        llm_output_text_utf8=segment_text_utf8,
        duplex_turn_id=7,
        duplex_epoch=3,
    )
    first["meta"].pop("prompt_cache_id")

    output = _forward(model, [first])
    prompt_key = model._request_prompt_keys["voice-a"]
    prompt = model._runtime_prompts[prompt_key]
    prompt_cache_id, prompt_wav = prompt.cache_id, prompt.path
    assert prompt_cache_id.startswith("runtime-ref-")
    assert Path(prompt_wav).is_file()
    torch.testing.assert_close(
        output.multimodal_outputs["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert output.multimodal_outputs["meta.duplex_turn_id"][0].item() == 7
    assert output.multimodal_outputs["meta.duplex_epoch"][0].item() == 3

    final = _info("voice-a", 1, [3, 4], last_chunk=True)
    final["meta"].pop("prompt_cache_id")
    final["meta"]["tts_is_last_chunk"] = True
    output = _forward(model, [final])

    assert output.multimodal_outputs["meta.tts_is_last_chunk"][0].item() is True
    assert model._request_prompt_keys["voice-a"] == prompt_key
    model.on_requests_finished(["voice-a"])
    assert "voice-a" not in model._request_prompt_keys
    assert prompt_key not in model._runtime_prompts
    assert not Path(prompt_wav).exists()
    assert (prompt_cache_id, prompt_wav) not in model.backend._prompt_features

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.ltx2.ltx2_audio_cuda_graph import (
    LTX2AudioCUDAGraphConfig,
    LTX2AudioCUDAGraphRunner,
    make_ltx2_audio_graph_key,
)
from vllm_omni.diffusion.models.ltx2.ltx2_audio_transformer import LTX2AudioStaticConditioning


def test_config_defaults_and_preserves_unrelated_additional_config():
    assert LTX2AudioCUDAGraphConfig.from_additional_config(None) == LTX2AudioCUDAGraphConfig()
    assert LTX2AudioCUDAGraphConfig.from_additional_config({"unrelated": {"enabled": True}}) == (
        LTX2AudioCUDAGraphConfig()
    )
    assert LTX2AudioCUDAGraphConfig.from_additional_config(
        {
            "unrelated": {"enabled": True},
            "ltx2_audio_cuda_graph": {"enabled": True, "max_entries": 8},
        }
    ) == LTX2AudioCUDAGraphConfig(enabled=True, max_entries=8)


def test_config_accepts_strictly_increasing_audio_length_buckets():
    config = LTX2AudioCUDAGraphConfig.from_additional_config(
        {
            "ltx2_audio_cuda_graph": {
                "enabled": True,
                "max_entries": 4,
                "audio_length_buckets": [1, 5.0, 10],
            }
        }
    )

    assert config.audio_length_buckets == (1.0, 5.0, 10.0)


@pytest.mark.parametrize(
    ("additional_config", "error", "message"),
    [
        (True, TypeError, "additional_config must be a mapping"),
        ({"ltx2_audio_cuda_graph": None}, TypeError, "must be a mapping"),
        ({"ltx2_audio_cuda_graph": True}, TypeError, "must be a mapping"),
        ({"ltx2_audio_cuda_graph": {"enable": True}}, ValueError, "Unknown"),
        ({"ltx2_audio_cuda_graph": {"enabled": 1}}, TypeError, "enabled must be a bool"),
        ({"ltx2_audio_cuda_graph": {"max_entries": True}}, ValueError, "positive integer"),
        ({"ltx2_audio_cuda_graph": {"max_entries": 0}}, ValueError, "positive integer"),
        ({"ltx2_audio_cuda_graph": {"max_entries": "4"}}, ValueError, "positive integer"),
        (
            {"ltx2_audio_cuda_graph": {"enabled": True, "audio_length_buckets": "1,5"}},
            TypeError,
            "list or tuple",
        ),
        (
            {"ltx2_audio_cuda_graph": {"enabled": True, "audio_length_buckets": [1.0, 1.0]}},
            ValueError,
            "strictly increasing",
        ),
        (
            {"ltx2_audio_cuda_graph": {"enabled": True, "audio_length_buckets": [0.0]}},
            ValueError,
            "finite and positive",
        ),
        (
            {"ltx2_audio_cuda_graph": {"enabled": False, "audio_length_buckets": [1.0]}},
            ValueError,
            "enabled=true",
        ),
    ],
)
def test_config_rejects_invalid_model_options(additional_config, error, message):
    with pytest.raises(error, match=message):
        LTX2AudioCUDAGraphConfig.from_additional_config(additional_config)


def _cpu_inputs(*, tokens: int = 3, context_tokens: int = 2, mask: bool = False, perturb: bool = False):
    batch = 2
    return {
        "audio_hidden_states": torch.randn(batch, tokens, 4, dtype=torch.bfloat16),
        "audio_encoder_hidden_states": torch.randn(batch, context_tokens, 4, dtype=torch.bfloat16),
        "audio_timestep": torch.randn(batch, tokens),
        "audio_sigma": torch.randn(batch),
        "audio_coords": torch.randn(batch, 1, tokens, 2),
        "audio_attention_mask": torch.ones(batch, tokens, dtype=torch.bool) if mask else None,
        "perturbation_mask": torch.ones(batch, 1, 1, dtype=torch.bfloat16) if perturb else None,
        "stg_blocks": [28] if perturb else None,
    }


def _key(inputs):
    return make_ltx2_audio_graph_key(
        inputs["audio_hidden_states"],
        inputs["audio_encoder_hidden_states"],
        inputs["audio_attention_mask"],
        inputs["perturbation_mask"],
        inputs["stg_blocks"],
    )


def test_graph_key_uses_structure_not_values():
    first = _cpu_inputs(mask=True, perturb=True)
    second = _cpu_inputs(mask=True, perturb=True)
    assert _key(first) == _key(second)
    assert _key(first).audio_token_count == 3
    assert _key(first).context_token_count == 2


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("tokens", 4),
        ("context_tokens", 3),
        ("mask", True),
        ("perturb", True),
    ],
)
def test_graph_key_changes_for_structural_fields(change, value):
    base = _cpu_inputs()
    changed = _cpu_inputs(**{change: value})
    assert _key(base) != _key(changed)


def test_graph_key_canonicalizes_stg_blocks():
    inputs = _cpu_inputs(perturb=True)
    inputs["stg_blocks"] = [28, 4, 28]
    first = _key(inputs)
    inputs["stg_blocks"] = [4, 28]
    assert first == _key(inputs)
    assert first.stg_blocks == (4, 28)


class _EagerTransformer:
    def __call__(self, **kwargs):
        return kwargs["audio_hidden_states"] + 1


def test_incompatible_inputs_fall_back_without_failed_key():
    runner = LTX2AudioCUDAGraphRunner(_EagerTransformer(), device="cpu")
    inputs = _cpu_inputs()
    output = runner(**inputs)
    torch.testing.assert_close(output, inputs["audio_hidden_states"] + 1)
    assert output.shape == (2, 3, 4)
    assert runner.last_call_info["reason"] == "incompatible_inputs"
    assert runner.stats_snapshot()["eager"] == 1
    assert runner.stats_snapshot()["eager_incompatible_inputs"] == 1
    assert runner.stats_snapshot()["failed_key_count"] == 0


def test_tp_capture_scope_registers_distributed_graph_buffers(monkeypatch):
    runner = LTX2AudioCUDAGraphRunner(_EagerTransformer(), device="cuda")
    entered = []

    @contextmanager
    def fake_graph_capture(*, device):
        entered.append(device)
        yield

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.ltx2.ltx2_audio_cuda_graph.get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.ltx2.ltx2_audio_cuda_graph.graph_capture",
        fake_graph_capture,
    )

    with runner._tensor_parallel_capture_scope():
        assert entered == [torch.device("cuda")]


def test_mock_cache_hit_lru_and_eviction(monkeypatch):
    runner = LTX2AudioCUDAGraphRunner(_EagerTransformer(), max_graphs=2, device="cuda")
    monkeypatch.setattr(runner, "_inputs_are_compatible", lambda **_kwargs: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    captures = []

    def capture(**kwargs):
        captures.append(tuple(kwargs["hidden_states"].shape))
        tensor = kwargs["hidden_states"].clone()
        return (
            SimpleNamespace(
                graph=SimpleNamespace(replay=lambda: None),
                static_hidden_states=tensor,
                static_context=kwargs["context"].clone(),
                static_timestep=kwargs["timestep"].clone(),
                static_sigma=kwargs["sigma"].clone(),
                static_coords=kwargs["coords"].clone(),
                static_rotary_cos=None,
                static_rotary_sin=None,
                static_attention_mask=None,
                static_perturbation_mask=None,
                static_output=tensor,
            ),
            tensor,
        )

    monkeypatch.setattr(runner, "_capture", capture)
    one = _cpu_inputs(tokens=3)
    two = _cpu_inputs(tokens=4)
    three = _cpu_inputs(tokens=5)
    runner(**one)
    runner(**two)
    runner(**one)
    runner(**three)
    assert captures == [(2, 3, 4), (2, 4, 4), (2, 5, 4)]
    stats = runner.stats_snapshot()
    assert stats["hits"] == 1
    assert stats["captures"] == 3
    assert stats["evictions"] == 1
    assert list(runner._cache)[0] == _key(one)


def test_cache_miss_returns_warmup_output_without_replay(monkeypatch):
    runner = LTX2AudioCUDAGraphRunner(_EagerTransformer(), device="cuda")
    monkeypatch.setattr(runner, "_inputs_are_compatible", lambda **_kwargs: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = _cpu_inputs()
    replay = Mock()
    captured_output = inputs["audio_hidden_states"] + 1

    def capture(**kwargs):
        return (
            SimpleNamespace(
                graph=SimpleNamespace(replay=replay),
                static_hidden_states=kwargs["hidden_states"].clone(),
                static_context=kwargs["context"].clone(),
                static_timestep=kwargs["timestep"].clone(),
                static_sigma=kwargs["sigma"].clone(),
                static_coords=kwargs["coords"].clone(),
                static_rotary_cos=None,
                static_rotary_sin=None,
                static_attention_mask=None,
                static_perturbation_mask=None,
                static_output=torch.empty_like(captured_output),
            ),
            captured_output,
        )

    monkeypatch.setattr(runner, "_capture", capture)

    before = runner.stats_snapshot()
    output = runner(**inputs)

    replay.assert_not_called()
    torch.testing.assert_close(output, captured_output)
    assert output.data_ptr() != captured_output.data_ptr()
    assert runner.last_call_info["mode"] == "capture"

    runner(**inputs)

    replay.assert_called_once_with()
    delta = runner.record_request_stats(before)
    assert delta["calls"] == 2
    assert delta["captures"] == 1
    assert delta["hits"] == 1
    assert delta["eager"] == 0
    assert delta["cache_size"] == 1
    assert runner.last_request_stats == delta


def test_request_scope_copies_static_conditioning_once_per_request(monkeypatch):
    runner = LTX2AudioCUDAGraphRunner(_EagerTransformer(), device="cuda")
    monkeypatch.setattr(runner, "_inputs_are_compatible", lambda **_kwargs: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def capture(**kwargs):
        tensor = kwargs["hidden_states"].clone()
        return (
            SimpleNamespace(
                graph=SimpleNamespace(replay=lambda: None),
                static_hidden_states=tensor,
                static_context=kwargs["context"].clone(),
                static_timestep=kwargs["timestep"].clone(),
                static_sigma=kwargs["sigma"].clone(),
                static_coords=kwargs["coords"].clone(),
                static_rotary_cos=None,
                static_rotary_sin=None,
                static_attention_mask=None,
                static_perturbation_mask=None,
                static_output=tensor,
                request_generation=None,
            ),
            tensor,
        )

    monkeypatch.setattr(runner, "_capture", capture)
    first = _cpu_inputs()
    changed = {**first, "audio_encoder_hidden_states": first["audio_encoder_hidden_states"] + 5}

    with runner.request_scope():
        runner(**first)
        runner(**changed)
        entry = runner._cache[_key(first)]
        torch.testing.assert_close(entry.static_context, first["audio_encoder_hidden_states"])

    assert runner.last_request_stats["calls"] == 2
    assert runner.last_request_stats["captures"] == 1
    assert runner.last_request_stats["hits"] == 1

    with runner.request_scope():
        runner(**changed)
        torch.testing.assert_close(entry.static_context, changed["audio_encoder_hidden_states"])

    assert runner.last_request_stats["calls"] == 1
    assert runner.last_request_stats["captures"] == 0
    assert runner.last_request_stats["hits"] == 1


def test_capture_failure_is_bounded_and_not_retried(monkeypatch):
    runner = LTX2AudioCUDAGraphRunner(_EagerTransformer(), max_graphs=1, device="cuda")
    monkeypatch.setattr(runner, "_inputs_are_compatible", lambda **_kwargs: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attempts = 0

    def fail(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("capture failed")

    monkeypatch.setattr(runner, "_capture", fail)
    first = _cpu_inputs(tokens=3)
    second = _cpu_inputs(tokens=4)
    runner(**first)
    runner(**first)
    runner(**second)
    assert attempts == 2
    stats = runner.stats_snapshot()
    assert stats["capture_failures"] == 2
    assert stats["eager_capture_failure"] == 2
    assert stats["eager_previous_capture_failure"] == 1
    assert stats["failed_key_count"] == 1


def test_clear_resets_lifecycle(monkeypatch):
    runner = LTX2AudioCUDAGraphRunner(_EagerTransformer(), device="cpu")
    runner._pool = object()
    runner._stats["calls"] = 2
    runner._failed_keys[_key(_cpu_inputs())] = None
    runner.clear()
    assert runner._pool is None
    assert runner.stats_snapshot()["calls"] == 0
    assert runner.stats_snapshot()["failed_key_count"] == 0
    assert runner.last_request_stats == {}


def test_active_outer_capture_uses_eager_without_mutating_cache(monkeypatch):
    runner = LTX2AudioCUDAGraphRunner(_EagerTransformer(), device="cuda")
    monkeypatch.setattr(runner, "_inputs_are_compatible", lambda **_kwargs: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    capture = Mock(side_effect=AssertionError("must not capture"))
    monkeypatch.setattr(runner, "_capture", capture)
    inputs = _cpu_inputs()

    output = runner(**inputs)

    torch.testing.assert_close(output, inputs["audio_hidden_states"] + 1)
    assert runner.last_call_info == {"mode": "eager", "reason": "active_capture"}
    assert runner.stats_snapshot()["captures"] == 0
    assert runner.stats_snapshot()["cache_size"] == 0
    assert runner.stats_snapshot()["eager_active_capture"] == 1
    capture.assert_not_called()


class _TinyAudioTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16, device="cuda")

    def forward(
        self,
        audio_hidden_states,
        audio_encoder_hidden_states,
        audio_timestep,
        *,
        audio_sigma,
        audio_coords,
        audio_attention_mask=None,
        audio_static_conditioning=None,
        attention_kwargs=None,
    ):
        if audio_static_conditioning is not None:
            audio_encoder_hidden_states = audio_static_conditioning.encoder_hidden_states
        out = self.proj(audio_hidden_states)
        out = out + audio_encoder_hidden_states.mean(dim=1, keepdim=True)
        out = out + audio_timestep.unsqueeze(-1).to(out.dtype) * 0.01
        out = out + audio_sigma[:, None, None].to(out.dtype) * 0.02
        if audio_static_conditioning is None:
            position = audio_coords[:, 0, :, :1]
        else:
            position = audio_static_conditioning.rotary_emb[0][:, 0, :, :1]
        out = out + position.to(out.dtype) * 0.03
        if audio_attention_mask is not None:
            out = out * audio_attention_mask.unsqueeze(-1)
        perturbation = (attention_kwargs or {}).get("ltx_perturbation_kwargs", {})
        if perturbation:
            out = out + perturbation["audio_self_attention_mask"] * 0.04
        return out


def _cuda_inputs(*, tokens=3, value=1.0, mask=True, perturb=True):
    batch = 2
    return {
        "audio_hidden_states": torch.full((batch, tokens, 4), value, dtype=torch.bfloat16, device="cuda"),
        "audio_encoder_hidden_states": torch.full((batch, 2, 4), value + 1, dtype=torch.bfloat16, device="cuda"),
        "audio_timestep": torch.full((batch, tokens), value + 2, dtype=torch.float32, device="cuda"),
        "audio_sigma": torch.full((batch,), value + 3, dtype=torch.float32, device="cuda"),
        "audio_coords": torch.full((batch, 1, tokens, 2), value + 4, dtype=torch.float32, device="cuda"),
        "audio_attention_mask": torch.ones(batch, tokens, dtype=torch.bool, device="cuda") if mask else None,
        "perturbation_mask": (
            torch.full((batch, 1, 1), value + 5, dtype=torch.bfloat16, device="cuda") if perturb else None
        ),
        "stg_blocks": [28] if perturb else None,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_matches_eager_replays_values_and_owns_output():
    torch.manual_seed(7)
    transformer = _TinyAudioTransformer().eval()
    runner = LTX2AudioCUDAGraphRunner(transformer, max_graphs=2)
    first_inputs = _cuda_inputs(value=1)
    second_inputs = _cuda_inputs(value=2)
    with torch.inference_mode():
        first_eager = runner._call_transformer(
            hidden_states=first_inputs["audio_hidden_states"],
            context=first_inputs["audio_encoder_hidden_states"],
            timestep=first_inputs["audio_timestep"],
            sigma=first_inputs["audio_sigma"],
            coords=first_inputs["audio_coords"],
            attention_mask=first_inputs["audio_attention_mask"],
            perturbation_mask=first_inputs["perturbation_mask"],
            stg_blocks=(28,),
        ).clone()
        first_graph = runner(**first_inputs)
        preserved = first_graph.clone()
        second_eager = runner._call_transformer(
            hidden_states=second_inputs["audio_hidden_states"],
            context=second_inputs["audio_encoder_hidden_states"],
            timestep=second_inputs["audio_timestep"],
            sigma=second_inputs["audio_sigma"],
            coords=second_inputs["audio_coords"],
            attention_mask=second_inputs["audio_attention_mask"],
            perturbation_mask=second_inputs["perturbation_mask"],
            stg_blocks=(28,),
        ).clone()
        second_graph = runner(**second_inputs)
    torch.accelerator.synchronize("cuda")
    torch.testing.assert_close(first_graph, first_eager, rtol=0, atol=0)
    torch.testing.assert_close(second_graph, second_eager, rtol=0, atol=0)
    torch.testing.assert_close(first_graph, preserved, rtol=0, atol=0)
    assert first_graph.data_ptr() != runner._cache[_key(first_inputs)].static_output.data_ptr()
    assert runner.stats_snapshot()["captures"] == 1
    assert runner.stats_snapshot()["hits"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_replays_request_scoped_static_conditioning():
    transformer = _TinyAudioTransformer().eval()
    runner = LTX2AudioCUDAGraphRunner(transformer, max_graphs=1)
    first_inputs = _cuda_inputs(value=1)
    second_inputs = _cuda_inputs(value=2)

    def conditioning(inputs, value):
        batch, tokens, _ = inputs["audio_hidden_states"].shape
        rotary = torch.full((batch, 1, tokens, 2), value, dtype=torch.bfloat16, device="cuda")
        return LTX2AudioStaticConditioning(
            encoder_hidden_states=inputs["audio_encoder_hidden_states"],
            rotary_emb=(rotary, rotary + 1),
        )

    first_conditioning = conditioning(first_inputs, 6)
    second_conditioning = conditioning(second_inputs, 7)
    with torch.inference_mode():
        first_expected = runner._call_transformer(
            hidden_states=first_inputs["audio_hidden_states"],
            context=first_conditioning.encoder_hidden_states,
            timestep=first_inputs["audio_timestep"],
            sigma=first_inputs["audio_sigma"],
            coords=None,
            attention_mask=first_inputs["audio_attention_mask"],
            perturbation_mask=first_inputs["perturbation_mask"],
            stg_blocks=(28,),
            rotary_cos=first_conditioning.rotary_emb[0],
            rotary_sin=first_conditioning.rotary_emb[1],
        ).clone()
        first_actual = runner(**first_inputs, audio_static_conditioning=first_conditioning)
        second_expected = runner._call_transformer(
            hidden_states=second_inputs["audio_hidden_states"],
            context=second_conditioning.encoder_hidden_states,
            timestep=second_inputs["audio_timestep"],
            sigma=second_inputs["audio_sigma"],
            coords=None,
            attention_mask=second_inputs["audio_attention_mask"],
            perturbation_mask=second_inputs["perturbation_mask"],
            stg_blocks=(28,),
            rotary_cos=second_conditioning.rotary_emb[0],
            rotary_sin=second_conditioning.rotary_emb[1],
        ).clone()
        second_actual = runner(**second_inputs, audio_static_conditioning=second_conditioning)
    torch.accelerator.synchronize("cuda")

    torch.testing.assert_close(first_actual, first_expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second_actual, second_expected, rtol=0.0, atol=0.0)
    assert runner.stats_snapshot()["captures"] == 1
    assert runner.stats_snapshot()["hits"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_private_pool_supports_two_serial_signatures():
    transformer = _TinyAudioTransformer().eval()
    runner = LTX2AudioCUDAGraphRunner(transformer, max_graphs=2)
    short = _cuda_inputs(tokens=3, value=1, mask=False, perturb=False)
    long = _cuda_inputs(tokens=5, value=2, mask=False, perturb=False)
    with torch.inference_mode():
        short_first = runner(**short)
        long_first = runner(**long)
        short_second = runner(**short)
    torch.accelerator.synchronize("cuda")
    torch.testing.assert_close(short_first, short_second, rtol=0, atol=0)
    assert long_first.shape == (2, 5, 4)
    assert runner.stats_snapshot()["captures"] == 2
    assert runner.stats_snapshot()["hits"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_can_capture_and_replay_from_sequential_python_threads():
    transformer = _TinyAudioTransformer().eval()
    runner = LTX2AudioCUDAGraphRunner(transformer, max_graphs=1)
    inputs = [_cuda_inputs(value=1), _cuda_inputs(value=2)]
    outputs: list[torch.Tensor | None] = [None, None]
    errors: list[BaseException] = []

    def run(index: int) -> None:
        try:
            with torch.inference_mode():
                outputs[index] = runner(**inputs[index])
            torch.accelerator.synchronize("cuda")
        except BaseException as exc:  # noqa: BLE001 - propagate thread failures
            errors.append(exc)

    first = threading.Thread(target=run, args=(0,))
    first.start()
    first.join()
    second = threading.Thread(target=run, args=(1,))
    second.start()
    second.join()

    assert not errors
    with torch.inference_mode():
        expected = [
            runner._call_transformer(
                hidden_states=value["audio_hidden_states"],
                context=value["audio_encoder_hidden_states"],
                timestep=value["audio_timestep"],
                sigma=value["audio_sigma"],
                coords=value["audio_coords"],
                attention_mask=value["audio_attention_mask"],
                perturbation_mask=value["perturbation_mask"],
                stg_blocks=(28,),
            )
            for value in inputs
        ]
    torch.accelerator.synchronize("cuda")
    assert outputs[0] is not None and outputs[1] is not None
    torch.testing.assert_close(outputs[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], expected[1], rtol=0, atol=0)
    assert runner.stats_snapshot()["captures"] == 1
    assert runner.stats_snapshot()["hits"] == 1
    assert runner.stats_snapshot()["eager"] == 0

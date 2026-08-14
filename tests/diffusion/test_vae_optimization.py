# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
import torch.nn as nn

import vllm_omni.diffusion.vae_optimization as vae_optimization
from vllm_omni.diffusion.vae_optimization import (
    configure_vae_runtime,
    finalize_vae_stage_durations,
    get_vae_optimization_capabilities,
    prepare_vae_optimization_config,
    resolve_vae_optimization,
    setup_vae_compile,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class _ParallelConfig:
    vae_parallel_mode: str = "tile"


@dataclass
class _Config:
    model_class_name: str = "MiniMaxH3Pipeline"
    vae_optimization_profile: str = "safe"
    vae_stack_tiling: str | bool | None = None
    vae_compile: str | bool | None = None
    vae_compile_max_shape_buckets: int = 4
    vae_use_tiling: bool = False
    enable_diffusion_pipeline_profiler: bool = False
    enforce_eager: bool = False
    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)


def _config(**overrides: Any) -> _Config:
    config = _Config()
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_h3_declares_complete_runtime_capabilities():
    capabilities = get_vae_optimization_capabilities("MiniMaxH3Pipeline")

    assert capabilities.tiled_decode
    assert capabilities.stacked_tiles
    assert capabilities.compilation
    assert capabilities.independent_process_group
    assert not capabilities.spatial_sharding


@pytest.mark.parametrize(
    ("profile", "stack_tiling", "compile_mode", "diagnostics"),
    [
        ("safe", "false", "false", False),
        ("optimized", "auto", "auto", False),
        ("diagnostic", "false", "false", True),
    ],
)
def test_profile_defaults(profile, stack_tiling, compile_mode, diagnostics):
    settings = resolve_vae_optimization(_config(vae_optimization_profile=profile))

    assert settings.stack_tiling == stack_tiling
    assert settings.compile == compile_mode
    assert settings.diagnostics is diagnostics


def test_prepare_profile_enables_required_runtime_features():
    config = _config(vae_optimization_profile="optimized")

    settings = prepare_vae_optimization_config(config)

    assert settings.stack_tiling == "auto"
    assert config.vae_use_tiling is True
    assert config.vae_stack_tiling == "auto"
    assert config.vae_compile == "auto"


def test_diagnostic_profile_enables_component_profiler_before_load():
    config = _config(vae_optimization_profile="diagnostic")

    prepare_vae_optimization_config(config)

    assert config.enable_diffusion_pipeline_profiler is True


def test_safe_profile_rejects_fast_path_override():
    with pytest.raises(ValueError, match="profile='safe'"):
        resolve_vae_optimization(_config(vae_stack_tiling="true"))


def test_explicit_unsupported_feature_fails_validation():
    with pytest.raises(ValueError, match="does not declare stacked-tile"):
        resolve_vae_optimization(
            _config(
                model_class_name="UnknownPipeline",
                vae_optimization_profile="diagnostic",
                vae_stack_tiling="true",
            )
        )


def test_unsupported_auto_feature_falls_back_to_eager():
    settings = resolve_vae_optimization(
        _config(
            model_class_name="UnknownPipeline",
            vae_optimization_profile="optimized",
        )
    )

    assert settings.stack_tiling == "false"
    assert settings.compile == "false"


def test_spatial_sharding_fails_for_h3_during_startup_validation():
    with pytest.raises(ValueError, match="does not declare VAE 'spatial_shard_height'"):
        resolve_vae_optimization(_config(parallel_config=_ParallelConfig(vae_parallel_mode="spatial_shard_height")))


def test_student_profile_requires_model_specific_artifact():
    with pytest.raises(ValueError, match="post-trained decoder artifact"):
        resolve_vae_optimization(_config(vae_optimization_profile="student"))


def test_vae_compile_is_independent_from_eager_dit():
    settings = resolve_vae_optimization(
        _config(
            vae_optimization_profile="diagnostic",
            vae_compile="true",
            enforce_eager=True,
        )
    )

    assert settings.compile == "true"


class _FakeRemoteModel:
    def __init__(
        self,
        *,
        fail_stacked_once: bool = False,
        failure_type: type[Exception] = RuntimeError,
        fail_collective_once: bool = False,
        collective_failure_type: type[Exception] = RuntimeError,
    ):
        self.stack_tiling = False
        self.vae_ratio = 8
        self.fail_stacked_once = fail_stacked_once
        self.failure_type = failure_type
        self.fail_collective_once = fail_collective_once
        self.collective_failure_type = collective_failure_type
        self.calls: list[bool] = []

    def split_tiles(self, length, is_decoder=False):
        del is_decoder
        count = 4 if length > 8 else 1
        return list(range(count)), [length] * count, [0] * max(0, count - 1)

    def tiled_decode(self, value):
        return value + 1

    def _run_tile_tasks(self, value):
        return value + 2

    def _all_gather_tiled_results(self, value):
        if self.fail_collective_once:
            self.fail_collective_once = False
            raise self.collective_failure_type("synthetic tile collective failure")
        return value + 3


class _FakeVideoVae:
    def __init__(self, remote_model, *, parallel_size=2):
        self.model = remote_model
        self.parallel_size = parallel_size

    def decode_latent(self, latent):
        stacked = self.model.stack_tiling
        self.model.calls.append(stacked)
        if stacked and self.model.fail_stacked_once:
            self.model.fail_stacked_once = False
            raise self.model.failure_type("synthetic stacked allocation failure")
        self.model._all_gather_tiled_results(latent)
        return latent + 1


class _FakeAudioVae:
    def decode_latent(self, latent):
        return latent + 2


@dataclass
class _Pipeline:
    od_config: _Config
    video_vae: _FakeVideoVae
    audio_vae: _FakeAudioVae
    enable_diffusion_pipeline_profiler: bool
    _profiler_lock: Any = None
    _stage_durations: dict[str, float] = field(default_factory=dict)


def _pipeline(
    config,
    *,
    fail_stacked_once=False,
    failure_type=RuntimeError,
    fail_collective_once=False,
    collective_failure_type=RuntimeError,
):
    remote = _FakeRemoteModel(
        fail_stacked_once=fail_stacked_once,
        failure_type=failure_type,
        fail_collective_once=fail_collective_once,
        collective_failure_type=collective_failure_type,
    )
    return _Pipeline(
        od_config=config,
        video_vae=_FakeVideoVae(remote),
        audio_vae=_FakeAudioVae(),
        enable_diffusion_pipeline_profiler=config.enable_diffusion_pipeline_profiler,
    )


def test_stacked_tiling_auto_uses_validated_multi_tile_shape(monkeypatch):
    config = _config(vae_optimization_profile="optimized", vae_compile="false")
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    configure_vae_runtime(pipeline, config)
    result = pipeline.video_vae.decode_latent(torch.zeros(1, 2, 2, 2, 2))

    assert torch.equal(result, torch.ones(1, 2, 2, 2, 2))
    assert pipeline.video_vae.model.calls == [True]
    assert pipeline._vae_stack_tiling_controller.last_decision["stacked"] is True
    assert pipeline._vae_stack_tiling_controller.last_decision["decision"] == "validated"
    assert pipeline.video_vae.model.stack_tiling is False


def test_runtime_configuration_is_idempotent(monkeypatch):
    config = _config(vae_optimization_profile="optimized", vae_compile="false")
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    configure_vae_runtime(pipeline, config)
    controller = pipeline._vae_stack_tiling_controller
    configure_vae_runtime(pipeline, config)
    pipeline.video_vae.decode_latent(torch.zeros(1, 2, 2, 2, 2))

    assert pipeline._vae_stack_tiling_controller is controller
    assert pipeline.video_vae.model.calls == [True]


def test_explicit_stack_mode_still_requires_multiple_local_tiles():
    config = _config(
        vae_optimization_profile="diagnostic",
        vae_stack_tiling="true",
        vae_compile="false",
    )
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(config)
    pipeline.video_vae.parallel_size = 32
    configure_vae_runtime(pipeline, config)

    pipeline.video_vae.decode_latent(torch.zeros(1, 2, 2, 2, 2))

    assert pipeline.video_vae.model.calls == [False]
    assert pipeline._vae_last_diagnostics["decision"] == "fewer_than_two_local_tiles"


def test_stack_mode_falls_back_before_decode_when_memory_is_insufficient(monkeypatch):
    config = _config(
        vae_optimization_profile="diagnostic",
        vae_stack_tiling="true",
        vae_compile="false",
    )
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(config)
    monkeypatch.setattr(vae_optimization._StackedTileController, "_has_memory_headroom", lambda *args: False)
    configure_vae_runtime(pipeline, config)

    pipeline.video_vae.decode_latent(torch.zeros(1, 2, 2, 2, 2))

    assert pipeline.video_vae.model.calls == [False]
    assert pipeline._vae_last_diagnostics["decision"] == "insufficient_memory_headroom"


def test_stacked_tiling_pre_collective_oom_retries_eager_and_does_not_poison_next_request(monkeypatch):
    config = _config(vae_optimization_profile="optimized", vae_compile="false")
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(config, fail_stacked_once=True, failure_type=torch.OutOfMemoryError)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    configure_vae_runtime(pipeline, config)
    latent = torch.zeros(1, 2, 2, 2, 2)

    first = pipeline.video_vae.decode_latent(latent)
    second = pipeline.video_vae.decode_latent(latent)

    assert torch.equal(first, latent + 1)
    assert torch.equal(second, latent + 1)
    assert pipeline.video_vae.model.calls == [True, False, True]
    assert pipeline._vae_stack_tiling_controller.fallbacks == 1
    assert pipeline.video_vae.model.stack_tiling is False


def test_stacked_tiling_oom_propagates_when_pre_collective_state_cannot_be_proved(monkeypatch):
    config = _config(vae_optimization_profile="optimized", vae_compile="false")
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(config, fail_stacked_once=True, failure_type=torch.OutOfMemoryError)
    monkeypatch.setattr(vae_optimization._StackedTileController, "_install_collective_marker", lambda self: None)
    configure_vae_runtime(pipeline, config)

    with pytest.raises(torch.OutOfMemoryError, match="synthetic stacked allocation failure"):
        pipeline.video_vae.decode_latent(torch.zeros(1, 2, 2, 2, 2))

    assert pipeline.video_vae.model.calls == [True]
    assert pipeline._vae_stack_tiling_controller.fallbacks == 0


def test_non_oom_stacked_runtime_failure_propagates_without_replay(monkeypatch):
    config = _config(vae_optimization_profile="optimized", vae_compile="false")
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(config, fail_stacked_once=True, failure_type=RuntimeError)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    configure_vae_runtime(pipeline, config)
    latent = torch.zeros(1, 2, 2, 2, 2)

    with pytest.raises(RuntimeError, match="synthetic stacked allocation failure"):
        pipeline.video_vae.decode_latent(latent)

    assert pipeline.video_vae.model.calls == [True]
    assert pipeline._vae_stack_tiling_controller.fallbacks == 0
    assert pipeline.video_vae.model.stack_tiling is False


@pytest.mark.parametrize("failure_type", [RuntimeError, torch.OutOfMemoryError])
def test_tile_collective_failure_propagates_without_unsafe_replay(monkeypatch, failure_type):
    config = _config(vae_optimization_profile="optimized", vae_compile="false")
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(
        config,
        fail_collective_once=True,
        collective_failure_type=failure_type,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    configure_vae_runtime(pipeline, config)
    latent = torch.zeros(1, 2, 2, 2, 2)

    with pytest.raises(failure_type, match="synthetic tile collective failure"):
        pipeline.video_vae.decode_latent(latent)
    second = pipeline.video_vae.decode_latent(latent)

    assert torch.equal(second, latent + 1)
    assert pipeline.video_vae.model.calls == [True, True]
    assert pipeline._vae_stack_tiling_controller.fallbacks == 0


def test_component_observability_and_tile_merge_derivation(monkeypatch):
    config = _config(vae_optimization_profile="diagnostic")
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(config)
    monkeypatch.setattr(vae_optimization.current_omni_platform, "is_available", lambda: False)
    configure_vae_runtime(pipeline, config)

    pipeline.video_vae.decode_latent(torch.zeros(1, 2, 1, 1, 1))
    pipeline.audio_vae.decode_latent(torch.zeros(1))
    pipeline.video_vae.model.tiled_decode(1)
    pipeline.video_vae.model._run_tile_tasks(1)
    pipeline.video_vae.model._all_gather_tiled_results(1)
    durations = finalize_vae_stage_durations(pipeline._stage_durations)

    assert durations["video_vae.decode_latent"] >= 0
    assert durations["audio_vae.decode_latent"] >= 0
    assert durations["video_vae.tile_decode"] >= 0
    assert durations["video_vae.tile_communication"] >= 0
    assert durations["video_vae.tile_merge"] >= 0
    assert pipeline._vae_last_diagnostics["decision"] == "disabled"
    assert len(pipeline._vae_last_diagnostics["latent_sha256"]) == 64


def test_diagnostic_latent_fingerprint_is_stable_across_requests(monkeypatch):
    config = _config(vae_optimization_profile="diagnostic")
    prepare_vae_optimization_config(config)
    pipeline = _pipeline(config)
    monkeypatch.setattr(vae_optimization.current_omni_platform, "is_available", lambda: False)
    configure_vae_runtime(pipeline, config)
    latent = torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 2, 2, 2)

    pipeline.video_vae.decode_latent(latent)
    first = dict(pipeline._vae_last_diagnostics)
    pipeline.video_vae.decode_latent(latent.clone())
    second = dict(pipeline._vae_last_diagnostics)

    assert first["latent_sha256"] == second["latent_sha256"]
    assert first["cold"] is True
    assert second["cold"] is False


def test_diagnostic_latent_fingerprint_binds_shape_and_dtype():
    float_tensor = torch.zeros(1, dtype=torch.float32)
    byte_tensor = torch.zeros(4, dtype=torch.uint8)

    assert float_tensor.numel() * float_tensor.element_size() == byte_tensor.numel() * byte_tensor.element_size()
    assert vae_optimization._latent_sha256(float_tensor) != vae_optimization._latent_sha256(byte_tensor)


class TransformerBlock(nn.Module):
    def forward(self, value):
        return value + 1


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = TransformerBlock()


@dataclass
class _DecoderOwner:
    decoder: _FakeDecoder


@dataclass
class _CompileVideoVae:
    model: _DecoderOwner


@dataclass
class _CompilePipeline:
    video_vae: _CompileVideoVae
    _vae_optimization_settings: Any


def _compile_pipeline(decoder: _FakeDecoder, settings: Any) -> _CompilePipeline:
    return _CompilePipeline(
        video_vae=_CompileVideoVae(model=_DecoderOwner(decoder=decoder)),
        _vae_optimization_settings=settings,
    )


def test_bounded_compile_falls_back_per_shape_bucket(monkeypatch):
    config = _config(
        vae_optimization_profile="diagnostic",
        vae_compile="true",
        vae_compile_max_shape_buckets=1,
    )
    settings = prepare_vae_optimization_config(config)
    decoder = _FakeDecoder()
    pipeline = _compile_pipeline(decoder, settings)
    compile_calls = []

    def compile_that_fails(function, dynamic=False):
        compile_calls.append((function, dynamic))

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic compile failure")

        return fail

    monkeypatch.setattr(vae_optimization.current_omni_platform, "supports_torch_inductor", lambda: True)
    monkeypatch.setattr(torch, "compile", compile_that_fails)

    assert setup_vae_compile(pipeline) == 1
    first = decoder.block(torch.zeros(1))
    second = decoder.block(torch.zeros(2))
    third = decoder.block(torch.zeros(1))

    assert torch.equal(first, torch.ones(1))
    assert torch.equal(second, torch.ones(2))
    assert torch.equal(third, torch.ones(1))
    assert len(compile_calls) == 1
    assert len(decoder.block.forward._vllm_omni_failed_buckets) == 1


def test_synchronous_compile_failure_falls_back_and_is_not_retried(monkeypatch):
    config = _config(vae_optimization_profile="diagnostic", vae_compile="true")
    settings = prepare_vae_optimization_config(config)
    decoder = _FakeDecoder()
    pipeline = _compile_pipeline(decoder, settings)
    calls = 0

    def compile_that_raises(function, dynamic=False):
        del function, dynamic
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic synchronous compile failure")

    monkeypatch.setattr(vae_optimization.current_omni_platform, "supports_torch_inductor", lambda: True)
    monkeypatch.setattr(torch, "compile", compile_that_raises)

    setup_vae_compile(pipeline)
    first = decoder.block(torch.zeros(1))
    second = decoder.block(torch.zeros(1))

    assert torch.equal(first, torch.ones(1))
    assert torch.equal(second, torch.ones(1))
    assert calls == 1

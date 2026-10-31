# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

import vllm_omni
from vllm_omni.config.stage_config import resolve_deploy_yaml
from vllm_omni.model_executor.models.voxcpm2.runtime_config import (
    _VoxCPM2RuntimeConfig,
    reject_outer_full_cudagraph,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeOmniPlatform:
    """Deterministic platform stand-in so cadence defaults are assertable on any runner.

    Also counts ``is_cuda()`` calls so the tests can assert the fake was
    actually consulted: from_vllm_config resolves the platform at call time
    today, and these guards fail loudly if it ever switches to a module-level
    import that would silently bypass the faked platform.
    """

    def __init__(self, is_cuda: bool) -> None:
        self._is_cuda = is_cuda
        self.is_cuda_calls = 0

    def is_cuda(self) -> bool:
        self.is_cuda_calls += 1
        return self._is_cuda


@pytest.mark.parametrize(
    ("uses_v2", "is_cuda", "expected"),
    [
        (False, False, (2, 1)),
        (False, True, (2, 1)),
        (True, False, (2, 1)),
        (True, True, (1, 3)),
    ],
)
def test_runner_specific_audio_defaults(monkeypatch, uses_v2, is_cuda, expected):
    # Incremental VAE is a CUDA-only per-patch feature: CUDA v2 gets the MRV2
    # audio profile (1/3); every other combination keeps the v1 cadence (2/1).
    # The base deploy yaml no longer pins the audio cadence keys, so the
    # (2, 1) dataclass defaults ARE the shipped v1 behavior — changing them
    # silently changes every deployment that does not pin the keys itself.
    import vllm_omni.platforms as platforms_mod

    # Patch the backing global, not the lazy attribute: assigning
    # current_omni_platform writes a real module attribute that permanently
    # shadows the lazy __getattr__ after monkeypatch restores it.
    # An earlier test in the same process may have resolved the real platform
    # into a module attribute (a direct assignment); delete that residue so
    # the from-import falls back to the lazy __getattr__ and consults the
    # fake below. Without this, the real CUDA platform answers and the fake
    # records zero is_cuda() calls (buildkite 18587).
    monkeypatch.delattr(platforms_mod, "current_omni_platform", raising=False)
    fake_platform = _FakeOmniPlatform(is_cuda)
    monkeypatch.setattr(platforms_mod, "_current_omni_platform", fake_platform)
    config = _VoxCPM2RuntimeConfig.from_vllm_config(
        SimpleNamespace(model_config=SimpleNamespace(use_v2_model_runner=uses_v2))
    )
    assert config.enable_incremental_vae is (uses_v2 and is_cuda)
    assert (config.vae_decode_every, config.audio_emit_every) == expected
    if uses_v2:
        # The v2 cadence depends on the platform read at call time; assert the
        # fake was consulted so a module-level import can't silently bypass it.
        assert fake_platform.is_cuda_calls >= 1
    else:
        # The v1 cadence must stay platform-independent: from_vllm_config
        # should not consult the platform on the legacy path at all.
        assert fake_platform.is_cuda_calls == 0


def test_mrv2_explicit_legacy_audio_override():
    config = _VoxCPM2RuntimeConfig.from_vllm_config(
        SimpleNamespace(
            model_config=SimpleNamespace(
                use_v2_model_runner=True,
                voxcpm2_runtime_config={"enable_incremental_vae": False, "vae_decode_every": 3, "audio_emit_every": 1},
            )
        )
    )
    assert not config.enable_incremental_vae
    assert config.vae_decode_every == 3
    assert config.audio_emit_every == 1


def _runtime_config_from_deploy_yaml(yaml_path: Path, use_v2_model_runner: bool) -> _VoxCPM2RuntimeConfig:
    merged = resolve_deploy_yaml(yaml_path)
    runtime_dict = next(
        stage["engine_extras"]["hf_overrides"]["voxcpm2_runtime_config"]
        for stage in merged["stages"]
        if "voxcpm2_runtime_config" in stage.get("engine_extras", {}).get("hf_overrides", {})
    )
    return _VoxCPM2RuntimeConfig.from_vllm_config(
        SimpleNamespace(
            model_config=SimpleNamespace(use_v2_model_runner=use_v2_model_runner, voxcpm2_runtime_config=runtime_dict)
        )
    )


def test_deploy_yaml_resolves_effective_audio_defaults():
    # The base yaml does not pin the audio cadence keys; the dataclass
    # defaults decide for v1 and the shipped MRV2 overlay re-pins them for v2.
    deploy_dir = Path(vllm_omni.__file__).resolve().parent / "deploy"
    v1 = _runtime_config_from_deploy_yaml(deploy_dir / "voxcpm2.yaml", use_v2_model_runner=False)
    assert v1.vae_decode_every == 2
    assert v1.audio_emit_every == 1

    v2 = _runtime_config_from_deploy_yaml(deploy_dir / "voxcpm2_mrv2.yaml", use_v2_model_runner=True)
    assert v2.vae_decode_every == 1
    assert v2.audio_emit_every == 3


@pytest.mark.parametrize(
    ("is_cuda", "expected"),
    [
        (False, (2, 1)),  # non-CUDA v2 keeps the v1 cadence (incremental VAE is CUDA-only)
        (True, (1, 3)),  # CUDA v2 must not inherit v1 cadence values from the base yaml
    ],
)
def test_v2_overlay_on_base_yaml_uses_runner_defaults(monkeypatch, tmp_path, is_cuda, expected):
    # A thin overlay built on the base yaml (base_config + model_runner: v2,
    # no explicit cadence keys) must resolve the audio cadence through
    # runtime_config.py per platform, not inherit stale base-yaml pins.
    import vllm_omni.platforms as platforms_mod

    # An earlier test in the same process may have resolved the real platform
    # into a module attribute (a direct assignment); delete that residue so
    # the from-import falls back to the lazy __getattr__ and consults the
    # fake below. Without this, the real CUDA platform answers and the fake
    # records zero is_cuda() calls (buildkite 18587).
    monkeypatch.delattr(platforms_mod, "current_omni_platform", raising=False)
    fake_platform = _FakeOmniPlatform(is_cuda)
    monkeypatch.setattr(platforms_mod, "_current_omni_platform", fake_platform)
    deploy_yaml = Path(vllm_omni.__file__).resolve().parent / "deploy" / "voxcpm2.yaml"
    overlay = tmp_path / "voxcpm2_v2_overlay.yaml"
    overlay.write_text(f"base_config: {deploy_yaml}\nmodel_runner: v2\n")

    config = _runtime_config_from_deploy_yaml(overlay, use_v2_model_runner=True)
    assert (config.vae_decode_every, config.audio_emit_every) == expected
    # Same call-time guard as the defaults test: the fake platform must be
    # the one from_vllm_config consults.
    assert fake_platform.is_cuda_calls >= 1


@dataclass
class _ModelConfig:
    voxcpm2_runtime_config: dict[str, object]


@dataclass
class _VllmConfig:
    model_config: _ModelConfig


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("inference_timesteps", 4, 4),
        ("inference_timesteps", "6", 6),
        ("inference_timesteps", -3, 2),
        ("inference_timesteps", 0, 2),
        ("inference_timesteps", 1, 2),
        ("inference_timesteps", 2, 2),
        ("cfg_value", "1.5", 1.5),
        ("cfg_cutoff_ratio", "0.5", 0.5),
        ("cfg_cutoff_ratio", -1.0, 0.0),
        ("cfg_cutoff_ratio", 2.0, 1.0),
    ],
)
def test_generation_config(key: str, value: object, expected: int | float) -> None:
    vllm_config = _VllmConfig(_ModelConfig({key: value}))

    config = _VoxCPM2RuntimeConfig.from_vllm_config(vllm_config)

    assert getattr(config, key) == expected


def test_generation_config_defaults() -> None:
    config = _VoxCPM2RuntimeConfig()

    assert config.inference_timesteps == 10
    assert config.cfg_value == 2.0
    assert config.cfg_cutoff_ratio == 1.0
    assert config.mixed_graph_cache_limit == 0


def test_mixed_graph_cache_limit_rejects_negative():
    with pytest.raises(ValueError, match="mixed_graph_cache_limit"):
        _VoxCPM2RuntimeConfig.from_vllm_config(_VllmConfig(_ModelConfig({"mixed_graph_cache_limit": -1})))


def test_incremental_vae_disables_legacy_batching_without_conflict_warning(monkeypatch) -> None:
    from vllm_omni.model_executor.models.voxcpm2 import runtime_config

    warnings = []
    monkeypatch.setattr(runtime_config.logger, "warning", lambda *args: warnings.append(args))
    config = _VoxCPM2RuntimeConfig(
        enable_incremental_vae=True, vae_decode_every=1, audio_emit_every=3, enable_batched_vae_decode=True
    )._normalized()
    assert config.enable_incremental_vae
    assert not config.enable_batched_vae_decode
    assert config.audio_emit_every == 3
    assert not warnings


class _CountingEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x, mu, t, cond, dt):
        self.calls += 1
        return mu[:, :1, None].expand_as(x)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cfg_value_rejects_nonfinite(value: str) -> None:
    with pytest.raises(ValueError, match="cfg_value must be finite"):
        _VoxCPM2RuntimeConfig.from_vllm_config(_VllmConfig(_ModelConfig({"cfg_value": value})))


@pytest.mark.parametrize("timesteps", [-3, 0, 1, 2, 4, 10])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("cfg_value", [1.5, 2.0])
@pytest.mark.parametrize("cutoff", [0.0, 1.0])
def test_generation_config_runs_estimator(timesteps: int, batch_size: int, cfg_value: float, cutoff: float) -> None:
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        _CFMBufferManager,
        _optimized_solve_euler_with_noise,
    )

    config = _VoxCPM2RuntimeConfig.from_vllm_config(
        _VllmConfig(
            _ModelConfig({"inference_timesteps": timesteps, "cfg_value": cfg_value, "cfg_cutoff_ratio": cutoff})
        )
    )
    cfm = torch.nn.Module()
    cfm.estimator = _CountingEstimator()
    buffers = _CFMBufferManager(
        device=torch.device("cpu"),
        dtype=torch.float32,
        feat_dim=2,
        patch_size=2,
        dit_hidden_size=2,
        max_batch_size=batch_size,
    )
    noise = torch.ones(batch_size, 2, 2)
    output = _optimized_solve_euler_with_noise(
        cfm,
        mu=torch.ones(batch_size, 2),
        patch_size=2,
        cond=torch.zeros_like(noise),
        noise=noise,
        n_timesteps=config.inference_timesteps,
        cfg_value=config.cfg_value,
        buffers=buffers,
        cfg_cutoff_ratio=config.cfg_cutoff_ratio,
    )

    assert cfm.estimator.calls == max(2, timesteps) - 1
    assert torch.isfinite(output).all()
    assert not torch.equal(output, noise)
    torch.testing.assert_close(noise, torch.ones_like(noise))
    if cutoff == 1.0:
        t_span = buffers.get_t_span(config.inference_timesteps)
        expected = noise - cfg_value * (t_span[1] - t_span[-1])
        torch.testing.assert_close(output, expected)


def test_exact_patch_batch_requires_incremental_unified_decode():
    with pytest.raises(ValueError, match="Exact patch batches"):
        _VoxCPM2RuntimeConfig(enable_exact_patch_batch=True)._normalized()
    config = _VoxCPM2RuntimeConfig(
        enable_exact_patch_batch=True, enable_incremental_vae=True, vae_decode_every=1
    )._normalized()
    assert config.enable_exact_patch_batch


def _graph_config(mode, *, uses_v2=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(use_v2_model_runner=uses_v2),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
    )


@pytest.mark.parametrize("mode_name", ["FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"])
def test_mrv1_rejects_outer_full_cudagraph(mode_name):
    with pytest.raises(ValueError, match=f"cudagraph_mode={mode_name}.*PIECEWISE"):
        reject_outer_full_cudagraph(_graph_config(CUDAGraphMode[mode_name]))


@pytest.mark.parametrize(
    ("mode_name", "uses_v2"),
    [("NONE", False), ("PIECEWISE", False), ("FULL_AND_PIECEWISE", True)],
)
def test_supported_graph_modes_pass(mode_name, uses_v2):
    reject_outer_full_cudagraph(_graph_config(CUDAGraphMode[mode_name], uses_v2=uses_v2))

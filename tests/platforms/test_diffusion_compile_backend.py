# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion compile backend unit tests with mocked optional dependencies."""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@dataclass
class _BackendConfig:
    diffusion_compile_backend: str = "auto"
    diffusion_compile_aclgraph: bool = False


@dataclass
class _ParallelConfig:
    world_size: int = 1
    sequence_parallel_size: int = 1
    use_hsdp: bool = False


@dataclass
class _CompileConfig:
    diffusion_compile_backend: str = "mindiesd"
    model_class_name: str = "QwenImagePipeline"
    diffusion_compile_granularity: str = "regional"
    diffusion_compile_dynamic: bool = False
    diffusion_compile_aclgraph: bool = False
    dtype: torch.dtype = torch.bfloat16
    quantization_config: object | None = None
    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    enable_distributed_layerwise_offload: bool = False
    cache_backend: str | None = "none"
    lora_path: str | None = None
    enable_sleep_mode: bool = False


@dataclass
class _FusionPatterns:
    enable_wan_residual_gate: bool = True
    enable_qwen_residual_gate: bool = True


@pytest.fixture
def npu_platform(monkeypatch):
    """Load the Omni NPU class without importing the Ascend runtime."""
    ascend = ModuleType("vllm_ascend.platform")
    setattr(ascend, "NPUPlatform", type("NPUPlatform", (), {}))
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", ascend)

    path = Path(__file__).resolve().parents[2] / "vllm_omni/platforms/npu/platform.py"
    spec = importlib.util.spec_from_file_location("_test_npu_compile_platform", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NPUOmniPlatform


@pytest.fixture
def compile_config():
    return _CompileConfig()


@pytest.fixture
def mindiesd(monkeypatch):
    module = ModuleType("mindiesd.compilation")

    class CompilationConfig:
        aclgraph_only = False
        aclgraph_with_compile = False
        aclgraph_lazy_capture = False
        safe_output_mode = True
        fusion_patterns = _FusionPatterns()

    backend = Mock()
    setattr(module, "CompilationConfig", CompilationConfig)
    setattr(module, "MindieSDBackend", Mock(return_value=backend))
    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", module)
    return module


@pytest.mark.parametrize(
    "platform",
    [OmniPlatformEnum.CUDA, OmniPlatformEnum.ROCM, OmniPlatformEnum.XPU, OmniPlatformEnum.MUSA],
)
@pytest.mark.parametrize("requested", ["auto", "inductor"])
def test_inductor_platforms_keep_backend(monkeypatch, platform, requested):
    monkeypatch.setattr(OmniPlatform, "_omni_enum", platform, raising=False)
    monkeypatch.setattr(OmniPlatform, "supports_torch_inductor", lambda: True)

    assert OmniPlatform.get_diffusion_compile_backend(_BackendConfig(requested)) == "inductor"


def test_platform_rejects_foreign_backend(monkeypatch):
    monkeypatch.setattr(OmniPlatform, "_omni_enum", OmniPlatformEnum.CUDA, raising=False)

    with pytest.raises(ValueError, match="not supported"):
        OmniPlatform.get_diffusion_compile_backend(_BackendConfig("mindiesd"))


def test_inductor_platform_rejects_npu_aclgraph(monkeypatch):
    config = _CompileConfig(diffusion_compile_backend="inductor", diffusion_compile_aclgraph=True)
    monkeypatch.setattr(OmniPlatform, "_omni_enum", OmniPlatformEnum.CUDA, raising=False)

    with pytest.raises(ValueError, match="NPU MindIE-SD backend only"):
        OmniPlatform.get_diffusion_compile_backend(config)


def test_unavailable_inductor_is_not_passed_as_default_backend(monkeypatch):
    monkeypatch.setattr(OmniPlatform, "_omni_enum", OmniPlatformEnum.UNSPECIFIED, raising=False)
    monkeypatch.setattr(OmniPlatform, "supports_torch_inductor", lambda: False)

    assert OmniPlatform.get_diffusion_compile_backend(_BackendConfig("auto")) is None
    with pytest.raises(ValueError, match="not supported"):
        OmniPlatform.get_diffusion_compile_backend(_BackendConfig("inductor"))


def test_npu_auto_falls_back_when_mindiesd_is_unavailable(npu_platform, compile_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", None)
    compile_config.diffusion_compile_backend = "auto"

    assert npu_platform.get_diffusion_compile_backend(compile_config) is None


def test_npu_auto_selects_available_mindiesd(npu_platform, compile_config, mindiesd):
    compile_config.diffusion_compile_backend = "auto"

    backend = npu_platform.get_diffusion_compile_backend(compile_config)

    assert backend is mindiesd.MindieSDBackend.return_value
    mindiesd.MindieSDBackend.assert_called_once_with()


def test_npu_resolves_explicit_backend(npu_platform, compile_config, mindiesd):
    assert npu_platform.supports_torch_inductor() is False

    backend = npu_platform.get_diffusion_compile_backend(compile_config)

    assert backend is mindiesd.MindieSDBackend.return_value
    mindiesd.MindieSDBackend.assert_called_once_with()
    assert mindiesd.CompilationConfig.aclgraph_only is False
    assert mindiesd.CompilationConfig.aclgraph_with_compile is False
    assert mindiesd.CompilationConfig.aclgraph_lazy_capture is False
    assert mindiesd.CompilationConfig.safe_output_mode is True


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("diffusion_compile_backend", "inductor", "requires"),
        ("diffusion_compile_dynamic", True, "diffusion_compile_dynamic=False"),
        ("dtype", torch.float16, "BF16"),
        ("quantization_config", object(), "unquantized"),
        ("enable_cpu_offload", True, "offload"),
        ("enable_layerwise_offload", True, "offload"),
        ("enable_distributed_layerwise_offload", True, "offload"),
        ("cache_backend", "cache_dit", "cache_backend"),
        ("lora_path", "adapter", "LoRA"),
        ("enable_sleep_mode", True, "sleep"),
    ],
)
def test_npu_rejects_explicit_incompatible_config(
    npu_platform,
    compile_config,
    mindiesd,
    field_name,
    value,
    message,
):
    setattr(compile_config, field_name, value)

    with pytest.raises(ValueError, match=message):
        npu_platform.get_diffusion_compile_backend(compile_config)
    mindiesd.MindieSDBackend.assert_not_called()


def test_npu_auto_falls_back_for_incompatible_config(npu_platform, compile_config, mindiesd):
    compile_config.diffusion_compile_backend = "auto"
    compile_config.diffusion_compile_dynamic = True

    assert npu_platform.get_diffusion_compile_backend(compile_config) is None
    mindiesd.MindieSDBackend.assert_not_called()


def test_npu_rejects_multiple_devices(npu_platform, compile_config, mindiesd):
    compile_config.parallel_config.world_size = 2

    with pytest.raises(ValueError, match="single NPU"):
        npu_platform.get_diffusion_compile_backend(compile_config)
    mindiesd.MindieSDBackend.assert_not_called()


@pytest.mark.parametrize("field_name", ["use_hsdp", "sequence_parallel_size"])
def test_npu_rejects_full_compile_parallelism(npu_platform, compile_config, mindiesd, field_name):
    compile_config.diffusion_compile_granularity = "full"
    setattr(compile_config.parallel_config, field_name, True if field_name == "use_hsdp" else 2)

    with pytest.raises(ValueError, match="HSDP or sequence parallelism"):
        npu_platform.get_diffusion_compile_backend(compile_config)
    mindiesd.MindieSDBackend.assert_not_called()


def test_npu_accepts_full_static_compile(npu_platform, compile_config, mindiesd):
    compile_config.diffusion_compile_granularity = "full"

    assert npu_platform.get_diffusion_compile_backend(compile_config) is mindiesd.MindieSDBackend.return_value


def test_npu_missing_dependency_has_actionable_error(npu_platform, compile_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", None)

    with pytest.raises(RuntimeError, match="could not be imported") as error:
        npu_platform.get_diffusion_compile_backend(compile_config)
    assert isinstance(error.value.__cause__, ImportError)


def test_npu_aclgraph_does_not_silently_fall_back(npu_platform, compile_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", None)
    compile_config.diffusion_compile_backend = "auto"
    compile_config.diffusion_compile_aclgraph = True

    with pytest.raises(RuntimeError, match="could not be imported"):
        npu_platform.get_diffusion_compile_backend(compile_config)


def test_npu_resets_aclgraph_state_by_default(npu_platform, compile_config, mindiesd):
    mindiesd.CompilationConfig.aclgraph_only = True
    mindiesd.CompilationConfig.aclgraph_with_compile = True
    mindiesd.CompilationConfig.aclgraph_lazy_capture = True
    mindiesd.CompilationConfig.safe_output_mode = False

    npu_platform.get_diffusion_compile_backend(compile_config)

    assert mindiesd.CompilationConfig.aclgraph_only is False
    assert mindiesd.CompilationConfig.aclgraph_with_compile is False
    assert mindiesd.CompilationConfig.aclgraph_lazy_capture is False
    assert mindiesd.CompilationConfig.safe_output_mode is True


def test_npu_enables_aclgraph_only_when_requested(npu_platform, compile_config, mindiesd):
    compile_config.diffusion_compile_aclgraph = True

    npu_platform.get_diffusion_compile_backend(compile_config)

    assert mindiesd.CompilationConfig.aclgraph_only is False
    assert mindiesd.CompilationConfig.aclgraph_with_compile is True
    assert mindiesd.CompilationConfig.aclgraph_lazy_capture is True
    assert mindiesd.CompilationConfig.safe_output_mode is True


@pytest.mark.parametrize(
    ("model_class_name", "wan_enabled", "qwen_enabled"),
    [
        ("QwenImagePipeline", False, True),
        ("QwenImageEditPipeline", False, True),
        ("WanPipeline", True, False),
        ("FluxPipeline", False, False),
    ],
)
def test_npu_selects_model_family_residual_pattern(
    npu_platform,
    compile_config,
    mindiesd,
    model_class_name,
    wan_enabled,
    qwen_enabled,
):
    compile_config.model_class_name = model_class_name

    npu_platform.get_diffusion_compile_backend(compile_config)

    patterns = mindiesd.CompilationConfig.fusion_patterns
    assert patterns.enable_wan_residual_gate is wan_enabled
    assert patterns.enable_qwen_residual_gate is qwen_enabled


def test_npu_preserves_disabled_residual_pattern(npu_platform, compile_config, mindiesd):
    patterns = mindiesd.CompilationConfig.fusion_patterns
    patterns.enable_qwen_residual_gate = False

    npu_platform.get_diffusion_compile_backend(compile_config)

    assert patterns.enable_qwen_residual_gate is False


def test_npu_supports_mindiesd_without_qwen_pattern_flag(npu_platform, compile_config, mindiesd):
    del mindiesd.CompilationConfig.fusion_patterns.enable_qwen_residual_gate

    assert npu_platform.get_diffusion_compile_backend(compile_config) is mindiesd.MindieSDBackend.return_value


def test_npu_rejects_incompatible_compilation_api(npu_platform, compile_config, mindiesd):
    del mindiesd.CompilationConfig.aclgraph_only

    with pytest.raises(RuntimeError, match="compilation API"):
        npu_platform.get_diffusion_compile_backend(compile_config)


def test_npu_rejects_missing_fusion_patterns_api(npu_platform, compile_config, mindiesd):
    del mindiesd.CompilationConfig.fusion_patterns

    with pytest.raises(RuntimeError, match="fusion_patterns"):
        npu_platform.get_diffusion_compile_backend(compile_config)


def test_npu_backend_constructor_error_propagates(npu_platform, compile_config, mindiesd):
    mindiesd.MindieSDBackend.side_effect = TypeError("incompatible constructor")

    with pytest.raises(TypeError, match="incompatible constructor"):
        npu_platform.get_diffusion_compile_backend(compile_config)

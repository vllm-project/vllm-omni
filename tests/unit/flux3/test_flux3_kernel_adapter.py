# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for Flux3 kernel selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.flux3 import flux3_kernel_adapter_factory
from vllm_omni.diffusion.models.flux3.flux3_kernel_adapter import Flux3KernelAdapter
from vllm_omni.diffusion.models.flux3.flux3_kernel_adapter_factory import (
    Flux3PrimsTSConfig,
    Flux3PrimsTSMode,
    Flux3PrimsTSSupport,
    build_flux3_kernel_adapter,
    probe_prims_ts_support,
    resolve_prims_ts_support,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("mode", list(Flux3PrimsTSMode))
def test_prims_ts_mode_parsing(mode: Flux3PrimsTSMode) -> None:
    config = Flux3PrimsTSConfig.from_additional_config({"prims_ts": {"mode": mode.value}})

    assert config.mode is mode


def test_prims_ts_defaults_to_auto() -> None:
    assert Flux3PrimsTSConfig.from_additional_config(None).mode is Flux3PrimsTSMode.AUTO
    assert Flux3PrimsTSConfig.from_additional_config({}).mode is Flux3PrimsTSMode.AUTO
    assert Flux3PrimsTSConfig.from_additional_config({"prims_ts": {}}).mode is Flux3PrimsTSMode.AUTO


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        ({"mode": True}, TypeError),
        ({"mode": "sometimes"}, ValueError),
        ({"mode": "off", "unknown": True}, ValueError),
    ],
)
def test_prims_ts_rejects_invalid_config(value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        Flux3PrimsTSConfig.from_additional_config({"prims_ts": value})


def test_probe_rejects_non_cuda_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        flux3_kernel_adapter_factory,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: False),
    )

    support = probe_prims_ts_support()

    assert support.available is False
    assert support.reason == "PrimsTS requires a CUDA device"


def test_probe_rejects_unsupported_cuda_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    capability = SimpleNamespace(to_int=lambda: 120)
    monkeypatch.setattr(
        flux3_kernel_adapter_factory,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda: capability,
        ),
    )

    support = probe_prims_ts_support()

    assert support.available is False
    assert support.reason == "PrimsTS requires one of SM100, SM103, SM107; found SM120"


def test_auto_mode_falls_back_when_flashinfer_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    capability = SimpleNamespace(to_int=lambda: 100)
    monkeypatch.setattr(
        flux3_kernel_adapter_factory,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda: capability,
        ),
    )

    def missing_flashinfer(_name: str) -> None:
        raise ModuleNotFoundError("FlashInfer is not installed")

    monkeypatch.setattr(flux3_kernel_adapter_factory.importlib, "import_module", missing_flashinfer)
    od_config = SimpleNamespace(additional_config={"prims_ts": {"mode": "auto"}})
    quant_config = SimpleNamespace(
        get_name=lambda: "modelopt",
        quant_method="FP8_PER_CHANNEL_PER_TOKEN",
    )

    adapter = build_flux3_kernel_adapter(
        qk_norm_rope=lambda q, k, _v, _norm, _rope: (q, k),
        od_config=od_config,
        quant_config=quant_config,
    )

    assert type(adapter) is Flux3KernelAdapter


def test_required_mode_raises_when_backend_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    unavailable = Flux3PrimsTSSupport(False, "test backend is unavailable")
    monkeypatch.setattr(flux3_kernel_adapter_factory, "probe_prims_ts_support", lambda: unavailable)
    config = Flux3PrimsTSConfig(mode=Flux3PrimsTSMode.REQUIRED)

    with pytest.raises(RuntimeError, match="test backend is unavailable"):
        resolve_prims_ts_support(config)


def test_off_mode_does_not_probe_optional_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_probe() -> Flux3PrimsTSSupport:
        raise AssertionError("disabled mode must not probe the optional backend")

    monkeypatch.setattr(flux3_kernel_adapter_factory, "probe_prims_ts_support", unexpected_probe)

    support = resolve_prims_ts_support(Flux3PrimsTSConfig(mode=Flux3PrimsTSMode.OFF))

    assert support.available is False
    assert support.reason == "disabled by configuration"


def test_adapter_factory_owns_backend_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    available = Flux3PrimsTSSupport(True)
    monkeypatch.setattr(flux3_kernel_adapter_factory, "probe_prims_ts_support", lambda _symbols: available)
    od_config = SimpleNamespace(additional_config={"prims_ts": {"mode": "auto"}})
    quant_config = SimpleNamespace(
        get_name=lambda: "modelopt",
        quant_method="FP8_PER_CHANNEL_PER_TOKEN",
        LinearMethodCls=object,
    )

    adapter = build_flux3_kernel_adapter(
        qk_norm_rope=lambda q, k, _v, _norm, _rope: (q, k),
        od_config=od_config,
        quant_config=quant_config,
    )

    assert adapter.prims_ts_config.mode is Flux3PrimsTSMode.AUTO
    assert adapter.prims_ts_support is available
    assert adapter.quant_config is quant_config
    assert adapter.quant_format == "fp8"
    assert adapter.backend == "prims_ts"


def test_adapter_factory_selects_modelopt_nvfp4(monkeypatch: pytest.MonkeyPatch) -> None:
    available = Flux3PrimsTSSupport(True)
    monkeypatch.setattr(flux3_kernel_adapter_factory, "probe_prims_ts_support", lambda _symbols: available)
    od_config = SimpleNamespace(additional_config={"prims_ts": {"mode": "required"}})
    quant_config = SimpleNamespace(get_name=lambda: "modelopt_fp4", quant_method="NVFP4")

    adapter = build_flux3_kernel_adapter(
        qk_norm_rope=lambda q, k, _v, _norm, _rope: (q, k),
        od_config=od_config,
        quant_config=quant_config,
    )

    assert adapter.quant_format == "nvfp4"
    assert adapter.backend == "prims_ts"


def test_adapter_factory_rejects_modelopt_w4a16_nvfp4() -> None:
    od_config = SimpleNamespace(additional_config={"prims_ts": {"mode": "required"}})
    quant_config = SimpleNamespace(get_name=lambda: "modelopt_fp4", quant_method="W4A16_NVFP4")

    with pytest.raises(RuntimeError, match="requires a ModelOpt FP8_PER_CHANNEL_PER_TOKEN or NVFP4"):
        build_flux3_kernel_adapter(
            qk_norm_rope=lambda q, k, _v, _norm, _rope: (q, k),
            od_config=od_config,
            quant_config=quant_config,
        )


def test_auto_mode_keeps_baseline_for_incompatible_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_probe(_symbols: tuple[str, ...]) -> Flux3PrimsTSSupport:
        raise AssertionError("an incompatible checkpoint must not probe the optional backend")

    monkeypatch.setattr(flux3_kernel_adapter_factory, "probe_prims_ts_support", unexpected_probe)
    od_config = SimpleNamespace(additional_config={"prims_ts": {"mode": "auto"}})

    adapter = build_flux3_kernel_adapter(
        qk_norm_rope=lambda q, k, _v, _norm, _rope: (q, k),
        od_config=od_config,
        quant_config=None,
    )

    assert adapter.backend == "vllm"


def test_required_mode_rejects_incompatible_checkpoint() -> None:
    od_config = SimpleNamespace(additional_config={"prims_ts": {"mode": "required"}})

    with pytest.raises(RuntimeError, match="ModelOpt FP8_PER_CHANNEL_PER_TOKEN or NVFP4"):
        build_flux3_kernel_adapter(
            qk_norm_rope=lambda q, k, _v, _norm, _rope: (q, k),
            od_config=od_config,
            quant_config=None,
        )


@pytest.mark.parametrize(
    ("od_fields", "reason"),
    [
        ({"dtype": torch.float16}, "requires bfloat16 model execution"),
        ({"parallel_config": SimpleNamespace(use_hsdp=True)}, "does not currently support HSDP"),
    ],
)
def test_required_mode_rejects_unsupported_execution_config(
    od_fields: dict[str, object],
    reason: str,
) -> None:
    od_config = SimpleNamespace(
        additional_config={"prims_ts": {"mode": "required"}},
        **od_fields,
    )
    quant_config = SimpleNamespace(
        get_name=lambda: "modelopt",
        quant_method="FP8_PER_CHANNEL_PER_TOKEN",
    )

    with pytest.raises(RuntimeError, match=reason):
        build_flux3_kernel_adapter(
            qk_norm_rope=lambda q, k, _v, _norm, _rope: (q, k),
            od_config=od_config,
            quant_config=quant_config,
        )

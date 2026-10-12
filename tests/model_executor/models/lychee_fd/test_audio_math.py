# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU dispatch contracts for the optional released GELU toolchain."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.lychee_fd import audio_math as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class SimulatedCudaInput(torch.Tensor):
    """CPU-backed tensor exposing CUDA dispatch metadata without CUDA work."""

    @property
    def device(self):
        return torch.device("cuda", 0)


@pytest.fixture(autouse=True)
def clear_path_cache():
    module.released_libdevice_path.cache_clear()
    yield
    module.released_libdevice_path.cache_clear()


def matching_runtime(monkeypatch, capability=(8, 0)):
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu132")
    monkeypatch.setattr(torch.version, "cuda", "13.2")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 92000)
    monkeypatch.setattr(
        module,
        "current_omni_platform",
        SimpleNamespace(is_cuda=lambda: True, get_device_capability=lambda device: capability),
    )


def simulated_inputs():
    storage = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=torch.bfloat16)
    return storage, storage.as_subclass(SimulatedCudaInput)


def test_cpu_gelu_uses_torch_without_toolchain_lookup(monkeypatch):
    monkeypatch.setattr(module, "released_libdevice_path", lambda: pytest.fail("CPU must not inspect libdevice"))
    inputs = torch.randn(16, dtype=torch.bfloat16)
    torch.testing.assert_close(module.released_gelu(inputs), F.gelu(inputs))


def test_missing_default_toolchain_uses_torch_gelu(monkeypatch, tmp_path):
    matching_runtime(monkeypatch)
    monkeypatch.delenv("LYCHEE_RELEASED_LIBDEVICE_PATH", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-toolchain"))
    storage, inputs = simulated_inputs()
    actual = module.released_gelu(inputs).as_subclass(torch.Tensor)
    torch.testing.assert_close(actual, F.gelu(storage))
    assert module.released_libdevice_path() is None


@pytest.mark.parametrize("change", ["hardware", "torch", "cuda", "cudnn", "platform"])
def test_unqualified_runtime_uses_torch_without_released_toolchain(monkeypatch, change):
    matching_runtime(monkeypatch)
    if change == "hardware":
        monkeypatch.setattr(module.current_omni_platform, "get_device_capability", lambda device: (9, 0))
    elif change == "torch":
        monkeypatch.setattr(torch, "__version__", "2.14.0+cu132")
    elif change == "cuda":
        monkeypatch.setattr(torch.version, "cuda", "12.8")
    elif change == "cudnn":
        monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 91900)
    else:
        monkeypatch.setattr(module.current_omni_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(module, "released_libdevice_path", lambda: pytest.fail("must not inspect released toolchain"))
    storage, inputs = simulated_inputs()
    torch.testing.assert_close(module.released_gelu(inputs).as_subclass(torch.Tensor), F.gelu(storage))


def test_explicit_missing_libdevice_is_a_configuration_error(monkeypatch, tmp_path):
    matching_runtime(monkeypatch)
    monkeypatch.setenv("LYCHEE_RELEASED_LIBDEVICE_PATH", str(tmp_path / "explicit-missing.bc"))
    _, inputs = simulated_inputs()
    with pytest.raises(RuntimeError, match="explicit released libdevice path"):
        module.released_gelu(inputs)


def test_available_matching_toolchain_keeps_released_kernel_dispatch(monkeypatch, tmp_path):
    matching_runtime(monkeypatch)
    path = tmp_path / "libdevice.10.bc"
    path.write_text("CPU dispatch fixture; never compiled")
    monkeypatch.setenv("LYCHEE_RELEASED_LIBDEVICE_PATH", str(path))
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(inputs, output, count, **kwargs):
                calls.append((grid, count, kwargs))
                output.copy_(F.gelu(inputs))

            return launch

    monkeypatch.setattr(module, "_released_gelu_kernel", Kernel())
    storage, inputs = simulated_inputs()
    torch.testing.assert_close(module.released_gelu(inputs).as_subclass(torch.Tensor), F.gelu(storage))
    assert len(calls) == 1
    assert calls[0][1] == storage.numel()
    assert calls[0][2]["extern_libs"] == {"libdevice": str(path)}
    assert calls[0][2]["enable_fp_fusion"] is False

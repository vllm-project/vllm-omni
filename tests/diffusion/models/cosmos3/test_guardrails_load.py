# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Test the torch.load wrapper without importing model or accelerator runtimes."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def load_guardrails(monkeypatch):
    def load(platform, cuda_available=False):
        # Use an explicit signature so duplicate map_location arguments raise
        # exactly as they do in torch.load, instead of being accepted by a Mock.
        def torch_load(f, map_location=None, pickle_module=None, *, weights_only=True, mmap=None):
            return f, map_location, pickle_module, weights_only, mmap

        dependencies = {
            "numpy": {"ndarray": object},
            "torch": {
                "load": torch_load,
                "cuda": SimpleNamespace(is_available=lambda: cuda_available),
            },
            "vllm.logger": {"init_logger": lambda name: None},
            "vllm_omni.diffusion.models.progress_bar": {"_is_rank_zero": lambda: True},
            "vllm_omni.errors": {"GuardrailViolationError": RuntimeError},
            "vllm_omni.platforms": {
                "current_omni_platform": SimpleNamespace(
                    is_npu=lambda: platform == "npu", is_xpu=lambda: platform == "xpu"
                )
            },
            "cosmos_guardrail": {"CosmosSafetyChecker": object},
        }
        for name, attrs in dependencies.items():
            module = ModuleType(name)
            module.__dict__.update(attrs)
            monkeypatch.setitem(sys.modules, name, module)

        path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/cosmos3/guardrails.py"
        spec = importlib.util.spec_from_file_location("_test_cosmos3_guardrails", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return sys.modules["torch"].load

    return load


@pytest.mark.parametrize("platform", ["npu", "xpu"])
@pytest.mark.parametrize("location", ["cpu", None, {"cuda:0": "cpu"}, lambda storage, location: storage])
def test_preserves_explicit_map_location(load_guardrails, platform, location):
    load = load_guardrails(platform)
    # Positional and keyword forms must agree, including explicit None.
    assert load("checkpoint.pt", location)[1] is location
    assert load("checkpoint.pt", map_location=location)[1] is location


@pytest.mark.parametrize(
    ("platform", "cuda_available", "expected"),
    [("npu", False, "cpu"), ("xpu", False, "cpu"), ("cpu", False, None), ("npu", True, None)],
)
def test_default_map_location(load_guardrails, platform, cuda_available, expected):
    load = load_guardrails(platform, cuda_available)
    assert load("checkpoint.pt")[1] == expected


def test_forwards_remaining_load_arguments(load_guardrails):
    load = load_guardrails("npu")
    pickle_module = object()
    assert load("checkpoint.pt", "cpu", pickle_module, weights_only=False, mmap=True) == (
        "checkpoint.pt",
        "cpu",
        pickle_module,
        False,
        True,
    )


def test_duplicate_map_location_still_raises(load_guardrails):
    load = load_guardrails("npu")
    with pytest.raises(TypeError, match="multiple values.*map_location"):
        load("checkpoint.pt", "cpu", map_location="cpu")

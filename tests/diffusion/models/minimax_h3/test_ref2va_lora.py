# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU arithmetic and binding contracts for the immutable Ref2VA adapter."""

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class StartupFlags:
    lora_path: str | None = None
    lora_backend: str = "distill"
    lora_scale: float = 1.0


@pytest.fixture
def adapter():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/ref2va_lora.py"
    spec = importlib.util.spec_from_file_location("_ref2va_lora_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_has_mixed_ranks_and_all_259_targets(adapter):
    targets = adapter.expected_targets()
    assert len(targets) == 259
    assert sum(rank == 16 for _, _, rank in targets.values()) == 51
    assert sum(rank == 64 for _, _, rank in targets.values()) == 208
    assert targets["blocks.0.attn.qkv_proj"] == (5376, 21504, 64)
    assert targets["token_refiner.blocks.1.attn.out_proj"] == (7168, 5376, 64)


def test_forward_delta_keeps_base_output_dtype_and_bias_separate(adapter):
    x = torch.tensor([[1.2, -0.7, 0.3]], dtype=torch.float32)
    a = torch.tensor([[1, 2, -1]], dtype=torch.bfloat16)
    b = torch.tensor([[2], [-3]], dtype=torch.bfloat16)
    baseline = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
    result = adapter._ForwardDelta(a, b)(x, baseline)
    expected = baseline + F.linear(F.linear(x.bfloat16(), a), b).float()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == baseline.dtype
    assert torch.equal(baseline, torch.tensor([[0.5, 1.5]]))


def test_bad_artifact_hash_is_rejected_before_tensor_loading(adapter, tmp_path):
    path = tmp_path / adapter.FILENAME
    path.write_bytes(b"not the supported checkpoint")
    with pytest.raises(ValueError, match="SHA256"):
        adapter.load_checked(path)


def test_startup_adapter_is_opt_in_and_requires_fixed_scale(adapter, tmp_path):
    assert adapter.resolve_adapter_path(StartupFlags()) is None
    assert adapter.resolve_adapter_path(StartupFlags(lora_path="other.safetensors")) is None
    path = tmp_path / adapter.FILENAME
    path.touch()
    config = StartupFlags(lora_path=str(path))
    assert adapter.resolve_adapter_path(config) == path
    config.lora_scale = 0.5
    with pytest.raises(ValueError, match="lora-scale"):
        adapter.resolve_adapter_path(config)


@pytest.mark.parametrize(
    "task,steps,dynamic,valid",
    [
        ("ref2va", 8, False, True),
        ("t2va", 8, False, False),
        ("ref2va", 7, False, False),
        ("ref2va", 8, True, False),
    ],
)
def test_startup_adapter_request_contract(task, steps, dynamic, valid):
    source = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Pipeline")
    method = next(
        node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_sigma_positions"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"OmniClientError": ValueError}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(source), "exec"),
        namespace,
    )

    class Pipeline:
        _ref2va_lora_path = Path("startup-adapter")
        _fasth3 = None

        def _sigma_schedule_for_request(self, sampling, task):
            return None

    @dataclass
    class Sampling:
        num_inference_steps: int
        lora_request: object | None

    request = Sampling(steps, object() if dynamic else None)
    if valid:
        assert namespace["_resolve_sigma_positions"](Pipeline(), task, request) == (None, 8)
    else:
        with pytest.raises(ValueError, match="adapter"):
            namespace["_resolve_sigma_positions"](Pipeline(), task, request)


@torch.inference_mode()
def test_binding_adds_delta_after_base_projection_and_clears_adaln(adapter, monkeypatch, mocker):
    class Projection(torch.nn.Module):
        input_size = 3
        output_size = 2

        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.ones(2, 3, dtype=torch.bfloat16))

        def forward(self, x):
            return F.linear(x, self.weight), None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = Projection()
            self.adaln_cache = mocker.Mock(spec=["clear"])

        def forward(self, *, x):
            return self.project(x)

    distributed = ModuleType("vllm.distributed")
    distributed.__dict__["get_tensor_model_parallel_world_size"] = lambda: 1
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setattr(adapter, "expected_targets", lambda: {"project": (3, 2, 1)})
    a = torch.tensor([[1, 2, -1]], dtype=torch.bfloat16)
    b = torch.tensor([[2], [-3]], dtype=torch.bfloat16)
    monkeypatch.setattr(adapter, "load_checked", lambda path: {"project": (a, b)})
    model = Model()
    x = torch.tensor([[1, 2, 3]], dtype=torch.bfloat16)
    baseline, _ = model(x=x)
    weights = model.project.weight.clone()
    adapter.install_adapter(model, Path("unused"))
    actual, bias = model(x=x)
    torch.testing.assert_close(actual, baseline + F.linear(F.linear(x, a), b), rtol=0, atol=0)
    assert bias is None
    torch.testing.assert_close(model.project.weight, weights, rtol=0, atol=0)
    assert set(dict(model.project.named_buffers())) == {"weight", "ref2va_delta.a", "ref2va_delta.b"}
    model.adaln_cache.clear.assert_called_once_with()

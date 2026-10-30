# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers import fused_swiglu
from vllm_omni.diffusion.models.qwen_image_21 import mlp_fusion
from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def config(backend="native", **overrides):
    fields = dict(enforce_eager=True, dtype=torch.bfloat16, additional_config={"qwen21_mlp_backend": backend})
    fields.update(overrides)
    return OmniDiffusionConfig(**fields)


def test_native_does_not_probe_optional_dependency(monkeypatch):
    def unexpected_probe():
        raise AssertionError("native must not probe MindIE-SD")

    monkeypatch.setattr(mlp_fusion, "fused_swiglu_available", unexpected_probe)
    assert mlp_fusion.select_mlp_backend(config(), unquantized=True) == "native"


@pytest.mark.parametrize("backend", [False, True, None, "merged", "auto", []])
def test_invalid_backend(backend):
    with pytest.raises(ValueError, match="qwen21_mlp_backend"):
        mlp_fusion.select_mlp_backend(config(backend), unquantized=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("enforce_eager", False),
        ("dtype", torch.float32),
        ("lora_path", "adapter"),
        ("enable_cpu_offload", True),
        ("enable_layerwise_offload", True),
        ("enable_distributed_layerwise_offload", True),
        ("diffusion_offload_config", {"mode": "module", "components": ["dit"]}),
    ],
)
def test_unsupported_configuration_falls_back(monkeypatch, field, value):
    cfg = config("eagle", **{field: value})
    monkeypatch.setattr(mlp_fusion, "fused_swiglu_available", lambda: True)
    assert mlp_fusion.select_mlp_backend(cfg, unquantized=True) == "native"


@pytest.mark.parametrize("field", ["tensor_parallel_size", "sequence_parallel_size", "use_hsdp"])
def test_parallel_configuration_falls_back(monkeypatch, field):
    cfg = config("eagle")
    setattr(cfg.parallel_config, field, True if field == "use_hsdp" else 2)
    monkeypatch.setattr(mlp_fusion, "fused_swiglu_available", lambda: True)
    assert mlp_fusion.select_mlp_backend(cfg, unquantized=True) == "native"


@pytest.mark.parametrize(
    "unquantized,available,expected", [(True, True, "eagle"), (False, True, "native"), (True, False, "native")]
)
def test_backend_selection(monkeypatch, unquantized, available, expected):
    monkeypatch.setattr(mlp_fusion, "fused_swiglu_available", lambda: available)
    assert mlp_fusion.select_mlp_backend(config("eagle"), unquantized=unquantized) == expected


@pytest.fixture
def local_linears(monkeypatch):
    class LocalLinear(torch.nn.Linear):
        def __init__(self, input_size, output_size, bias=False, **kwargs):
            super().__init__(input_size, output_size, bias=bias, dtype=torch.bfloat16)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return super().forward(x)

    for name in ("ColumnParallelLinear", "RowParallelLinear", "ReplicatedLinear"):
        monkeypatch.setattr(model, name, LocalLinear)
    monkeypatch.setattr(fused_swiglu.CustomOp, "dispatch_forward", lambda op: op.forward_native)


@pytest.fixture
def mlp(local_linears):
    torch.manual_seed(42)
    return model.QwenImage21SwiGLUFeedForward(8, 24, mlp_backend="eagle").eval()


def reference(layer, x):
    return F.linear(F.silu(F.linear(x, layer.gate_layer.weight)) * F.linear(x, layer.proj.weight), layer.out.weight)


def test_packing_preserves_checkpoint_keys_and_storage(mlp):
    before = {name: value.clone() for name, value in mlp.state_dict().items()}
    with torch.inference_mode():
        assert mlp.prepare_fused_weights()
    assert not mlp._gate_up_weight.is_inference()
    assert set(mlp.state_dict()) == set(before)
    for name, value in mlp.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    storage = mlp._gate_up_weight.untyped_storage().data_ptr()
    assert mlp.gate_layer.weight.untyped_storage().data_ptr() == storage
    assert mlp.proj.weight.untyped_storage().data_ptr() == storage
    assert mlp.proj.weight.storage_offset() == mlp.gate_layer.weight.numel()


def test_forward_uses_complete_ffn_without_repacking(mlp, monkeypatch):
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    with torch.inference_mode():
        assert mlp.prepare_fused_weights()
        expected = reference(mlp, x)

        def unexpected_cat(*args, **kwargs):
            raise AssertionError("forward must not pack weights")

        monkeypatch.setattr(torch, "cat", unexpected_cat)
        actual = mlp(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(proj.calls == 0 for proj in (mlp.gate_layer, mlp.proj, mlp.out))


def test_checkpoint_reload_updates_packed_storage(mlp):
    with torch.inference_mode():
        assert mlp.prepare_fused_weights()
    state = {name: value + 0.125 for name, value in mlp.state_dict().items()}
    mlp.load_state_dict(state)
    with torch.inference_mode():
        x = torch.randn(1, 5, 8, dtype=torch.bfloat16)
        torch.testing.assert_close(mlp(x), reference(mlp, x), rtol=0, atol=0)


@pytest.mark.parametrize("change", ["unpacked", "native", "replace_weight", "dtype", "training", "compile", "grad"])
def test_fallback_uses_current_modules(mlp, monkeypatch, change):
    assert mlp.prepare_fused_weights()
    dtype = torch.bfloat16
    if change == "unpacked":
        mlp._gate_up_weight = None
    elif change == "native":
        mlp.mlp_backend = "native"
    elif change == "replace_weight":
        mlp.gate_layer.weight = torch.nn.Parameter(mlp.gate_layer.weight.clone() + 0.25)
    elif change == "dtype":
        mlp.float()
        dtype = torch.float32
    elif change == "training":
        mlp.train()
    elif change == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    x = torch.randn(1, 5, 8, dtype=dtype)
    with torch.set_grad_enabled(change == "grad"):
        torch.testing.assert_close(mlp(x), reference(mlp, x), rtol=0, atol=0)
    assert all(proj.calls == 1 for proj in (mlp.gate_layer, mlp.proj, mlp.out))


def test_replaced_projection_keeps_its_forward(mlp):
    class Adapter(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.weight = base.weight

        def forward(self, x):
            return self.base(x) + 0.5

    assert mlp.prepare_fused_weights()
    mlp.gate_layer = Adapter(mlp.gate_layer)
    with torch.inference_mode():
        x = torch.randn(1, 5, 8, dtype=torch.bfloat16)
        expected = mlp.out(F.silu(mlp.gate_layer(x)) * mlp.proj(x))
        torch.testing.assert_close(mlp(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["native", "eagle"])
def test_model_configuration_and_weight_loading(local_linears, monkeypatch, backend):
    monkeypatch.setattr(mlp_fusion, "fused_swiglu_available", lambda: True)
    monkeypatch.setattr(model, "QwenImage21Attention", lambda **kwargs: torch.nn.Identity())
    transformer = model.QwenImage21Transformer2DModel(
        config(backend), num_layers=1, num_attention_heads=1, attention_head_dim=128, context_in_dim=128
    ).eval()
    mlp = transformer.transformer_blocks[0].img_mlp
    assert transformer.mlp_backend == mlp.mlp_backend == backend
    checkpoint = [(name, value.clone()) for name, value in transformer.named_parameters()]
    with torch.inference_mode():
        loaded = transformer.load_weights(checkpoint)
    assert set(name for name, _ in checkpoint) <= loaded
    assert (mlp._gate_up_weight is not None) == (backend == "eagle")


def test_projection_hook_uses_original_modules(mlp):
    assert mlp.prepare_fused_weights()
    calls = []
    with mlp.out.register_forward_hook(lambda *args: calls.append(1)), torch.inference_mode():
        x = torch.randn(1, 5, 8, dtype=torch.bfloat16)
        torch.testing.assert_close(mlp(x), reference(mlp, x), rtol=0, atol=0)
    assert calls == [1]

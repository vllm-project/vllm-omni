# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU unit tests for the HunyuanImage 3.0 shared-layout adapter.

The adapter is pure layout logic: shared-boundary filtering, checkpoint-name
filtering, weight-spec construction and view attachment.  These tests exercise
that contract with synthetic ``torch`` tensors on CPU, using the same
private-namespace import as the backend suite so the enclosing ``vllm_omni``
package is never initialized.
"""

import importlib
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tests.distributed.weight_sharing.helpers import load_class_methods, load_weight_sharing

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DIM = 8


class _Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # RMSNorm-style: weight only, no bias.
        self.weight = nn.Parameter(torch.randn(DIM))


class _Proj(nn.Module):
    def __init__(self, out: int, inp: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out, inp))
        self.bias = nn.Parameter(torch.randn(out))


class _Gate(nn.Module):
    def __init__(self, out: int, inp: int) -> None:
        super().__init__()
        # The shared router is float32 on both stages.
        self.weight = nn.Parameter(torch.randn(out, inp, dtype=torch.float32))


class _Experts(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.routed_experts = nn.Module()
        self.routed_experts.register_parameter("w13_weight", nn.Parameter(torch.randn(2 * dim, dim)))
        self.routed_experts.register_parameter("w2_weight", nn.Parameter(torch.randn(dim, dim)))


class _Attn(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv_proj = _Proj(3 * dim, dim)
        self.o_proj = _Proj(dim, dim)


class _MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = _Gate(dim, dim)
        self.experts = _Experts(dim)


class _Layer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.input_layernorm = _Norm()
        self.post_attention_layernorm = _Norm()
        self.self_attn = _Attn(dim)
        self.mlp = _MLP(dim)


class _Backbone(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, DIM)
        self.layers = nn.ModuleList(_Layer(DIM) for _ in range(num_layers))
        self.norm = _Norm()


def _expected_shared_names(num_layers: int = 2) -> set[str]:
    names = {"embed_tokens.weight", "norm.weight"}
    for i in range(num_layers):
        names.update(
            {
                f"layers.{i}.input_layernorm.weight",
                f"layers.{i}.post_attention_layernorm.weight",
                f"layers.{i}.self_attn.qkv_proj.weight",
                f"layers.{i}.self_attn.qkv_proj.bias",
                f"layers.{i}.self_attn.o_proj.weight",
                f"layers.{i}.self_attn.o_proj.bias",
                f"layers.{i}.mlp.gate.weight",
                f"layers.{i}.mlp.experts.routed_experts.w13_weight",
                f"layers.{i}.mlp.experts.routed_experts.w2_weight",
            }
        )
    return names


@pytest.fixture(scope="module")
def adapter():
    ws = load_weight_sharing()
    return importlib.import_module(f"{ws.__name__}.adapters.hunyuan_image3")


@pytest.fixture(scope="module")
def session_module():
    ws = load_weight_sharing()
    return importlib.import_module(f"{ws.__name__}.session")


@pytest.fixture(scope="module")
def spec_builder():
    ws = load_weight_sharing()
    return importlib.import_module(f"{ws.__name__}.adapters").build_weight_spec


@pytest.fixture(scope="module")
def backbone() -> _Backbone:
    return _Backbone()


def test_is_shared_param_filters_runtime_paths(adapter):
    for name in _expected_shared_names():
        assert adapter.is_shared_param(name), name
    for name in (
        "lm_head.weight",
        "vae.decoder.weight",
        "layers.0.self_attn.attn.k_scale",
        "layers.0.mlp.experts.routed_experts.w13_weight_scale",
        "layers.0.self_attn.image_attn.something",
    ):
        assert not adapter.is_shared_param(name), name


def test_shared_backbone_names_matches_module(adapter, backbone):
    assert set(adapter.shared_backbone_names(backbone)) == _expected_shared_names()


def test_collect_shared_bindings(adapter, backbone):
    bindings = adapter.collect_shared_bindings(backbone)
    assert list(bindings) == sorted(_expected_shared_names())
    assert all(isinstance(value, torch.Tensor) for value in bindings.values())
    # The router must be float32 per the fusion contract.
    assert bindings["layers.0.mlp.gate.weight"].dtype == torch.float32


def test_collect_shared_bindings_detaches_parameter_subclasses(adapter):
    from vllm.model_executor.parameter import ModelWeightParameter

    backbone = nn.Module()
    backbone.embed_tokens = nn.Module()
    backbone.embed_tokens.register_parameter(
        "weight", ModelWeightParameter.__new__(ModelWeightParameter, torch.randn(16, DIM))
    )
    backbone.norm = nn.Module()
    backbone.norm.register_parameter("weight", ModelWeightParameter.__new__(ModelWeightParameter, torch.randn(DIM)))
    assert any(type(param) is ModelWeightParameter for _, param in backbone.named_parameters())

    bindings = adapter.collect_shared_bindings(backbone)
    assert set(bindings) == {"embed_tokens.weight", "norm.weight"}
    assert all(type(value) is torch.Tensor for value in bindings.values()), [type(value) for value in bindings.values()]


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_build_weight_spec(spec_builder, device):
    # Arbitrary model names and non-contiguous views exercise the common builder.
    bindings = {"z.weight": torch.empty(2, 3, device=device).t(), "a.weight": torch.empty(4, device=device)}
    spec = spec_builder("model-id", "revision", bindings, layout="test-layout")
    assert spec.model_id == "model-id"
    assert spec.checkpoint_revision == "revision"
    assert spec.layout == "test-layout"
    assert [tensor.name for tensor in spec.tensors] == sorted(bindings)
    assert spec == spec_builder("model-id", "revision", dict(reversed(bindings.items())), layout="test-layout")
    for tensor in spec.tensors:
        param = bindings[tensor.name]
        assert tensor.shape == tuple(param.shape)
        assert tensor.stride == tuple(param.stride())
        assert tensor.dtype == str(param.dtype).removeprefix("torch.")


def test_build_weight_spec_rejects_empty_bindings(spec_builder, contracts):
    with pytest.raises(contracts.CompatibilityError, match="no shared weight bindings"):
        spec_builder("model-id", "revision", {}, layout="test-layout")


def test_build_weight_spec_requires_layout(spec_builder, contracts):
    bindings = {"weight": torch.empty(1)}
    with pytest.raises(contracts.CompatibilityError, match="nonempty layout"):
        spec_builder("model-id", "revision", bindings, layout="")
    with pytest.raises(TypeError, match="layout"):
        spec_builder("model-id", "revision", bindings)


def test_publish_collects_bindings_once(adapter, backbone, session_module, monkeypatch):
    group = session_module.SharedWeightGroupConfig(
        group_id="group",
        generation=0,
        session_id="session",
        provider_id="ar",
        consumer_id="dit",
        role="provider",
        rank=0,
        tp_size=1,
        device_uuid="GPU-test",
        model_id="model",
        checkpoint_revision="revision",
        layout="hunyuan_image3_backbone_v1",
    )
    enumerate_params = Mock(wraps=backbone.named_parameters)
    monkeypatch.setattr(backbone, "named_parameters", enumerate_params)
    backend = Mock()
    adapter.publish_backbone(backend, SimpleNamespace(model=backbone), group=group, device_index=0)
    enumerate_params.assert_called_once_with()
    backend.prepare_export.assert_called_once()
    bindings, spec, _ = backend.prepare_export.call_args.args
    assert list(bindings) == [tensor.name for tensor in spec.tensors] == sorted(_expected_shared_names())
    assert spec.layout == group.layout
    for name, tensor in bindings.items():
        assert tensor.data_ptr() == backbone.get_parameter(name).data_ptr()


def test_is_shared_checkpoint_name(adapter):
    shared = (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.query_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.gate.wg.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.gate_and_up_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.bias",
        "model.layers.0.mlp.shared_mlp.gate_proj.weight",
        "model.layers.0.mlp.shared_mlp.gate_up_proj.weight",
        "model.layers.0.mlp.shared_mlp.gate_and_up_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.3.down_proj.weight",
        "model.layers.0.mlp.gate_proj_bias",
    )
    for name in shared:
        assert adapter.is_shared_checkpoint_name(name), name

    private = (
        "lm_head.weight",
        "vae.decoder.conv_out.weight",
        "vision_model.embeddings.weight",
        "model.vision_aligner.weight",
        "model.layers.0.self_attn.attn.k_scale",
        "model.final_layer.weight",
        "model.patch_embed.weight",
    )
    for name in private:
        assert not adapter.is_shared_checkpoint_name(name), name


@pytest.mark.parametrize("suffix", ["weight", "bias"])
def test_private_load_filters_fused_shared_mlp(adapter, suffix):
    loader_cls = load_class_methods(
        "vllm_omni/diffusion/model_loader/diffusers_loader.py",
        "DiffusersPipelineLoader",
        ["load_weights"],
        {"time": time, "logger": Mock()},
    )
    shared = f"model.layers.0.mlp.shared_mlp.gate_up_proj.{suffix}"
    checkpoint_key = f"model.layers.0.mlp.shared_mlp.gate_and_up_proj.{suffix}"
    private = "model.final_layer.weight"
    private_tensor = torch.tensor([7.0])
    model = Mock()
    model.named_parameters.return_value = []

    def load_weights(weights):
        received = dict(weights)
        assert set(received) == {private}
        assert received[private] is private_tensor
        return {private}

    model.load_weights.side_effect = load_weights
    loader = loader_cls()
    loader.counter_before_loading_weights = time.perf_counter()
    loader._get_expected_parameter_names = Mock(return_value={shared, private})
    loader.get_all_weights = Mock(return_value=iter([(checkpoint_key, torch.ones(1)), (private, private_tensor)]))
    loader.load_weights(
        model, planned_weights=[shared], disallow_weights=[shared], weight_name_filter=adapter.is_shared_checkpoint_name
    )


def test_attach_imported_weights_shares_storage(adapter, backbone):
    bindings = adapter.collect_shared_bindings(backbone)
    data_ptrs = {name: tensor.data_ptr() for name, tensor in bindings.items()}

    # Pass the same tensors back as the "imported" views and verify the module
    # parameters are replaced with Parameters that share the same storage.
    adapter.attach_imported_weights(backbone, dict(bindings))

    for name, ptr in data_ptrs.items():
        param = backbone.get_parameter(name)
        assert isinstance(param, nn.Parameter), name
        assert param.requires_grad is False, name
        assert param.data_ptr() == ptr, name


def test_attach_imported_weights_rejects_mismatch(adapter, backbone):
    bindings = adapter.collect_shared_bindings(backbone)
    missing = dict(bindings)
    missing_name = next(iter(missing))
    missing.pop(missing_name)
    with pytest.raises(adapter.CompatibilityError) as exc:
        adapter.attach_imported_weights(backbone, missing)
    assert (
        str(exc.value) == f"imported views do not match the shared backbone layout (missing={[missing_name]}, extra=[])"
    )

    extra = dict(bindings)
    extra["layers.0.self_attn.not_shared.weight"] = torch.randn(DIM, DIM)
    with pytest.raises(adapter.CompatibilityError) as exc:
        adapter.attach_imported_weights(backbone, extra)
    assert str(exc.value) == (
        "imported views do not match the shared backbone layout "
        "(missing=[], extra=['layers.0.self_attn.not_shared.weight'])"
    )


def test_validate_supported_layout(adapter):
    class _Config:
        pass

    ok = _Config()
    ok.use_cla = False
    ok.num_experts = 32
    adapter.validate_supported_layout(ok)

    cla = _Config()
    cla.use_cla = True
    cla.num_experts = 32
    with pytest.raises(adapter.CompatibilityError):
        adapter.validate_supported_layout(cla)

    per_layer = _Config()
    per_layer.use_cla = False
    per_layer.num_experts = [8, 8]
    with pytest.raises(adapter.CompatibilityError):
        adapter.validate_supported_layout(per_layer)


def test_session_builders(session_module):
    group = session_module.SharedWeightGroupConfig(
        group_id="group",
        generation=0,
        session_id="session",
        provider_id="provider",
        consumer_id="consumer",
        role="provider",
        rank=1,
        tp_size=4,
        device_uuid="GPU-00000000-0000-0000-0000-000000000000",
        model_id="model",
        checkpoint_revision="revision",
        layout="hunyuan_image3_backbone_v1",
    )
    assert group.payload is None

    identity = session_module.build_identity(group)
    assert identity.group_id == "group"
    assert identity.rank == 1
    assert identity.tp_size == 4
    assert identity.protocol_version == 1

    context = session_module.build_session_context(group, 3)
    assert context.device_index == 3
    assert context.identity == identity

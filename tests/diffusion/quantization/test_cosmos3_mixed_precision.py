# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.mixed_precision import (
    Cosmos3MixedPrecisionConfig,
    Cosmos3MixedPrecisionRuntime,
    resolve_mixed_precision_config,
)
from vllm_omni.diffusion.models.cosmos3.mixed_precision import runtime as runtime_impl
from vllm_omni.diffusion.models.cosmos3.mixed_precision.checkpoint import read_checkpoint_policy
from vllm_omni.diffusion.models.cosmos3.mixed_precision.runtime import (
    Cosmos3MixedPrecisionLinearMethod,
)
from vllm_omni.diffusion.models.cosmos3.mixed_precision.strategy import (
    Fp8W8A8W8A16Strategy,
    Nvfp4W4A4W4A16Strategy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _policy(
    *,
    first_steps: int = 3,
    last_steps: int = 3,
    reasoner: str = "a16",
    scope: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "type": "first_last_n",
        "index_space": "denoising_loop_iteration",
        "scope": scope or ["transformer"],
        "default_mode": "native",
        "first_steps": {"count": first_steps, "mode": "a16"},
        "last_steps": {"count": last_steps, "mode": "a16"},
        "overlap": "a16",
        "reasoner": reasoner,
    }


def _fake_quant_config(name: str, *, serialized: bool = True):
    if name == "modelopt":
        return SimpleNamespace(
            get_name=lambda: name,
            is_checkpoint_fp8_serialized=serialized,
        )
    if name == "modelopt_fp4":
        return SimpleNamespace(
            get_name=lambda: name,
            quant_method="NVFP4",
            is_checkpoint_nvfp4_serialized=serialized,
        )
    return SimpleNamespace(get_name=lambda: name)


def _checkpoint_od_config(
    quant_config,
    *,
    policy: object | None = None,
    additional_config: dict[str, object] | None = None,
):
    quant_config_name = quant_config.get_name()
    quant_algo = {
        "modelopt": "FP8",
        "modelopt_fp4": "NVFP4",
    }.get(quant_config_name, "FP8")
    disk_quant_config: dict[str, object] = {
        "quant_method": quant_config_name,
        "quant_algo": quant_algo,
    }
    if policy is not None:
        disk_quant_config["runtime"] = {"diffusion_step_policy": policy}
    return SimpleNamespace(
        tf_model_config=SimpleNamespace(
            params={"quantization_config": disk_quant_config},
            quant_config=quant_config,
        ),
        additional_config=additional_config,
    )


def _swizzle_blockscale_cpu(scale: torch.Tensor) -> torch.Tensor:
    rows, cols = scale.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    padded = torch.zeros(
        (1, padded_rows, padded_cols),
        dtype=scale.dtype,
        device=scale.device,
    )
    padded[0, :rows, :cols] = scale
    padded = padded.reshape(1, padded_rows // 128, 4, 32, padded_cols // 4, 4)
    return padded.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(
        padded_rows,
        padded_cols,
    )


def test_config_parses_nested_schedule_and_reasoner_policy() -> None:
    config = Cosmos3MixedPrecisionConfig.from_additional_config(
        {
            "cosmos3_mixed_precision": {
                "first_steps": 2,
                "last_steps": 4,
                "reasoner": "native",
            }
        }
    )

    assert config is not None
    assert config.reasoner == "native"
    assert [index for index in range(10) if config.use_high_precision(index, 10)] == [0, 1, 6, 7, 8, 9]


@pytest.mark.parametrize(
    ("first_steps", "last_steps", "selected"),
    [
        (0, 2, [5, 6]),
        (2, 0, [0, 1]),
        (0, 0, []),
        (4, 4, list(range(7))),
    ],
)
def test_schedule_boundaries_and_overlap(
    first_steps: int,
    last_steps: int,
    selected: list[int],
) -> None:
    config = Cosmos3MixedPrecisionConfig(
        first_steps=first_steps,
        last_steps=last_steps,
    )
    assert [index for index in range(7) if config.use_high_precision(index, 7)] == selected


def test_one_step_request_honors_boundary_precision() -> None:
    config = Cosmos3MixedPrecisionConfig(first_steps=1, last_steps=1)
    assert config.use_high_precision(0, 1)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"cosmos3_mixed_precision": True}, "must be a mapping"),
        ({"cosmos3_mixed_precision": {"first_steps": -1}}, "non-negative"),
        ({"cosmos3_mixed_precision": {"reasoner": "fp16"}}, "must be one of"),
        ({"cosmos3_mixed_precision": {"enabled": 0}}, "must be a boolean"),
        (
            {"cosmos3_mixed_precision": {"enabled": False, "first_steps": 1}},
            "cannot be combined",
        ),
        ({"cosmos3_mixed_precision": {"unknown": 1}}, "Unknown"),
    ],
)
def test_config_rejects_invalid_values(values: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Cosmos3MixedPrecisionConfig.from_additional_config(values)


def test_config_presence_enables_defaults() -> None:
    assert Cosmos3MixedPrecisionConfig.from_additional_config({}) is None
    config = Cosmos3MixedPrecisionConfig.from_additional_config(
        {"cosmos3_mixed_precision": {}}
    )
    assert config == Cosmos3MixedPrecisionConfig()


def test_noop_config_does_not_install_runtime() -> None:
    assert (
        Cosmos3MixedPrecisionConfig.from_additional_config(
            {
                "cosmos3_mixed_precision": {
                    "first_steps": 0,
                    "last_steps": 0,
                    "reasoner": "native",
                }
            }
        )
        is None
    )


def test_additional_config_distinguishes_absent_and_disabled() -> None:
    assert Cosmos3MixedPrecisionConfig.resolve_additional_config({}) == (False, None)
    assert Cosmos3MixedPrecisionConfig.resolve_additional_config({"cosmos3_mixed_precision": {"enabled": False}}) == (
        True,
        None,
    )
    assert Cosmos3MixedPrecisionConfig.resolve_additional_config(
        {
            "cosmos3_mixed_precision": {
                "first_steps": 0,
                "last_steps": 0,
                "reasoner": "native",
            }
        }
    ) == (True, None)


def test_checkpoint_policy_round_trip_through_transformer_config() -> None:
    from vllm_omni.diffusion.data import TransformerConfig

    disk = {
        "_class_name": "Cosmos3VFMTransformer",
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "runtime": {"diffusion_step_policy": _policy(first_steps=2, last_steps=4)},
        },
    }

    tf_config = TransformerConfig.from_dict(disk)
    assert tf_config.quant_config is not None
    assert tf_config.quant_config.get_name() == "modelopt"
    assert (
        tf_config.to_dict()["quantization_config"]["runtime"]
        == disk["quantization_config"]["runtime"]
    )
    assert read_checkpoint_policy(SimpleNamespace(tf_model_config=tf_config)) == Cosmos3MixedPrecisionConfig(
        first_steps=2,
        last_steps=4,
        reasoner="a16",
    )


@pytest.mark.parametrize("name", ["modelopt", "modelopt_fp4"])
def test_checkpoint_policy_supports_serialized_modelopt_formats(name: str) -> None:
    od_config = _checkpoint_od_config(_fake_quant_config(name), policy=_policy())
    assert read_checkpoint_policy(od_config) == Cosmos3MixedPrecisionConfig()


@pytest.mark.parametrize("name", ["modelopt", "modelopt_fp4"])
def test_checkpoint_policy_rejects_nonserialized_modelopt_formats(name: str) -> None:
    od_config = _checkpoint_od_config(
        _fake_quant_config(name, serialized=False),
        policy=_policy(),
    )
    with pytest.raises(ValueError, match="serialized ModelOpt"):
        read_checkpoint_policy(od_config)


def test_checkpoint_policy_rejects_mixed_modelopt_format() -> None:
    with pytest.raises(ValueError, match="serialized ModelOpt FP8 or NVFP4"):
        read_checkpoint_policy(_checkpoint_od_config(_fake_quant_config("modelopt_mixed"), policy=_policy()))


def test_checkpoint_policy_missing_metadata_preserves_ordinary_path() -> None:
    assert read_checkpoint_policy(_checkpoint_od_config(_fake_quant_config("modelopt"))) is None


def test_checkpoint_policy_rejects_malformed_runtime_container() -> None:
    od_config = _checkpoint_od_config(_fake_quant_config("modelopt"))
    od_config.tf_model_config.params["quantization_config"]["runtime"] = True
    with pytest.raises(TypeError, match="runtime must be a mapping"):
        read_checkpoint_policy(od_config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("type", "sigma_ranges", "type"),
        ("index_space", "scheduler_timestep", "index_space"),
        ("default_mode", "a16", "default_mode"),
        ("overlap", "native", "overlap"),
        ("reasoner", "fp16", "reasoner"),
    ],
)
def test_checkpoint_policy_rejects_invalid_schema(field: str, value: object, message: str) -> None:
    policy = _policy()
    policy[field] = value
    with pytest.raises(ValueError, match=message):
        read_checkpoint_policy(_checkpoint_od_config(_fake_quant_config("modelopt"), policy=policy))


@pytest.mark.parametrize(
    ("step_range", "message"),
    [
        ({"count": -1, "mode": "a16"}, "non-negative"),
        ({"count": True, "mode": "a16"}, "non-negative"),
        ({"count": 1, "mode": "native"}, "mode"),
        ({"mode": "a16"}, "Missing"),
        ({"count": 1, "mode": "a16", "sigma": 1.0}, "Unknown"),
    ],
)
def test_checkpoint_policy_rejects_invalid_step_range(step_range: object, message: str) -> None:
    policy = _policy()
    policy["first_steps"] = step_range
    with pytest.raises((TypeError, ValueError), match=message):
        read_checkpoint_policy(_checkpoint_od_config(_fake_quant_config("modelopt"), policy=policy))


def test_checkpoint_policy_rejects_unsupported_quantization_config() -> None:
    with pytest.raises(ValueError, match="serialized ModelOpt FP8 or NVFP4"):
        read_checkpoint_policy(_checkpoint_od_config(_fake_quant_config("compressed-tensors"), policy=_policy()))


def test_checkpoint_policy_for_other_scope_is_not_applied() -> None:
    od_config = _checkpoint_od_config(
        _fake_quant_config("modelopt"),
        policy=_policy(scope=["vae"]),
    )
    assert read_checkpoint_policy(od_config) is None


def test_additional_config_overrides_and_disables_checkpoint_policy() -> None:
    checkpoint = _checkpoint_od_config(
        _fake_quant_config("modelopt"),
        policy=_policy(first_steps=1, last_steps=1),
    )
    assert resolve_mixed_precision_config(checkpoint) == (
        Cosmos3MixedPrecisionConfig(first_steps=1, last_steps=1),
        "checkpoint",
    )

    checkpoint.additional_config = {
        "cosmos3_mixed_precision": {
            "first_steps": 2,
            "last_steps": 0,
            "reasoner": "native",
        }
    }
    assert resolve_mixed_precision_config(checkpoint) == (
        Cosmos3MixedPrecisionConfig(first_steps=2, last_steps=0, reasoner="native"),
        "additional_config",
    )

    checkpoint.additional_config = {"cosmos3_mixed_precision": {"enabled": False}}
    assert resolve_mixed_precision_config(checkpoint) == (None, "additional_config_disabled")


def test_runtime_allows_standard_offload_and_rejects_distributed(monkeypatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    monkeypatch.setattr(
        transformer_cosmos3,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    od_config = SimpleNamespace(
        enable_cpu_offload=True,
        enable_layerwise_offload=True,
        enable_distributed_layerwise_offload=False,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    transformer_cosmos3._validate_mixed_precision_runtime(
        Cosmos3MixedPrecisionConfig(),
        od_config,
    )
    od_config.enable_distributed_layerwise_offload = True
    with pytest.raises(ValueError, match="distributed layer-wise offload"):
        transformer_cosmos3._validate_mixed_precision_runtime(
            Cosmos3MixedPrecisionConfig(),
            od_config,
        )
    od_config.enable_distributed_layerwise_offload = False
    od_config.max_num_seqs = 2
    with pytest.raises(ValueError, match="one active request"):
        transformer_cosmos3._validate_mixed_precision_runtime(
            Cosmos3MixedPrecisionConfig(),
            od_config,
        )


class _Layer(torch.nn.Module):
    def __init__(self, output_size: int, input_size: int) -> None:
        super().__init__()
        self.output_size_per_partition = output_size
        self.input_size_per_partition = input_size


class _BaseMethod:
    def __init__(self, *, nvfp4: bool = False) -> None:
        self.processed = False
        self.apply_calls = 0
        self.fp8_linear = object()
        if nvfp4:
            from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
                CutlassNvFp4LinearKernel,
            )

            self.kernel = object.__new__(CutlassNvFp4LinearKernel)
        else:
            self.kernel = None

    def create_weights(self, *args, **kwargs) -> None:
        pass

    def process_weights_after_loading(self, layer) -> None:
        self.processed = True
        if layer.weight.dtype == torch.float8_e4m3fn:
            layer.weight = layer.weight.t().contiguous()
            layer.weight_scale = layer.weight_scale.max().reshape(1)
        elif layer.weight.dtype == torch.uint8:
            layer.weight_scale = _swizzle_blockscale_cpu(layer.weight_scale)
            layer.weight_global_scale = layer.weight_scale_2.max().float().reshape(1)
            del layer.weight_scale_2

    def apply(self, layer, x, bias=None):
        del bias
        self.apply_calls += 1
        return torch.full(
            (*x.shape[:-1], layer.output_size_per_partition),
            17,
            dtype=x.dtype,
            device=x.device,
        )


def _fp8_layer(output_size: int = 2, input_size: int = 4) -> _Layer:
    layer = _Layer(output_size, input_size)
    layer.weight = torch.ones(output_size, input_size, dtype=torch.float8_e4m3fn)
    layer.weight_scale = torch.tensor([0.5], dtype=torch.float32)
    return layer


def _nvfp4_layer(output_size: int = 128, input_size: int = 64) -> _Layer:
    layer = _Layer(output_size, input_size)
    layer.weight = torch.full((output_size, input_size // 2), 0x22, dtype=torch.uint8)
    layer.weight_scale = torch.ones(output_size, input_size // 16, dtype=torch.float8_e4m3fn)
    layer.weight_scale_2 = torch.ones(1)
    return layer


def _runtime_and_method(
    strategy,
    layer: _Layer,
    *,
    path: str = "generation",
    reasoner: str = "native",
):
    config = Cosmos3MixedPrecisionConfig(
        first_steps=1,
        last_steps=1,
        reasoner=reasoner,
    )
    runtime = Cosmos3MixedPrecisionRuntime(config)
    base = _BaseMethod(nvfp4=isinstance(strategy, Nvfp4W4A4W4A16Strategy))
    method = Cosmos3MixedPrecisionLinearMethod(
        base,
        strategy,
        runtime,
        f"{path}.linear",
        path,
    )
    method.process_weights_after_loading(layer)
    return runtime, base, method


@pytest.mark.parametrize(
    ("strategy", "layer"),
    [
        (Fp8W8A8W8A16Strategy(), _fp8_layer()),
        (Nvfp4W4A4W4A16Strategy(), _nvfp4_layer()),
    ],
)
def test_generation_dispatches_native_middle_and_a16_edges(strategy, layer) -> None:
    runtime, base, method = _runtime_and_method(strategy, layer)
    x = torch.ones(1, layer.input_size_per_partition, dtype=torch.bfloat16)

    runtime.set_step(1, 3)
    assert torch.equal(
        method.apply(layer, x),
        torch.full((1, layer.output_size_per_partition), 17, dtype=x.dtype),
    )
    runtime.set_step(0, 3)
    assert torch.equal(
        method.apply(layer, x),
        torch.nn.functional.linear(x, strategy.materialize(layer)),
    )
    assert base.apply_calls == 1


@pytest.mark.parametrize(
    ("strategy", "layer"),
    [
        (Fp8W8A8W8A16Strategy(), _fp8_layer()),
        (Nvfp4W4A4W4A16Strategy(), _nvfp4_layer()),
    ],
)
def test_a16_uses_only_live_native_weights(strategy, layer) -> None:
    runtime, base, method = _runtime_and_method(strategy, layer)
    assert base.processed
    assert not hasattr(layer, "_cosmos3_precision_weight")
    assert not hasattr(layer, "_cosmos3_precision_weight_scale")

    runtime.set_step(0, 3)
    output = method.apply(
        layer,
        torch.ones(1, layer.input_size_per_partition, dtype=torch.bfloat16),
    )
    assert output.shape == (1, layer.output_size_per_partition)


def test_a16_follows_live_weight_storage_after_offload_rebinding() -> None:
    layer = _fp8_layer()
    strategy = Fp8W8A8W8A16Strategy()
    runtime, _, method = _runtime_and_method(strategy, layer)
    runtime.set_step(0, 3)
    x = torch.ones(1, 4, dtype=torch.bfloat16)
    before = method.apply(layer, x)

    layer.weight = torch.zeros_like(layer.weight)
    after = method.apply(layer, x)
    assert not torch.equal(before, after)
    assert torch.count_nonzero(after) == 0


def test_fp8_rejects_smoothquant() -> None:
    layer = _fp8_layer()
    layer.pre_quant_scale = torch.ones(1)
    with pytest.raises(ValueError, match="SmoothQuant"):
        Fp8W8A8W8A16Strategy().validate_before_processing(
            _BaseMethod(),
            layer,
            "gen.linear",
        )


def test_fp8_rejects_non_tensorwise_scales() -> None:
    layer = _fp8_layer(output_size=3, input_size=2)
    layer.weight_scale = torch.tensor([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="tensorwise FP8"):
        Fp8W8A8W8A16Strategy().validate_before_processing(
            _BaseMethod(),
            layer,
            "gen.linear",
        )


def test_nvfp4_reference_materializes_live_cutlass_weights() -> None:
    layer = _nvfp4_layer()
    strategy = Nvfp4W4A4W4A16Strategy()
    _runtime_and_method(strategy, layer)
    output = strategy.materialize(layer)
    assert output.shape == (128, 64)
    assert torch.equal(output, torch.ones_like(output))


def test_nvfp4_rejects_fused_global_scales() -> None:
    layer = _nvfp4_layer()
    layer.weight_scale_2 = torch.tensor([1.0, 2.0])
    strategy = Nvfp4W4A4W4A16Strategy()
    with pytest.raises(ValueError, match="NVFP4 scales"):
        strategy.validate_before_processing(
            _BaseMethod(nvfp4=True),
            layer,
            "gen.qkv",
        )


@pytest.mark.parametrize(
    ("policy", "uses_native"),
    [("a16", False), ("native", True)],
)
def test_reasoner_policy_is_independent_of_generation_step(
    policy: str,
    uses_native: bool,
) -> None:
    layer = _fp8_layer()
    runtime, base, method = _runtime_and_method(
        Fp8W8A8W8A16Strategy(),
        layer,
        path="reasoner",
        reasoner=policy,
    )
    runtime.set_step(1, 3)
    method.apply(layer, torch.ones(1, 4, dtype=torch.bfloat16))
    assert (base.apply_calls == 1) is uses_native


def test_reset_clears_generation_state() -> None:
    runtime = Cosmos3MixedPrecisionRuntime(
        Cosmos3MixedPrecisionConfig(first_steps=1, last_steps=1)
    )
    runtime.set_step(0, 5)
    assert runtime.use_high_precision("generation")
    runtime.reset()
    assert not runtime.use_high_precision("generation")


def test_install_discovers_generation_and_opt_in_reasoner(monkeypatch) -> None:
    class _FakeLinear(torch.nn.Module):
        def __init__(self, prefix: str) -> None:
            super().__init__()
            self.prefix = prefix
            self.input_size_per_partition = 4
            self.output_size_per_partition = 2
            self.quant_method = SimpleNamespace(name="fp8")

    monkeypatch.setattr(runtime_impl, "LinearBase", _FakeLinear)
    strategy = Fp8W8A8W8A16Strategy()
    monkeypatch.setattr(strategy, "accepts", lambda method: getattr(method, "name", None) == "fp8")
    monkeypatch.setattr(runtime_impl, "_STRATEGIES", (strategy,))
    runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig(reasoner="a16"))
    reasoner = _FakeLinear("reasoner.q_proj")
    generation = [
        _FakeLinear("generation.q_proj"),
        _FakeLinear("generation.out_proj"),
    ]
    transformer = SimpleNamespace(
        language_model=SimpleNamespace(layers=torch.nn.Sequential(reasoner)),
        gen_layers=torch.nn.Sequential(*generation),
    )
    runtime.install(transformer)
    assert isinstance(reasoner.quant_method, Cosmos3MixedPrecisionLinearMethod)
    assert all(isinstance(layer.quant_method, Cosmos3MixedPrecisionLinearMethod) for layer in generation)


def test_pipeline_helpers_forward_and_reset_schedule() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    calls = []
    pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
    pipeline.transformer = SimpleNamespace(
        set_mixed_precision_step=lambda step, count: calls.append((step, count)),
        reset_mixed_precision=lambda: calls.append("reset"),
    )
    pipeline._set_mixed_precision_step(2, 7)
    pipeline._reset_mixed_precision()
    assert calls == [(2, 7), "reset"]

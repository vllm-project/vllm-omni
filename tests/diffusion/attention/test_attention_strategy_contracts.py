# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Component ownership and execution contracts, without model-name dispatch."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention import layer as layer_mod
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.contracts import MethodCapabilities, StrategyModelSupport
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.strategy import (
    AttentionOperation,
    finalize_attention_strategy,
    validate_pipeline_attention_strategy,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def local_attention(monkeypatch):
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: NoParallelAttention())
    monkeypatch.setattr(
        layer_mod, "get_attn_backend_for_role", lambda **kw: (SDPABackend, kw["attention_config"].default)
    )


class NewlySupportedTransformer(torch.nn.Module):
    attention_strategy_support = StrategyModelSupport()

    def __init__(self, component):
        super().__init__()
        self.attn = layer_mod.build_attention(
            num_heads=1,
            head_size=4,
            causal=False,
            softmax_scale=0.5,
            prefix="arbitrary.prefix.without.layer.numbers",
            role="example.self",
            operation=AttentionOperation("same_identity", 17, "example.self", None, component=component),
        )
        finalize_attention_strategy(self)

    def forward(self, x):
        return self._attention_strategy_runner(x)

    def forward_with_attention_layout(self, x, *, attention_layout=None):
        return self.attn.for_layout(attention_layout)(x, x, x)


def configuration(**options):
    return SimpleNamespace(
        diffusion_attention_config=AttentionConfig(**options),
        parallel_config=SimpleNamespace(ring_degree=1),
        diffusion_kv_cache_dtype=None,
    )


def test_component_scoped_inventory_is_complete_and_not_shared_mutable_state(local_attention):
    config = configuration(
        presets={"base": "FLASH_ATTN", "other": "CUDNN_ATTN"},
        layout={"default": "base", "overrides": [{"component": "second", "layers": [17], "use": "other"}]},
    )
    with set_current_diffusion_config(config):
        first = NewlySupportedTransformer("first")
        original_plan = first._attention_strategy_runner.plan
        second = NewlySupportedTransformer("second")
    assert first._attention_strategy_runner.plan is original_plan
    assert original_plan.operations[0].component == "first"
    assert second._attention_strategy_runner.plan.operations[0].component == "second"
    pipeline = SimpleNamespace(attention_strategy_components=("first", "second"), first=first, second=second)
    validate_pipeline_attention_strategy(pipeline, config)
    assert first.attn.for_layout(0).attn_spec.backend == "FLASH_ATTN"
    assert second.attn.for_layout(0).attn_spec.backend == "CUDNN_ATTN"
    assert second.attn.for_layout(0).layer_idx == 17
    assert second.attn.for_layout(0).attention_execution is second.attention_execution
    with pytest.raises(FrozenInstanceError):
        second.attention_execution.host_prepared = True
    with pytest.raises(ValueError, match="Unknown layers|empty attention selector"):
        validate_pipeline_attention_strategy(
            SimpleNamespace(attention_strategy_components=("first",), first=first), config
        )


def test_model_support_is_declared_not_name_based(local_attention, monkeypatch):
    config = configuration(presets={"base": "FLASH_ATTN"}, layout={"default": "base"})
    monkeypatch.setattr(NewlySupportedTransformer, "attention_strategy_support", None)
    with set_current_diffusion_config(config), pytest.raises(ValueError, match="support contract"):
        NewlySupportedTransformer("first")


def test_actual_provider_must_opt_in(local_attention, monkeypatch):
    config = configuration(presets={"base": "FLASH_ATTN"}, layout={"default": "base"})
    monkeypatch.setattr(SDPABackend, "strategy_capabilities", None)
    with set_current_diffusion_config(config), pytest.raises(ValueError, match="execution contract"):
        NewlySupportedTransformer("first")


def test_backend_options_are_validated_by_selected_executor(local_attention, monkeypatch):
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

    seen = []
    initialize = SDPAImpl.__init__

    def initialize_with_options(self, *args, backend_kwargs=None, **kwargs):
        if backend_kwargs:
            seen.append(backend_kwargs)
            if backend_kwargs["topk"] != 8:
                raise ValueError("Provider requires topk=8")
        initialize(self, *args, backend_kwargs=backend_kwargs, **kwargs)

    monkeypatch.setattr(SDPAImpl, "__init__", initialize_with_options)
    for topk in (8, 16):
        # Configuration normalization must not import or admit a backend.
        config = configuration(
            presets={"base": {"backend": "FASTVIDEO_VSA", "fastvideo_vsa_topk": topk}},
            layout={"default": "base"},
        )
        with set_current_diffusion_config(config):
            if topk == 8:
                NewlySupportedTransformer("first")
            else:
                with pytest.raises(ValueError, match="Provider requires"):
                    NewlySupportedTransformer("first")
    assert seen == [{"topk": 8}, {"topk": 16}]


def test_legacy_executor_uses_same_selection_interface(local_attention):
    with set_current_diffusion_config(configuration(default="FLASH_ATTN")):
        executor = layer_mod.build_attention(num_heads=1, head_size=4, causal=False, softmax_scale=0.5)
    assert executor.for_layout() is executor
    assert not executor.attention_execution.local_tensor_forward
    with pytest.raises(ValueError):
        executor.for_layout(0)


def test_compiled_strategy_preserves_pre_local_post_boundary(local_attention, monkeypatch):
    class TensorStages(NoParallelAttention):
        def __init__(self, scale):
            self.scale = scale

        def pre_attention(self, query, key, value, metadata):
            return query, key, value * self.scale, metadata, 3.0

        def post_attention(self, output, context):
            return output + context

    config = configuration(
        presets={"base": "FLASH_ATTN", "other": "CUDNN_ATTN"},
        layouts={"first": {"default": "base"}, "second": {"default": "other"}},
        schedule={
            "coordinate": "step_index",
            "phases": [{"until": 1, "layout": "first"}, {"until": None, "layout": "second"}],
        },
    )
    with set_current_diffusion_config(config):
        model = NewlySupportedTransformer("first")
    for layout, scale in enumerate((2.0, 4.0)):
        # Nontrivial tensor stages make bypassing either boundary observable.
        # This tests the integration seam, not distributed communication.
        model.attn.for_layout(layout)._no_parallel_strategy = TensorStages(scale)

    def unexpected_context_read():
        raise AssertionError("Fixed local attention must not inspect mutable request context")

    monkeypatch.setattr(layer_mod, "get_forward_context", unexpected_context_read)
    monkeypatch.setattr(layer_mod, "is_forward_context_available", unexpected_context_read)
    graphs: list[torch.fx.GraphModule] = []
    executions = []

    def backend(graph, inputs):
        index = len(graphs)
        graphs.append(graph)

        def execute(*args):
            executions.append(index)
            return graph.forward(*args)

        return execute

    torch._dynamo.reset()
    try:
        model._attention_strategy_runner.compile(backend=backend, dynamic=True)
        for length in (5, 7):
            x = torch.randn(1, length, 1, 4)
            q = x.transpose(1, 2)
            dense = torch.nn.functional.scaled_dot_product_attention(q, q, q).transpose(1, 2)
            for step, scale in ((0, 2.0), (1, 4.0), (0, 2.0)):
                with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=2)):
                    torch.testing.assert_close(model(x), dense * scale + 3.0)
        assert len(graphs) == 2
        assert executions == [0, 1, 0, 0, 1, 0]
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("compiled", [False, True])
def test_model_copy_and_state_dict_preserve_strategy(local_attention, compiled):
    import io
    from copy import deepcopy

    config = configuration(presets={"base": "FLASH_ATTN"}, layout={"default": "base"})
    with set_current_diffusion_config(config):
        model = NewlySupportedTransformer("first")
        restored = NewlySupportedTransformer("first")
    model.register_buffer("stored_weights", torch.tensor([3.0, 7.0]))
    restored.register_buffer("stored_weights", torch.zeros(2))
    if compiled:
        model._attention_strategy_runner.compile(backend="eager")
    x = torch.randn(1, 3, 1, 4)
    expected = model(x)
    if compiled:
        with pytest.raises(TypeError, match="process-local.*state_dict"):
            deepcopy(model)
    else:
        copied = deepcopy(model)
        torch.testing.assert_close(copied(x), expected)
        assert copied._attention_strategy_runner._model is copied
        serialized = io.BytesIO()
        torch.save(model, serialized)
        serialized.seek(0)
        loaded = torch.load(serialized, weights_only=False)
        torch.testing.assert_close(loaded(x), expected)
        assert loaded._attention_strategy_runner._model is loaded
    torch.testing.assert_close(model(x), expected)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored.stored_weights, model.stored_weights)
    torch.testing.assert_close(restored(x), expected)
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "parallel",
    [
        {"ring_degree": 2},
        {"allgather_degree": 2},
        {"tensor_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"data_parallel_size": 2},
        {"use_hsdp": True},
        {"cfg_parallel_size": 2},
    ],
)
def test_strategy_rejects_unsupported_parallel_combinations(parallel):
    from vllm_omni.diffusion.attention.contracts import validate_strategy_parallel

    config = configuration(presets={"base": "FLASH_ATTN"}, layout={"default": "base"})
    for name, value in parallel.items():
        setattr(config.parallel_config, name, value)
    with pytest.raises(ValueError, match="single-device"):
        validate_strategy_parallel(config)


def test_backend_graph_break_is_allowed(local_attention, monkeypatch):
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

    monkeypatch.setattr(SDPAImpl, "forward", torch.compiler.disable(SDPAImpl.forward))
    config = configuration(presets={"base": "TORCH_SDPA"}, layout={"default": "base"})
    with set_current_diffusion_config(config):
        model = NewlySupportedTransformer("first")
    x = torch.randn(1, 5, 1, 4)
    expected = model(x)
    model._attention_strategy_runner.compile(backend="eager")
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("options", [{"local_execution": False}, {"uses_request_context": True}])
def test_eager_admission_still_requires_strategy_owned_dispatch(options):
    capabilities = MethodCapabilities(**{"local_execution": True, **options})
    with pytest.raises(ValueError, match="immutable local execution"):
        capabilities.validate_strategy()


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_strategy_admits_pure_ulysses(enforce_eager):
    from vllm_omni.diffusion.attention.contracts import AttentionExecutionEnvironment, validate_strategy_parallel

    config = configuration(presets={"base": "FLASH_ATTN"}, layout={"default": "base"})
    config.num_gpus = 2
    config.parallel_config = SimpleNamespace(ulysses_degree=2, world_size=2, sequence_parallel_size=2)
    config.enforce_eager = enforce_eager
    config.diffusion_compile_granularity = "regional"
    validate_strategy_parallel(config)
    with pytest.raises(ValueError, match="does not declare Ulysses"):
        AttentionExecutionEnvironment.for_model(config, StrategyModelSupport())
    environment = AttentionExecutionEnvironment.for_model(config, StrategyModelSupport(ulysses=True))
    assert environment.ulysses_degree == 2
    config.parallel_config.ulysses_mode = "advanced_uaa"
    with pytest.raises(ValueError, match="strict Ulysses"):
        validate_strategy_parallel(config)
    config.parallel_config.ulysses_mode = "strict"
    config.enforce_eager = False
    config.diffusion_compile_granularity = "full"
    with pytest.raises(ValueError, match="regional compilation or eager"):
        validate_strategy_parallel(config)

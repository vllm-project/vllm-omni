# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-forward capture, compilation and replay with CPU or real CUDA providers."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context


def configure_strategy_test(monkeypatch, device, role):
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            pytest.skip("Requires Hopper SM90")
        pytest.importorskip("flash_attn.cute")
    torch.manual_seed(0)
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
    from vllm_omni.diffusion.data import AttentionConfig

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: NoParallelAttention())
    # Substitute just the CUDA provider with CPU numerical attention. Preserve
    # the real model, layout selection, metadata, weights and compiler.
    if device == "cpu":
        monkeypatch.setattr(
            layer_mod, "get_attn_backend_for_role", lambda **kw: (SDPABackend, kw["attention_config"].default)
        )
    alternate: dict[str, Any] = (
        {"backend": "CUDNN_ATTN"}
        if device == "cpu"
        else {
            "name": "block_sparse",
            "config": {
                "block_size": [64, 64],
                "selection": {"name": "block_topk", "config": {"target_sparsity": 0.5}},
                "backend": {"require": "FLASH_ATTN", "implementation": "auto"},
            },
        }
    )
    attention_config = AttentionConfig(
        presets={"dense": {"backend": "FLASH_ATTN"}, "alternate": alternate},
        layouts={
            "dense": {"default": "dense"},
            "mixed": {"default": "dense", "overrides": [{"attention_role": role, "use": "alternate"}]},
        },
        schedule={
            "coordinate": "step_index",
            "phases": [{"until": 1, "layout": "dense"}, {"until": None, "layout": "mixed"}],
        },
    )
    steps: tuple[int, ...] = (0, 1)
    if device == "cuda":
        # Three complete forwards: dense, one sparse block, both sparse blocks.
        attention_config = AttentionConfig(
            presets={"dense": {"backend": "FLASH_ATTN"}, "alternate": alternate},
            layouts={
                "dense": {"default": "dense"},
                "mixed": {
                    "default": "dense",
                    "overrides": [{"attention_role": role, "layers": [0], "use": "alternate"}],
                },
                "sparse": {"default": "dense", "overrides": [{"attention_role": role, "use": "alternate"}]},
            },
            schedule={
                "coordinate": "step_index",
                "phases": [
                    {"until": 1, "layout": "dense"},
                    {"until": 2, "layout": "mixed"},
                    {"until": None, "layout": "sparse"},
                ],
            },
        )
        steps = (0, 1, 2)
    common = dict(
        diffusion_attention_config=attention_config,
        parallel_config=SimpleNamespace(ring_degree=1, ulysses_degree=1),
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        diffusion_kv_cache_dtype=None,
    )
    return common, steps


def move_strategy_inputs(value, device, dtype):
    if isinstance(value, torch.Tensor):
        dtype = dtype if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=dtype)
    if isinstance(value, dict):
        return {key: move_strategy_inputs(item, device, dtype) for key, item in value.items()}
    return value


def check_strategy_capture_and_reuse(model, inputs, device, compiler_backend, steps, tolerances):
    references = {}
    for step in steps:
        with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
            references[step] = model(**inputs)

    graphs = []

    def backend(graph, example_inputs):
        compiled = torch._inductor.compile(graph, example_inputs) if compiler_backend == "inductor" else graph.forward
        # Inductor may request that Dynamo restart tracing with specialized
        # dimensions. Count completed compilations, not abandoned attempts.
        graphs.append(graph)
        return compiled

    torch._dynamo.reset()
    model._attention_strategy_runner.compile(backend=backend, dynamic=True)
    runner = model._attention_strategy_runner
    runner.set_warmup(True)
    try:
        with override_forward_context(ForwardContext(denoise_step_idx=0, total_denoise_steps=2)):
            torch.testing.assert_close(model(**inputs), references[0], **tolerances)
    finally:
        runner.set_warmup(False)
    assert not runner.warmup_status()["layouts_pending"]
    for step in steps:
        with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
            torch.testing.assert_close(model(**inputs), references[step], **tolerances)
    compiled_count = len(graphs)
    assert compiled_count == len(steps)
    for step in steps:
        with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
            torch.testing.assert_close(model(**inputs), references[step], **tolerances)
    assert len(graphs) == compiled_count
    if device == "cuda":
        sparse_calls = [
            sum("vllm_omni.block_sparse_request" in str(node.target) for node in graph.graph.nodes) for graph in graphs
        ]
        assert sparse_calls == [0, 1, 2]
    return graphs, compiled_count

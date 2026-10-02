# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strategy layouts reuse regional graphs across model and layer CPU transfers."""

from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.cosmos3.test_cosmos3_transformer import _tiny_cosmos3_config
from tests.helpers.attention_strategy import configure_strategy_test, move_strategy_inputs
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.parametrize(
    ("device", "offload_mode"),
    [
        pytest.param("cpu", "model", marks=pytest.mark.cpu),
        pytest.param("cuda", "model", marks=pytest.mark.cuda),
        pytest.param("cuda", "layerwise", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
@torch.inference_mode()
def test_strategy_offload_preserves_outputs_and_reuses_regions(monkeypatch, device, compiled, offload_mode):
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3 as cosmos

    common, steps = configure_strategy_test(monkeypatch, device, "cosmos3.gen")
    monkeypatch.setattr(cosmos, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(cosmos, "_get_ulysses_state", lambda: (1, 0, None))
    config = _tiny_cosmos3_config(num_hidden_layers=2)
    if device == "cuda":
        config.update(
            hidden_size=256, head_dim=128, intermediate_size=512, rope_scaling={"mrope_section": [24, 20, 20]}
        )
    cfg = SimpleNamespace(**common, tf_model_config=config)
    with set_current_diffusion_config(cfg):
        model = cosmos.Cosmos3VFMTransformer(cfg).to(device=device, dtype=common["dtype"])
    model.post_load_weights()
    model.eval()
    for weight in model.parameters():
        weight.fill_(1) if weight.ndim == 1 else weight.normal_(std=0.02)
    side = 16 if device == "cuda" else 2
    inputs = move_strategy_inputs(
        dict(
            hidden_states=torch.randn(1, 2, 1, side, side),
            timestep=torch.ones(1),
            text_ids=torch.zeros(1, 3, dtype=torch.long),
            text_mask=torch.ones(1, 3, dtype=torch.long),
            video_shape=(1, side, side),
        ),
        device,
        common["dtype"],
    )
    references = {}
    for step in steps:
        with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
            references[step] = model(**inputs)
    model.reset_cache()
    weights = {name: id(weight) for name, weight in model.named_parameters()}
    runner = model._attention_strategy_runner
    graphs, executions, swaps = [], [], []
    offloader = None
    if offload_mode == "model":
        model.enable_model_cpu_offload(device=torch.device(device), pin_memory=False)
        assert all(weight.device.type == "cpu" for weight in model.gen_layers.parameters())
        assert all(weight.device.type == "cpu" for weight in model.language_model.layers.parameters())
        activate = model._activate_model_cpu_offload_component

        def record_swap(name):
            assert not torch.compiler.is_compiling()
            if model._active_model_cpu_offload_component != name:
                swaps.append(name)
            activate(name)

        monkeypatch.setattr(model, "_activate_model_cpu_offload_component", record_swap)
    else:
        from vllm_omni.diffusion.offloader.base import OffloadConfig
        from vllm_omni.diffusion.offloader.config import OffloadStrategy
        from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend

        class Pipeline(torch.nn.Module):
            _dit_modules = ["transformer.language_model", "transformer"]
            _encoder_modules = []
            _vae_modules = []
            _resident_modules = []

        pipeline = Pipeline()
        pipeline.transformer = model
        offloader = LayerWiseOffloadBackend(OffloadConfig(strategy=OffloadStrategy.LAYER_WISE), torch.device(device))
        offloader.enable(pipeline)
        assert len(offloader._dit_hooks) == 4
        for hook in offloader._dit_hooks:
            prefetch = hook.prefetch_layer

            def record_prefetch(non_blocking=True, prefetch=prefetch):
                assert not torch.compiler.is_compiling()
                swaps.append("layer")
                return prefetch(non_blocking=non_blocking)

            monkeypatch.setattr(hook, "prefetch_layer", record_prefetch)

    def backend(graph, example_inputs):
        graphs.append(graph)
        execute = torch._inductor.compile(graph, example_inputs) if device == "cuda" else graph.forward

        def run(*args):
            executions.append(graph)
            return execute(*args)

        return run

    torch._dynamo.reset()
    try:
        if compiled:
            with pytest.raises(ValueError, match="offloading requires regional"):
                runner.compile()
            runner.compile(granularity="regional", backend=backend, dynamic=True)
            assert runner.warmup_status()["compile_granularity"] == "regional"
        for request in range(2):
            model.reset_cache()
            # A fresh request loads UND after GEN was resident on the prior one.
            runner.set_warmup(request == 0)
            for step in steps:
                before = len(executions)
                with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
                    torch.testing.assert_close(model(**inputs), references[step], atol=0.005, rtol=0.005)
                if compiled:
                    assert len(executions) > before
                if offloader is None:
                    assert all(weight.device.type == "cpu" for weight in model.language_model.layers.parameters())
                    assert all(weight.device.type == device for weight in model.gen_layers.parameters())
                else:
                    # The ring retains the first block for the next invocation.
                    for blocks in (model.language_model.layers, model.gen_layers):
                        assert next(blocks[0].parameters()).numel() > 0
                        assert next(blocks[-1].parameters()).numel() == 0
            runner.set_warmup(False)
            if request == 0:
                graph_count = len(graphs)
                assert not runner.warmup_status()["layouts_pending"]
            else:
                assert len(graphs) == graph_count
        if offloader is None:
            assert swaps == ["reasoner", "generator", "reasoner", "generator"]
        else:
            assert len(swaps) >= 2 * (2 + 2 * len(steps))
        assert weights == {name: id(weight) for name, weight in model.named_parameters()}
        if compiled and device == "cuda":
            assert any(
                "vllm_omni.block_sparse_request" in str(node.target) for graph in graphs for node in graph.graph.nodes
            )
    finally:
        runner.set_warmup(False)
        if offloader is None:
            model.disable_model_cpu_offload()
        else:
            offloader.disable()
        torch._dynamo.reset()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-rank strategy parity: Gloo/SDPA locally, NCCL/FA4 on two Hopper GPUs.

Run the GPU cases with pytest -m cuda; this test spawns both ranks itself.
No checkpoint download is needed. CPU cases exercise real collectives but
substitute SDPA for the CUDA providers; GPU cases exercise real sparse kernels.
"""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model]


def _cpu_group(rank, monkeypatch):
    from vllm_omni.diffusion.attention.parallel import factory
    from vllm_omni.diffusion.distributed import parallel_state, sp_sharding

    def all_gather(tensor, dim):
        outputs = [torch.empty_like(tensor) for _ in range(2)]
        dist.all_gather(outputs, tensor.contiguous())
        return torch.cat(outputs, dim=dim)

    group = SimpleNamespace(
        ulysses_group=dist.group.WORLD,
        ulysses_world_size=2,
        ulysses_rank=rank,
        ring_world_size=1,
        all_gather=all_gather,
    )
    # Keep the production sharding hooks and all-to-all implementation. Only
    # replace platform group discovery so the CPU gate needs no CUDA devices.
    for module in (parallel_state, sp_sharding, factory):
        monkeypatch.setattr(module, "get_sp_group", lambda: group)
        monkeypatch.setattr(module, "get_sequence_parallel_world_size", lambda: 2)
    for module in (parallel_state, sp_sharding):
        monkeypatch.setattr(module, "get_sequence_parallel_rank", lambda: rank)
    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_rank", lambda: rank)
    monkeypatch.setattr(parallel_state, "get_ring_parallel_world_size", lambda: 1)


@torch.inference_mode()
def _worker(rank, device_type, offload_mode, init_method):
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    from tests.diffusion.models.cosmos3.test_cosmos3_transformer import _tiny_cosmos3_config
    from tests.helpers.attention_strategy import configure_strategy_test, move_strategy_inputs
    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
    from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
    from vllm_omni.diffusion.hooks import apply_sequence_parallel
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3 as cosmos
    from vllm_omni.platforms import current_omni_platform

    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if device_type == "cuda" else "cpu")
    offloader = None
    with pytest.MonkeyPatch.context() as patch:
        if device_type == "cuda":
            current_omni_platform.set_device(device)
            parallel_state.init_distributed_environment(
                world_size=2, rank=rank, local_rank=rank, distributed_init_method=init_method, backend="nccl"
            )
            parallel_state.initialize_model_parallel(sequence_parallel_size=2, ulysses_degree=2)
        else:
            dist.init_process_group(
                "gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=120)
            )
            _cpu_group(rank, patch)
        try:
            factory = layer_mod.build_parallel_attention_strategy
            common, steps = configure_strategy_test(patch, device_type, "cosmos3.gen")
            patch.setattr(layer_mod, "build_parallel_attention_strategy", factory)
            patch.setattr(cosmos, "get_tensor_model_parallel_world_size", lambda: 1)
            model_config = _tiny_cosmos3_config(num_hidden_layers=2, hidden_size=16, num_attention_heads=4)
            if device_type == "cuda":
                model_config.update(
                    hidden_size=512, head_dim=128, intermediate_size=512, rope_scaling={"mrope_section": [24, 20, 20]}
                )

            def configuration(degree):
                return SimpleNamespace(
                    **{**common, "parallel_config": DiffusionParallelConfig(ulysses_degree=degree)},
                    num_gpus=degree,
                    tf_model_config=model_config,
                    enforce_eager=True,
                    diffusion_compile_granularity="regional",
                )

            def context(config, step):
                return override_forward_context(
                    ForwardContext(
                        omni_diffusion_config=config,
                        denoise_step_idx=step,
                        total_denoise_steps=len(steps),
                        sp_plan_hooks_applied=config.num_gpus > 1,
                    )
                )

            with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=device_type))):
                reference_config, sp_config = configuration(1), configuration(2)
                with set_current_diffusion_config(reference_config), context(reference_config, 0):
                    reference = cosmos.Cosmos3VFMTransformer(reference_config).to(device=device, dtype=common["dtype"])
                reference.post_load_weights()
                reference.eval()
                for weight in reference.parameters():
                    weight.fill_(1) if weight.ndim == 1 else weight.normal_(std=0.02)
                with set_current_diffusion_config(sp_config), context(sp_config, 0):
                    model = cosmos.Cosmos3VFMTransformer(sp_config).to(device=device, dtype=common["dtype"])
                model.load_state_dict(reference.state_dict())
                model.post_load_weights()
                model.eval()
                apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=2), model._sp_plan)
                assert model.gen_layers[0].cross_attention.attn.for_layout(0).parallel_strategy.name == "ulysses"
                shapes = (16, 17) if device_type == "cuda" else (3, 4)
                inputs = [
                    move_strategy_inputs(
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
                    for side in shapes
                ]
                references = {}
                for request in range(2):
                    for index, values in enumerate(inputs):
                        reference.reset_cache()
                        for step in steps:
                            with context(reference_config, step):
                                references[request, index, step] = reference(
                                    **{**values, "text_ids": values["text_ids"] + request}
                                )
                del reference
                identities = {name: id(weight) for name, weight in model.named_parameters()}
                if offload_mode == "model":
                    model.enable_model_cpu_offload(device=device, pin_memory=True)
                elif offload_mode == "layerwise":
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
                    offloader = LayerWiseOffloadBackend(OffloadConfig(strategy=OffloadStrategy.LAYER_WISE), device)
                    offloader.enable(pipeline)
                    assert len(offloader._dit_hooks) == 4
                graphs, executions = [], []
                sparse_calls = []
                if device_type == "cuda":
                    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
                    from vllm_omni.diffusion.attention.block_sparse import BlockSparseAttention
                    from vllm_omni.diffusion.attention.capabilities import (
                        ExecutionContext,
                        ParallelStrategy,
                        SupportStatus,
                    )

                    owners = {
                        id(executor.attention): (index, executor)
                        for index, block in enumerate(model.gen_layers)
                        for executor in block.cross_attention.attn._strategy_variants
                        if isinstance(executor.attention, BlockSparseAttention)
                    }
                    dispatch = BlockSparseAttention._dispatch_request

                    def record_sparse(impl, query, key, value, prefix):
                        output = dispatch(impl, query, key, value, prefix)
                        index, executor = owners[id(impl)]
                        # Observe real execution inside the opaque request op,
                        # not graph capture or fake-tensor preparation.
                        result = executor.resolve_execution_path(
                            ExecutionContext(platform="cuda"),
                            query,
                            key,
                            value,
                            AttentionMetadata(extra={"protected_kv_prefix": prefix}),
                            inputs_are_local=True,
                        )
                        assert result.parallel_strategy is ParallelStrategy.ULYSSES
                        assert result.support.status is SupportStatus.SUPPORTED
                        sparse_calls.append(index)
                        return output

                    patch.setattr(BlockSparseAttention, "_dispatch_request", record_sparse)

                def backend(graph, example_inputs):
                    compiled = (
                        torch._inductor.compile(graph, example_inputs) if device_type == "cuda" else graph.forward
                    )
                    graphs.append(graph)

                    def run(*args):
                        executions.append(graph)
                        return compiled(*args)

                    return run

                runner = model._attention_strategy_runner
                with pytest.raises(ValueError, match="regional compilation or eager"):
                    runner.compile()
                for compiled in (False, True):
                    if compiled:
                        runner.compile(granularity="regional", backend=backend, dynamic=True)
                    for request in range(2):
                        runner.set_warmup(compiled and request == 0)
                        for index, values in enumerate(inputs):
                            model.reset_cache()
                            for step in steps:
                                before = len(executions)
                                sparse_calls.clear()
                                with context(sp_config, step):
                                    actual = model(**{**values, "text_ids": values["text_ids"] + request})
                                torch.testing.assert_close(
                                    actual,
                                    references[request, index, step],
                                    atol=0.005 if device_type == "cuda" else 1e-5,
                                    rtol=0.016 if device_type == "cuda" else 1e-5,
                                )
                                if compiled:
                                    assert len(executions) > before
                                if device_type == "cuda":
                                    expected_calls = [0, 0, 1] if runner.warming else ([], [0], [0, 1])[step]
                                    assert sparse_calls == expected_calls, (rank, request, step, sparse_calls)
                        runner.set_warmup(False)
                        if request == 0:
                            graph_count = len(graphs)
                        else:
                            assert len(graphs) == graph_count, "Repeated request recompiled warmed regions"
                    if compiled:
                        assert not runner.warmup_status()["layouts_pending"]
                assert identities == {name: id(weight) for name, weight in model.named_parameters()}
                if device_type == "cuda":
                    assert any(
                        "vllm_omni.block_sparse_request" in str(node.target)
                        for graph in graphs
                        for node in graph.graph.nodes
                    )
                if offloader is not None:
                    offloader.disable()
                if offload_mode == "model":
                    model.disable_model_cpu_offload()
                dist.barrier()
        finally:
            torch._dynamo.reset()
            if device_type == "cuda":
                parallel_state.destroy_distributed_env()
            else:
                dist.destroy_process_group()


@pytest.mark.parametrize(
    ("device_type", "offload_mode"),
    [
        pytest.param("cpu", "none", marks=pytest.mark.cpu),
        pytest.param("cuda", "none", marks=pytest.mark.cuda),
        pytest.param("cuda", "model", marks=pytest.mark.cuda),
        pytest.param("cuda", "layerwise", marks=pytest.mark.cuda),
    ],
)
def test_strategy_ulysses_two_rank_parity(tmp_path, device_type, offload_mode):
    from vllm_omni.platforms import current_omni_platform

    if device_type == "cuda" and current_omni_platform.get_device_count() < 2:
        pytest.skip("Requires two Hopper GPUs for real NCCL and FA4 sparse attention")
    torch.multiprocessing.spawn(
        _worker, args=(device_type, offload_mode, f"file://{tmp_path / 'rendezvous'}"), nprocs=2
    )

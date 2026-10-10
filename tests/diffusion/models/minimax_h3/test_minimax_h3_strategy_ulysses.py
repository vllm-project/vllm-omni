# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 strategy parity with real Gloo collectives or two-GPU NCCL/FA4 execution."""

from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model]


@torch.inference_mode()
def _worker(rank, device_type, offload_mode, init_method):
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    from vllm import distributed as vllm_distributed
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    from tests.diffusion.models.cosmos3.test_cosmos3_strategy_ulysses import _cpu_group
    from tests.diffusion.models.minimax_h3.test_minimax_h3_sparse_ulysses import _configuration, _inputs
    from tests.helpers.attention_strategy import configure_strategy_test
    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.block_sparse import BlockSparseAttention
    from vllm_omni.diffusion.attention.capabilities import ExecutionContext, ParallelStrategy, SupportStatus
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
    from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
    from vllm_omni.diffusion.hooks import apply_sequence_parallel
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.offloader.base import OffloadConfig
    from vllm_omni.diffusion.offloader.config import OffloadStrategy
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend
    from vllm_omni.diffusion.offloader.sequential_backend import (
        ModelLevelOffloadBackend,
        sequential_offload_component,
    )
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
            patch.setattr(parallel_state, "get_allgather_parallel_world_size", lambda: 1)
        try:
            factory = layer_mod.build_parallel_attention_strategy
            common, _ = configure_strategy_test(patch, device_type, "minimax_h3.dit")
            patch.setattr(layer_mod, "build_parallel_attention_strategy", factory)
            patch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
            patch.setattr(vllm_distributed, "get_tensor_model_parallel_world_size", lambda: 1)
            # CPU substitutes SDPA for both providers but retains all three
            # layouts, production packing, sharding and all-to-all operations.
            attention = AttentionConfig(
                presets=common["diffusion_attention_config"].presets,
                layouts={
                    "dense": {"default": "dense"},
                    "mixed": {
                        "default": "dense",
                        "overrides": [{"attention_role": "minimax_h3.dit", "layers": [0], "use": "alternate"}],
                    },
                    "sparse": {
                        "default": "dense",
                        "overrides": [{"attention_role": "minimax_h3.dit", "use": "alternate"}],
                    },
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
            configs = (_configuration(1), _configuration(2))
            for config in configs:
                config.diffusion_attention_config = attention
                config.enforce_eager = True
                config.diffusion_compile_granularity = "regional"

            def context(config, step):
                return override_forward_context(
                    ForwardContext(
                        omni_diffusion_config=config,
                        denoise_step_idx=step,
                        total_denoise_steps=3,
                        sp_plan_hooks_applied=config.num_gpus > 1,
                    )
                )

            with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=device_type))):
                with set_current_diffusion_config(configs[0]), context(configs[0], 0):
                    reference = h3.MiniMaxH3DiTModel(configs[0]).to(device).eval()
                for weight in reference.parameters():
                    weight.fill_(1) if weight.ndim == 1 else weight.normal_(std=0.02)
                reference.post_load_weights()
                with set_current_diffusion_config(configs[1]), context(configs[1], 0):
                    model = h3.MiniMaxH3DiTModel(configs[1]).to(device).eval()
                model.load_state_dict(reference.state_dict())
                model.post_load_weights()
                apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=2), model._sp_plan)
                assert model.token_refiner.blocks[0].attn.attention.for_layout(0).skip_sequence_parallel
                assert model.blocks[0].attn.attention.for_layout(0).parallel_strategy.name == "ulysses"
                inputs = [_inputs(length, device) for length in (256, 257)]
                expected = {}
                for request in range(2):
                    for index, values in enumerate(inputs):
                        for step in range(3):
                            with context(configs[0], step):
                                expected[request, index, step] = reference(
                                    **{**values, "prompt_embeds": values["prompt_embeds"] + request * 0.1}
                                )
                del reference
                pipeline = object.__new__(MiniMaxH3Pipeline)
                torch.nn.Module.__init__(pipeline)
                pipeline.transformer = model
                pipeline._dit_modules = ["transformer"]
                pipeline._encoder_modules = []
                pipeline._vae_modules = []
                if offload_mode == "model":
                    offloader = ModelLevelOffloadBackend(OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL), device)
                elif offload_mode == "layerwise":
                    offloader = LayerWiseOffloadBackend(OffloadConfig(strategy=OffloadStrategy.LAYER_WISE), device)
                if offloader is not None:
                    offloader.enable(pipeline)
                    if offload_mode == "layerwise":
                        assert len(offloader._dit_hooks) == len(model.blocks)
                        assert all(p.device == device for p in model.token_refiner.parameters())
                graphs, executions, sparse_calls = [], [], []
                owners = {
                    id(executor.attention): executor
                    for block in model.blocks
                    for executor in block.attn.attention._strategy_variants
                    if isinstance(executor.attention, BlockSparseAttention)
                }
                dispatch = BlockSparseAttention._dispatch_request

                def record_sparse(impl, q, k, v, prefix):
                    assert q.shape[1] in (256, 257)  # Packed padding is trimmed.
                    assert q.shape[2] == 1  # Two heads split across two ranks.
                    sparse_calls.append(q.shape[1])
                    output = dispatch(impl, q, k, v, prefix)
                    path = owners[id(impl)].resolve_execution_path(
                        ExecutionContext(platform="cuda"),
                        q,
                        k,
                        v,
                        AttentionMetadata(extra={"protected_kv_prefix": prefix}),
                        inputs_are_local=True,
                    )
                    assert path.parallel_strategy is ParallelStrategy.ULYSSES
                    assert path.support.status is SupportStatus.SUPPORTED
                    return output

                patch.setattr(BlockSparseAttention, "_dispatch_request", record_sparse)

                def backend(graph, args):
                    compiled = torch._inductor.compile(graph, args) if device_type == "cuda" else graph.forward
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
                            for step in range(3):
                                before = len(executions)
                                sparse_calls.clear()
                                lifecycle = (
                                    sequential_offload_component(model) if offload_mode == "model" else nullcontext()
                                )
                                with context(configs[1], step), lifecycle:
                                    # Production denoise inputs carry rank-local precomputed RoPE.
                                    rope = model.prepare_rope_table(
                                        values["img_position_ids"], seq_len=values["x"].shape[1]
                                    )
                                    assert rope.shape[0] == values["x"].shape[1] // 2
                                    actual = model(
                                        **{
                                            **values,
                                            "rope_table": rope if step else None,
                                            "prompt_embeds": values["prompt_embeds"] + request * 0.1,
                                        }
                                    )
                                torch.testing.assert_close(
                                    actual, expected[request, index, step], atol=0.005, rtol=0.02
                                )
                                if device_type == "cuda":
                                    assert len(sparse_calls) == (3 if runner.warming else step)
                                if compiled:
                                    assert len(executions) > before
                                if offload_mode == "model":
                                    assert all(p.device.type == "cpu" for p in model.parameters())
                        runner.set_warmup(False)
                        if request == 0:
                            warmed = len(graphs)
                        else:
                            assert len(graphs) == warmed, "Repeated request recompiled warmed regions"
                    if compiled:
                        assert not runner.warmup_status()["layouts_pending"]
                dist.barrier()
        finally:
            if offloader is not None:
                offloader.disable()
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
        pytest.skip("Requires two Hopper GPUs for NCCL and FA4 sparse attention")
    torch.multiprocessing.spawn(
        _worker, args=(device_type, offload_mode, f"file://{tmp_path / 'rendezvous'}"), nprocs=2
    )

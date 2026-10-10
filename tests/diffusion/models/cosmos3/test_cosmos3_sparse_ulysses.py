# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Static sparse Ulysses: native FA4 head shards and two-rank Cosmos3 parity."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cuda]


@pytest.fixture(autouse=True)
def require_hopper():
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 9 or capability.minor != 0:
        pytest.skip("Requires Hopper SM90 and the 64x64 FA4 sparse kernel")
    pytest.importorskip("flash_attn.cute")


def sparse_spec():
    return {
        "name": "block_sparse",
        "config": {
            "block_size": [64, 64],
            "selection": {"name": "block_topk", "config": {"target_sparsity": 0.75}},
            "backend": {"require": "FLASH_ATTN", "implementation": "auto"},
        },
    }


def move_inputs(values, device):
    return {
        key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in values.items()
    }


@torch.inference_mode()
def test_sparse_head_shards_and_padding_on_one_gpu(monkeypatch):
    """Validate the sparse math separately from the two-GPU communication gate."""
    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.capabilities import (
        CompilationMode,
        ExecutionContext,
        ParallelStrategy,
        SupportStatus,
    )
    from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, DiffusionParallelConfig
    from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: NoParallelAttention())

    def executor(degree):
        config = SimpleNamespace(
            parallel_config=DiffusionParallelConfig(ulysses_degree=degree),
            diffusion_attention_config=AttentionConfig(default=sparse_spec()),
            diffusion_kv_cache_dtype=None,
        )
        with set_current_diffusion_config(config):
            return layer_mod.Attention(
                num_heads=4,
                num_kv_heads=2,
                head_size=128,
                causal=False,
                softmax_scale=128**-0.5,
            )

    q = torch.randn(1, 1026, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1029, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    baseline = executor(1)
    expected = baseline(
        q[:, :-1].contiguous(),
        k[:, :-1].contiguous(),
        v[:, :-1].contiguous(),
        AttentionMetadata(extra={"protected_kv_prefix": 3}),
    )
    parts = []
    for rank in range(2):

        class HeadShard(NoParallelAttention):
            @property
            def enabled(self):
                return True

            @property
            def name(self):
                return "ulysses"

            def pre_attention(self, query, key, value, metadata):
                return (
                    query[:, :, 2 * rank : 2 * rank + 2].contiguous(),
                    key[:, :, rank : rank + 1].contiguous(),
                    value[:, :, rank : rank + 1].contiguous(),
                    metadata,
                    None,
                )

        attention = executor(2)
        assert attention.attention.num_heads == 2
        assert attention.attention.num_kv_heads == 1
        attention.parallel_strategy = HeadShard()
        with override_forward_context(ForwardContext(sp_plan_hooks_applied=True, _sp_shard_depth=1)):
            local = (
                q[:, :, 2 * rank : 2 * rank + 2].contiguous()[:, :-1].contiguous(),
                k[:, :, rank : rank + 1].contiguous()[:, :-1].contiguous(),
                v[:, :, rank : rank + 1].contiguous()[:, :-1].contiguous(),
            )
            metadata = AttentionMetadata(extra={"protected_kv_prefix": 3})
            context = ExecutionContext(platform="cuda", require_fullgraph=True)
            pending = attention.resolve_execution_path(context, *local, metadata, inputs_are_local=True)
            assert pending.support.status is SupportStatus.UNSUPPORTED
            assert "not prepared" in pending.support.reason
            parts.append(
                attention(q, k, v, AttentionMetadata(extra={"protected_kv_prefix": 3, "ulysses_sp_padding": 1}))
            )
            result = attention.resolve_execution_path(context, *local, metadata, inputs_are_local=True)
            assert result.support.status is SupportStatus.SUPPORTED
            assert result.parallel_strategy is ParallelStrategy.ULYSSES
            assert result.compilation_mode is CompilationMode.EAGER_ONLY
            assert result.requested_support(context).status is SupportStatus.UNSUPPORTED
            raw = attention.resolve_execution_path(context, q, k, v, metadata)
            assert raw.support.status is SupportStatus.UNSUPPORTED
            assert "post-communication" in raw.support.reason
            # The local kernel must not claim ownership of the wrapper.
            kernel = attention.attention.resolve_execution_path(
                ExecutionContext(platform="cuda", parallel_strategy=ParallelStrategy.ULYSSES), *local, metadata
            )
            assert kernel.support.status is SupportStatus.UNSUPPORTED

            def unexpected_execution(*args, **kwargs):
                raise AssertionError("Capability queries must not execute kernels or collectives")

            monkeypatch.setattr(attention.parallel_strategy, "pre_attention", unexpected_execution)
            monkeypatch.setattr(attention.attention, "_execute", unexpected_execution)
            assert (
                attention.resolve_execution_path(context, *local, metadata, inputs_are_local=True).support.status
                is SupportStatus.SUPPORTED
            )
            changed = tuple(t[:, :-1] for t in local)
            assert (
                attention.resolve_execution_path(context, *changed, metadata, inputs_are_local=True).support.status
                is SupportStatus.UNSUPPORTED
            )
            attention.use_ring = True
            assert (
                attention.resolve_execution_path(context, *local, metadata, inputs_are_local=True).support.status
                is SupportStatus.UNSUPPORTED
            )
            attention.use_ring = False
            with override_forward_context(
                ForwardContext(
                    omni_diffusion_config=SimpleNamespace(parallel_config=SimpleNamespace(ulysses_mode="advanced_uaa")),
                    sp_plan_hooks_applied=True,
                    _sp_shard_depth=1,
                )
            ):
                rejected = attention.resolve_execution_path(context, *local, metadata, inputs_are_local=True)
                assert rejected.support.status is SupportStatus.UNSUPPORTED
                assert "strict Ulysses" in rejected.support.reason
                with pytest.raises(ValueError, match="strict Ulysses"):
                    attention(q, k, v, metadata)
    actual = torch.cat(parts, dim=2)
    torch.testing.assert_close(actual[:, :-1], expected, atol=0.005, rtol=0.016)
    assert torch.count_nonzero(actual[:, -1]) == 0


@torch.inference_mode()
def _worker(rank, offload_mode, init_method, degree=2):
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    import torch.distributed as dist
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    from tests.diffusion.models.cosmos3.test_cosmos3_transformer import _tiny_cosmos3_config
    from vllm_omni.diffusion.compile import regionally_compile
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, DiffusionParallelConfig
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
    from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
    from vllm_omni.diffusion.hooks import apply_sequence_parallel
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3 as cosmos
    from vllm_omni.platforms import current_omni_platform

    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device(f"cuda:{rank}")
    offloader = None
    with pytest.MonkeyPatch.context() as patch:
        current_omni_platform.set_device(device)
        parallel_state.init_distributed_environment(
            world_size=degree, rank=rank, local_rank=rank, distributed_init_method=init_method, backend="nccl"
        )
        parallel_state.initialize_model_parallel(sequence_parallel_size=degree, ulysses_degree=degree)
        try:
            from vllm.model_executor import parameter
            from vllm.model_executor.layers import linear

            for module in (linear, parameter):
                patch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
                patch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
            common = dict(
                diffusion_attention_config=AttentionConfig(
                    default="FLASH_ATTN",
                    per_role={"cosmos3.gen": sparse_spec()},
                ),
                dtype=torch.bfloat16,
                diffusion_kv_cache_dtype=None,
            )
            steps = (0, 1)
            patch.setattr(cosmos, "get_tensor_model_parallel_world_size", lambda: 1)
            model_config = _tiny_cosmos3_config(num_hidden_layers=2, hidden_size=16, num_attention_heads=4)
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
                        sp_plan_hooks_applied=config.num_gpus > 1,
                    )
                )

            with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
                reference_config, sp_config = configuration(1), configuration(degree)
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
                if degree > 1:
                    apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=degree), model._sp_plan)
                    assert model.gen_layers[0].cross_attention.attn.parallel_strategy.name == "ulysses"
                # More than eight candidate blocks: exercise actual sparse selection.
                shapes = (32, 33)
                inputs = [
                    move_inputs(
                        dict(
                            hidden_states=torch.randn(1, 2, 1, side, side),
                            timestep=torch.ones(1),
                            text_ids=torch.zeros(1, 3, dtype=torch.long),
                            text_mask=torch.ones(1, 3, dtype=torch.long),
                            video_shape=(1, side, side),
                        ),
                        device,
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
                    for executor in (block.cross_attention.attn,)
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
                    assert result.parallel_strategy is (
                        ParallelStrategy.ULYSSES if degree > 1 else ParallelStrategy.NONE
                    )
                    assert result.support.status is SupportStatus.SUPPORTED
                    sparse_calls.append(index)
                    return output

                patch.setattr(BlockSparseAttention, "_dispatch_request", record_sparse)

                def backend(graph, example_inputs):
                    compiled = torch._inductor.compile(graph, example_inputs)
                    graphs.append(graph)

                    def run(*args):
                        executions.append(graph)
                        return compiled(*args)

                    return run

                for compiled in (False, True):
                    if compiled:
                        regionally_compile(model, backend=backend, dynamic=True, fullgraph=False)
                    for request in range(2):
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
                                    atol=0.005,
                                    rtol=0.016,
                                )
                                if compiled:
                                    assert len(executions) > before
                                expected_calls = [0, 1]
                                assert sparse_calls == expected_calls, (rank, request, step, sparse_calls)
                        if request == 0:
                            graph_count = len(graphs)
                        else:
                            assert len(graphs) == graph_count, "Repeated request recompiled warmed regions"
                assert identities == {name: id(weight) for name, weight in model.named_parameters()}
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
            parallel_state.destroy_distributed_env()


@pytest.mark.parallel
@pytest.mark.parametrize("offload_mode", ["none", "model", "layerwise"])
def test_static_sparse_ulysses_two_rank_parity(tmp_path, offload_mode):
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.get_device_count() < 2:
        pytest.skip("Requires two Hopper GPUs for NCCL/FA4 validation")
    torch.multiprocessing.spawn(
        _worker,
        args=(offload_mode, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=2,
    )


@pytest.mark.parametrize("offload_mode", ["model", "layerwise"])
def test_static_sparse_offloading(tmp_path, offload_mode):
    _worker(0, offload_mode, f"file://{tmp_path / 'rendezvous'}", degree=1)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real sparse MiniMax H3 execution with Ulysses and CPU offloading."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cuda]


@pytest.fixture(autouse=True)
def require_hopper():
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 9 or capability.minor != 0:
        pytest.skip("Requires Hopper SM90 and FA4")
    pytest.importorskip("flash_attn.cute")


def _configuration(degree):
    from vllm_omni.diffusion.data import AttentionConfig, DiffusionParallelConfig

    return SimpleNamespace(
        tf_model_config=dict(
            num_layers=2,
            token_refiner_num_layers=1,
            hidden_size=256,
            num_attention_heads=2,
            attention_head_dim=128,
            ffn_hidden_size=512,
            latents_dim=2,
            audio_latents_dim=2,
            patch_size=(1, 1, 1),
            text_dim=256,
            timestep_input_dim=32,
            time_embed_hidden_size=64,
            time_embed_dim=128,
            adaln_out_features=18 * 256,
            final_adaln_out_features=2 * 256,
            rope_inv_freq_len=16,
        ),
        diffusion_attention_config=AttentionConfig(
            default="FLASH_ATTN",
            per_role={
                "minimax_h3.dit": {
                    "name": "block_sparse",
                    "config": {
                        "block_size": [64, 64],
                        "selection": {"name": "block_topk", "config": {"target_sparsity": 0.75}},
                        "backend": {"require": "FLASH_ATTN", "implementation": "auto"},
                    },
                }
            },
        ),
        parallel_config=DiffusionParallelConfig(ulysses_degree=degree, data_parallel_size=1),
        dtype=torch.bfloat16,
        diffusion_kv_cache_dtype=None,
        num_gpus=degree,
    )


def _inputs(real_length, device):
    # Single document followed by suffix padding; both totals divide by U=2.
    total = ((real_length + 63) // 64) * 64
    text = torch.arange(3, device=device)
    audio = torch.arange(3, 7, device=device)
    video = torch.arange(7, real_length, device=device)
    tags = torch.full((total,), 2, device=device, dtype=torch.long)
    tags[:3], tags[3:7], tags[real_length:] = 0, 1, -1
    return dict(
        x=torch.randn(1, total, 2, device=device),
        audio_x=torch.randn(1, total, 2, device=device),
        img_position_ids=torch.zeros(1, total, 3, device=device, dtype=torch.long),
        unique_timesteps=torch.tensor([0.5], device=device),
        inverse_indices=torch.zeros(total, device=device, dtype=torch.long),
        update_mask=torch.ones(video.numel(), device=device),
        token_tags=tags,
        prompt_embeds=torch.randn(3, 256, device=device, dtype=torch.bfloat16),
        img_pos_info={"position_ids": video},
        audio_pos_info={"position_ids": audio},
        text_pos_info={"position_ids": text},
        img_pos_for_infer_output_info={"position_ids": video},
        packed_seq_params={
            "cu_seqlens_q": torch.tensor([0, real_length, total], device=device, dtype=torch.int32),
            "max_seqlen_q": real_length,
        },
        refiner_packed_seq_params={
            "cu_seqlens_q": torch.tensor([0, 3], device=device, dtype=torch.int32),
            "max_seqlen_q": 3,
        },
    )


@torch.inference_mode()
def _run_case(rank, degree, offload_mode):
    from vllm import distributed as vllm_distributed
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from vllm_omni.diffusion.attention.block_sparse import BlockSparseAttention
    from vllm_omni.diffusion.compile import regionally_compile
    from vllm_omni.diffusion.config import set_current_diffusion_config
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

    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device(f"cuda:{rank}")
    configs = (_configuration(1), _configuration(degree))

    def context(config):
        return override_forward_context(
            ForwardContext(
                omni_diffusion_config=config,
                sp_plan_hooks_applied=config.num_gpus > 1,
            )
        )

    offloader = None
    with (
        pytest.MonkeyPatch.context() as patch,
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))),
    ):
        patch.setattr(vllm_distributed, "get_tensor_model_parallel_world_size", lambda: 1)
        for module in (linear, parameter):
            patch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
            patch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        patch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
        with set_current_diffusion_config(configs[0]), context(configs[0]):
            reference = h3.MiniMaxH3DiTModel(configs[0]).to(device=device).eval()
        for weight in reference.parameters():
            weight.fill_(1) if weight.ndim == 1 else weight.normal_(std=0.02)
        reference.post_load_weights()
        inputs = [_inputs(length, device) for length in (1024, 1089)]
        with context(configs[0]):
            expected = [reference(**values) for values in inputs]
        with set_current_diffusion_config(configs[1]), context(configs[1]):
            model = h3.MiniMaxH3DiTModel(configs[1]).to(device=device).eval()
        model.load_state_dict(reference.state_dict())
        model.post_load_weights()
        del reference
        if degree > 1:
            apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=degree), model._sp_plan)
        assert model.token_refiner.blocks[0].attn.attention.skip_sequence_parallel
        for block in model.blocks:
            assert block.attn.attention.attention.num_heads == 2 // degree
        # Use the production pipeline's discovery and offload lifecycle,
        # without loading encoder/VAE checkpoints.
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
        if offloader:
            offloader.enable(pipeline)
            assert offloader.enabled
            if offload_mode == "layerwise":
                assert len(offloader._dit_hooks) == len(model.blocks)
                # Ordinary layerwise offload keeps non-DiT-block state resident;
                # only distributed layerwise offload streams nested submodules.
                assert all(p.device == device for p in model.token_refiner.parameters())
        dispatch = BlockSparseAttention._dispatch_request
        sparse_calls = []

        def record(impl, q, k, v, prefix):
            assert q.shape[1] in (1024, 1089)  # Padding removed before selection.
            assert q.shape[2] == 2 // degree
            sparse_calls.append(q.shape[1])
            return dispatch(impl, q, k, v, prefix)

        patch.setattr(BlockSparseAttention, "_dispatch_request", record)
        graphs, executions = [], []

        def backend(graph, args):
            compiled = torch._inductor.compile(graph, args)
            graphs.append(graph)

            def execute(*args):
                executions.append(graph)
                return compiled(*args)

            return execute

        try:
            for compiled in (False, True):
                if compiled:
                    regionally_compile(model, backend=backend, dynamic=True, fullgraph=False)
                for repeat in range(2):
                    for values, output in zip(inputs, expected, strict=True):
                        sparse_calls.clear()
                        before = len(executions)
                        lifecycle = sequential_offload_component(model) if offload_mode == "model" else nullcontext()
                        with context(configs[1]), lifecycle:
                            actual = model(**values)
                        torch.testing.assert_close(actual, output, atol=0.005, rtol=0.02)
                        assert len(sparse_calls) == len(model.blocks)
                        if compiled:
                            assert len(executions) > before
                        if offload_mode == "model":
                            assert all(p.device.type == "cpu" for p in model.parameters())
                    if repeat == 0:
                        warmed = len(graphs)
                    else:
                        assert len(graphs) == warmed
        finally:
            if offloader:
                offloader.disable()
            torch._dynamo.reset()


def _worker(rank, offload_mode, init_method):
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device(torch.device(f"cuda:{rank}"))
    parallel_state.init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=init_method,
        backend="nccl",
    )
    parallel_state.initialize_model_parallel(sequence_parallel_size=2, ulysses_degree=2)
    try:
        _run_case(rank, 2, offload_mode)
    finally:
        parallel_state.destroy_distributed_env()


@pytest.mark.parametrize("offload_mode", ["model", "layerwise"])
def test_sparse_offloading(offload_mode):
    _run_case(0, 1, offload_mode)


@pytest.mark.parallel
@pytest.mark.parametrize("offload_mode", ["none", "model", "layerwise"])
def test_sparse_ulysses_offloading(tmp_path, offload_mode):
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.get_device_count() < 2:
        pytest.skip("Requires two Hopper GPUs")
    torch.multiprocessing.spawn(_worker, args=(offload_mode, f"file://{tmp_path / 'rendezvous'}"), nprocs=2)

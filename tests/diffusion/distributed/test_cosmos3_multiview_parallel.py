# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real collective and denoising parity for Cosmos3 Multiview-AV.

The small transformer has production GQA/head dimensions but needs no exported
checkpoint or media. Full-resolution performance remains a separate benchmark.
"""

from __future__ import annotations

import gc
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel]

# (CFGP, CP, TP, HSDP shard size, HSDP replica size)
TWO_GPU_CASES = [(2, 1, 1, 0, 1), (1, 2, 1, 0, 1), (1, 1, 2, 0, 1), (1, 1, 1, 2, 1), (2, 1, 1, 2, 1), (1, 2, 1, 2, 1)]
SCALE_CASES = [
    (1, 4, 1, 0, 1),
    (1, 8, 1, 0, 1),
    (2, 2, 1, 0, 1),
    (2, 4, 1, 0, 1),
    (1, 1, 4, 0, 1),
    (1, 1, 8, 0, 1),
    (2, 1, 2, 0, 1),
    (1, 2, 2, 0, 1),
    (2, 2, 2, 0, 1),
    (1, 1, 1, 4, 1),
    (1, 1, 1, 8, 1),
    (2, 2, 1, 4, 1),
    (2, 4, 1, 8, 1),
    (2, 2, 1, 2, 2),
]


def _config(case):
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

    cfg, cp, tp, shard, replicate = case
    parallel = DiffusionParallelConfig(
        cfg_parallel_size=cfg,
        ulysses_degree=cp,
        tensor_parallel_size=tp,
        use_hsdp=bool(shard),
        hsdp_shard_size=shard or -1,
        hsdp_replicate_size=replicate,
    )
    config = OmniDiffusionConfig.from_kwargs(
        model="test", dtype=torch.bfloat16, parallel_config=parallel, diffusion_attention_backend="TORCH_SDPA"
    )
    config.tf_model_config = {
        "backbone_type": "cosmos3_multiview",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 128,
        "vocab_size": 32,
        "latent_patch_size": 1,
        "latent_channel": 2,
        "rope_scaling": {"mrope_section": [24, 20, 20]},
    }
    return config


def _pipeline(config, device):
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer

    # Exercise the real denoising loop with synthetic latents; VAE/media and
    # tokenizer construction are deliberately outside this checkpoint-free test.
    pipeline = Cosmos3MultiviewPipeline.__new__(Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = config
    pipeline.device = device
    pipeline.dtype = config.dtype
    pipeline.transformer = Cosmos3MultiviewVFMTransformer(od_config=config).to(dtype=config.dtype)
    pipeline._cache_dit_requires_paired_cfg = False
    pipeline.progress_bar = lambda xs: xs
    return pipeline


def _denoise(pipeline, backend, request_index):
    from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import MultiviewLayout
    from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

    # 66/110 GEN tokens cut across all eleven cameras and require padding at CP4/8.
    frames_per_view = 3 + 2 * request_index
    frames = 11 * frames_per_view
    rng = torch.Generator().manual_seed(42 + request_index)
    latents = torch.randn(1, 2, frames, 1, 1, generator=rng).to(pipeline.device, pipeline.dtype)
    control = torch.randn(latents.shape, generator=rng).to(pipeline.device, pipeline.dtype)
    condition = torch.randn(latents.shape, generator=rng).to(pipeline.device, pipeline.dtype)
    mask = torch.ones(1, 1, frames, 1, 1, device=pipeline.device, dtype=pipeline.dtype)
    if request_index:
        mask[:, :, ::frames_per_view] = 0
        latents = mask * latents + (1 - mask) * condition
    pos_len, neg_len = 7 + request_index, 3 + request_index
    positive = torch.arange(pos_len, device=pipeline.device).unsqueeze(0)
    negative = torch.arange(neg_len, device=pipeline.device).unsqueeze(0) + 10
    pipeline.scheduler = FlowUniPCMultistepScheduler()
    pipeline.scheduler.set_timesteps(3, device=pipeline.device)
    und_calls = []
    handle = pipeline.transformer.language_model.register_forward_pre_hook(
        lambda _module, args: und_calls.append((args[0].shape[1], args[0][0, 0].item()))
    )
    result = pipeline.diffuse(
        latents=latents,
        timesteps=pipeline.scheduler.timesteps,
        cond_ids=positive,
        cond_mask=torch.ones_like(positive),
        uncond_ids=negative,
        uncond_mask=torch.ones_like(negative),
        guidance_scale=6.0 if request_index == 0 else 1.0,
        velocity_mask=mask,
        condition_latents=condition,
        shared_kwargs={
            "video_shape": (frames, 1, 1),
            "fps": 30.0,
            "noisy_frame_mask": mask,
            "control_latents": [control],
            "temporal_position_period": frames_per_view,
            "multiview_layout": MultiviewLayout(
                11,
                frames,
                1,
                1,
                backend=backend,
                max_und_tokens=64,
                decomposed_temporal_window_seconds=0.5,
                control_attends_sensor=True,
            ),
        },
    )
    handle.remove()
    if request_index:
        expected_calls = [(pos_len, 0)]
    elif pipeline.od_config.parallel_config.cfg_parallel_size == 2:
        from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_rank

        expected_calls = [(pos_len, 0)] if get_classifier_free_guidance_rank() == 0 else [(neg_len, 10)]
    else:
        expected_calls = [(pos_len, 0), (neg_len, 10)]
    assert und_calls == expected_calls, "UND should encode only this rank's branches, once per request"
    assert torch.isfinite(result).all()
    if request_index:
        torch.testing.assert_close(result[:, :, ::frames_per_view], condition[:, :, ::frames_per_view], atol=0, rtol=0)
    return result


def _worker(rank, world_size, port, case, backend, compile_blocks):
    from vllm.config import DeviceConfig, VllmConfig

    from vllm_omni.diffusion.compile import regionally_compile
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.distributed.hsdp import HSDPInferenceConfig, apply_hsdp_to_model
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
    from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
    from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel

    device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(device)
    # Bound collective hangs, including failures during HSDP cache population.
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    try:
        init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank)
        initialize_model_parallel(data_parallel_size=world_size)
        vllm_config = VllmConfig(device_config=DeviceConfig(device="cuda"))
        reference_config = _config((1, 1, 1, 0, 1))
        with (
            set_forward_context(vllm_config=vllm_config, omni_diffusion_config=reference_config),
            set_current_diffusion_config(reference_config),
            torch.no_grad(),
        ):
            reference = _pipeline(reference_config, device)
            torch.manual_seed(7)
            for name, parameter in reference.named_parameters():
                if "norm" in name and parameter.ndim == 1:
                    parameter.fill_(1)
                else:
                    parameter.normal_(std=0.02)
            reference.transformer.post_load_weights()
            weights = {name: value.cpu().clone() for name, value in reference.state_dict().items()}
            reference.transformer.to(device).eval()
            expected = [_denoise(reference, backend, i).cpu() for i in range(2)]
            full_numel = sum(p.numel() for p in reference.transformer.parameters())
            del reference
        gc.collect()
        torch.accelerator.empty_cache()
        destroy_model_parallel()

        cfg, cp, tp, shard, replicate = case
        config = _config(case)
        initialize_model_parallel(
            cfg_parallel_size=cfg,
            sequence_parallel_size=cp,
            ulysses_degree=cp,
            tensor_parallel_size=tp,
            use_hsdp=bool(shard),
            fully_shard_degree=shard or 1,
        )
        with (
            set_forward_context(vllm_config=vllm_config, omni_diffusion_config=config),
            set_current_diffusion_config(config),
            torch.no_grad(),
        ):
            pipeline = _pipeline(config, device)
            # Use the production TP-aware remapping/weight loading path.
            pipeline.load_weights(iter(weights.items()))
            if shard:
                apply_hsdp_to_model(
                    pipeline.transformer,
                    HSDPInferenceConfig(enabled=True, hsdp_shard_size=shard, hsdp_replicate_size=replicate),
                    target_device=device,
                )
                from torch.distributed.tensor import DTensor

                assert isinstance(pipeline.transformer.gen_layers[0].cross_attention.to_q.weight, DTensor)
                assert pipeline.transformer.time_embedder.linear_1.weight.dtype == torch.float32
                assert pipeline.transformer.time_embedder.linear_1.weight.device == device
            else:
                pipeline.transformer.to(device)
            if tp > 1 or shard:
                local_numel = sum(
                    (p.to_local() if hasattr(p, "to_local") else p).numel() for p in pipeline.transformer.parameters()
                )
                assert local_numel < full_numel
            if cp > 1:
                apply_sequence_parallel(
                    pipeline.transformer, SequenceParallelConfig(ulysses_degree=cp), pipeline.transformer._sp_plan
                )
                get_forward_context().sp_plan_hooks_applied = True
            if compile_blocks:
                regionally_compile(pipeline.transformer, dynamic=False)
            # Repeat with new text lengths, camera geometry, and guidance disabled
            # to exercise cache resets and identical collective schedules.
            for i in range(2):
                actual = _denoise(pipeline, backend, i)
                torch.testing.assert_close(actual.cpu(), expected[i], atol=2e-2, rtol=2e-2)
                gathered = [torch.empty_like(actual) for _ in range(world_size)]
                dist.all_gather(gathered, actual)
                for other in gathered:
                    torch.testing.assert_close(actual, other, atol=0, rtol=0)
    finally:
        destroy_distributed_env()


def _run(case, backend, port, compile_blocks=False):
    cfg, cp, tp, shard, replicate = case
    world_size = max(cfg * cp * tp, shard * replicate)
    if not torch.cuda.is_available() or torch.accelerator.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA devices")
    if backend == "fa4":
        if torch.cuda.get_device_capability()[0] != 10:
            pytest.skip("FA4 requires datacenter Blackwell")
        pytest.importorskip("flash_attn.cute")
    torch.multiprocessing.spawn(_worker, args=(world_size, port, case, backend, compile_blocks), nprocs=world_size)


@pytest.mark.parametrize("case", TWO_GPU_CASES)
@pytest.mark.parametrize("backend", ["triton", "fa4"])
@hardware_test(res={"cuda": "B200"}, num_cards=2)
def test_multiview_two_gpu_denoising(case, backend, unused_tcp_port):
    _run(case, backend, unused_tcp_port)


@pytest.mark.parametrize("case", SCALE_CASES)
@pytest.mark.parametrize("backend", ["triton", "fa4"])
@hardware_test(res={"cuda": "B200"}, num_cards=8)
def test_multiview_scaling_denoising(case, backend, unused_tcp_port):
    _run(case, backend, unused_tcp_port)


@pytest.mark.parametrize("case", [(1, 2, 1, 0, 1), (2, 1, 1, 2, 1), (1, 1, 2, 0, 1)])
@pytest.mark.parametrize("backend", ["triton", "fa4"])
@hardware_test(res={"cuda": "B200"}, num_cards=2)
def test_multiview_compiled_denoising(case, backend, unused_tcp_port):
    _run(case, backend, unused_tcp_port, compile_blocks=True)


def _recompile_worker(rank, port):
    from torch._dynamo.testing import CompileCounterWithBackend

    from vllm_omni.diffusion.models.cosmos3 import multiview_flex_attention as sparse
    from vllm_omni.diffusion.models.cosmos3.multiview_parallel import multiview_ulysses_attention

    device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    try:
        torch.manual_seed(7)
        q = torch.randn(1, 66, 8, 128, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, 66, 2, 128, dtype=torch.bfloat16, device=device)
        v = torch.randn_like(k)
        layout = sparse.MultiviewLayout(11, 33, 1, 1, max_und_tokens=64)
        counter = CompileCounterWithBackend("inductor")
        sparse._compiled_flex_attention = torch.compile(sparse.torch_flex_attention, backend=counter, dynamic=False)
        for text_len in (3, 7, 35, 47):
            ku = torch.randn(1, text_len, 2, 128, dtype=torch.bfloat16, device=device)
            output = multiview_ulysses_attention(
                *(t.chunk(2, dim=1)[rank].contiguous() for t in (q, k, v)),
                ku,
                torch.randn_like(ku),
                sparse.MultiviewAttentionContext(layout, {}),
                group=dist.group.WORLD,
                rank=rank,
                world_size=2,
            )
            assert torch.isfinite(output).all()
        assert counter.frame_count == 1, f"CP sparse kernel recompiled for text length: {counter.frame_count} graphs"
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_multiview_cp_prompt_lengths_do_not_recompile(unused_tcp_port):
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    torch.multiprocessing.spawn(_recompile_worker, args=(unused_tcp_port,), nprocs=2)

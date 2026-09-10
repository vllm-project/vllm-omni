# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tiny LingBot numeric gates with real SP/TP groups, loaders and attention.

FP32 direct tests use batch two and native SDPA; batch-one direct/paged tests
use BF16 and the production FlashAttention path. No checkpoint is downloaded.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel]

_CASES = [
    pytest.param("direct", torch.float32, 2, id="direct-fp32"),
    pytest.param("direct", torch.bfloat16, 1, id="direct-bf16"),
    pytest.param("paged", torch.bfloat16, 1, id="paged-bf16"),
]
_HEADS, _HEAD_DIM, _LAYERS = 8, 32, 2
_FRAMES, _SIDE, _TOKENS_PER_FRAME = 3, 32, 256


def _model(dtype):
    from vllm_omni.diffusion.models.lingbot_world.transformer import CausalLingBotWorldTransformer3DModel

    model = CausalLingBotWorldTransformer3DModel(
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        num_layers=_LAYERS,
        in_channels=4,
        out_channels=4,
        text_dim=8,
        freq_dim=16,
        ffn_dim=512,
        patch_size=(1, 2, 2),
        sink_size=1,
        num_frames_per_block=_FRAMES,
        sliding_window_num_frames=6,
        rope_max_seq_len=32,
    ).eval()
    return model.to(device=torch.device("cuda", torch.accelerator.current_device_index()), dtype=dtype)


def _checkpoint(model):
    weights = {}
    generator = torch.Generator().manual_seed(17)
    for name, param in sorted(model.named_parameters()):
        data = torch.randn(param.shape, generator=generator) * 0.02
        if "norm" in name and name.endswith("weight"):
            data.fill_(1)
        if name.endswith("modulation") and data.shape[-2] == 6:
            data[..., 2, :] = 1
            data[..., 5, :] = 1
        data = data.to(param.dtype)
        param.copy_(data)
        if ".self_attn.qkv." in name:
            for component, chunk in zip(("q", "k", "v"), data.chunk(3), strict=True):
                weights[name.replace(".self_attn.qkv.", f".self_attn.{component}.")] = chunk
        else:
            weights[name] = data
    return weights


def _rollout(model, mode, dtype, batch):
    from vllm_omni.diffusion.models.lingbot_world.transformer import LingBotTransformerCache
    from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
    from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

    device = next(model.parameters()).device
    state = None
    if mode == "paged":
        kv = ARDiffusionKVCache(
            ARDiffusionKVConfig(enable=True, chunk_size=_TOKENS_PER_FRAME, window_chunks=5, sink_chunks=1),
            num_layers=_LAYERS,
            num_kv_heads=model.blocks[0].self_attn.num_sp_heads,
            head_size=_HEAD_DIM,
            dtype=dtype,
            block_size=_TOKENS_PER_FRAME,
            max_model_len=4096,
            available_bytes=1 << 27,
            kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
            session_capacity=1,
            frames_per_block=_FRAMES,
            max_scratch_tokens_per_branch=_FRAMES * _TOKENS_PER_FRAME,
            device=device,
        )
        state = ARDiffusionKVState(kv, "numeric", {"main": kv.begin_request("numeric")}, num_layers=_LAYERS)
        cache = LingBotTransformerCache(self_attention=[], cross_attention=[None] * _LAYERS)
    else:
        cache = model.allocate_cache(
            batch_size=batch, latent_height=_SIDE, latent_width=_SIDE, device=device, dtype=dtype
        )
    generator = torch.Generator().manual_seed(123)
    text = torch.randn(batch, 5, 8, generator=generator).to(device, dtype)
    outputs = []
    try:
        for start in range(0, 12, _FRAMES):
            latent = torch.randn(batch, 4, _FRAMES, _SIDE, _SIDE, generator=generator).to(device, dtype)
            camera = torch.randn(batch, 384, _FRAMES, _SIDE, _SIDE, generator=generator).to(device, dtype)
            for commit in (False, False, True):
                if state is not None:
                    cache.self_attention = state.get_kv_caches(
                        "main", seq_len=_FRAMES * _TOKENS_PER_FRAME, commit_current=commit
                    )
                output = model(
                    latent,
                    torch.full((batch,), 500.0, device=device),
                    text,
                    camera,
                    cache=cache,
                    start_frame=start,
                    update_cache=commit,
                )
                outputs.append(output.float().cpu())
                if state is not None:
                    state.commit_paged_context("main")
            if state is None:
                for layer in cache.self_attention:
                    assert layer.absolute_end == (start + _FRAMES) * _TOKENS_PER_FRAME
                    assert layer.last_start == start * _TOKENS_PER_FRAME
                    assert layer.end == min((start + _FRAMES) * _TOKENS_PER_FRAME, 6 * _TOKENS_PER_FRAME)
                # Replacing the current block must not advance the cache cursor.
                model(
                    latent,
                    torch.full((batch,), 500.0, device=device),
                    text,
                    camera,
                    cache=cache,
                    start_frame=start,
                    update_cache=True,
                )
                assert cache.self_attention[0].absolute_end == (start + _FRAMES) * _TOKENS_PER_FRAME
        cross = [
            (layer.key.detach().float().cpu(), layer.value.detach().float().cpu()) for layer in cache.cross_attention
        ]
        for layer in cache.cross_attention:
            assert layer.key.is_contiguous() and layer.value.is_contiguous()
            assert layer.key.untyped_storage().nbytes() == layer.key.numel() * layer.key.element_size()
        return torch.stack(outputs), cross
    finally:
        if state is not None:
            state.close()


def _worker(rank, world_size, sp_size, tp_size, mode, dtype, batch, rendezvous):
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        destroy_model_parallel,
        get_sp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.registry import _apply_sequence_parallel_if_enabled
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device(torch.device("cuda", rank))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", world_size=world_size, rank=rank, timeout=timedelta(seconds=90)
    )
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, backend="nccl")
    try:
        with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
            for baseline in (True, False):
                sp, tp = (1, 1) if baseline else (sp_size, tp_size)
                initialize_model_parallel(
                    data_parallel_size=world_size // (sp * tp),
                    sequence_parallel_size=sp,
                    ulysses_degree=sp,
                    tensor_parallel_size=tp,
                )
                config = OmniDiffusionConfig(
                    model=str(Path(rendezvous).parent),
                    dtype=dtype,
                    enforce_eager=True,
                    parallel_config=DiffusionParallelConfig(
                        data_parallel_size=world_size // (sp * tp),
                        sequence_parallel_size=sp,
                        ulysses_degree=sp,
                        tensor_parallel_size=tp,
                    ),
                    diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
                )
                with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
                    model = _model(dtype)
                    if baseline:
                        weights = _checkpoint(model)
                    else:
                        model.load_weights(iter(weights.items()))
                        # Exact FP32 sentinels independently pin scatter/gather axes and head ownership.
                        group = get_sp_group()
                        heads = _HEADS // tp
                        full = torch.arange(12 * heads * 2, device="cuda", dtype=torch.float32).reshape(1, 12, heads, 2)
                        full += get_tensor_model_parallel_rank() * 10000
                        local = full.chunk(sp, dim=1)[group.ulysses_rank].contiguous()
                        exchanged = SeqAllToAll4D.apply(group.ulysses_group, local, 2, 1, False)
                        torch.testing.assert_close(exchanged, full.chunk(sp, dim=2)[group.ulysses_rank], rtol=0, atol=0)
                        restored = SeqAllToAll4D.apply(group.ulysses_group, exchanged, 1, 2, False)
                        torch.testing.assert_close(restored, local, rtol=0, atol=0)
                    _apply_sequence_parallel_if_enabled(SimpleNamespace(transformer=model), config)
                    result, cross = _rollout(model, mode, dtype, batch)
                    if baseline:
                        expected, expected_cross = result, cross
                    else:
                        assert expected.abs().max() > 1e-3
                        bound = 1e-5 if dtype == torch.float32 else 1e-2
                        error = (result - expected).double().norm() / expected.double().norm()
                        assert error <= bound, f"rank={rank}, TP{tp} SP{sp}: relative L2 {error.item():.3g} > {bound}"
                        local_heads = _HEADS // tp // sp
                        first = (
                            get_tensor_model_parallel_rank() * (_HEADS // tp)
                            + get_sp_group().ulysses_rank * local_heads
                        )
                        for (k, v), (ek, ev) in zip(cross, expected_cross, strict=True):
                            tolerance = 1e-5 if dtype == torch.float32 else 2e-2
                            torch.testing.assert_close(
                                k, ek[:, :, first : first + local_heads], rtol=tolerance, atol=tolerance
                            )
                            torch.testing.assert_close(
                                v, ev[:, :, first : first + local_heads], rtol=tolerance, atol=tolerance
                            )
                        if rank == 0:
                            print(f"{mode} {dtype} TP{tp} SP{sp}: relative L2={error.item():.3g}", flush=True)
                    del model
                torch.distributed.barrier()
                destroy_model_parallel()
    finally:
        destroy_distributed_env()


def _run(tmp_path: Path, sp, tp, mode, dtype, batch):
    if torch.accelerator.device_count() < sp * tp:
        pytest.skip(f"requires {sp * tp} GPUs")
    torch.multiprocessing.spawn(
        _worker, args=(sp * tp, sp, tp, mode, dtype, batch, str(tmp_path / "rendezvous")), nprocs=sp * tp
    )


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize(("mode", "dtype", "batch"), _CASES)
def test_sp2_matches_sp1(tmp_path, mode, dtype, batch):
    _run(tmp_path, 2, 1, mode, dtype, batch)


@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize(("mode", "dtype", "batch"), _CASES)
def test_sp4_matches_sp1(tmp_path, mode, dtype, batch):
    _run(tmp_path, 4, 1, mode, dtype, batch)


@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize(("mode", "dtype", "batch"), _CASES)
def test_tp2_sp2_matches_sp1(tmp_path, mode, dtype, batch):
    _run(tmp_path, 2, 2, mode, dtype, batch)

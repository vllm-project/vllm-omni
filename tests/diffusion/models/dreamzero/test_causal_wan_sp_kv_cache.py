# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import socket
import tempfile

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
from vllm_omni.diffusion.models.dreamzero.causal_wan_model import CausalWanModel
from vllm_omni.platforms import current_omni_platform


pytestmark = [pytest.mark.diffusion, pytest.mark.parallel]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _set_dist_env(*, rank: int, world_size: int, master_port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"


def _run_cached_attention_case(
    local_rank: int,
    world_size: int,
    master_port: int,
    input_file: str,
    output_file: str,
    num_heads: int,
    head_size: int,
    ulysses_degree: int,
    use_joint: bool = False,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    _set_dist_env(rank=local_rank, world_size=world_size, master_port=master_port)
    init_distributed_environment(world_size=world_size, rank=local_rank)
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        cfg_parallel_size=1,
        ulysses_mode="strict",
    )
    od_config = OmniDiffusionConfig(model="test", dtype=torch.float32, parallel_config=parallel_config)

    try:
        with set_forward_context(omni_diffusion_config=od_config), set_current_diffusion_config(od_config):
            attn = Attention(
                num_heads=num_heads,
                head_size=head_size,
                causal=False,
                softmax_scale=1.0 / (head_size**0.5),
            ).to(device=device, dtype=torch.float32)

            with np.load(input_file, allow_pickle=False) as payload:
                q1_full = torch.from_numpy(payload["q1"]).to(device=device)
                k1_full = torch.from_numpy(payload["k1"]).to(device=device)
                v1_full = torch.from_numpy(payload["v1"]).to(device=device)
                q2_full = torch.from_numpy(payload["q2"]).to(device=device)
                k2_full = torch.from_numpy(payload["k2"]).to(device=device)
                v2_full = torch.from_numpy(payload["v2"]).to(device=device)
                if use_joint:
                    jq1 = torch.from_numpy(payload["jq1"]).to(device=device)
                    jk1 = torch.from_numpy(payload["jk1"]).to(device=device)
                    jv1 = torch.from_numpy(payload["jv1"]).to(device=device)
                    jq2 = torch.from_numpy(payload["jq2"]).to(device=device)
                    jk2 = torch.from_numpy(payload["jk2"]).to(device=device)
                    jv2 = torch.from_numpy(payload["jv2"]).to(device=device)
                else:
                    jq1 = jk1 = jv1 = jq2 = jk2 = jv2 = None

            if world_size == 1:
                q1, k1, v1 = q1_full, k1_full, v1_full
                q2, k2, v2 = q2_full, k2_full, v2_full
            else:
                q1 = torch.tensor_split(q1_full, world_size, dim=1)[local_rank].contiguous()
                k1 = torch.tensor_split(k1_full, world_size, dim=1)[local_rank].contiguous()
                v1 = torch.tensor_split(v1_full, world_size, dim=1)[local_rank].contiguous()
                q2 = torch.tensor_split(q2_full, world_size, dim=1)[local_rank].contiguous()
                k2 = torch.tensor_split(k2_full, world_size, dim=1)[local_rank].contiguous()
                v2 = torch.tensor_split(v2_full, world_size, dim=1)[local_rank].contiguous()
                get_forward_context()._sp_shard_depth = 1

            cache = None
            metadata_1 = (
                AttentionMetadata(joint_query=jq1, joint_key=jk1, joint_value=jv1, joint_strategy="rear")
                if use_joint
                else None
            )
            metadata_2 = (
                AttentionMetadata(joint_query=jq2, joint_key=jk2, joint_value=jv2, joint_strategy="rear")
                if use_joint
                else None
            )
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                _, cache = attn.forward_with_kv_cache(
                    q1,
                    k1,
                    v1,
                    cache,
                    max_cache_len=16,
                    attn_metadata=metadata_1,
                )
                out_local, cache = attn.forward_with_kv_cache(
                    q2,
                    k2,
                    v2,
                    cache,
                    max_cache_len=16,
                    attn_metadata=metadata_2,
                )

            local_cache_shape = torch.tensor(cache.shape, device=device, dtype=torch.int64)
            gathered_cache_shapes = [torch.empty_like(local_cache_shape) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_cache_shapes, local_cache_shape)

            if world_size == 1:
                out_full = out_local
                cache_full = cache
                cache_shapes = torch.stack(gathered_cache_shapes).cpu().numpy()
            else:
                if use_joint:
                    joint_len = jq2.shape[1]
                    video_out_local = out_local[:, : q2.shape[1]]
                    joint_out_local = out_local[:, q2.shape[1] :]
                    if joint_out_local.shape[1] != joint_len:
                        raise AssertionError(
                            f"Expected joint output length {joint_len}, got {joint_out_local.shape[1]}."
                        )
                    gathered_out = [torch.empty_like(video_out_local) for _ in range(world_size)]
                    torch.distributed.all_gather(gathered_out, video_out_local)
                else:
                    joint_out_local = None
                    gathered_out = [torch.empty_like(out_local) for _ in range(world_size)]
                    torch.distributed.all_gather(gathered_out, out_local)

                gathered_cache = [torch.empty_like(cache) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_cache, cache)

                if local_rank == 0:
                    out_full = torch.cat(gathered_out, dim=1)
                    if use_joint:
                        out_full = torch.cat([out_full, joint_out_local], dim=1)
                    cache_full = torch.cat(gathered_cache, dim=3)
                    cache_shapes = torch.stack(gathered_cache_shapes).cpu().numpy()
                else:
                    out_full = None
                    cache_full = None
                    cache_shapes = None

            if local_rank == 0:
                np.savez(
                    output_file,
                    out=out_full.detach().cpu().numpy(),
                    cache=cache_full.detach().cpu().numpy(),
                    cache_shapes=cache_shapes,
                )
    finally:
        destroy_distributed_env()


def _run_tiny_causal_wan_case(
    local_rank: int,
    world_size: int,
    master_port: int,
    input_file: str,
    output_file: str,
    ulysses_degree: int,
    use_prefill: bool = False,
    use_qk_norm: bool = False,
    dtype_name: str = "float32",
    num_layers: int = 1,
    num_frame_per_block: int = 1,
    num_heads: int = 4,
    action_horizon: int = 2,
    model_type: str = "t2v",
    frame_seqlen: int = 4,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    _set_dist_env(rank=local_rank, world_size=world_size, master_port=master_port)
    init_distributed_environment(world_size=world_size, rank=local_rank)
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        cfg_parallel_size=1,
        ulysses_mode="strict",
    )
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    od_config = OmniDiffusionConfig(model="test", dtype=dtype, parallel_config=parallel_config)

    try:
        from vllm.config import DeviceConfig, VllmConfig

        vllm_config = VllmConfig(device_config=DeviceConfig(device=current_omni_platform.device_type))
        with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config), set_current_diffusion_config(
            od_config
        ):
            torch.manual_seed(1234)
            current_omni_platform.manual_seed(1234)
            in_dim = 4 if model_type == "i2v" else 2
            model = CausalWanModel(
                model_type=model_type,
                patch_size=(1, 1, 1),
                frame_seqlen=frame_seqlen,
                text_len=8,
                in_dim=in_dim,
                dim=32 if num_heads == 4 else 80,
                ffn_dim=64 if num_heads == 4 else 160,
                freq_dim=8,
                text_dim=16,
                out_dim=2,
                num_heads=num_heads,
                num_layers=num_layers,
                hidden_size=16,
                action_dim=3,
                max_state_dim=5,
                qk_norm=use_qk_norm,
                cross_attn_norm=use_qk_norm,
                num_frame_per_block=num_frame_per_block,
                num_action_per_block=action_horizon,
                num_state_per_block=1,
            ).to(device=device, dtype=dtype)
            model.eval()

            if world_size > 1:
                apply_sequence_parallel(
                    model,
                    SequenceParallelConfig(ulysses_degree=ulysses_degree, ring_degree=1),
                    model._sp_plan,
                )
                get_forward_context().sp_plan_hooks_applied = True

            with np.load(input_file, allow_pickle=False) as payload:
                x = torch.from_numpy(payload["x"]).to(device=device, dtype=dtype)
                timestep = torch.from_numpy(payload["timestep"]).to(device=device)
                context = torch.from_numpy(payload["context"]).to(device=device, dtype=dtype)
                action = torch.from_numpy(payload["action"]).to(device=device, dtype=dtype)
                timestep_action = torch.from_numpy(payload["timestep_action"]).to(device=device)
                state = torch.from_numpy(payload["state"]).to(device=device, dtype=dtype)
                y = torch.from_numpy(payload["y"]).to(device=device, dtype=dtype) if "y" in payload.files else None
                clip_feature = (
                    torch.from_numpy(payload["clip_feature"]).to(device=device, dtype=dtype)
                    if "clip_feature" in payload.files
                    else None
                )

            seq_len = int(x.shape[2] * x.shape[3] * x.shape[4])
            batch_size = x.shape[0]
            head_dim = model.dim // model.num_heads
            num_cache_heads = model.kv_cache_num_heads(ulysses_degree=ulysses_degree)
            kv_cache = [
                torch.empty(
                    2,
                    batch_size,
                    0,
                    num_cache_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(model.num_layers)
            ]

            with torch.no_grad(), torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                if use_prefill:
                    _, _, kv_cache = model(
                        x=x,
                        timestep=timestep,
                        context=context,
                        seq_len=seq_len,
                        kv_cache=kv_cache,
                        crossattn_cache=[],
                        current_start_frame=0,
                        y=y,
                        clip_feature=clip_feature,
                    )

                video_out, action_out, updated_cache = model(
                    x=x,
                    timestep=timestep,
                    context=context,
                    seq_len=seq_len,
                    kv_cache=kv_cache,
                    crossattn_cache=[],
                    current_start_frame=1 if use_prefill else 0,
                    y=y,
                    clip_feature=clip_feature,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                )

            cache = updated_cache[0]
            if world_size > 1:
                gathered_cache = [torch.empty_like(cache) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_cache, cache)
                if local_rank == 0:
                    cache_out = torch.cat(gathered_cache, dim=3)
                else:
                    cache_out = None
            else:
                cache_out = cache

            if local_rank == 0:
                np.savez(
                    output_file,
                    video=video_out.detach().float().cpu().numpy(),
                    action=action_out.detach().float().cpu().numpy(),
                    cache=cache_out.detach().float().cpu().numpy(),
                )
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
def test_dreamzero_cached_attention_ulysses_keeps_kv_cache_head_sharded(world_size: int) -> None:
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")

    batch_size = 1
    seq_len_1 = 4
    seq_len_2 = 4
    num_heads = 4
    head_size = 8

    base_port = _find_free_port()
    sp_port = _find_free_port()
    while sp_port == base_port:
        sp_port = _find_free_port()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(0)
        q1 = torch.randn(batch_size, seq_len_1, num_heads, head_size, dtype=torch.float32)
        k1 = torch.randn(batch_size, seq_len_1, num_heads, head_size, dtype=torch.float32)
        v1 = torch.randn(batch_size, seq_len_1, num_heads, head_size, dtype=torch.float32)
        q2 = torch.randn(batch_size, seq_len_2, num_heads, head_size, dtype=torch.float32)
        k2 = torch.randn(batch_size, seq_len_2, num_heads, head_size, dtype=torch.float32)
        v2 = torch.randn(batch_size, seq_len_2, num_heads, head_size, dtype=torch.float32)
        np.savez(
            input_file,
            q1=q1.numpy(),
            k1=k1.numpy(),
            v1=v1.numpy(),
            q2=q2.numpy(),
            k2=k2.numpy(),
            v2=v2.numpy(),
        )

        torch.multiprocessing.spawn(
            _run_cached_attention_case,
            args=(1, base_port, input_file, baseline_file, num_heads, head_size, 1),
            nprocs=1,
        )
        torch.multiprocessing.spawn(
            _run_cached_attention_case,
            args=(world_size, sp_port, input_file, sp_file, num_heads, head_size, world_size),
            nprocs=world_size,
        )

        with np.load(baseline_file, allow_pickle=False) as baseline_payload:
            baseline_out = torch.from_numpy(baseline_payload["out"])
            baseline_cache = torch.from_numpy(baseline_payload["cache"])
        with np.load(sp_file, allow_pickle=False) as sp_payload:
            sp_out = torch.from_numpy(sp_payload["out"])
            sp_cache = torch.from_numpy(sp_payload["cache"])
            sp_cache_shapes = sp_payload["cache_shapes"]

        expected_cache_shape = (2, batch_size, seq_len_1 + seq_len_2, num_heads // world_size, head_size)
        for rank in range(world_size):
            assert tuple(sp_cache_shapes[rank]) == expected_cache_shape
        assert sp_out.shape == baseline_out.shape
        assert sp_cache.shape == baseline_cache.shape
        torch.testing.assert_close(sp_out, baseline_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_cache, baseline_cache, atol=1e-5, rtol=1e-5)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass


@pytest.mark.parametrize("world_size", [2, 4])
def test_dreamzero_cached_attention_ulysses_keeps_joint_tokens_out_of_kv_cache(world_size: int) -> None:
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")

    batch_size = 1
    seq_len_1 = 4
    seq_len_2 = 4
    joint_len = 3
    num_heads = 4
    head_size = 8

    base_port = _find_free_port()
    sp_port = _find_free_port()
    while sp_port == base_port:
        sp_port = _find_free_port()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(1)
        q1 = torch.randn(batch_size, seq_len_1, num_heads, head_size, dtype=torch.float32)
        k1 = torch.randn(batch_size, seq_len_1, num_heads, head_size, dtype=torch.float32)
        v1 = torch.randn(batch_size, seq_len_1, num_heads, head_size, dtype=torch.float32)
        q2 = torch.randn(batch_size, seq_len_2, num_heads, head_size, dtype=torch.float32)
        k2 = torch.randn(batch_size, seq_len_2, num_heads, head_size, dtype=torch.float32)
        v2 = torch.randn(batch_size, seq_len_2, num_heads, head_size, dtype=torch.float32)
        jq1 = torch.randn(batch_size, joint_len, num_heads, head_size, dtype=torch.float32)
        jk1 = torch.randn(batch_size, joint_len, num_heads, head_size, dtype=torch.float32)
        jv1 = torch.randn(batch_size, joint_len, num_heads, head_size, dtype=torch.float32)
        jq2 = torch.randn(batch_size, joint_len, num_heads, head_size, dtype=torch.float32)
        jk2 = torch.randn(batch_size, joint_len, num_heads, head_size, dtype=torch.float32)
        jv2 = torch.randn(batch_size, joint_len, num_heads, head_size, dtype=torch.float32)
        np.savez(
            input_file,
            q1=q1.numpy(),
            k1=k1.numpy(),
            v1=v1.numpy(),
            q2=q2.numpy(),
            k2=k2.numpy(),
            v2=v2.numpy(),
            jq1=jq1.numpy(),
            jk1=jk1.numpy(),
            jv1=jv1.numpy(),
            jq2=jq2.numpy(),
            jk2=jk2.numpy(),
            jv2=jv2.numpy(),
        )

        torch.multiprocessing.spawn(
            _run_cached_attention_case,
            args=(1, base_port, input_file, baseline_file, num_heads, head_size, 1, True),
            nprocs=1,
        )
        torch.multiprocessing.spawn(
            _run_cached_attention_case,
            args=(world_size, sp_port, input_file, sp_file, num_heads, head_size, world_size, True),
            nprocs=world_size,
        )

        with np.load(baseline_file, allow_pickle=False) as baseline_payload:
            baseline_out = torch.from_numpy(baseline_payload["out"])
            baseline_cache = torch.from_numpy(baseline_payload["cache"])
        with np.load(sp_file, allow_pickle=False) as sp_payload:
            sp_out = torch.from_numpy(sp_payload["out"])
            sp_cache = torch.from_numpy(sp_payload["cache"])
            sp_cache_shapes = sp_payload["cache_shapes"]

        expected_cache_shape = (2, batch_size, seq_len_1 + seq_len_2, num_heads // world_size, head_size)
        for rank in range(world_size):
            assert tuple(sp_cache_shapes[rank]) == expected_cache_shape
        assert sp_out.shape == baseline_out.shape
        assert sp_cache.shape == baseline_cache.shape
        torch.testing.assert_close(sp_out, baseline_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_cache, baseline_cache, atol=1e-5, rtol=1e-5)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass


def test_tiny_causal_wan_model_ulysses_matches_single_rank_with_actions() -> None:
    world_size = 2
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")

    base_port = _find_free_port()
    sp_port = _find_free_port()
    while sp_port == base_port:
        sp_port = _find_free_port()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(5)
        x = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32)
        timestep = torch.zeros(1, 2, dtype=torch.float32)
        context = torch.randn(1, 8, 16, dtype=torch.float32)
        action = torch.randn(1, 2, 3, dtype=torch.float32)
        timestep_action = torch.zeros(1, 2, dtype=torch.float32)
        state = torch.randn(1, 1, 5, dtype=torch.float32)
        np.savez(
            input_file,
            x=x.numpy(),
            timestep=timestep.numpy(),
            context=context.numpy(),
            action=action.numpy(),
            timestep_action=timestep_action.numpy(),
            state=state.numpy(),
        )

        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(1, base_port, input_file, baseline_file, 1),
            nprocs=1,
        )
        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(world_size, sp_port, input_file, sp_file, world_size),
            nprocs=world_size,
        )

        with np.load(baseline_file, allow_pickle=False) as baseline_payload:
            baseline_video = torch.from_numpy(baseline_payload["video"])
            baseline_action = torch.from_numpy(baseline_payload["action"])
            baseline_cache = torch.from_numpy(baseline_payload["cache"])
        with np.load(sp_file, allow_pickle=False) as sp_payload:
            sp_video = torch.from_numpy(sp_payload["video"])
            sp_action = torch.from_numpy(sp_payload["action"])
            sp_cache = torch.from_numpy(sp_payload["cache"])

        assert sp_video.shape == baseline_video.shape
        assert sp_action.shape == baseline_action.shape
        assert sp_cache.shape == baseline_cache.shape
        torch.testing.assert_close(sp_video, baseline_video, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_action, baseline_action, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_cache, baseline_cache, atol=1e-5, rtol=1e-5)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass


def test_tiny_causal_wan_model_ulysses_matches_single_rank_after_prefill() -> None:
    world_size = 2
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")

    base_port = _find_free_port()
    sp_port = _find_free_port()
    while sp_port == base_port:
        sp_port = _find_free_port()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(6)
        x = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32)
        timestep = torch.zeros(1, 2, dtype=torch.float32)
        context = torch.randn(1, 8, 16, dtype=torch.float32)
        action = torch.randn(1, 2, 3, dtype=torch.float32)
        timestep_action = torch.zeros(1, 2, dtype=torch.float32)
        state = torch.randn(1, 1, 5, dtype=torch.float32)
        np.savez(
            input_file,
            x=x.numpy(),
            timestep=timestep.numpy(),
            context=context.numpy(),
            action=action.numpy(),
            timestep_action=timestep_action.numpy(),
            state=state.numpy(),
        )

        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(1, base_port, input_file, baseline_file, 1, True),
            nprocs=1,
        )
        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(world_size, sp_port, input_file, sp_file, world_size, True),
            nprocs=world_size,
        )

        with np.load(baseline_file, allow_pickle=False) as baseline_payload:
            baseline_video = torch.from_numpy(baseline_payload["video"])
            baseline_action = torch.from_numpy(baseline_payload["action"])
            baseline_cache = torch.from_numpy(baseline_payload["cache"])
        with np.load(sp_file, allow_pickle=False) as sp_payload:
            sp_video = torch.from_numpy(sp_payload["video"])
            sp_action = torch.from_numpy(sp_payload["action"])
            sp_cache = torch.from_numpy(sp_payload["cache"])

        assert sp_video.shape == baseline_video.shape
        assert sp_action.shape == baseline_action.shape
        assert sp_cache.shape == baseline_cache.shape
        torch.testing.assert_close(sp_video, baseline_video, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_action, baseline_action, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_cache, baseline_cache, atol=1e-5, rtol=1e-5)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass


@pytest.mark.parametrize("world_size", [2, 4])
def test_tiny_causal_wan_model_ulysses_matches_single_rank_after_prefill_with_qk_norm(world_size: int) -> None:
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")

    base_port = _find_free_port()
    sp_port = _find_free_port()
    while sp_port == base_port:
        sp_port = _find_free_port()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(7)
        x = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32)
        timestep = torch.zeros(1, 2, dtype=torch.float32)
        context = torch.randn(1, 8, 16, dtype=torch.float32)
        action = torch.randn(1, 2, 3, dtype=torch.float32)
        timestep_action = torch.zeros(1, 2, dtype=torch.float32)
        state = torch.randn(1, 1, 5, dtype=torch.float32)
        np.savez(
            input_file,
            x=x.numpy(),
            timestep=timestep.numpy(),
            context=context.numpy(),
            action=action.numpy(),
            timestep_action=timestep_action.numpy(),
            state=state.numpy(),
        )

        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(1, base_port, input_file, baseline_file, 1, True, True, "bfloat16", 8, 2),
            nprocs=1,
        )
        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(world_size, sp_port, input_file, sp_file, world_size, True, True, "bfloat16", 8, 2),
            nprocs=world_size,
        )

        with np.load(baseline_file, allow_pickle=False) as baseline_payload:
            baseline_video = torch.from_numpy(baseline_payload["video"])
            baseline_action = torch.from_numpy(baseline_payload["action"])
            baseline_cache = torch.from_numpy(baseline_payload["cache"])
        with np.load(sp_file, allow_pickle=False) as sp_payload:
            sp_video = torch.from_numpy(sp_payload["video"])
            sp_action = torch.from_numpy(sp_payload["action"])
            sp_cache = torch.from_numpy(sp_payload["cache"])

        assert sp_video.shape == baseline_video.shape
        assert sp_action.shape == baseline_action.shape
        assert sp_cache.shape == baseline_cache.shape
        torch.testing.assert_close(sp_video, baseline_video, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_action, baseline_action, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_cache, baseline_cache, atol=1e-5, rtol=1e-5)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass


@pytest.mark.parametrize("world_size", [2, 4])
def test_tiny_causal_wan_model_ulysses_matches_single_rank_with_realistic_action_horizon(
    world_size: int,
) -> None:
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")

    base_port = _find_free_port()
    sp_port = _find_free_port()
    while sp_port == base_port:
        sp_port = _find_free_port()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(8)
        x = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32)
        timestep = torch.zeros(1, 2, dtype=torch.float32)
        context = torch.randn(1, 8, 16, dtype=torch.float32)
        action = torch.randn(1, 24, 3, dtype=torch.float32)
        timestep_action = torch.zeros(1, 24, dtype=torch.float32)
        state = torch.randn(1, 1, 5, dtype=torch.float32)
        np.savez(
            input_file,
            x=x.numpy(),
            timestep=timestep.numpy(),
            context=context.numpy(),
            action=action.numpy(),
            timestep_action=timestep_action.numpy(),
            state=state.numpy(),
        )

        common_args = (True, True, "bfloat16", 4, 2, 40, 24)
        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(1, base_port, input_file, baseline_file, 1, *common_args),
            nprocs=1,
        )
        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(world_size, sp_port, input_file, sp_file, world_size, *common_args),
            nprocs=world_size,
        )

        with np.load(baseline_file, allow_pickle=False) as baseline_payload:
            baseline_video = torch.from_numpy(baseline_payload["video"])
            baseline_action = torch.from_numpy(baseline_payload["action"])
            baseline_cache = torch.from_numpy(baseline_payload["cache"])
        with np.load(sp_file, allow_pickle=False) as sp_payload:
            sp_video = torch.from_numpy(sp_payload["video"])
            sp_action = torch.from_numpy(sp_payload["action"])
            sp_cache = torch.from_numpy(sp_payload["cache"])

        assert sp_video.shape == baseline_video.shape
        assert sp_action.shape == baseline_action.shape
        assert sp_cache.shape == baseline_cache.shape
        torch.testing.assert_close(sp_video, baseline_video, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_action, baseline_action, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_cache, baseline_cache, atol=1e-5, rtol=1e-5)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass


@pytest.mark.parametrize("world_size", [2, 4])
def test_tiny_causal_wan_i2v_ulysses_matches_single_rank_with_clip_and_y(
    world_size: int,
) -> None:
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")

    base_port = _find_free_port()
    sp_port = _find_free_port()
    while sp_port == base_port:
        sp_port = _find_free_port()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_in:
        input_file = f_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_base:
        baseline_file = f_base.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f_sp:
        sp_file = f_sp.name

    try:
        torch.manual_seed(9)
        x = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32)
        y = torch.randn(1, 2, 2, 2, 2, dtype=torch.float32)
        timestep = torch.zeros(1, 2, dtype=torch.float32)
        context = torch.randn(1, 8, 16, dtype=torch.float32)
        clip_feature = torch.randn(1, 257, 1280, dtype=torch.float32)
        action = torch.randn(1, 24, 3, dtype=torch.float32)
        timestep_action = torch.zeros(1, 24, dtype=torch.float32)
        state = torch.randn(1, 1, 5, dtype=torch.float32)
        np.savez(
            input_file,
            x=x.numpy(),
            y=y.numpy(),
            timestep=timestep.numpy(),
            context=context.numpy(),
            clip_feature=clip_feature.numpy(),
            action=action.numpy(),
            timestep_action=timestep_action.numpy(),
            state=state.numpy(),
        )

        common_args = (True, True, "bfloat16", 4, 2, 40, 24, "i2v")
        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(1, base_port, input_file, baseline_file, 1, *common_args),
            nprocs=1,
        )
        torch.multiprocessing.spawn(
            _run_tiny_causal_wan_case,
            args=(world_size, sp_port, input_file, sp_file, world_size, *common_args),
            nprocs=world_size,
        )

        with np.load(baseline_file, allow_pickle=False) as baseline_payload:
            baseline_video = torch.from_numpy(baseline_payload["video"])
            baseline_action = torch.from_numpy(baseline_payload["action"])
            baseline_cache = torch.from_numpy(baseline_payload["cache"])
        with np.load(sp_file, allow_pickle=False) as sp_payload:
            sp_video = torch.from_numpy(sp_payload["video"])
            sp_action = torch.from_numpy(sp_payload["action"])
            sp_cache = torch.from_numpy(sp_payload["cache"])

        assert sp_video.shape == baseline_video.shape
        assert sp_action.shape == baseline_action.shape
        assert sp_cache.shape == baseline_cache.shape
        torch.testing.assert_close(sp_video, baseline_video, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_action, baseline_action, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sp_cache, baseline_cache, atol=1e-5, rtol=1e-5)
    finally:
        for path in (input_file, baseline_file, sp_file):
            try:
                os.remove(path)
            except OSError:
                pass

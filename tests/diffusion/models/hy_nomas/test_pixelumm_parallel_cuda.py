# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.config import set_current_vllm_config

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import build_decoder
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "tencent/Hy-Nomas-Diffusers"
REVISION = "a1ae04d11ae4ea176d1f76de634c115944f9882d"


def _load_decoder(root: Path, device: torch.device):
    root = root / "decoder"
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    model = build_decoder(config, device)

    def weights():
        for name in sorted(set(index.values())):
            with safe_open(root / name, framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    yield key, shard.get_tensor(key)

    model.load_weights(weights())
    return model


def _run_stream(model, latent):
    session = model.create_streaming_decode_session()
    parts = []
    positions = []
    for chunk, frames, final in ((latent[:, :, :7], 28, False), (latent[:, :, 7:], 49, True)):
        session.final_segment = final
        parts.append(session.push(chunk, pixel_frames=frames))
        positions.append(session.committed_pixel_frames)
    parts.append(session.finish())
    positions.append(session.committed_pixel_frames)
    assert session.finished and positions[-1] == 77
    return parts, positions


def _decode_worker(rank: int, rendezvous: str, root: str):
    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, device_id=device, timeout=timedelta(minutes=10)
    )
    world_ready = False
    try:
        parallel = DiffusionParallelConfig(sequence_parallel_size=2, ulysses_degree=2)
        config = OmniDiffusionConfig(model=root, dtype=torch.bfloat16, parallel_config=parallel, enforce_eager=True)
        vllm_config = create_diffusion_vllm_config(device, config)
        with (
            set_forward_context(vllm_config=vllm_config, omni_diffusion_config=config),
            set_current_vllm_config(vllm_config),
            set_current_diffusion_config(config),
        ):
            init_distributed_environment(world_size=2, rank=rank, local_rank=rank, backend="nccl")
            world_ready = True
            initialize_model_parallel(sequence_parallel_size=2, ulysses_degree=2, backend="nccl")
            model = _load_decoder(Path(root), device)
            generator = torch.Generator().manual_seed(42)
            latent = torch.randn((1, 32, 20, 8, 8), generator=generator, dtype=torch.bfloat16).to(device)
            calls = []
            hook = model.decoder.register_forward_hook(lambda *_: calls.append(1))
            with (
                torch.inference_mode(),
                sdpa_kernel(
                    [
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.MATH,
                    ]
                ),
            ):
                expected = model.decode(latent, pixel_frames=77).sample
                assert len(calls) == 3
                calls.clear()
                expected_parts, expected_positions = _run_stream(model, latent)
                dist.barrier()
                model.configure_distributed_decode(get_sp_group().ulysses_group)
                calls.clear()
                actual = model.decode(latent, pixel_frames=77).sample
                assert len(calls) == (2 if rank == 0 else 1)
                actual_parts, actual_positions = _run_stream(model, latent)
                dist.barrier()
            hook.remove()
            assert actual_positions == expected_positions
            if rank == 0:
                assert actual.shape[2] == 77 and sum(part.shape[2] for part in actual_parts if part.numel()) == 77
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert len(actual_parts) == len(expected_parts)
                for part, expected_part in zip(actual_parts, expected_parts):
                    torch.testing.assert_close(part, expected_part, rtol=0, atol=0)
            else:
                assert actual.numel() == 0
                assert all(part.numel() == 0 for part in actual_parts)
    finally:
        if world_ready:
            destroy_distributed_env()
        elif dist.is_initialized():
            dist.destroy_process_group()


@hardware_test(res={"cuda": ["H200", "B200"]}, num_cards=2)
def test_real_decoder_window_parallel_matches_serial(tmp_path):
    local_model = os.environ.get("HY_NOMAS_MODEL")
    root = (
        Path(local_model)
        if local_model
        else Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=["decoder/*"]))
    )
    mp.spawn(_decode_worker, args=((tmp_path / "rendezvous").as_uri(), str(root)), nprocs=2, join=True)

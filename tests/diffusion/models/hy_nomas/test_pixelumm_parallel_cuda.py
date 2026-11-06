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

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import load_component_weights
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import build_decoder
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

    load_component_weights(model, weights())
    return model


def _run_stream(model, latent):
    session = model.create_streaming_decode_session()
    parts = []
    positions = []
    for chunk, frames, final in ((latent[:, :, :7], 28, False), (latent[:, :, 7:], 29, True)):
        session.final_segment = final
        parts.append(session.push(chunk, pixel_frames=frames))
        positions.append(session.committed_pixel_frames)
    parts.append(session.finish())
    positions.append(session.committed_pixel_frames)
    assert session.finished and positions[-1] == 57
    return parts, positions


def _decode_worker(rank: int, rendezvous: str, root: str):
    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, device_id=device, timeout=timedelta(minutes=10)
    )
    try:
        model = _load_decoder(Path(root), device)
        generator = torch.Generator().manual_seed(42)
        latent = torch.randn((1, 32, 15, 8, 8), generator=generator, dtype=torch.bfloat16).to(device)
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
            expected = model.decode(latent, pixel_frames=57).sample
            assert len(calls) == 2
            calls.clear()
            expected_parts, expected_positions = _run_stream(model, latent)
            dist.barrier()
            model.configure_distributed_decode(dist.group.WORLD)
            calls.clear()
            actual = model.decode(latent, pixel_frames=57).sample
            assert len(calls) == 1
            actual_parts, actual_positions = _run_stream(model, latent)
            dist.barrier()
        hook.remove()
        assert actual_positions == expected_positions
        if rank == 0:
            assert actual.shape[2] == 57 and sum(part.shape[2] for part in actual_parts if part.numel()) == 57
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert len(actual_parts) == len(expected_parts)
            for part, expected_part in zip(actual_parts, expected_parts):
                torch.testing.assert_close(part, expected_part, rtol=0, atol=0)
        else:
            assert actual.numel() == 0
            assert all(part.numel() == 0 for part in actual_parts)
    finally:
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

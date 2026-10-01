# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Multi-GPU numerical-correctness tests for the Ming-Image Qwen VAE decode paths.

Every case is compared against the same VAE's single-rank full (non-tiled)
decode. ``tile`` reports the lossy tile overlap/blend path as-is, while
``spatial_shard_height`` / ``spatial_shard_width`` are expected to be fp-lossless
within the tolerances below.
"""

import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
    DistributedAutoencoderKLQwenImage,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

_QWEN_VAE_MODEL = "inclusionAI/Ming-Image-0.1-Design-Layer"
_QWEN_VAE_SUBFOLDER = "vae"
_WORLD_SIZE = 2
_TOLERANCE_MAX_ABS = 1e-2
_TOLERANCE_MEAN_ABS = 1e-3
_TOLERANCE_PSNR_DB = 50.0

# (batch, z_dim, frames, height, width). The 7-batch case mirrors six decomposed
# layers plus the composite frame; the 16x16 latent case stays below the tile
# threshold and must therefore fall back to full non-tiled decode in ``tile``.
_LATENT_CASES = (
    (1, 16, 1, 128, 128),
    (7, 16, 1, 128, 128),
    (1, 16, 1, 16, 16),
)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return math.inf if mse == 0 else 10 * math.log10(4.0 / mse)


def _record_case(return_dict: dict, case_id: int, mode: str, reference: torch.Tensor, actual: torch.Tensor) -> None:
    diff = (actual.float() - reference.float()).abs()
    return_dict[f"{case_id}:{mode}:max_abs_diff"] = diff.max().item()
    return_dict[f"{case_id}:{mode}:mean_abs_diff"] = diff.mean().item()
    return_dict[f"{case_id}:{mode}:psnr"] = _psnr(actual, reference)
    return_dict[f"{case_id}:{mode}:shape"] = tuple(actual.shape)
    channel_names = ("R", "G", "B", "A")
    channels = actual.shape[1]
    for channel in range(min(channels, len(channel_names))):
        channel_diff = diff[:, channel].abs()
        return_dict[f"{case_id}:{mode}:channel_{channel_names[channel]}_max_abs_diff"] = channel_diff.max().item()
        return_dict[f"{case_id}:{mode}:channel_{channel_names[channel]}_mean_abs_diff"] = channel_diff.mean().item()


def _load_qwen_vae(model: str, dtype: torch.dtype) -> DistributedAutoencoderKLQwenImage:
    local_vae_only = os.path.isdir(model) and not os.path.isdir(os.path.join(model, _QWEN_VAE_SUBFOLDER))
    load_kwargs: dict[str, object] = {"torch_dtype": dtype}
    if not local_vae_only:
        load_kwargs["subfolder"] = _QWEN_VAE_SUBFOLDER
    return DistributedAutoencoderKLQwenImage.from_pretrained(model, **load_kwargs)


def _worker(rank: int, split_dim: str, return_dict: dict, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(device)
    dtype = torch.float32
    backend = current_omni_platform.dist_backend
    init_distributed_environment(world_size=_WORLD_SIZE, rank=rank, local_rank=rank, backend=backend)
    initialize_model_parallel(sequence_parallel_size=_WORLD_SIZE, ulysses_degree=_WORLD_SIZE, backend=backend)

    model = os.environ.get("VLLM_OMNI_TEST_QWEN_VAE", _QWEN_VAE_MODEL)
    try:
        for case_id, (batch, z_dim, frames, height, width) in enumerate(_LATENT_CASES):
            vae = _load_qwen_vae(model, dtype).to(device=device, dtype=dtype).eval()
            generator = torch.Generator(device=device).manual_seed(0)
            latents = torch.randn(
                (batch, z_dim, frames, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )

            with torch.inference_mode():
                # Single-rank full decode: the ground truth for every comparison.
                vae.use_tiling = False
                vae.set_parallel_size(1, mode="tile")
                reference = vae.decode(latents, return_dict=False)[0].float()

                # Tile / tile-parallel decode. For the small latent shape this
                # stays under the tile threshold and therefore runs the same
                # full non-tiled decoder.
                vae.use_tiling = True
                vae.set_parallel_size(_WORLD_SIZE, mode="tile")
                tiled = vae.decode(latents, return_dict=False)[0].float()

                # Spatially-sharded decode along the requested dimension.
                vae.use_tiling = True
                vae.set_parallel_size(_WORLD_SIZE, mode=f"spatial_shard_{split_dim}")
                sharded = vae.decode(latents, return_dict=False)[0].float()

            if rank == 0:
                _record_case(return_dict, case_id, "tile", reference, tiled)
                _record_case(return_dict, case_id, f"spatial_shard_{split_dim}", reference, sharded)

            del vae
            torch.accelerator.empty_cache()
    finally:
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": ["B200"]}, num_cards=_WORLD_SIZE)
@pytest.mark.parametrize("split_dim", ["height", "width"])
def test_qwen_spatial_shard_decode_matches_reference(split_dim: str) -> None:
    manager = mp.get_context("spawn").Manager()
    return_dict = manager.dict()
    master_port = str(29600 + (1 if split_dim == "width" else 0))

    mp.spawn(
        _worker,
        args=(split_dim, return_dict, master_port),
        nprocs=_WORLD_SIZE,
        join=True,
    )

    for case_id, shape in enumerate(_LATENT_CASES):
        assert f"{case_id}:tile:shape" in return_dict, f"rank 0 did not report tile case {shape}"
        spatial_mode = f"spatial_shard_{split_dim}"
        assert f"{case_id}:{spatial_mode}:shape" in return_dict, f"rank 0 did not report {spatial_mode} case {shape}"

        # Tile is intentionally lossy for above-threshold shapes and is not
        # gated; for the small below-threshold shape it is exact (PSNR inf).
        tile_psnr = return_dict[f"{case_id}:tile:psnr"]
        assert tile_psnr >= 0
        assert return_dict[f"{case_id}:tile:shape"] == return_dict[f"{case_id}:{spatial_mode}:shape"]

        print(
            f"case {shape}: tile max={return_dict[f'{case_id}:tile:max_abs_diff']:.6e} "
            f"mean={return_dict[f'{case_id}:tile:mean_abs_diff']:.6e} psnr={tile_psnr:.2f}"
        )
        spatial_max = return_dict[f"{case_id}:{spatial_mode}:max_abs_diff"]
        spatial_mean = return_dict[f"{case_id}:{spatial_mode}:mean_abs_diff"]
        spatial_psnr = return_dict[f"{case_id}:{spatial_mode}:psnr"]
        print(f"case {shape}: {spatial_mode} max={spatial_max:.6e} mean={spatial_mean:.6e} psnr={spatial_psnr:.2f}")
        channel_report = " ".join(
            f"{name}:{return_dict[f'{case_id}:{spatial_mode}:channel_{name}_max_abs_diff']:.3e}/"
            f"{return_dict[f'{case_id}:{spatial_mode}:channel_{name}_mean_abs_diff']:.3e}"
            for name in ("R", "G", "B", "A")
        )
        print(f"case {shape}: {spatial_mode} channels (max/mean abs diff): {channel_report}")
        assert spatial_max <= _TOLERANCE_MAX_ABS, (
            f"{spatial_mode} max_abs_diff {spatial_max} exceeds {_TOLERANCE_MAX_ABS} for {shape}"
        )
        assert spatial_mean <= _TOLERANCE_MEAN_ABS, (
            f"{spatial_mode} mean_abs_diff {spatial_mean} exceeds {_TOLERANCE_MEAN_ABS} for {shape}"
        )
        assert spatial_psnr >= _TOLERANCE_PSNR_DB, (
            f"{spatial_mode} PSNR {spatial_psnr} below {_TOLERANCE_PSNR_DB} for {shape}"
        )
        # The RGBA channel that motivated this work must stay in the same
        # tolerance band as the RGB channels.
        alpha_max = return_dict[f"{case_id}:{spatial_mode}:channel_A_max_abs_diff"]
        rgb_max = max(return_dict[f"{case_id}:{spatial_mode}:channel_{name}_max_abs_diff"] for name in ("R", "G", "B"))
        assert alpha_max <= max(rgb_max, _TOLERANCE_MAX_ABS)

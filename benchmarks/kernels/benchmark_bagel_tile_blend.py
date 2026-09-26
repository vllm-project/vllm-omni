# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare BAGEL VAE tile merging. Decoder compute and communication are not timed."""

import argparse
import json
import statistics
import time

import torch

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import GridSpec
from vllm_omni.diffusion.models.bagel.autoencoder import DistributedAutoEncoder


class OriginalMerge(DistributedAutoEncoder):
    """The two scalar blend loops from main."""

    def blend_v(self, above: torch.Tensor, current: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(above.shape[-2], current.shape[-2], blend_extent)
        for y in range(blend_extent):
            alpha = y / blend_extent
            current[:, :, y, :] = above[:, :, -blend_extent + y, :] * (1 - alpha) + current[:, :, y, :] * alpha
        return current

    def blend_h(self, left: torch.Tensor, current: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(left.shape[-1], current.shape[-1], blend_extent)
        for x in range(blend_extent):
            alpha = x / blend_extent
            current[:, :, :, x] = left[:, :, :, -blend_extent + x] * (1 - alpha) + current[:, :, :, x] * alpha
        return current


def merger(cls: type[DistributedAutoEncoder]) -> DistributedAutoEncoder:
    # The merge methods do not use weights or a process group.
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.tile_sample_stride_height = 448
    model.tile_sample_stride_width = 448
    return model


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(81)
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "args": vars(args)}))
    models = {"main": merger(OriginalMerge), "PR": merger(DistributedAutoEncoder)}
    for size in args.sizes:
        positions = list(range(0, size, 448))
        grid_size = len(positions)
        source = torch.randn(grid_size**2, 1, 3, 512, 512, device="cuda", dtype=getattr(torch, args.dtype))
        packed = source.clone()
        tiles = {
            (i, j): packed[i * grid_size + j, ..., : min(512, size - y), : min(512, size - x)]
            for i, y in enumerate(positions)
            for j, x in enumerate(positions)
        }
        spec = GridSpec(
            (2, 3),
            (grid_size, grid_size),
            {"sample_height": size, "sample_width": size, "blend_height": 64, "blend_width": 64},
        )
        reference = models["main"].decode_tile_merge(tiles, spec)
        packed.copy_(source)
        assert torch.equal(reference, models["PR"].decode_tile_merge(tiles, spec))
        del reference
        samples: dict[str, list[float]] = {name: [] for name in models}
        peaks = {}
        for name, model in models.items():
            for _ in range(5):
                packed.copy_(source)
                model.decode_tile_merge(tiles, spec)
            torch.accelerator.synchronize()
            packed.copy_(source)
            allocated = torch.accelerator.memory.memory_allocated()
            torch.accelerator.memory.reset_peak_memory_stats()
            output = model.decode_tile_merge(tiles, spec)
            torch.accelerator.synchronize()
            peaks[name] = (torch.accelerator.memory.max_memory_allocated() - allocated) / 2**20
            del output
        for round_ in range(args.rounds):
            for name in list(models) if round_ % 2 == 0 else list(reversed(models)):
                # Restore mutated inputs outside the timed region.
                packed.copy_(source)
                torch.accelerator.synchronize()
                start = time.perf_counter()
                output = models[name].decode_tile_merge(tiles, spec)
                torch.accelerator.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000)
                del output
        print(
            json.dumps(
                {
                    "size": size,
                    "extra_peak_MiB": peaks,
                    "results": {
                        name: {"median_ms": statistics.median(values), "samples_ms": values}
                        for name, values in samples.items()
                    },
                }
            )
        )


if __name__ == "__main__":
    main()

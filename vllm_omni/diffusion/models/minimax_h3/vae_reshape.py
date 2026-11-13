# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local cross-chunk scheduling for the released H3 tiled decoder.

The native temporal loop and spatial merge execute unchanged. Only tile
network work is scheduled ahead, then replayed into the native spatial merge.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from functools import wraps

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)
_MISSING = object()


@contextmanager
def _reference_ffn_projection_batches(model, reference_batch_size):
    """Keep the FFN expansion Linear's original NPU GEMM batch shape.

    The released unstacked decoder invokes w1 with one sample batch per
    spatial tile. Combining tiles changes that Linear's floating-point
    result on Ascend even when each sample's input is identical. Only this
    projection uses the reference batch; the remaining network work stays
    packed. No dtype, weight, input value or output coordinate changes.
    """
    previous = []

    def wrap(original):
        @wraps(original)
        def forward(value):
            if value.ndim != 3 or value.shape[0] <= reference_batch_size:
                return original(value)
            if value.shape[0] % reference_batch_size:
                raise RuntimeError("Packed FFN input disagrees with the native tile batch")
            return torch.cat([original(part) for part in value.split(reference_batch_size, dim=0)], dim=0)

        return forward

    try:
        for block in getattr(model.decoder, "transformer_blocks", ()):
            projection = getattr(getattr(block, "ff", None), "w1", None)
            if not isinstance(projection, torch.nn.Linear):
                continue
            saved = projection.__dict__.get("forward", _MISSING)
            previous.append((projection, saved))
            projection.forward = wrap(projection.forward)
        yield len(previous)
    finally:
        for projection, saved in reversed(previous):
            if saved is _MISSING:
                del projection.forward
            else:
                projection.forward = saved


class _TilePool:
    def __init__(self, adapter, latent, *, all_outputs):
        self.adapter, self.model = adapter, adapter.model
        self.latent, self.all_outputs = latent, all_outputs
        model = self.model
        # Use the released decode_temporal scalar plan; unsupported boundary
        # branches keep their original path instead of changing their splits.
        self.step = int(model.tokens_chunk_size)
        self.length = self.step + int(model.token_overlap)
        self.chunks = (int(latent.shape[2]) + int(model.token_drop)) // self.step - int(model.token_drop > 0)
        self.y = model.split_tiles(int(latent.shape[-2]) * model.vae_ratio, True)
        self.x = model.split_tiles(int(latent.shape[-1]) * model.vae_ratio, True)
        self.tiles = len(self.y[0]) * len(self.x[0])
        self.total = self.chunks * self.tiles
        self.batching = os.environ.get("H3_VIDEO_DECODER_TILE_BATCHING", "1") == "1"
        self.batch = int(os.environ.get("H3_VIDEO_DECODER_TILE_BATCH_SIZE", "4")) if self.batching else 1
        if self.batch <= 0:
            raise ValueError("H3_VIDEO_DECODER_TILE_BATCH_SIZE must be a positive integer")
        self.next_item = self.next_chunk = 0
        self.pending = {}
        self.state = None
        self.output_spec = None
        self.original_tiled = model.tiled_decode
        self.stats = {
            "active": True,
            "chunks": self.chunks,
            "tiles": self.tiles,
            "total_items": self.total,
            "max_tile_batch": self.batch,
            "reference_ffn_batch_size": int(latent.shape[0]),
            "wave_items": [],
            "local_batch_sizes": [],
        }

    def _tile(self, item):
        chunk, tile = divmod(item, self.tiles)
        row, col = divmod(tile, len(self.x[0]))
        ratio = self.model.vae_ratio
        y, h = self.y[0][row] // ratio, self.y[1][row] // ratio
        x, w = self.x[0][col] // ratio, self.x[1][col] // ratio
        begin = chunk * self.step
        return self.latent[:, :, begin : begin + self.length, y : y + h, x : x + w]

    def _wave(self):
        model = self.model
        if self.state is None:
            # The adapter's rank-local/component context has already entered.
            self.state = dict(self.adapter._native_parallel_state())
            self.size = int(self.state["sp_size"]) if model.parallel_tiling else 1
            self.rank = int(self.state["sp_rank"]) if model.parallel_tiling else 0
            self.stats.update(cards=self.size, rank=self.rank)
        begin = self.next_item
        end = min(begin + self.size * self.batch, self.total)
        views = [self._tile(item) for item in range(begin, end)]
        indices = list(range(self.rank, len(views), self.size))
        # Reuse the existing tile batching, decode wrappers, projections,
        # quantized/fused modules and forward hooks without changing their math.
        outputs = model._run_tile_tasks(views, indices, model.decode, self.batching)
        if self.size > 1:
            dist.barrier(group=self.state["sp_process_group"])
            outputs = self._gather(outputs, len(views))
        for offset, output in enumerate(outputs):
            self.pending[begin + offset] = output
        self.stats["wave_items"].append(end - begin)
        self.stats["local_batch_sizes"].append(len(indices))
        self.next_item = end

    def _gather(self, outputs, count):
        """Gather a round-robin wave, including ranks with an empty tail.

        The checkpoint's original gather rejects an empty local task list.
        This request-owned equal-shape gather pads only the rank-local task
        dimension and restores the original global tile order. Metadata is
        shared once because eligible temporal/spatial tiles have fixed shape.
        """
        group = self.state["sp_process_group"]
        dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.uint8)
        if self.output_spec is None:
            metadata = torch.zeros(7, device=self.latent.device, dtype=torch.int64)
            if self.rank == 0:
                output = outputs[0]
                metadata[0] = output.ndim
                metadata[1] = dtypes.index(output.dtype) if output.dtype in dtypes else -1
                if output.ndim <= 5:
                    metadata[2 : 2 + output.ndim] = metadata.new_tensor(output.shape)
            dist.broadcast(metadata, src=dist.get_global_rank(group, 0), group=group)
            values = metadata.tolist()
            if not 0 < values[0] <= 5 or not 0 <= values[1] < len(dtypes):
                raise RuntimeError("Unsupported native cross-chunk tile output")
            self.output_spec = (tuple(values[2 : 2 + values[0]]), dtypes[values[1]])
        shape, dtype = self.output_spec
        max_local = (count + self.size - 1) // self.size
        local = torch.zeros((max_local, *shape), dtype=dtype, device=self.latent.device)
        if outputs:
            local[: len(outputs)].copy_(torch.stack(outputs))
        gathered = [torch.empty_like(local) for _ in range(self.size)]
        dist.all_gather(gathered, local, group=group)
        return [gathered[item % self.size][item // self.size] for item in range(count)]

    def decode_chunk(self, clip):
        chunk = self.next_chunk
        if chunk >= self.chunks or int(clip.shape[2]) != self.length:
            raise RuntimeError("H3 cross-chunk scheduler disagrees with the native temporal plan")
        target = (chunk + 1) * self.tiles
        while self.next_item < target:
            self._wave()
        ordered = [self.pending.pop(item) for item in range(chunk * self.tiles, target)]
        self.next_chunk += 1
        skip = getattr(self.model, "_skip_decode_output", None)
        if not self.all_outputs and skip is not None and skip():
            return None
        # Replay ONLY the already-decoded tiles into the original row-major
        # spatial blend/crop. No network work or second gather occurs here.
        model = self.model
        previous_parallel = model.parallel_tiling
        previous_tasks = model.__dict__.get("_run_tile_tasks", _MISSING)
        model.parallel_tiling = False
        model._run_tile_tasks = lambda *args, **kwargs: ordered
        try:
            return self.original_tiled(clip)
        finally:
            model.parallel_tiling = previous_parallel
            if previous_tasks is _MISSING:
                del model._run_tile_tasks
            else:
                model._run_tile_tasks = previous_tasks


def _supported(model, latent):
    return (
        latent.ndim == 5
        and not getattr(model, "training", True)
        and getattr(model, "use_3d_conv", False)
        and getattr(model, "use_vit_decoder", False)
        and getattr(model, "decoder_tiling", False)
        # Native stacking groups tiles by spatial rank within each chunk.
        # Retain that path until those exact batch groups can be scheduled.
        and not getattr(model, "stack_tiling", False)
        and not getattr(getattr(model, "decoder", None), "spatial_parallel", False)
        and not getattr(model, "isolated_first_frame", False)
        and not getattr(model, "isolated_last_frame", False)
        and int(getattr(model, "tokens_chunk_size", 0)) > 0
        and (int(latent.shape[2]) + int(model.token_drop)) % int(model.tokens_chunk_size) == 0
        and int(latent.shape[2]) >= int(model.tokens_chunk_size) + int(model.token_overlap)
        and all(
            callable(getattr(model, name, None))
            for name in ("split_tiles", "decode", "tiled_decode", "_run_tile_tasks")
        )
    )


@contextmanager
def _cross_chunk_tiling(adapter, latent, *, all_outputs):
    model = adapter.model
    if not _supported(model, latent):
        model._h3_vae_reshape_last_stats = {"active": False, "reason": "unsupported native boundary/parallel mode"}
        logger.info("H3_VAE_RESHAPE inactive: %s", model._h3_vae_reshape_last_stats["reason"])
        yield
        return
    pool = _TilePool(adapter, latent, all_outputs=all_outputs)
    previous = model.__dict__.get("tiled_decode", _MISSING)
    model.tiled_decode = pool.decode_chunk
    try:
        with _reference_ffn_projection_batches(model, int(latent.shape[0])) as projections:
            pool.stats["reference_ffn_projections"] = projections
            yield
            if pool.next_chunk == 0:
                pool.stats.update(active=False, status="native-rank-skip")
                return
            if pool.next_chunk != pool.chunks or pool.next_item != pool.total or pool.pending:
                raise RuntimeError("H3 cross-chunk decode did not consume its complete native plan")
            pool.stats["status"] = "complete"
            pool.stats["idle_item_slots"] = (-pool.total) % pool.size
    finally:
        if previous is _MISSING:
            del model.tiled_decode
        else:
            model.tiled_decode = previous
        model._h3_vae_reshape_last_stats = pool.stats
        logger.info("H3_VAE_RESHAPE %s", json.dumps(pool.stats, sort_keys=True))


def install_cross_chunk_tiling(adapter):
    """Install after the native model is ready; flags must precede construction."""
    if os.environ.get("H3_VIDEO_DECODER_CROSS_CHUNK", "0") != "1":
        return
    if getattr(adapter, "_h3_vae_reshape_installed", False):
        return
    for name in ("decode_latent", "decode_with_chunks"):
        original = getattr(adapter, name, None)
        if original is None:
            continue

        def wrap(function, all_outputs):
            @wraps(function)
            @torch.inference_mode()
            def decode(latent, *args, **kwargs):
                # Keep the adapter's existing latent denormalization contract;
                # the scheduler slices the same tensor the native path decodes.
                denormalized = adapter._denormalize_latent(latent)
                with _cross_chunk_tiling(adapter, denormalized, all_outputs=all_outputs):
                    return function(latent, *args, **kwargs)

            return decode

        setattr(adapter, name, wrap(original, name == "decode_with_chunks"))
    adapter._h3_vae_reshape_installed = True


__all__ = ["install_cross_chunk_tiling"]

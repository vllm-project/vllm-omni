# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded cross-clip scheduling for the checkpoint's independent VAE tiles."""

from collections.abc import Callable
from dataclasses import asdict
from typing import Protocol

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from .vae_schedule import TileTask, allocate_tiles, plan_temporal_schedule

logger = init_logger(__name__)


class TiledVideoDecoder(Protocol):
    """Checkpoint primitives needed to schedule independent spatial tiles."""

    training: bool
    decoder_parallel: bool
    decoder_tiling: bool
    parallel_tiling: bool
    stack_tiling: bool
    vae_ratio: int
    vae_ratio_t: int
    tokens_chunk_size: int
    token_overlap: int
    use_vit_decoder: bool

    def split_tiles(self, length: int, decode: bool) -> tuple[list[int], list[int], list[int]]: ...
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...
    def _adaptive_decode(self, z: torch.Tensor) -> torch.Tensor: ...
    def blend(self, a: torch.Tensor, b: torch.Tensor, overlap: int, dim: int) -> torch.Tensor: ...
    def _run_tile_tasks(
        self,
        tiles: list[torch.Tensor],
        indices: list[int],
        forward: Callable[[torch.Tensor], torch.Tensor],
        stack: bool,
    ) -> list[torch.Tensor]: ...
    def _all_gather_tiled_results(self, tasks: list[torch.Tensor], count: int) -> list[torch.Tensor]: ...


def temporal_scheduling_unavailable_reason(model: object) -> str | None:
    """Unsupported checkpoints keep their original decode implementation."""
    methods = ("split_tiles", "_run_tile_tasks", "_all_gather_tiled_results", "decode", "blend")
    missing = [name for name in methods if not callable(getattr(model, name, None))]
    if missing:
        return f"unsupported_checkpoint_tile_methods:{missing}"
    if not getattr(model, "use_vit_decoder", False):
        return "unverified_decoder_context_contract"
    transform = getattr(getattr(model, "processor", None), "transform_rev", None)
    if not callable(getattr(model, "_decode_temporal_streaming", None)) or any(
        not isinstance(getattr(transform, name, None), (list, tuple)) or len(getattr(transform, name)) != 3
        for name in ("mean", "std")
    ):
        return "unsupported_checkpoint_stream_contract"
    return None


def _validate_decode_state(model: TiledVideoDecoder, rank: int, world_size: int) -> None:
    if model.training or getattr(model, "decoder_parallel", False):
        raise ValueError("VAE temporal windows require inference with decoder_parallel disabled")
    if type(world_size) is not int or type(rank) is not int or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid VAE tile rank {rank} for group size {world_size}")


def describe_tiles(
    model: TiledVideoDecoder,
    shape: tuple[int, ...],
    clip: int,
    context: tuple[int, int, int, int],
) -> tuple[TileTask, ...]:
    """Metadata only: preserve N, T, causal context and native output position."""
    n, c, t, h, w = shape
    ratio = model.vae_ratio
    ys, heights, _ = model.split_tiles(h * ratio, True)
    xs, widths, _ = model.split_tiles(w * ratio, True)
    return tuple(
        TileTask(clip, i * len(xs) + j, n, (n, c, t, height // ratio, width // ratio), context, (y, x))
        for i, (y, height) in enumerate(zip(ys, heights))
        for j, (x, width) in enumerate(zip(xs, widths))
    )


def select_temporal_window(
    model: TiledVideoDecoder,
    z: torch.Tensor,
    head: torch.Tensor | None,
    tail: torch.Tensor | None,
    num_chunks: int,
    *,
    rank: int,
    world_size: int,
    process_group: dist.ProcessGroup | None,
) -> int:
    """Choose from task geometry and safe storage; never execute a trial decode."""
    _validate_decode_state(model, rank, world_size)
    if z.ndim != 5 or type(num_chunks) is not int or num_chunks < 1:
        raise ValueError("VAE scheduling requires a 5D latent and positive temporal clip count")
    reason = None
    if not model.decoder_tiling or not model.parallel_tiling or world_size == 1:
        reason = "tiling_disabled_or_single_rank"
    elif model.stack_tiling:
        reason = "native_stacked_batch_contract"
    elif not getattr(model, "use_vit_decoder", False):
        reason = "unverified_decoder_context_contract"
    elif num_chunks == 1:
        reason = "single_temporal_clip"
    if reason:
        logger.info(
            "H3 VAE scheduling requested=true selected=false executed=false group=%d window=1 reason=%s",
            world_size,
            reason,
        )
        return 1
    clips = []
    for i in range(num_chunks):
        offset = i * model.tokens_chunk_size
        stop = min(offset + model.tokens_chunk_size + model.token_overlap, z.shape[2])
        h = head.shape[2] if i == 0 and head is not None else 0
        t = tail.shape[2] if i == num_chunks - 1 and tail is not None else 0
        shape = (*z.shape[:2], stop - offset + h + t, *z.shape[-2:])
        clips.append(describe_tiles(model, shape, i, (offset, stop, h, t)))
    module = torch.get_device_module(z.device)
    free, _ = module.mem_get_info(z.device)
    # Reserve >=75% of free memory for native activations/fragmentation and
    # bound incremental live outputs/gather storage without changing precision.
    budget = min(2 * 1024**3, free // 4)
    control_device = "cpu" if dist.get_backend(process_group) == "gloo" else z.device
    shared = torch.tensor(budget, dtype=torch.int64, device=control_device)
    dist.all_reduce(shared, op=dist.ReduceOp.MIN, group=process_group)
    budget = int(shared.item())
    plan = plan_temporal_schedule(
        tuple(clips),
        world_size,
        budget_bytes=budget,
        # RGB float32 decoded tiles; t*ratio_t is a conservative temporal
        # bound, including a short tail. N and spatial dimensions remain real.
        output_bytes_per_latent_pixel=3 * 4 * model.vae_ratio_t * model.vae_ratio**2,
        input_element_bytes=z.element_size(),
    )
    logger.info(
        "H3 VAE scheduling requested=true selected=%s executed=false group=%d input=%s tiles=%s "
        "window=%d balancing_period=%d candidates=%s budget=%d reason=%s",
        plan.optimized,
        world_size,
        tuple(z.shape),
        [t.shape for t in clips[0]],
        plan.window,
        plan.balancing_period,
        [asdict(c) for c in plan.candidates],
        budget,
        plan.reason,
    )
    return plan.window


def decode_temporal_window(
    model: TiledVideoDecoder,
    clips: list[torch.Tensor],
    *,
    rank: int,
    world_size: int,
    process_group: dist.ProcessGroup | None,
    contexts: list[tuple[int, int, int, int]] | None = None,
) -> list[torch.Tensor]:
    """Decode independent tiles, then restore native spatial/temporal order.

    Only equal-shaped tiles share a collective or a native stacked batch.
    A bucket smaller than the process group is decoded on every rank: the
    checkpoint gather rejects empty local task lists. The choice depends only
    on replicated input metadata, so all ranks follow the same collectives.
    """
    _validate_decode_state(model, rank, world_size)
    if not model.decoder_tiling or not model.parallel_tiling or world_size == 1 or len(clips) == 1:
        # A short final window may have fewer spatial tiles than ranks. Avoid
        # the checkpoint's empty-rank gather while retaining native arithmetic.
        parallel = model.parallel_tiling
        try:
            if clips and len(describe_tiles(model, clips[0].shape, 0, (0, 0, 0, 0))) < world_size:
                model.parallel_tiling = False
            return [model._adaptive_decode(clip) for clip in clips]
        finally:
            model.parallel_tiling = parallel

    # Each entry keeps its clip and row-major spatial tile identity. Never
    # batch a short tail with a full clip or an edge tile with a different shape.
    entries = []
    descriptors = []
    layouts = []
    results: list[dict[int, torch.Tensor]] = []
    for clip_index, clip in enumerate(clips):
        ratio = model.vae_ratio
        y_idx, y_len, y_overlap = model.split_tiles(clip.shape[-2] * ratio, True)
        x_idx, x_len, x_overlap = model.split_tiles(clip.shape[-1] * ratio, True)
        layouts.append((len(y_idx), len(x_idx), y_overlap, x_overlap))
        results.append({})
        for i, (y, height) in enumerate(zip(y_idx, y_len)):
            for j, (x, width) in enumerate(zip(x_idx, x_len)):
                tile = clip[..., y // ratio : y // ratio + height // ratio, x // ratio : x // ratio + width // ratio]
                entries.append((clip_index, i * len(x_idx) + j, tile))
        context = contexts[clip_index] if contexts is not None else (0, clip.shape[2], 0, 0)
        descriptors.extend(describe_tiles(model, clip.shape, clip_index, context))

    for bucket in allocate_tiles(tuple(descriptors), world_size):
        distributed = len({i for share in bucket.ranks for i in share}) >= world_size
        slots = max(map(len, bucket.ranks))
        # Reindex LPT ownership into the native gather's round-robin wire order.
        # Pad with references to decoded outputs, never with decoder forwards.
        wire = [share[k] if k < len(share) else None for k in range(slots) for share in bucket.ranks]
        tiles = [entries[i if i is not None else bucket.ranks[k % world_size][0]][2] for k, i in enumerate(wire)]
        indices = list(range(rank, len(tiles), world_size))[: len(bucket.ranks[rank])]
        decoded = model._run_tile_tasks(tiles, indices, model.decode, model.stack_tiling)
        if distributed:
            decoded.extend([decoded[0]] * (slots - len(decoded)))
            dist.barrier(group=process_group)
            decoded = model._all_gather_tiled_results(decoded, len(tiles))
            if len(decoded) != len(tiles):
                raise RuntimeError("VAE tile gather returned the wrong number of tiles")
            selected = [(i, value) for i, value in zip(wire, decoded) if i is not None]
        else:
            selected = list(zip(bucket.ranks[rank], decoded))
        logger.info(
            "H3 VAE scheduling requested=true selected=true executed=%s group=%d window=%d "
            "shape=%s rank_tasks=%s actual_batch=%d rank_work_units=%s",
            distributed,
            world_size,
            len(clips),
            bucket.shape,
            [len(s) for s in bucket.ranks],
            bucket.shape[0],
            [sum(descriptors[i].cost for i in s) for s in bucket.ranks],
        )
        for index, output in selected:
            clip_index, tile_index, _ = entries[index]
            results[clip_index][tile_index] = output.to(clips[clip_index].device)

    outputs = []
    for tiles, (height, width, y_overlap, x_overlap) in zip(results, layouts):
        rows = [[tiles[i * width + j] for j in range(width)] for i in range(height)]
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # Match native tiled_decode: blend against the raw neighbors,
                # above before left, then crop overlap. Do not mutate rows.
                if i > 0:
                    tile = model.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = model.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
                if i < height - 1:
                    tile = tile[..., : -y_overlap[i], :]
                if j < width - 1:
                    tile = tile[..., :, : -x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        outputs.append(torch.cat(result_rows, dim=-2))
    return outputs

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


def _reference_projection_modules(model):
    """Preserve shape-sensitive FFN and attention output Linear calls."""
    for block in getattr(model.decoder, "transformer_blocks", ()):
        projections = [
            getattr(getattr(block, "ff", None), "w1", None),
            getattr(getattr(block, "attn", None), "to_out", None),
        ]
        if not getattr(model, "stack_tiling", False):
            # With no native tile stacking the reference calls are smaller;
            # QKV and FFN contraction also vary for narrower tile sequences.
            projections.extend(
                [
                    getattr(getattr(block, "attn", None), "to_qkv", None),
                    getattr(getattr(block, "ff", None), "w2", None),
                ]
            )
        for projection in projections:
            if isinstance(projection, torch.nn.Linear):
                yield projection
    projection = getattr(model.decoder, "proj_out", None)
    if isinstance(projection, torch.nn.Linear):
        yield projection


@contextmanager
def _reference_ffn_projection_batches(model, reference_batch_size):
    """Keep the sensitive decoder projections' original NPU GEMM batch shapes.

    The released unstacked decoder invokes w1 with one sample batch per
    spatial tile. Combining tiles changes that Linear's floating-point
    result on Ascend even when each sample's input is identical. The FFN expansion and attention output
    projections use reference batches; the remaining network work stays
    packed. No dtype, weight, input value or output coordinate changes.
    """
    previous = []

    def wrap(original):
        @wraps(original)
        def forward(value):
            if value.ndim != 3:
                return original(value)
            if callable(reference_batch_size):
                groups = reference_batch_size()
                if not groups:
                    return original(value)
                if sum(groups) != value.shape[0]:
                    raise RuntimeError("Packed FFN input disagrees with native stack groups")
                if len(groups) == 1:
                    return original(value)
            else:
                if value.shape[0] <= reference_batch_size:
                    return original(value)
                if value.shape[0] % reference_batch_size:
                    raise RuntimeError("Packed FFN input disagrees with the native tile batch")
                groups = reference_batch_size
            return torch.cat([original(part) for part in value.split(groups, dim=0)], dim=0)

        return forward

    try:
        for projection in _reference_projection_modules(model):
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


@contextmanager
def _reference_ffn_projection_outputs(model, reference_groups):
    """Evaluate native GEMMs into disjoint slices without a concatenation copy."""
    previous = []
    activity = {"fast_calls": 0, "gemm_calls": 0, "weight_prepares": 0, "fallback_calls": 0, "native_output_calls": 0}
    previous_activity = model.__dict__.get("_h3_projection_out_activity", _MISSING)
    model._h3_projection_out_activity = activity

    def wrap(projection, original, standard, native_only=False):
        cached_key = cached_weight = cached_bias = None

        @wraps(original)
        def forward(value):
            nonlocal cached_key, cached_weight, cached_bias
            groups = reference_groups()
            if value.ndim != 3 or not groups:
                return original(value)
            if sum(groups) != value.shape[0]:
                raise RuntimeError("Packed FFN input disagrees with native stack groups")
            if native_only:
                # Keep the final projection's exact original operator. This
                # narrow result is copied once per decoder forward, unlike
                # the repeated wide FFN expansion outputs.
                activity["native_output_calls"] += 1
                activity["gemm_calls"] += len(groups)
                if len(groups) == 1:
                    return original(value)
                return torch.cat([original(part) for part in value.split(groups, dim=0)], dim=0)
            enabled = torch.is_autocast_enabled(value.device.type)
            dtype = torch.get_autocast_dtype(value.device.type) if enabled else value.dtype
            supported = (
                standard
                and projection.bias is not None
                and value.is_contiguous()
                and not torch.is_grad_enabled()
                and value.dtype != torch.float64
                and projection.weight.dtype != torch.float64
                and (enabled or (projection.weight.dtype == value.dtype and projection.bias.dtype == value.dtype))
            )
            if not supported:
                activity["fallback_calls"] += 1
                return torch.cat([original(part) for part in value.split(groups, dim=0)], dim=0)
            activity["fast_calls"] += 1
            key = (projection.weight.data_ptr(), projection.bias.data_ptr(), value.device, dtype)
            if cached_key != key:
                cached_weight = projection.weight.to(dtype=dtype)
                cached_bias = projection.bias.to(dtype=dtype)
                cached_key = key
                activity["weight_prepares"] += 1
            inputs = value.to(dtype=dtype)
            result = torch.empty((*value.shape[:-1], projection.out_features), device=value.device, dtype=dtype)
            with torch.autocast(device_type=value.device.type, enabled=False):
                for part, output in zip(inputs.split(groups, dim=0), result.split(groups, dim=0)):
                    activity["gemm_calls"] += 1
                    torch.addmm(
                        cached_bias,
                        part.reshape(-1, part.shape[-1]),
                        cached_weight.t(),
                        out=output.view(-1, projection.out_features),
                    )
            return result

        return forward

    try:
        for projection in _reference_projection_modules(model):
            saved = projection.__dict__.get("forward", _MISSING)
            previous.append((projection, saved))
            standard = type(projection) is torch.nn.Linear and saved is _MISSING
            projection.forward = wrap(
                projection,
                projection.forward,
                standard,
                native_only=projection is getattr(model.decoder, "proj_out", None),
            )
        yield len(previous)
    finally:
        for projection, saved in reversed(previous):
            if saved is _MISSING:
                del projection.forward
            else:
                projection.forward = saved
        if previous_activity is _MISSING:
            del model._h3_projection_out_activity
        else:
            model._h3_projection_out_activity = previous_activity


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


class _NativeStackTilePool(_TilePool):
    """Pack adjacent chunks while retaining each rank's native tile groups."""

    def __init__(self, adapter, latent, *, all_outputs):
        super().__init__(adapter, latent, all_outputs=all_outputs)
        self.reference_groups = None
        self.native_fallback = False
        self.stats.update(native_stack_groups=True, reference_group_batches=[])

    def _initialize_native_plan(self):
        if self.state is not None:
            return
        self.state = dict(self.adapter._native_parallel_state())
        self.size = int(self.state["sp_size"]) if self.model.parallel_tiling else 1
        self.rank = int(self.state["sp_rank"]) if self.model.parallel_tiling else 0
        self.native_indices = [self.model._local_tile_indices(self.tiles, rank, self.size) for rank in range(self.size)]
        largest = max(map(len, self.native_indices))
        self.chunk_batch = max(1, self.batch // largest)
        self.native_fallback = not self.batching or largest > self.batch
        self.stats.update(
            cards=self.size,
            rank=self.rank,
            native_tile_group_sizes=list(map(len, self.native_indices)),
            max_chunks_per_wave=self.chunk_batch,
        )
        if self.native_fallback:
            self.stats.update(
                active=False,
                status="native-group-fallback",
                reason="Native stack group exceeds the requested packing limit",
            )

    def _wave(self):
        self._initialize_native_plan()
        begin_chunk = self.next_item // self.tiles
        end_chunk = min(begin_chunk + self.chunk_batch, self.chunks)
        plan = [
            [chunk * self.tiles + tile for chunk in range(begin_chunk, end_chunk) for tile in indices]
            for indices in self.native_indices
        ]
        local = plan[self.rank]
        views = [self._tile(item) for item in local]
        self.reference_groups = (len(self.native_indices[self.rank]) * int(self.latent.shape[0]),) * (
            end_chunk - begin_chunk
        )
        try:
            outputs = self.model._run_tile_tasks(views, list(range(len(views))), self.model.decode, True)
        finally:
            self.reference_groups = None
        if self.size > 1:
            dist.barrier(group=self.state["sp_process_group"])
            max_local = max(map(len, plan))
            # Reuse the equal-shape gather with a padded rectangular layout,
            # then restore the original chunk/tile coordinates from the plan.
            gathered = super()._gather(outputs, self.size * max_local)
            for rank, items in enumerate(plan):
                for offset, item in enumerate(items):
                    self.pending[item] = gathered[rank + offset * self.size]
        else:
            self.pending.update(zip(local, outputs))
        self.next_item = end_chunk * self.tiles
        self.stats["wave_items"].append((end_chunk - begin_chunk) * self.tiles)
        self.stats["local_batch_sizes"].append(len(local))
        self.stats["reference_group_batches"].append(
            [len(self.native_indices[self.rank]) * int(self.latent.shape[0])] * (end_chunk - begin_chunk)
        )

    def decode_chunk(self, clip):
        self._initialize_native_plan()
        if self.native_fallback:
            self.next_chunk += 1
            self.next_item += self.tiles
            return self.original_tiled(clip)
        return super().decode_chunk(clip)


def _pack_native_groups(groups, size, limit):
    bins = [[] for _ in range(size)]
    counts = [0] * size
    for group in sorted(groups, key=lambda item: -len(item)):
        choices = [rank for rank in range(size) if counts[rank] + len(group) <= limit]
        if not choices:
            return None
        rank = max(choices, key=lambda item: (counts[item], -item))
        bins[rank].append(group)
        counts[rank] += len(group)
    return tuple(tuple(items) for items in bins)


class _BalancedNativeStackTilePool(_NativeStackTilePool):
    """Move indivisible native groups across replicated decoder ranks."""

    def __init__(self, adapter, latent, *, all_outputs):
        super().__init__(adapter, latent, all_outputs=all_outputs)
        self.wave_index = 0
        self.stats.update(schedule="balanced-native-groups", native_groups_per_wave=[])

    def _initialize_native_plan(self):
        super()._initialize_native_plan()
        if hasattr(self, "wave_plans") or self.native_fallback:
            return
        self.stats.pop("max_chunks_per_wave", None)
        key = (self.size, self.batch, self.chunks, tuple(tuple(items) for items in self.native_indices))
        cached = self.model.__dict__.get("_h3_native_group_plan_cache")
        if cached is not None and cached[0] == key:
            self.wave_plans = cached[1]
            self._check_schedule_benefit()
            return
        groups = [
            tuple(chunk * self.tiles + tile for tile in items)
            for chunk in range(self.chunks)
            for items in self.native_indices
            if items
        ]
        plans = []
        cursor = 0
        while cursor < len(groups):
            selected = []
            packed = None
            while cursor < len(groups):
                trial = selected + [groups[cursor]]
                trial_plan = _pack_native_groups(trial, self.size, self.batch)
                if trial_plan is None:
                    break
                selected, packed = trial, trial_plan
                cursor += 1
            if not selected:
                raise RuntimeError("Native group cannot fit its packing limit")
            if cursor == len(groups):
                total = sum(map(len, selected))
                limit = max(max(map(len, selected)), (total + self.size - 1) // self.size)
                while limit < self.batch:
                    compact = _pack_native_groups(selected, self.size, limit)
                    if compact is not None:
                        packed = compact
                        break
                    limit += 1
            plans.append(packed)
        self.wave_plans = tuple(plans)
        self.model._h3_native_group_plan_cache = (key, self.wave_plans)
        self._check_schedule_benefit()

    def _check_schedule_benefit(self):
        native_max_load = max(map(len, self.native_indices)) * self.chunks
        planned_max_load = max(
            sum(len(group) for wave in self.wave_plans for group in wave[rank]) for rank in range(self.size)
        )
        if len(self.wave_plans) >= self.chunks and planned_max_load >= native_max_load:
            self.native_fallback = True
            self.stats.update(
                active=False,
                status="native-group-no-gain",
                reason="Native groups already have the same wave count and maximum tile load",
            )

    def _wave(self):
        self._initialize_native_plan()
        groups_by_rank = self.wave_plans[self.wave_index]
        plan = [[item for group in groups for item in group] for groups in groups_by_rank]
        local = plan[self.rank]
        views = [self._tile(item) for item in local]
        reference = tuple(len(group) * int(self.latent.shape[0]) for group in groups_by_rank[self.rank])
        self.reference_groups = reference
        try:
            outputs = self.model._run_tile_tasks(views, list(range(len(views))), self.model.decode, True)
        finally:
            self.reference_groups = None
        if self.size > 1:
            dist.barrier(group=self.state["sp_process_group"])
            max_local = max(map(len, plan))
            gathered = _TilePool._gather(self, outputs, self.size * max_local)
            for rank, items in enumerate(plan):
                for offset, item in enumerate(items):
                    self.pending[item] = gathered[rank + offset * self.size]
        else:
            self.pending.update(zip(local, outputs))
        count = sum(map(len, plan))
        self.next_item += count
        self.wave_index += 1
        self.stats["wave_items"].append(count)
        self.stats["local_batch_sizes"].append(len(local))
        self.stats["reference_group_batches"].append(list(reference))
        self.stats["native_groups_per_wave"].append(len(groups_by_rank[self.rank]))


def _supported(model, latent):
    return (
        latent.ndim == 5
        and not getattr(model, "training", True)
        and getattr(model, "use_3d_conv", False)
        and getattr(model, "use_vit_decoder", False)
        and getattr(model, "decoder_tiling", False)
        and (not getattr(model, "stack_tiling", False) or callable(getattr(model, "_local_tile_indices", None)))
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
    original_tokens = int(latent.shape[2]) if latent.ndim == 5 else None
    pad_tokens = 0
    # Mirror decode_temporal's token copies while retaining its own final
    # frame trimming and blending. No arithmetic or latent value changes.
    if (
        latent.ndim == 5
        and latent.shape[2] > 0
        and not getattr(model, "training", True)
        and not getattr(model, "isolated_first_frame", False)
        and not getattr(model, "isolated_last_frame", False)
        and int(getattr(model, "tokens_chunk_size", 0)) > 0
    ):
        pad_tokens = -(int(latent.shape[2]) + int(model.token_drop)) % int(model.tokens_chunk_size)
        if pad_tokens:
            latent = torch.cat([latent, latent[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    if not _supported(model, latent):
        model._h3_vae_reshape_last_stats = {"active": False, "reason": "unsupported native boundary/parallel mode"}
        logger.info("H3_VAE_RESHAPE inactive: %s", model._h3_vae_reshape_last_stats["reason"])
        yield
        return
    pool_class = _BalancedNativeStackTilePool if getattr(model, "stack_tiling", False) else _TilePool
    pool = pool_class(adapter, latent, all_outputs=all_outputs)
    pool.stats.update(original_latent_tokens=original_tokens, native_pad_tokens=pad_tokens)
    previous = model.__dict__.get("tiled_decode", _MISSING)
    model.tiled_decode = pool.decode_chunk
    try:
        reference = (lambda: pool.reference_groups) if isinstance(pool, _NativeStackTilePool) else int(latent.shape[0])
        context = (
            _reference_ffn_projection_outputs
            if isinstance(pool, _NativeStackTilePool)
            else _reference_ffn_projection_batches
        )
        with context(model, reference) as projections:
            pool.stats["reference_projection_count"] = projections
            pool.stats["reference_ffn_projections"] = sum(
                isinstance(getattr(getattr(block, "ff", None), "w1", None), torch.nn.Linear)
                for block in getattr(model.decoder, "transformer_blocks", ())
            )
            pool.stats["reference_attention_output_projections"] = sum(
                isinstance(getattr(getattr(block, "attn", None), "to_out", None), torch.nn.Linear)
                for block in getattr(model.decoder, "transformer_blocks", ())
            )
            pool.stats["reference_final_output_projections"] = int(
                isinstance(getattr(model.decoder, "proj_out", None), torch.nn.Linear)
            )
            pool.stats["reference_attention_qkv_projections"] = (
                sum(
                    isinstance(getattr(getattr(block, "attn", None), "to_qkv", None), torch.nn.Linear)
                    for block in getattr(model.decoder, "transformer_blocks", ())
                )
                if not getattr(model, "stack_tiling", False)
                else 0
            )
            pool.stats["reference_ffn_contraction_projections"] = (
                sum(
                    isinstance(getattr(getattr(block, "ff", None), "w2", None), torch.nn.Linear)
                    for block in getattr(model.decoder, "transformer_blocks", ())
                )
                if not getattr(model, "stack_tiling", False)
                else 0
            )
            if isinstance(pool, _NativeStackTilePool):
                pool.stats["projection_out"] = model._h3_projection_out_activity
            yield
            if isinstance(pool, _NativeStackTilePool) and pool.native_fallback:
                return
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-clip tile scheduling for MiniMax H3 reference-video encoding."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType
from typing import Any, cast

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_CONFIG_KEY = "h3_vae_encode_tile_balance_max_tasks"
_ACTIVE = ContextVar("h3_vae_encode_tile_balance", default=False)


@contextmanager
def video_encode_tile_balance_context() -> Iterator[None]:
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def resolve_video_tile_balance_max_tasks(config: Mapping[str, Any] | None) -> int | None:
    """Read the opt-in task-window size from H3 additional config."""

    value = (config or {}).get(_CONFIG_KEY)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{_CONFIG_KEY} must be a positive integer, got {value!r}")
    return value


def _task_ids(start: int, end: int, rank: int, world_size: int) -> list[int]:
    first = start + ((rank - start) % world_size)
    return list(range(first, end, world_size))


def _windows(chunks: int, tiles: int, max_tasks: int) -> list[dict[str, Any]]:
    clips_per_window = max(1, max_tasks // tiles)
    windows = []
    for clip_start in range(0, chunks, clips_per_window):
        clip_end = min(chunks, clip_start + clips_per_window)
        task_start = clip_start * tiles
        task_end = clip_end * tiles
        windows.append(
            {
                "clip_start": clip_start,
                "clip_end": clip_end,
                "task_start": task_start,
                "task_end": task_end,
            }
        )
    return windows


def _plan(model: Any, x: torch.Tensor, world_size: int, max_tasks: int) -> dict[str, Any]:
    y_idx, y_len, y_overlap = model.split_tiles(int(x.shape[-2]), False)
    x_idx, x_len, x_overlap = model.split_tiles(int(x.shape[-1]), False)
    clip_length = int(model.clip_length)
    orig_frames = int(x.shape[2])
    pad_frames = (-orig_frames) % clip_length
    chunks = (orig_frames + pad_frames) // clip_length
    tiles = len(y_idx) * len(x_idx)
    return {
        "clip_length": clip_length,
        "orig_frames": orig_frames,
        "pad_frames": pad_frames,
        "chunks": chunks,
        "tiles": tiles,
        "tile_grid": (len(y_idx), len(x_idx)),
        "y_idx": y_idx,
        "y_len": y_len,
        "y_overlap": y_overlap,
        "x_idx": x_idx,
        "x_len": x_len,
        "x_overlap": x_overlap,
        "windows": _windows(chunks, tiles, max_tasks),
        "world_size": world_size,
    }


def _unsupported(model: Any, x: torch.Tensor, plan: Mapping[str, Any]) -> bool:
    return any(
        (
            model.training,
            not model.encoder_tiling,
            model.encoder_parallel,
            model.stack_tiling,
            model.isolated_first_frame,
            model.isolated_last_frame,
            model.isolated_key_frame,
            not callable(getattr(model, "encode", None)),
            hasattr(model.encoder, "loss_info"),
            x.dtype != torch.float32,
            x.shape[2] <= 0,
            plan["chunks"] <= 1,
            plan["world_size"] <= 1,
            plan["tiles"] < plan["world_size"],
            len(set(plan["y_len"])) != 1,
            len(set(plan["x_len"])) != 1,
        )
    )


def _clip(model: Any, x: torch.Tensor, clip_index: int, plan: Mapping[str, Any]) -> torch.Tensor:
    start = clip_index * plan["clip_length"]
    end = start + plan["clip_length"]
    clip = x[:, :, start : min(end, plan["orig_frames"])]
    if end > plan["orig_frames"]:
        clip = torch.cat(
            [clip, clip[:, :, -1:].repeat(1, 1, end - plan["orig_frames"], 1, 1)],
            dim=2,
        )
    return clip


def _blend_clip(model: Any, encoded: Sequence[torch.Tensor], x: torch.Tensor, plan: Mapping[str, Any]) -> torch.Tensor:
    row_count, column_count = plan["tile_grid"]
    rows: list[list[torch.Tensor | None]] = [[None for _ in range(column_count)] for _ in range(row_count)]
    for index, tile in enumerate(encoded):
        rows[index // column_count][index % column_count] = tile.to(x.device)
    y_overlap = [overlap // model.vae_ratio for overlap in plan["y_overlap"]]
    x_overlap = [overlap // model.vae_ratio for overlap in plan["x_overlap"]]
    result_rows = []
    for row_index, optional_row in enumerate(rows):
        if any(tile is None for tile in optional_row):
            raise RuntimeError("tile-balance blend received an incomplete row")
        row = cast(list[torch.Tensor], optional_row)
        result_row = []
        for column_index, tile in enumerate(row):
            if row_index > 0:
                tile = model.blend(rows[row_index - 1][column_index], tile, y_overlap[row_index - 1], dim=-2)
            if column_index > 0:
                tile = model.blend(row[column_index - 1], tile, x_overlap[column_index - 1], dim=-1)
            if row_index < len(rows) - 1 and y_overlap[row_index] > 0:
                tile = tile[..., : -y_overlap[row_index], :]
            if column_index < len(row) - 1 and x_overlap[column_index] > 0:
                tile = tile[..., :, : -x_overlap[column_index]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def _reassemble(
    gathered: Sequence[torch.Tensor],
    task_start: int,
    task_end: int,
    world_size: int,
) -> list[torch.Tensor]:
    if len(gathered) != world_size:
        raise RuntimeError(f"tile-balance gather returned {len(gathered)} ranks, expected {world_size}")
    encoded: list[torch.Tensor | None] = [None] * (task_end - task_start)
    for source_rank, tensors in enumerate(gathered):
        source_ids = _task_ids(task_start, task_end, source_rank, world_size)
        if int(tensors.shape[0]) != len(source_ids):
            raise RuntimeError("tile-balance gather returned an unexpected task count")
        for task_index, tensor in zip(source_ids, tensors):
            encoded[task_index - task_start] = tensor
    if any(tile is None for tile in encoded):
        raise RuntimeError("tile-balance gather left a missing task")
    return cast(list[torch.Tensor], encoded)


def _run(
    model: Any,
    x: torch.Tensor,
    plan: Mapping[str, Any],
    rank: int,
    group: Any,
    all_gather_var_shape: Any,
) -> torch.Tensor:
    outputs = []
    column_count = plan["tile_grid"][1]
    for window in plan["windows"]:
        task_start = window["task_start"]
        task_end = window["task_end"]
        local_ids = _task_ids(task_start, task_end, rank, plan["world_size"])
        tasks = []
        clips: dict[int, torch.Tensor] = {}
        for task_index in local_ids:
            clip_index, tile_index = divmod(task_index, plan["tiles"])
            row, column = divmod(tile_index, column_count)
            clip = clips.get(clip_index)
            if clip is None:
                clip = _clip(model, x, clip_index, plan)
                clips[clip_index] = clip
            tile = clip[
                ...,
                plan["y_idx"][row] : plan["y_idx"][row] + plan["y_len"][row],
                plan["x_idx"][column] : plan["x_idx"][column] + plan["x_len"][column],
            ]
            tasks.append(model.encode(tile))
        if plan["world_size"] > 1:
            stacked = torch.stack(tasks)
            gathered = all_gather_var_shape(stacked, group=group)
            encoded = _reassemble(gathered, task_start, task_end, plan["world_size"])
        else:
            encoded = tasks
        for local_clip in range(window["clip_end"] - window["clip_start"]):
            start = local_clip * plan["tiles"]
            outputs.append(_blend_clip(model, encoded[start : start + plan["tiles"]], x, plan))
    result = torch.cat(outputs, dim=2)
    if model.token_drop > 0:
        result = result[:, :, : -model.token_drop]
    return result


def install_video_tile_balance(adapter: Any, max_tasks: int | None) -> bool:
    """Install the opt-in NPU scheduler after the temporal encode fork."""

    if max_tasks is None or adapter._device_target.type != "npu" or adapter.decode_only:
        return False
    if max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    model = adapter.model
    if getattr(model, "_omni_video_tile_balance_installed", False):
        return True
    original = getattr(model, "encode_temporal", None)
    gather_method = getattr(model, "_all_gather_tiled_results", None)
    gather_function = getattr(gather_method, "__func__", gather_method)
    gather_namespace = getattr(gather_function, "__globals__", None)
    all_gather_var_shape = gather_namespace.get("all_gather_var_shape") if isinstance(gather_namespace, dict) else None
    if not callable(original) or not callable(all_gather_var_shape):
        logger.warning_once("MiniMax H3 checkpoint does not expose encode tile-balance collectives")
        return False

    def encode_temporal(instance: Any, x: torch.Tensor) -> torch.Tensor:
        if not _ACTIVE.get() or x.ndim != 5 or not instance.parallel_tiling:
            return original(x)
        state = adapter._native_parallel_state()
        world_size = int(state.get("sp_size", 1))
        rank = int(state.get("sp_rank", 0))
        plan = _plan(instance, x, world_size, max_tasks)
        if _unsupported(instance, x, plan):
            return original(x)
        return _run(instance, x, plan, rank, state["sp_process_group"], all_gather_var_shape)

    model.encode_temporal = MethodType(encode_temporal, model)
    model._omni_video_tile_balance_installed = True
    return True

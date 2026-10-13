# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni audio NPUGraphs with per-replay FIA sequence boundaries."""

import inspect

import torch
import torch_npu
from vllm_ascend.worker.encoder_acl_graph import (
    EncoderAclGraphManager,
    get_encoder_graph_params,
    set_encoder_forward_context,
    set_encoder_graph_params,
    update_encoder_graph_params,
    weak_ref_workspaces,
)

from vllm_omni.model_executor.models.qwen3_omni.audio_encoder_cudagraph import Qwen3OmniAudioEncoderCudaGraphs
from vllm_omni.model_executor.models.qwen3_omni.vision_encoder_cudagraph import pad_attention_bounds
from vllm_omni.worker.encoder_cudagraph import SingleReplayEncoderCudaGraphManager

# vllm-ascend keys its FIA registry by budget, or by (path, budget) once
# encoder graphs gained multiple paths.
_PATH_KEYED_REGISTRY = "path" in inspect.signature(update_encoder_graph_params).parameters


def _registry_key(budget: int):
    return ("default", budget) if _PATH_KEYED_REGISTRY else budget


def _rebind_fia(update_stream, budget: int, bounds: list[int], last_bounds: dict) -> None:
    """Refresh a replay's FIA sequence lengths unless the graph already holds them.

    A graph task keeps the arguments of its last update, so an unchanged
    replay skips one host FIA launch per layer. Its attention tasks still wait
    for one event per layer, which must be recorded on every replay.
    """
    lengths = tuple(bounds)
    if last_bounds.get(budget) == lengths:
        events = get_encoder_graph_params().events.get(_registry_key(budget), [])
        with torch.npu.stream(update_stream):
            for event in events:
                event.record(update_stream)
        return
    # Drop the entry first: a failed update must not leave stale lengths cached.
    last_bounds.pop(budget, None)
    with set_encoder_forward_context(budget, False, cu_seqlens_cpu=torch.tensor(bounds, dtype=torch.int32)):
        update_encoder_graph_params(update_stream, budget)
    last_bounds[budget] = lengths


class SingleReplayEncoderNpuGraphManager(SingleReplayEncoderCudaGraphManager, EncoderAclGraphManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_fia_bounds = {}

    def capture(self, graph_pool):
        self._last_fia_bounds = {}
        super().capture(graph_pool)

    def clear(self):
        super().clear()
        set_encoder_graph_params([])
        self.update_stream = None
        self._last_fia_bounds = {}

    @staticmethod
    def _host_cu_seqlens(mm_kwargs, total: int, max_sequences: int) -> list[int]:
        # Vision attention runs per frame. Build the FIA host lengths from the
        # host grid instead of reading the padded device buffer back, which
        # would block the host once per frame-chunk replay.
        grid = mm_kwargs["video_grid_thw" if "video_grid_thw" in mm_kwargs else "image_grid_thw"]
        grid = grid if isinstance(grid, list) else grid.tolist()
        bounds = [0]
        for t, h, w in grid:
            frame, start = int(h) * int(w), bounds[-1]
            bounds.extend(start + frame * (i + 1) for i in range(int(t)))
        return pad_attention_bounds(bounds, total, max_sequences)

    def _run_budget_graph(self, mm_kwargs, token_budget, path="default", axis_keys=()):
        if axis_keys:
            raise NotImplementedError("Encoder ACL graphs with capture axes are not supported.")
        num_items = len(self._get_item_specs(mm_kwargs))
        graph_set = self._get_graph_set(path)
        if token_budget not in graph_set:
            self.graph_misses += num_items
            return None
        graph_meta = graph_set[token_budget]
        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs, self.max_batch_size, self.max_frames_per_batch, path
        )
        for key, buf in graph_meta.input_buffers.items():
            src = replay.values.get(key)
            if src is None:
                continue
            if src.ndim == 0:
                buf.copy_(src)
            else:
                self.config.padding_logics.get(key, self._copy_padded_buffer)(buf, src)
        bounds = self._host_cu_seqlens(
            mm_kwargs,
            graph_meta.input_buffers["pixel_values"].shape[0],
            graph_meta.input_buffers["cu_seqlens"].shape[0] - 1,
        )
        if self.update_stream is None:
            self.update_stream = torch.npu.Stream()
        # Without the host sync, the next FIA update must still wait until the
        # previous replay of this graph has consumed its sequence lengths.
        self.update_stream.wait_stream(torch.npu.current_stream())
        graph_meta.graph.replay()
        _rebind_fia(self.update_stream, token_budget, bounds, self._last_fia_bounds)
        self.graph_hits += num_items
        return graph_meta.output_buffer


class Qwen3OmniAudioEncoderNpuGraphs(Qwen3OmniAudioEncoderCudaGraphs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Vision uses positive token budgets. Audio needs separate entries for
        # every (CNN chunks, output tokens) pair, including equal token counts.
        keys = [*self.capture_shapes, *(("w", width) for width in self.short_widths)]
        self._budget_keys = {shape: -i for i, shape in enumerate(keys, start=1)}
        self.update_stream = None
        self._last_fia_bounds = {}

    def capture(self, pool):
        self._last_fia_bounds = {}
        # Conv2D must use ACLNN: the internal-format ACL-op path cannot be
        # captured. Restore the process setting after startup capture.
        allow_internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        try:
            torch.npu.config.allow_internal_format = False
            super().capture(pool)
        finally:
            torch.npu.config.allow_internal_format = allow_internal_format != b"disable"

    def _capture_graph(self, forward, pool, key):
        if get_encoder_graph_params() is None:
            set_encoder_graph_params([])
        params = get_encoder_graph_params()
        budget = self._budget_keys[key]
        registry_key = _registry_key(budget)
        params.events[registry_key] = []
        params.handles[registry_key] = []
        params.attn_params[registry_key] = []
        params.workspaces[registry_key] = None
        graph = torch.npu.NPUGraph()
        with set_encoder_forward_context(budget, True), torch.npu.graph(graph, pool=pool):
            output = forward()
        weak_ref_workspaces()
        return graph, output

    def _replay_graph(self, captured, key, boundaries):
        if self.update_stream is None:
            self.update_stream = torch.npu.Stream()
        # Enqueue this dependency before replay, which waits for FIA updates.
        # Waiting on the replay itself here would create a stream cycle.
        self.update_stream.wait_stream(torch.npu.current_stream())
        captured.graph.replay()
        _rebind_fia(self.update_stream, self._budget_keys[key], boundaries, self._last_fia_bounds)

    def clear(self):
        super().clear()
        params = get_encoder_graph_params()
        if params is not None:
            for budget in self._budget_keys.values():
                for mapping in (params.events, params.handles, params.attn_params, params.workspaces):
                    mapping.pop(_registry_key(budget), None)
        self.update_stream = None
        self._last_fia_bounds = {}

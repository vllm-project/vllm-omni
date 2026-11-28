# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CUDA-graph runner for the audex causal speech decoder's steady-state
streaming chunks.

The streaming decode is launch-bound, not compute-bound: one steady chunk
(``chunk_frames`` emit frames + ``lookahead_steps`` lookahead frames) costs
~500 kernel launches for a few microseconds of GPU work each. Steady-state
shapes are constant per chunk, so the whole per-chunk decode
(``AudexCausalSpeechDecoderModel.decode_cached``) is captured once per KV
bucket and replayed per chunk:

- The graph reads/writes a static shared KV workspace sized to ``max_frames``
  (per-bucket graphs use prefix views of the same buffer, so all buckets
  share one allocation). The stock attention mask ``key_pos <= input_pos``
  hides the padded region, so replay output is bit-identical to eager.
- The session's ``CausalCodecDecoderCache`` stays authoritative: valid KV is
  copied into the workspace before each replay (skipped when the workspace
  provably already holds this cache's state) and the new frames are appended
  back with the same ``torch.cat`` semantics as the eager path. Concurrent
  sessions therefore interleave safely.
- Inputs reach the graph through a ring of pinned host slots guarded by CUDA
  events: the host may run arbitrarily far ahead of the GPU, and overwriting
  a single pinned buffer would race with its pending H2D copy.

Any chunk the runner cannot serve (position past ``max_frames``, incomplete
lookahead, non-matching emit width) returns ``None`` and the caller falls
back to the eager session path, which is always correct.
"""

from __future__ import annotations

import weakref

import torch

from .modeling_audex_causal_speech_decoder import CausalCodecDecoderCache

_PINNED_SLOTS = 32


class _GraphCaptureCache:
    """``CausalCodecDecoderCache`` stand-in over the runner's workspace.

    Used to capture one steady chunk (and, inside the graph, to replay it):
    ``input_positions`` derives positions from a device-side ``pos_base``
    buffer, and ``update`` writes the new K/V into the workspace with
    ``index_copy_`` instead of growing a ``torch.cat`` cache.
    """

    def __init__(self, runner: AudexCausalSpeechDecoderGraphRunner, kv_views: list[tuple]):
        self.runner = runner
        self.kv_views = kv_views
        self.position = 0  # host-side bookkeeping; never read during replay

    def input_positions(self, length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device).unsqueeze(0) + self.runner.pos_base.view(1, 1)
        self._positions = positions
        return positions

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        index = self._positions.view(-1)
        k, v = self.kv_views[layer_idx]
        k.index_copy_(2, index, key)
        v.index_copy_(2, index, value)
        return k, v

    def advance(self, length: int) -> None:
        self.position += length


class AudexCausalSpeechDecoderGraphRunner:
    """Replays CUDA-graphed steady-state chunks for one decoder model.

    One runner is shared by all sessions of a stage; per-session state lives
    in each session's stock ``CausalCodecDecoderCache``.
    """

    def __init__(
        self,
        model,
        *,
        emit_frames: int,
        bucket_stride: int = 64,
        max_frames: int = 1024,
        device: torch.device | None = None,
    ):
        if emit_frames <= 0:
            raise ValueError(f"emit_frames must be positive, got {emit_frames}")
        if bucket_stride <= 0:
            raise ValueError(f"bucket_stride must be positive, got {bucket_stride}")
        if max_frames % bucket_stride != 0 or max_frames < emit_frames:
            raise ValueError(
                f"max_frames must be a multiple of bucket_stride and >= emit_frames, "
                f"got max_frames={max_frames}, bucket_stride={bucket_stride}, emit_frames={emit_frames}"
            )

        config = model.config
        self.model = model
        self.emit_frames = int(emit_frames)
        self.lookahead_steps = int(model.lookahead_steps)
        self.bucket_stride = int(bucket_stride)
        self.max_frames = int(max_frames)

        device = device or next(model.parameters()).device
        if device.type != "cuda":
            raise ValueError(f"CUDA graph runner requires a CUDA device, got {device}")

        # Graphs bake in the rope cache's storage pointer, and replays cannot
        # grow it lazily the way the eager path does when a position lands
        # past its end. Grow it once here — value-preserving, since rope
        # entries are index-deterministic — so every replay position of every
        # bucket is in-bounds. (The blocks share one rotary instance.)
        rotary = model.module.backbone.transformers[0].att.rotary_embed
        if rotary.cache.size(0) < self.max_frames:
            rotary.build_rope_cache(self.max_frames)

        head_dim = config.hidden_dim // config.heads
        shape = (1, config.heads, self.max_frames, head_dim)
        # [layer] -> (k, v) over the shared workspace; per-bucket graphs use
        # k[:, :, :bucket] views, so every bucket shares this allocation
        self.workspace: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.zeros(shape, device=device), torch.zeros(shape, device=device)) for _ in range(config.depth)
        ]

        self.static_idx = torch.zeros(1, self.emit_frames, 1, dtype=torch.long, device=device)
        self.static_look = (
            torch.zeros(1, self.lookahead_steps, 1, dtype=torch.long, device=device) if self.lookahead_steps else None
        )
        self.pos_base = torch.zeros(1, dtype=torch.long, device=device)

        self._pin_idx = [
            torch.zeros(1, self.emit_frames, 1, dtype=torch.long, pin_memory=True) for _ in range(_PINNED_SLOTS)
        ]
        self._pin_look = (
            [torch.zeros(1, self.lookahead_steps, 1, dtype=torch.long, pin_memory=True) for _ in range(_PINNED_SLOTS)]
            if self.lookahead_steps
            else [None] * _PINNED_SLOTS
        )
        self._pin_pos = [torch.zeros(1, dtype=torch.long, pin_memory=True) for _ in range(_PINNED_SLOTS)]
        self._events = [torch.cuda.Event() for _ in range(_PINNED_SLOTS)]
        self._slot = 0

        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        self._pool = None  # shared capture memory pool (handle from graph.pool())
        # workspace skip-invalidation state: the cache (weakref) whose KV the
        # workspace currently holds, and up to which position
        self._ws_owner: weakref.ReferenceType | None = None
        self._ws_upto = -1

        torch.accelerator.synchronize()
        for event in self._events:
            event.record()  # all slots initially free

    # ------------------------------------------------------------------
    # bucket math
    # ------------------------------------------------------------------
    def bucket_for(self, end_position: int) -> int | None:
        """KV bucket covering frames [0, end_position), or None past the cap."""
        if end_position > self.max_frames:
            return None
        stride = self.bucket_stride
        return max(stride, ((end_position + stride - 1) // stride) * stride)

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------
    def _capture(self, bucket: int) -> torch.cuda.CUDAGraph:
        kv_views = [(k[:, :, :bucket], v[:, :, :bucket]) for k, v in self.workspace]
        capture_cache = _GraphCaptureCache(self, kv_views)

        def body() -> torch.Tensor:
            vq_emb = self.model.audex_speech_token_embedder.get_output_from_indices(self.static_idx)
            lookahead_vq_emb = None
            if self.static_look is not None:
                lookahead_vq_emb = self.model.audex_speech_token_embedder.get_output_from_indices(self.static_look)
            return self.model.decode_cached(vq_emb, capture_cache, lookahead_vq_emb=lookahead_vq_emb)

        with torch.inference_mode():
            # pos_base may hold a stale position from an earlier replay; the
            # capture must write inside this bucket's views, so start at 0
            self.pos_base.zero_()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    body()
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            pool_kwargs = {"pool": self._pool} if self._pool is not None else {}
            with torch.cuda.graph(graph, **pool_kwargs):
                output = body()
            if self._pool is None:
                self._pool = graph.pool()

        # the warmup/capture runs wrote garbage into the workspace's first
        # frames; clear them (masked out anyway, but keep state clean)
        for k, v in kv_views:
            k.zero_()
            v.zero_()
        self._ws_owner = None
        self._ws_upto = -1
        self._graphs[bucket] = graph
        self._outputs[bucket] = output
        return graph

    # ------------------------------------------------------------------
    # steady-chunk decode
    # ------------------------------------------------------------------
    def decode(
        self,
        codes_emit: list[int],
        codes_look: list[int],
        cache: CausalCodecDecoderCache,
    ) -> torch.Tensor | None:
        """Decode one steady chunk, updating ``cache`` in place.

        Returns the chunk waveform as a view of a static graph output buffer
        (clone it before holding it past the next replay), or ``None`` when
        this chunk cannot be served (caller falls back to the eager path).
        """
        if len(codes_emit) != self.emit_frames:
            return None
        if self.lookahead_steps and len(codes_look) != self.lookahead_steps:
            return None
        position = cache.position
        end = position + self.emit_frames
        bucket = self.bucket_for(end)
        if bucket is None:
            return None
        graph = self._graphs.get(bucket)
        if graph is None:
            graph = self._capture(bucket)

        with torch.inference_mode():
            # Copy this session's valid KV into the shared workspace. Skip
            # only when the workspace provably already holds this exact
            # cache's state (same live cache object, same position, no eager
            # chunk in between: any eager decode advances the position or
            # switches sessions first).
            owner = self._ws_owner() if self._ws_owner is not None else None
            skip_copy_in = owner is cache and self._ws_upto == position
            if not skip_copy_in:
                for layer, (k_ws, v_ws) in enumerate(self.workspace):
                    kv = cache.key_values.get(layer)
                    if kv is not None and position:
                        k_ws[:, :, :position].copy_(kv[0])
                        v_ws[:, :, :position].copy_(kv[1])

            slot = self._slot
            event = self._events[slot]
            event.synchronize()  # this slot's previous H2D copy has executed
            self._pin_idx[slot][0, :, 0] = torch.as_tensor(codes_emit, dtype=torch.long)
            if self.static_look is not None:
                self._pin_look[slot][0, :, 0] = torch.as_tensor(codes_look, dtype=torch.long)
            self._pin_pos[slot][0] = position
            self.static_idx.copy_(self._pin_idx[slot], non_blocking=True)
            if self.static_look is not None:
                self.static_look.copy_(self._pin_look[slot], non_blocking=True)
            self.pos_base.copy_(self._pin_pos[slot], non_blocking=True)
            graph.replay()
            event.record()
            self._slot = (slot + 1) % _PINNED_SLOTS

            # Append the new frames to the session's cache with the same
            # semantics as CausalCodecDecoderCache.update on the eager path.
            for layer, (k_ws, v_ws) in enumerate(self.workspace):
                new_k = k_ws[:, :, position:end]
                new_v = v_ws[:, :, position:end]
                kv = cache.key_values.get(layer)
                if kv is None:
                    cache.key_values[layer] = (new_k.clone(), new_v.clone())
                else:
                    cache.key_values[layer] = (
                        torch.cat([kv[0], new_k], dim=2),
                        torch.cat([kv[1], new_v], dim=2),
                    )
            cache.position = end
            self._ws_owner = weakref.ref(cache)
            self._ws_upto = end
            return self._outputs[bucket]

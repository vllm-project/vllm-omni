# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched streaming AudioVAE decode for Ming3 Stage 1.

The shared AudioVAE decoder streams one request at a time through a
Transformers Qwen2Model with a per-request DynamicCache, so a step with N
active streams launches N full 24-layer passes. Here every request that has
new latents in a step goes through one batched pass: each row attends to its
own cached window (the last ``sliding_window - 1`` positions, kept in a
slot-indexed pool) plus its new tokens, under the decoder's causal sliding-
window mask. The linear upsampler and the ISTFT head keep their per-request
streaming state and run per row.

On CUDA in half precision the rows are packed along the token dimension and
run through :class:`PackedQwen2` (fused norms, RoPE and SiLU, FlashAttention):
each stream's keys live in a ring of paged KV blocks, since attention reaches
back only one window, and the pass replays from CUDA graphs captured per
packed-token bucket. Elsewhere the rows are padded to one length and the
layer stack replays per (batch, length) bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from vllm.platforms import current_platform

from vllm_omni.utils.device_copy import DeviceStager

from .packed_qwen2 import PackedQwen2, PagedKV

# Rows per pass and decoder tokens per row (four per latent): the ramp chunks
# and steady 25-latent chunks fall inside these buckets.
GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
GRAPH_LENGTHS = (16, 32, 64, 128)
# Packed path: tokens of all rows per pass.
PACKED_GRAPH_TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048)
# Decoder tokens one stream may reach (four per latent).
MAX_STREAM_POSITIONS = 16384


@dataclass
class _StreamState:
    slot: int
    position: int = 0  # decoder tokens consumed so far (absolute RoPE position)
    cached: int = 0  # valid tokens in the slot's window (<= window - 1)
    upsample: Any = None
    audio_buffer: torch.Tensor | None = None
    window_buffer: torch.Tensor | None = None


class BatchedStreamingVAEDecoder:
    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        max_streams: int,
        kv_block_size: int = 64,
        ring_blocks: int = 4,
    ):
        self.decoder = decoder
        qwen = decoder.decoder
        config = qwen.config
        if not getattr(config, "use_sliding_window", False) or config.sliding_window is None:
            raise ValueError("Batched Ming3 VAE decode expects a sliding-window decoder")
        if any(layer_type != "sliding_attention" for layer_type in getattr(config, "layer_types", [])):
            raise ValueError("Batched Ming3 VAE decode expects sliding attention in every layer")
        self.layers = qwen.layers
        self.norm = qwen.norm
        self.rotary = qwen.rotary_emb
        self.window = int(config.sliding_window)
        self.cache_len = self.window - 1
        attention = self.layers[0].self_attn
        self.head_dim = attention.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = attention.scaling
        parameter = next(decoder.parameters())
        packed = parameter.device.type == "cuda" and parameter.dtype in (torch.float16, torch.bfloat16)
        self._packed: PackedQwen2 | None = PackedQwen2(qwen, MAX_STREAM_POSITIONS) if packed else None
        self._max_streams = max_streams
        # The extra slot takes the window writes of batch-padding rows in a
        # graph replay; it is never handed to a stream.
        self._scratch_slot = max_streams
        if self._packed is None:
            self.kv = torch.zeros(
                (len(self.layers), 2, max_streams + 1, self.num_kv_heads, self.cache_len, self.head_dim),
                device=parameter.device,
                dtype=parameter.dtype,
            )
        else:
            # Stream s owns physical blocks [s * ring_blocks, (s + 1) * ring_blocks); logical
            # block j maps to block j % ring_blocks, so a pass may add up to one ring
            # minus one window of tokens per stream.
            if ring_blocks * kv_block_size < self.window + 1:
                raise ValueError("The KV ring must hold more than one attention window")
            self._kv_block = kv_block_size
            self._ring = ring_blocks
            self._max_row_tokens = ring_blocks * kv_block_size - self.cache_len
            keys, values = self._packed.new_paged_caches(max_streams * ring_blocks, kv_block_size)
            logical = torch.arange(-(-MAX_STREAM_POSITIONS // kv_block_size), device=parameter.device)
            first = torch.arange(max_streams, device=parameter.device)[:, None] * ring_blocks
            block_table = (first + logical[None, :] % ring_blocks).to(torch.int32)
            self._paged = (keys, values, block_table)
            self._token_stager = DeviceStager(torch.long, capacity=2 * PACKED_GRAPH_TOKENS[-1])
            self._row_stager = DeviceStager(torch.int32, capacity=2 * max_streams + 1)
            self._packed_graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
            self._packed_static: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._free = list(range(max_streams - 1, -1, -1))
        self._streams: dict[str, _StreamState] = {}
        self._meta_stager = DeviceStager(torch.long, capacity=4 * GRAPH_BATCH_SIZES[-1] + 4 * max_streams)
        # (batch, length) -> (graph, static output); inputs live in the shared static buffers.
        self._graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self._graph_batches: list[int] = []
        self._graph_lengths: list[int] = []
        self._graph_hidden: torch.Tensor | None = None
        self._graph_meta: torch.Tensor | None = None

    @torch.inference_mode()
    def capture_graphs(
        self,
        batch_sizes: tuple[int, ...] = GRAPH_BATCH_SIZES,
        lengths: tuple[int, ...] = GRAPH_LENGTHS,
    ) -> None:
        """Capture the layer stack for each (batch, length) bucket; other shapes run eagerly."""
        parameter = next(self.decoder.parameters())
        if parameter.device.type != "cuda":
            return
        if self._packed is not None:
            self._capture_packed_graphs()
            return
        batch_sizes = tuple(sorted({size for size in batch_sizes if size <= self._scratch_slot}))
        lengths = tuple(sorted(set(lengths)))
        hidden_size = self.norm.weight.shape[0]
        self._graph_hidden = torch.zeros(
            (batch_sizes[-1], lengths[-1], hidden_size), device=parameter.device, dtype=parameter.dtype
        )
        # Rows of positions, cached lengths, new lengths and slots; capture
        # writes only the scratch slot.
        self._graph_meta = torch.zeros((4, batch_sizes[-1]), device=parameter.device, dtype=torch.long)
        self._graph_meta[3].fill_(self._scratch_slot)
        pool = current_platform.get_global_graph_pool()
        stream = torch.cuda.Stream(device=parameter.device)
        for batch in reversed(batch_sizes):
            for length in reversed(lengths):
                hidden = self._graph_hidden[:batch, :length]
                meta = self._graph_meta[:, :batch]
                # Warm up per shape so library plans exist before capture.
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self._stack(hidden, meta)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    output = self._stack(hidden, meta)
                self._graphs[(batch, length)] = (graph, output)
        self._graph_batches = list(batch_sizes)
        self._graph_lengths = list(lengths)

    def _capture_packed_graphs(self, tokens: tuple[int, ...] = PACKED_GRAPH_TOKENS) -> None:
        parameter = next(self.decoder.parameters())
        rows = self._max_streams
        hidden = torch.zeros((tokens[-1], self.norm.weight.shape[0]), device=parameter.device, dtype=parameter.dtype)
        # Token rows: positions, cache slots (negative: no write). Row columns:
        # cumulative query lengths, then keys per stream after the pass.
        token_meta = torch.zeros((2, tokens[-1]), device=parameter.device, dtype=torch.long)
        token_meta[1].fill_(-1)
        row_meta = torch.zeros(2 * rows + 1, device=parameter.device, dtype=torch.int32)
        self._packed_static = (hidden, token_meta, row_meta)
        pool = current_platform.get_global_graph_pool()
        stream = torch.cuda.Stream(device=parameter.device)
        for count in sorted(tokens, reverse=True):
            # Capture runs with no sequences: nothing is read or written.
            def run(count=count):
                return self._packed_stack(
                    hidden[:count], token_meta[:, :count], row_meta, min(count, self._max_row_tokens)
                )

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                output = run()
            self._packed_graphs[count] = (graph, output)

    def _packed_stack(
        self, hidden: torch.Tensor, token_meta: torch.Tensor, row_meta: torch.Tensor, max_row_tokens: int
    ) -> torch.Tensor:
        rows = self._max_streams
        keys, values, block_table = self._paged
        kv = PagedKV(
            keys,
            values,
            slot_mapping=token_meta[1],
            seqused_k=row_meta[rows + 1 :],
            block_table=block_table,
            max_seqlen_k=MAX_STREAM_POSITIONS,
        )
        return self._packed(hidden, token_meta[0], row_meta[: rows + 1], max_row_tokens, kv)

    def _forward_packed(self, states: list[_StreamState], hiddens: list[torch.Tensor]) -> list[torch.Tensor]:
        parts: list[list[torch.Tensor]] = [[] for _ in hiddens]
        done = [0] * len(hiddens)
        pending = list(range(len(hiddens)))
        while pending:
            # A pass may add at most one ring minus one window per stream.
            take = [(row, hiddens[row][done[row] : done[row] + self._max_row_tokens]) for row in pending]
            for (row, _), output in zip(take, self._packed_pass([(states[row], piece) for row, piece in take])):
                parts[row].append(output)
                done[row] += output.shape[0]
            pending = [row for row in pending if done[row] < hiddens[row].shape[0]]
        return [chunks[0] if len(chunks) == 1 else torch.cat(chunks) for chunks in parts]

    def _packed_pass(self, take: list[tuple[_StreamState, torch.Tensor]]) -> list[torch.Tensor]:
        """Run one packed pass over ``(stream, new tokens)``; return each stream's hidden states."""
        rows = self._max_streams
        # FlashAttention reads the packed tokens in sequence (stream slot) order.
        order = sorted(range(len(take)), key=lambda index: take[index][0].slot)
        lengths = np.zeros(rows, dtype=np.int64)
        used = np.zeros(rows, dtype=np.int64)
        positions, slots = [], []
        for index in order:
            state, piece = take[index]
            count, start = piece.shape[0], state.position
            if start + count > MAX_STREAM_POSITIONS:
                raise RuntimeError(f"Ming3 AudioVAE stream exceeds {MAX_STREAM_POSITIONS} decoder tokens")
            lengths[state.slot], used[state.slot] = count, start + count
            position = np.arange(start, start + count)
            block = state.slot * self._ring + (position // self._kv_block) % self._ring
            positions.append(position)
            slots.append(block * self._kv_block + position % self._kv_block)
        total = int(lengths.sum())
        hidden = torch.cat([take[index][1] for index in order])
        max_row_tokens = int(lengths.max())
        bucket = next((count for count in sorted(self._packed_graphs) if count >= total), None)
        width = bucket or total
        token_meta = np.zeros((2, width), dtype=np.int64)
        token_meta[1].fill(-1)
        token_meta[0, :total] = np.concatenate(positions)
        token_meta[1, :total] = np.concatenate(slots)
        row_meta = np.concatenate(([0], np.cumsum(lengths), used))
        device = hidden.device
        token_view = self._token_stager(token_meta.reshape(-1), device).view(2, width)
        row_view = self._row_stager(row_meta, device)
        if bucket is None:
            output = self._packed_stack(hidden, token_view.clone(), row_view.clone(), max_row_tokens)
        else:
            assert self._packed_static is not None
            static_hidden, static_tokens, static_rows = self._packed_static
            graph, static_output = self._packed_graphs[bucket]
            static_hidden[:total].copy_(hidden)
            static_tokens[:, :bucket].copy_(token_view)
            static_rows.copy_(row_view)
            graph.replay()
            # A long row takes several passes; each replay rewrites the static output.
            output = static_output[:total].clone()

        results: list[torch.Tensor | None] = [None] * len(take)
        for index, piece in zip(order, output.split([take[index][1].shape[0] for index in order]), strict=True):
            state = take[index][0]
            state.position += piece.shape[0]
            results[index] = piece
        return results  # type: ignore[return-value]

    def _graph_bucket(self, batch: int, length: int) -> tuple[int, int] | None:
        graph_batch = next((size for size in self._graph_batches if size >= batch), None)
        graph_length = next((size for size in self._graph_lengths if size >= length), None)
        if graph_batch is None or graph_length is None:
            return None
        return graph_batch, graph_length

    def release(self, request_id: str) -> None:
        state = self._streams.pop(request_id, None)
        if state is not None:
            self._free.append(state.slot)

    @torch.inference_mode()
    def decode(self, items: list[tuple[str, torch.Tensor, bool]]) -> list[torch.Tensor]:
        """Decode ``(request_id, latents [Patches, PatchSize, Dim], finished)``; return flat waveforms."""
        decoder = self.decoder
        rows: list[tuple[int, _StreamState, torch.Tensor]] = []
        waveforms: list[torch.Tensor | None] = [None] * len(items)
        # One projection for every row's latents.
        flat = [latent.reshape(-1, latent.shape[-1]) for _, latent, _ in items]
        projected = decoder.fc1(torch.cat(flat)).split([rows.shape[0] for rows in flat])
        for index, ((request_id, latent, finished), hidden) in enumerate(zip(items, projected, strict=True)):
            state = self._streams.get(request_id)
            if state is None:
                state = _StreamState(slot=self._free.pop())
                self._streams[request_id] = state
            hidden, state.upsample = decoder.upsampling(hidden.unsqueeze(0), state=state.upsample, is_last=finished)
            if hidden is None or hidden.shape[1] == 0:
                waveforms[index] = latent.new_zeros((0,), dtype=torch.float32)
            else:
                rows.append((index, state, hidden[0]))

        if rows:
            # One padded pass for every row: the pass is launch-bound, so a
            # second pass for short rows costs more than their padding.
            forward = self._forward_rows if self._packed is None else self._forward_packed
            outputs = forward([state for _, state, _ in rows], [hidden for _, _, hidden in rows])
            for (index, state, _), hidden in zip(rows, outputs, strict=True):
                finished = items[index][2]
                audio, _, state.audio_buffer, state.window_buffer = decoder.head(
                    hidden.unsqueeze(0),
                    streaming=True,
                    audio_buffer=state.audio_buffer,
                    window_buffer=state.window_buffer,
                    last_chunk=finished,
                )
                waveforms[index] = audio.reshape(-1).to(torch.float32)
        for request_id, _, finished in items:
            if finished:
                self.release(request_id)
        return waveforms

    def _forward_rows(self, states: list[_StreamState], hiddens: list[torch.Tensor]) -> list[torch.Tensor]:
        device = hiddens[0].device
        lengths = [hidden.shape[0] for hidden in hiddens]
        batch, max_len = len(hiddens), max(lengths)
        hidden = torch.nn.utils.rnn.pad_sequence(hiddens, batch_first=True)
        rows = [
            (state.position, state.cached, length, state.slot) for state, length in zip(states, lengths, strict=True)
        ]
        bucket = self._graph_bucket(batch, max_len)
        if bucket is None:
            # Small per-row integers reach the device without a host sync; one
            # copy keeps the per-layer reads off host memory.
            hidden = self._stack(hidden, self._meta_stager(rows, device).T.clone())
        else:
            graph, output = self._graphs[bucket]
            # Batch-padding rows have no tokens and write the scratch slot.
            rows += [(0, 0, 0, self._scratch_slot)] * (bucket[0] - batch)
            assert self._graph_meta is not None and self._graph_hidden is not None
            self._graph_meta[:, : bucket[0]].copy_(self._meta_stager(rows, device).T)
            self._graph_hidden[:batch, :max_len].copy_(hidden)
            graph.replay()
            hidden = output

        for state, length in zip(states, lengths, strict=True):
            state.position += length
            state.cached = min(self.cache_len, state.cached + length)
        return [hidden[row, :length] for row, length in enumerate(lengths)]

    def _stack(self, hidden: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Run the decoder layers over padded rows; ``meta`` holds positions, cached and new lengths, slots."""
        device = hidden.device
        batch, max_len = hidden.shape[0], hidden.shape[1]
        positions, cached, row_lengths, slots = meta[0], meta[1], meta[2], meta[3]

        offsets = torch.arange(max_len, device=device)
        cos, sin = self.rotary(hidden, positions[:, None] + offsets[None, :])
        # Keys are [cached window (right-aligned), new tokens]; relative positions
        # are -cache_len..-1 for the window and 0..max_len-1 for new tokens.
        key_rel = torch.arange(-self.cache_len, max_len, device=device)
        query = offsets[:, None]
        allowed = (key_rel[None, :] <= query) & (query - key_rel[None, :] < self.window)
        valid_key = (key_rel[None, :] >= -cached[:, None]) & (key_rel[None, :] < row_lengths[:, None])
        mask = allowed[None] & valid_key[:, None, :]
        # Padded query rows attend to themselves only, so softmax stays finite.
        padded_query = offsets[None, :] >= row_lengths[:, None]
        mask = mask | (padded_query[:, :, None] & (key_rel[None, None, :] == query[None]))
        mask = mask[:, None]
        # New window: the last cache_len of [window, new tokens] for each row.
        keep = row_lengths[:, None] + torch.arange(self.cache_len, device=device)[None, :]

        for layer_index, layer in enumerate(self.layers):
            attention = layer.self_attn
            residual = hidden
            normed = layer.input_layernorm(hidden)
            query_states = attention.q_proj(normed).view(batch, max_len, -1, self.head_dim).transpose(1, 2)
            key_states = attention.k_proj(normed).view(batch, max_len, -1, self.head_dim).transpose(1, 2)
            value_states = attention.v_proj(normed).view(batch, max_len, -1, self.head_dim).transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            keys = torch.cat([self.kv[layer_index, 0, slots], key_states], dim=2)
            values = torch.cat([self.kv[layer_index, 1, slots], value_states], dim=2)
            # vLLM disables cuDNN attention, and no other fused kernel takes a
            # boolean mask with grouped KV heads; expanded heads keep this on
            # the memory-efficient kernel instead of the fp32 math fallback.
            attended = F.scaled_dot_product_attention(
                query_states,
                keys.repeat_interleave(self.kv_groups, dim=1),
                values.repeat_interleave(self.kv_groups, dim=1),
                attn_mask=mask,
                scale=self.scaling,
            )
            hidden = residual + attention.o_proj(attended.transpose(1, 2).reshape(batch, max_len, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            gather = keep[:, None, :, None].expand(-1, self.num_kv_heads, -1, self.head_dim)
            self.kv[layer_index, 0, slots] = torch.gather(keys, 2, gather)
            self.kv[layer_index, 1, slots] = torch.gather(values, 2, gather)
        return self.norm(hidden)


__all__ = ["BatchedStreamingVAEDecoder"]

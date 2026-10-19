# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio tower graphs with host-prepared chunk and attention metadata."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d

logger = init_logger(__name__)


def audio_chunk_metadata(lengths: list[int], window: int, infer_window: int):
    """Match the eager tower's CNN chunks and per-audio attention windows."""
    chunks = []
    outputs = []
    for length in lengths:
        if length <= 0:
            raise ValueError("Audio feature lengths must be positive")
        full, tail = divmod(length, window)
        current = [window] * full + ([tail] if tail else [])
        chunks.extend(current)
        outputs.append(sum((n + 7) // 8 for n in current))
    width = max(chunks)
    cnn_width = (width + 7) // 8
    indices = [i * cnn_width + j for i, length in enumerate(chunks) for j in range((length + 7) // 8)]
    attn_window = cnn_width * (infer_window // window)
    boundaries = [0]
    for length in outputs:
        while length:
            step = min(length, attn_window)
            boundaries.append(boundaries[-1] + step)
            length -= step
    return chunks, indices, boundaries, outputs


def _copy_chunked_features(
    features: torch.Tensor, lengths: list[int], chunks: list[int], destination: torch.Tensor
) -> None:
    """Pack clips into fixed-width CNN rows, batching long runs of windows."""
    count, mel_bins, window = destination.shape
    if features.shape[1] == count * window:
        destination.copy_(features.reshape(mel_bins, count, window).transpose(0, 1))
        return

    # For short-clip batches, per-clip tail clearing/copying costs more than
    # pad_sequence's single fill and per-window copies. Keep that path when
    # there are at most two windows per clip on average.
    if count <= 2 * len(lengths):
        padded = torch.nn.utils.rnn.pad_sequence(features.T.split(chunks), batch_first=True).transpose(1, 2)
        destination.copy_(padded)
        return

    # Copy whole runs of full windows, not one kernel per window. Only each
    # clip's final row needs padding; clear it before copying the live tail.
    offset = row = 0
    for length in lengths:
        full, tail = divmod(length, window)
        end = offset + full * window
        if full:
            destination[row : row + full].copy_(features[:, offset:end].reshape(mel_bins, full, window).transpose(0, 1))
        row += full
        if tail:
            destination[row, :, tail:].zero_()
            destination[row, :, :tail].copy_(features[:, end : end + tail])
            row += 1
        offset += length


def replay_audio_inputs(chunks, indices, boundaries, cnn_width, cu_capacity, attn_window):
    """Replay one full-width graph for every tail of this chunk count.

    Short rows are already zero-padded to the CNN window, matching eager
    ``pad_sequence``. Gathering the padded CNN positions keeps the captured
    GEMM shape fixed. Those positions are trailing attention sequences, each no
    longer than the captured ``max_seqlen``, so attention cannot mix them with
    live tokens. GEMM shape changes can still affect floating-point rounding.
    The caller discards the extra positions.
    """
    full_tokens = len(chunks) * cnn_width
    live_tokens = len(indices)
    if live_tokens == full_tokens:
        planned = list(boundaries)
        replay_indices = list(indices)
    else:
        if attn_window <= 0:
            return None
        pad_indices = [
            row * cnn_width + col for row, length in enumerate(chunks) for col in range((length + 7) // 8, cnn_width)
        ]
        if live_tokens + len(pad_indices) != full_tokens or boundaries[-1] != live_tokens:
            return None
        replay_indices = list(indices) + pad_indices
        planned = list(boundaries)
        cursor = live_tokens
        while cursor < full_tokens:
            cursor = min(full_tokens, cursor + attn_window)
            planned.append(cursor)
    expected_end = live_tokens if live_tokens == full_tokens else full_tokens
    if not planned or planned[-1] != expected_end or len(planned) > cu_capacity:
        return None
    planned.extend([planned[-1]] * (cu_capacity - len(planned)))
    return replay_indices, planned, live_tokens


def audio_forward_prepared(tower, features, indices, cu_seqlens, max_seqlen):
    """The full eager CNN and transformer, with fixed-shape gather indices."""
    embeddings = []
    for chunk in features.unsqueeze(1).split(tower.conv_chunksize, dim=0):
        hidden = F.gelu(tower.conv2d1(chunk))
        hidden = F.gelu(tower.conv2d2(hidden))
        embeddings.append(F.gelu(tower.conv2d3(hidden)))
    hidden = torch.cat(embeddings, dim=0) if len(embeddings) > 1 else embeddings[0]
    b, c, f, t = hidden.shape
    hidden, _ = tower.conv_out(hidden.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
    hidden = hidden + tower.positional_embedding.positional_embedding[:t].to(hidden.dtype).unsqueeze(0)
    hidden = hidden.flatten(0, 1).index_select(0, indices)
    for layer in tower.layers:
        hidden = layer(hidden, cu_seqlens, max_seqlen)
    hidden = tower.ln_post(hidden)
    hidden, _ = tower.proj1(hidden)
    hidden = tower.act(hidden)
    hidden, _ = tower.proj2(hidden)
    return hidden


@dataclass
class AudioGraph:
    graph: Any
    features: torch.Tensor
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    output: torch.Tensor
    metadata_key: tuple[int, ...] | None = None


class Qwen3OmniAudioEncoderCudaGraphs:
    """Capture one full-width CNN/transformer graph per chunk count.

    A short tail row replays that graph. Its padded CNN positions are separate
    attention sequences and are dropped, avoiding a graph per tail width.
    Padded GEMM dimensions may change rounding. One clip shorter than the CNN
    window replays a graph captured at that exact width. A batch of several
    such clips, or a chunk count that was not captured, stays eager.
    """

    def __init__(self, tower, budgets=None, extra_shapes=(), short_widths=None):
        self.tower = tower
        # 1..64 chunks covers a 60-second clip, including its one-frame tail
        # (61 chunks). Larger batches, such as four 30-second clips, stay eager:
        # those shapes dominated the captured workspace.
        self.budgets = tuple(range(1, 65)) if budgets is None else tuple(budgets)
        cnn_width = (tower.n_window * 2 + 7) // 8
        shapes = {(count, count * cnn_width) for count in self.budgets}
        shapes.update(extra_shapes)
        self.capture_shapes = sorted(shapes)
        # A clip shorter than one CNN window has a narrower conv than the
        # full-width graph. Replaying that graph zero-pads the row and changes
        # the tail. Capture each narrower width for a single clip; mixed short
        # clips still run eager. Explicit budgets keep the previous set so
        # tests do not capture 99 extra graphs unless they ask for them.
        window = tower.n_window * 2
        if short_widths is None:
            self.short_widths = tuple(range(1, window)) if budgets is None else ()
        else:
            self.short_widths = tuple(short_widths)
        self.graphs = {}
        self.graph_hits = 0
        self.graph_misses = 0
        self._last_lengths: tuple[int, ...] | None = None
        self._last_replay: tuple | None = None

    def num_graphs_to_capture(self) -> int:
        return len(self.capture_shapes) + len(self.short_widths)

    def capture(self, pool):
        self._last_lengths = None
        self._last_replay = None
        tower = self.tower
        weight = tower.conv2d1.weight
        window = tower.n_window * 2
        cnn_width = (window + 7) // 8
        max_seqlen = torch.tensor(cnn_width * (tower.n_window_infer // window), dtype=torch.int32)
        if not self.capture_shapes and not self.short_widths:
            return
        # Replays are serialized on the runner stream and outputs are cloned.
        # All shapes can therefore use prefixes of one persistent input arena.
        # Keep metadata separate: each graph caches its last clip boundaries.
        feature_buffer = torch.zeros(
            max((count for count, _ in self.capture_shapes), default=1),
            tower.num_mel_bins,
            window,
            device=weight.device,
            dtype=weight.dtype,
        )
        # Capture the largest shapes first so smaller graphs reuse the pool
        # instead of accumulating progressively larger convolution workspaces.
        for count, tokens in reversed(self.capture_shapes):
            features = feature_buffer[:count]
            indices = torch.arange(tokens, device=weight.device)
            attn_window = cnn_width * (tower.n_window_infer // window)
            boundaries = list(range(0, tokens, attn_window)) + [tokens] if attn_window > 0 else [0, tokens]
            # Room for one live sequence per chunk plus padded tail positions
            # split into captured attention windows.
            max_pad = max(count - 1, 0) * max(cnn_width - 1, 0)
            pad_steps = (max_pad + attn_window - 1) // attn_window if attn_window > 0 and max_pad else 0
            capacity = max(count + 2, count + 1 + pad_steps, len(boundaries))
            boundaries += [tokens] * (capacity - len(boundaries))
            cu = torch.tensor(boundaries, device=weight.device, dtype=torch.int32)
            with torch.inference_mode():
                for _ in range(2):
                    audio_forward_prepared(tower, features, indices, cu, max_seqlen)
                graph, output = self._capture_graph(
                    lambda: audio_forward_prepared(tower, features, indices, cu, max_seqlen), pool, (count, tokens)
                )
            self.graphs[count, tokens] = AudioGraph(graph, features, indices, cu, output)
        for width in self.short_widths:
            tokens = (width + 7) // 8
            features = feature_buffer[:1, :, :width]
            indices = torch.arange(tokens, device=weight.device)
            cu = torch.tensor([0, tokens], device=weight.device, dtype=torch.int32)
            with torch.inference_mode():
                for _ in range(2):
                    audio_forward_prepared(tower, features, indices, cu, max_seqlen)
                graph, output = self._capture_graph(
                    lambda features=features, indices=indices, cu=cu: audio_forward_prepared(
                        tower, features, indices, cu, max_seqlen
                    ),
                    pool,
                    ("w", width),
                )
            self.graphs["w", width] = AudioGraph(graph, features, indices, cu, output)
        logger.info(
            "Captured %d Qwen3-Omni audio encoder graphs (%d full-width, %d single-clip short widths)",
            len(self.graphs),
            len(self.capture_shapes),
            len(self.short_widths),
        )

    def clear(self):
        self.graphs.clear()
        self._last_lengths = None
        self._last_replay = None

    def _capture_graph(self, forward, pool, key):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            output = forward()
        return graph, output

    def _replay_graph(self, captured, key, boundaries):
        captured.graph.replay()

    def _prepare_replay(self, lengths: list[int]):
        tower = self.tower
        window = tower.n_window * 2
        chunks, indices, boundaries, outputs = audio_chunk_metadata(lengths, window, tower.n_window_infer)
        count = len(chunks)
        cnn_width = (window + 7) // 8
        live_tokens = len(indices)
        if max(chunks) != window:
            # Eager's conv sees the longest row, not a full window of zeros.
            # Only a single clip has a captured exact-width graph.
            if len(chunks) == 1 and ("w", chunks[0]) in self.graphs:
                captured = self.graphs["w", chunks[0]]
                if len(boundaries) > captured.cu_seqlens.numel():
                    self.graph_misses += len(lengths)
                    return None
                planned = list(boundaries)
                planned += [live_tokens] * (captured.cu_seqlens.numel() - len(planned))
                return chunks, indices, planned, outputs, ("w", chunks[0]), live_tokens
            self.graph_misses += len(lengths)
            if self.graph_misses <= 3 or self.graph_misses % 100 == 0:
                logger.info(
                    "Qwen3-Omni audio graph fallback: chunks=%d width=%d hits=%d misses=%d",
                    len(chunks),
                    max(chunks) if chunks else 0,
                    self.graph_hits,
                    self.graph_misses,
                )
            return None
        key = (count, live_tokens)
        # Tail rows have fewer output tokens than the full-width graph. Replay
        # that graph unless an exact tail shape was captured explicitly.
        use_full_width = key not in self.graphs
        if use_full_width:
            key = (count, count * cnn_width)
        if key not in self.graphs:
            self.graph_misses += len(lengths)
            if self.graph_misses <= 3 or self.graph_misses % 100 == 0:
                logger.info(
                    "Qwen3-Omni audio graph fallback: chunks=%d width=%d hits=%d misses=%d",
                    len(chunks),
                    max(chunks) if chunks else 0,
                    self.graph_hits,
                    self.graph_misses,
                )
            return None
        captured = self.graphs[key]
        if use_full_width:
            attn_window = cnn_width * (tower.n_window_infer // window)
            planned = replay_audio_inputs(
                chunks, indices, boundaries, cnn_width, captured.cu_seqlens.numel(), attn_window
            )
            if planned is None:
                self.graph_misses += len(lengths)
                return None
            indices, boundaries, live_tokens = planned
        elif len(boundaries) > captured.cu_seqlens.numel():
            self.graph_misses += len(lengths)
            return None
        else:
            boundaries = list(boundaries)
            boundaries += [live_tokens] * (captured.cu_seqlens.numel() - len(boundaries))
        return chunks, indices, boundaries, outputs, key, live_tokens

    def execute(self, input_features, lengths: list[int]):
        if not self.graphs:
            return None
        metadata_key = tuple(lengths)
        # Repeated lengths need neither token-by-token host planning nor H2D
        # metadata uploads. Keep just one host plan so arbitrary request lengths
        # cannot grow a cache; graph-local metadata_key still guards uploads
        # after another request has used the same graph.
        if self._last_lengths == metadata_key:
            prepared = self._last_replay
        else:
            prepared = self._prepare_replay(lengths)
            if prepared is not None:
                self._last_lengths = metadata_key
                self._last_replay = prepared
        if prepared is None:
            return None
        chunks, indices, boundaries, outputs, key, live_tokens = prepared
        captured = self.graphs[key]
        _copy_chunked_features(input_features, lengths, chunks, captured.features)
        if captured.metadata_key != metadata_key:
            # Upload straight into the stable graph buffers. A temporary device
            # tensor followed by copy_ adds an allocation and a D2D copy for
            # each metadata array on every changed-length replay.
            async_tensor_h2d(indices, out=captured.indices)
            async_tensor_h2d(boundaries, out=captured.cu_seqlens)
            captured.metadata_key = metadata_key
        self._replay_graph(captured, key, boundaries)
        self.graph_hits += len(lengths)
        if self.graph_hits <= 3 or self.graph_hits % 64 == 0:
            logger.info("Qwen3-Omni audio encoder graphs: hits=%d misses=%d", self.graph_hits, self.graph_misses)
        # Request embeddings outlive the next replay in the encoder cache.
        return captured.output[:live_tokens].clone().split(outputs)

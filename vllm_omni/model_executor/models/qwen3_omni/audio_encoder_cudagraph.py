# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio tower graphs with host-prepared chunk and attention metadata."""

from dataclasses import dataclass

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
    graph: torch.cuda.CUDAGraph
    features: torch.Tensor
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    output: torch.Tensor


class Qwen3OmniAudioEncoderCudaGraphs:
    """Capture exact CNN/output sizes without padding live matrix dimensions.

    Clip boundaries and tail gather indices remain replay inputs. Unsupported
    shape signatures and shorter CNN widths use the original eager tower.
    """

    def __init__(self, tower, budgets=None, extra_shapes=()):
        self.tower = tower
        self.budgets = tuple(range(1, 65)) if budgets is None else tuple(budgets)
        cnn_width = (tower.n_window * 2 + 7) // 8
        shapes = {(count, count * cnn_width) for count in self.budgets}
        # Preserve both CNN and projection/GEMM dimensions. Padding these
        # dimensions can amplify BF16 rounding at sensitive audio tokens.
        # Short clips and the common 30/60-second processor tails are captured
        # explicitly; other lengths keep the original eager arithmetic.
        for count in (1, 2, 3, 4, 31, 61):
            if count in self.budgets:
                shapes.update((count, (count - 1) * cnn_width + tail) for tail in range(1, cnn_width))
        if budgets is None:
            # Batches of up to four 30-second clips, or two 60-second clips,
            # may each have the processor's extra one-frame tail.
            for full in (30, 60, 90, 120):
                for tails in range(full // 30 + 1):
                    shapes.add((full + tails, full * cnn_width + tails))
        shapes.update(extra_shapes)
        self.capture_shapes = sorted(shapes)
        self.graphs = {}
        self.graph_hits = 0
        self.graph_misses = 0

    def capture(self, pool):
        tower = self.tower
        weight = tower.conv2d1.weight
        window = tower.n_window * 2
        cnn_width = (window + 7) // 8
        max_seqlen = torch.tensor(cnn_width * (tower.n_window_infer // window), dtype=torch.int32)
        # Capture the largest shapes first so smaller graphs reuse the pool
        # instead of accumulating progressively larger convolution workspaces.
        for count, tokens in reversed(self.capture_shapes):
            features = torch.zeros(count, tower.num_mel_bins, window, device=weight.device, dtype=weight.dtype)
            indices = torch.arange(tokens, device=weight.device)
            boundaries = list(range(0, tokens, cnn_width * (tower.n_window_infer // window))) + [tokens]
            boundaries += [tokens] * (count + 2 - len(boundaries))
            cu = torch.tensor(boundaries, device=weight.device, dtype=torch.int32)
            with torch.inference_mode():
                for _ in range(2):
                    audio_forward_prepared(tower, features, indices, cu, max_seqlen)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    output = audio_forward_prepared(tower, features, indices, cu, max_seqlen)
            self.graphs[count, tokens] = AudioGraph(graph, features, indices, cu, output)
        logger.info("Captured %d Qwen3-Omni audio encoder CUDA graphs with exact CNN/token shapes", len(self.graphs))

    def execute(self, input_features, lengths: list[int]):
        if not self.graphs:
            return None
        tower = self.tower
        window = tower.n_window * 2
        chunks, indices, boundaries, outputs = audio_chunk_metadata(lengths, window, tower.n_window_infer)
        key = len(chunks), len(indices)
        if key not in self.graphs or max(chunks) != window:
            self.graph_misses += len(lengths)
            if self.graph_misses <= 3 or self.graph_misses % 100 == 0:
                logger.info(
                    "Qwen3-Omni audio graph fallback: chunks=%d width=%d hits=%d misses=%d",
                    len(chunks),
                    max(chunks),
                    self.graph_hits,
                    self.graph_misses,
                )
            return None
        captured = self.graphs[key]
        # pad_sequence includes zero tail padding and overwrites every row.
        split = input_features.T.split(chunks)
        padded = torch.nn.utils.rnn.pad_sequence(split, batch_first=True).transpose(1, 2)
        captured.features.copy_(padded)
        live_tokens = len(indices)
        boundaries += [live_tokens] * (captured.cu_seqlens.numel() - len(boundaries))
        captured.indices.copy_(async_tensor_h2d(torch.tensor(indices, dtype=torch.long), device=input_features.device))
        captured.cu_seqlens.copy_(
            async_tensor_h2d(torch.tensor(boundaries, dtype=torch.int32), device=input_features.device)
        )
        captured.graph.replay()
        self.graph_hits += len(lengths)
        if self.graph_hits <= 3 or self.graph_hits % 64 == 0:
            logger.info("Qwen3-Omni audio encoder graphs: hits=%d misses=%d", self.graph_hits, self.graph_misses)
        # Request embeddings outlive the next replay in the encoder cache.
        return captured.output[:live_tokens].clone().split(outputs)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused SigLIP encoder layers and CUDA graphs for MiniCPM-o 4.5's packed vision encode.

``SiglipVisionTransformer.forward_packed`` runs each of the 27 layers as
LayerNorm, three projections, attention, a copy of the attention output into
a packed buffer, the output projection, a residual add, LayerNorm, fc1, GELU,
fc2 and another residual add: 13 kernels per layer, six of them full passes
over the activations. Here a layer is six kernels:

* one packed QKV GEMM (the q/k/v weights stacked once; the modules keep views
  of the stacked weight, so no memory is added);
* attention per run of equal grids, reading q/k/v as strided views of the
  packed projection and returning its output in the token layout the next
  GEMM reads, with no copy for the single-run case;
* the output projection without its bias, then one pass that adds the bias
  and the residual, stores the new residual in place and applies
  ``layer_norm2`` (``residual_layer_norm``);
* fc1 with its bias and the tanh GELU as the cuBLASLt epilogue
  (``torch._addmm_activation``);
* fc2 without its bias, then the same residual pass into the next layer's
  ``layer_norm1`` (the encoder's ``post_layernorm`` after the last layer).

The math per token is the eager layer's; the residual add and the LayerNorm
statistics are computed in fp32 inside one pass instead of rounding the sum
to the model dtype first.

A duplex camera frame always has one patch grid, and a runner step encodes
up to ``vision_batch_size`` of them, so ``VisionGraphEncoder`` captures the
packed tower + resampler per ``(grid, batch bucket)`` and replays it: at
batch 1..4 most of an eager encode is kernel launch time. Batches pad up to
their bucket with zero slices; a slice attends only to its own patches in
both the tower and the resampler, so padding never reaches a real slice.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .ops import residual_layer_norm

logger = init_logger(__name__)

# Batch buckets of the vision graphs; the checkpoint's vision_batch_size (16)
# bounds a chunk, so it is the largest bucket.
DEFAULT_GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16)
# Distinct (grid, bucket) graphs kept at once; duplex uses one grid.
_MAX_GRAPHS = 16
# A (grid, bucket) is captured on its third encode, so one-off shapes (the
# startup profile run, a single turn-mode image) stay eager.
_CAPTURE_AFTER = 2


def supports_fused_layers(vpm: nn.Module) -> bool:
    """Whether ``encode_packed_fused`` covers ``vpm``'s layers (SigLIP with a GELU MLP).

    An explicit ``eager`` attention implementation (the A/B reference) keeps
    the unfused layers, since the fused ones always attend through SDPA.
    """
    layers = getattr(getattr(vpm, "encoder", None), "layers", None)
    if not layers or getattr(vpm, "post_layernorm", None) is None:
        return False
    if getattr(getattr(vpm, "config", None), "_attn_implementation", None) == "eager":
        return False
    layer = layers[0]
    attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    return (
        attn is not None
        and mlp is not None
        and all(getattr(attn, name, None) is not None for name in ("q_proj", "k_proj", "v_proj", "out_proj"))
        and getattr(mlp.config, "hidden_act", None) in ("gelu_pytorch_tanh", "gelu")
    )


def packed_qkv(attn: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(3E, E)`` q/k/v weight (and bias) of ``attn``, stacked once.

    The q/k/v parameters are then re-pointed at row views of the stacked
    tensors, so the old storage is freed and the eager path keeps working.
    """
    cached = attn.__dict__.get("_packed_qkv")
    projections = (attn.q_proj, attn.k_proj, attn.v_proj)
    if cached is not None and all(
        p.weight.data_ptr() == cached[0][i * p.weight.shape[0]].data_ptr() for i, p in enumerate(projections)
    ):
        return cached
    with torch.no_grad():
        weight = torch.cat([p.weight for p in projections])
        biases = [p.bias for p in projections]
        bias = torch.cat(biases) if all(b is not None for b in biases) else None
        rows = int(projections[0].weight.shape[0])
        for index, projection in enumerate(projections):
            projection.weight.data = weight[index * rows : (index + 1) * rows]
            if bias is not None:
                projection.bias.data = bias[index * rows : (index + 1) * rows]
    attn.__dict__["_packed_qkv"] = (weight, bias)
    return weight, bias


def _attend(attn: nn.Module, qkv: torch.Tensor, seq_groups: Sequence[tuple[int, int, int]]) -> torch.Tensor:
    """``(tokens, E)`` attention output of packed ``(tokens, 3E)`` q/k/v, each item attending to itself."""
    heads, head_dim = attn.num_heads, attn.head_dim
    embed = heads * head_dim
    outputs = []
    for start, count, seq_len in seq_groups:
        run = qkv[start : start + count * seq_len].view(count, seq_len, 3, heads, head_dim)
        query, key, value = (run[:, :, index].transpose(1, 2) for index in range(3))
        out = F.scaled_dot_product_attention(query, key, value, scale=attn.scale)
        outputs.append(out.transpose(1, 2).reshape(count * seq_len, embed))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def _mlp_up(mlp: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    fc1 = mlp.fc1
    if mlp.config.hidden_act == "gelu_pytorch_tanh" and fc1.bias is not None and hidden.is_cuda:
        # cuBLASLt's GELU epilogue is the tanh approximation.
        return torch._addmm_activation(fc1.bias, hidden, fc1.weight.t(), use_gelu=True)
    return mlp.activation_fn(fc1(hidden))


def encode_packed_fused(
    vpm: nn.Module, hidden_states: torch.Tensor, seq_groups: Sequence[tuple[int, int, int]]
) -> torch.Tensor:
    """The encoder layers and ``post_layernorm`` of ``vpm.forward_packed`` over ``(tokens, E)`` embeddings.

    The residual stream is updated in place in a contiguous copy of
    ``hidden_states`` (the patch embedding hands over a transposed view).
    """
    layers = vpm.encoder.layers
    residual = hidden_states.contiguous().unsqueeze(0)
    first = layers[0].layer_norm1
    normed = residual_layer_norm(residual, weight=first.weight, bias=first.bias, eps=first.eps)
    for index, layer in enumerate(layers):
        attn = layer.self_attn
        qkv_weight, qkv_bias = packed_qkv(attn)
        attended = _attend(attn, F.linear(normed[0], qkv_weight, qkv_bias), seq_groups)
        norm = layer.layer_norm2
        normed = residual_layer_norm(
            residual,
            F.linear(attended, attn.out_proj.weight).unsqueeze(0),
            y_bias=attn.out_proj.bias,
            weight=norm.weight,
            bias=norm.bias,
            eps=norm.eps,
            residual_out=residual,
        )
        mlp = layer.mlp
        last = index + 1 == len(layers)
        norm = vpm.post_layernorm if last else layers[index + 1].layer_norm1
        normed = residual_layer_norm(
            residual,
            F.linear(_mlp_up(mlp, normed[0]), mlp.fc2.weight).unsqueeze(0),
            y_bias=mlp.fc2.bias,
            weight=norm.weight,
            bias=norm.bias,
            eps=norm.eps,
            residual_out=None if last else residual,
        )
    return normed[0]


def _bucket(count: int, sizes: Sequence[int]) -> int | None:
    for size in sizes:
        if count <= size:
            return size
    return None


class VisionGraphEncoder:
    """CUDA graphs of the packed SigLIP tower + resampler for single-grid chunks.

    ``encode(pixels, height, width, count)`` returns ``(count, queries, D)``
    for ``count`` slices of one ``(height, width)`` patch grid packed as
    ``(1, C, patch, count * height * width * patch)``, or ``None`` when no
    graph covers the chunk (the caller runs it eagerly). A ``(grid, bucket)``
    is captured once it recurs (``_CAPTURE_AFTER``) and kept in LRU order.
    """

    def __init__(self, vpm: nn.Module, resampler: nn.Module, *, batch_sizes: Sequence[int] = DEFAULT_GRAPH_BATCH_SIZES):
        self.vpm = vpm
        self.resampler = resampler
        self.batch_sizes = tuple(sorted({int(b) for b in batch_sizes if int(b) > 0}))
        self._graphs: OrderedDict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]] = (
            OrderedDict()
        )
        self._failed: set[tuple[int, int, int]] = set()
        self._seen: dict[tuple[int, int, int], int] = {}
        self.enabled = True

    def encode(self, pixels: torch.Tensor, height: int, width: int, count: int) -> torch.Tensor | None:
        if not self.enabled or torch.cuda.is_current_stream_capturing():
            return None
        bucket = _bucket(count, self.batch_sizes)
        if bucket is None:
            return None
        key = (int(height), int(width), bucket)
        entry = self._graphs.get(key)
        if entry is None:
            if key in self._failed:
                return None
            seen = self._seen.get(key, 0)
            if seen < _CAPTURE_AFTER:
                if len(self._seen) >= 4 * _MAX_GRAPHS:
                    self._seen.clear()
                self._seen[key] = seen + 1
                return None
            entry = self._capture(key, pixels)
            if entry is None:
                return None
        self._graphs.move_to_end(key)
        static_pixels, static_out, graph = entry
        width_px = int(pixels.shape[-1])
        static_pixels[..., :width_px].copy_(pixels)
        if width_px < int(static_pixels.shape[-1]):
            static_pixels[..., width_px:].zero_()
        graph.replay()
        return static_out[:count].clone()

    def _run(self, pixels: torch.Tensor, runs: list[tuple[int, int, int]]) -> torch.Tensor:
        return self.resampler.forward_packed(self.vpm.forward_packed(pixels, runs), runs)

    def _capture(self, key: tuple[int, int, int], like: torch.Tensor) -> tuple | None:
        height, width, bucket = key
        runs = [(height, width, bucket)]
        patch = int(like.shape[-2])
        static_pixels = torch.zeros(
            (*like.shape[:-1], bucket * height * width * patch), device=like.device, dtype=like.dtype
        )
        try:
            stream = torch.cuda.Stream(device=like.device)
            stream.wait_stream(torch.cuda.current_stream(like.device))
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(2):
                    self._run(static_pixels, runs)
            torch.cuda.current_stream(like.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # thread_local: other threads of the worker may use CUDA meanwhile.
            with (
                torch.inference_mode(),
                torch.cuda.graph(
                    graph, pool=current_platform.get_global_graph_pool(), capture_error_mode="thread_local"
                ),
            ):
                static_out = self._run(static_pixels, runs)
        except Exception:
            logger.warning("MiniCPM-o vision CUDA graph capture failed for %s; staying eager", key, exc_info=True)
            self._failed.add(key)
            return None
        if len(self._graphs) >= _MAX_GRAPHS:
            _, (_, _, evicted) = self._graphs.popitem(last=False)
            evicted.reset()
        self._graphs[key] = (static_pixels, static_out, graph)
        logger.info("Captured MiniCPM-o vision CUDA graph grid=%dx%d batch=%d", height, width, bucket)
        return self._graphs[key]

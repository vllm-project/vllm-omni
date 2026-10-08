# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused packed SigLIP layers and CUDA graphs for MiniCPM-o 4.5 vision encode."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from copy import copy
from math import prod
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import ModelConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.platforms import current_platform
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

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
    """Whether ``encode_packed_fused`` covers ``vpm`` (SigLIP with a GELU MLP)."""
    layers = getattr(getattr(vpm, "encoder", None), "layers", None)
    if not layers or getattr(vpm, "post_layernorm", None) is None:
        return False
    if getattr(getattr(vpm, "config", None), "_attn_implementation", None) == "eager":
        return False
    attn, mlp = getattr(layers[0], "self_attn", None), getattr(layers[0], "mlp", None)
    return all(
        getattr(attn, name, None) is not None for name in ("q_proj", "k_proj", "v_proj", "out_proj")
    ) and getattr(getattr(mlp, "config", None), "hidden_act", None) in ("gelu_pytorch_tanh", "gelu")


def packed_qkv(attn: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(3E, E)`` q/k/v weight (and bias) of ``attn``, stacked once."""
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


def _add_norm(residual: torch.Tensor, norm: nn.Module, y=None, y_bias=None, residual_out=None) -> torch.Tensor:
    """``norm(residual + y + y_bias)`` for a ``(tokens, E)`` ``y``, the sum also written to ``residual_out``."""
    y = None if y is None else y.unsqueeze(0)
    return residual_layer_norm(
        residual, y, y_bias=y_bias, weight=norm.weight, bias=norm.bias, eps=norm.eps, residual_out=residual_out
    )


def encode_packed_fused(
    vpm: nn.Module, hidden_states: torch.Tensor, seq_groups: Sequence[tuple[int, int, int]]
) -> torch.Tensor:
    """The encoder layers and ``post_layernorm`` of ``vpm.forward_packed`` over ``(tokens, E)`` embeddings.

    The residual stream is updated in place in a contiguous copy of
    ``hidden_states`` (the patch embedding hands over a transposed view).
    """
    layers = vpm.encoder.layers
    residual = hidden_states.contiguous().unsqueeze(0)
    normed = _add_norm(residual, layers[0].layer_norm1)
    for index, layer in enumerate(layers):
        attn, mlp = layer.self_attn, layer.mlp
        qkv_weight, qkv_bias = packed_qkv(attn)
        attended = _attend(attn, F.linear(normed[0], qkv_weight, qkv_bias), seq_groups)
        y = F.linear(attended, attn.out_proj.weight)
        normed = _add_norm(residual, layer.layer_norm2, y, attn.out_proj.bias, residual_out=residual)
        last = index + 1 == len(layers)
        norm = vpm.post_layernorm if last else layers[index + 1].layer_norm1
        y = F.linear(_mlp_up(mlp, normed[0]), mlp.fc2.weight)
        normed = _add_norm(residual, norm, y, mlp.fc2.bias, residual_out=None if last else residual)
    return normed[0]


def _copy_pixels_padded(dst: torch.Tensor, src: torch.Tensor) -> None:
    width_px = int(src.shape[-1])
    dst[..., :width_px].copy_(src)
    if width_px < int(dst.shape[-1]):
        dst[..., width_px:].zero_()


class _PackedVisionAdapter(SupportsEncoderCudaGraph):
    """Adapter exposing the packed SigLIP tower + resampler to vLLM's EncoderCudaGraphManager."""

    supports_encoder_cudagraph: ClassVar[Literal[True]] = True

    def __init__(self, forward: Any, runs: Any, pixel_shape: Any, output_shape: torch.Size):
        self.forward, self.runs, self.pixel_shape, self.output_shape = forward, runs, pixel_shape, output_shape
        self.tokens = prod(output_shape[:-1]) or 1

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixels"],
            out_hidden_size=self.output_shape[-1],
            padding_logics={"pixels": _copy_pixels_padded},
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        return self.tokens, self.tokens

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        return [EncoderItemSpec(input_size=self.tokens, output_tokens=self.tokens)]

    def select_encoder_cudagraph_items(self, mm_kwargs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        return dict(mm_kwargs)

    def prepare_encoder_cudagraph_capture_inputs(self, *args: Any, **kwargs: Any) -> EncoderCudaGraphCaptureInputs:
        device = kwargs.get("device") or args[3]
        dtype = kwargs.get("dtype") or args[4]
        return EncoderCudaGraphCaptureInputs({"pixels": torch.zeros(self.pixel_shape, dtype=dtype, device=device)})

    def prepare_encoder_cudagraph_replay_buffers(
        self, mm_kwargs: dict[str, Any], *args: Any, **kwargs: Any
    ) -> EncoderCudaGraphReplayBuffers:
        return EncoderCudaGraphReplayBuffers(mm_kwargs)

    def encoder_cudagraph_forward(self, inputs: dict[str, torch.Tensor], path: str = "default") -> torch.Tensor:
        return self.forward(inputs["pixels"], self.runs)

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any], path: str = "default") -> torch.Tensor:
        return self.encoder_cudagraph_forward(mm_kwargs, path)

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        dest[0] = outputs["default"].clone()


class VisionGraphEncoder:
    """CUDA graphs of the packed SigLIP tower + resampler for single-grid chunks."""

    def __init__(
        self,
        vpm: nn.Module,
        resampler: nn.Module,
        *,
        batch_sizes: Sequence[int] = DEFAULT_GRAPH_BATCH_SIZES,
        vllm_config: VllmConfig | None = None,
    ):
        self.vpm, self.resampler, self.vllm_config = vpm, resampler, vllm_config
        self.batch_sizes = tuple(sorted({int(b) for b in batch_sizes if int(b) > 0}))
        # (h, w, bucket) -> EncoderCudaGraphManager, least recently used first.
        self._managers: OrderedDict[tuple[int, int, int], EncoderCudaGraphManager] = OrderedDict()
        self._failed: set[tuple[int, int, int]] = set()
        self._seen: dict[tuple[int, int, int], int] = {}

    def encode(self, pixels: torch.Tensor, height: int, width: int, count: int) -> torch.Tensor | None:
        if torch.cuda.is_current_stream_capturing():
            return None
        bucket = min((size for size in self.batch_sizes if count <= size), default=None)
        if bucket is None:
            return None
        key = (int(height), int(width), bucket)
        manager = self._managers.get(key)
        if manager is None:
            if key in self._failed:
                return None
            seen = self._seen.get(key, 0)
            if seen < _CAPTURE_AFTER:
                if len(self._seen) >= 4 * _MAX_GRAPHS:
                    self._seen.clear()
                self._seen[key] = seen + 1
                return None
            manager = self._capture(key, pixels)
            if manager is None:
                return None
        self._managers.move_to_end(key)
        return manager.execute({"pixels": pixels})[0][:count]

    def _run(self, pixels: torch.Tensor, runs: list[tuple[int, int, int]]) -> torch.Tensor:
        return self.resampler.forward_packed(self.vpm.forward_packed(pixels, runs), runs)

    def _capture(self, key: tuple[int, int, int], like: torch.Tensor) -> EncoderCudaGraphManager | None:
        height, width, bucket = key
        runs, patch = [(height, width, bucket)], int(like.shape[-2])
        pixel_shape = (*like.shape[:-1], bucket * height * width * patch)
        dummy_pixels = torch.zeros(pixel_shape, device=like.device, dtype=like.dtype)
        try:
            stream = torch.cuda.Stream(device=like.device)
            stream.wait_stream(torch.cuda.current_stream(like.device))
            with torch.cuda.stream(stream), torch.inference_mode():
                out = self._run(dummy_pixels, runs)
                for _ in range(1):
                    self._run(dummy_pixels, runs)
            torch.cuda.current_stream(like.device).wait_stream(stream)
            adapter = _PackedVisionAdapter(self._run, runs, pixel_shape, out.shape)
            config = copy(self.vllm_config) if self.vllm_config is not None else VllmConfig()
            if getattr(config, "model_config", None) is None:
                config.model_config = ModelConfig.__new__(ModelConfig)
            if getattr(config.model_config, "multimodal_config", None) is None:
                config.model_config.multimodal_config = MultiModalConfig()
            cc = config.compilation_config = copy(config.compilation_config)
            cc.encoder_cudagraph_token_budgets = [adapter.tokens]
            cc.encoder_cudagraph_max_vision_items_per_batch, cc.encoder_cudagraph_max_frames_per_batch = 1, 0
            config.parallel_config = copy(config.parallel_config)
            config.parallel_config.tensor_parallel_size = 1

            manager = EncoderCudaGraphManager(config, like.device, like.dtype, adapter)
            with torch.cuda.stream(stream):
                manager.capture(graph_pool=current_platform.get_global_graph_pool())
            torch.cuda.current_stream(like.device).wait_stream(stream)
        except Exception:
            logger.warning("MiniCPM-o vision CUDA graph capture failed for %s; staying eager", key, exc_info=True)
            self._failed.add(key)
            return None
        if len(self._managers) >= _MAX_GRAPHS:
            _, evicted = self._managers.popitem(last=False)
            evicted.clear()
        self._managers[key] = manager
        logger.info("Captured MiniCPM-o vision CUDA graph grid=%dx%d batch=%d via manager", height, width, bucket)
        return manager

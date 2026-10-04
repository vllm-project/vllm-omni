# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-shape MiniCPM adapters for vLLM encoder CUDA graph management."""

from collections.abc import Callable, Hashable
from copy import copy
from math import prod

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)


def _copy_exact(target: torch.Tensor, source: torch.Tensor) -> None:
    target.copy_(source)


class _ExactShapeEncoder(SupportsEncoderCudaGraph):
    """One already-packed local encoder batch is one indivisible manager item.

    Packing and distributed placement belong to the caller. Keeping the batch
    intact preserves attention masks and avoids introducing padding semantics.
    """

    def __init__(
        self, forward: Callable[..., torch.Tensor], inputs: tuple[torch.Tensor | None, ...], output_shape: torch.Size
    ):
        self.forward = forward
        self.templates = tuple(None if x is None else (x.shape, x.dtype, x.device) for x in inputs)
        self.output_shape = output_shape
        self.tokens = prod(output_shape[:-1]) or 1
        self.keys = [str(i) for i, x in enumerate(inputs) if x is not None]

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image", "audio"],
            buffer_keys=self.keys,
            out_hidden_size=self.output_shape[-1],
            padding_logics=dict.fromkeys(self.keys, _copy_exact),
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        return self.tokens, self.tokens

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, torch.Tensor]) -> list[EncoderItemSpec]:
        return [EncoderItemSpec(input_size=self.tokens, output_tokens=self.tokens)]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, torch.Tensor], indices: list[int]
    ) -> dict[str, torch.Tensor]:
        assert indices == [0]
        return dict(mm_kwargs)

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = (),
    ) -> EncoderCudaGraphCaptureInputs:
        return EncoderCudaGraphCaptureInputs(
            {
                str(i): torch.zeros(shape, dtype=input_dtype, device=input_device)
                for i, template in enumerate(self.templates)
                if template is not None
                for shape, input_dtype, input_device in [template]
            }
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, torch.Tensor],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        return EncoderCudaGraphReplayBuffers(mm_kwargs)

    def encoder_cudagraph_forward(self, inputs: dict[str, torch.Tensor], path: str = "default") -> torch.Tensor:
        return self.forward(*(inputs.get(str(i)) for i in range(len(self.templates))))

    def encoder_eager_forward(self, mm_kwargs: dict[str, torch.Tensor], path: str = "default") -> torch.Tensor:
        return self.encoder_cudagraph_forward(mm_kwargs, path)

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> None:
        output = outputs["default"]
        # Callers retain embeddings across subsequent replays.
        dest[0] = output.clone() if clone else output


class EncoderCudaGraph:
    """Capture repeated shapes without padding or changing attention semantics.

    The owner must bypass this object for training, stateful KV caches and
    data-dependent Python branches. Each entry owns a private graph pool;
    callers get a clone so a later replay cannot overwrite retained embeddings.
    Unseen shapes after the cap run eagerly instead of growing GPU memory.
    """

    def __init__(self, forward: Callable[..., torch.Tensor], vllm_config: VllmConfig, *, max_graphs: int = 4):
        self.forward = forward
        self.vllm_config = vllm_config
        self.max_graphs = max_graphs
        self.graphs: dict[tuple, EncoderCudaGraphManager] = {}
        self._seen: dict[tuple, torch.Size] = {}

    def __call__(self, *inputs: torch.Tensor | None) -> torch.Tensor:
        tensors = [value for value in inputs if value is not None]
        if (
            not tensors
            or any(value.device.type != "cuda" for value in tensors)
            or len({value.device for value in tensors}) != 1
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.forward(*inputs)
        # Do not share mutable replay buffers across independent CUDA streams.
        key = (
            torch.cuda.current_stream(tensors[0].device).cuda_stream,
            tuple(None if x is None else (x.shape, x.dtype, x.device) for x in inputs),
        )
        entry = self.graphs.get(key)
        if entry is None:
            if len(self.graphs) >= self.max_graphs:
                return self.forward(*inputs)
            if key not in self._seen:
                # Avoid paying capture latency for one-off shapes. Bound even
                # the admission metadata for streams with arbitrary resolutions.
                if len(self._seen) >= self.max_graphs * 4:
                    self._seen.pop(next(iter(self._seen)))
                output = self.forward(*inputs)
                self._seen[key] = output.shape
                return output
            entry = self._capture(inputs, self._seen[key])
            self.graphs[key] = entry
            self._seen.pop(key)
        return entry.execute({str(i): x for i, x in enumerate(inputs) if x is not None})[0]

    def _capture(self, inputs: tuple[torch.Tensor | None, ...], output_shape: torch.Size) -> EncoderCudaGraphManager:
        tensor = next(x for x in inputs if x is not None)
        adapter = _ExactShapeEncoder(self.forward, inputs, output_shape)
        # Do not mutate the runner configuration. This adapter has already
        # packed and placed a local batch, so manager-level DP must be disabled.
        config = copy(self.vllm_config)
        config.compilation_config = copy(config.compilation_config)
        config.compilation_config.encoder_cudagraph_token_budgets = [adapter.tokens]
        config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 1
        config.compilation_config.encoder_cudagraph_max_frames_per_batch = 0
        config.parallel_config = copy(config.parallel_config)
        config.parallel_config.tensor_parallel_size = 1
        manager = EncoderCudaGraphManager(config, tensor.device, tensor.dtype, adapter)
        stream = torch.cuda.Stream(device=tensor.device)
        stream.wait_stream(torch.cuda.current_stream(tensor.device))
        with torch.cuda.stream(stream):
            manager.capture(graph_pool=torch.cuda.graph_pool_handle())
        torch.cuda.current_stream(tensor.device).wait_stream(stream)
        return manager

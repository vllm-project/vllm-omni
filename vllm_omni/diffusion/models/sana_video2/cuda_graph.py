# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.logger import init_logger

from .transformer_sana_video2 import SanaVideo2TransformerModel

logger = init_logger(__name__)


@dataclass
class _GraphEntry:
    capture: BreakableCUDAGraphCapture
    inputs: tuple[torch.Tensor | None, ...]
    ropes: dict[str, torch.Tensor]
    output: torch.Tensor


class SanaVideo2CudaGraphRunner:
    """Explicitly capture fixed shapes; replay hits and run misses eagerly."""

    def __init__(self, model: SanaVideo2TransformerModel) -> None:
        self.model = model
        self.entries: dict[tuple, _GraphEntry] = {}

    @staticmethod
    def _signature(inputs: tuple[torch.Tensor | None, ...]) -> tuple:
        return tuple(None if x is None else (tuple(x.shape), x.dtype, x.device) for x in inputs)

    @torch.no_grad()
    def capture(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> None:
        inputs = (hidden_states, timestep, encoder_hidden_states, encoder_attention_mask)
        self.model.validate_inputs(*inputs)
        if hidden_states.device.type != "cuda":
            raise ValueError("SANA-Video 2.0 CUDA graphs require CUDA inputs")
        key = self._signature(inputs)
        if key in self.entries:
            return
        static = tuple(None if x is None else x.clone() for x in inputs)
        ropes = self.model.prepare_rotary_emb(hidden_states)
        with torch.accelerator.device_index(hidden_states.device.index):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            # Each shape owns a pool so different signatures cannot reuse live storage.
            capture = BreakableCUDAGraphCapture(pool=torch.cuda.graph_pool_handle())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.model.forward_tensor(*static, ropes)
                stream.synchronize()
                with capture:
                    output = self.model.forward_tensor(*static, ropes)
            torch.cuda.current_stream().wait_stream(stream)
        self.entries[key] = _GraphEntry(capture, static, ropes, output)
        logger.info("SANA-Video 2.0 captured %d graph segment(s) for %s", capture.num_graphs, key)

    @torch.no_grad()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = (hidden_states, timestep, encoder_hidden_states, encoder_attention_mask)
        entry = self.entries.get(self._signature(inputs))
        if entry is None:
            return self.model(*inputs)
        self.model.validate_inputs(*inputs)
        for static, live in zip(entry.inputs, inputs):
            if static is not None:
                static.copy_(live)
        entry.capture.replay()
        # Callers may retain this output across later replays.
        return entry.output.clone()

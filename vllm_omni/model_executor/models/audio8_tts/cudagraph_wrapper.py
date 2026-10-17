# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shape-bucketed CUDA graphs for the Audio8 codec's causal paths."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .codec import ArkttsCodec

logger = init_logger(__name__)


def decoder_capture_sizes(chunk: int, context: int, initial: int) -> tuple[int, ...]:
    """Full windows emitted by the configured stage input processor.

    Short terminal windows use the next larger captured window. Request-level
    overrides of the initial chunk are intentionally outside this contract.
    """
    if chunk <= 0 or context < 0 or initial < 0:
        raise ValueError("Invalid Audio8 codec chunk configuration")
    initial = min(initial, chunk)
    sizes = {chunk, chunk + context}
    if initial:
        sizes.update(range(initial, chunk + 1, initial))
        coverage = chunk // initial * initial
        sizes.add(chunk + min(context, coverage))
    return tuple(sorted(sizes))


class Audio8CodecCUDAGraphWrapper:
    """Group ragged requests and lazily capture only the required batch shapes."""

    def __init__(
        self,
        codec: ArkttsCodec,
        *,
        batch_size: int,
        decoder_sizes: tuple[int, ...] = (),
    ) -> None:
        self.codec = codec
        self.batch_size = batch_size
        self.decoder_sizes = tuple(sorted(set(decoder_sizes)))
        # Keys are (path, captured batch size, captured sequence length).
        self._graphs: dict[tuple[str, int, int], torch.cuda.CUDAGraph] = {}
        self._inputs: dict[tuple[str, int, int], torch.Tensor] = {}
        self._outputs: dict[tuple[str, int, int], torch.Tensor] = {}
        self._lengths: dict[tuple[str, int, int], torch.Tensor] = {}

    def capture_decoder(self) -> None:
        """Pre-capture configured lengths at geometric batch sizes up to capacity."""
        batch_sizes = []
        size = 1
        while size < self.batch_size:
            batch_sizes.append(size)
            size *= 2
        batch_sizes.append(self.batch_size)
        for length in self.decoder_sizes:
            for batch_size in batch_sizes:
                self._capture("decode", batch_size, length)

    def _capture(self, kind: str, batch_size: int, size: int) -> None:
        key = (kind, batch_size, size)
        if key in self._graphs:
            return
        component = "encoder" if kind == "encode" else "decoder"
        device = next(self.codec.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("Audio8 codec CUDA graphs require a CUDA device")
        if kind == "encode":
            static_input = torch.zeros((batch_size, 1, size), device=device, dtype=next(self.codec.parameters()).dtype)
            static_lengths = torch.full((batch_size,), size, device=device, dtype=torch.long)

            def run() -> torch.Tensor:
                return self.codec._encode_eager(static_input, static_lengths)[0]

        else:
            static_input = torch.zeros((batch_size, 10, size), device=device, dtype=torch.long)
            static_lengths = torch.empty(0, device=device, dtype=torch.long)

            def run() -> torch.Tensor:
                return self.codec._decode_eager(static_input)

        # Warm up on a separate stream before capture, including the codec's
        # length-indexed attention mask and RoPE caches.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                run()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
            output = run()
        self._graphs[key] = graph
        self._inputs[key] = static_input
        self._outputs[key] = output
        self._lengths[key] = static_lengths
        logger.info(
            "Captured Audio8 codec %s CUDA graph: batch=%d length=%d",
            component,
            batch_size,
            size,
        )

    @staticmethod
    def _normalize(item: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "encode":
            if item.ndim == 1:
                item = item.unsqueeze(0)
            if item.ndim != 2 or item.shape[0] != 1:
                raise ValueError("Each encoder request must have shape [samples] or [1, samples]")
        elif item.ndim != 2 or item.shape[0] != 10:
            raise ValueError("Each decoder request must have shape [10, frames]")
        if item.shape[-1] <= 0:
            raise ValueError("Audio8 codec inputs must have positive length")
        return item

    def _bucket(self, kind: str, length: int) -> int:
        if kind == "encode":
            half_second_frames = math.ceil(self.codec.sample_rate / (2 * self.codec.frame_length))
            bucket_samples = half_second_frames * self.codec.frame_length
            return math.ceil(length / bucket_samples) * bucket_samples
        for size in self.decoder_sizes:
            if length <= size:
                return size
        raise ValueError(f"Decoder input of {length} frames exceeds configured graph buckets {self.decoder_sizes}")

    def _graph_key(
        self, kind: str, batch_size: int, bucket: int, *, exact_length: bool = False
    ) -> tuple[str, int, int]:
        """Reuse the smallest suitable graph before capturing a new shape."""
        candidates = [
            key
            for key in self._graphs
            if key[0] == kind and key[1] >= batch_size and (key[2] == bucket if exact_length else key[2] >= bucket)
        ]
        if candidates:
            return min(candidates, key=lambda key: (key[1], key[2]))
        if kind == "decode":
            raise RuntimeError(f"No pre-captured Audio8 decoder graph for batch={batch_size}, bucket={bucket}")
        self._capture(kind, batch_size, bucket)
        return kind, batch_size, bucket

    @torch.inference_mode()
    def _run(self, kind: str, requests: list[torch.Tensor]) -> list[torch.Tensor]:
        if not requests:
            return []
        normalized = [self._normalize(item, kind) for item in requests]
        groups: dict[int, list[int]] = defaultdict(list)
        for index, item in enumerate(normalized):
            groups[self._bucket(kind, item.shape[-1])].append(index)
        graph_groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for bucket in sorted(groups):
            indices = groups[bucket]
            key = self._graph_key(kind, len(indices), bucket)
            graph_groups[key].extend(indices)

        while any(len(indices) > key[1] for key, indices in graph_groups.items()):
            regrouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
            for key, indices in graph_groups.items():
                selected = (
                    self._graph_key(kind, len(indices), key[2], exact_length=kind == "encode")
                    if len(indices) > key[1]
                    else key
                )
                regrouped[selected].extend(indices)
            graph_groups = regrouped

        results: list[torch.Tensor | None] = [None] * len(requests)
        for key, indices in graph_groups.items():
            graph_size = key[2]
            static_input = self._inputs[key]
            static_input.zero_()
            for slot, index in enumerate(indices):
                item = normalized[index]
                static_input[slot, :, : item.shape[-1]].copy_(item)
            self._graphs[key].replay()
            output = self._outputs[key]
            for slot, index in enumerate(indices):
                length = normalized[index].shape[-1]
                valid = (
                    math.ceil(length / self.codec.frame_length)
                    if kind == "encode"
                    else length * output.shape[-1] // graph_size
                )
                results[index] = output[slot, :, :valid].clone()
        assert all(item is not None for item in results)
        return results  # type: ignore[return-value]

    def encode(self, requests: list[torch.Tensor]) -> list[torch.Tensor]:
        return self._run("encode", requests)

    def decode(self, requests: list[torch.Tensor]) -> list[torch.Tensor]:
        return self._run("decode", requests)


__all__ = ["Audio8CodecCUDAGraphWrapper", "decoder_capture_sizes"]

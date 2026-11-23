# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in, single-row Qwen2 decode graphs with request-isolated static KV."""

from contextlib import contextmanager
from threading import Lock

import torch
from transformers import Qwen2Model, StaticCache
from vllm.logger import init_logger

logger = init_logger(__name__)


class Qwen2DecodeGraph:
    """Keep captured addresses alive, while resetting KV for every request.

    Prefill stays eager. Explicit positions and a fixed-size attention mask avoid
    reading the device-side cache length while capturing or replaying decode.
    The request lock covers prefill, all decode steps and exception cleanup.
    """

    def __init__(self, model: Qwen2Model, *, max_cache_len: int):
        self.model = model
        self.max_cache_len = max_cache_len
        self.cache: StaticCache | None = None
        self.active = False
        self._lock = Lock()
        self._failed = False
        self._graph = None
        self._embed = None
        self._position = None
        self._slots = None
        self._output = None
        self._completion = None
        self._retired_resources = None

    def _disable(self) -> None:
        self._failed = True
        self.active = False
        # Keep failed buffers alive until the last request's completion event
        # finishes; their allocations may belong to a different CUDA stream.
        self._retired_resources = (self.cache, self._graph, self._embed, self._position, self._slots, self._output)
        self.cache = None
        self._graph = None
        self._embed = self._position = self._slots = self._output = None

    @contextmanager
    def request(self, inputs_embeds: torch.Tensor):
        with self._lock:
            device = next(self.model.parameters()).device
            stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
            if self._completion is not None:
                stream.wait_event(self._completion)
                if self._completion.query():
                    self._retired_resources = None
            elif stream is None:
                self._retired_resources = None
            supported = (
                inputs_embeds.device.type == "cuda"
                and inputs_embeds.device == device
                and inputs_embeds.ndim == 3
                and inputs_embeds.shape[0] == 1
                and inputs_embeds.shape[-1] == self.model.config.hidden_size
                and 0 < inputs_embeds.shape[1] < self.max_cache_len
                and self.model.config._attn_implementation == "sdpa"
                and not getattr(self.model.config, "use_sliding_window", False)
                and all(
                    kind == "full_attention"
                    for kind in (getattr(self.model.config, "layer_types", None) or ["full_attention"])
                )
            )
            if supported and not self._failed and self._graph is None:
                try:
                    self._capture(inputs_embeds)
                except Exception:
                    # A partially populated cache must never reach real prefill.
                    self._disable()
                    logger.warning("Talker LLM CUDA graph capture failed; using eager decode", exc_info=True)
            self.active = supported and self._graph is not None
            try:
                if self.cache is not None:
                    self.cache.reset()
                yield
            finally:
                self.active = False
                try:
                    if self.cache is not None:
                        self.cache.reset()
                finally:
                    # A CPU lock orders submissions, not GPU completion. Each
                    # streaming producer can submit on a different CUDA stream.
                    if stream is not None:
                        if self._completion is None:
                            self._completion = torch.cuda.Event()
                        self._completion.record(stream)

    def _capture(self, inputs_embeds: torch.Tensor) -> None:
        self.cache = StaticCache(config=self.model.config, max_cache_len=self.max_cache_len)
        self._embed = torch.zeros_like(inputs_embeds[:, :1])
        self._position = torch.zeros((1, 1), device=inputs_embeds.device, dtype=torch.long)
        self._slots = torch.arange(self.max_cache_len, device=inputs_embeds.device)

        def forward():
            mask = (self._slots <= self._position.reshape(1)).reshape(1, 1, 1, self.max_cache_len)
            return self.model(
                inputs_embeds=self._embed,
                position_ids=self._position,
                attention_mask={"full_attention": mask},
                past_key_values=self.cache,
                use_cache=True,
            ).last_hidden_state

        current = torch.cuda.current_stream(inputs_embeds.device)
        stream = torch.cuda.Stream(device=inputs_embeds.device)
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.cache.reset()
                    forward()
            current.wait_stream(stream)
            # Older Transformers caches advance a Python counter. Capturing them
            # would bake a fixed write position into every replay.
            if not all(
                isinstance(getattr(layer, "cumulative_length", None), torch.Tensor)
                and layer.cumulative_length.device.type == "cuda"
                for layer in self.cache.layers
            ):
                raise RuntimeError("Talker LLM CUDA graphs require device-side static-cache positions")
            self.cache.reset()
            stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._output = forward()
        finally:
            current.wait_stream(stream)
        self.cache.reset()
        self._graph = graph
        logger.info(
            "Captured Talker LLM CUDA graph: decode_shape=%s max_cache_len=%d",
            tuple(self._embed.shape),
            self.max_cache_len,
        )

    def decode(self, inputs_embeds: torch.Tensor, *, position: int) -> torch.Tensor | None:
        """Return an owned output tensor, or None for the caller's eager path."""
        if not self.active or self._graph is None or self._embed is None:
            return None
        if (
            inputs_embeds.shape != self._embed.shape
            or inputs_embeds.device != self._embed.device
            or inputs_embeds.dtype != self._embed.dtype
            or not 0 <= position < self.max_cache_len
        ):
            # An eager multi-token step changes cache length by more than one.
            # Keep the rest of this request eager rather than guessing positions.
            self.active = False
            return None
        self._embed.copy_(inputs_embeds)
        self._position.fill_(position)
        try:
            self._graph.replay()
        except Exception:
            # Replay may already have written KV. Retrying this step eagerly
            # would append it twice; fail this request and recover on the next.
            self._disable()
            logger.warning("Talker LLM CUDA graph replay failed; disabling graphs", exc_info=True)
            raise
        return self._output.clone()

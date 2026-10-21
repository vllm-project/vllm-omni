# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-shape CUDA graph for one Wan2.2 DiT denoise step.

Mirrors the AuK DiT graph wrapper (``auk/cudagraph_wrapper.py``): only the
transformer forward of a single denoise step is captured; the flow-matching
scheduler and the CFG combine stay with the caller. The graph is keyed by the
exact input shapes. Unlike AuK, whose target lengths vary continuously and
need bucket rounding, Wan2.2 serving shapes are discrete tiers (resolution ×
frame count, fixed 512-token text embeddings), so exact keys keep the cache
small without any padding or attention-mask handling. Timestep is a mutable
scalar buffer, letting every step of a request replay one graph.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Keyword arguments that are safe to capture. Anything else (image
# embeddings, pipeline-parallel intermediate tensors, ...) falls back to
# eager execution so the wrapper never changes an unverified path.
_CAPTURABLE_KWARGS = frozenset({"timestep", "encoder_hidden_states", "return_dict"})


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_hidden_states: torch.Tensor
    static_timestep: torch.Tensor
    static_encoder_hidden_states: torch.Tensor
    static_out: torch.Tensor


class Wan22DiTCUDAGraphWrapper:
    """Replay one Wan2.2 transformer forward per denoise step.

    The wrapper sits on the ``current_model`` call inside
    ``WanPipeline22.predict_noise``. CFG runs as two sequential calls through
    this wrapper (conditional and unconditional); their combine happens in the
    caller and is deliberately kept out of the graph.
    """

    def __init__(self, transformer: torch.nn.Module, *, enabled: bool = True, max_graphs: int = 8) -> None:
        self.transformer = transformer
        self.enabled = bool(enabled)
        self.max_graphs = max(1, int(max_graphs))
        self._cache: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self._pool_handle: int | None = None
        logger.info(
            "Initialized Wan2.2 DiT CUDA graph cache: enabled=%s slots=%d",
            self.enabled,
            self.max_graphs,
        )

    @staticmethod
    def _key(hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> tuple:
        return (tuple(hidden_states.shape), tuple(timestep.shape), tuple(encoder_hidden_states.shape))

    def _retire_graph_generation_if_full(self) -> None:
        """Retire all graphs together so none outlive shared workspaces."""
        if len(self._cache) >= self.max_graphs:
            self._cache.clear()

    def _forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        result = self.transformer(
            hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        return result[0]

    @torch.no_grad()
    def __call__(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # ``None``-valued and empty-dict keyword arguments (the denoise loop
        # normalizes ``attention_kwargs=None`` to ``{}``) are equivalent to
        # omitting them and stay capturable.
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if v is not None and not (isinstance(v, dict) and not v)
        }
        timestep = kwargs.get("timestep")
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        capturable = (
            self.enabled
            and timestep is not None
            and encoder_hidden_states is not None
            and set(kwargs) <= _CAPTURABLE_KWARGS
            and hidden_states.device.type == "cuda"
            and not torch.cuda.is_current_stream_capturing()
        )
        if not capturable:
            result = self.transformer(hidden_states, **kwargs)
            return result if isinstance(result, torch.Tensor) else result[0]

        key = self._key(hidden_states, timestep, encoder_hidden_states)
        entry = self._cache.get(key)
        if entry is None:
            self._retire_graph_generation_if_full()
            try:
                entry = self._capture(hidden_states, timestep, encoder_hidden_states)
            except Exception:
                self._cache.clear()
                self.enabled = False
                logger.exception(
                    "Disabling Wan2.2 DiT CUDA graphs after capture failure for key=%s; "
                    "subsequent requests will use eager execution.",
                    key,
                )
                raise
            self._cache[key] = entry
        else:
            self._cache.move_to_end(key)

        entry.static_hidden_states.copy_(hidden_states)
        entry.static_timestep.copy_(timestep)
        entry.static_encoder_hidden_states.copy_(encoder_hidden_states)
        entry.graph.replay()
        return entry.static_out.clone()

    def _capture(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> _GraphEntry:
        static_hidden_states = hidden_states.clone()
        static_timestep = timestep.clone()
        static_encoder_hidden_states = encoder_hidden_states.clone()
        # Warm-up runs any lazy kernel selection outside the capture.
        for _ in range(3):
            self._forward(static_hidden_states, static_timestep, static_encoder_hidden_states)
        if self._pool_handle is None:
            self._pool_handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool_handle):
            static_out = self._forward(static_hidden_states, static_timestep, static_encoder_hidden_states)

        logger.info(
            "Captured Wan2.2 DiT single-step CUDA graph: latent=%s timestep=%s text=%s",
            tuple(hidden_states.shape),
            tuple(timestep.shape),
            tuple(encoder_hidden_states.shape),
        )
        return _GraphEntry(
            graph=graph,
            static_hidden_states=static_hidden_states,
            static_timestep=static_timestep,
            static_encoder_hidden_states=static_encoder_hidden_states,
            static_out=static_out,
        )


__all__ = ["Wan22DiTCUDAGraphWrapper"]

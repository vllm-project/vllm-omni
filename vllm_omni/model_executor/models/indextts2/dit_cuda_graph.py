"""Per-shape CUDA graph capture for the S2Mel DiT transformer core.

Profiling showed the DiT estimator forward is CPU launch-bound (the GPU
is idle ~85% of CFM wall time). Within one request the CFM runs 12 Euler
steps with *identical* tensor shapes (T_in never changes), so the graph
captured on the first step is replayed by every subsequent step. Across
requests T_in varies, so graphs are cached per shape with a small LRU;
a cache miss costs roughly one extra eager step (warm run + capture).

Capture safety relies on the IndexTTS2 S2Mel decoder only enabling this
runner for full-mask shapes:

- ``x_lens == T`` for every sample (the decoder feeds full-length sequences), so
  the attention padding mask is all-True and dense flash attention is
  mathematically identical. The decoder sets ``_assume_full_mask`` on
  the gpt_fast ``Attention`` modules so the backend skips its
  ``torch.any(~mask)`` D2H sync, which would abort stream capture.

Modeled on ``CUDAGraphGLMTTSDiTWrapper``
(vllm_omni/model_executor/models/glm_tts/glm_tts_dit_wrapper.py).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_c: torch.Tensor
    static_pos: torch.Tensor
    static_mask: torch.Tensor | None
    static_out: torch.Tensor


class CUDAGraphDiTRunner:
    """Replay the gpt_fast Transformer forward via per-shape CUDA graphs."""

    def __init__(self, transformer: torch.nn.Module, *, max_graphs: int = 4, enabled: bool = True) -> None:
        self.transformer = transformer
        self.max_graphs = int(max_graphs)
        self.enabled = bool(enabled)
        self._cache: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self.last_call_info: dict[str, object] = {}

    def __call__(
        self,
        x_in: torch.Tensor,
        c: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.enabled or x_in.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            self.last_call_info = {
                "mode": "eager",
                "reason": "ineligible",
                "shape": tuple(x_in.shape),
                "cache_size": len(self._cache),
            }
            return self.transformer(x_in, c, input_pos, mask)

        mask_key = None if mask is None else (tuple(mask.shape), mask.dtype)
        device_key = (x_in.device.type, x_in.device.index)
        key = (
            device_key,
            tuple(x_in.shape),
            x_in.dtype,
            tuple(c.shape),
            c.dtype,
            tuple(input_pos.shape),
            input_pos.dtype,
            mask_key,
        )
        entry = self._cache.get(key)
        if entry is None:
            entry = self._capture(x_in, c, input_pos, mask)
            if entry is None:
                self.last_call_info = {
                    "mode": "eager",
                    "reason": "capture_failed",
                    "shape": tuple(x_in.shape),
                    "cache_size": len(self._cache),
                }
                return self.transformer(x_in, c, input_pos, mask)
            self._cache[key] = entry
            while len(self._cache) > self.max_graphs:
                _, evicted = self._cache.popitem(last=False)
                del evicted
            # Kernels are only recorded (not executed) during capture, so
            # replay once to produce the first real output.
            entry.graph.replay()
            self.last_call_info = {
                "mode": "graph",
                "reason": "capture",
                "shape": tuple(x_in.shape),
                "cache_size": len(self._cache),
            }
            return entry.static_out.clone()

        self._cache.move_to_end(key)
        entry.static_x.copy_(x_in)
        entry.static_c.copy_(c)
        entry.static_pos.copy_(input_pos)
        if entry.static_mask is not None and mask is not None:
            entry.static_mask.copy_(mask)
        entry.graph.replay()
        self.last_call_info = {
            "mode": "graph",
            "reason": "hit",
            "shape": tuple(x_in.shape),
            "cache_size": len(self._cache),
        }
        return entry.static_out.clone()

    def _capture(
        self,
        x_in: torch.Tensor,
        c: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> _GraphEntry | None:
        static_x = x_in.clone()
        static_c = c.clone()
        static_pos = input_pos.clone()
        # The mask content is constant (all-True; see module docstring) and
        # ignored by Attention via `_assume_full_mask`, but keep a static
        # clone so the captured graph never dereferences a freed tensor.
        static_mask = mask.contiguous().clone() if mask is not None else None
        try:
            # Eager warm run: cuDNN/flash-attn autotuning and lazy inits
            # must happen outside stream capture.
            with torch.no_grad():
                _ = self.transformer(static_x, static_c, static_pos, static_mask)
            torch.accelerator.synchronize(x_in.device)

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_out = self.transformer(static_x, static_c, static_pos, static_mask)
        except Exception:
            logger.warning(
                "Disabling S2Mel DiT CUDA graphs after capture failure for shape=%s",
                tuple(x_in.shape),
                exc_info=True,
            )
            self.enabled = False
            return None
        logger.debug("Captured S2Mel DiT CUDA graph for shape=%s", tuple(x_in.shape))
        return _GraphEntry(
            graph=graph,
            static_x=static_x,
            static_c=static_c,
            static_pos=static_pos,
            static_mask=static_mask,
            static_out=static_out,
        )

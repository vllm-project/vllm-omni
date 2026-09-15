# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-shape CUDA graph for one AuK DiT denoise step."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer

logger = init_logger(__name__)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_x_mask: torch.Tensor
    static_text: torch.Tensor
    static_c_mask: torch.Tensor
    static_ref: torch.Tensor
    static_ref_mask: torch.Tensor
    static_timestep: torch.Tensor
    static_cfg: torch.Tensor | None
    static_out: torch.Tensor


class AuKCUDAGraphWrapper:
    """Replay one DiT denoise step and leave Euler scheduling to the caller.

    The graph is keyed by the target, text and reference sequence lengths plus
    whether the CFG branch is enabled. Timestep and the CFG strength are
    mutable scalar buffers, so all Euler steps and CFG values within one path
    reuse one graph.
    """

    _TARGET_ALIGNMENT = 64
    _TEXT_ALIGNMENT = 64
    _REF_ALIGNMENT = 50

    def __init__(self, dit: AuKTransformer, *, enabled: bool = True, max_graphs: int = 32) -> None:
        self.dit = dit
        self.enabled = bool(enabled)
        self.max_graphs = max(1, int(max_graphs))
        self._cache: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self._pool_handle: int | None = None

    @classmethod
    def _bucket_inputs(
        cls,
        x: torch.Tensor,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_bucket = round_up(x.shape[1], cls._TARGET_ALIGNMENT)
        text_bucket = round_up(text.shape[1], cls._TEXT_ALIGNMENT)
        ref_bucket = round_up(ref.shape[1], cls._REF_ALIGNMENT)
        x_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        return (
            F.pad(x, (0, 0, 0, target_bucket - x.shape[1])),
            F.pad(x_mask, (0, target_bucket - x_mask.shape[1]), value=False),
            F.pad(text, (0, 0, 0, text_bucket - text.shape[1])),
            F.pad(c_mask, (0, text_bucket - c_mask.shape[1]), value=False),
            F.pad(ref, (0, 0, 0, ref_bucket - ref.shape[1])),
            F.pad(ref_mask, (0, ref_bucket - ref_mask.shape[1]), value=False),
        )

    @staticmethod
    def _key(
        x: torch.Tensor,
        text: torch.Tensor,
        ref: torch.Tensor,
        uses_cfg: bool,
    ) -> tuple[int, int, int, bool]:
        return (x.shape[1], text.shape[1], ref.shape[1], uses_cfg)

    @torch.no_grad()
    def __call__(
        self,
        *,
        x: torch.Tensor,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timestep: torch.Tensor,
        cfg_strength: float,
    ) -> torch.Tensor:
        inputs = (x, text, c_mask, ref, ref_mask, timestep)
        uses_cfg = cfg_strength >= 1e-5
        if not self.enabled or x.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            if uses_cfg:
                return self._run_cfg(x, None, text, c_mask, ref, ref_mask, timestep, cfg_strength=cfg_strength)
            return self._run(x, None, text, c_mask, ref, ref_mask, timestep)

        target_frames = x.shape[1]
        x, x_mask, text, c_mask, ref, ref_mask = self._bucket_inputs(x, text, c_mask, ref, ref_mask)
        inputs = (x, x_mask, text, c_mask, ref, ref_mask, timestep)
        key = self._key(x, text, ref, uses_cfg)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._capture(*inputs, cfg_strength=cfg_strength, uses_cfg=uses_cfg)
            if len(self._cache) >= self.max_graphs:
                self._cache.popitem(last=False)
            self._cache[key] = entry
        else:
            self._cache.move_to_end(key)

        entry.static_x.copy_(x)
        entry.static_x_mask.copy_(x_mask)
        entry.static_text.copy_(text)
        entry.static_c_mask.copy_(c_mask)
        entry.static_ref.copy_(ref)
        entry.static_ref_mask.copy_(ref_mask)
        entry.static_timestep.copy_(timestep)
        if entry.static_cfg is not None:
            entry.static_cfg.fill_(cfg_strength)
        entry.graph.replay()
        return entry.static_out[:, :target_frames].clone()

    def _run(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.dit(x, text, timestep, mask=x_mask, c_mask=c_mask, ref=ref, ref_mask=ref_mask)

    def _run_cfg(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timestep: torch.Tensor,
        *,
        cfg_strength: torch.Tensor | float,
    ) -> torch.Tensor:
        pred = self.dit(
            x,
            text,
            timestep,
            mask=x_mask,
            c_mask=c_mask,
            ref=ref,
            ref_mask=ref_mask,
            cfg_infer=True,
            cache=True,
        )
        conditional, unconditional = pred.chunk(2, dim=0)
        return conditional + (conditional - unconditional) * cfg_strength

    def _capture(
        self,
        *inputs: torch.Tensor,
        cfg_strength: float,
        uses_cfg: bool,
    ) -> _GraphEntry:
        static_inputs = tuple(value.clone() for value in inputs)
        static_cfg = None
        if uses_cfg:
            static_cfg = torch.empty((), device=static_inputs[0].device, dtype=torch.float32)
            static_cfg.fill_(cfg_strength)
        try:
            for _ in range(3):
                if uses_cfg:
                    assert static_cfg is not None
                    self._run_cfg(*static_inputs, cfg_strength=static_cfg)
                else:
                    self._run(*static_inputs)
            # CFG warm-up populates AuKTransformer's Python-side projected-text
            # cache. Clear it before capture so the graph includes
            # project_text(static_text), rather than closing over the first
            # request's projection and ignoring later static_text updates.
            self.dit.clear_cache()
            if static_cfg is not None:
                static_cfg.fill_(cfg_strength)
            if self._pool_handle is None:
                self._pool_handle = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool_handle):
                # Capture the actual CFG path. The scalar value remains a
                # mutable buffer, so cfg=2 and cfg=3 share this graph.
                if uses_cfg:
                    assert static_cfg is not None
                    static_out = self._run_cfg(*static_inputs, cfg_strength=static_cfg)
                else:
                    static_out = self._run(*static_inputs)
        finally:
            self.dit.clear_cache()

        logger.info(
            "Captured AuK DiT single-step CUDA graph: target_frames=%d text_tokens=%d ref_frames=%d cfg=%s",
            inputs[0].shape[1],
            inputs[2].shape[1],
            inputs[4].shape[1],
            cfg_strength,
        )
        return _GraphEntry(
            graph=graph,
            static_x=static_inputs[0],
            static_x_mask=static_inputs[1],
            static_text=static_inputs[2],
            static_c_mask=static_inputs[3],
            static_ref=static_inputs[4],
            static_ref_mask=static_inputs[5],
            static_timestep=static_inputs[6],
            static_cfg=static_cfg,
            static_out=static_out,
        )


__all__ = ["AuKCUDAGraphWrapper"]

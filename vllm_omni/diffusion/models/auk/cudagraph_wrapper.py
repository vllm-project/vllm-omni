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

from vllm_omni.diffusion.models.auk.auk_transformer import AuKStepContext, AuKTransformer

logger = init_logger(__name__)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_ctx: AuKStepContext
    static_timestep: torch.Tensor
    static_cfg: torch.Tensor | None
    static_step: torch.Tensor | None
    static_out: torch.Tensor


class AuKCUDAGraphWrapper:
    """Replay one DiT denoise step and leave Euler scheduling to the caller.

    The graph is keyed by the target, text and reference sequence lengths,
    whether the CFG branch is enabled and, when the caller passes the time
    grid, its length. Timestep (or the step index), and the CFG strength are
    mutable scalar buffers, so all Euler steps and CFG values within one path
    reuse one graph. With the time grid, the adaLN modulations of every step
    are part of the per-request context and the graph selects its row.

    Only :meth:`AuKTransformer.step` is captured. The per-request context
    (text projection, reference embedding, padding biases, rotary tables) is
    built by :meth:`AuKTransformer.prepare` on the first step of a request and
    copied into the graph's static context, so the replayed steps skip it.
    """

    _TARGET_ALIGNMENT = 32
    _TEXT_ALIGNMENT = 32
    _REF_ALIGNMENT = 50

    def __init__(self, dit: AuKTransformer, *, enabled: bool = True, max_graphs: int = 32) -> None:
        self.dit = dit
        self.enabled = bool(enabled)
        self.max_graphs = max(1, int(max_graphs))
        self._cache: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self._pool_handle: int | None = None
        # Key whose static context holds the current request, and the context
        # the eager fallback reuses across the steps of one request.
        self._active_key: tuple | None = None
        self._eager_ctx: AuKStepContext | None = None
        logger.info("Initialized AuK DiT lazy CUDA graph cache: enabled=%s slots=%d", self.enabled, self.max_graphs)

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
        steps: int = 0,
    ) -> tuple[int, int, int, bool, int]:
        return (x.shape[1], text.shape[1], ref.shape[1], uses_cfg, steps)

    def _retire_graph_generation_if_full(self) -> None:
        """Retire all graphs together so none outlive shared workspaces."""
        if len(self._cache) >= self.max_graphs:
            self._cache.clear()
            self._active_key = None

    def _prepare(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        uses_cfg: bool,
        timesteps: torch.Tensor | None = None,
    ) -> AuKStepContext:
        return self.dit.prepare(
            text,
            target_len=x.shape[1],
            mask=x_mask,
            c_mask=c_mask,
            ref=ref,
            ref_mask=ref_mask,
            cfg_infer=uses_cfg,
            timesteps=timesteps,
        )

    def _step(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        ctx: AuKStepContext,
        cfg_strength: torch.Tensor | float,
        step_index: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """One guided (or unguided) velocity; the CFG combine is part of the graph."""
        velocity = self.dit.step(x, timestep, ctx, step_index=step_index)
        if ctx.branches == 1:
            return velocity
        conditional, unconditional = velocity.chunk(2, dim=0)
        return conditional + (conditional - unconditional) * cfg_strength

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
        new_request: bool = True,
        timesteps: torch.Tensor | None = None,
        step_index: int | None = None,
    ) -> torch.Tensor:
        """Velocity of ``x`` at ``timestep``.

        ``new_request`` marks the first step of a request, whose conditioning
        (``text``, ``c_mask``, ``ref``, ``ref_mask``) is then prepared anew; the
        later steps of the request pass ``False`` and reuse it. The default
        prepares on every call, which is always correct.

        ``timesteps`` (the times every step of the request runs at) and
        ``step_index`` (this step's position in them) let the adaLN
        modulations be computed once per request instead of once per step.
        """
        uses_cfg = cfg_strength >= 1e-5
        if timesteps is None or step_index is None:
            timesteps, step_index = None, None
        if not self.enabled or x.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            if new_request or self._eager_ctx is None:
                self._eager_ctx = self._prepare(x, None, text, c_mask, ref, ref_mask, uses_cfg, timesteps)
            return self._step(x, timestep, self._eager_ctx, cfg_strength, step_index)

        target_frames = x.shape[1]
        x, x_mask, text, c_mask, ref, ref_mask = self._bucket_inputs(x, text, c_mask, ref, ref_mask)
        steps = 0 if timesteps is None else int(timesteps.numel())
        key = self._key(x, text, ref, uses_cfg, steps)
        entry = self._cache.get(key)
        ctx = None
        if entry is None or new_request or key != self._active_key:
            ctx = self._prepare(x, x_mask, text, c_mask, ref, ref_mask, uses_cfg, timesteps)
        if entry is None:
            # Same Retirement as #6587, details discussed in #7469
            self._retire_graph_generation_if_full()
            try:
                assert ctx is not None
                entry = self._capture(x, ctx, timestep, cfg_strength=cfg_strength, step_index=step_index)
            except Exception:
                self._cache.clear()
                self._active_key = None
                self.enabled = False
                logger.exception(
                    "Disabling AuK DiT CUDA graphs after capture failure for key=%s; "
                    "subsequent requests will use eager execution.",
                    key,
                )
                raise
            self._cache[key] = entry
        else:
            self._cache.move_to_end(key)
            if ctx is not None:
                entry.static_ctx.copy_(ctx)
        self._active_key = key

        entry.static_x.copy_(x)
        entry.static_timestep.copy_(timestep)
        if entry.static_step is not None:
            entry.static_step.fill_(step_index)
        if entry.static_cfg is not None:
            entry.static_cfg.fill_(cfg_strength)
        entry.graph.replay()
        return entry.static_out[:, :target_frames].clone()

    def _capture(
        self,
        x: torch.Tensor,
        ctx: AuKStepContext,
        timestep: torch.Tensor,
        *,
        cfg_strength: float,
        step_index: int | None = None,
    ) -> _GraphEntry:
        static_x = x.clone()
        static_ctx = ctx.clone()
        static_timestep = timestep.clone()
        static_step = None
        if static_ctx.modulation is not None and step_index is not None:
            static_step = torch.full((), step_index, device=static_x.device, dtype=torch.long)
        static_cfg = None
        if ctx.branches == 2:
            static_cfg = torch.empty((), device=static_x.device, dtype=torch.float32)
            static_cfg.fill_(cfg_strength)
        scale = static_cfg if static_cfg is not None else cfg_strength
        # Warm-up runs any lazy torch.compile outside the capture.
        for _ in range(3):
            self._step(static_x, static_timestep, static_ctx, scale, static_step)
        if self._pool_handle is None:
            self._pool_handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool_handle):
            # The CFG scalar stays a mutable buffer, so cfg=2 and cfg=3 share this graph.
            static_out = self._step(static_x, static_timestep, static_ctx, scale, static_step)

        logger.info(
            "Captured AuK DiT single-step CUDA graph: target_frames=%d text_tokens=%d ref_frames=%d cfg=%s",
            x.shape[1],
            ctx.c.shape[1],
            0 if ctx.prompt is None else ctx.prompt.shape[1],
            cfg_strength,
        )
        return _GraphEntry(
            graph=graph,
            static_x=static_x,
            static_ctx=static_ctx,
            static_timestep=static_timestep,
            static_cfg=static_cfg,
            static_step=static_step,
            static_out=static_out,
        )


__all__ = ["AuKCUDAGraphWrapper"]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bucketed CUDA graphs for AuK DiT steps and complete Euler loops."""

from __future__ import annotations

import gc
import traceback
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


@dataclass
class _LoopGraphEntry:
    graph: torch.cuda.CUDAGraph
    static_x_in: torch.Tensor
    static_ctx: AuKStepContext
    static_cfg: torch.Tensor | None
    static_x_out: torch.Tensor
    static_dts: torch.Tensor
    static_timesteps: torch.Tensor
    static_step_indices: torch.Tensor


class AuKCUDAGraphWrapper:
    """Replay one DiT denoise step or the entire diffusion loop with CUDA graphs.

    The graph is keyed by the target, text and reference sequence lengths,
    whether the CFG branch is enabled and, when the caller passes the time
    grid, its length. Timestep (or the step index), and the CFG strength are
    mutable scalar buffers, so all Euler steps and CFG values within one path
    reuse one graph. With the time grid, the adaLN modulations of every step
    are part of the per-request context and the graph selects its row.

    The single-step path captures :meth:`AuKTransformer.step`; the loop path
    also captures the Euler updates. The per-request context
    (text projection, reference embedding, padding biases, rotary tables) is
    built by :meth:`AuKTransformer.prepare` on the first step of a request and
    copied into the graph's static context, so the replayed steps skip it.
    """

    _TARGET_ALIGNMENT = 32
    _TEXT_ALIGNMENT = 32
    _REF_ALIGNMENT = 50
    _BATCH_LADDER = (1, 2, 4, 8, 16, 32, 64, 128)
    _MAX_OOM_LOOP_RETRIES = 1
    _OOM_COOLDOWN_CALLS = 16

    @classmethod
    def _batch_bucket(cls, batch_size: int) -> int:
        """Find the smallest batch ladder step >= batch_size."""
        for b in cls._BATCH_LADDER:
            if batch_size <= b:
                return b
        return batch_size

    @staticmethod
    def _is_oom_capture_error(exc: BaseException) -> bool:
        """Return True when a capture failure is caused by CUDA out-of-memory."""
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, torch.cuda.OutOfMemoryError):
                return True
            if isinstance(cur, RuntimeError) and "out of memory" in str(cur).lower():
                return True
            cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
        return False

    def __init__(self, dit: AuKTransformer, *, enabled: bool = True, max_graphs: int = 32) -> None:
        self.dit = dit
        self.enabled = bool(enabled)
        self.max_graphs = max(1, int(max_graphs))
        self._cache: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self._loop_cache: OrderedDict[tuple, _LoopGraphEntry] = OrderedDict()
        # Deterministic capture failures (and OOM-exhausted keys) stay eager for
        # the process lifetime; transient OOMs cool down and retry once after
        # generation retirement.
        self._uncapturable_loop_keys: set[tuple] = set()
        self._oom_loop_cooldowns: dict[tuple, int] = {}
        self._oom_loop_retry_counts: dict[tuple, int] = {}
        # Single pool shared by single-step and loop graphs, retired atomically.
        # Safe because both __call__ and sample_loop clone static outputs before return.
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
        x_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_bucket = round_up(x.shape[1], cls._TARGET_ALIGNMENT)
        text_bucket = round_up(text.shape[1], cls._TEXT_ALIGNMENT)
        ref_bucket = round_up(ref.shape[1], cls._REF_ALIGNMENT)
        if x_mask is None:
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
    ) -> tuple[int, int, int, int, bool, int]:
        return (x.shape[0], x.shape[1], text.shape[1], ref.shape[1], uses_cfg, steps)

    def _retire_graph_generation_if_full(self) -> None:
        """Retire all graphs together so none outlive shared workspaces."""
        if len(self._cache) + len(self._loop_cache) >= self.max_graphs:
            self._cache.clear()
            self._loop_cache.clear()
            self._active_key = None
            self._pool_handle = None
            # Retirement frees the pool held by the previous graph generation,
            # so keys that only hit transient CUDA OOM can retry once; deterministic
            # (or OOM-exhausted) failures remain in _uncapturable_loop_keys.
            self._oom_loop_cooldowns.clear()

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
        x_mask: torch.Tensor | None = None,
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
                self._eager_ctx = self._prepare(x, x_mask, text, c_mask, ref, ref_mask, uses_cfg, timesteps)
            step_idx: torch.Tensor | int | None = step_index
            if step_index is not None and self._eager_ctx.modulation is not None:
                step_idx = torch.full((), step_index, device=x.device, dtype=torch.long)
            return self._step(x, timestep, self._eager_ctx, cfg_strength, step_idx)

        target_frames = x.shape[1]
        x, x_mask, text, c_mask, ref, ref_mask = self._bucket_inputs(x, text, c_mask, ref, ref_mask, x_mask)
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
                self._loop_cache.clear()
                self._oom_loop_cooldowns.clear()
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
            "Captured AuK DiT single-step CUDA graph: batch_size=%d "
            "target_frames=%d text_tokens=%d ref_frames=%d cfg=%s",
            x.shape[0],
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

    def _run_eager_loop(
        self,
        *,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timesteps: torch.Tensor,
        cfg_strength: float,
    ) -> torch.Tensor:
        """Run the multi-step Euler denoising loop eagerly without capturing CUDA graphs."""
        n_steps = int(timesteps.numel()) - 1
        if n_steps <= 0:
            return x
        uses_cfg = cfg_strength >= 1e-5
        ctx = self._prepare(x, x_mask, text, c_mask, ref, ref_mask, uses_cfg, timesteps[:-1])
        step_indices = torch.arange(n_steps, device=x.device, dtype=torch.long) if ctx.modulation is not None else None
        for i in range(n_steps):
            step_idx = None if step_indices is None else step_indices[i]
            vel = self._step(x, timesteps[i], ctx, cfg_strength, step_idx)
            dt = (timesteps[i + 1] - timesteps[i]).to(x.device, x.dtype)
            x = x + dt * vel
        return x

    @torch.no_grad()
    def sample_loop(
        self,
        *,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timesteps: torch.Tensor,
        cfg_strength: float,
    ) -> torch.Tensor:
        """Sample latents over the full Euler time grid inside a single CUDA graph."""
        n_steps = int(timesteps.numel()) - 1
        uses_cfg = cfg_strength >= 1e-5
        if not self.enabled or x.device.type != "cuda" or torch.cuda.is_current_stream_capturing() or n_steps <= 0:
            return self._run_eager_loop(
                x=x,
                x_mask=x_mask,
                text=text,
                c_mask=c_mask,
                ref=ref,
                ref_mask=ref_mask,
                timesteps=timesteps,
                cfg_strength=cfg_strength,
            )

        target_frames = x.shape[1]
        orig_batch = x.shape[0]
        orig_x_mask = x_mask
        target_bucket = round_up(target_frames, self._TARGET_ALIGNMENT)
        text_bucket = round_up(text.shape[1], self._TEXT_ALIGNMENT)
        ref_bucket = round_up(ref.shape[1], self._REF_ALIGNMENT)
        batch_bucket = self._batch_bucket(orig_batch)
        key = (batch_bucket, target_bucket, text_bucket, ref_bucket, uses_cfg, n_steps)
        if key in self._uncapturable_loop_keys:
            return self._run_eager_loop(
                x=x,
                x_mask=orig_x_mask,
                text=text,
                c_mask=c_mask,
                ref=ref,
                ref_mask=ref_mask,
                timesteps=timesteps,
                cfg_strength=cfg_strength,
            )
        cooldown = self._oom_loop_cooldowns.get(key)
        if cooldown is not None:
            if cooldown > 1:
                self._oom_loop_cooldowns[key] = cooldown - 1
            else:
                del self._oom_loop_cooldowns[key]
            return self._run_eager_loop(
                x=x,
                x_mask=orig_x_mask,
                text=text,
                c_mask=c_mask,
                ref=ref,
                ref_mask=ref_mask,
                timesteps=timesteps,
                cfg_strength=cfg_strength,
            )
        entry = self._loop_cache.get(key)
        hit_key = key if entry is not None else None
        if entry is None:
            hit_key, entry = self._larger_loop_entry(
                orig_batch, target_bucket, text_bucket, ref_bucket, uses_cfg, n_steps
            )
        capture_failed = False
        capture_is_oom = False
        if entry is None:
            try:
                entry = self._prepare_and_capture_loop(
                    batch_bucket=batch_bucket,
                    x=x,
                    x_mask=orig_x_mask,
                    text=text,
                    c_mask=c_mask,
                    ref=ref,
                    ref_mask=ref_mask,
                    timesteps=timesteps,
                    uses_cfg=uses_cfg,
                    cfg_strength=cfg_strength,
                )
            except Exception as exc:
                self._cache.clear()
                self._loop_cache.clear()
                self._active_key = None
                self._pool_handle = None
                self._oom_loop_cooldowns.clear()
                capture_is_oom = self._is_oom_capture_error(exc)
                if capture_is_oom:
                    retries = self._oom_loop_retry_counts.get(key, 0) + 1
                    self._oom_loop_retry_counts[key] = retries
                    if retries <= self._MAX_OOM_LOOP_RETRIES:
                        self._oom_loop_cooldowns[key] = self._OOM_COOLDOWN_CALLS
                        logger.warning(
                            "AuK DiT multi-step CUDA graph capture hit transient OOM for key=%s "
                            "(attempt %d/%d); falling back to eager until next cache retirement.",
                            key,
                            retries,
                            self._MAX_OOM_LOOP_RETRIES + 1,
                            exc_info=True,
                        )
                    else:
                        self._uncapturable_loop_keys.add(key)
                        logger.warning(
                            "AuK DiT multi-step CUDA graph capture failed with repeated OOM for key=%s "
                            "after %d attempts; keeping eager loop permanently.",
                            key,
                            retries,
                            exc_info=True,
                        )
                else:
                    self._uncapturable_loop_keys.add(key)
                    logger.warning(
                        "AuK DiT multi-step CUDA graph capture failed for key=%s; falling back to eager loop.",
                        key,
                        exc_info=True,
                    )
                cur_exc: BaseException | None = exc
                seen_exc: set[int] = set()
                while cur_exc is not None and id(cur_exc) not in seen_exc:
                    seen_exc.add(id(cur_exc))
                    if cur_exc.__traceback__ is not None:
                        traceback.clear_frames(cur_exc.__traceback__)
                    cur_exc = cur_exc.__cause__ or cur_exc.__context__
                capture_failed = True
            else:
                self._loop_cache[key] = entry
                self._oom_loop_cooldowns.pop(key, None)
                self._oom_loop_retry_counts.pop(key, None)
        else:
            assert hit_key is not None
            self._loop_cache.move_to_end(hit_key)
            x_buck, x_mask_buck, text_buck, c_mask_buck, ref_buck, ref_mask_buck = self._bucket_inputs(
                x, text, c_mask, ref, ref_mask, orig_x_mask
            )
            x_buck, x_mask_buck, text_buck, c_mask_buck, ref_buck, ref_mask_buck = self._fit_loop_inputs(
                entry, x_buck, x_mask_buck, text_buck, c_mask_buck, ref_buck, ref_mask_buck
            )
            ctx = self._prepare(
                x_buck, x_mask_buck, text_buck, c_mask_buck, ref_buck, ref_mask_buck, uses_cfg, timesteps[:-1]
            )
            entry.static_ctx.copy_(ctx)
            entry.static_x_in.copy_(x_buck)

        if capture_failed:
            if capture_is_oom:
                gc.collect()
                if torch.cuda.is_available():
                    torch.accelerator.empty_cache()
            return self._run_eager_loop(
                x=x,
                x_mask=orig_x_mask,
                text=text,
                c_mask=c_mask,
                ref=ref,
                ref_mask=ref_mask,
                timesteps=timesteps,
                cfg_strength=cfg_strength,
            )

        assert entry is not None
        if entry.static_cfg is not None:
            entry.static_cfg.fill_(cfg_strength)
        entry.static_timesteps.copy_(timesteps.float())
        entry.static_dts.copy_(torch.diff(timesteps.float()))
        entry.graph.replay()
        # The graph owns this buffer and overwrites it on the next replay.
        return entry.static_x_out[:orig_batch, :target_frames].clone()

    def _prepare_and_capture_loop(
        self,
        *,
        batch_bucket: int,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timesteps: torch.Tensor,
        uses_cfg: bool,
        cfg_strength: float,
    ) -> _LoopGraphEntry:
        """Retire full caches, pad bucketed inputs, prepare context, and capture a loop graph."""
        self._retire_graph_generation_if_full()
        x_buck, x_mask_buck, text_buck, c_mask_buck, ref_buck, ref_mask_buck = self._bucket_inputs(
            x, text, c_mask, ref, ref_mask, x_mask
        )
        if batch_bucket > x_buck.shape[0]:
            x_buck = self._pad_batch_time(x_buck, batch_bucket, x_buck.shape[1])
            x_mask_buck = self._pad_batch_time(x_mask_buck, batch_bucket, x_mask_buck.shape[1], mask=True)
            text_buck = self._pad_batch_time(text_buck, batch_bucket, text_buck.shape[1])
            c_mask_buck = self._pad_c_mask(c_mask_buck, batch_bucket, c_mask_buck.shape[1])
            ref_buck = self._pad_batch_time(ref_buck, batch_bucket, ref_buck.shape[1])
            ref_mask_buck = self._pad_batch_time(ref_mask_buck, batch_bucket, ref_mask_buck.shape[1], mask=True)
        ctx = self._prepare(
            x_buck, x_mask_buck, text_buck, c_mask_buck, ref_buck, ref_mask_buck, uses_cfg, timesteps[:-1]
        )
        return self._capture_loop(
            x_buck,
            ctx,
            timesteps=timesteps,
            cfg_strength=cfg_strength,
        )

    def _larger_loop_entry(
        self,
        batch: int,
        frames: int,
        text: int,
        ref: int,
        uses_cfg: bool,
        n_steps: int,
    ) -> tuple[tuple | None, _LoopGraphEntry | None]:
        """Reuse a nearby batch graph with the same sequence buckets.

        A 127-request wave pads into the B=128 serving graph instead of capturing
        on the first packet. Batch padding is limited to twice the requested size.
        """
        best: tuple[int, tuple, _LoopGraphEntry] | None = None
        for cache_key, entry in self._loop_cache.items():
            cached_b, cached_f, cached_t, cached_r, cfg, steps = cache_key
            if (cached_f, cached_t, cached_r, cfg, steps) != (frames, text, ref, uses_cfg, n_steps):
                continue
            if not batch <= cached_b <= batch * 2:
                continue
            if best is None or cached_b < best[0]:
                best = (cached_b, cache_key, entry)
        return (None, None) if best is None else (best[1], best[2])

    @staticmethod
    def _pad_batch_time(tensor: torch.Tensor, batch: int, seq: int, *, mask: bool = False) -> torch.Tensor:
        """Right-pad a ``[B, T, ...]`` (or ``[B, T]`` mask) to ``(batch, seq)``."""
        pad_b = batch - tensor.shape[0]
        pad_t = seq - tensor.shape[1]
        if pad_b == 0 and pad_t == 0:
            return tensor
        if tensor.ndim == 3:
            return F.pad(tensor, (0, 0, 0, pad_t, 0, pad_b))
        return F.pad(tensor, (0, pad_t, 0, pad_b), value=False if mask else 0)

    @staticmethod
    def _pad_c_mask(c_mask: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
        """Pad ``c_mask`` to ``(batch, seq)`` while keeping at least one valid token on padded rows."""
        orig_b = c_mask.shape[0]
        padded = AuKCUDAGraphWrapper._pad_batch_time(c_mask, batch, seq, mask=True)
        if batch > orig_b and padded.shape[1] > 0:
            padded = padded.clone()
            padded[orig_b:, 0] = True
        return padded

    def _fit_loop_inputs(
        self,
        entry: _LoopGraphEntry,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad bucketed inputs to the static buffers of a larger warmed loop graph."""
        graph_b, graph_frames = int(entry.static_x_in.shape[0]), int(entry.static_x_in.shape[1])
        graph_text = int(entry.static_ctx.c.shape[1])
        graph_ref = 0 if entry.static_ctx.prompt is None else int(entry.static_ctx.prompt.shape[1])
        return (
            self._pad_batch_time(x, graph_b, graph_frames),
            self._pad_batch_time(x_mask, graph_b, graph_frames, mask=True),
            self._pad_batch_time(text, graph_b, graph_text),
            self._pad_c_mask(c_mask, graph_b, graph_text),
            self._pad_batch_time(ref, graph_b, graph_ref),
            self._pad_batch_time(ref_mask, graph_b, graph_ref, mask=True),
        )

    def _capture_loop(
        self,
        x: torch.Tensor,
        ctx: AuKStepContext,
        *,
        timesteps: torch.Tensor,
        cfg_strength: float,
    ) -> _LoopGraphEntry:
        static_x_in = x.clone()
        static_ctx = ctx.clone()
        static_cfg = None
        if ctx.branches == 2:
            static_cfg = torch.empty((), device=x.device, dtype=torch.float32)
            static_cfg.fill_(cfg_strength)
        scale = static_cfg if static_cfg is not None else cfg_strength

        n_steps = int(timesteps.numel()) - 1
        static_timesteps = timesteps.clone().float()
        static_dts = torch.diff(static_timesteps).float()
        static_step_indices = torch.arange(n_steps, device=x.device, dtype=torch.long)

        def run_loop(cur_x: torch.Tensor) -> torch.Tensor:
            for i in range(n_steps):
                vel = self._step(cur_x, static_timesteps[i], static_ctx, scale, step_index=static_step_indices[i])
                cur_x = cur_x + static_dts[i] * vel
            return cur_x

        for _ in range(3):
            run_loop(static_x_in)

        if self._pool_handle is None:
            self._pool_handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool_handle):
            static_x_out = run_loop(static_x_in)

        logger.info(
            "Captured AuK DiT multi-step (%d steps) CUDA graph: "
            "batch_size=%d target_frames=%d text_tokens=%d ref_frames=%d cfg=%s",
            n_steps,
            x.shape[0],
            x.shape[1],
            ctx.c.shape[1],
            0 if ctx.prompt is None else ctx.prompt.shape[1],
            cfg_strength,
        )
        return _LoopGraphEntry(
            graph=graph,
            static_x_in=static_x_in,
            static_ctx=static_ctx,
            static_cfg=static_cfg,
            static_x_out=static_x_out,
            static_dts=static_dts,
            static_timesteps=static_timesteps,
            static_step_indices=static_step_indices,
        )


__all__ = ["AuKCUDAGraphWrapper"]

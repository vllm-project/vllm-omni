# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import time
from collections import OrderedDict

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput, KVPrefetchJob
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionKVCacheSpec,
    SupportsARDiffusionPipeline,
    SupportsARDiffusionWarmup,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

logger = init_logger(__name__)


def resolve_ar_diffusion_kv_config(od_config: OmniDiffusionConfig) -> ARDiffusionKVConfig:
    """Resolve optional deployment overrides and enable AR-Diffusion KV."""
    raw = getattr(od_config, "ar_diffusion_kv_config", None)
    if raw is None:
        model_config = getattr(od_config, "model_config", None)
        if isinstance(model_config, dict):
            raw = model_config.get("ar_diffusion_kv_config")
    if isinstance(raw, ARDiffusionKVConfig):
        return dataclasses.replace(raw, enable=True)
    if isinstance(raw, dict):
        return ARDiffusionKVConfig(**{**raw, "enable": True})
    return ARDiffusionKVConfig(enable=True)


class ARDiffusionModelRunner(DiffusionModelRunner):
    """Own AR KV pools and session lifetime for a capable diffusion pipeline.

    The runner never inspects model internals. At load time it consumes an
    :class:`ARDiffusionKVCacheSpec`; during execution it binds a runner-owned
    :class:`ARDiffusionKVState` through the pipeline's context manager. Sessions
    persist across requests and end on explicit close, LRU eviction, or failed
    forward. A request with ``extra_args["reset"]`` releases the old session
    before executing with a fresh one.
    """

    _WARMUP_SID = "__ardiffusion_warmup__"
    _MAX_RESIDENT_SESSIONS = 1

    def __init__(self, vllm_config: object, od_config: OmniDiffusionConfig, device: torch.device) -> None:
        super().__init__(vllm_config, od_config, device)
        self.ar_diffusion_kv_config = resolve_ar_diffusion_kv_config(od_config)
        self.kv_cache: ARDiffusionKVCache | None = None
        self._ar_diffusion_capability: SupportsARDiffusionPipeline | None = None
        self._ar_diffusion_kv_cache_spec: ARDiffusionKVCacheSpec | None = None
        self._sessions: OrderedDict[str, ARDiffusionKVState] = OrderedDict()
        self._session_capacity = 0
        self._perf_e2e_times: list[float] = []

    @staticmethod
    def _require_capability(pipeline: object) -> SupportsARDiffusionPipeline:
        if not isinstance(pipeline, SupportsARDiffusionPipeline):
            raise TypeError(
                "ARDiffusionEngine requires the loaded pipeline to implement "
                "SupportsARDiffusionPipeline: ar_diffusion_kv_cache_spec(), "
                "bind_ar_diffusion_state(), reset_ar_diffusion_session(), and "
                "close_ar_diffusion_session(). Select the default DiffusionEngine "
                "or add an explicit AR-Diffusion capability adapter to the model."
            )
        return pipeline

    def load_model(self, *args: object, **kwargs: object) -> None:
        super().load_model(*args, **kwargs)
        if self.pipeline is None:
            return
        self._preallocate_kv_cache()
        if not self.od_config.enforce_eager and self.ar_diffusion_kv_config.warmup_cudagraph:
            self._warmup_ar_rollout()

    def _available_memory_bytes(self) -> int:
        if self.device is None or torch.device(self.device).type != "cuda":
            raise RuntimeError("AR-Diffusion KV preallocation currently requires a CUDA device")
        return int(torch.cuda.mem_get_info(self.device)[0])

    def _preallocate_kv_cache(self, *, available_bytes: int | None = None) -> None:
        """Build pools solely from the pipeline capability and runner config."""
        if bool(getattr(self.od_config, "step_execution", False)):
            raise ValueError(
                "ARDiffusionModelRunner currently supports request-mode execution only; "
                "step_execution=True would bypass per-request AR session binding."
            )
        max_num_seqs = int(getattr(self.od_config, "max_num_seqs", 1) or 1)
        if max_num_seqs > 1:
            raise ValueError(
                "AR-Diffusion paged KV supports max_num_seqs=1 (single-sequence "
                f"rollouts); got max_num_seqs={max_num_seqs}."
            )
        capability = self._require_capability(self.pipeline)
        if bool(getattr(self.pipeline, "supports_request_batch", False)):
            raise ValueError(
                "ARDiffusionModelRunner currently supports one request at a time; "
                "request-batch execution would bypass per-request AR session binding."
            )
        spec = capability.ar_diffusion_kv_cache_spec()
        if not isinstance(spec, ARDiffusionKVCacheSpec):
            raise TypeError(
                f"ar_diffusion_kv_cache_spec() must return ARDiffusionKVCacheSpec, got {type(spec).__name__}"
            )

        config = dataclasses.replace(
            self.ar_diffusion_kv_config,
            chunk_size=spec.tokens_per_frame,
            window_chunks=self.ar_diffusion_kv_config.window_chunks or spec.window_frames,
            sink_chunks=self.ar_diffusion_kv_config.sink_chunks or spec.sink_frames,
            reset_at_boundary=self.ar_diffusion_kv_config.reset_at_boundary or spec.reset_at_boundary,
        )
        self.ar_diffusion_kv_config = config
        self._ar_diffusion_capability = capability
        self._ar_diffusion_kv_cache_spec = spec
        self._session_capacity = min(spec.session_capacity, self._MAX_RESIDENT_SESSIONS)
        self.kv_cache = ARDiffusionKVCache(
            config,
            num_layers=spec.num_layers,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=self.od_config.dtype,
            block_size=spec.tokens_per_frame,
            max_model_len=spec.max_model_len,
            available_bytes=self._available_memory_bytes() if available_bytes is None else available_bytes,
            kv_branches=spec.kv_branches,
            session_capacity=self._session_capacity,
            cross_attention_lengths=spec.cross_attention_lengths,
            frames_per_block=spec.frames_per_block,
            max_scratch_tokens_per_branch=spec.max_scratch_tokens_per_branch,
            device=self.device,
        )
        logger.info(
            "AR-Diffusion KV cache: blocks=%d layers=%d local_kv_heads=%d head_size=%d "
            "tokens/frame=%d frames/block=%d window=%d sink=%d kv_branches=%s cross=%s "
            "resident_capacity=%d requested_capacity=%d",
            self.kv_cache.num_blocks,
            spec.num_layers,
            spec.num_kv_heads,
            spec.head_size,
            spec.tokens_per_frame,
            spec.frames_per_block,
            config.window_chunks,
            config.sink_chunks,
            [(kv_branch.name, kv_branch.local_index) for kv_branch in spec.kv_branches],
            spec.cross_attention_lengths,
            self._session_capacity,
            spec.session_capacity,
        )

    def _new_session_state(self, session_id: str) -> ARDiffusionKVState:
        if self.kv_cache is None or self._ar_diffusion_kv_cache_spec is None:
            raise RuntimeError("AR-Diffusion session requested before KV cache initialization")
        adapters = {
            kv_branch.name: self.kv_cache.begin_request(f"ar::{session_id}::{kv_branch.name}")
            for kv_branch in self._ar_diffusion_kv_cache_spec.kv_branches
        }
        return ARDiffusionKVState(
            self.kv_cache,
            session_id,
            adapters,
            num_layers=self._ar_diffusion_kv_cache_spec.num_layers,
        )

    def _release_session(
        self,
        session_id: str,
        *,
        reset_model: bool,
        reason: str,
        suppress_errors: bool = False,
    ) -> None:
        """One release path for reset, close, eviction, and failed forwards."""
        state = self._sessions.pop(session_id, None)
        errors: list[Exception] = []
        if state is not None:
            try:
                state.close()
            except Exception as exc:  # noqa: BLE001 - notify the pipeline even if pool cleanup fails
                errors.append(exc)
        capability = self._ar_diffusion_capability
        if capability is not None:
            try:
                if reset_model:
                    capability.reset_ar_diffusion_session(session_id)
                else:
                    capability.close_ar_diffusion_session(session_id)
            except Exception as exc:  # noqa: BLE001 - preserve all lifecycle cleanup attempts
                errors.append(exc)
        logger.debug("AR-Diffusion released session=%s reason=%s", session_id, reason)
        if errors:
            if suppress_errors:
                logger.warning(
                    "AR-Diffusion session=%s cleanup after %s had %d error(s): %s",
                    session_id,
                    reason,
                    len(errors),
                    errors,
                )
            else:
                raise errors[0]

    def reset_session(self, session_id: str) -> None:
        """Release KV and notify the pipeline to reset model-owned state."""
        self._release_session(session_id, reset_model=True, reason="reset")

    def close_session(self, session_id: str) -> None:
        """Release KV and notify the pipeline to drop model-owned state."""
        self._release_session(session_id, reset_model=False, reason="close")

    def _get_or_create_session(self, session_id: str) -> ARDiffusionKVState:
        state = self._sessions.get(session_id)
        if state is None:
            while len(self._sessions) >= self._session_capacity:
                oldest = next(iter(self._sessions))
                self._release_session(oldest, reset_model=False, reason="lru_eviction")
            state = self._new_session_state(session_id)
            self._sessions[session_id] = state
        self._sessions.move_to_end(session_id)
        return state

    @staticmethod
    def _request_session(req: OmniDiffusionRequest) -> tuple[str, dict]:
        extra_args = req.sampling_params.extra_args or {}
        return str(extra_args.get("session_id") or "default"), extra_args

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
    ) -> DiffusionOutput:
        if self.kv_cache is None:
            return super().execute_model(req, kv_prefetch_job=kv_prefetch_job)
        capability = self._ar_diffusion_capability
        if capability is None:
            raise RuntimeError("AR-Diffusion capability missing after KV cache initialization")

        session_id, extra_args = self._request_session(req)
        if extra_args.get("reset", False):
            self.reset_session(session_id)
        state = self._get_or_create_session(session_id)
        started = time.perf_counter()
        try:
            with capability.bind_ar_diffusion_state(session_id, state):
                output = super().execute_model(req, kv_prefetch_job=kv_prefetch_job)
            if self.device is not None and torch.device(self.device).type == "cuda":
                torch.accelerator.synchronize(self.device)
        except Exception:
            self._release_session(
                session_id,
                reset_model=False,
                reason="forward_exception",
                suppress_errors=True,
            )
            logger.warning(
                "AR-Diffusion forward failed for session=%s; KV and model state were released",
                session_id,
            )
            raise
        self._perf_e2e_times.append(time.perf_counter() - started)
        if extra_args.get("close_session", False):
            self.close_session(session_id)
        return output

    def execute_model_batch(
        self,
        scheduler_output: DiffusionSchedulerOutput,
        od_config: OmniDiffusionConfig,
    ) -> BatchRunnerOutput:
        """Reject request batching until batch-aware AR state binding exists."""
        raise RuntimeError(
            "ARDiffusionModelRunner does not support request-batch execution; use request mode with max_num_seqs=1."
        )

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Reject step execution until step-aware AR state binding exists."""
        raise RuntimeError(
            "ARDiffusionModelRunner does not support step execution; use request mode with step_execution=False."
        )

    def _warmup_ar_rollout(self) -> None:
        """Run model-provided warmup requests, or safely skip when absent."""
        if self.kv_cache is None or self.pipeline is None:
            return
        if not isinstance(self.pipeline, SupportsARDiffusionWarmup):
            logger.info("AR-Diffusion pipeline provides no warmup requests; skipping rollout warmup")
            return
        sid = self._WARMUP_SID
        free_before = self.kv_cache.manager.block_pool.get_num_free_blocks()
        try:
            for request in self.pipeline.ar_diffusion_warmup_requests(sid):
                self.execute_model(request)
        except Exception as exc:  # noqa: BLE001 - warmup must not make model load fail
            logger.warning("AR-Diffusion rollout warmup failed (%s); using lazy capture", exc)
        finally:
            self._release_session(
                sid,
                reset_model=False,
                reason="warmup_complete",
                suppress_errors=True,
            )
            self._perf_e2e_times.clear()
            free_after = self.kv_cache.manager.block_pool.get_num_free_blocks()
            if free_after != free_before:
                logger.warning(
                    "AR-Diffusion warmup did not restore the KV pool (free %d -> %d)",
                    free_before,
                    free_after,
                )

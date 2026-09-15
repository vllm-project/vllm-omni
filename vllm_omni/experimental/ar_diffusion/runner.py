# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import dataclasses
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import supports_step_execution
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
from vllm_omni.experimental.ar_diffusion.release_events import (
    ARDiffusionReleaseEventLog,
    SessionGenerationUnsupportedError,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest
from vllm_omni.experimental.world_models.session_state import SessionStateLostError
from vllm_omni.platforms import current_omni_platform

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

    def __init__(self, vllm_config: object, od_config: OmniDiffusionConfig, device: torch.device) -> None:
        super().__init__(vllm_config, od_config, device)
        self.ar_diffusion_kv_config = resolve_ar_diffusion_kv_config(od_config)
        self.kv_cache: ARDiffusionKVCache | None = None
        self._ar_diffusion_capability: SupportsARDiffusionPipeline | None = None
        self._ar_diffusion_kv_cache_spec: ARDiffusionKVCacheSpec | None = None
        self._sessions: OrderedDict[str, ARDiffusionKVState] = OrderedDict()
        self._session_capacity = 0
        self._perf_e2e_times: list[float] = []
        # Wall-clock start of each request's in-flight stepwise chunk, so
        # ``_perf_e2e_times`` keeps one entry per AR block in both execution
        # modes instead of one entry per denoise step.
        self._stepwise_chunk_started: dict[str, float] = {}
        # A coordinated stage reports releases upward and lets the orchestrator
        # own reset/close ordering; a single-stage deployment does its own.
        self.lifecycle_externally_coordinated = bool(getattr(od_config, "coordinated_session_lifecycle", False))
        stage_id = getattr(od_config, "stage_id", None)
        self.release_events = ARDiffusionReleaseEventLog(
            stage_id=-1 if stage_id is None else int(stage_id),
        )

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
        # Warmup releases real rollouts, but they are not user sessions.
        self.release_events.set_ready()

    def _available_memory_bytes(self) -> int:
        if self.device is None or torch.device(self.device).type != "cuda":
            raise RuntimeError("AR-Diffusion KV preallocation currently requires a CUDA device")
        return int(torch.cuda.mem_get_info(self.device)[0])

    def _preallocate_kv_cache(self, *, available_bytes: int | None = None) -> None:
        """Build pools solely from the pipeline capability and runner config."""
        if bool(getattr(self.od_config, "step_execution", False)) and not supports_step_execution(self.pipeline):
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
            session_capacity=spec.session_capacity,
            cross_attention_lengths=spec.cross_attention_lengths,
            frames_per_block=spec.frames_per_block,
            max_scratch_tokens_per_branch=spec.max_scratch_tokens_per_branch,
            model_owned_state_bytes_per_session=spec.model_owned_state_bytes_per_session,
            device=self.device,
        )
        self._session_capacity = self.kv_cache.session_capacity
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
        # Record before raising: peers must learn this stage dropped the session
        # even when local cleanup only partly succeeded.
        self.release_events.record(session_id, reason=reason, cleanup_failed=bool(errors))
        # The event above captured the generation, so the binding can go; keeping
        # one per session id ever seen would grow without bound.
        self.release_events.forget_generation(session_id)
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

    def suppress_release_events(self, session_id: str):
        """Context manager marking a release as coordinator-driven."""
        return self.release_events.coordinated(session_id)

    def get_ar_diffusion_release_events(self) -> list[dict[str, object]]:
        """Report self-initiated releases without consuming them."""
        return self.release_events.pending()

    def ack_ar_diffusion_release_events(self, event_ids: Sequence[str]) -> int:
        """Drop release records the coordinator has finished acting on."""
        return self.release_events.acknowledge(event_ids)

    def register_ar_diffusion_generation(self, session_id: str, generation: int) -> bool:
        """Bind one generation to both the release log and the loaded model.

        The runner owns registration for a stage that has one: reporting only to
        its own event log would leave the model unable to fence a stale payload.
        A model that does not implement the hook cannot be fenced, so on a
        coordinated topology that is a failure rather than a silent success.
        """
        pipeline = getattr(self, "pipeline", None)
        register = getattr(pipeline, "register_session_generation", None)
        if not callable(register):
            if self.lifecycle_externally_coordinated:
                raise SessionGenerationUnsupportedError(
                    f"{type(pipeline).__name__} does not implement register_session_generation(), so "
                    "it cannot reject a payload from a replaced generation. A coordinated topology "
                    "requires it on every participant."
                )
            self.release_events.register_generation(session_id, generation)
            return True
        if not bool(register(session_id, int(generation))):
            raise SessionGenerationUnsupportedError(
                f"{type(pipeline).__name__}.register_session_generation() refused generation "
                f"{generation} for session {session_id!r}."
            )
        # Only after the model accepted it, so a failed registration cannot leave
        # the log stamping releases with a generation the model never took.
        self.release_events.register_generation(session_id, generation)
        return True

    def has_ar_diffusion_session(self, session_id: str) -> bool:
        """Whether runner-owned KV for ``session_id`` is currently resident."""
        return str(session_id) in self._sessions

    def _reject_unknown_continuation(self, session_id: str, *, reset: bool) -> None:
        """Refuse a continuation of a session this runner no longer holds.

        Otherwise ``_get_or_create_session`` evicts a healthy session to make
        room for the unknown one, and only then does the pipeline reject the
        request. Opt-in, since only a coordinated topology can rely on the
        coordinator having already retired what it released.
        """
        if reset or not self.lifecycle_externally_coordinated:
            return
        if session_id in self._sessions:
            return
        raise SessionStateLostError(
            f"AR-Diffusion session {session_id!r} has no resident KV on this stage, so this "
            "request cannot continue it. Its paged KV was released (explicit close, LRU "
            "eviction, or a failed forward) and cannot be rebuilt from a later chunk. Begin a "
            'new rollout with extra_args["reset"]=True.'
        )

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
    def _request_session(
        req: OmniDiffusionRequest,
    ) -> tuple[str, dict, ARDiffusionTickRequest | None]:
        extra_args = req.sampling_params.extra_args or {}
        tick = ARDiffusionTickRequest.from_extra_args(extra_args)
        if tick is not None:
            return tick.session_id, extra_args, tick
        return str(extra_args.get("session_id") or "default"), extra_args, None

    @contextmanager
    def _bound_ar_session(
        self,
        session_id: str,
        *,
        description: str,
        synchronize: bool = True,
    ) -> Iterator[None]:
        """Bind runner-owned KV for one runner invocation and fail closed.

        The pipeline sees the state only inside this context. On any failure the
        session is released before the exception propagates, so a later request
        for the same id cannot resume half-written pages.
        """
        capability = self._ar_diffusion_capability
        if capability is None:
            raise RuntimeError("AR-Diffusion capability missing after KV cache initialization")
        state = self._get_or_create_session(session_id)
        try:
            with capability.bind_ar_diffusion_state(session_id, state):
                yield
            if synchronize:
                current_omni_platform.synchronize()
        except Exception:
            self._release_session(
                session_id,
                reset_model=False,
                reason="forward_exception",
                suppress_errors=True,
            )
            logger.warning(
                "AR-Diffusion %s failed for session=%s; KV and model state were released",
                description,
                session_id,
            )
            raise

    def _release_scheduler_finished_sessions(self, scheduler_output: DiffusionSchedulerOutput) -> None:
        """Close sessions the scheduler retired, including aborted requests.

        A stepwise request that runs to completion is closed as soon as its
        final chunk is emitted; an aborted one never produces that output, so
        its KV would otherwise linger until LRU eviction. Session ids equal
        request ids on this path, and tick sessions are keyed by their own ids,
        so the membership check leaves them untouched.
        """
        for request_id in getattr(scheduler_output, "finished_req_ids", None) or ():
            self._stepwise_chunk_started.pop(request_id, None)
            if request_id in self._sessions:
                self._release_session(
                    request_id,
                    reset_model=False,
                    reason="close",
                    suppress_errors=True,
                )

    def execute_model(
        self,
        req: OmniDiffusionRequest,
        kv_prefetch_job: KVPrefetchJob | None = None,
    ) -> DiffusionOutput:
        if self.kv_cache is None:
            return super().execute_model(req, kv_prefetch_job=kv_prefetch_job)
        if self._ar_diffusion_capability is None:
            raise RuntimeError("AR-Diffusion capability missing after KV cache initialization")

        session_id, extra_args, tick = self._request_session(req)
        reset = tick.reset if tick is not None else bool(extra_args.get("reset", False))
        close_session = tick.close_session if tick is not None else bool(extra_args.get("close_session", False))
        # The orchestrator already retired the old generation before admission
        # and closes after the request itself; repeating the reset here would
        # discard the conditioning the upstream stage just produced. The
        # begin/reset intent still reaches the pipeline through extra_args.
        if reset and not self.lifecycle_externally_coordinated:
            self.reset_session(session_id)
        self._reject_unknown_continuation(session_id, reset=reset)
        started = time.perf_counter()
        with self._bound_ar_session(session_id, description="forward"):
            output = super().execute_model(req, kv_prefetch_job=kv_prefetch_job)
        self._perf_e2e_times.append(time.perf_counter() - started)
        if close_session and not self.lifecycle_externally_coordinated:
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
        """Bind runner-owned KV for one stepwise invocation, then inherit the step loop."""
        pipeline = getattr(self, "pipeline", None)
        if pipeline is None or not supports_step_execution(pipeline):
            raise RuntimeError(
                "ARDiffusionModelRunner does not support step execution; use request mode with step_execution=False."
            )
        if self.kv_cache is None:
            return super().execute_stepwise(scheduler_output)
        if self._ar_diffusion_capability is None:
            raise RuntimeError("AR-Diffusion capability missing after KV cache initialization")
        # Retire sessions the scheduler dropped before serving this cycle: a
        # completed request is closed when its final chunk is emitted, an
        # aborted one only here.
        self._release_scheduler_finished_sessions(scheduler_output)
        request_ids = list(getattr(scheduler_output, "scheduled_request_ids", ()))
        if not request_ids:
            return super().execute_stepwise(scheduler_output)
        if len(request_ids) != 1:
            raise RuntimeError(
                "AR-Diffusion step execution supports one request at a time; "
                f"got {len(request_ids)} scheduled requests."
            )
        request_id = request_ids[0]
        session_id = request_id
        chunk_started = self._stepwise_chunk_started.setdefault(request_id, time.perf_counter())
        try:
            # Denoise steps inside one chunk need no device barrier; the chunk
            # boundary below is the meaningful one, and syncing per step would
            # cost one synchronize per denoise step instead of per AR block.
            with self._bound_ar_session(session_id, description="stepwise", synchronize=False):
                output = super().execute_stepwise(scheduler_output)
            runner_output = output.get_request_output(request_id)
            finished = bool(runner_output is not None and runner_output.finished)
            result = None if runner_output is None else runner_output.result
            errored = bool(result is not None and getattr(result, "error", None))
            chunk_emitted = result is not None or finished or errored
            if chunk_emitted:
                # The barrier is where an asynchronous device error surfaces, so
                # it must sit inside the same fail-closed scope as the step
                # itself rather than leave cleanup to the scheduler's next cycle.
                current_omni_platform.synchronize()
        except Exception:
            self._fail_closed_stepwise(request_id)
            raise
        # One entry per emitted AR block keeps ``_perf_e2e_times`` comparable
        # with request mode, where one execute_model() is one block.
        if chunk_emitted:
            self._perf_e2e_times.append(time.perf_counter() - chunk_started)
            self._stepwise_chunk_started.pop(request_id, None)
        if finished or errored:
            self._release_session(
                session_id,
                reset_model=False,
                reason="forward_exception" if errored else "close",
                suppress_errors=True,
            )
        return output

    def _fail_closed_stepwise(self, request_id: str) -> None:
        """Drop everything a failed stepwise invocation left behind.

        A failure inside the step is already released by ``_bound_ar_session``;
        a failure at the chunk barrier happens after that context closed, so the
        session is still present and is released here.
        """
        self._stepwise_chunk_started.pop(request_id, None)
        state_cache = getattr(self, "state_cache", None)
        if isinstance(state_cache, dict):
            state_cache.pop(request_id, None)
        if request_id in self._sessions:
            self._release_session(request_id, reset_model=False, reason="forward_exception", suppress_errors=True)
            logger.warning(
                "AR-Diffusion stepwise chunk barrier failed for session=%s; KV and model state were released",
                request_id,
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

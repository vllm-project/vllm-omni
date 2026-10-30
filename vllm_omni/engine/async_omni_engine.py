# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AsyncOmniEngine: the turn-based engine (add_request / streaming updates / CFG companions / interaction)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.multimodal.cache import ShmObjectStoreSenderCache
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_hints import KvHintsEnvelope

from vllm_omni.data_entry_keys import REQUEST_ARTIFACT_DIRS_KEY, TRANSFORM_OWNED_META_KEYS
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_engine_utils import (
    apply_omni_final_stage_metadata,
    inject_global_id,
    scope_stage_replica_mm_uuids,
    upgrade_to_omni_request,
)
from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AddCompanionRequestMessage,
    EngineQueueMessage,
    InteractionMessage,
    StageSubmissionMessage,
)
from vllm_omni.engine.omni_engine_base import OmniEngineBase, StageRuntimeInfo
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorBase
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.inputs.data import OmniInteractionPrompt, OmniSamplingParams, disable_watermarking

logger = init_logger(__name__)


class AsyncOmniEngine(OmniEngineBase):
    """Turn-based engine used by ``Omni`` / ``AsyncOmni``."""

    def _create_orchestrator(self, **orchestrator_kwargs: Any) -> OrchestratorBase:
        return Orchestrator(**orchestrator_kwargs)

    # ---- request helpers ----

    _DEFAULT_MM_HASHER_ALGORITHM = "blake3"

    def _resolve_mm_hasher_algorithm(self) -> str:
        """Return the MM hash algorithm configured for stage 0.

        Upstream vLLM 0.29 removed ``MultiModalHasher``'s ``_get_mm_hasher_algorithm``
        helper and the ``VLLM_MM_HASHER_ALGORITHM`` env var; the algorithm now lives
        only on ``MultiModalConfig.mm_hasher_algorithm`` (``--mm-hasher-algorithm``,
        default ``"blake3"``). Resolve it from stage-0's config so the pre-computed
        replica-scoped uuid uses the same algorithm as the stage-0 renderer built by
        ``build_stage0_input_processor``. Falls back to the upstream default when the
        engine has no stage config (unit tests built via ``object.__new__``) or the
        model has no multimodal config.
        """

        stage_vllm_configs = getattr(self, "stage_vllm_configs", None) or []
        if not stage_vllm_configs:
            return self._DEFAULT_MM_HASHER_ALGORITHM

        model_config = getattr(stage_vllm_configs[0], "model_config", None)
        if model_config is None:
            return self._DEFAULT_MM_HASHER_ALGORITHM

        mm_config = None
        get_multimodal_config = getattr(model_config, "get_multimodal_config", None)
        if callable(get_multimodal_config):
            try:
                mm_config = get_multimodal_config()
            except ValueError:
                # Model is not multimodal.
                mm_config = None
        if mm_config is None:
            mm_config = getattr(model_config, "multimodal_config", None)

        return getattr(mm_config, "mm_hasher_algorithm", None) or self._DEFAULT_MM_HASHER_ALGORITHM

    def _ensure_stage_replica_mm_uuids(
        self,
        prompt: Any,
        *,
        stage_id: int,
        replica_id: int,
    ) -> Any:
        """Make multimodal processor-cache keys local to a stage replica.

        vLLM's frontend multimodal sender cache is process-global, while each
        vllm-omni stage replica owns a separate EngineCore receiver cache. If
        two requests with the same image are routed to different stage-0
        replicas, a plain content hash can make the sender omit the tensor for
        a replica that has never received it. Prefixing user/content UUIDs with
        the selected replica keeps cache reuse within the receiver that owns it.

        Returns a shallow copy carrying the scoped UUIDs; the caller's prompt is
        left untouched. Writing the scoped UUIDs back would make the next
        request built from that dict reuse them as user UUIDs and key its new
        media under the old media's hash.
        """
        return scope_stage_replica_mm_uuids(
            prompt,
            stage_id=stage_id,
            replica_id=replica_id,
            model_id=str(getattr(self, "model", "")),
            mm_hasher_algorithm=self._resolve_mm_hasher_algorithm(),
        )

    @staticmethod
    def _stage_pool_replica_count(stage_pool: Any) -> int:
        try:
            live_num_replicas = getattr(stage_pool, "live_num_replicas", None)
            if live_num_replicas is not None:
                return int(live_num_replicas)
        except Exception:
            pass

        try:
            live_replica_ids = getattr(stage_pool, "live_replica_ids", None)
            if callable(live_replica_ids):
                return len(live_replica_ids())
        except Exception:
            pass

        try:
            clients = getattr(stage_pool, "clients", None)
            if clients is not None:
                return sum(1 for client in clients if client is not None)
        except Exception:
            pass

        return int(getattr(stage_pool, "num_replicas", 1) or 1)

    @staticmethod
    def _stage_pool_is_distributed(stage_pool: Any) -> bool:
        try:
            is_distributed = getattr(stage_pool, "is_distributed", None)
            if is_distributed is not None:
                return bool(is_distributed() if callable(is_distributed) else is_distributed)
        except Exception:
            pass

        return getattr(stage_pool, "_hub", None) is not None

    def _scope_stage0_multimodal_cache_to_replica(
        self,
        request_id: str,
        prompt: Any,
    ) -> tuple[int | None, Any]:
        """Return the preselected stage-0 replica and the prompt to process.

        The returned prompt is a scoped copy when scoping applies, else
        ``prompt`` itself.
        """
        stage_pools = getattr(self, "stage_pools", None)
        if isinstance(prompt, EngineCoreRequest) or not stage_pools:
            return None, prompt

        stage0_pool = stage_pools[0]
        # TODO: Currently only supports the ar -> dit process.
        # Future scenarios (e.g., dit -> ar) need to be added, which will require modifications here.
        if stage0_pool.stage_type == "diffusion" or self._stage_pool_replica_count(stage0_pool) <= 1:
            return None, prompt

        prompts = prompt if isinstance(prompt, list) else [prompt]
        if not any(isinstance(p, dict) and p.get("multi_modal_data") for p in prompts):
            return None, prompt

        if self._stage_pool_is_distributed(stage0_pool):
            preselect_replica_id = getattr(stage0_pool, "preselect_replica_id", None)
            if not callable(preselect_replica_id):
                logger.debug(
                    "[AsyncOmniEngine] Skipping stage-0 multimodal cache scoping for distributed routing "
                    "without preselect support req=%s",
                    request_id,
                )
                return None, prompt
            replica_id = preselect_replica_id(request_id)
            if replica_id is None:
                logger.debug(
                    "[AsyncOmniEngine] Skipping stage-0 multimodal cache scoping for distributed routing "
                    "because no serviceable replica is available yet req=%s",
                    request_id,
                )
                return None, prompt
        else:
            replica_id = stage0_pool.select_replica_id(request_id)

        scoped = [self._ensure_stage_replica_mm_uuids(p, stage_id=0, replica_id=replica_id) for p in prompts]

        logger.debug(
            "[AsyncOmniEngine] Scoped multimodal cache keys to stage-0 replica-%s for req=%s",
            replica_id,
            request_id,
        )
        return replica_id, (scoped if isinstance(prompt, list) else scoped[0])

    def _build_add_request_message(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        final_output_stage_ids: Sequence[int] | None = None,
        arrival_time: float | None = None,
        lora_request: Any = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        *,
        kv_hints: KvHintsEnvelope | None = None,
        resumable: bool = False,
        message_type: Literal["add_request", "streaming_update"] = "add_request",
        entry_stage_id: int = 0,
    ) -> StageSubmissionMessage:
        """Build an add_request message after stage-0 preprocessing."""
        request_timestamp = float(arrival_time) if arrival_time is not None else time.time()
        effective_sampling_params_list: list[OmniSamplingParams] = (
            list(cast(Sequence[OmniSamplingParams], sampling_params_list))
            if sampling_params_list is not None
            else list(self.default_sampling_params_list)
        )
        if not effective_sampling_params_list:
            raise ValueError(
                f"Missing sampling params for stage 0. Got {len(effective_sampling_params_list)} stage params."
            )

        # Opting out on any stage opts the whole request out; pooling stages never watermark
        if any(
            params.watermarking is False
            for params in effective_sampling_params_list
            if isinstance(params, OmniSamplingParams)
        ):
            effective_sampling_params_list = disable_watermarking(effective_sampling_params_list)

        params = effective_sampling_params_list[0]

        # Keep the original prompt for downstream stages (they need the raw
        # dict, e.g. for multi_modal_data).
        if isinstance(prompt, dict):
            raw_info = prompt.get("additional_information")
            if isinstance(raw_info, dict):
                raw_meta = raw_info.get("meta")
                if isinstance(raw_meta, dict):
                    for key in TRANSFORM_OWNED_META_KEYS:
                        raw_meta.pop(key, None)
        # The artifact-dirs key is server-reserved: the prompt transform records
        # where it wrote preprocessed artifacts, and the cleanup paths rmtree
        # whatever it holds. Drop anything a client request carries under it
        # before the transform runs, so cleanup can only ever delete
        # server-written paths (CodeQL py/path-injection).
        if isinstance(prompt, dict):
            prompt.pop(REQUEST_ARTIFACT_DIRS_KEY, None)
        original_prompt = prompt
        preselected_stage0_replica: int | None = None
        request_artifact_dirs: list[str] = []

        stage_type = self.stage_metadata[0].stage_type
        if stage_type == "diffusion" and kv_hints is not None:
            raise ValueError("kv_hints require an AR/LLM stage with a KV cache")
        output_prompt_text: Any = None
        _preprocess_ms = 0.0
        if entry_stage_id:
            # The request bypasses stage 0: its prompt stays raw for the entry
            # stage's input processor (Orchestrator._build_entry_stage_request).
            for item in prompt if isinstance(prompt, list) else [prompt]:
                inject_global_id(item, request_id)
            output_prompt_text = prompt_text
            if output_prompt_text is None and isinstance(prompt, dict):
                output_prompt_text = prompt.get("prompt")
        elif stage_type != "diffusion" and not isinstance(prompt, EngineCoreRequest):
            # Stage transforms and downstream stages must share the same
            # request identity, including when the transform replaces the
            # prompt object.
            if isinstance(prompt, dict):
                inject_global_id(prompt, request_id)
            elif isinstance(prompt, list):
                for item in prompt:
                    inject_global_id(item, request_id)

            prompt_transform_func = getattr(self, "prompt_transform_func", None)
            if prompt_transform_func is not None:
                prompt = prompt_transform_func(
                    copy.copy(prompt),
                    effective_sampling_params_list,
                )
                if isinstance(prompt, dict):
                    raw_dirs = prompt.pop(REQUEST_ARTIFACT_DIRS_KEY, None)
                    if isinstance(raw_dirs, list) and all(isinstance(path, str) for path in raw_dirs):
                        request_artifact_dirs = list(raw_dirs)
                        if isinstance(original_prompt, dict):
                            original_prompt[REQUEST_ARTIFACT_DIRS_KEY] = request_artifact_dirs

            if isinstance(prompt, dict):
                inject_global_id(prompt, request_id)
            elif isinstance(prompt, list):
                for item in prompt:
                    inject_global_id(item, request_id)

            preselected_stage0_replica, prompt = self._scope_stage0_multimodal_cache_to_replica(
                request_id,
                prompt,
            )

            # Full input processing (tokenization, multimodal, etc.)
            assert self.input_processor is not None

            def _release_failed_preprocess() -> None:
                """Undo a failed stage-0 preprocess: free the reserved replica
                binding and drop artifacts the transform already wrote. Reads
                only server-tracked state — never the client prompt."""
                if preselected_stage0_replica is not None and self.stage_pools:
                    self.stage_pools[0].release_binding(request_id)
                for artifact_dir in request_artifact_dirs:
                    shutil.rmtree(artifact_dir, ignore_errors=True)

            _t_preprocess = time.perf_counter()
            try:
                request = self.input_processor.process_inputs(
                    request_id=request_id,
                    prompt=prompt,
                    params=params,
                    supported_tasks=self.supported_tasks,
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                    resumable=resumable,
                    **({"kv_hints": kv_hints} if kv_hints is not None else {}),
                )
            except Exception:
                _release_failed_preprocess()
                raise
            try:
                _preprocess_ms = (time.perf_counter() - _t_preprocess) * 1000.0
                # TODO (Peiqi): add this for Qwen3-TTS only. Other models don't have
                # additional_information field in the prompt.
                request = upgrade_to_omni_request(request, prompt)

                if isinstance(request, OmniEngineCoreRequest) and request.additional_information is not None:
                    processed_info = deserialize_additional_information(request.additional_information)
                    processed_meta = processed_info.get("meta")
                    if isinstance(processed_meta, dict):
                        if isinstance(original_prompt, dict):
                            original_info = dict(original_prompt.get("additional_information") or {})
                            original_meta = dict(original_info.get("meta") or {})
                            original_meta.update(processed_meta)
                            original_info["meta"] = original_meta
                            original_prompt["additional_information"] = original_info

                if reasoning_ended is not None:
                    request.reasoning_ended = reasoning_ended

                # process_inputs() leaves external_req_id unset, so we need to set it to the
                # orchestrator request_id to find the correct req_state in Orchestrator.request_states.
                request.external_req_id = request_id
                request = apply_omni_final_stage_metadata(request, final_stage_id)

                # Registration with stage 0's output processor is deferred to the
                # orchestrator thread (see Orchestrator._handle_add_request), which
                # now routes admission through StagePool.submit_initial().
                output_prompt_text = prompt_text
                if output_prompt_text is None and isinstance(original_prompt, dict):
                    output_prompt_text = original_prompt.get("prompt")
                prompt = request
            except Exception:
                _release_failed_preprocess()
                raise
        else:
            request_artifact_dirs = []

        return StageSubmissionMessage(
            type=message_type,
            request_id=request_id,
            prompt=prompt,
            original_prompt=original_prompt,
            output_prompt_text=output_prompt_text,
            sampling_params_list=effective_sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=list(final_output_stage_ids) if final_output_stage_ids is not None else None,
            preprocess_ms=_preprocess_ms,
            request_timestamp=request_timestamp,
            enqueue_ts=time.perf_counter(),
            request_artifact_dirs=request_artifact_dirs or None,
            entry_stage_id=entry_stage_id,
            lora_request=lora_request,
        )

    def _build_cfg_companions(
        self,
        parent_id: str,
        original_prompt: Any,
        stage0_params: Any,
        sampling_params_list: list[Any],
    ) -> list[AddCompanionRequestMessage]:
        """Expand a prompt into its CFG companions, without enqueueing any.

        Construction is separated from admission so a guided request is
        all-or-nothing. A model whose guidance is mandatory cannot decode a
        request whose companion never arrived: the pair never completes, the
        request occupies scheduler and KV capacity for the scheduler's whole
        hold budget, and then produces no audio. Raising here instead means the
        caller learns immediately and nothing was admitted.

        Raises:
            Exception: Whatever prompt expansion or input processing raised.
                The caller is expected to let it reach the client.
        """
        assert self.prompt_expand_func is not None
        expanded = self.prompt_expand_func(original_prompt, stage0_params)
        if not expanded:
            return []

        companions: list[AddCompanionRequestMessage] = []
        assert self.input_processor is not None
        for ep in expanded:
            cid = f"{parent_id}{ep.request_id_suffix}"
            companion_prompt = ep.prompt

            companion_params, companion_spl = ep.apply_overrides(stage0_params, sampling_params_list)

            if isinstance(companion_prompt, dict):
                inject_global_id(companion_prompt, cid)

            request = self.input_processor.process_inputs(
                request_id=cid,
                prompt=companion_prompt,
                params=companion_params,
                supported_tasks=self.supported_tasks,
            )
            # Same restore the parent request gets: the upstream input
            # processor drops omni-only prompt fields, so without this the
            # companion reaches the worker with no additional_information at
            # all. That is where ``global_request_id`` lives, and it is what
            # was just injected above, so skipping it silently undoes the
            # injection: the model sees the companion row with no id and
            # cannot match it to its conditioned partner.
            request = upgrade_to_omni_request(request, companion_prompt)
            request.external_req_id = cid
            # Companions are stage-0-final for ordinary downstream payloads,
            # but diffusion still needs their CFG KV caches.
            request = apply_omni_final_stage_metadata(request, 0, force_kv_transfer=True)

            # Registration of this companion on stage-0's output processor is
            # deferred to Orchestrator._handle_add_companion, which routes
            # admission through StagePool.submit_initial(..., affinity_request_id=...).
            companions.append(
                AddCompanionRequestMessage(
                    companion_id=cid,
                    parent_id=parent_id,
                    role=ep.role,
                    prompt=request,
                    companion_prompt_text=companion_prompt,
                    sampling_params_list=companion_spl,
                )
            )
        return companions

    def _enqueue_cfg_companions(
        self,
        parent_id: str,
        original_prompt: Any,
        stage0_params: Any,
        sampling_params_list: list[Any],
    ) -> None:
        """Build and enqueue CFG companions, tolerating a build failure.

        Kept for callers that admit the parent first and cannot roll it back.
        Prefer building with :meth:`_build_cfg_companions` before the parent is
        admitted, so the pair is atomic.
        """
        try:
            companions = self._build_cfg_companions(parent_id, original_prompt, stage0_params, sampling_params_list)
        except Exception:
            logger.exception("[AsyncOmniEngine] CFG companion build failed for req %s", parent_id)
            return
        for companion in companions:
            self.request_queue.sync_q.put(companion)
        if not companions:
            return

        logger.info(
            "[AsyncOmniEngine] CFG expansion for req %s: %d companions",
            parent_id,
            len(companions),
        )

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        final_output_stage_ids: Sequence[int] | None = None,
        arrival_time: float | None = None,
        lora_request: Any = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        *,
        kv_hints: KvHintsEnvelope | None = None,
        resumable: bool = False,
        entry_stage_id: int = 0,
    ) -> None:
        """Process stage-0 input locally, then send to the Orchestrator.

        Input processing and output
        processor registration happen here in the caller's thread, avoiding
        a queue + coroutine-switch round-trip.  The Orchestrator receives a
        ready-to-submit OmniEngineCoreRequest.
        """
        msg, companions = self._prepare_add_request(
            request_id=request_id,
            prompt=prompt,
            prompt_text=prompt_text,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            reasoning_ended=reasoning_ended,
            resumable=resumable,
            entry_stage_id=entry_stage_id,
            kv_hints=kv_hints,
        )
        try:
            self.request_queue.sync_q.put(msg)
        except BaseException:
            for artifact_dir in msg.request_artifact_dirs or ():
                shutil.rmtree(artifact_dir, ignore_errors=True)
            raise
        finally:
            if isinstance(msg.original_prompt, dict):
                msg.original_prompt.pop(REQUEST_ARTIFACT_DIRS_KEY, None)
        for companion in companions:
            self.request_queue.sync_q.put(companion)
        if companions:
            logger.info(
                "[AsyncOmniEngine] CFG expansion for req %s: %d companions",
                request_id,
                len(companions),
            )

    def _prepare_add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        final_output_stage_ids: Sequence[int] | None = None,
        arrival_time: float | None = None,
        lora_request: Any = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        *,
        kv_hints: KvHintsEnvelope | None = None,
        resumable: bool = False,
        entry_stage_id: int = 0,
    ) -> tuple[StageSubmissionMessage, list[AddCompanionRequestMessage]]:
        """Build the ADD message plus CFG companions without enqueueing."""
        # Client requests must never bring the server-reserved artifact-dirs
        # key: every cleanup handler below rmtrees the paths it finds, and a
        # raise before _build_add_request_message's own sanitization (e.g. the
        # sampling-params ValueError) would otherwise leave a client-supplied
        # value reachable by that cleanup (CodeQL py/path-injection).
        if isinstance(prompt, dict):
            prompt.pop(REQUEST_ARTIFACT_DIRS_KEY, None)
        msg: StageSubmissionMessage | None = None
        try:
            msg = self._build_add_request_message(
                request_id=request_id,
                prompt=prompt,
                prompt_text=prompt_text,
                sampling_params_list=sampling_params_list,
                final_stage_id=final_stage_id,
                final_output_stage_ids=final_output_stage_ids,
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                reasoning_ended=reasoning_ended,
                resumable=resumable,
                entry_stage_id=entry_stage_id,
                **({"kv_hints": kv_hints} if kv_hints is not None else {}),
            )
            # CFG companions are built before the parent is admitted, so the group
            # is all-or-nothing: a build failure raises here, nothing is enqueued,
            # and the caller sees the error. Admitting the parent first would leave
            # an orphan holding scheduler and KV capacity that can never complete,
            # because a model whose guidance is mandatory cannot decode a request
            # whose companion never arrived.
            companions: list[AddCompanionRequestMessage] = []
            if self.prompt_expand_func is not None and final_stage_id > 0:
                effective_spl = msg.sampling_params_list
                stage0_params = effective_spl[0] if effective_spl else None
                if stage0_params is not None:
                    companions = self._build_cfg_companions(
                        request_id, msg.original_prompt, stage0_params, effective_spl
                    )
        except BaseException:
            # Companion construction runs under this guard too: a failure after
            # the ADD message exists must still release its preprocessed
            # artifacts and ownership key, since nothing was enqueued and the
            # enqueue-path cleanup never runs. Dirs are read only from the
            # prepared message — never from the client prompt:
            # _build_add_request_message cleans its own extraction on any
            # internal failure, and the entry sanitization above already
            # dropped whatever a client supplied under the key.
            if msg is not None:
                for artifact_dir in msg.request_artifact_dirs or ():
                    shutil.rmtree(artifact_dir, ignore_errors=True)
            raise
        assert msg is not None
        return msg, companions

    async def _put_request_queue(self, item: EngineQueueMessage) -> None:
        """Enqueue one message without blocking the caller's event loop.

        The janus queue's async facade is owned by the orchestrator loop —
        the first loop that awaits it — and janus >= 2 raises RuntimeError
        when any other loop touches it, even with queue capacity free. Route
        the put through the owning loop instead: while the queue is full the
        bridged put stays a cancellable task on that loop, and a cancelled
        caller waits for the owning-loop task to settle before unwinding, so
        an item that already resumed (a slot freed) lands before any abort
        RPC the caller sends next — FIFO then aborts it instead of letting
        the abort overtake an orphan ADD (#8678).
        """
        owner_loop = getattr(self, "orchestrator_loop", None)
        if owner_loop is None or owner_loop is asyncio.get_running_loop():
            await self.request_queue.async_q.put(item)
            return
        await self._bridge_owned_put(lambda: self.request_queue.async_q.put(item), {"parent_enqueued": False})

    async def _put_request_queue_group(
        self,
        msg: StageSubmissionMessage,
        companions: list[AddCompanionRequestMessage],
        outcome: dict[str, bool],
    ) -> None:
        """Enqueue an ADD and its CFG companions as one cancellable unit.

        Same owning-loop routing as :meth:`_put_request_queue`; the whole
        group runs as a single task on that loop, so cancellation between
        companion puts is handled where the queue state is visible (see
        :meth:`_owned_enqueue_group`). ``outcome["parent_enqueued"]`` is set
        to whether the parent landed — the caller needs that to scope its
        cleanup — on every exit, including cancellation.
        """
        owner_loop = getattr(self, "orchestrator_loop", None)
        if owner_loop is None or owner_loop is asyncio.get_running_loop():
            await self._owned_enqueue_group(msg, companions, outcome)
            return
        await self._bridge_owned_put(lambda: self._owned_enqueue_group(msg, companions, outcome), outcome)

    async def _bridge_owned_put(
        self,
        owned_put_factory: Callable[[], Any],
        outcome: dict[str, bool],
    ) -> None:
        """Await an owning-loop put, settling it before propagating cancel."""
        settled: concurrent.futures.Future = concurrent.futures.Future()
        started = threading.Event()
        caller_cancelled = threading.Event()

        async def _run() -> None:
            started.set()
            try:
                # cf.cancel() can complete the proxy future before this
                # coroutine is first scheduled; a body that starts afterwards
                # must still not run the put behind a caller that gave up.
                if caller_cancelled.is_set():
                    raise asyncio.CancelledError
                await owned_put_factory()
            finally:
                # Settle only here — after the group's own finally finalized
                # `outcome` — so a cancelled caller reads the unwound
                # operation's verdict, not the state at cf.cancel() time.
                if not settled.done():
                    settled.set_result(bool(outcome.get("parent_enqueued", False)))

        def _settle_cancelled_before_start(done: concurrent.futures.Future) -> None:
            # cf.cancel() completes the proxy future synchronously, before the
            # owning-loop task unwinds. Settle here only when the coroutine
            # never started — nothing can have landed, so False is final; a
            # started put settles from _run's finally after its unwind.
            if done.cancelled():
                caller_cancelled.set()
                if not started.is_set() and not settled.done():
                    settled.set_result(False)

        cf = asyncio.run_coroutine_threadsafe(_run(), getattr(self, "orchestrator_loop"))
        cf.add_done_callback(_settle_cancelled_before_start)
        try:
            await asyncio.wrap_future(cf)
        except asyncio.CancelledError:
            cf.cancel()
            # A started put settles only once its owning-loop unwind finished;
            # waiting for that keeps the cancel-then-abort ordering honest.
            try:
                await asyncio.wrap_future(settled)
            except asyncio.CancelledError:
                pass
            raise

    async def _owned_enqueue_group(
        self,
        msg: StageSubmissionMessage,
        companions: list[AddCompanionRequestMessage],
        outcome: dict[str, bool],
    ) -> None:
        """Put the ADD and its companions on the queue's owning loop.

        Runs as one cancellable unit. If teardown happens after the parent
        landed but before every companion did, the parent is revoked here —
        it would otherwise hold scheduler and KV capacity for a CFG group
        that can never complete. If nothing landed, the prepared artifacts
        are removed. ``outcome["parent_enqueued"]`` reports whether the
        parent landed, on every exit, so the frontend can scope its cleanup.
        """
        parent_enqueued = False
        try:
            await self.request_queue.async_q.put(msg)
            parent_enqueued = True
            for companion in companions:
                await self.request_queue.async_q.put(companion)
        except asyncio.CancelledError:
            if parent_enqueued:
                self._spawn_orphan_parent_revoke(msg.request_id)
            else:
                for artifact_dir in msg.request_artifact_dirs or ():
                    shutil.rmtree(artifact_dir, ignore_errors=True)
            raise
        except BaseException:
            if not parent_enqueued:
                for artifact_dir in msg.request_artifact_dirs or ():
                    shutil.rmtree(artifact_dir, ignore_errors=True)
            raise
        finally:
            outcome["parent_enqueued"] = parent_enqueued

    def _spawn_orphan_parent_revoke(self, request_id: str) -> None:
        """Revoke a half-enqueued CFG parent as soon as the queue has room.

        Detached on the owning loop so the revoked caller unwinds at once;
        FIFO places the abort behind the parent ADD, so the Orchestrator
        aborts a request it actually admitted.
        """
        pending = getattr(self, "_orphan_revokes", None)
        if pending is None:
            pending = self._orphan_revokes = set[asyncio.Task[None]]()
        tracked: set[asyncio.Task[None]] = pending

        async def _revoke() -> None:
            try:
                await self.request_queue.async_q.put(AbortRequestMessage(request_ids=[request_id]))
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Failed to revoke half-enqueued CFG parent %s", request_id)
            finally:
                tracked.discard(revoke_task)

        revoke_task = asyncio.get_running_loop().create_task(_revoke())
        tracked.add(revoke_task)

    async def _reset_frontend_mm_cache(self) -> None:
        """Drop sender-cache entries recorded by a submit that never landed.

        ``input_processor.process_inputs`` registers the request's media in
        the process-global multimodal sender cache before the queue put; a
        revoked submission means EngineCore never received the request, and a
        later request with the same media would send only a cache reference
        and fail with "Multi-modal processor cache miss". Clearing the whole
        cache is coarse but this path only runs on rare cancelled submits.

        The shm sender cache is skipped: it hands EngineCore shared-memory
        addresses, so an entry whose request never landed is still readable,
        and ``clear_cache()`` there resets the ring buffer under requests
        that are still in flight. The cost is that the revoked request's
        writer reference is never balanced by a reader, so its buffer stays
        pinned and later media may fall back to full-payload IPC once the
        ring fills; releasing it needs a writer-release API from vLLM.
        """
        processor = getattr(self, "input_processor", None)
        renderer = getattr(processor, "renderer", None)
        if isinstance(getattr(renderer, "mm_processor_cache", None), ShmObjectStoreSenderCache):
            return
        clear = getattr(renderer, "clear_mm_cache_async", None)
        if clear is None:
            return
        try:
            await clear()
        except Exception:
            logger.exception("Failed to clear the frontend multimodal cache after a revoked submit")

    async def add_request_async(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        final_output_stage_ids: Sequence[int] | None = None,
        arrival_time: float | None = None,
        lora_request: Any = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        *,
        kv_hints: KvHintsEnvelope | None = None,
        resumable: bool = False,
        entry_stage_id: int = 0,
    ) -> None:
        """Async add_request API.

        Same local input processing as ``add_request``, but the queue put is
        awaited through the janus async facade: a full request queue suspends
        this coroutine instead of blocking the event loop on ``sync_q.put``,
        and the wait is cancellable (sleep/pause-abort cancel pending submits
        so their admission slots release instead of deadlocking the loop).
        """
        msg, companions = self._prepare_add_request(
            request_id=request_id,
            prompt=prompt,
            prompt_text=prompt_text,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            reasoning_ended=reasoning_ended,
            resumable=resumable,
            entry_stage_id=entry_stage_id,
            **({"kv_hints": kv_hints} if kv_hints is not None else {}),
        )
        outcome: dict[str, bool] = {"parent_enqueued": False}
        try:
            await self._put_request_queue_group(msg, companions, outcome)
        except BaseException:
            # Artifact cleanup for a group that never landed is handled by
            # _owned_enqueue_group, which knows whether the parent was
            # admitted. Reset the frontend sender cache only then:
            # process_inputs recorded this request's media before the put, so
            # a never-admitted submission must not leave later same-media
            # requests with cache-miss failures. For a half-enqueued group the
            # parent DID land (its orphan revoke is in flight) and the cache
            # entries belong to that real request — clearing the
            # process-global cache here would race the Orchestrator processing
            # the ADD and force unrelated media re-sends.
            if not outcome["parent_enqueued"]:
                await self._reset_frontend_mm_cache()
            raise
        finally:
            if isinstance(msg.original_prompt, dict):
                msg.original_prompt.pop(REQUEST_ARTIFACT_DIRS_KEY, None)
        if companions:
            logger.info(
                "[AsyncOmniEngine] CFG expansion for req %s: %d companions",
                request_id,
                len(companions),
            )

    def add_streaming_update(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        final_output_stage_ids: Sequence[int] | None = None,
        arrival_time: float | None = None,
        lora_request: Any = None,
        *,
        resumable: bool = True,
    ) -> None:
        """Send an incremental streaming update for an existing request."""
        msg = self._build_add_request_message(
            request_id=request_id,
            prompt=prompt,
            prompt_text=prompt_text,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            resumable=resumable,
            message_type="streaming_update",
        )
        self.request_queue.sync_q.put(msg)

    async def add_streaming_update_async(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        prompt_text: str | None = None,
        sampling_params_list: Sequence[Any] | None = None,
        final_stage_id: int = 0,
        final_output_stage_ids: Sequence[int] | None = None,
        arrival_time: float | None = None,
        lora_request: Any = None,
        *,
        resumable: bool = True,
    ) -> None:
        """Async wrapper for add_streaming_update().

        Builds the same streaming_update message but enqueues it through
        :meth:`_put_request_queue`: a full request queue suspends this
        coroutine (cancellable) instead of blocking the frontend event loop
        inside ``sync_q.put``, which would re-create the #8678 wedge for
        streaming requests against a paused engine.
        """
        msg = self._build_add_request_message(
            request_id=request_id,
            prompt=prompt,
            prompt_text=prompt_text,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            resumable=resumable,
            message_type="streaming_update",
        )
        await self._put_request_queue(msg)

    def submit_interaction(
        self,
        request_id: str,
        interaction: OmniInteractionPrompt,
    ) -> None:
        """Send an interaction control message to the Orchestrator."""
        if self.request_queue is None:
            raise RuntimeError("request_queue is not initialized")

        self.request_queue.sync_q.put_nowait(
            InteractionMessage(
                request_id=request_id,
                interaction=interaction,
            )
        )

    async def submit_interaction_async(
        self,
        request_id: str,
        interaction: OmniInteractionPrompt,
    ) -> None:
        """Async interaction API."""
        self.submit_interaction(request_id, interaction)


# ``StageRuntimeInfo`` moved to ``omni_engine_base``; keep the historical import path.
__all__ = ["AsyncOmniEngine", "StageRuntimeInfo"]

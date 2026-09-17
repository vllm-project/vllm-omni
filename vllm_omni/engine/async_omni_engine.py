# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AsyncOmniEngine: the turn-based engine (add_request / streaming updates / CFG companions / interaction)."""

from __future__ import annotations

import copy
import shutil
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.data_entry_keys import REQUEST_ARTIFACT_DIRS_KEY, TRANSFORM_OWNED_META_KEYS
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_engine_utils import (
    apply_omni_final_stage_metadata,
    inject_global_id,
    upgrade_to_omni_request,
)
from vllm_omni.engine.messages import (
    AddCompanionRequestMessage,
    InteractionMessage,
    StageSubmissionMessage,
)
from vllm_omni.engine.omni_engine_base import OmniEngineBase, StageRuntimeInfo
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorBase
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.inputs.data import OmniInteractionPrompt, OmniSamplingParams

logger = init_logger(__name__)


class AsyncOmniEngine(OmniEngineBase):
    """Turn-based engine used by ``Omni`` / ``AsyncOmni``."""

    def _create_orchestrator(self, **orchestrator_kwargs: Any) -> OrchestratorBase:
        return Orchestrator(**orchestrator_kwargs)

    # ---- request helpers ----

    @staticmethod
    def _iter_multimodal_items(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _ensure_stage_replica_mm_uuids(
        self,
        prompt: Any,
        *,
        stage_id: int,
        replica_id: int,
    ) -> None:
        """Make multimodal processor-cache keys local to a stage replica.

        vLLM's frontend multimodal sender cache is process-global, while each
        vllm-omni stage replica owns a separate EngineCore receiver cache. If
        two requests with the same image are routed to different stage-0
        replicas, a plain content hash can make the sender omit the tensor for
        a replica that has never received it. Prefixing user/content UUIDs with
        the selected replica keeps cache reuse within the receiver that owns it.
        """

        if not isinstance(prompt, dict):
            return

        mm_data = prompt.get("multi_modal_data")
        if not isinstance(mm_data, dict) or not mm_data:
            return

        from vllm.config.multimodal import _get_mm_hasher_algorithm
        from vllm.multimodal.hasher import MultiModalHasher

        existing_uuids = prompt.get("multi_modal_uuids")
        if not isinstance(existing_uuids, dict):
            existing_uuids = {}

        model_id = str(getattr(self, "model", ""))
        scoped_uuids: dict[str, list[str | None]] = dict(existing_uuids)
        for modality, raw_items in mm_data.items():
            items = self._iter_multimodal_items(raw_items)
            if not items:
                continue

            modality_existing = existing_uuids.get(modality)
            if not isinstance(modality_existing, list):
                modality_existing = [modality_existing] if modality_existing is not None else []

            modality_uuids: list[str | None] = []
            for idx, item in enumerate(items):
                user_uuid = modality_existing[idx] if idx < len(modality_existing) else None
                if user_uuid is not None:
                    base_uuid = str(user_uuid)
                elif item is None:
                    base_uuid = None
                else:
                    base_uuid = MultiModalHasher.hash_kwargs(
                        _get_mm_hasher_algorithm(),
                        model_id=model_id,
                        **{modality: item},
                    )

                if base_uuid is None:
                    modality_uuids.append(None)
                else:
                    modality_uuids.append(f"stage{stage_id}:rep{replica_id}:{base_uuid}")

            scoped_uuids[modality] = modality_uuids

        if scoped_uuids:
            prompt["multi_modal_uuids"] = scoped_uuids

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
    ) -> int | None:
        stage_pools = getattr(self, "stage_pools", None)
        if isinstance(prompt, EngineCoreRequest) or not stage_pools:
            return None

        stage0_pool = stage_pools[0]
        # TODO: Currently only supports the ar -> dit process.
        # Future scenarios (e.g., dit -> ar) need to be added, which will require modifications here.
        if stage0_pool.stage_type == "diffusion" or self._stage_pool_replica_count(stage0_pool) <= 1:
            return None

        prompts = prompt if isinstance(prompt, list) else [prompt]
        if not any(isinstance(p, dict) and p.get("multi_modal_data") for p in prompts):
            return None

        if self._stage_pool_is_distributed(stage0_pool):
            preselect_replica_id = getattr(stage0_pool, "preselect_replica_id", None)
            if not callable(preselect_replica_id):
                logger.debug(
                    "[AsyncOmniEngine] Skipping stage-0 multimodal cache scoping for distributed routing "
                    "without preselect support req=%s",
                    request_id,
                )
                return None
            replica_id = preselect_replica_id(request_id)
            if replica_id is None:
                logger.debug(
                    "[AsyncOmniEngine] Skipping stage-0 multimodal cache scoping for distributed routing "
                    "because no serviceable replica is available yet req=%s",
                    request_id,
                )
                return None
        else:
            replica_id = stage0_pool.select_replica_id(request_id)

        for p in prompts:
            self._ensure_stage_replica_mm_uuids(
                p,
                stage_id=0,
                replica_id=replica_id,
            )

        logger.debug(
            "[AsyncOmniEngine] Scoped multimodal cache keys to stage-0 replica-%s for req=%s",
            replica_id,
            request_id,
        )
        return replica_id

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
        resumable: bool = False,
        message_type: Literal["add_request", "streaming_update"] = "add_request",
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
        original_prompt = prompt
        preselected_stage0_replica: int | None = None
        request_artifact_dirs: list[str] = []

        stage_type = self.stage_metadata[0].stage_type
        output_prompt_text: Any = None
        _preprocess_ms = 0.0
        if stage_type != "diffusion" and not isinstance(prompt, EngineCoreRequest):
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
                if isinstance(prompt, dict):
                    prompt.pop(REQUEST_ARTIFACT_DIRS_KEY, None)
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

            preselected_stage0_replica = self._scope_stage0_multimodal_cache_to_replica(
                request_id,
                prompt,
            )

            # Full input processing (tokenization, multimodal, etc.)
            assert self.input_processor is not None
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
                )
            except Exception:
                if preselected_stage0_replica is not None and self.stage_pools:
                    self.stage_pools[0].release_binding(request_id)
                for artifact_dir in request_artifact_dirs:
                    shutil.rmtree(artifact_dir, ignore_errors=True)
                raise
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

            # Restore external_req_id to the original user-facing request_id.
            # InputProcessor.process_inputs() renames request_id to an internal
            # UUID (saving the original in external_req_id), but then overwrites
            # external_req_id with the new internal ID. We need external_req_id
            # to match the key used in Orchestrator.request_states so that
            # output routing (output.request_id lookup) can find the req_state.
            request.external_req_id = request_id
            request = apply_omni_final_stage_metadata(request, final_stage_id)

            # Registration with stage 0's output processor is deferred to the
            # orchestrator thread (see Orchestrator._handle_add_request), which
            # now routes admission through StagePool.submit_initial().
            output_prompt_text = prompt_text
            if output_prompt_text is None and isinstance(original_prompt, dict):
                output_prompt_text = original_prompt.get("prompt")
            prompt = request
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
        resumable: bool = False,
    ) -> None:
        """Process stage-0 input locally, then send to the Orchestrator.

        Input processing and output
        processor registration happen here in the caller's thread, avoiding
        a queue + coroutine-switch round-trip.  The Orchestrator receives a
        ready-to-submit OmniEngineCoreRequest.
        """
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
            )
        except BaseException:
            if isinstance(prompt, dict):
                for artifact_dir in prompt.pop(REQUEST_ARTIFACT_DIRS_KEY, None) or ():
                    if isinstance(artifact_dir, str):
                        shutil.rmtree(artifact_dir, ignore_errors=True)
            raise
        # CFG companions are built before the parent is admitted, so the group
        # is all-or-nothing: a build failure raises here, nothing is enqueued,
        # and the caller sees the error. Admitting the parent first would leave
        # an orphan holding scheduler and KV capacity that can never complete,
        # because a model whose guidance is mandatory cannot decode a request
        # whose companion never arrived.
        companions: list[AddCompanionRequestMessage] = []
        try:
            if self.prompt_expand_func is not None and final_stage_id > 0:
                effective_spl = msg.sampling_params_list
                stage0_params = effective_spl[0] if effective_spl else None
                if stage0_params is not None:
                    companions = self._build_cfg_companions(
                        request_id, msg.original_prompt, stage0_params, effective_spl
                    )

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
        resumable: bool = False,
    ) -> None:
        """Async add_request API."""
        self.add_request(
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
        """Async wrapper for add_streaming_update()."""
        self.add_streaming_update(
            request_id=request_id,
            prompt=prompt,
            prompt_text=prompt_text,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            resumable=resumable,
        )

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

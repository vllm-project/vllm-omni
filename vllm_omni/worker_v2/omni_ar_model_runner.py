# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniARModelRunner — autoregressive stage runner on MR V2.

Extends ``OmniGPUModelRunner`` with:

* ``OmniOutput`` post-processing in ``sample_tokens``
* Per-request ``pooler_output`` construction (hidden + multimodal slices)
* Async D2H copy via ``OmniAsyncOutput`` for non-blocking output transfer
* Cross-stage KV extraction before state cleanup
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any, cast

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    get_ep_all2all_manager,
)
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu.eplb_utils import step_eplb_after

from vllm_omni.data_entry_keys import OmniPayload, flatten_payload
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.utils.mm_outputs import partition_flat_payload
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner
from vllm_omni.worker_v2.omni_sampler import sample_with_output
from vllm_omni.worker_v2.output_snapshot import PackedOutputSnapshot, RequestOutputSnapshot, pack_output_snapshot
from vllm_omni.worker_v2.streaming_audio import StreamingAudioOutput

logger = init_logger(__name__)
_ASYNC_MM_SNAPSHOT_MAX_BUCKETS_PER_SLOT = 64


def _copy_mm_to_snapshot_slot(value: Any, slot: dict[tuple[Any, ...], torch.Tensor], path: tuple[Any, ...] = ()) -> Any:
    if not path and isinstance(value, dict):
        packed = pack_output_snapshot(value, slot, max_buckets=_ASYNC_MM_SNAPSHOT_MAX_BUCKETS_PER_SLOT)
        if packed is not None:
            return packed
    if isinstance(value, torch.Tensor):
        bucket_key = path + (tuple(value.shape), value.dtype, value.device)
        cached = slot.get(bucket_key)
        if cached is None:
            if len(slot) >= _ASYNC_MM_SNAPSHOT_MAX_BUCKETS_PER_SLOT:
                return value.detach().clone()
            cached = torch.empty_like(value)
            slot[bucket_key] = cached
        cached.copy_(value)
        return cached
    if isinstance(value, dict):
        return {key: _copy_mm_to_snapshot_slot(item, slot, path + (key,)) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_mm_to_snapshot_slot(item, slot, path + (index,)) for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_copy_mm_to_snapshot_slot(item, slot, path + (index,)) for index, item in enumerate(value))
    return value


def _has_cuda_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device.type == "cuda"
    if isinstance(value, dict):
        return any(_has_cuda_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_cuda_tensor(item) for item in value)
    return False


def _guard_graph_replay_for_pooler_copy(
    main_stream: Any,
    copy_event: Any,
    *,
    need_pooler: bool,
    async_chunk: bool,
) -> None:
    """Keep static graph outputs alive until non-snapshotted D2H completes."""
    if need_pooler and not async_chunk:
        main_stream.wait_event(copy_event)


class OmniARModelRunner(OmniGPUModelRunner):
    """AR stage runner. Produces per-request hidden states + multimodal outputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kv_transfer_manager: OmniKVTransferManager | None = None
        self._kv_extracted_req_ids: list[str] | None = None
        self._async_mm_snapshot_slots: list[dict[tuple[Any, ...], torch.Tensor]] = [{} for _ in range(4)]
        self._async_mm_snapshot_events: list[torch.cuda.Event | None] = [None] * 4
        self._async_mm_snapshot_pending = [False] * 4
        self._async_mm_snapshot_cursor = 0
        self._last_multimodal_snapshot_slot: int | None = None

    def _ensure_kv_transfer_manager(self) -> OmniKVTransferManager:
        if self.kv_transfer_manager is None:
            self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
        return self.kv_transfer_manager

    # ------------------------------------------------------------------
    # execute_model: KV transfer pre-hook + delegate to super
    # ------------------------------------------------------------------

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Any | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
    ) -> Any:
        if not dummy_run:
            self._handle_kv_transfer_pre(scheduler_output)
        return super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            is_profile=is_profile,
            context_len=context_len,
            valid_dummy_state_slots=valid_dummy_state_slots,
            randomize_inputs=randomize_inputs,
        )

    # ------------------------------------------------------------------
    # sample_tokens: OmniOutput handling + pooler_output + async D2H
    # ------------------------------------------------------------------

    def sample(self, hidden_states, input_batch, grammar_output):
        # An explicitly declared model-state hook may return already determined
        # tokens. Unsupported sampling features retain the upstream path.
        sample_determined = getattr(type(self.model_state), "sample_determined_tokens", None)
        if (
            sample_determined is not None
            and grammar_output is None
            and self.batch_sharder is None
            and self.pp_handler is None
            and input_batch.num_draft_tokens == 0
            and input_batch.num_reqs > 0
        ):
            output = sample_determined(self.model_state, input_batch, self.sampler)
            if output is not None:
                return output, output.num_sampled, output.num_rejected
        return super().sample(hidden_states, input_batch, grammar_output)

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> OmniAsyncOutput | OmniModelRunnerOutput | ModelRunnerOutput | None:
        kv_extracted = self._kv_extracted_req_ids
        self._kv_extracted_req_ids = None

        if self.execute_model_state is None:
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        ec_connector_output = self.execute_model_state.ec_connector_output
        pending_aux_output = None
        if self.aux_output_connector is not None:
            pending_aux_output = self.aux_output_connector.prepare_output(input_batch)
        self.execute_model_state = None

        # --- Omni: reconstruct raw model output and post-process ---
        aux = self._last_aux_output
        self._last_aux_output = None
        multimodal_outputs = self._last_multimodal_outputs
        self._last_multimodal_outputs = None
        snapshot_slot = None
        self._last_multimodal_snapshot_slot = None
        raw_output = self._reconstruct_raw_model_output(
            hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
            aux=aux,
        )
        text_hidden, multimodal_outputs = self.model_state.postprocess_model_output(
            raw_output, input_batch, self.req_states
        )
        if bool(getattr(self.model_config, "async_chunk", False)) and multimodal_outputs:
            multimodal_outputs = self._retain_multimodal_outputs(multimodal_outputs)
            snapshot_slot = self._last_multimodal_snapshot_slot
            self._last_multimodal_snapshot_slot = None

        # --- Standard v2 sampling ---
        sampling_context = nullcontext()
        model_sampling_context = getattr(self.model, "mrv2_sampling_context", None)
        if callable(model_sampling_context) and getattr(self.model, "logitsprocs_need_output_token_ids", False):
            # seq_lens includes this forward's input, but not its sampled token:
            # final prefill has zero prior outputs; the first decode has one.
            sampling_context = model_sampling_context(
                req_ids=input_batch.req_ids,
                num_output_tokens=(
                    input_batch.seq_lens[: input_batch.num_reqs]
                    - self.req_states.prompt_len.gpu[input_batch.idx_mapping]
                ),
            )
        with sampling_context:
            sampling_output = sample_with_output(
                self.sampler, self.sample, text_hidden, input_batch, self.req_states, grammar_output
            )
        sampler_output = sampling_output.sampler_output
        num_sampled, num_rejected = sampling_output.num_sampled, sampling_output.num_rejected
        if sampling_output.multimodal_outputs is not None:
            multimodal_outputs = sampling_output.multimodal_outputs
        run_eager_mtp = getattr(self.model_state, "run_eager_mtp", None)
        if multimodal_outputs and run_eager_mtp is not None:
            run_eager_mtp(
                input_batch,
                text_hidden,
                sampler_output.sampled_token_ids,
                multimodal_outputs,
                self._dispatch_mtp_batch_descriptor,
            )
        publish_sampled = getattr(self.model_state, "publish_sampled_embeddings", None)
        extra_outputs = (
            publish_sampled(input_batch, sampler_output.sampled_token_ids)
            if (
                multimodal_outputs
                and callable(publish_sampled)
                and bool(getattr(self.model_config, "async_chunk", False))
                and getattr(self.model_config, "engine_output_type", "text") != "text"
            )
            else None
        )
        if self.pp_handler is not None:
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )

        # --- Omni: prompt logprobs ---
        assert self.prompt_logprobs_worker is not None
        # Mirror the current parent GPUModelRunner call exactly.
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            text_hidden,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
        )
        prompt_token_id_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_token_id_logprobs(
            self.model.compute_logits,
            text_hidden,
            input_batch,
            self.req_states.prompt_len.np,
        )

        # --- Omni: pooler_output ---
        engine_output_type = getattr(self.vllm_config.model_config, "engine_output_type", "text")
        need_pooler = engine_output_type != "text"

        # --- Build base output ---
        model_runner_output = OmniModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore[arg-type]
            prompt_logprobs_dict=prompt_logprobs_dict,
            prompt_token_id_logprobs_dict=prompt_token_id_logprobs_dict,
            kv_connector_output=None,
        )
        model_runner_output.kv_extracted_req_ids = kv_extracted
        model_runner_output._async_chunk = bool(getattr(self.model_config, "async_chunk", False))

        prepare_streaming = getattr(self.model_state, "prepare_streaming_audio_output", None)
        streaming_audio = (
            prepare_streaming(input_batch, self.req_states, multimodal_outputs)
            if need_pooler and multimodal_outputs and callable(prepare_streaming)
            else None
        )

        # --- Async D2H via OmniAsyncOutput ---
        materialize_native = self._uses_native_output_materializer()
        async_output = OmniAsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
            text_hidden=text_hidden if need_pooler and sampling_output.include_hidden_states else None,
            multimodal_outputs=multimodal_outputs if need_pooler else None,
            input_batch=input_batch if need_pooler else None,
            async_chunk=bool(getattr(self.model_config, "async_chunk", False)),
            finalize_output=(None if materialize_native else self._finalize_native_data_plane_output),
            finalize_multimodal=sampling_output.finalize_multimodal,
            check_ep_fault=self.check_ep_fault,
            pending_aux_output=pending_aux_output,
            streaming_audio=streaming_audio,
            extra_multimodal_outputs=extra_outputs,
        )
        self._release_multimodal_snapshot(snapshot_slot, async_output.copy_event)
        _guard_graph_replay_for_pooler_copy(
            self.main_stream,
            async_output.copy_event,
            need_pooler=need_pooler
            and (sampling_output.include_hidden_states or not sampling_output.owns_multimodal_outputs),
            async_chunk=bool(getattr(self.model_config, "async_chunk", False)),
        )

        # Postprocess AFTER creating async output (so copy_event is
        # recorded before postprocess, matching upstream pattern).
        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )
        model_runner_output.kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        model_runner_output.ec_connector_output = ec_connector_output

        self._reserve_native_data_plane_outputs(list(model_runner_output.req_ids))
        if materialize_native:
            return self._materialize_native_output(async_output)
        return async_output

    def _retain_multimodal_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        if not bool(getattr(self.model_config, "async_chunk", False)) or not outputs:
            return outputs
        if getattr(self.model, "mm_outputs_fresh_per_step", False):
            # Freshly allocated each step and filled after sampling; nothing
            # overwrites them before the host copy.
            return outputs
        # A producer-side PackedOutputSnapshot already owns its device slabs
        # for this forward and carries the event that makes them visible to
        # the output copy stream. Repacking it would add a second GPU copy.
        if isinstance(outputs, PackedOutputSnapshot):
            return outputs
        slot_index = self._async_mm_snapshot_cursor
        if self._async_mm_snapshot_pending[slot_index]:
            raise RuntimeError("Async multimodal snapshot ring exhausted before sample_tokens().")
        event = self._async_mm_snapshot_events[slot_index]
        if event is not None:
            self.main_stream.wait_event(event)
        if _has_cuda_tensor(outputs):
            with torch.cuda.stream(self.main_stream):
                snapshot = _copy_mm_to_snapshot_slot(outputs, self._async_mm_snapshot_slots[slot_index])
        else:
            snapshot = _copy_mm_to_snapshot_slot(outputs, self._async_mm_snapshot_slots[slot_index])
        self._async_mm_snapshot_pending[slot_index] = True
        self._last_multimodal_snapshot_slot = slot_index
        self._async_mm_snapshot_cursor = (slot_index + 1) % len(self._async_mm_snapshot_slots)
        return snapshot

    def _release_multimodal_snapshot(self, slot_index: int | None, copy_event: torch.cuda.Event) -> None:
        if slot_index is None:
            return
        self._async_mm_snapshot_events[slot_index] = copy_event
        self._async_mm_snapshot_pending[slot_index] = False

    # ------------------------------------------------------------------
    # pooler_output construction
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_raw_model_output(
        *,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict[str, Any] | None,
        aux: Any | None,
    ) -> Any:
        if multimodal_outputs:
            return OmniOutput(
                text_hidden_states=hidden_states,
                multimodal_outputs=cast(OmniPayload, multimodal_outputs),
            )
        if aux is not None:
            return (hidden_states, aux)
        return hidden_states

    @staticmethod
    def _build_pooler_output_from_cpu(
        hidden_cpu: torch.Tensor,
        mm_cpu: dict[str, Any],
        query_start_loc_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
        padded_total_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Build per-request payloads, excluding any CUDA Graph padding."""
        # Graph-padded hidden states and unpadded multimodal outputs can
        # coexist. Use the scheduled token count to identify the real axis.
        total = int(num_scheduled_tokens[:num_reqs].sum())
        pooler: list[dict[str, Any]] = []
        for i in range(num_reqs):
            start = int(query_start_loc_np[i])
            end = start + int(num_scheduled_tokens[i])
            payload: dict[str, Any] = {"hidden": hidden_cpu[start:end].clone()}
            for k, v in mm_cpu.items():
                payload[k] = _slice_pooler_value(
                    v,
                    req_index=i,
                    start=start,
                    end=end,
                    total_tokens=total,
                    padded_total_tokens=padded_total_tokens,
                )
            pooler.append(flatten_payload(payload))
        return pooler

    @staticmethod
    def _build_async_chunk_outputs_from_mm(
        mm_outputs: dict[str, Any],
        query_start_loc_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_reqs: int,
        total_tokens: int,
        padded_total_tokens: int | None = None,
    ) -> tuple[list[dict[str, Any] | None] | None, list[dict[str, Any] | None] | None]:
        """Build async-chunk payloads without materializing hidden on CPU."""
        if not mm_outputs:
            return None, None

        inter_stage_list: list[dict[str, Any] | None] = []
        client_mm_list: list[dict[str, Any] | None] = []
        for i in range(num_reqs):
            start = int(query_start_loc_np[i])
            end = start + int(num_scheduled_tokens[i])
            payload = {
                key: _slice_pooler_value(
                    value,
                    req_index=i,
                    start=start,
                    end=end,
                    total_tokens=total_tokens,
                    padded_total_tokens=padded_total_tokens,
                )
                for key, value in mm_outputs.items()
            }
            inter_stage, client_mm = partition_flat_payload(flatten_payload(payload))
            inter_stage_list.append(inter_stage or None)
            client_mm_list.append(client_mm or None)

        return (
            None if all(item is None for item in inter_stage_list) else inter_stage_list,
            None if all(item is None for item in client_mm_list) else client_mm_list,
        )

    # ------------------------------------------------------------------
    # KV transfer
    # ------------------------------------------------------------------

    def _handle_kv_transfer_pre(self, scheduler_output: SchedulerOutput) -> None:
        finished: dict = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if not finished:
            return

        kv_caches = getattr(self, "kv_caches", None)
        if kv_caches is None:
            return

        if hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished.items():
                try:
                    meta = self.model.get_kv_transfer_metadata(req_id)
                    if meta:
                        existing = data.get("custom_metadata") or {}
                        existing.update(meta)
                        data["custom_metadata"] = existing
                except Exception:
                    logger.warning(
                        "Failed to get KV transfer metadata for %s",
                        req_id,
                        exc_info=True,
                    )

        mgr = self._ensure_kv_transfer_manager()
        self._kv_extracted_req_ids = mgr.handle_finished_requests_kv_transfer(
            finished_reqs=finished,
            kv_caches=kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
            request_id_resolver=self._resolve_global_request_id,
        )

    def _resolve_global_request_id(self, req_id: str) -> str:
        req_idx = self.req_states.req_id_to_index.get(req_id)
        if req_idx is None:
            return req_id
        info = self.model_state.intermediate_buffer.buffers[req_idx]
        global_id = info.get("global_request_id")
        if isinstance(global_id, list):
            global_id = global_id[0] if global_id else None
        if isinstance(global_id, bytes):
            return global_id.decode("utf-8")
        return str(global_id) if global_id else req_id


# ======================================================================
# OmniAsyncOutput — async D2H for Omni AR outputs
# ======================================================================


def _async_copy_to_np(
    x: torch.Tensor,
    *,
    copy_stream: torch.cuda.Stream | None = None,
    pin_memory: bool | None = None,
) -> np.ndarray:
    return _async_copy_tensor(
        x,
        copy_stream=copy_stream,
        pin_memory=pin_memory,
    ).numpy()


def _async_copy_tensor(
    x: torch.Tensor,
    *,
    copy_stream: torch.cuda.Stream | None = None,
    pin_memory: bool | None = None,
) -> torch.Tensor:
    x = x.detach()
    if x.device.type == "cpu":
        return x.clone()
    if pin_memory is None:
        pin_memory = PIN_MEMORY
    cpu = torch.empty_like(x, device="cpu", pin_memory=pin_memory)
    cpu.copy_(x, non_blocking=pin_memory)
    if copy_stream is None:
        copy_stream = torch.cuda.current_stream()
    x.record_stream(copy_stream)
    return cpu


def _async_copy_mm_value(
    value: Any,
    *,
    copy_stream: torch.cuda.Stream | None = None,
    pin_memory: bool | None = None,
) -> Any:
    if isinstance(value, torch.Tensor):
        return _async_copy_tensor(
            value,
            copy_stream=copy_stream,
            pin_memory=pin_memory,
        )
    if isinstance(value, dict):
        return {
            key: _async_copy_mm_value(
                val,
                copy_stream=copy_stream,
                pin_memory=pin_memory,
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        if len(value) > 1 and isinstance(value[0], torch.Tensor) and value[0].device.type == "cpu":
            first = value[0]
            if all(val is first for val in value):
                # One shared per-step host tensor (e.g. every request's sample
                # rate): one copy serves all entries.
                return [_async_copy_tensor(first)] * len(value)
        return [
            _async_copy_mm_value(
                val,
                copy_stream=copy_stream,
                pin_memory=pin_memory,
            )
            for val in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _async_copy_mm_value(
                val,
                copy_stream=copy_stream,
                pin_memory=pin_memory,
            )
            for val in value
        )
    return value


def _async_copy_mm(
    mm_outputs: dict | None,
    total_tokens: int,
    *,
    copy_stream: torch.cuda.Stream | None = None,
    pin_memory: bool | None = None,
) -> dict[str, Any]:
    """Non-blocking D2H copy of multimodal output tensors."""
    if not mm_outputs:
        return {}
    if isinstance(mm_outputs, PackedOutputSnapshot):
        if mm_outputs.producer_event is not None:
            mm_outputs.producer_event.wait(copy_stream)
        return mm_outputs.copy_to_cpu(
            lambda tensor: _async_copy_tensor(tensor, copy_stream=copy_stream, pin_memory=pin_memory)
        )
    return {
        key: _async_copy_mm_value(
            value,
            copy_stream=copy_stream,
            pin_memory=pin_memory,
        )
        for key, value in mm_outputs.items()
    }


def _merge_payload_trees(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """``base`` with ``extra`` merged in; nested mappings merge key by key."""
    merged = dict(base)
    for key, value in extra.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_payload_trees(current, value)
        else:
            merged[key] = value
    return merged


def _slice_pooler_value(
    value: Any,
    *,
    req_index: int,
    start: int,
    end: int,
    total_tokens: int,
    padded_total_tokens: int | None = None,
    request_scoped: bool = False,
) -> Any:
    if isinstance(value, torch.Tensor):
        token_axis_sizes = {total_tokens}
        if padded_total_tokens is not None:
            token_axis_sizes.add(padded_total_tokens)
        if not request_scoped and value.dim() > 0 and value.shape[0] in token_axis_sizes:
            return value[start:end].contiguous()
        return value.clone()
    if isinstance(value, dict):
        return {
            key: _slice_pooler_value(
                val,
                req_index=req_index,
                start=start,
                end=end,
                total_tokens=total_tokens,
                padded_total_tokens=padded_total_tokens,
                request_scoped=request_scoped,
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        if not value:
            return []
        # Lists at the batch level contain request-owned payloads. Their
        # tensors use local axes even when a size coincides with batch tokens.
        values = value if request_scoped else [value[req_index] if req_index < len(value) else value[0]]
        owned = [
            _slice_pooler_value(
                item,
                req_index=req_index,
                start=start,
                end=end,
                total_tokens=total_tokens,
                request_scoped=True,
            )
            for item in values
        ]
        return owned if request_scoped else owned[0]
    return value


def _ensure_tensor_values(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert a flattened payload to strictly ``dict[str, torch.Tensor]``.

    Non-tensor scalars (int/float/bool) are wrapped with ``torch.tensor()``;
    values that cannot be safely converted are dropped. Enforces the tensor-only
    invariant required by ``OmniEngineCoreOutput.multimodal_output`` (the channel
    the async_chunk stage-input processor reads). Mirrors the V1 runner helper.
    """
    result: dict[str, torch.Tensor] = {}
    for key, val in payload.items():
        if isinstance(val, torch.Tensor):
            result[key] = val
        elif isinstance(val, (int, float, bool)):
            result[key] = torch.tensor(val)
        elif isinstance(val, (list, tuple)):
            try:
                result[key] = torch.tensor(val)
            except (ValueError, TypeError, RuntimeError):
                logger.warning(
                    "Dropping non-tensorizable multimodal output key '%s' (type=%s).",
                    key,
                    type(val).__name__,
                )
        else:
            logger.warning(
                "Dropping non-tensor multimodal output key '%s' (type=%s).",
                key,
                type(val).__name__,
            )
    return result


class OmniAsyncOutput(AsyncModelRunnerOutput):
    """Async D2H copy for Omni AR model outputs.

    Mirrors upstream ``AsyncOutput`` but additionally handles
    ``pooler_output`` (hidden states + multimodal outputs) via
    non-blocking copies on the copy stream.
    """

    def __init__(
        self,
        model_runner_output: OmniModelRunnerOutput,
        sampler_output: Any,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event | None = None,
        text_hidden: torch.Tensor | None = None,
        multimodal_outputs: dict | None = None,
        input_batch: Any | None = None,
        async_chunk: bool = False,
        finalize_output: Any | None = None,
        finalize_multimodal: Any | None = None,
        check_ep_fault: bool = False,
        pending_aux_output: Any | None = None,
        streaming_audio: StreamingAudioOutput | None = None,
        extra_multimodal_outputs: tuple[dict[str, Any], torch.cuda.Event] | None = None,
    ):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.pending_aux_output = pending_aux_output
        self.copy_event = copy_event if copy_event is not None else torch.cuda.Event(blocking=True)
        self._async_chunk = bool(async_chunk)
        self._finalize_output = finalize_output
        self._finalize_multimodal = finalize_multimodal
        self._has_fault: torch.Tensor | None = None

        # Snapshot input_batch metadata needed for pooler_output slicing
        self._need_pooler = text_hidden is not None or (
            (self._async_chunk or finalize_multimodal is not None) and bool(multimodal_outputs)
        )
        self._query_start_loc_np: np.ndarray | None = None
        self._num_scheduled_tokens: np.ndarray | None = None
        self._num_reqs: int = 0
        self._total_tokens: int = 0
        self._padded_total_tokens: int = 0
        if self._need_pooler and input_batch is not None:
            self._query_start_loc_np = input_batch.query_start_loc_np.copy()
            self._num_scheduled_tokens = np.array(input_batch.num_scheduled_tokens, dtype=np.int32)
            self._num_reqs = input_batch.num_reqs
            if self._query_start_loc_np.shape[0] > self._num_reqs:
                self._total_tokens = int(self._query_start_loc_np[self._num_reqs])
            else:
                self._total_tokens = int(self._num_scheduled_tokens[: self._num_reqs].sum())
            self._padded_total_tokens = int(getattr(input_batch, "num_tokens_after_padding", self._total_tokens))

        # Perform all D2H copies on the copy stream (non-blocking).
        import contextlib

        @contextlib.contextmanager
        def _stream(to_stream, from_stream):
            try:
                torch.cuda.set_stream(to_stream)
                yield
            finally:
                torch.cuda.set_stream(from_stream)

        with _stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            pin_memory = PIN_MEMORY

            # Sampled token ids
            self.sampled_token_ids_np = _async_copy_to_np(
                sampler_output.sampled_token_ids,
                copy_stream=copy_stream,
                pin_memory=pin_memory,
            )
            self.num_sampled_tokens_np = _async_copy_to_np(
                num_sampled_tokens,
                copy_stream=copy_stream,
                pin_memory=pin_memory,
            )
            self.sampling_mask_tensors = None
            if sampler_output.sampling_mask_tensors is not None:
                self.sampling_mask_tensors = sampler_output.sampling_mask_tensors.to_cpu_nonblocking()
            if pending_aux_output is not None:
                pending_aux_output.enqueue_cpu_copy(
                    num_sampled=self.num_sampled_tokens_np,
                    num_rejected=_async_copy_to_np(
                        sampler_output.num_rejected,
                        copy_stream=copy_stream,
                        pin_memory=pin_memory,
                    ),
                )

            # Logprobs
            self.logprobs_tensors = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = sampler_output.logprobs_tensors.to_cpu_nonblocking()
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = _async_copy_to_np(
                    sampler_output.num_nans,
                    copy_stream=copy_stream,
                    pin_memory=pin_memory,
                )

            # Prompt logprobs
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            self.prompt_token_id_logprobs_dict = {
                k: _async_copy_tensor(v, copy_stream=copy_stream, pin_memory=pin_memory)
                for k, v in self.model_runner_output.prompt_token_id_logprobs_dict.items()
            }
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)

            # Pooler output (hidden + multimodal) — async D2H
            self._hidden_cpu: torch.Tensor | None = None
            self._mm_cpu: dict[str, Any] = {}
            self._mm_snapshot: dict[str, Any] = {}
            self._streaming_audio = (
                streaming_audio.to_cpu(
                    copy_stream, partial(_async_copy_tensor, copy_stream=copy_stream, pin_memory=pin_memory)
                )
                if streaming_audio is not None
                else None
            )
            if self._need_pooler and (self._async_chunk or self._finalize_multimodal is not None):
                # CUDA graph replay reuses the model's output buffers. Take
                # ownership directly in pinned host memory on the output copy
                # stream so deferred finalization never performs a blocking
                # D2H copy on the runner thread.
                if (
                    streaming_audio is None
                    or self._finalize_multimodal is not None
                    or extra_multimodal_outputs is not None
                ):
                    # PCM owns its request partition. Generic codes/meta are
                    # discarded unless a finalizer or extra payload needs them.
                    self._mm_snapshot = _async_copy_mm(
                        multimodal_outputs,
                        self._total_tokens,
                        copy_stream=copy_stream,
                        pin_memory=pin_memory,
                    )
                if extra_multimodal_outputs:
                    # Produced after sampling on the producer stream; copy only
                    # once its completion event has been observed.
                    extra_outputs, extra_ready = extra_multimodal_outputs
                    copy_stream.wait_event(extra_ready)
                    self._mm_snapshot = _merge_payload_trees(
                        self._mm_snapshot,
                        _async_copy_mm(
                            extra_outputs,
                            self._total_tokens,
                            copy_stream=copy_stream,
                            pin_memory=pin_memory,
                        ),
                    )
            elif self._need_pooler and text_hidden is not None:
                self._hidden_cpu = _async_copy_tensor(
                    text_hidden,
                    copy_stream=copy_stream,
                    pin_memory=pin_memory,
                )
                total_tokens = text_hidden.shape[0]
                self._mm_cpu = _async_copy_mm(
                    multimodal_outputs,
                    total_tokens,
                    copy_stream=copy_stream,
                    pin_memory=pin_memory,
                )

            self.copy_event.record(copy_stream)

    def get_output(self) -> OmniModelRunnerOutput:
        self.copy_event.synchronize()

        # Sampled token ids
        sampled_token_ids: list[list[int]] = self.sampled_token_ids_np.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids
        if self.sampling_mask_tensors is not None:
            self.model_runner_output.sampling_masks = self.sampling_mask_tensors.tolists()
        if self.pending_aux_output is not None:
            self.model_runner_output.aux_output_connector_output = self.pending_aux_output.process_output()
        self.model_runner_output.sampled_token_ids_materialized = True

        # Logprobs
        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )
        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        self.model_runner_output.prompt_token_id_logprobs_dict = self.prompt_token_id_logprobs_dict

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: one or more ranks "
                f"timed out during dispatch/combine. Mask: {mask.cpu().tolist()}"
            )

        # Pooler output. Populate two channels from the same per-request payloads,
        # mirroring the V1 runner:
        #   * pooler_output  -> sync/full-payload path (inline pooling_output bridge)
        #   * multimodal_outputs -> wire multimodal_output, which the async_chunk
        #     stage-input processor (talker2code2wav_async_chunk) reads for codes.
        if self._need_pooler and (self._async_chunk or self._finalize_multimodal is not None):
            if self._finalize_multimodal is not None:
                self._mm_snapshot = self._finalize_multimodal(self._mm_snapshot, num_sampled_tokens)
            pooler_inter: list[dict[str, Any] | None] | None
            pooler_client: list[dict[str, Any] | None] | None
            if isinstance(self._mm_snapshot, RequestOutputSnapshot):
                pooler_inter, pooler_client = self._mm_snapshot.inter_stage, self._mm_snapshot.client
                if len(pooler_inter) != self._num_reqs or (
                    pooler_client is not None and len(pooler_client) != self._num_reqs
                ):
                    raise ValueError("Model-owned output snapshot does not match the request batch")
            elif self._streaming_audio is None:
                # In-stage PCM already owns its request partition. Building
                # generic code payloads here would immediately discard them.
                pooler_inter, pooler_client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
                    self._mm_snapshot,
                    self._query_start_loc_np,
                    self._num_scheduled_tokens,
                    self._num_reqs,
                    self._total_tokens,
                    self._padded_total_tokens,
                )
            if self._streaming_audio is not None:
                pooler_inter, pooler_client = None, self._streaming_audio.get_output()
                # PCM chunks are host tensors this output owns (fresh, or views of its
                # own D2H copy) and the sample rate is a shared read-only scalar:
                # publish them without a per-request host copy.
                copy_payload = _ensure_tensor_values
            else:

                def copy_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
                    return _ensure_tensor_values(_async_copy_mm_value(payload))

            self.model_runner_output.pooler_output = None if self._async_chunk else pooler_inter
            self.model_runner_output.inter_stage_outputs = pooler_inter
            self.model_runner_output.multimodal_outputs = (
                [copy_payload(p) if p else {} for p in pooler_client] if pooler_client else None
            )
        elif self._need_pooler and self._hidden_cpu is not None:
            pooler_output = OmniARModelRunner._build_pooler_output_from_cpu(
                self._hidden_cpu,
                self._mm_cpu,
                self._query_start_loc_np,
                self._num_scheduled_tokens,
                self._num_reqs,
                self._padded_total_tokens,
            )
            pooler_payload = cast(list[dict[str, Any] | None], pooler_output) if pooler_output else None
            self.model_runner_output.pooler_output = pooler_payload
            self.model_runner_output.inter_stage_outputs = pooler_payload
            self.model_runner_output.multimodal_outputs = (
                [_ensure_tensor_values(p) if p else {} for p in pooler_payload] if pooler_payload else None
            )

        if self._finalize_output is not None:
            return self._finalize_output(self.model_runner_output)
        return self.model_runner_output

"""OmniARModelRunner — autoregressive stage runner on MR V2.

Extends ``OmniGPUModelRunner`` with:

* ``OmniOutput`` post-processing in ``sample_tokens``
* Per-request ``pooler_output`` construction (hidden + multimodal slices)
* Async D2H copy via ``OmniAsyncOutput`` for non-blocking output transfer
* Cross-stage KV extraction before state cleanup
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

logger = init_logger(__name__)


class OmniARModelRunner(OmniGPUModelRunner):
    """AR stage runner. Produces per-request hidden states + multimodal outputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kv_transfer_manager: OmniKVTransferManager | None = None
        self._kv_extracted_req_ids: list[str] | None = None

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
    ) -> Any:
        if not dummy_run:
            self._handle_kv_transfer_pre(scheduler_output)
        return super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            is_profile=is_profile,
        )

    # ------------------------------------------------------------------
    # sample_tokens: OmniOutput handling + pooler_output + async D2H
    # ------------------------------------------------------------------

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> OmniAsyncOutput | OmniModelRunnerOutput | ModelRunnerOutput | None:
        kv_extracted = self._kv_extracted_req_ids
        self._kv_extracted_req_ids = None

        if self.execute_model_state is None:
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        # kv_connector_output is no longer a field on vLLM 0.23.0's ExecuteModelState;
        # the base execute_model stashes it here after post_forward().
        kv_connector_output = self._last_kv_connector_output
        self._last_kv_connector_output = None
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            from vllm.v1.worker.gpu.pp_utils import pp_receive

            sampled, num_sampled, num_rejected = pp_receive(
                input_batch.num_reqs,
                max_sample_len=self.num_speculative_steps + 1,
            )
            # vLLM 0.23.0 renamed postprocess() -> postprocess_sampled() and now
            # takes idx_mapping (+ query_start_loc) instead of the InputBatch.
            self.postprocess_sampled(
                input_batch.idx_mapping,
                sampled,
                num_sampled,
                num_rejected,
                input_batch.query_start_loc,
            )
            return None

        # --- Omni: reconstruct raw model output and post-process ---
        aux = self._last_aux_output
        self._last_aux_output = None
        multimodal_outputs = self._last_multimodal_outputs
        self._last_multimodal_outputs = None
        raw_output = self._reconstruct_raw_model_output(
            hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
            aux=aux,
        )
        text_hidden, multimodal_outputs = self.model_state.postprocess_model_output(
            raw_output, input_batch, self.req_states
        )

        # --- Standard v2 sampling ---
        sampler_output, num_sampled, num_rejected = self._sample_with_prompt_token_compat(
            text_hidden,
            input_batch,
            grammar_output,
        )

        if self.use_pp:
            from vllm.v1.worker.gpu.pp_utils import pp_broadcast

            pp_broadcast(sampler_output.sampled_token_ids, num_sampled, num_rejected)

        # --- Omni: prompt logprobs ---
        assert self.prompt_logprobs_worker is not None
        # vLLM 0.23.0 dropped the prefill_len / num_computed_prefill_tokens args;
        # compute_prompt_logprobs now reads prefill_len_np /
        # num_computed_prefill_tokens_np from input_batch instead. Mirror the
        # parent GPUModelRunner's 6-arg call exactly.
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            text_hidden,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
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
            kv_connector_output=kv_connector_output,
        )
        model_runner_output.kv_extracted_req_ids = kv_extracted

        # --- Async D2H via OmniAsyncOutput ---
        async_output = OmniAsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
            text_hidden=text_hidden if need_pooler else None,
            multimodal_outputs=multimodal_outputs if need_pooler else None,
            input_batch=input_batch if need_pooler else None,
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

        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

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
                multimodal_outputs=multimodal_outputs,
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
    ) -> list[dict[str, Any]]:
        """Build pooler_output from already-CPU tensors."""
        total = hidden_cpu.shape[0]
        pooler: list[dict[str, Any]] = []
        for i in range(num_reqs):
            start = int(query_start_loc_np[i])
            end = start + int(num_scheduled_tokens[i])
            payload: dict[str, Any] = {"hidden": hidden_cpu[start:end]}
            for k, v in mm_cpu.items():
                payload[k] = _slice_pooler_value(
                    v,
                    req_index=i,
                    start=start,
                    end=end,
                    total_tokens=total,
                )
            pooler.append(flatten_payload(payload))
        return pooler

    def _sample_with_prompt_token_compat(
        self,
        text_hidden: torch.Tensor,
        input_batch: Any,
        grammar_output: GrammarOutput | None,
    ) -> tuple[Any, Any, Any]:
        """Run upstream sampling while restoring V1 prompt-id compatibility.

        Some Omni AR stages sample from a logits vocabulary smaller than the
        tokenizer/input vocabulary.  V1 corrected ``prompt_token_ids`` after it
        had the real logits tensor.  MR V2's upstream ``sample`` owns logits
        computation, so wrap ``compute_logits`` for this call and clamp from the
        actual logits shape instead of guessing from config.
        """
        compute_logits = getattr(self.model, "compute_logits", None)
        if not callable(compute_logits):
            return self.sample(text_hidden, input_batch, grammar_output)

        def compute_logits_with_prompt_token_compat(*args: Any, **kwargs: Any) -> Any:
            logits = compute_logits(*args, **kwargs)
            logits_shape = getattr(logits, "shape", ()) if logits is not None else ()
            logits_vocab_size = logits_shape[-1] if logits_shape else None
            if isinstance(logits_vocab_size, int):
                self._clamp_sampling_prompt_token_ids(input_batch, logits_vocab_size)
            return logits

        model_dict = getattr(self.model, "__dict__", {})
        had_instance_compute_logits = isinstance(model_dict, dict) and "compute_logits" in model_dict
        original_instance_compute_logits = model_dict.get("compute_logits") if had_instance_compute_logits else None
        setattr(self.model, "compute_logits", compute_logits_with_prompt_token_compat)
        try:
            return self.sample(text_hidden, input_batch, grammar_output)
        finally:
            if had_instance_compute_logits:
                setattr(self.model, "compute_logits", original_instance_compute_logits)
            else:
                try:
                    delattr(self.model, "compute_logits")
                except AttributeError:
                    setattr(self.model, "compute_logits", compute_logits)

    @staticmethod
    def _clamp_sampling_prompt_token_ids(
        input_batch: Any,
        logits_vocab_size: int | None,
    ) -> None:
        """Clamp sampler prompt IDs to the stage logits vocabulary.

        V1 Omni AR runner did this after computing logits.  MR V2 samples via
        the upstream helper, so normalize the metadata before that call.
        """
        if logits_vocab_size is None or logits_vocab_size <= 0:
            return
        if getattr(input_batch, "vocab_size", logits_vocab_size) <= logits_vocab_size:
            return

        sampling_metadata = getattr(input_batch, "sampling_metadata", None)
        if sampling_metadata is None or getattr(sampling_metadata, "no_penalties", False):
            return

        prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return

        max_token_id = logits_vocab_size - 1
        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids.clamp_(max=max_token_id)
            return

        if isinstance(prompt_token_ids, list):
            sampling_metadata.prompt_token_ids = [
                [min(int(tok), max_token_id) for tok in ids] if isinstance(ids, list) else min(int(ids), max_token_id)
                for ids in prompt_token_ids
            ]

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
        )


# ======================================================================
# OmniAsyncOutput — async D2H for Omni AR outputs
# ======================================================================


def _async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


def _async_copy_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.to("cpu", non_blocking=True)


def _async_copy_mm_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _async_copy_tensor(value)
    if isinstance(value, dict):
        return {key: _async_copy_mm_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_async_copy_mm_value(val) for val in value]
    return value


def _async_copy_mm(mm_outputs: dict | None, total_tokens: int) -> dict[str, Any]:
    """Non-blocking D2H copy of multimodal output tensors."""
    if not mm_outputs:
        return {}
    cpu: dict[str, Any] = {}
    for k, v in mm_outputs.items():
        try:
            cpu[k] = _async_copy_mm_value(v)
        except Exception:
            logger.exception("Error async-copying multimodal output %s", k)
    return cpu


def _slice_pooler_value(
    value: Any,
    *,
    req_index: int,
    start: int,
    end: int,
    total_tokens: int,
) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and value.shape[0] == total_tokens:
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
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        if not value:
            return []
        elem = value[req_index] if req_index < len(value) else value[0]
        return _slice_pooler_value(
            elem,
            req_index=req_index,
            start=start,
            end=end,
            total_tokens=total_tokens,
        )
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
    ):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_event = copy_event if copy_event is not None else torch.cuda.Event()

        # Snapshot input_batch metadata needed for pooler_output slicing
        self._need_pooler = text_hidden is not None
        self._query_start_loc_np: np.ndarray | None = None
        self._num_scheduled_tokens: np.ndarray | None = None
        self._num_reqs: int = 0
        if self._need_pooler and input_batch is not None:
            self._query_start_loc_np = input_batch.query_start_loc_np.copy()
            self._num_scheduled_tokens = np.array(input_batch.num_scheduled_tokens, dtype=np.int32)
            self._num_reqs = input_batch.num_reqs

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

            # Sampled token ids
            self.sampled_token_ids_np = _async_copy_to_np(sampler_output.sampled_token_ids)
            self.num_sampled_tokens_np = _async_copy_to_np(num_sampled_tokens)

            # Logprobs
            self.logprobs_tensors = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = sampler_output.logprobs_tensors.to_cpu_nonblocking()
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = _async_copy_to_np(sampler_output.num_nans)

            # Prompt logprobs
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }

            # Pooler output (hidden + multimodal) — async D2H
            self._hidden_cpu: torch.Tensor | None = None
            self._mm_cpu: dict[str, Any] = {}
            if self._need_pooler and text_hidden is not None:
                self._hidden_cpu = _async_copy_tensor(text_hidden)
                total_tokens = text_hidden.shape[0]
                self._mm_cpu = _async_copy_mm(multimodal_outputs, total_tokens)

            self.copy_event.record(copy_stream)

    def get_output(self) -> OmniModelRunnerOutput:
        self.copy_event.synchronize()

        # Sampled token ids
        sampled_token_ids: list[list[int]] = self.sampled_token_ids_np.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        # Logprobs
        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )
        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict

        # Pooler output. Populate two channels from the same per-request payloads,
        # mirroring the V1 runner:
        #   * pooler_output  -> sync/full-payload path (inline pooling_output bridge)
        #   * multimodal_outputs -> wire multimodal_output, which the async_chunk
        #     stage-input processor (talker2code2wav_async_chunk) reads for codes.
        if self._need_pooler and self._hidden_cpu is not None:
            pooler_output = OmniARModelRunner._build_pooler_output_from_cpu(
                self._hidden_cpu,
                self._mm_cpu,
                self._query_start_loc_np,
                self._num_scheduled_tokens,
                self._num_reqs,
            )
            self.model_runner_output.pooler_output = pooler_output
            self.model_runner_output.multimodal_outputs = [
                _ensure_tensor_values(p) if p else {} for p in pooler_output
            ]

        return self.model_runner_output

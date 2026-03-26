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

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
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
    ) -> Any:
        if not dummy_run:
            self._handle_kv_transfer_pre(scheduler_output)
        return super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
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
        kv_connector_output = self.execute_model_state.kv_connector_output
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            from vllm.v1.worker.gpu.pp_utils import pp_receive

            sampled, num_sampled, num_rejected = pp_receive(
                input_batch.num_reqs,
                max_sample_len=self.num_speculative_steps + 1,
            )
            self.postprocess(input_batch, sampled, num_sampled, num_rejected)
            return None

        # --- Omni: reconstruct raw model output and post-process ---
        aux = self._last_aux_output
        self._last_aux_output = None
        raw_output: Any = hidden_states
        if aux is not None:
            raw_output = (hidden_states, aux)
        text_hidden, multimodal_outputs = self.model_state.postprocess_model_output(
            raw_output, input_batch, self.req_states
        )

        # --- Standard v2 sampling ---
        sampler_output, num_sampled, num_rejected = self.sample(
            text_hidden, input_batch, grammar_output
        )

        if self.use_pp:
            from vllm.v1.worker.gpu.pp_utils import pp_broadcast

            pp_broadcast(
                sampler_output.sampled_token_ids, num_sampled, num_rejected
            )

        # --- Omni: prompt logprobs ---
        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            text_hidden,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
            self.req_states.prefill_len.np,
            self.req_states.num_computed_prefill_tokens,
        )

        # --- Omni: pooler_output ---
        engine_output_type = getattr(
            self.vllm_config.model_config, "engine_output_type", "text"
        )
        need_pooler = engine_output_type != "text"

        # --- Build base output ---
        model_runner_output = OmniModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={
                rid: i for i, rid in enumerate(input_batch.req_ids)
            },
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
            copy_event=self.output_copy_event,
            text_hidden=text_hidden if need_pooler else None,
            multimodal_outputs=multimodal_outputs if need_pooler else None,
            input_batch=input_batch if need_pooler else None,
        )

        # Postprocess AFTER creating async output (so copy_event is
        # recorded before postprocess, matching upstream pattern).
        self.postprocess(
            input_batch,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
        )

        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    # ------------------------------------------------------------------
    # pooler_output construction
    # ------------------------------------------------------------------

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
                if isinstance(v, torch.Tensor) and v.shape[0] == total:
                    payload[k] = v[start:end].contiguous()
                elif isinstance(v, dict):
                    payload[k] = {
                        sk: sv[start:end].contiguous() for sk, sv in v.items()
                    }
                elif isinstance(v, list):
                    elem = v[i] if i < len(v) else v[0]
                    if isinstance(elem, torch.Tensor):
                        elem = elem.clone()
                    payload[k] = elem
                else:
                    payload[k] = v
            pooler.append(payload)
        return pooler

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


def _async_copy_mm(
    mm_outputs: dict | None, total_tokens: int
) -> dict[str, Any]:
    """Non-blocking D2H copy of multimodal output tensors."""
    if not mm_outputs:
        return {}
    cpu: dict[str, Any] = {}
    for k, v in mm_outputs.items():
        try:
            if isinstance(v, torch.Tensor) and v.shape[0] == total_tokens:
                cpu[k] = _async_copy_tensor(v)
            elif isinstance(v, dict):
                sub: dict[str, torch.Tensor] = {}
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor) and sv.shape[0] == total_tokens:
                        sub[str(sk)] = _async_copy_tensor(sv)
                if sub:
                    cpu[k] = sub
            elif isinstance(v, list) and v:
                cpu[k] = [
                    (_async_copy_tensor(el) if isinstance(el, torch.Tensor) else el)
                    for el in v
                ]
        except Exception:
            logger.exception("Error async-copying multimodal output %s", k)
    return cpu


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
        copy_event: torch.cuda.Event,
        text_hidden: torch.Tensor | None = None,
        multimodal_outputs: dict | None = None,
        input_batch: Any | None = None,
    ):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_event = copy_event

        # Snapshot input_batch metadata needed for pooler_output slicing
        self._need_pooler = text_hidden is not None
        self._query_start_loc_np: np.ndarray | None = None
        self._num_scheduled_tokens: np.ndarray | None = None
        self._num_reqs: int = 0
        if self._need_pooler and input_batch is not None:
            self._query_start_loc_np = input_batch.query_start_loc_np.copy()
            self._num_scheduled_tokens = np.array(
                input_batch.num_scheduled_tokens, dtype=np.int32
            )
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
            self.sampled_token_ids_np = _async_copy_to_np(
                sampler_output.sampled_token_ids
            )
            self.num_sampled_tokens_np = _async_copy_to_np(num_sampled_tokens)

            # Logprobs
            self.logprobs_tensors = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
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

        # Pooler output
        if self._need_pooler and self._hidden_cpu is not None:
            self.model_runner_output.pooler_output = (
                OmniARModelRunner._build_pooler_output_from_cpu(
                    self._hidden_cpu,
                    self._mm_cpu,
                    self._query_start_loc_np,
                    self._num_scheduled_tokens,
                    self._num_reqs,
                )
            )

        return self.model_runner_output

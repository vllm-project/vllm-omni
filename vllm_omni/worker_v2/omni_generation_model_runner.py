"""OmniGenerationModelRunner — non-autoregressive stage runner on MR V2.

Used for stages like Code2Wav that convert codec codes to audio waveforms.
No token sampling or logits computation — model output goes directly into
``pooler_output``.  Inherits from ``OmniGPUModelRunner`` for intermediate
buffer and lifecycle hooks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.model_runner import (
    BatchDescriptor,
    IntermediateTensors,
    get_uniform_token_count,
)

from vllm_omni.core.sched.output import OmniCachedRequestData, OmniNewRequestData
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.forward_compat import add_forward_compat_kwargs
from vllm_omni.worker_v2.omni_ar_model_runner import _ensure_tensor_values
from vllm_omni.worker_v2.omni_model_runner import (
    OmniGPUModelRunner,
    _make_execute_model_state,
)

logger = init_logger(__name__)


class OmniGenerationModelRunner(OmniGPUModelRunner):
    """Non-autoregressive generation runner (e.g. Code2Wav).

    Overrides ``execute_model`` to skip the tensor-only assertion and
    ``sample_tokens`` to construct ``pooler_output`` from multimodal
    model outputs without performing token sampling.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gen_model_output: Any = None
        self._gen_input_batch: Any = None
        self._gen_kv_connector_output: Any = None
        # Placeholder for ExecuteModelState.hidden_states — allocated
        # once and reused every step to avoid per-forward allocation.
        self._dummy_hidden = torch.zeros(1, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------
    # Async-chunk support: replace prompt_token_ids for cached requests
    # ------------------------------------------------------------------

    def _handle_async_chunk_updates(self, scheduler_output: SchedulerOutput) -> None:
        """In-place update cached requests whose prompt_token_ids changed.

        In async_chunk mode, the ``ChunkTransferAdapter`` replaces
        ``Request.prompt_token_ids`` with new codec frames for each
        chunk and resets ``num_computed_tokens`` to 0.  The scheduler
        propagates the new ``prompt_token_ids`` via
        ``OmniCachedRequestData``.

        Instead of remove + re-add (which involves free_indices churn
        and redundant model_state init), we update the existing slot
        in-place.  This is safe for Code2Wav because:
        - No KV cache / rope state to reinitialize
        - staged writes are applied once at the end

        The old intermediate buffer for this slot is cleared here; the
        inherited ``OmniGPUModelRunner.update_requests`` (called right after
        this method in ``execute_model``) writes the current chunk state.
        """
        cached = scheduler_output.scheduled_cached_reqs
        if not cached.req_ids:
            return

        if not isinstance(cached, OmniCachedRequestData):
            return

        new_prompt_ids = cached.prompt_token_ids
        if not new_prompt_ids:
            return

        updated = False
        released_chunks: list[OmniNewRequestData] = []

        for i, req_id in enumerate(cached.req_ids):
            new_ids = new_prompt_ids.get(req_id)
            if new_ids is None:
                continue

            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is None:
                block_ids = cached.new_block_ids[i]
                released_chunks.append(
                    OmniNewRequestData(
                        req_id=req_id,
                        prompt_token_ids=new_ids,
                        mm_features=[],
                        sampling_params=None,
                        pooling_params=None,
                        block_ids=block_ids if block_ids is not None else tuple(),
                        num_computed_tokens=0,
                        lora_request=None,
                        prompt_embeds=None,
                        prefill_token_ids=new_ids,
                        additional_information=cached.additional_information.get(req_id),
                    )
                )
                continue

            self.model_state.intermediate_buffer.remove_request(req_idx)

            # In-place update token state — same slot, no remove/re-add.
            # .np[] = direct write (no GPU buffer); stage_write = GPU-synced.
            n = len(new_ids)
            self.req_states.prompt_len.np[req_idx] = n
            self.req_states.prefill_len.np[req_idx] = n
            self.req_states.total_len.stage_write_elem(req_idx, n)
            self.req_states.all_token_ids.stage_write(req_idx, 0, new_ids)
            self.req_states.num_computed_tokens.stage_write_elem(req_idx, 0)
            self.req_states.num_computed_prefill_tokens[req_idx] = 0

            updated = True

        if released_chunks:
            self.add_requests(SimpleNamespace(scheduled_new_reqs=released_chunks))
        if updated:
            self.req_states.apply_staged_writes()

    def _release_generation_slots(self, input_batch: Any) -> None:
        if not getattr(self.model_config, "async_chunk", False):
            return
        model_state = getattr(self, "model_state", None)
        remove_request = getattr(self, "_remove_request", None)
        if model_state is None or remove_request is None:
            return
        for i in range(input_batch.num_reqs):
            req_id = input_batch.req_ids[i]
            req_idx = int(input_batch.idx_mapping_np[i])
            model_state.remove_request(req_idx)
            remove_request(req_id)

    # ------------------------------------------------------------------
    # profile / warmup — skip sampler since there are no logits
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Generation models have no KV cache — skip profiling.

        Code2Wav shares GPU memory with the Talker stage (same device);
        its memory footprint is managed via ``gpu_memory_utilization``
        config, not profiled dynamically.  Running the real model with
        random input_ids causes out-of-bounds indexing in codec lookup
        tables.
        """
        torch.accelerator.synchronize()

    # ------------------------------------------------------------------
    # execute_model — run the generation model, store raw output
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            # Handle async_chunk prompt_token_ids replacement for cached
            # requests BEFORE add/update — update the existing slot
            # in-place with the new chunk's tokens.
            self._handle_async_chunk_updates(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                return self.kv_connector.no_forward(scheduler_output)

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)
        batch_desc, _ = self._dispatch_batch_descriptor(
            num_reqs=num_reqs,
            num_toks=num_toks,
            uniform_tok_count=uniform_tok_count,
            use_eager=is_profile,
        )

        if batch_desc.num_tokens == 0:
            return self.kv_connector.no_forward(scheduler_output)

        if not dummy_run:
            input_batch = self.prepare_inputs(scheduler_output, batch_desc)
        else:
            from vllm.v1.worker.gpu.input_batch import InputBatch

            input_batch = InputBatch.make_dummy(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
            )

        attn_metadata = None
        slot_mappings_by_layer = None

        inputs_embeds = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            # vLLM 0.23 dropped the req_states arg from get_mm_embeddings.
            inputs_embeds = self.model_state.get_mm_embeddings(
                scheduler_output.scheduled_encoder_inputs,
                input_batch,
            )

        model_inputs = {
            "input_ids": input_batch.input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": intermediate_tensors,
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        add_forward_compat_kwargs(model_inputs, input_batch, self.sampler)

        batch_descriptor = BatchDescriptor(
            num_tokens=input_batch.num_tokens_after_padding,
            has_lora=self.lora_config is not None,
        )
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=input_batch.num_tokens_after_padding,
            cudagraph_runtime_mode=batch_desc.cg_mode,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mappings_by_layer,
        ):
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.model(**model_inputs)

        kv_connector_output = self.kv_connector.post_forward(scheduler_output.finished_req_ids)

        # Convert raw model output to OmniOutput.
        if not isinstance(model_output, OmniOutput):
            buffer_list = self.model_state.intermediate_buffer.gather(input_batch)
            try:
                model_output = self.model.make_omni_output(
                    model_output,
                    model_intermediate_buffer=buffer_list,
                    runtime_additional_information=buffer_list,
                )
            except Exception:
                logger.error(
                    "make_omni_output failed; returning empty output",
                    exc_info=True,
                )
                self._gen_model_output = None
                self.execute_model_state = _make_execute_model_state(
                    input_batch=input_batch,
                    attn_metadata=None,
                    slot_mappings_by_layer=None,
                    hidden_states=self._dummy_hidden,
                    aux_hidden_states=None,
                    finished_req_ids=scheduler_output.finished_req_ids,
                    kv_connector_output=kv_connector_output,
                    num_tokens_across_dp=None,
                )
                return None

        self._gen_model_output = model_output
        self._gen_input_batch = input_batch
        self._gen_kv_connector_output = kv_connector_output

        # ExecuteModelState is required by the upstream engine loop
        # (EngineCore checks execute_model_state is not None before
        # calling sample_tokens).
        self.execute_model_state = _make_execute_model_state(
            input_batch=input_batch,
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=self._dummy_hidden,
            aux_hidden_states=None,
            finished_req_ids=scheduler_output.finished_req_ids,
            kv_connector_output=kv_connector_output,
            num_tokens_across_dp=None,
        )
        return None

    # ------------------------------------------------------------------
    # sample_tokens — build pooler_output, no actual sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput | None = None) -> OmniModelRunnerOutput | None:
        model_output = self._gen_model_output
        input_batch = self._gen_input_batch
        kv_connector_output = self._gen_kv_connector_output
        self._gen_model_output = None
        self._gen_input_batch = None
        self._gen_kv_connector_output = None
        self.execute_model_state = None

        if model_output is None or input_batch is None:
            return None

        num_reqs = input_batch.num_reqs

        # Mark all scheduled tokens as computed so the scheduler does
        # not re-schedule them.  Unlike AR stages we do NOT call
        # self.postprocess() — that kernel advances num_computed_tokens
        # by 1 and emits sampled tokens, which would cause check_stop
        # to fire.  Instead, set num_computed_tokens = prompt_len
        # directly, matching V1's behavior.
        for i in range(num_reqs):
            req_idx = int(input_batch.idx_mapping_np[i])
            prompt_len = int(self.req_states.prompt_len.np[req_idx])
            self.req_states.num_computed_tokens.stage_write_elem(req_idx, prompt_len)
        self.req_states.num_computed_tokens.apply_write()

        # Build pooler_output from OmniOutput.multimodal_outputs (dict).
        pooler_output = self._build_pooler_output(model_output, num_reqs)

        req_ids = input_batch.req_ids

        # Generation models don't do token sampling.  Return one empty
        # list per request so the scheduler does NOT trigger check_stop
        # (which would prematurely finish the request).  The request
        # stays RUNNING until the orchestrator marks it done via
        # chunk_transfer_adapter.finished_requests.
        sampled_token_ids: list[list[int]] = [[] for _ in range(len(req_ids))]
        self._release_generation_slots(input_batch)

        # Per-request multimodal_output (tensor-only) built from the same payloads.
        # OmniGenerationScheduler indexes this as mm_outputs[req_index], and for the
        # final (code2wav) stage it becomes the user-facing outputs[0].multimodal_output.
        # Must be a per-request list (not the raw dict), matching the V1 runner.
        multimodal_outputs = [_ensure_tensor_values(p) if p else {} for p in pooler_output]

        return OmniModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            pooler_output=pooler_output,
            multimodal_outputs=multimodal_outputs,
            kv_connector_output=kv_connector_output,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pooler_output(
        model_output: OmniOutput,
        num_reqs: int,
    ) -> list[dict[str, Any] | None]:
        """Extract per-request pooler payloads from model output.

        Code2Wav's ``make_omni_output`` returns
        ``{"model_outputs": [tensor_per_req, ...], "sr": [...]}``,
        so each value is a ``list`` with ``len == num_reqs``.
        """
        mm = model_output.multimodal_outputs
        if not isinstance(mm, dict):
            logger.warning(
                "Unexpected multimodal_outputs type: %s; returning empty pooler_output",
                type(mm).__name__ if mm is not None else "None",
            )
            return [None] * num_reqs

        pooler: list[dict[str, Any] | None] = []
        for i in range(num_reqs):
            payload: dict[str, Any] = {}
            for key, val in mm.items():
                # Primary path: val is list[Tensor] with len == num_reqs
                # (Code2Wav make_omni_output format).
                if isinstance(val, list) and len(val) == num_reqs:
                    out = val[i]
                    payload[key] = out.detach().cpu().contiguous() if isinstance(out, torch.Tensor) else out
                elif isinstance(val, torch.Tensor):
                    if val.dim() > 0 and val.shape[0] == num_reqs:
                        payload[key] = val[i].detach().cpu().contiguous()
                    else:
                        payload[key] = val.detach().cpu().contiguous()
                else:
                    payload[key] = val
            pooler.append(payload)
        return pooler

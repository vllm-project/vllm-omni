"""OmniGenerationModelRunner — non-autoregressive stage runner on MR V2.

Used for stages like Code2Wav that convert codec codes to audio waveforms.
No token sampling or logits computation — model output goes directly into
``pooler_output``.  Inherits from ``OmniGPUModelRunner`` for intermediate
buffer and lifecycle hooks.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.model_runner import (
    BatchDescriptor,
    ExecuteModelState,
    IntermediateTensors,
    get_uniform_token_count,
)

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

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

    # ------------------------------------------------------------------
    # profile / warmup — skip sampler since there are no logits
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Generation models have no KV cache — skip full dummy forward.

        Only allocate a small dummy tensor to measure baseline memory.
        Running the real model with random input_ids causes out-of-bounds
        indexing in codec lookup tables.
        """
        dummy = torch.empty(1, device=self.device, dtype=self.dtype)
        del dummy
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
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                return self.kv_connector.no_forward(scheduler_output)

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)
        batch_desc = self.cudagraph_manager.dispatch(num_reqs, num_toks, uniform_tok_count)

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
            inputs_embeds = self.model_state.get_mm_embeddings(
                scheduler_output.scheduled_encoder_inputs,
                input_batch,
                self.req_states,
            )

        model_inputs = {
            "input_ids": input_batch.input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": intermediate_tensors,
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }

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

        kv_connector_output = self.kv_connector.post_forward(scheduler_output)

        # Convert via make_omni_output if available
        if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
            buffer_list = self.model_state.intermediate_buffer.gather(input_batch)
            try:
                model_output = self.model.make_omni_output(
                    model_output,
                    model_intermediate_buffer=buffer_list,
                    runtime_additional_information=buffer_list,
                )
            except Exception:
                logger.error(
                    "make_omni_output failed for generation output; "
                    "raw output will be used as multimodal_outputs",
                    exc_info=True,
                )

        self._gen_model_output = model_output
        self._gen_input_batch = input_batch
        self._gen_kv_connector_output = kv_connector_output

        # Set execute_model_state so _dummy_run (profile/warmup) doesn't
        # trip on the assertion.  hidden_states is a dummy tensor because
        # generation stages don't go through the normal sampling path.
        dummy_hidden = torch.zeros(
            1,
            dtype=self.dtype,
            device=self.device,
        )
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=dummy_hidden,
            aux_hidden_states=None,
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

        # Resolve the EOS token once.  Used both for postprocess
        # (advancing num_computed_tokens) and for the returned
        # sampled_token_ids (triggering scheduler's check_stop).
        # OmniModelConfig may not expose eos_token_id directly;
        # fall back through hf_text_config, then default to 0.
        eos_id = getattr(self.model_config, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(
                getattr(self.model_config, "hf_text_config", None),
                "eos_token_id",
                None,
            )
        if eos_id is None:
            eos_id = 0

        # Advance num_computed_tokens via the upstream postprocess
        # kernel.  Without this the scheduler re-schedules the same
        # tokens every step (infinite loop).
        dummy_sampled = torch.full(
            (num_reqs, 1), eos_id, dtype=torch.long, device=self.device
        )
        dummy_num_sampled = torch.ones(
            num_reqs, dtype=torch.int32, device=self.device
        )
        dummy_num_rejected = torch.zeros(
            num_reqs, dtype=torch.int32, device=self.device
        )
        self.postprocess(
            input_batch, dummy_sampled, dummy_num_sampled, dummy_num_rejected
        )

        # Build pooler_output from multimodal model outputs.
        multimodal_outputs: dict | list | torch.Tensor | None = None
        if isinstance(model_output, OmniOutput):
            multimodal_outputs = model_output.multimodal_outputs
        else:
            multimodal_outputs = model_output

        pooler_output: list[object] = []
        if isinstance(multimodal_outputs, torch.Tensor):
            for i in range(num_reqs):
                pooler_output.append(
                    {"model_outputs": multimodal_outputs[i].detach().cpu().contiguous()}
                )
        elif isinstance(multimodal_outputs, list):
            for out in multimodal_outputs:
                pooler_output.append(
                    {"model_outputs": out.detach().cpu().contiguous() if out is not None else None}
                )
        elif isinstance(multimodal_outputs, dict):
            for i in range(num_reqs):
                mm_payload: dict[str, Any] = {}
                for key, val in multimodal_outputs.items():
                    if isinstance(val, torch.Tensor):
                        if val.dim() > 0 and val.shape[0] == num_reqs:
                            mm_payload[key] = val[i].detach().cpu().contiguous()
                        else:
                            mm_payload[key] = val.detach().cpu().contiguous()
                    elif isinstance(val, list) and len(val) == num_reqs:
                        out = val[i]
                        mm_payload[key] = (
                            out.detach().cpu().contiguous()
                            if isinstance(out, torch.Tensor) else out
                        )
                    else:
                        mm_payload[key] = val
                pooler_output.append(mm_payload)
        else:
            pooler_output = [None] * num_reqs

        req_ids = input_batch.req_ids[:]

        # Generation models don't do token sampling, but V2's scheduler
        # relies on sampled_token_ids to detect request completion via
        # check_stop().  Emit one EOS token per request so the scheduler
        # marks them FINISHED_STOPPED and frees the request slot.
        sampled_token_ids: list[list[int]] = [[eos_id]] * len(req_ids)

        return OmniModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            pooler_output=pooler_output,
            multimodal_outputs=(
                multimodal_outputs if isinstance(multimodal_outputs, dict)
                else {"model_outputs": multimodal_outputs}
            ),
            kv_connector_output=kv_connector_output,
        )

"""AR GPU Model Runner for vLLM-omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

from typing import Optional, Union, Any, List
import numpy as np

import torch

from vllm import envs
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    IntermediateTensors,
    get_pp_group,
    get_tp_group,
    has_kv_transfer_group,
    set_forward_context,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_omni.engine import PromptEmbedsPayload, AdditionalInformationPayload
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


class ARModelRunner(OmniGPUModelRunner):
    """Autoregressive GPU model runner that returns hidden states per request.

    This runner follows the same preparation and forward path as GPUModelRunner
    (inputs assembly, multi-modal handling, TP/PP/DP integration, CUDA graphs),
    and additionally performs lightweight sampling so that sampled tokens are
    available in outputs. Hidden representations are taken at the same indices
    that GPUModelRunner would use for sampling/logits (i.e. `logits_indices`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[OmniModelRunnerOutput, IntermediateTensors]:
        # Update internal state with the new schedule
        self._update_states(scheduler_output)

        # Decode per-request prompt_embeds / additional_hidden_states payloads (if present) into CPU tensors
        try:
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", [])
            if new_reqs:
                import numpy as np
                import torch
                for nr in new_reqs:
                    req_id = getattr(nr, "req_id", None) or getattr(nr, "request_id", None)
                    if req_id is None:
                        continue
                    # prompt_embeds
                    payload_pe = getattr(nr, "prompt_embeds", None)
                    if payload_pe is not None:
                        if isinstance(payload_pe, torch.Tensor):
                            pe_cpu = payload_pe.detach().to("cpu").contiguous()
                        elif isinstance(payload_pe, PromptEmbedsPayload):
                            dt = np.dtype(getattr(payload_pe, "dtype", "float32"))
                            arr = np.frombuffer(payload_pe.data, dtype=dt)
                            arr = arr.reshape(payload_pe.shape)
                            pe_cpu = torch.from_numpy(arr)
                        else:
                            pe_cpu = None
                        if pe_cpu is not None and req_id in self.requests:
                            setattr(self.requests[req_id], "prompt_embeds_cpu", pe_cpu)
                    # additional_information
                    payload_info = getattr(nr, "additional_information", None)
                    if payload_info is not None:
                        info_dict = {}
                        if isinstance(payload_info, dict):
                            # Already decoded
                            info_dict = payload_info
                        elif isinstance(payload_info, AdditionalInformationPayload):
                            for k, entry in payload_info.entries.items():
                                if entry.tensor_data is not None:
                                    dt = np.dtype(getattr(entry, "tensor_dtype", "float32"))
                                    arr = np.frombuffer(entry.tensor_data, dtype=dt)
                                    arr = arr.reshape(entry.tensor_shape)
                                    info_dict[k] = torch.from_numpy(arr)
                                else:
                                    info_dict[k] = entry.list_data
                        if info_dict and req_id in self.requests:
                            setattr(self.requests[req_id], "additional_information_cpu", info_dict)
        except Exception:
            pass

        # If there's no work to do, either return empty output or kv-only path
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output,
                                                self.vllm_config)

        # Prepare decoder inputs and attention metadata
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata, num_scheduled_tokens_np,
         spec_decode_common_attn_metadata) = self._prepare_inputs(
             scheduler_output)

        # Determine number of input tokens for this iteration
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if (self.compilation_config.pass_config.enable_sequence_parallelism
                    and tp_size > 1):
                from vllm.utils import round_up  # lazy local import
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # DP padding
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # Multimodal handling (encode and gather embeddings if needed)
        if self.is_multimodal_model:
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        # Always assemble inputs_embeds on first PP rank; overlay per-request prompt_embeds and collect additional_hidden_states for prefill only
        if get_pp_group().is_first_rank:
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids[:num_scheduled_tokens],
                multimodal_embeddings=(mm_embeds or None)
                if self.is_multimodal_model else None,
            )

            # Copy into persistent buffer to enable CUDA Graph capture
            self.inputs_embeds[:num_scheduled_tokens].copy_(
                inputs_embeds_scheduled)

            # Reset per-step additional information collector
            if hasattr(self, "_forward_additional_information"):
                self._forward_additional_information = None

            # Overlay custom prompt_embeds per request for the prompt portion; collect additional_information (tensor/list) for prefill portion only
            for req_index, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                pe_cpu = getattr(req_state, "prompt_embeds_cpu", None)
                addi_cpu = getattr(req_state, "additional_information_cpu", None)
                num_computed_tokens = int(
                    self.input_batch.num_computed_tokens_cpu[req_index])
                prompt_len = len(req_state.prompt_token_ids)
                prompt_remaining = max(0, prompt_len - num_computed_tokens)
                sched_tokens = int(num_scheduled_tokens_np[req_index])
                overlay_len = min(sched_tokens, prompt_remaining)
                if overlay_len <= 0:
                    continue
                if pe_cpu is not None:
                    src = pe_cpu[num_computed_tokens:
                                 num_computed_tokens + overlay_len].to(
                                      dtype=self.dtype,
                                      device=self.device,
                                      non_blocking=True)
                    start_offset = int(self.query_start_loc_cpu[req_index])
                    self.inputs_embeds[start_offset:start_offset + overlay_len] \
                        .copy_(src)
                # For additional_information: handle arbitrary keys
                if addi_cpu is not None and isinstance(addi_cpu, dict):
                    # Lazy init collector dict
                    if not hasattr(self, "_forward_additional_information") or \
                       self._forward_additional_information is None:
                        self._forward_additional_information = {}
                    # Process tensors (slice by scheduled prompt range) and lists (append per-request)
                    for k, v in addi_cpu.items():
                        if isinstance(v, torch.Tensor):
                            # Slice along token dimension for prefill part
                            try:
                                seg = v[num_computed_tokens:
                                        num_computed_tokens + overlay_len].to(
                                            dtype=self.dtype,
                                            device=self.device,
                                            non_blocking=True)
                            except Exception:
                                # Fallback: move whole tensor if slicing fails
                                seg = v.to(dtype=self.dtype,
                                           device=self.device,
                                           non_blocking=True)
                            prev_val = self._forward_additional_information.get(k)
                            self._forward_additional_information[k] = (
                                torch.cat([prev_val, seg], dim=0)
                                if isinstance(prev_val, torch.Tensor) else seg.clone())
                        elif isinstance(v, list):
                            prev_val = self._forward_additional_information.get(k)
                            if prev_val is None:
                                self._forward_additional_information[k] = [v]
                            elif isinstance(prev_val, list):
                                self._forward_additional_information[k].append(v)
                            else:
                                # Mixed types: wrap existing into list
                                self._forward_additional_information[k] = [prev_val, v]

            input_ids = self.input_ids[:num_input_tokens]  # preserved for APIs
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            model_mm_kwargs = (self._extract_mm_kwargs(scheduler_output)
                               if self.is_multimodal_model else {})
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
            model_mm_kwargs = {}

        # Positions/mRoPE
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        # Handle pipeline-parallel intermediate tensors
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Some attention backends only support CUDA Graphs in pure decode.
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

        # Forward pass
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
        ), self.maybe_get_kv_connector_output(
                scheduler_output) as kv_connector_output:

            model_kwargs_extra = {}
            # Only pass additional_information for the prefill part
            if hasattr(self, "_forward_additional_information") and \
               self._forward_additional_information is not None and \
               isinstance(self._forward_additional_information, dict):
                model_kwargs_extra["additional_information"] = self._forward_additional_information
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **MultiModalKwargs.as_kwargs(
                    model_mm_kwargs,
                    device=self.device,
                ),
                sampling_metadata=self.input_batch.sampling_metadata,
                logits_index=logits_indices,
                sampler=self.sampler,
                **model_kwargs_extra,
            )

        if self.use_aux_hidden_state_outputs:
            hidden_states, _aux_hidden_states = model_output
        else:
            hidden_states = model_output

        text_hidden_states, multimodal_outputs = (
            self.extract_multimodal_outputs(hidden_states))

        # Mid PP stages return intermediate tensors unmodified
        if not get_pp_group().is_last_rank:
            assert isinstance(text_hidden_states, IntermediateTensors)
            text_hidden_states.kv_connector_output = kv_connector_output
            return text_hidden_states

        # Broadcast PP output for external_launcher (torchrun)
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            assert isinstance(text_hidden_states, IntermediateTensors)
            if not broadcast_pp_output:
                text_hidden_states.kv_connector_output = kv_connector_output
                return text_hidden_states
            get_pp_group().send_tensor_dict(text_hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(text_hidden_states, num_scheduled_tokens,
                                  num_scheduled_tokens_np, kv_connector_output)

            sample_hidden_states = text_hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed (with spec decode)
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # Handle partial prefill: discard sampled tokens and rewind RNG
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                discard_sampled_tokens_req_indices.append(i)

        # Move CPU sync parts
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Prompt logprobs if needed
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            text_hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Parse valid sampled tokens
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        # Cache sampled tokens
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue
            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Speculative decoding draft tokens if configured
        if not self.speculative_config:
            spec_token_ids = None
        else:
            assert spec_decode_common_attn_metadata is not None
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampling_metadata,
                text_hidden_states,
                sample_hidden_states,
                _aux_hidden_states if '_aux_hidden_states' in locals() else None,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )

        # Convert to per-request tensors on CPU
        pooler_output: list[Optional[torch.Tensor]] = []
        prev_logits_index = 0
        for logits_index in logits_indices:
            pooler_output.append(text_hidden_states[prev_logits_index:logits_index+1].detach().to("cpu").contiguous())
            prev_logits_index = logits_index + 1


        self.eplb_step()

        return OmniModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=pooler_output if self.vllm_config.model_config.engine_output_type != "text" else None,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )
    
    @torch.inference_mode()
    def extract_multimodal_outputs(self, hidden_states: Union[torch.Tensor, List[torch.Tensor]]) -> dict:
        if hasattr(self.model, "have_multimodal_outputs") and self.model.have_multimodal_outputs:
            text_hidden_states = hidden_states.text_hidden_states
            multimodal_outputs = hidden_states.multimodal_outputs

        elif isinstance(hidden_states, torch.Tensor):
            text_hidden_states = hidden_states
            multimodal_outputs = {}
        elif isinstance(hidden_states, List):
            text_hidden_states = hidden_states[0]
            multimodal_outputs = {}
        else:
            raise ValueError(f"Invalid hidden states type: {type(hidden_states)}")
        return text_hidden_states, multimodal_outputs
    
    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        capture_attn_cudagraph: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        attn_metadata: Optional[dict[str, Any]] = None
        if capture_attn_cudagraph:
            attn_metadata = {}

            # Make sure max_model_len is used at the graph capture time.
            self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                           non_blocking=True)

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs +
                                                                 1],
                    seq_lens=self.seq_lens[:num_reqs],
                    seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=num_tokens,
                    block_table_tensor=self.input_batch.block_table[
                        kv_cache_group_id].get_device_tensor()[:num_reqs],
                    slot_mapping=self.input_batch.
                    block_table[kv_cache_group_id].slot_mapping[:num_tokens],
                    causal=True)

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    attn_metadata_i = attn_group.metadata_builder\
                        .build_for_cudagraph_capture(common_attn_metadata)
                    for layer_name in kv_cache_group_spec.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
                model_mm_kwargs = self._dummy_mm_kwargs(num_reqs)
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
                model_mm_kwargs = {}

            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False)

            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **MultiModalKwargs.as_kwargs(
                        model_mm_kwargs,
                        device=self.device,
                    ),
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens)

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
        return hidden_states, hidden_states[logit_indices]


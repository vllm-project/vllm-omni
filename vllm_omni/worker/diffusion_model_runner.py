# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Union, List, Any
import numpy as np

import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    EMPTY_MODEL_RUNNER_OUTPUT,
    IntermediateTensors,
    get_pp_group,
    has_kv_transfer_group,
    set_forward_context,
)
from vllm.multimodal.inputs import MultiModalKwargs

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.forward_context import BatchDescriptor
from vllm.v1.spec_decode.eagle import EagleProposer


class DiffusionModelRunner(GPUModelRunner):
    """Diffusion model runner for vLLM-omni (non-autoregressive).

    - Reuses GPUModelRunner preparation, multimodal handling, and TP/PP/DP glue.
    - Does not compute logits or perform token sampling.
    - Executes diffusion process and returns tensors via `pooler_output`.
    """

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        self._update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output,
                                                self.vllm_config)

        # Prepare decoder inputs and attention metadata (for batch/order mapping)
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata, num_scheduled_tokens_np,
         spec_decode_common_attn_metadata) = self._prepare_inputs(
             scheduler_output)

        # Input token count for this iteration (not used by diffusion, but
        # retained to keep DP padding/ordering consistent)
        num_input_tokens = scheduler_output.total_num_scheduled_tokens
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # Multimodal conditioning (e.g., text/audio/video encoders)
        if self.is_multimodal_model:
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        # Build inputs to mirror AR runner: input_ids/positions/embeds
        if self.is_multimodal_model and get_pp_group().is_first_rank:
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids[:scheduler_output.total_num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )
            self.inputs_embeds[:scheduler_output.total_num_scheduled_tokens].copy_(
                inputs_embeds_scheduled)
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            model_mm_kwargs = self._extract_mm_kwargs(scheduler_output)
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
            model_mm_kwargs = {}

        # Positions (mrope or standard)
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        # Intermediate tensors sync for PP (if any)
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Set forward context mainly for resource management and kv connector
        skip_cuda_graphs = True  # diffusion path does not rely on cuda graphs here
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
        ), self.maybe_get_kv_connector_output(
                scheduler_output) as kv_connector_output:

            if not get_pp_group().is_last_rank:
                # For non-last PP stages, pass through intermediate tensors.
                assert intermediate_tensors is not None
                intermediate_tensors.kv_connector_output = kv_connector_output
                return intermediate_tensors

            outputs = self._run_diffusion(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                multimodal_kwargs=model_mm_kwargs,
                logits_indices=logits_indices,
            )

        # Ensure one tensor per request, map to CPU for output struct
        pooler_output: List[Optional[torch.Tensor]] = []
        if isinstance(outputs, torch.Tensor):
            # If model returned a single stacked tensor, split by requests
            assert outputs.shape[0] == self.input_batch.num_reqs
            for i in range(self.input_batch.num_reqs):
                pooler_output.append(outputs[i].detach().cpu())
        elif isinstance(outputs, list):
            for out in outputs:
                pooler_output.append(out.detach().cpu() if out is not None else None)
        else:
            raise RuntimeError("Unsupported diffusion output type")

        self.eplb_step()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits={},
            multimodal_outputs={},
        )

    def _run_diffusion(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
        multimodal_kwargs: dict,
        logits_indices: torch.Tensor,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Runs the diffusion process and returns per-request tensors.

        Tries model interfaces in the following order for maximal compatibility:
        1) model.sample(condition=..., **kwargs)
        2) model.forward(condition=..., **kwargs)
        3) model.diffuse(condition=..., **kwargs)
        """
        # Keep inputs identical to AR runner
        kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **MultiModalKwargs.as_kwargs(multimodal_kwargs, device=self.device),
            sampling_metadata=self.input_batch.sampling_metadata,
            logits_index=logits_indices,
            sampler=self.sampler,
        )

        # For Qwen 2.5 Omni's current implementation, we only support the forward method
        if hasattr(self.model, "forward"):
            return self.model.forward(**kwargs)
        
        # if hasattr(self.model, "sample"):
        #     return self.model.sample(**kwargs)
        # if hasattr(self.model, "forward"):
        #     return self.model.forward(**kwargs)
        # if hasattr(self.model, "diffuse"):
        #     return self.model.diffuse(**kwargs)

        raise RuntimeError(
            "The loaded model does not expose diffusion interfaces 'sample', "
            "'forward', or 'diffuse'. Please implement one of them or adapt the runner.")
    
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
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        force_attention: bool = False,
        uniform_decode: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
        """
        assert cudagraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = num_tokens // 2
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [
                num_prefill_tokens
            ]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            num_reqs = num_tokens // max_query_len
            assert num_reqs <= max_num_reqs, \
                "Do not capture num_reqs > max_num_reqs for uniform batch"
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] += num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        attn_metadata: Optional[dict[str, Any]] = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            attn_metadata = {}

            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                # Make sure max_model_len is used at the graph capture time.
                seq_lens = self.max_model_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=max_query_len,
                    max_seq_len=self.max_model_len,
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
                                            num_scheduled_tokens, remove_lora):
            model_kwargs = self._init_model_kwargs(num_tokens)
            if (self.supports_mm_inputs
                    and not self.model_config.is_encoder_decoder):
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            else:
                input_ids = self.input_ids.gpu[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens]
            else:
                positions = self.positions.gpu[:num_tokens]

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
            if cudagraph_runtime_mode == CUDAGraphMode.NONE:
                batch_descriptor = None
            else:
                # filter out the valid batch descriptor
                _cg_mode, batch_descriptor = \
                    self.cudagraph_dispatcher.dispatch(
                        BatchDescriptor(num_tokens=num_tokens,
                                        uniform_decode=uniform_decode))
                # sanity check
                assert cudagraph_runtime_mode == _cg_mode, (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}.")

            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
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


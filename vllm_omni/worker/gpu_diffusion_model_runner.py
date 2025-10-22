# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Union, List
import numpy as np
import gc
import logging

import torch

from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    EMPTY_MODEL_RUNNER_OUTPUT,
    IntermediateTensors,
    get_pp_group,
    has_kv_transfer_group,
    set_forward_context,
)
from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs
from vllm.multimodal.inputs import MultiModalKwargs

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


logger = logging.getLogger(__name__)


class DiffusionModelRunner(OmniGPUModelRunner):
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
    ) -> Union[OmniModelRunnerOutput, IntermediateTensors]:
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
        _, multimodal_outputs = (
            self.extract_multimodal_outputs(outputs))

        # Ensure one tensor per request, map to CPU for output struct
        pooler_output: List[Optional[torch.Tensor]] = []
        if isinstance(multimodal_outputs, torch.Tensor):
            # If model returned a single stacked tensor, split by requests
            assert outputs.shape[0] == self.input_batch.num_reqs
            for i in range(self.input_batch.num_reqs):
                pooler_output.append(outputs[i].detach().to("cpu").contiguous())
        elif isinstance(multimodal_outputs, list):
            for out in outputs:
                pooler_output.append(out.detach().to("cpu").contiguous() if out is not None else None)
        elif isinstance(multimodal_outputs, dict):
            for out in multimodal_outputs.values():
                pooler_output.append(out.detach().to("cpu").contiguous() if out is not None else None)
        else:
            raise RuntimeError("Unsupported diffusion output type")

        self.eplb_step()

        return OmniModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits={},
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

        attn_metadata: Optional[dict[str, dict]] = None
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
                    query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
                    seq_lens=self.seq_lens[:num_reqs],
                    seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=num_tokens,
                    block_table_tensor=self.input_batch.block_table[
                        kv_cache_group_id].get_device_tensor()[:num_reqs],
                    slot_mapping=self.input_batch.block_table[
                        kv_cache_group_id].slot_mapping[:num_tokens],
                    causal=True)

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    attn_metadata_i = attn_group.metadata_builder \
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

            # Diffusion path: avoid CUDA graphs; we only use context for resource wiring
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
                    sampler=None,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            # Extract multimodal outputs if present; we ignore them here because
            # dummy run returns tensors only. The actual diffusion runner returns
            # multimodal outputs via pooler_output in execute_model.
            text_hidden_states, _ = self.extract_multimodal_outputs(hidden_states)

            if not skip_eplb:
                self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return text_hidden_states, None
    
    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        logger.warning("Dummy sampler run is not implemented for diffusion model")
        return None

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache, similar to base but
        # without any logits/sampler warming.
        if self.is_multimodal_model:
            mm_budget = self.mm_budget
            assert mm_budget is not None

            # TODO: handle encoder-decoder models once supported.
            if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                (
                    dummy_modality,
                    max_tokens,
                ) = mm_budget.get_modality_with_max_tokens()
                (
                    max_mm_items_per_prompt,
                    max_mm_items_per_batch,
                ) = mm_budget.get_max_items(dummy_modality, max_tokens)

                batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                    dummy_modality,
                    max_mm_items_per_batch,
                )

                dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                    **batched_dummy_mm_inputs)

                sanity_check_mm_encoder_outputs(
                    dummy_encoder_outputs,
                    expected_num_items=max_mm_items_per_batch,
                )

                self.encoder_cache["tmp"] = dict(
                    enumerate(dummy_encoder_outputs))

        hidden_states, _ = self._dummy_run(self.max_num_tokens, is_profile=True)
        if get_pp_group().is_last_rank:
            pass  # No sampler/pooler warmup for diffusion
        self._sync_device()
        del hidden_states
        self.encoder_cache.clear()
        gc.collect()

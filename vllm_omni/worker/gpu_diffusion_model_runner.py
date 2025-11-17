from __future__ import annotations

import gc
import logging
from typing import Any, List, Optional, Union

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncGPUModelRunnerOutput,
    IntermediateTensors,
    PerLayerAttnMetadata,
    get_pp_group,
    set_forward_context,
)
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

logger = logging.getLogger(__name__)


class GPUDiffusionModelRunner(OmniGPUModelRunner):
    """Diffusion model runner for vLLM-omni (non-autoregressive).

    - Reuses GPUModelRunner preparation, multimodal handling, and TP/PP/DP glue.
    - Does not compute logits or perform token sampling.
    - Executes diffusion process and returns tensors via `pooler_output`.
    """

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
        ubatch_slices: Optional[UBatchSlices] = None,
        num_tokens_after_padding: Optional[torch.Tensor] = None,
    ) -> tuple[
        int,
        int,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[IntermediateTensors],
        dict[str, Any],
    ]:

        # Input token count for this iteration (not used by diffusion, but
        # retained to keep DP padding/ordering consistent)
        num_input_tokens = scheduler_output.total_num_scheduled_tokens
        num_pad, num_tokens_after_padding = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if (
            self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        ):
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids.gpu[:num_input_tokens],
                multimodal_embeddings=mm_embeds or None,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_input_tokens].copy_(inputs_embeds_scheduled)

            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_input_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = (
                self.is_token_ids.gpu[:num_input_tokens]
                .nonzero(as_tuple=False)
                .squeeze(1)
            )
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.get_input_embeddings(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs(num_input_tokens)
            input_ids = self.input_ids.gpu[:num_input_tokens]
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if (
            self.model_config.is_encoder_decoder
            and scheduler_output.scheduled_encoder_inputs
        ):
            encoder_inputs = self._extract_encoder_inputs(scheduler_output)
            model_kwargs.update(encoder_inputs)

        return (
            num_input_tokens,
            num_input_tokens,
            num_tokens_after_padding,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[OmniModelRunnerOutput, IntermediateTensors]:
        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)
                if not scheduler_output.total_num_scheduled_tokens:
                    return EMPTY_MODEL_RUNNER_OUTPUT

                # Prepare the decoder inputs.
                (
                    attn_metadata,
                    logits_indices,
                    spec_decode_metadata,
                    num_scheduled_tokens_np,
                    spec_decode_common_attn_metadata,
                    max_query_len,
                    ubatch_slices,
                    num_tokens_after_padding,
                ) = self._prepare_inputs(scheduler_output)

            # Input token count for this iteration (not used by diffusion, but
            # retained to keep DP padding/ordering consistent)
            (
                num_scheduled_tokens,
                num_input_tokens,
                num_tokens_across_dp,
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
            ) = self._preprocess(
                scheduler_output,
                intermediate_tensors,
                ubatch_slices,
                num_tokens_after_padding,
            )

        cudagraph_runtime_mode = CUDAGraphMode.NONE
        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=None,
                ubatch_slices=ubatch_slices,
            ),
            record_function_or_nullcontext("Forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):

            outputs = self._run_diffusion(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                multimodal_kwargs=model_kwargs,
                logits_indices=logits_indices,
            )

        _, multimodal_outputs = self.extract_multimodal_outputs(outputs)
        # Ensure one tensor per request, map to CPU for output struct
        pooler_output: List[Optional[torch.Tensor]] = []
        if isinstance(multimodal_outputs, torch.Tensor):
            # If model returned a single stacked tensor, split by requests
            assert multimodal_outputs.shape[0] == self.input_batch.num_reqs
            for i in range(self.input_batch.num_reqs):
                pooler_output.append(
                    multimodal_outputs[i].detach().to("cpu").contiguous()
                )
        elif isinstance(multimodal_outputs, list):
            for out in multimodal_outputs:
                pooler_output.append(
                    out.detach().to("cpu").contiguous() if out is not None else None
                )
        elif isinstance(multimodal_outputs, dict):
            for out in multimodal_outputs.values():
                pooler_output.append(
                    out.detach().to("cpu").contiguous() if out is not None else None
                )
        else:
            raise RuntimeError("Unsupported diffusion output type")

        output = OmniModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits={},
        )

        if not self.use_async_scheduling:
            return output

        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=[],
            invalid_req_indices=[],
            async_output_copy_stream=self.async_output_copy_stream,
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
        # TODO: add the diffuse method for other models

        raise RuntimeError(
            "The loaded model does not expose diffusion interfaces 'sample', "
            "'forward', or 'diffuse'. Please implement one of them or adapt the runner."
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: Optional[CUDAGraphMode] = None,
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
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
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
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL,
        }

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
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

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
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)

        num_tokens_after_padding = None

        # If we failed to microbatch, currently need to resynchronize
        # TODO(lucas,sage): we should be able to avoid this second sync by
        #  refactoring `get_dp_padding_ubatch` and `get_dp_padding` into
        #  a single `coordinate_batch_across_dp` function.
        if num_tokens_after_padding is None:
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
            num_tokens_after_padding = num_tokens + num_pad
        else:
            num_tokens_across_dp = num_tokens_after_padding
            num_tokens_after_padding = int(num_tokens_after_padding[0].item())

        attn_metadata: Optional[PerLayerAttnMetadata] = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            attn_metadata = {}

            seq_lens = max_query_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups
            ):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[  # noqa: E501
                        :num_reqs
                    ],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=max_query_len,
                    max_seq_len=self.max_model_len,
                    block_table_tensor=self.input_batch.block_table[
                        kv_cache_group_id
                    ].get_device_tensor(num_reqs),
                    slot_mapping=self.input_batch.block_table[
                        kv_cache_group_id
                    ].slot_mapping.gpu[:num_tokens],
                    causal=True,
                )
                for attn_group in self.attn_groups[kv_cache_group_id]:

                    assert type(attn_metadata) is dict
                    attn_metadata_i = attn_group.get_metadata_builder().build_for_cudagraph_capture(  # noqa: E501
                        common_attn_metadata
                    )
                    for layer_name in attn_group.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(
            self.lora_config, num_scheduled_tokens, remove_lora
        ):
            model_kwargs = self._init_model_kwargs(num_tokens)
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = self._init_model_kwargs(num_tokens)
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
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False
                )

            # filter out the valid batch descriptor
            _cg_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    BatchDescriptor(
                        num_tokens=num_tokens_after_padding,
                        uniform_decode=uniform_decode,
                    )
                )
                if not is_profile
                else (CUDAGraphMode.NONE, None)
            )
            if cudagraph_runtime_mode is not None:
                # we allow forcing NONE when the dispatcher disagrees to support
                # warm ups for cudagraph capture
                assert (
                    cudagraph_runtime_mode == CUDAGraphMode.NONE
                    or cudagraph_runtime_mode == _cg_mode
                ), (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}."
                )
            else:
                cudagraph_runtime_mode = _cg_mode

            with (
                self.maybe_randomize_inputs(input_ids),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_after_padding,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                ),
            ):
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

        # logit_indices = np.cumsum(num_scheduled_tokens) - 1
        hidden_states, _ = self.extract_multimodal_outputs(hidden_states)
        return hidden_states, None

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        logger.warning("Dummy sampler run is not implemented for diffusion model")
        return None

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            if self.model_config.multimodal_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."
                )
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    # NOTE: Currently model is profiled with a single non-text
                    # modality with the max possible input tokens even when
                    # it supports multiple.
                    dummy_modality = mm_budget.get_modality_with_max_tokens()
                    max_mm_items_per_batch = mm_budget.max_items_per_batch_by_modality[
                        dummy_modality
                    ]

                    logger.info(
                        "Encoder cache will be initialized with a budget of "
                        "%s tokens, and profiled with %s %s items of the "
                        "maximum feature size.",
                        encoder_budget,
                        max_mm_items_per_batch,
                        dummy_modality,
                    )

                    # Create dummy batch of multimodal inputs.
                    batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                        dummy_modality,
                        max_mm_items_per_batch,
                    )

                    # Run multimodal encoder.
                    dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                        **batched_dummy_mm_inputs
                    )

                    sanity_check_mm_encoder_outputs(
                        dummy_encoder_outputs,
                        expected_num_items=max_mm_items_per_batch,
                    )

                    # NOTE: This happens when encoder cache needs to store
                    # the embeddings that encoder outputs are scattered onto.
                    # In this case we create dummy embeddings of size
                    # (encode_budget, hidden_size) and scatter encoder
                    # output into it.
                    encoder_output_shape = dummy_encoder_outputs[0].shape
                    if encoder_output_shape[0] < encoder_budget:
                        expanded_outputs = []
                        for output in dummy_encoder_outputs:
                            expanded = output.new_zeros(
                                (encoder_budget, encoder_output_shape[-1])
                            )
                            num_tokens = output.shape[0]
                            expanded[:num_tokens].copy_(output)
                            expanded_outputs.append(expanded)

                        dummy_encoder_outputs = expanded_outputs

                    # Cache the dummy encoder outputs.
                    self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, _ = self._dummy_run(self.max_num_tokens, is_profile=True)
        if get_pp_group().is_last_rank:
            pass
        self._sync_device()
        del hidden_states
        self.encoder_cache.clear()
        gc.collect()

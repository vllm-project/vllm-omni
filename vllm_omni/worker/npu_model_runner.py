"""NPU Model Runner base class for vLLM-omni.

Provides multimodality extensions for NPU model runners, including payload
decoding and multimodal output extraction.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

import math
import numpy as np
import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import BatchDescriptor, DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.sampling_params import SamplingType
from vllm.utils import cdiv, round_up
from vllm.v1.attention.backends.utils import CommonAttentionMetadata, split_attn_metadata
from vllm.v1.outputs import KVConnectorOutput, LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.utils import is_residual_scattered_for_sp, MultiModalBudget
from vllm.v1.worker.gpu_model_runner import IntermediateTensors, PerLayerAttnMetadata
from vllm.v1.worker.ubatch_splitting import ubatch_split
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.npu_input_batch import CachedRequestState
from vllm_ascend.utils import enable_sp, vllm_version_is, lmhead_tp_enable

from vllm_omni.engine import AdditionalInformationPayload, PromptEmbedsPayload

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class OmniNPUModelRunner(NPUModelRunner):
    """Base class for NPU model runners with multimodality support.

    Extends NPUModelRunner with:
    - Payload decoding (prompt_embeds, additional_information)
    - Multimodal output extraction
    - Additional information update merging
    - Multimodal initialization (mm_budget, mm_registry, supports_mm_inputs)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.is_multimodal_raw_input_only_model = (
            self.model_config.is_multimodal_raw_input_only_model)
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config)
        
        self._mm_budget = None

    @property
    def mm_budget(self):
        if self._mm_budget is None:
            self._mm_budget = MultiModalBudget(
                self.model_config,
                self.scheduler_config,
                self.mm_registry,
            ) if self.supports_mm_inputs else None
        return self._mm_budget

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input NPU tensors for the model.

        The SamplingMetadata is updated and copied to the NPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            # Handle backward compatibility for mm_features/mm_kwargs
            backward_kwargs = {}
            if vllm_version_is("0.11.0"):
                backward_kwargs["mm_features"] = getattr(new_req_data, "mm_features", None)
            else:
                backward_kwargs["mm_kwargs"] = getattr(new_req_data, "mm_kwargs", None)
                backward_kwargs["mm_hashes"] = getattr(new_req_data, "mm_hashes", None)
                backward_kwargs["mm_positions"] = getattr(new_req_data, "mm_positions", None)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
                **backward_kwargs,
            )
            self.requests[req_id] = req_state

            # If prompt embeddings are provided, decode and attach to request state
            try:
                if getattr(new_req_data, "prompt_embeds", None) is not None:
                    payload = new_req_data.prompt_embeds
                    if isinstance(payload, PromptEmbedsPayload):
                        dtype = getattr(np, payload.dtype)
                        arr = np.frombuffer(payload.data, dtype=dtype)
                        arr = arr.reshape(payload.shape)
                        pe_cpu = torch.from_numpy(arr)
                    elif isinstance(payload, torch.Tensor):
                        pe_cpu = payload.detach().to("cpu").contiguous()
                    else:
                        pe_cpu = None
                    # Store temporarily on CPU; later moved to device in builder
                    if pe_cpu is not None:
                        setattr(self.requests[req_id], "prompt_embeds_cpu", pe_cpu)
                        # Also replace payload with Tensor for user visibility in
                        # scheduler_output
                        try:
                            new_req_data.prompt_embeds = pe_cpu  # type: ignore[assignment]
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"Error decoding prompt embeds: {e}")
            # Decode additional_information payloads (dictionary)
            try:
                if getattr(new_req_data, "additional_information", None) is not None:
                    payload_info = new_req_data.additional_information
                    info_dict = {}
                    if isinstance(payload_info, dict):
                        info_dict = payload_info
                    elif isinstance(payload_info, AdditionalInformationPayload):
                        for k, entry in payload_info.entries.items():
                            if entry.tensor_data is not None:
                                dt = np.dtype(
                                    getattr(entry, "tensor_dtype", "float32")
                                )
                                arr = np.frombuffer(entry.tensor_data, dtype=dt)
                                arr = arr.reshape(entry.tensor_shape)
                                info_dict[k] = torch.from_numpy(arr)
                            else:
                                info_dict[k] = entry.list_data
                    if info_dict:
                        setattr(
                            self.requests[req_id],
                            "additional_information_cpu",
                            info_dict,
                        )
            except Exception as e:
                logger.error(f"Error decoding additional information: {e}")
                pass

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                if vllm_version_is("0.11.0"):
                    self._init_mrope_positions(self.requests[req_id])
                else:
                    self._init_mrope_positions_0102(self.requests[req_id])

            reqs_to_add.append(self.requests[req_id])

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (
                    num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ()
            )
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index
                ] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    @torch.inference_mode()
    def extract_multimodal_outputs(
        self, hidden_states: Union[torch.Tensor, List[torch.Tensor]]
    ) -> tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor], dict]]:
        """Extract multimodal outputs from hidden states."""
        if (
            hasattr(self.model, "have_multimodal_outputs")
            and self.model.have_multimodal_outputs
        ):
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

    def _sample(
        self, logits: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata]
    ) -> SamplerOutput:
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
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids
            if hasattr(self, "_update_states_after_model_execute"):
                self._update_states_after_model_execute(output_token_ids)

        return sampler_output

    def _bookkeeping_sync(
        self, scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput, logits: Optional[torch.Tensor],
        hidden_states: torch.Tensor, num_scheduled_tokens: int
    ) -> tuple[
        dict[str, int],
        Optional[LogprobsLists],
        list[list[int]],
        dict[str, Optional[LogprobsTensors]],
        list[str],
        dict[str, int],
        list[int],
    ]:
        num_nans_in_logits = {}
        # placeholder for now, TODO: _get_nans_in_logits()

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        invalid_req_indices = []
        
        if not self.use_async_scheduling:
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                valid_sampled_token_ids = sampled_token_ids.tolist()
            else:
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
            discard_sampled_tokens_req_indices = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                          scheduler_output.num_scheduled_tokens[req_id])
                if seq_len < req_state.num_tokens:
                    discard_sampled_tokens_req_indices.append(i)
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
        else:
            valid_sampled_token_ids = []
            discard_sampled_tokens_req_indices = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                          scheduler_output.num_scheduled_tokens[req_id])
                if seq_len < req_state.num_tokens:
                    discard_sampled_tokens_req_indices.append(i)
            invalid_req_indices = discard_sampled_tokens_req_indices
            invalid_req_indices_set = set(invalid_req_indices)
            assert sampled_token_ids.shape[-1] == 1

            self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_sampled_token_ids_invalid_indices = invalid_req_indices_set
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                    if i not in invalid_req_indices_set
                }

        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.model_config.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.model_config.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    def get_dp_padding(
        self, num_tokens: int
    ) -> tuple[int, Optional[torch.Tensor]]:
        """Determines the total number of tokens that each rank will run."""
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank

        if dp_size == 1 or self.vllm_config.model_config.enforce_eager:
            return 0, None

        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                                dp_size,
                                                device="cpu",
                                                dtype=torch.int32)
        return max_tokens_across_dp_cpu - num_tokens, num_tokens_after_padding

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        """Calculate input token count with padding if needed."""
        if (self.use_aclgraph and num_scheduled_tokens
                <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size for ACLGraph.
            # Note: pad_for_cudagraph works for both CUDA graphs and ACLGraph
            return self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)

        # Eager mode.
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if (self.compilation_config.pass_config.enable_sequence_parallelism
                and tp_size > 1):
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _generate_dummy_run_hidden_states(self, with_prefill,
                                          is_torchair_compile, input_ids,
                                          positions, attn_metadata, num_tokens,
                                          intermediate_tensors, inputs_embeds,
                                          model_kwargs=None):
        """Override to support model_kwargs for multimodal/pooling models."""
        if model_kwargs is None:
            model_kwargs = {}
        
        hidden_states = self.model(input_ids=input_ids,
                                   positions=positions,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds,
                                   **model_kwargs)
        
        from vllm_ascend.compilation.acl_graph import update_attn_params, update_mla_attn_params
        
        forward_context = get_forward_context()
        assert forward_context is not None
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and \
            not forward_context.capturing:
            if self.vllm_config.model_config.use_mla:
                # FIXME: Try using `auto_dispatch_capture=True`
                update_mla_attn_params(self.update_stream, forward_context,
                                   positions.shape[0],
                                   self.speculative_config)
            else:
                update_attn_params(self.update_stream, forward_context,
                                   positions.shape[0])

        from vllm_ascend.spec_decode.interface import SpecDcodeType
        if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
            hidden_states, _ = hidden_states
        else:
            hidden_states = hidden_states
        return hidden_states

    def _init_model_kwargs(self, num_tokens: int):
        """Initialize model kwargs."""
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if param.extra_kwargs is not None and \
            (token_types := param.extra_kwargs.get(
                "compressed_token_type_ids")) is not None:
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens_cpu[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device)
        return model_kwargs

    def sync_and_slice_intermediate_tensors(
        self, num_tokens: int, intermediate_tensors: IntermediateTensors,
        sync_self: bool
    ) -> IntermediateTensors:
        """Sync and slice intermediate tensors for pipeline parallelism."""
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True)

        return IntermediateTensors({
            k: v[:num_tokens // tp] if k == "residual" and is_rs else v[:num_tokens]
            for k, v in self.intermediate_tensors.items()
        })

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> dict:
        """Extract multimodal kwargs."""
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        mm_kwargs = list[MultiModalKwargsItem]()
        for req in scheduler_output.scheduled_new_reqs:
            # Handle version compatibility
            if vllm_version_is("0.11.0"):
                mm_features = getattr(req, "mm_features", None)
            else:
                mm_features = getattr(req, "mm_kwargs", None)
            
            if mm_features:
                if isinstance(mm_features, list):
                    for feature in mm_features:
                        if hasattr(feature, "data") and feature.data is not None:
                            mm_kwargs.append(feature.data)
                elif isinstance(mm_features, dict):
                    return mm_features

        if not mm_kwargs:
            return {}

        model = cast(SupportsMultiModal, self.model)
        mm_kwargs_combined: dict = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=model.merge_by_field_config,
        ):
            mm_kwargs_combined.update(mm_kwargs_group)

        return mm_kwargs_combined

    def _extract_encoder_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, torch.Tensor]:
        """Extract encoder inputs for encoder-decoder models."""
        if hasattr(self, "_batch_mm_kwargs_from_scheduler"):
            mm_kwargs, _ = self._batch_mm_kwargs_from_scheduler(scheduler_output)
        else:
            return {}

        if not mm_kwargs:
            return {}

        model = cast(SupportsMultiModal, self.model)
        encoder_features = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=model.merge_by_field_config,
        ):
            encoder_features.update(mm_kwargs_group)

        return encoder_features

    @contextmanager
    def synchronize_input_prep(self):
        """Synchronize input preparation for async scheduling."""
        if getattr(self, 'use_async_scheduling', False):
            if not hasattr(self, "prepare_inputs_event") or self.prepare_inputs_event is None:
                self.prepare_inputs_event = torch.npu.Event()
                self.prepare_inputs_event.record(torch.npu.current_stream())
        
        if not hasattr(self, "prepare_inputs_event") or self.prepare_inputs_event is None:
            yield
            return

        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    @contextmanager
    def maybe_get_kv_connector_output(
        self, scheduler_output: "SchedulerOutput"
    ):
        """KV connector context manager."""
        if not has_kv_transfer_group():
            yield None
            return

        output = KVConnectorOutput()

        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase_V1)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata)

        kv_connector.start_load_kv(get_forward_context())
        try:
            yield output
        finally:
            kv_connector.wait_for_save()
            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids))
            output.kv_connector_stats = kv_connector.get_kv_connector_stats()
            kv_connector.clear_connector_metadata()

    def pad_out_ubatch_slice(self, ubatch_slices, num_total_tokens: int):
        """Pad ubatch slice for DBO (Dynamic Batch Overlap)."""
        from vllm.v1.worker.ubatch_utils import UBatchSlice
        if len(ubatch_slices) < 2:
            return
        padded_second_ubatch_slice = slice(ubatch_slices[1].token_slice.start,
                                           num_total_tokens)
        ubatch_slices[1] = UBatchSlice(padded_second_ubatch_slice,
                                       padded_second_ubatch_slice)

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """Step for EPLB (Expert Parallelism Load Balancing)."""
        if hasattr(self, "dynamic_eplb") and self.dynamic_eplb:
            if hasattr(self, "eplb_updator"):
                self.eplb_updator.forward_end()

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> dict:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None
        
        dummy_decoder_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_model_len,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_data = dummy_decoder_data.multi_modal_data
        
        dummy_mm_item = dummy_mm_data[modality][0]
        dummy_mm_items = [dummy_mm_item] * max_items_per_batch
        
        model = cast(SupportsMultiModal, self.model)
        return next(mm_kwargs_group
                    for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                        dummy_mm_items,
                        device=self.device,
                        pin_memory=getattr(self, "pin_memory", False),
                        merge_by_field_config=model.merge_by_field_config,
                    ))

    def _dummy_mm_kwargs(self, num_seqs: int) -> dict:
        """Return dummy multimodal kwargs for dummy runs."""
        if not self.is_multimodal_raw_input_only_model:
            return {}
        
        mm_budget = self.mm_budget
        assert mm_budget is not None
        
        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    @contextmanager
    def maybe_randomize_inputs(self, input_ids: Optional[torch.Tensor]):
        """Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set."""
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        
        if not randomize_inputs:
            yield
            return
        
        if input_ids is None:
            yield
            return
        
        import functools
        
        @functools.cache
        def rand_input_ids() -> torch.Tensor:
            return torch.randint_like(
                self.input_ids,
                low=0,
                high=self.model_config.get_vocab_size(),
                dtype=input_ids.dtype, 
            )
        
        logger.debug_once("Randomizing dummy data for DP Rank")
        input_ids.copy_(
            rand_input_ids()[:input_ids.size(0)],
            non_blocking=True
        )
        yield
        input_ids.fill_(0)

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a dummy forward pass to warm up/profile run or capture the ACL graph for the model.
        
        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: Used to control the behavior.
                - if not set will determine the aclgraph mode based on using
                    the self.aclgraph_dispatcher.
                - CUDAGraphMode.NONE: No aclgraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise aclgraph.
                - CUDAGraphMode.FULL: Full aclgraph, attention metadata is needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            allow_microbatching: If True, allow ubatch splitting if DBO is enabled.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run.
        """
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL,
        }

        if hasattr(self, "use_aclgraph") and self.use_aclgraph and enable_sp(self.vllm_config):
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_tokens = math.ceil(num_tokens / tp_size) * tp_size

        with_prefill = create_mixed_batch or (not uniform_decode and num_tokens > 1)
        
        if hasattr(self, "is_kv_producer") and self.is_kv_producer and \
           hasattr(self, "is_kv_consumer") and not self.is_kv_consumer:
            with_prefill = True

        if hasattr(self, "_sync_metadata_across_dp"):
            (num_tokens, num_tokens_across_dp, with_prefill, _) = \
                self._sync_metadata_across_dp(num_tokens, with_prefill, False)
        else:
            num_tokens_across_dp = None

        if hasattr(self, "_select_moe_comm_method"):
            moe_comm_type = self._select_moe_comm_method(num_tokens, with_prefill)
        else:
            moe_comm_type = None

        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        
        if create_mixed_batch:
            assert not uniform_decode
            num_decode_tokens = num_tokens // 2
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = cdiv(num_tokens, max_query_len)
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            if with_prefill:
                num_reqs = num_tokens
            else:
                decode_token_per_req = getattr(self, "decode_token_per_req", 1)
                num_reqs = (num_tokens + decode_token_per_req - 1) // decode_token_per_req
            num_reqs = min(num_reqs, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)

        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())

        ubatch_slices = None
        num_tokens_after_padding = None
        
        if self.parallel_config.enable_dbo and allow_microbatching:
            ubatch_slices, ubatch_num_tokens_after_padding = ubatch_split(
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                total_num_scheduled_tokens,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
            )
            if ubatch_num_tokens_after_padding is not None:
                num_tokens_after_padding = ubatch_num_tokens_after_padding * 2

        if num_tokens_after_padding is None:
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
            num_tokens_after_padding = num_tokens + num_pad
        else:
            if isinstance(num_tokens_after_padding, torch.Tensor):
                num_tokens_after_padding = int(num_tokens_after_padding[0].item())
            elif isinstance(num_tokens_after_padding, (list, np.ndarray)):
                num_tokens_after_padding = int(num_tokens_after_padding[0])

        attn_metadata: Optional[PerLayerAttnMetadata] = None
        
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            attn_metadata = {}
            if ubatch_slices is not None:
                attn_metadata = [dict() for _ in range(len(ubatch_slices))]

            if create_mixed_batch:
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len
            
            self.seq_lens_np[:num_reqs] = seq_lens
            self.seq_lens_np[num_reqs:] = 0
            if isinstance(seq_lens, list):
                self.seq_lens_cpu[:num_reqs] = torch.tensor(seq_lens, dtype=torch.int32)
            else:
                self.seq_lens_cpu[:num_reqs] = seq_lens
            self.seq_lens_cpu[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(
                self.seq_lens_cpu[:num_reqs], non_blocking=True)
                self.seq_lens[num_reqs:].fill_(0)

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc_np[0] = 0
            self.query_start_loc_np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc_cpu[0] = 0
            self.query_start_loc_cpu[1 : num_reqs + 1] = torch.from_numpy(cum_num_tokens)
            self.query_start_loc[:num_reqs + 1].copy_(
                self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups
            ):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc[: num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc_cpu[: num_reqs + 1],
                    seq_lens=self.seq_lens[:num_reqs],
                    seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[
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
                    ].slot_mapping[:num_tokens],
                    causal=True,
                )
                for attn_group in self.attn_groups[kv_cache_group_id]:
                    if ubatch_slices is not None:
                        common_attn_metadata_list = split_attn_metadata(
                            ubatch_slices, common_attn_metadata
                        )
                        for ubid, common_attn_metadata in enumerate(
                            common_attn_metadata_list
                        ):
                            assert common_attn_metadata.max_query_len == 1
                            attn_metadata_i = attn_group.get_metadata_builder(
                                ubatch_id=ubid
                            ).build_for_cudagraph_capture(common_attn_metadata)
                            for layer_name in attn_group.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][layer_name] = attn_metadata_i
                    else:
                        assert type(attn_metadata) is dict
                        attn_metadata_i = attn_group.get_metadata_builder().build_for_cudagraph_capture(
                            common_attn_metadata
                        )
                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(
            self.lora_config, num_scheduled_tokens, remove_lora
        ):
            model_kwargs = self._init_model_kwargs(num_tokens)
            
            # Prepare inputs (NPU uses direct tensor access, not .gpu buffers)
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
                model_kwargs = self._init_model_kwargs(num_tokens)
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            # Prepare intermediate_tensors if PP
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

            # Dispatch graph mode (check for aclgraph_dispatcher)
            if hasattr(self, "aclgraph_dispatcher") and not is_profile:
                _cg_mode, batch_descriptor = self.aclgraph_dispatcher.dispatch(
                    BatchDescriptor(
                        num_tokens=num_tokens_after_padding,
                        uniform_decode=uniform_decode,
                    )
                )
            else:
                _cg_mode, batch_descriptor = (CUDAGraphMode.NONE, None)
            
            # Map GPU parameter name to NPU internal name for clarity
            # cudagraph_runtime_mode (GPU signature) → aclgraph_runtime_mode (NPU internal)
            if cudagraph_runtime_mode is not None:
                assert (
                    cudagraph_runtime_mode == CUDAGraphMode.NONE
                    or cudagraph_runtime_mode == _cg_mode
                ), (
                    f"ACL graph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}."
                )
                aclgraph_runtime_mode = cudagraph_runtime_mode
            else:
                aclgraph_runtime_mode = _cg_mode

            # Adjust for ubatch if needed
            if ubatch_slices is not None:
                num_tokens_after_padding = ubatch_slices[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_after_padding

            original_in_profile_run = self.in_profile_run
            self.in_profile_run = is_profile

            if not self.in_profile_run and hasattr(self, "dynamic_eplb") and self.dynamic_eplb:
                if hasattr(self, "eplb_updator"):
                    self.eplb_updator.forward_before()

            need_dummy_logits = (not self.in_profile_run and lmhead_tp_enable())
            dummy_indices = None
            dummy_compute_logits = None
            
            if need_dummy_logits:
                max_num_reqs_across_dp = num_tokens if not with_prefill else max_num_reqs
                dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32, device=self.device)
                
                def dummy_compute_logits(hidden_states):
                    return self.model.compute_logits(hidden_states[dummy_indices])

            try:
                with self.maybe_randomize_inputs(input_ids), set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_after_padding,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=with_prefill,
                    in_profile_run=self.in_profile_run,
                    reserved_mc2_mask=getattr(self, "reserved_mc2_mask", None),
                    moe_comm_type=moe_comm_type,
                    num_actual_tokens=0,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    prefetch_stream=getattr(self, "prefetch_stream", None),
                    model_instance=self.model,
                    weight_prefetch_method=getattr(self, "weight_prefetch_method", None),
                ):
                    hidden_states = self._generate_dummy_run_hidden_states(
                        with_prefill=with_prefill,
                        is_torchair_compile=False,
                        input_ids=input_ids,
                        positions=positions,
                        attn_metadata=attn_metadata,
                        num_tokens=num_tokens,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        model_kwargs=model_kwargs,
                    )
                    
                    if need_dummy_logits:
                        dummy_compute_logits(hidden_states)

                if self.drafter:
                    self.drafter.dummy_run(
                        num_tokens=num_tokens,
                        with_prefill=with_prefill,
                        skip_attn=True,
                        num_reqs=num_reqs,
                        num_tokens_across_dp=num_tokens_across_dp,
                        aclgraph_runtime_mode=aclgraph_runtime_mode,
                        batch_descriptor=batch_descriptor,
                    )
                    if need_dummy_logits:
                        self.drafter.model.compute_logits(hidden_states[dummy_indices])
                
                if self.in_profile_run and hasattr(self, "dynamic_eplb") and self.dynamic_eplb:
                    if hasattr(self, "model"):
                        self.model.clear_all_moe_loads()
                if not self.in_profile_run and hasattr(self, "dynamic_eplb") and self.dynamic_eplb:
                    if hasattr(self, "eplb_updator"):
                        self.eplb_updator.take_update_info_from_eplb_process()
            finally:
                self.in_profile_run = original_in_profile_run

        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        hidden_states, _ = self.extract_multimodal_outputs(hidden_states)
        return hidden_states, hidden_states[logit_indices]

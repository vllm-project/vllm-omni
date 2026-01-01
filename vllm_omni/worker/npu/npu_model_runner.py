# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.interfaces import supports_mrope
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.utils import enable_sp, lmhead_tp_enable
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from vllm_omni.model_executor.models.output_templates import OmniOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class OmniNPUModelRunner(NPUModelRunner):
    """
    Base class for NPU model runners with multimodality support.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._omni_per_req_additional_information: dict[str, dict] | None = None
        self._omni_num_scheduled_tokens_np: np.ndarray | None = None
        self._omni_last_model_output: object | None = None

    def _init_mrope_positions(self, req_state: CachedRequestState):
        image_grid_thw = []
        video_grid_thw = []
        second_per_grid_ts = []
        audio_feature_lengths = []
        use_audio_in_video = False
        for mm_feature in req_state.mm_features:
            mm_item = mm_feature.data
            if mm_item is None:
                continue
            mm_input = mm_item.get_data()
            if (t := mm_input.get("image_grid_thw")) is not None:
                image_grid_thw.append(t.tolist())
            if (t := mm_input.get("video_grid_thw")) is not None:
                video_grid_thw.append(t.tolist())
            if (t := mm_input.get("second_per_grid_ts")) is not None:
                second_per_grid_ts.append(t)
            if (t := mm_input.get("audio_feature_lengths")) is not None:
                audio_feature_lengths.append(t)
            # Check for use_audio_in_video
            use_audio_in_video_value = mm_input.get("use_audio_in_video")
            if use_audio_in_video_value is not None:
                use_audio_in_video = bool(use_audio_in_video_value.item())

        if supports_mrope(self.model):
            req_state.mrope_positions, req_state.mrope_position_delta = self.model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                mm_features=req_state.mm_features,
                hf_config=self.model_config.hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )
        else:
            req_state.mrope_positions, req_state.mrope_position_delta = MRotaryEmbedding.get_input_positions_tensor(
                req_state.prompt_token_ids,
                hf_config=self.model_config.hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
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

            if sampling_params and sampling_params.sampling_type == SamplingType.RANDOM_SEED:
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

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            #  -------------------------------------- Omni-new -------------------------------------------------
            # If prompt embeddings are provided, decode and attach to inter_data
            try:
                if getattr(new_req_data, "prompt_embeds", None) is not None:
                    payload = new_req_data.prompt_embeds
                    dtype = getattr(np, payload.dtype)
                    arr = np.frombuffer(payload.data, dtype=dtype)
                    arr = arr.reshape(payload.shape)
                    pe_cpu = torch.from_numpy(arr)
                    # Store temporarily on CPU; later moved to device in builder
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
                    else:
                        from vllm_omni.engine import AdditionalInformationPayload

                        if isinstance(payload_info, AdditionalInformationPayload):
                            for k, entry in payload_info.entries.items():
                                if entry.tensor_data is not None:
                                    dt = np.dtype(getattr(entry, "tensor_dtype", "float32"))
                                    arr = np.frombuffer(entry.tensor_data, dtype=dt)
                                    arr = arr.reshape(entry.tensor_shape)
                                    info_dict[k] = torch.from_numpy(arr.copy())
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
            #  ------------------------------------------------------------------------------------------------
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

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
                num_new_tokens = num_computed_tokens + len(new_token_ids) - req_state.num_tokens
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
                self.input_batch.token_ids_cpu[req_index, start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
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
        self, hidden_states: torch.Tensor | list[torch.Tensor] | OmniOutput
    ) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor] | dict]:
        """Extract multimodal outputs from hidden states."""
        if hasattr(self.model, "have_multimodal_outputs") and self.model.have_multimodal_outputs:
            text_hidden_states = hidden_states.text_hidden_states
            multimodal_outputs = hidden_states.multimodal_outputs

        elif isinstance(hidden_states, torch.Tensor):
            text_hidden_states = hidden_states
            multimodal_outputs = {}
        elif isinstance(hidden_states, list):
            text_hidden_states = hidden_states[0]
            multimodal_outputs = {}
        else:
            raise ValueError(f"Invalid hidden states type: {type(hidden_states)}")
        return text_hidden_states, multimodal_outputs

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        aclgraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
    ) -> torch.Tensor:
        # only support eager mode and piecewise graph now
        assert aclgraph_runtime_mode is None or aclgraph_runtime_mode in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL,
        }
        # In multi-DP scenarios, there may be situations where all DP groups are executing dummy runs.
        # If sequence parallelism is enabled, it is essential to ensure that num_tokens is divisible by tp_size.
        if self.use_aclgraph and enable_sp(self.vllm_config):
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_tokens = math.ceil(num_tokens / tp_size) * tp_size

        # Force dummy run on prefill stage when this node is deemed as kv producer.
        if self.is_kv_producer and not self.is_kv_consumer:
            with_prefill = True

        # Padding for DP
        (num_tokens, num_tokens_across_dp, with_prefill) = self._sync_metadata_across_dp(num_tokens, with_prefill)

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
        max_num_reqs = self.max_num_reqs
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            if with_prefill:
                num_reqs = num_tokens
            else:
                num_reqs = (num_tokens + self.decode_token_per_req - 1) // self.decode_token_per_req
            num_reqs = min(num_reqs, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        if not self.in_profile_run and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        has_lora = True if self.lora_config and self.compilation_config.cudagraph_specialize_lora else False
        _ag_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora
        )

        num_tokens_padded = batch_descriptor.num_tokens
        num_reqs_padded = batch_descriptor.num_reqs if batch_descriptor.num_reqs is not None else num_reqs
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        moe_comm_type = self._select_moe_comm_method(num_tokens_padded)

        # filter out the valid batch descriptor
        if aclgraph_runtime_mode is not None:
            # we allow forcing NONE when the dispatcher disagrees to support
            # warm ups for aclgraph capture
            if aclgraph_runtime_mode != CUDAGraphMode.NONE and aclgraph_runtime_mode != _ag_mode:
                raise ValueError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}."
                )
        else:
            aclgraph_runtime_mode = _ag_mode

        # TODO(Mengqing): Set create_mixed_batch to False since it's only used in FI warmup
        # and not supported in ASCEND now. We could remove it in the future.
        attn_metadata = self._build_dummy_attn_metadata(
            False,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            force_attention=force_attention,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens, num_sampled_tokens):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and flashcomm1 are enabled,
                # during dummy_run the estimated space should divide num_tokens by tp_size;
                # otherwise, on non-first PP ranks it would effectively perform an extra all-gather,
                # leading to incorrect memory estimation and potentially causing OOM.
                actual_tokens = num_tokens
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    actual_tokens = num_tokens // tp_size
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=actual_tokens, dtype=self.dtype, device=self.device
                    )
                intermediate_tensors = IntermediateTensors(
                    {k: v[:num_tokens_padded] for k, v in self.intermediate_tensors.items()}
                )

            need_dummy_logits = not self.in_profile_run and lmhead_tp_enable()
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return
                if hasattr(self.drafter, "model") and hasattr(self.drafter.model, "compute_logits"):
                    return self.drafter.model.compute_logits(hidden_states[dummy_indices])

            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                with_prefill=with_prefill,
                in_profile_run=self.in_profile_run,
                # reserved_mc2_mask=self.reserved_mc2_mask,
                moe_comm_type=moe_comm_type,
                num_actual_tokens=0,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                prefetch_stream=self.prefetch_stream,
                model_instance=self.model,
                weight_prefetch_method=self.weight_prefetch_method,
            ):
                hidden_states = self._generate_dummy_run_hidden_states(
                    input_ids, positions, num_tokens_padded, intermediate_tensors, inputs_embeds
                )
                dummy_compute_logits(hidden_states)

            if self.drafter:
                self.drafter.dummy_run(
                    num_tokens=num_tokens_padded,
                    with_prefill=with_prefill,
                    num_reqs=num_reqs_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    skip_attn=not force_attention,
                )
            if self.in_profile_run and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if not self.in_profile_run and self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()
                self.eplb_updator.forward_end()

            hidden_states, _ = self.extract_multimodal_outputs(hidden_states)

            return hidden_states

    def _decode_and_store_request_payloads(self, scheduler_output: "SchedulerOutput") -> None:
        """Decode per-request prompt_embeds and additional_information for newly
        scheduled requests and store them to CPU in the request state.
        This version avoids hard dependency on payload classes by duck-typing."""
        try:
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", [])
            if not new_reqs:
                return
            for nr in new_reqs:
                req_id = getattr(nr, "req_id", None) or getattr(nr, "request_id", None)
                if req_id is None:
                    continue
                # prompt_embeds
                payload_pe = getattr(nr, "prompt_embeds", None)
                pe_cpu = None
                if payload_pe is not None:
                    if isinstance(payload_pe, torch.Tensor):
                        pe_cpu = payload_pe.detach().to("cpu").contiguous()
                    else:
                        # Try duck-typing a payload with data/shape/dtype
                        data = getattr(payload_pe, "data", None)
                        shape = getattr(payload_pe, "shape", None)
                        if data is not None and shape is not None:
                            dt = np.dtype(getattr(payload_pe, "dtype", "float32"))
                            arr = np.frombuffer(data, dtype=dt)
                            arr = arr.reshape(shape)
                            pe_cpu = torch.from_numpy(arr.copy())
                if pe_cpu is not None and req_id in self.requests:
                    setattr(self.requests[req_id], "prompt_embeds_cpu", pe_cpu)
                # additional_information
                payload_info = getattr(nr, "additional_information", None)
                if payload_info is not None:
                    info_dict = {}
                    if isinstance(payload_info, dict):
                        info_dict = payload_info
                    else:
                        # Try duck-typing a payload with entries, each entry may have
                        # tensor_data/tensor_dtype/tensor_shape or list_data
                        entries = getattr(payload_info, "entries", None)
                        if isinstance(entries, dict):
                            for k, entry in entries.items():
                                tensor_data = getattr(entry, "tensor_data", None)
                                if tensor_data is not None:
                                    dt = np.dtype(getattr(entry, "tensor_dtype", "float32"))
                                    arr = np.frombuffer(tensor_data, dtype=dt)
                                    arr = arr.reshape(getattr(entry, "tensor_shape", ()))
                                    info_dict[k] = torch.from_numpy(arr.copy())
                                else:
                                    info_dict[k] = getattr(entry, "list_data", None)
                    if info_dict and req_id in self.requests:
                        setattr(self.requests[req_id], "additional_information_cpu", info_dict)
        except Exception as e:
            logger.error(f"Error decoding prompt_embeds / additional_information: {e}")

    def _gather_runtime_additional_information(self) -> list[dict]:
        """Gather per-request additional_information stored in request state in batch order."""
        per_req_runtime_info = []
        for req_id in self.input_batch.req_ids:
            req_state = self.requests.get(req_id)
            info = getattr(req_state, "additional_information_cpu", None) if req_state is not None else None
            if info and isinstance(info, dict):
                per_req_runtime_info.append(info)
                if "thinker_reply_part_per_request" in info:
                    q = info["thinker_reply_part_per_request"]
                    if hasattr(q, "shape"):
                        logger.debug(f"[OMNI] req={req_id} has thinker_reply_part_per_request queue shape: {q.shape}")
            else:
                per_req_runtime_info.append({})
        return per_req_runtime_info

    def _compute_request_token_spans(self, num_scheduled_tokens_np) -> list[tuple[int, int]]:
        """Compute (start, end) token spans for each request within the flattened step sequence."""
        req_token_spans: list[tuple[int, int]] = []
        for req_index in range(len(self.input_batch.req_ids)):
            start_offset = int(self.query_start_loc.cpu[req_index])
            sched_tokens = int(num_scheduled_tokens_np[req_index])
            req_token_spans.append((start_offset, start_offset + sched_tokens))
        return req_token_spans

    def _build_model_kwargs_extra(self) -> dict:
        """Build extra keyword arguments passed to the model for this step, including:
        - runtime_additional_information: per-request additional information stored in request state
        """
        model_kwargs_extra: dict[str, object] = {}
        try:
            model_kwargs_extra["runtime_additional_information"] = self._gather_runtime_additional_information()
        except Exception as e:
            logger.error(f"[OMNI DEBUG] Error building model_kwargs_extra: {e}")
            import traceback

            traceback.print_exc()
        return model_kwargs_extra

    def _collect_additional_information_for_prefill(
        self,
        num_scheduled_tokens_np: np.ndarray,
    ) -> dict[str, dict]:
        """Overlay per-request prompt_embeds for the prefill portion and collect
        additional_information slices for this step. Returns a map req_id -> dict."""
        for req_index, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            pe_cpu = getattr(req_state, "prompt_embeds_cpu", None)
            num_computed_tokens = int(self.input_batch.num_computed_tokens_cpu[req_index])
            prompt_len = len(req_state.prompt_token_ids)
            prompt_remaining = max(0, prompt_len - num_computed_tokens)
            sched_tokens = int(num_scheduled_tokens_np[req_index])
            overlay_len = min(sched_tokens, prompt_remaining)
            if overlay_len <= 0:
                continue
            if overlay_len > 0 and pe_cpu is not None:
                src = pe_cpu[num_computed_tokens : num_computed_tokens + overlay_len].to(
                    dtype=self.dtype, device=self.device, non_blocking=True
                )
                start_offset = int(self.query_start_loc.cpu[req_index])
                self.inputs_embeds[start_offset : start_offset + overlay_len].copy_(src)

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        """Inject omni-specific kwargs into forward and cache model output"""
        model_kwargs_extra = self._build_model_kwargs_extra()

        runtime_info = model_kwargs_extra.get("runtime_additional_information", [])
        if runtime_info:
            for i, info in enumerate(runtime_info):
                if info:
                    logger.debug(f"[OMNI] req[{i}] runtime_additional_information keys: {list(info.keys())}")

        model_output = super()._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
            **model_kwargs_extra,
        )
        # Cache model output so later sample_tokens can consume multimodal results.
        self._omni_last_model_output = model_output
        return model_output

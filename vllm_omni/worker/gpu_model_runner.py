from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.interfaces import supports_mm_encoder_only, supports_mrope
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, IntermediateTensors, PerLayerAttnMetadata
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices

from vllm_omni.model_executor.models.output_templates import OmniOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    xgr_torch_compile = LazyLoader(
        "xgr_torch_compile",
        globals(),
        "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile",
    )

logger = init_logger(__name__)


class OmniGPUModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._omni_per_req_additional_information: dict[str, dict] | None = None
        self._omni_num_scheduled_tokens_np: np.ndarray | None = None
        self._omni_last_model_output: object | None = None

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        # TODO move this model specific logic to a separate class
        if hasattr(self.model, "talker_mtp") and self.model.talker is not None:
            self.talker_mtp = self.model.talker_mtp
            cudagraph_mode = self.compilation_config.cudagraph_mode
            assert cudagraph_mode is not None
            if cudagraph_mode.has_full_cudagraphs():
                self.talker_mtp = CUDAGraphWrapper(
                    self.model.talker_mtp, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
                )
            hidden_size = self.model_config.hf_config.talker_config.text_config.hidden_size
            self.talker_mtp_input_ids = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
            self.talker_mtp_inputs_embeds = self._make_buffer(
                self.max_num_reqs, hidden_size, dtype=self.dtype, numpy=False
            )
            self.last_talker_hidden = self._make_buffer(self.max_num_reqs, hidden_size, dtype=self.dtype, numpy=False)
            self.text_step = self._make_buffer(self.max_num_reqs, hidden_size, dtype=self.dtype, numpy=False)

    def _init_mrope_positions(self, req_state: CachedRequestState):
        """Initialize M-RoPE positions for multimodal inputs.

        Extracts multimodal feature metadata (image grids, video grids,
        audio features) and computes M-RoPE positions for proper positional
        encoding of multimodal tokens.

        Args:
            req_state: Cached request state containing multimodal features

        Raises:
            AssertionError: If the model does not support M-RoPE
        """
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

        if supports_mrope(self.get_model()):
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
            self.num_prompt_logprobs.pop(req_id, None)
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
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
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

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(self.requests[req_id])

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # fist step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt length),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step does't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

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
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = self.input_batch.num_prompt_tokens[req_index] + num_output_tokens
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

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

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    @torch.inference_mode()
    def extract_multimodal_outputs(self, hidden_states: torch.Tensor | list[torch.Tensor] | OmniOutput) -> dict:
        if (
            hasattr(self.model, "have_multimodal_outputs")
            and self.model.have_multimodal_outputs
            and isinstance(hidden_states, OmniOutput)
        ):
            text_hidden_states = hidden_states.text_hidden_states
            multimodal_outputs = hidden_states.multimodal_outputs

        elif isinstance(hidden_states, torch.Tensor):
            text_hidden_states = hidden_states
            multimodal_outputs = {}
        elif isinstance(hidden_states, list) or isinstance(hidden_states, tuple):
            text_hidden_states = hidden_states[0]
            multimodal_outputs = {}
        else:
            raise ValueError(f"Invalid hidden states type: {type(hidden_states)}")
        return text_hidden_states, multimodal_outputs

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
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
            activate_lora: If False, dummy_run is performed without LoRAs.
        """
        if supports_mm_encoder_only(self.model):
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()

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
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=activate_lora,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            activate_lora,
            remove_lora,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device,
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(num_tokens_padded, None, False)

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                ),
            ):
                if getattr(self.model, "talker", None) is not None and hasattr(self.model, "talker_mtp"):
                    outputs = self.talker_mtp(
                        self.talker_mtp_input_ids.gpu[:num_tokens_padded],
                        self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded],
                        self.last_talker_hidden.gpu[:num_tokens_padded],
                        self.text_step.gpu[:num_tokens_padded],
                    )
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
            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (is_graph_capturing and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE)
                    or (not is_graph_capturing and cudagraph_runtime_mode != CUDAGraphMode.NONE)
                ) and not self.speculative_config.enforce_eager

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and activate_lora:
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                )

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

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
        logit_indices_device = torch.from_numpy(logit_indices).to(self.device, non_blocking=True)
        return hidden_states, hidden_states[logit_indices_device]

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

    def _process_additional_information_updates(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: object,
        num_scheduled_tokens_np: np.ndarray,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Process model-provided per-request additional_information updates and merge into request state."""
        try:
            # execute the custom postprocess function
            # TODO(Peiqi): do we have a more elegant way to do this?
            if hasattr(self.model, "has_postprocess") and self.model.has_postprocess:
                for req_index, req_id in enumerate(self.input_batch.req_ids):
                    if self.model_config.async_chunk:
                        req_infos = self._get_additional_information(scheduler_output, req_id)
                    else:
                        req_state = self.requests.get(req_id)
                        req_infos = (
                            getattr(req_state, "additional_information_cpu", None) if req_state is not None else None
                        )
                    start_offset = int(self.query_start_loc.cpu[req_index])
                    sched_tokens = int(num_scheduled_tokens_np[req_index])
                    s, e = start_offset, start_offset + sched_tokens
                    # only consider to store data into update dict.
                    hidden_states_slice = hidden_states[s:e]
                    update_dict = self.model.postprocess(hidden_states_slice, **req_infos)
                    self._merge_additional_information_update(req_id, update_dict)
        except Exception as e:
            logger.error(
                f"Error merging for requests:{self.input_batch.req_ids} "
                f"additional information update: {e}, with the multimodal_outputs "
                f"as {multimodal_outputs}"
            )
            import traceback

            traceback.print_exc()

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

    def _get_additional_information(self, scheduler_output: "SchedulerOutput", req_id: str) -> dict:
        req_infos = None
        req_state = self.requests.get(req_id)
        additional_information_cpu = getattr(req_state, "additional_information_cpu", None)
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id == req_id:
                payload_info = getattr(new_req, "additional_information", None)
                if payload_info is not None:
                    return payload_info

        if hasattr(scheduler_output.scheduled_cached_reqs, "additional_information"):
            cached_infos = getattr(scheduler_output.scheduled_cached_reqs, "additional_information", {})
            if isinstance(cached_infos, dict) and req_id in cached_infos:
                req_infos = cached_infos[req_id]
                if not isinstance(req_infos, dict):
                    req_infos = None

        if req_infos is None or req_infos.get("last_talker_hidden", None) is None:
            if req_infos is None:
                additional_information_cpu.pop("thinker_embeddings", None)
                req_infos = additional_information_cpu
            else:
                req_infos["last_talker_hidden"] = additional_information_cpu.get("last_talker_hidden", None)
                req_infos["num_processed_thinker_tokens"] = additional_information_cpu.get(
                    "num_processed_thinker_tokens", 0
                )
            if not isinstance(req_infos, dict):
                req_infos = None

        if req_infos is None:
            logger.warning(f"No additional_information found for req_id: {req_id}")

        return req_infos

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        """Align with v0.14.0 preprocess and omni's additional information handling."""
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        ec_connector_output = None

        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
            # Run the multimodal encoder if any.
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.embed_input_ids(
                self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and is_first_rank:
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
            token_ids_idx = self.is_token_ids.gpu[:num_scheduled_tokens].nonzero(as_tuple=False).squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
            input_ids = self.input_ids.gpu[:num_input_tokens]
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs()

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Run the encoder, just like we do with other multimodal inputs.
            # For an encoder-decoder model, our processing here is a bit
            # simpler, because the outputs are just passed to the decoder.
            # We are not doing any prompt replacement. We also will only
            # ever have a single encoder input.
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        req_ids = self.input_batch.req_ids
        num_scheduled_tokens_np = np.array(
            [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
            dtype=np.int32,
        )
        self._omni_num_scheduled_tokens_np = num_scheduled_tokens_np

        # Note: only prefill need collect additional_information for now.
        # Decode don't need per_req_additional_information anymore.
        if inputs_embeds is not None:
            # Prefill: overlay prompt_embeds and collect additional_information
            self._collect_additional_information_for_prefill(num_scheduled_tokens_np)

        if hasattr(self.model, "has_preprocess") and self.model.has_preprocess:
            # Overlay custom prompt_embeds per request for the prompt portion;
            # collect additional_information (tensor/list) for prefill portion only
            decode_req_ids = []
            for req_index, req_id in enumerate(self.input_batch.req_ids):
                # Try to get additional_information from multiple sources
                if self.vllm_config.model_config.async_chunk:
                    req_infos = self._get_additional_information(scheduler_output, req_id)
                else:
                    req_state = self.requests.get(req_id)
                    req_infos = (
                        getattr(req_state, "additional_information_cpu", None) if req_state is not None else None
                    )
                start_offset = int(self.query_start_loc.cpu[req_index])
                sched_tokens = int(num_scheduled_tokens_np[req_index])
                s, e = start_offset, start_offset + sched_tokens
                span_len = int(e) - int(s)

                # call the custom process function
                req_input_ids, req_embeds, update_dict = self.model.preprocess(
                    input_ids=input_ids[s:e], input_embeds=inputs_embeds[s:e], **req_infos
                )
                if hasattr(self.model, "talker_mtp") and span_len == 1:
                    last_talker_hidden, text_step = update_dict.pop("mtp_inputs")
                    decode_slice = slice(len(decode_req_ids), len(decode_req_ids) + 1)
                    self.talker_mtp_input_ids.gpu[decode_slice].copy_(req_input_ids)
                    self.talker_mtp_inputs_embeds.gpu[decode_slice].copy_(req_embeds)
                    self.last_talker_hidden.gpu[decode_slice].copy_(last_talker_hidden)
                    self.text_step.gpu[decode_slice].copy_(text_step)
                    decode_req_ids.append(req_id)

                # TODO(Peiqi): the merge stage could move out from the critical path
                self._merge_additional_information_update(req_id, update_dict)

                # update the inputs_embeds and input_ids
                seg_len = min(span_len, req_embeds.shape[0])
                inputs_embeds[s : s + seg_len] = req_embeds[:seg_len]
                if isinstance(req_input_ids, torch.Tensor) and req_input_ids.numel() == seg_len:
                    input_ids[s : s + seg_len] = req_input_ids

            # run talker mtp decode
            if hasattr(self.model, "talker_mtp"):
                self._talker_mtp_forward(decode_req_ids, inputs_embeds)

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    def _talker_mtp_forward(self, decode_req_ids: list[str], inputs_embeds: torch.Tensor) -> None:
        decode_batch_size = len(decode_req_ids)
        if decode_batch_size == 0:
            return
        _cudagraph_mode, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
            num_tokens=decode_batch_size,
            num_reqs=decode_batch_size,
            num_scheduled_tokens_np=np.ones(decode_batch_size, dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )
        req_input_ids = self.talker_mtp_input_ids.gpu[:decode_batch_size]
        req_embeds = self.talker_mtp_inputs_embeds.gpu[:decode_batch_size]
        last_talker_hidden = self.last_talker_hidden.gpu[:decode_batch_size]
        text_step = self.text_step.gpu[:decode_batch_size]
        with set_forward_context(
            None, self.vllm_config, cudagraph_runtime_mode=_cudagraph_mode, batch_descriptor=batch_desc
        ):
            req_embeds, code_predictor_codes = self.talker_mtp(req_input_ids, req_embeds, last_talker_hidden, text_step)
        # update the inputs_embeds and code_predictor_codes
        code_predictor_codes_cpu = code_predictor_codes.detach().to("cpu").contiguous()
        for idx, req_id in enumerate(decode_req_ids):
            req_index = self.input_batch.req_ids.index(req_id)
            start_offset = int(self.query_start_loc.cpu[req_index])
            inputs_embeds[start_offset : start_offset + 1] = req_embeds[idx : idx + 1]
            update_dict = {"code_predictor_codes": code_predictor_codes_cpu[idx : idx + 1]}
            self._merge_additional_information_update(req_id, update_dict)

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
        if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
            model_output = self.model.make_omni_output(model_output, **model_kwargs_extra)
        # Cache model output so later sample_tokens can consume multimodal results.
        self._omni_last_model_output = model_output
        return model_output

    def _merge_additional_information_update(self, req_id: str, upd: dict) -> None:
        req_state = self.requests.get(req_id)
        if req_state is None:
            return
        existing = getattr(req_state, "additional_information_cpu", {})
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        for k, v in upd.items():
            if isinstance(v, torch.Tensor):
                merged[k] = v.detach().to("cpu").contiguous()
            elif isinstance(v, list):
                merged[k] = [
                    (item.detach().to("cpu").contiguous() if isinstance(item, torch.Tensor) else item) for item in v
                ]
            else:
                merged[k] = v
        setattr(req_state, "additional_information_cpu", merged)

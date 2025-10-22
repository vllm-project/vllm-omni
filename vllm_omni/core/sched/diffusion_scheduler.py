from vllm.v1.core.sched.scheduler import SchedulerOutput, EngineCoreOutputs, Request, RequestStatus, SpecDecodingStats
from collections import defaultdict
from typing import Optional
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.engine import EngineCoreEventType
from vllm.distributed.kv_events import KVEventBatch
import time
from vllm_omni.core.sched.scheduler import OmniScheduler
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm.v1.engine import EngineCoreOutput
from vllm_omni.core.sched.output import OmniNewRequestData


class DiffusionScheduler(OmniScheduler):
    def schedule(self) -> SchedulerOutput:
        """Diffusion fast path:
        - Feed all input tokens of the request at once (if 0, allocate 1 placeholder token).
        - If the token budget cannot be satisfied at once, fall back to the default vLLM scheduling.
        """

        # 选出零 prompt 且使用 pooling（扩散结果经 pooler_output 回传）的请求
        token_budget = self.max_num_scheduled_tokens
        capacity = self.max_num_running_reqs - len(self.running)
        scheduled_timestamp = time.monotonic()

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []

        req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        structured_output_request_ids: dict[str, int] = {}

        # 临时队列：保持等待队列顺序，不破坏非扩散请求
        skipped_waiting_requests = create_request_queue(self.policy)

        # 快速通道挑选并调度（所有请求都视为扩散请求，不依赖 pooling_params）
        while self.waiting and token_budget > 0 and capacity > 0:
            request = self.waiting.peek_request()
            # 统一按扩散处理。若未来需要条件开关，可接入配置或请求标记。
            is_diffusion = True
            if not is_diffusion:
                # 暂存到跳过队列，稍后归还到等待队列头部
                self.waiting.pop_request()
                skipped_waiting_requests.prepend_request(request)
                continue

            # 一次性为该请求分配全部输入 token（若为 0，则分配 1 个占位 token）
            required_tokens = max(getattr(request, "num_prompt_tokens", 0), 1)
            if required_tokens > token_budget:
                # 无足够预算一次性完成该请求的输入处理，停止快速通道尝试
                break
            num_new_tokens = required_tokens
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # 无法分配（显存紧张等），停止快速通道尝试，回退到默认调度
                # 将当前 request 放回等待队列头
                # 注意：此处不改变原队列顺序
                break

            # 正式调度该请求
            request = self.waiting.pop_request()
            self.running.append(request)
            request.status = RequestStatus.RUNNING
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED,
                                     scheduled_timestamp)

            req_to_new_block_ids[request.request_id] = new_blocks.get_block_ids()
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            capacity -= 1
            scheduled_new_reqs.append(request)

        # 归还被跳过的等待请求
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # 若快速通道未调度任何请求，则回退到原始调度逻辑
        if not num_scheduled_tokens:
            return super().schedule()

        # 计算公共前缀块（与 v1 对齐）
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduled_spec_decode_tokens,
        )

        # 组装 SchedulerOutput
        new_reqs_data = [
            OmniNewRequestData.from_request(req, req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_block_ids,
        )

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # KVTransfer：封装元信息
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # 发布 KV 事件（与 v1 对齐）
        events = self.kv_cache_manager.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # 更新内部状态（推进 num_computed_tokens，释放 encoder 输入等）
        self._update_after_schedule(scheduler_output)
        return scheduler_output
    """
    Scheduler for the diffusion model.
    This scheduler is modified to stop the request immediately for the diffusion model.
    This is because the diffusion model can generate the final image/audio in one step.
    Note: This is just a minimal modification to the original scheduler, and there should be some further efforts to optimize the scheduler. 
    The original scheduler is still used for the AR model.
    """
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: OmniModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update the scheduler state based on the model runner output.

        This method is modified to stop the request immediately for the diffusion model.
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens, where is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]

            # 扩散请求：单步完成，直接标记完成并释放资源
            request.status = RequestStatus.FINISHED_STOPPED
            # 可选：标注停止原因，便于前端区分（不影响协议）
            request.stop_reason = request.stop_reason or "diffusion_done"
            kv_transfer_params = self._free_request(request)
            if status_before_stop == RequestStatus.RUNNING:
                stopped_running_reqs.add(request)
            else:
                stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # spec_token_ids comes from the model runner output
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                if self.structured_output_manager.should_advance(request):
                    metadata = request.structured_output_request
                    # Needs to happen after new_token_ids are accepted.
                    request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                        spec_token_ids[req_index])
                else:
                    request.spec_token_ids = spec_token_ids[req_index]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                    ))

            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = [
                req for req in self.running if req not in stopped_running_reqs
            ]
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if engine_core_outputs:
            # Return stats to only one of the front-ends.
            next(iter(engine_core_outputs.values())).scheduler_stats = (
                self.make_stats(spec_decoding_stats))

        return engine_core_outputs
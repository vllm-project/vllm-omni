# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Atomic CFG admission for full-attention AR models with unequal prompts.

Each branch retains its real prompt length and position IDs. Admission reserves
capacity for the complete generation of both branches, so KV pressure cannot
preempt one branch while its partner continues. Reservations account for pages;
the ordinary paged attention manager still allocates pages as tokens arrive.
"""

from collections.abc import Iterable
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.engine.serialization import deserialize_additional_information


class OmniCFGARScheduler(OmniARScheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.scheduler_config.async_scheduling or self.scheduler_config.enable_chunked_prefill:
            raise ValueError("Atomic CFG scheduling requires synchronous scheduling and complete prefills")
        if self.policy != SchedulingPolicy.FCFS or self.cache_config.enable_prefix_caching:
            raise ValueError("Atomic CFG scheduling requires FCFS and disabled prefix caching")
        if self.vllm_config.speculative_config is not None or self.connector is not None:
            raise ValueError("Atomic CFG scheduling does not support speculative decoding or KV connectors")
        if len(self.kv_cache_manager.kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("Atomic CFG reservations require one full-attention KV group")
        self._capacity = self.kv_cache_manager.block_pool.num_gpu_blocks - 1 - self.kv_cache_manager.watermark_blocks
        pair_pages = 2 * -(-self.max_model_len // self.block_size)
        if self._capacity < pair_pages or self.max_num_running_reqs < 2:
            raise ValueError("Atomic CFG requires KV capacity and max_num_seqs for two maximum-length branches")
        if self.max_num_scheduled_tokens < 2 * self.max_model_len:
            raise ValueError("Atomic CFG requires max_num_batched_tokens >= 2 * max_model_len")
        self._pairs: dict[str, dict[str, str]] = {}
        self._request_pairs: dict[str, str] = {}
        self._prefill_min_output_tokens = int(self.vllm_config.additional_config.get("cfg_prefill_delay_tokens", 0))
        if self._prefill_min_output_tokens < 0:
            raise ValueError("cfg_prefill_delay_tokens cannot be negative")

    def _should_defer_waiting_admission(self) -> bool:
        # Let fresh streaming requests accumulate playback audio before
        # another request's text/reference encoding occupies the same device.
        return any(request.num_output_tokens < self._prefill_min_output_tokens for request in self.running)

    def add_request(self, request: Request) -> None:
        payload = getattr(request, "additional_information", None)
        info = deserialize_additional_information(payload) if payload is not None else {}
        group = info.get("cfg_group")
        if group is not None:
            role = group["role"]
            suffix = group["uncond_suffix"]
            if role not in ("cond", "uncond") or not suffix:
                raise ValueError("CFG groups require cond/uncond roles and a companion suffix")
            external_id = request.external_req_id
            if role == "uncond":
                if not external_id.endswith(suffix):
                    raise ValueError("CFG companion request ID does not match its suffix")
                pair_id = external_id[: -len(suffix)]
            else:
                pair_id = external_id
            pair = self._pairs.setdefault(pair_id, {})
            if role in pair:
                raise ValueError("Duplicate CFG branch")
            pair[role] = request.request_id
            self._request_pairs[request.request_id] = pair_id
        super().add_request(request)

    def _reserved_pages(self, request: Request) -> int:
        length = min(self.max_model_len, request.num_prompt_tokens + request.max_tokens)
        return -(-length // self.block_size)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._drop_aborted_queued_requests()
        self._prune_pairs()
        original_waiting = self.waiting
        admitted = create_request_queue(self.policy)
        pending = {request.request_id: request for request in original_waiting}
        pages = self._capacity - sum(self._reserved_pages(request) for request in self.running)
        slots = self.max_num_running_reqs - len(self.running)
        tokens = self.max_num_scheduled_tokens - len(self.running)
        chosen: set[str] = set()
        for request in original_waiting:
            if request.request_id in chosen:
                continue
            pair_id = self._request_pairs.get(request.request_id)
            if pair_id is None:
                group = [request]
            else:
                pair = self._pairs[pair_id]
                if len(pair) != 2 or any(rid not in pending for rid in pair.values()):
                    continue
                group = [pending[pair[role]] for role in ("cond", "uncond")]
                # Offline callers also need both unequal-length prompts to
                # stop together when the longer branch reaches the context.
                limit = min(min(member.max_tokens, self.max_model_len - member.num_prompt_tokens) for member in group)
                for member in group:
                    member.max_tokens = limit
            needed_pages = sum(self._reserved_pages(member) for member in group)
            needed_tokens = sum(member.num_tokens - member.num_computed_tokens for member in group)
            if len(group) > slots or needed_pages > pages or needed_tokens > tokens:
                # Do not let later single-branch requests repeatedly occupy
                # the spare slot and starve an earlier complete CFG pair.
                break
            for member in group:
                admitted.add_request(member)
                chosen.add(member.request_id)
            pages -= needed_pages
            tokens -= needed_tokens
            slots -= len(group)
        original_waiting.remove_requests([pending[rid] for rid in chosen])
        self.waiting = admitted
        try:
            output = super().schedule(throttle_prefills)
        finally:
            original_waiting.prepend_requests(self.waiting)
            self.waiting = original_waiting
        for pair in self._pairs.values():
            if len(pair) != 2:
                continue
            ids = list(pair.values())
            scheduled = [rid in output.num_scheduled_tokens for rid in ids]
            if any(scheduled) != all(scheduled):
                raise RuntimeError("CFG atomic admission invariant violated: only one branch was scheduled")
            if all(scheduled):
                progress = [
                    self.requests[rid].num_computed_tokens - self.requests[rid].num_prompt_tokens for rid in ids
                ]
                if progress[0] != progress[1]:
                    raise RuntimeError("CFG branches reached different generation steps")
        return output

    def finish_requests(self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus) -> list[Request]:
        if request_ids is None:
            ids = None
        else:
            ids = {request_ids} if isinstance(request_ids, str) else set(request_ids)
            for rid in tuple(ids):
                pair_id = self._request_pairs.get(rid)
                if pair_id is not None:
                    ids.update(self._pairs[pair_id].values())
        finished = super().finish_requests(ids, finished_status)
        self._prune_pairs()
        return finished

    def _prune_pairs(self) -> None:
        for pair_id, pair in list(self._pairs.items()):
            if all(rid not in self.requests for rid in pair.values()):
                del self._pairs[pair_id]
                for rid in pair.values():
                    self._request_pairs.pop(rid, None)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-worker acoustic synthesis after releasing native AR slots."""

import copy
import time
from dataclasses import dataclass, field, replace

import torch
from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.request import RequestStatus, StreamingUpdate

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_cfg_ar_scheduler import OmniCFGARScheduler
from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.request import OmniRequest
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_worker import GPUARWorker

from .batched_synthesis import BatchedSynthesisQueue
from .request import AUTO_ABC, SEMANTIC_ARGS
from .yue2 import ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END, MUSIC_START, SAMPLE_RATE, STOP_TOKEN_IDS
from .yue2 import Yue2ForCausalLM as Yue2Backbone


@dataclass
class Yue2SchedulerOutput(OmniSchedulerOutput):
    # Remove runner AR state without cancelling the model's acoustic job.
    acoustic_req_ids: set[str] = field(default_factory=set)
    replaced_req_ids: set[str] = field(default_factory=set)


@dataclass
class Yue2ModelRunnerOutput(OmniModelRunnerOutput):
    acoustic_started: list[str] = field(default_factory=list)
    acoustic_ready: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)


class Yue2ForCausalLM(Yue2Backbone):
    def __init__(self, *, vllm_config, prefix=""):
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("YuE2 requires TP=PP=1")
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError("YuE2 requires synchronous scheduling")
        if vllm_config.model_config.use_v2_model_runner or vllm_config.speculative_config is not None:
            raise ValueError("YuE2 requires model runner v1 without speculative decoding")
        if vllm_config.cache_config.enable_prefix_caching or vllm_config.kv_transfer_config is not None:
            raise ValueError("YuE2 requires APC and external KV connectors disabled")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.acoustic_started: list[str] = []
        config = vllm_config.model_config.hf_config
        sizes = [
            getattr(config, name, default)
            for name, default in (
                ("yue2_nar_batch_size", 1),
                ("yue2_vae_batch_size", 1),
                ("yue2_vae_tile_frames", 1024),
            )
        ]
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes):
            raise ValueError("YuE2 acoustic batch and tile sizes must be positive integers")
        batch_wait_ms = getattr(config, "yue2_nar_batch_wait_ms", 0)
        batch_max_tokens = getattr(config, "yue2_nar_batch_max_tokens", 0)
        if isinstance(batch_max_tokens, bool) or not isinstance(batch_max_tokens, int) or batch_max_tokens < 0:
            raise ValueError("YuE2 NAR batch token budget must be a nonnegative integer")
        if sizes != [1, 1, 1024] or batch_wait_ms != 0 or batch_max_tokens:
            self._synthesis = BatchedSynthesisQueue(
                self, *sizes, batch_wait_ms=batch_wait_ms, batch_max_tokens=batch_max_tokens
            )

    def _hold_allowed(self, state):
        return False

    def _queue_synthesis(self, state):
        if isinstance(self._synthesis, BatchedSynthesisQueue):
            self._synthesis.submit(state)
        else:
            super()._queue_synthesis(state)

    def on_requests_finished(self, finished_req_ids):
        if isinstance(self._synthesis, BatchedSynthesisQueue):
            for request_id in finished_req_ids:
                self._synthesis.cancel_request(request_id)
        super().on_requests_finished(finished_req_ids)

    def _finish_request_safely(self, state, *, hit_end):
        if state.constants.skip_synthesis:
            return super()._finish_request_safely(state, hit_end=hit_end)
        self._queue_synthesis(state)
        self.acoustic_started.append(state.request_id)

    def poll_audio(self):
        if isinstance(self._synthesis, BatchedSynthesisQueue):
            return self._synthesis.poll()
        self._synthesis.pump()
        ready = {}
        # shortcut: scan native model states; index jobs if polling becomes measurable.
        for rid, state in self._states.items():
            job = state.job
            if job is None or not job.done():
                continue
            audio, failed = job.finish()
            state.job = None
            self._decode_t0.pop(rid, None)
            ready[rid] = {
                # Output transport must not own the job's pinned D2H buffer.
                "audio": audio.clone(memory_format=torch.contiguous_format),
                "sr": torch.tensor(SAMPLE_RATE, dtype=torch.int32),
                "meta.truncated": torch.tensor(int(state.truncated)),
                "meta.error": torch.tensor(int(failed)),
            }
        return ready

    def prepare_runner_inputs(self, **kwargs):
        result = super().prepare_runner_inputs(**kwargs)
        self._cfg_step = kwargs.get("sampling_extra_args") or []
        return result

    def sample(self, logits, sampling_metadata):
        if not any((args or {}).get("yue2_cfg_parent") is not None for args in self._cfg_step):
            return super().sample(logits, sampling_metadata)
        rows = {rid: i for i, (rid, _, _) in enumerate(self._step_rows)}
        pairs = []
        original_discard = self._step_discard
        discard = list(original_discard) if original_discard is not None else [False] * len(rows)
        for negative, args in enumerate(self._cfg_step):
            parent = (args or {}).get("yue2_cfg_parent")
            if parent is None:
                continue
            if parent not in rows:
                raise RuntimeError("YuE2 CFG branch scheduled without its parent")
            positive = rows[parent]
            if discard[positive] or discard[negative]:
                raise RuntimeError("YuE2 CFG requires complete paired prefills")
            # Only semantic columns are projected on semantic-only steps; the
            # unused -inf columns must not enter subtraction (-inf - -inf).
            lo, hi = MUSIC_END, CODEC_OFFSET + CODEC_SIZE
            cond, uncond = logits[positive, lo:hi], logits[negative, lo:hi]
            cond.copy_(uncond + args["yue2_cfg_scale"] * (cond - uncond))
            discard[negative] = True  # One RNG draw/history, on the positive row.
            pairs.append((positive, negative))
        self._step_discard = discard
        try:
            result = super().sample(logits, sampling_metadata)
        finally:
            self._step_discard = original_discard
        for positive, negative in pairs:
            result.sampled_token_ids[negative].copy_(result.sampled_token_ids[positive])
        return result


class Yue2ModelRunner(GPUARModelRunner):
    def _update_states(self, scheduler_output):
        # Only runner bookkeeping sees these IDs as finished. The model callback
        # still receives the genuine terminal/abort IDs from execute_model.
        detached = getattr(scheduler_output, "acoustic_req_ids", set())
        return super()._update_states(
            replace(scheduler_output, finished_req_ids=scheduler_output.finished_req_ids | detached)
        )

    def _with_audio(self, output):
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()
        ready = self.model.poll_audio()
        # The runner may hold a CUDAGraphWrapper: assignment would shadow the
        # underlying model's list on that wrapper instead of consuming it.
        started = list(self.model.acoustic_started)
        self.model.acoustic_started.clear()
        return Yue2ModelRunnerOutput(**vars(output), acoustic_started=started, acoustic_ready=ready)

    @torch.inference_mode()
    def execute_model(self, scheduler_output, *args, **kwargs):
        replaced = getattr(scheduler_output, "replaced_req_ids", set())
        if replaced:
            # A new prompt reuses the native request ID. Retire the old runner
            # and model phase state before admitting its replacement prefill.
            scheduler_output = replace(scheduler_output, finished_req_ids=scheduler_output.finished_req_ids | replaced)
        output = super().execute_model(scheduler_output, *args, **kwargs)
        if output is not None:
            output = self._with_audio(output)
            if not scheduler_output.total_num_scheduled_tokens and not output.acoustic_ready:
                # shortcut: poll every 1ms while AR is idle; use events if this becomes costly.
                time.sleep(0.001)
        return output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output):
        return self._with_audio(super().sample_tokens(grammar_output))


class Yue2Worker(GPUARWorker):
    model_runner_cls = Yue2ModelRunner


class Yue2SchedulerBase(OmniARScheduler):
    def _update_request_with_output(self, request, new_token_ids, is_stale=False):
        args = request.sampling_params.extra_args or {}
        if not is_stale and args.get(AUTO_ABC):
            request.append_output_token_ids(new_token_ids)
            tokens = list(request.output_token_ids)
            if ABC_END in tokens or len(tokens) >= request.max_tokens:
                hit_end = bool(tokens and tokens[-1] == ABC_END)
                abc = tokens[:-1] if hit_end else tokens
                if not abc or any(not 0 <= token < EOD for token in abc):
                    request.status = RequestStatus.FINISHED_ERROR
                    request.stop_reason = "YuE2 generated an empty or invalid ABC score"
                    return [], True
                request.yue2_abc_ids = abc
                request.yue2_abc_truncated = not hit_end
                prefix = list(request.prompt_token_ids) + abc + [ABC_END, MUSIC_START]
                if len(prefix) + args[SEMANTIC_ARGS]["yue2_max_audio_frames"] + 2 > self.max_model_len:
                    request.status = RequestStatus.FINISHED_ERROR
                    request.stop_reason = "YuE2 semantic phase exceeds the configured context"
                    return [], True
                request.yue2_next_prefix = prefix
            # ABC tokens are an internal phase; no client-visible completion.
            return [], False
        if getattr(request, "yue2_acoustic", False) and not is_stale:
            assert new_token_ids == [MUSIC_END]
            # Keep the end token for the final audio output, not this AR handoff.
            return [], False
        return super()._update_request_with_output(request, new_token_ids, is_stale=is_stale)

    def get_num_unfinished_requests(self):
        return super().get_num_unfinished_requests() + sum(
            bool(getattr(r, "yue2_acoustic", False)) and not r.is_finished() for r in self.requests.values()
        )

    def schedule(self, throttle_prefills=False):
        if isinstance(self, OmniCFGARScheduler) and self._pairs:
            output = OmniCFGARScheduler.schedule(self, throttle_prefills)
        else:
            output = OmniARScheduler.schedule(self, throttle_prefills)
        acoustic = {rid for rid, r in self.requests.items() if getattr(r, "yue2_acoustic", False)}
        replaced = set()
        for rid in output.num_scheduled_tokens:
            request = self.requests[rid]
            if getattr(request, "yue2_reset_phase", False):
                replaced.add(rid)
                request.yue2_reset_phase = False
        return Yue2SchedulerOutput(**vars(output), acoustic_req_ids=acoustic, replaced_req_ids=replaced)

    def update_from_output(self, scheduler_output, model_runner_output):
        started = getattr(model_runner_output, "acoustic_started", ())
        for rid in started:
            request = self.requests.get(rid)
            if request is not None and not request.is_finished():
                request.yue2_acoustic = True
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for rid in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(rid)
            prefix = getattr(request, "yue2_next_prefix", None)
            if request is None or request.is_finished() or prefix is None:
                continue
            args = dict(request.sampling_params.extra_args[SEMANTIC_ARGS])
            args["yue2_prefix_ids"] = prefix
            params = SamplingParams(
                max_tokens=args["yue2_max_audio_frames"] + 2,
                detokenize=False,
                stop_token_ids=list(STOP_TOKEN_IDS),
                extra_args=args,
            )
            self.running.remove(request)
            self._release_replaced_streaming_prompt_cache(request)
            update = StreamingUpdate(
                mm_features=None,
                prompt_token_ids=prefix,
                max_tokens=params.max_tokens,
                arrival_time=request.arrival_time,
                sampling_params=params,
            )
            update.additional_information = request.additional_information
            self._replace_streaming_session(request, update)
            request.max_tokens = params.max_tokens
            request.yue2_next_prefix = None
            request.yue2_reset_phase = True
            self.waiting.prepend_request(request)
        for rid in started:
            request = self.requests.get(rid)
            if request is None or request.is_finished():
                continue
            self.running.remove(request)
            self._inflight_prefills.discard(request)
            self._free_request_blocks(request)
            self.encoder_cache_manager.free(request)
        for rid, payload in getattr(model_runner_output, "acoustic_ready", {}).items():
            request = self.requests.get(rid)
            if request is None or request.is_finished():
                continue  # Abort won the race; normal finished-ID cleanup cancels the job.
            failed = bool(payload["meta.error"])
            if hasattr(request, "yue2_abc_ids"):
                payload["abc_token_ids"] = torch.tensor(request.yue2_abc_ids, dtype=torch.int64)
                payload["meta.abc_truncated"] = torch.tensor(request.yue2_abc_truncated)
            request.append_output_token_ids(MUSIC_END)
            request.status = RequestStatus.FINISHED_ERROR if failed else RequestStatus.FINISHED_STOPPED
            self._free_request(request)
            batch = outputs.setdefault(request.client_index, EngineCoreOutputs())
            # Audio completes after the parent scheduler's coalesced-output flush.
            held_tokens = self._held_token_ids.pop(rid, [])
            batch.outputs.append(
                OmniEngineCoreOutput(
                    request_id=rid,
                    new_token_ids=held_tokens + [MUSIC_END],
                    is_coalesced=bool(held_tokens),
                    finish_reason=FinishReason.ERROR if failed else FinishReason.STOP,
                    stop_reason="YuE2 synthesis failed" if failed else MUSIC_END,
                    multimodal_output=payload,
                )
            )
        # Merge, since the base scheduler may already have emitted abort IDs.
        for client, finished in (self.finished_req_ids_dict or {}).items():
            batch = outputs.setdefault(client, EngineCoreOutputs())
            batch.finished_requests = (batch.finished_requests or set()) | finished
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict.clear()
        return outputs


class Yue2Scheduler(Yue2SchedulerBase, OmniCFGARScheduler):
    """Reuse native atomic admission; expose only the parent outside stage 0."""

    def __init__(self, *args, **kwargs):
        # Ordinary songs retain native admission, including one-slot deployments.
        OmniARScheduler.__init__(self, *args, **kwargs)
        if self.scheduler_config.async_scheduling:
            raise ValueError("YuE2 requires synchronous scheduling")
        if self.policy != SchedulingPolicy.FCFS or self.cache_config.enable_prefix_caching:
            raise ValueError("YuE2 requires FCFS and disabled prefix caching")
        if self.vllm_config.speculative_config is not None or self.connector is not None:
            raise ValueError("YuE2 does not support speculative decoding or KV connectors")
        if len(self.kv_cache_manager.kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("YuE2 requires one full-attention KV group")
        self._capacity = self.kv_cache_manager.block_pool.num_gpu_blocks - 1 - self.kv_cache_manager.watermark_blocks
        self._pairs = {}
        self._request_pairs = {}
        self._prefill_min_output_tokens = 0
        self._internal_ids: set[str] = set()
        self._rejected: dict[str, str] = {}

    def get_num_unfinished_requests(self):
        return super().get_num_unfinished_requests() + len(self._rejected)

    def schedule(self, throttle_prefills=False):
        self.waiting.remove_requests([self.requests[rid] for rid in self._rejected if rid in self.requests])
        return super().schedule(throttle_prefills)

    def add_request(self, request):
        super().add_request(request)
        self._add_negative(request)

    def _add_negative(self, request):
        args = request.sampling_params.extra_args or {}
        if (
            args.get("yue2_phase") != "semantic"
            or args.get("yue2_cfg_scale", 1) == 1
            or args.get("yue2_cfg_parent") is not None
            or getattr(request, "yue2_cfg_created", False)
        ):
            return
        prefix = list(args["yue2_negative_prefix"])
        if args.get("yue2_cfg_needs_abc"):
            prefix += request.yue2_abc_ids + [ABC_END, MUSIC_START]
        lengths = (request.num_prompt_tokens, len(prefix))
        pages = sum(-(-min(self.max_model_len, n + request.max_tokens) // self.block_size) for n in lengths)
        if (
            min(self.max_num_running_reqs, self.max_num_active_reqs) < 2
            or self.scheduler_config.enable_chunked_prefill
            or sum(lengths) > self.max_num_scheduled_tokens
            or pages > self._capacity
        ):
            self._rejected[request.request_id] = (
                "YuE2 CFG requires complete prefills and sufficient AR slots, prefill token budget and KV capacity; "
                "disable chunked prefill, increase deployment capacity or reduce the request budget"
            )
            return
        # Branch-local sampling state must not mutate the parent's nested args.
        params = copy.deepcopy(request.sampling_params)
        child_id = request.request_id + "-yue2-negative"
        params.extra_args.update(
            yue2_prefix_ids=prefix,
            yue2_cfg_parent=request.request_id,
            yue2_skip_synthesis=True,
        )
        negative = OmniRequest(
            child_id,
            prefix,
            params,
            None,
            external_req_id=child_id,
            client_index=request.client_index,
            arrival_time=request.arrival_time,
            priority=request.priority,
        )
        self._internal_ids.add(child_id)
        request.yue2_cfg_created = True
        self._pairs[request.request_id] = {"cond": request.request_id, "uncond": child_id}
        self._request_pairs.update({request.request_id: request.request_id, child_id: request.request_id})
        OmniARScheduler.add_request(self, negative)

    def finish_requests(self, request_ids, finished_status):
        finished = super().finish_requests(request_ids, finished_status)
        # EngineCore uses this return value to send shutdown/abort outputs.
        return [request for request in finished if request.request_id not in self._internal_ids]

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        # Automatic ABC may have replaced a prompt during this update.
        for rid in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(rid)
            if request is not None and not request.is_finished():
                self._add_negative(request)
        for rid, reason in self._rejected.items():
            request = self.requests.get(rid)
            if request is None or request.is_finished():
                continue
            self.finish_requests(rid, RequestStatus.FINISHED_ERROR)
            batch = outputs.setdefault(request.client_index, EngineCoreOutputs())
            batch.outputs.append(
                OmniEngineCoreOutput(
                    request_id=rid,
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                    stop_reason=reason,
                )
            )
            batch.finished_requests = (batch.finished_requests or set()) | {rid}
        self._rejected.clear()
        for batch in outputs.values():
            batch.outputs = [output for output in batch.outputs if output.request_id not in self._internal_ids]
            if batch.finished_requests is not None:
                batch.finished_requests.difference_update(self._internal_ids)
        # Keep scheduler.finished_req_ids intact: the worker needs them to
        # release negative-row model/runner state, even though clients do not.
        self._internal_ids.intersection_update(self.requests)
        return outputs

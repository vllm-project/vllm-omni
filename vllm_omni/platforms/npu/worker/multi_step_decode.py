# SPDX-License-Identifier: Apache-2.0
"""Runner-side execution of multi-step decode windows on Ascend NPUs.

Companion to ``vllm_omni.core.sched.multi_step_decode`` (scheduler-side
planning).  When the scheduler rewrites a decode step into a K-token
window, this module replays the K decode steps inside a single
``execute_model`` call:

    step input -> model.preprocess (per-request embedding)
               -> forward -> make_omni_output / multimodal extraction
               -> compute_logits -> sampler -> next step input

Everything the engine loop would otherwise do once per token -- batch
layout rebuild, attention metadata, forward context, sampling and
inter-stage payload assembly -- runs K times here, but the host-side
engine steps (scheduler pass, IPC, output processing) run once per
window.  That is the whole point: for stages whose per-token device time
is far below the per-step host time, the engine step rate becomes the
throughput ceiling and windowing lifts it by ~K.

The final ``OmniModelRunnerOutput`` is stashed on the runner
(``_multi_step_pending_output``) for ``sample_tokens()``, preserving the
engine's execute_model -> sample_tokens contract.

Early exit: when every request's ``make_omni_output`` has already
signalled completion, the loop stops before K -- the remaining steps
would only burn device time on dead rows.  The scheduler-side
``reconcile_window_shortfall`` absorbs the unproduced tokens, so early
exit needs no special casing anywhere else.

Fail-closed contract: ``validate_multi_step_plan`` re-checks every static
invariant (platform, parallelism, model contract, sampling features)
before a window runs.  Per-step mutable state is deliberately NOT
re-checked here: under async scheduling the runner's input_batch view
lags the scheduler, so such checks would produce false refusals -- and a
refusal after the scheduler planned follow-up windows on top of this one
cannot be recovered by shrinking (the follow-up plans' KV positions
would skip this window's reserved slots, leaving holes of uninitialized
cache).  Phase and batch composition are gated scheduler-side with
authoritative state instead.  The residual static refusals shrink the
K-token schedule back to a plain single-token step
(``shrink_refused_multi_step_window``); the scheduler side reconciles the
reservation shortfall, so a refused window costs nothing but a skipped
optimization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphMode, CUDAGraphStat
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.utils import record_function_or_nullcontext
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.ops.rotary_embedding import update_cos_sin

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.utils.config import stage_sends_async_output
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.utils.mm_outputs import partition_payload_list

logger = init_logger(__name__)

# Diagnostic counters: scheduler-side plans vs runner-side executions.  A
# large planned>>executed gap means a gate here refuses systematically and
# every engine step pays the plan/allocate/refuse/reconcile churn for
# nothing (see the multi-step A/B analysis).
_refusal_counts: dict[str, int] = {}
_executed_windows = 0
_executed_steps = 0


def _refuse(reason: str) -> None:
    _refusal_counts[reason] = _refusal_counts.get(reason, 0) + 1
    if _refusal_counts[reason] == 1:
        logger.warning("Multi-step plan refused at runner: %s (first occurrence)", reason)


def validate_multi_step_plan(
    runner: Any, scheduler_output: SchedulerOutput
) -> dict[str, int] | None:
    """Validate the scheduler's window plan against runner-local state.

    Returns the plan (req_id -> K) when this runner can host the window,
    None to run the normal single-step path (fail-closed; the scheduler
    side reconciles the shortfall).
    """
    try:
        plan = getattr(scheduler_output, "multi_step_plan", None)
        if not plan:
            return _refuse("no_plan")
        window_sizes = {int(k) for k in plan.values()}
        if len(window_sizes) != 1:
            return _refuse("mixed_window_sizes")
        window_k = window_sizes.pop()
        if window_k < 2:
            return _refuse("window_k_lt_2")
        num_scheduled = scheduler_output.num_scheduled_tokens
        if set(num_scheduled) != set(plan):
            return _refuse("num_scheduled_mismatch")
        if any(int(n) != window_k for n in num_scheduled.values()):
            return _refuse("scheduled_not_k")
        if (
            scheduler_output.scheduled_new_reqs
            or scheduler_output.scheduled_spec_decode_tokens
            or scheduler_output.scheduled_encoder_inputs
        ):
            return _refuse("new_or_spec_or_encoder_reqs")
        # The window drives input feedback through the async-scheduling
        # fast path; anything else falls back to the standard loop.
        if not runner.use_async_scheduling:
            return _refuse("no_async_scheduling")
        if runner.num_spec_tokens or runner.speculative_config is not None:
            return _refuse("spec_decode")
        pp_group = get_pp_group()
        if pp_group.world_size != 1 or not pp_group.is_last_rank:
            return _refuse("pp")
        if get_tp_group().world_size != 1:
            return _refuse("tp")
        if (
            runner.vllm_config.parallel_config.data_parallel_size != 1
            or runner.pcp_size != 1
            or runner.dcp_size != 1
            or runner.use_cp
        ):
            return _refuse("dp_pcp_dcp_cp")
        if runner.lora_config is not None or runner.is_pooling_model:
            return _refuse("lora_or_pooling")
        if runner.model_config.is_encoder_decoder or runner.supports_mm_inputs:
            # Omni wrapper architectures host the multimodal encoders for the
            # earlier stages, so the talker stage legitimately reports
            # supports_mm_inputs even though its decode path consumes plain
            # token ids (preprocess() rebuilds embeddings from request-local
            # state).  Allow it only when the loaded model declares the
            # multi-step contract for THIS stage -- same (arch, stage) gate as
            # the scheduler side.
            stage = getattr(runner.model_config, "model_stage", None)
            declared_stages = getattr(runner.model, "supports_multi_step_stages", None)
            if not (
                getattr(runner.model, "supports_multi_step_decode", False)
                or (declared_stages is not None and stage in declared_stages)
            ):
                return _refuse("enc_dec_or_mm_inputs")
        if runner.omni_prefix_cache is not None:
            return _refuse("prefix_cache")
        if getattr(runner, "has_talker_mtp", False):
            return _refuse("talker_mtp")
        # NOTE: no duplex-hook / prefer_model_sampler refusal here — the
        # window loop mirrors _sample's custom-sampler branch (logit bias,
        # duplex hook, model sampler with prepared metadata), so models that
        # use them host windows with identical sampling semantics.
        if getattr(runner.cache_config, "mamba_cache_mode", "off") == "align":
            return _refuse("mamba_align")
        if runner.model_config.enable_return_routed_experts:
            return _refuse("routed_experts")
        if getattr(runner, "calculate_kv_scales", False):
            return _refuse("kv_scales")
        if getattr(runner, "dynamic_eplb", False):
            return _refuse("dynamic_eplb")
        if runner.debugger is not None:
            return _refuse("debugger")
        if runner.ascend_config.profiling_chunk_config.enabled:
            return _refuse("profiling_chunk")
        from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer

        if has_ec_transfer() and get_ec_transfer().is_producer:
            return _refuse("ec_transfer_producer")
        model = runner.model
        # Model capability contract (declared via supports_multi_step_decode):
        # preprocess() builds decode embeddings from request-local state,
        # make_omni_output advances that state in place.
        if not getattr(model, "has_preprocess", False) or not hasattr(model, "make_omni_output"):
            return _refuse("model_contract_missing")
        sampling_metadata = runner.input_batch.sampling_metadata
        if not sampling_metadata.no_penalties:
            return _refuse("penalties")
        if runner.input_batch.bad_words_token_ids:
            return _refuse("bad_words")
        if not runner.input_batch.no_allowed_token_ids:
            return _refuse("allowed_token_ids")
        if runner.input_batch.logprob_token_ids:
            return _refuse("logprob_token_ids")
        max_num_logprobs = sampling_metadata.max_num_logprobs
        if max_num_logprobs is not None and max_num_logprobs > 0:
            return _refuse("max_num_logprobs")
        if runner.num_prompt_logprobs:
            return _refuse("prompt_logprobs")
        # NOTE: no per-step mutable-state re-checks here (prefill phase,
        # batch composition).  Under async scheduling the runner's
        # input_batch view lags the scheduler -- the plan is validated
        # before _update_states syncs it -- so those checks would produce
        # false refusals.  A refusal after the scheduler planned follow-up
        # windows on top of this one is NOT recoverable by shrinking: the
        # follow-up plan's KV positions would skip this window's reserved
        # slots, leaving holes of uninitialized cache that corrupt every
        # later step.  The scheduler side already gates phase/batch with
        # authoritative state (_request_admits_multi_step), so the runner
        # only re-checks static configuration above.
        global _executed_windows
        _executed_windows += 1
        return plan
    except Exception:
        logger.exception("Multi-step plan validation failed; using single-step decode")
        return None


def shrink_refused_multi_step_window(scheduler_output: SchedulerOutput) -> None:
    """Shrink a runner-refused window back to a plain single-token step.

    The scheduler-side plan bumped ``num_scheduled_tokens`` to K and reserved
    K-1 extra KV slots per request.  If this runner refuses to host the
    window, executing that K-token schedule as a plain multi-token decode
    would be wrong twice over: the talker rebuilds every decode embedding
    from the request-local codec chain, so rows 1..K-1 would run with stale
    embeddings, and the step reports only one sampled token per request
    while KV writes land on all K slots.  The polluted cache slots and the
    phantom positions permanently corrupt every later step (the scheduler
    reconcile only rolls back accounting, not cache contents).

    Shrinking to one token per request makes the fallback bit-identical to a
    never-planned step; the scheduler-side ``reconcile_window_shortfall``
    rolls back the K-1 unused slot reservations.
    """
    plan = getattr(scheduler_output, "multi_step_plan", None)
    if not plan:
        return
    reclaimed = 0
    for req_id, planned in plan.items():
        scheduled = int(scheduler_output.num_scheduled_tokens.get(req_id, 0))
        if scheduled > 1:
            scheduler_output.num_scheduled_tokens[req_id] = 1
            reclaimed += scheduled - 1
    if reclaimed:
        scheduler_output.total_num_scheduled_tokens -= reclaimed


def execute_multi_step_window(
    runner: Any, scheduler_output: SchedulerOutput, plan: dict[str, int]
) -> None:
    """Replay the K decode steps of the window inside this one call.

    On return the assembled ``OmniModelRunnerOutput`` is stashed in
    ``runner._multi_step_pending_output`` for ``sample_tokens()``.
    """
    window_k = next(iter(plan.values()))

    global _executed_steps
    _executed_steps += window_k
    if _executed_steps % (window_k * 50) == window_k or _executed_steps == window_k:
        logger.info(
            "Multi-step counters: executed_windows=%d executed_steps=%d refusals=%s",
            _executed_windows, _executed_steps, dict(sorted(_refusal_counts.items())),
        )

    with record_function_or_nullcontext("multi_step_window"):
        with runner.synchronize_input_prep():
            deferred = runner._update_states(scheduler_output)
            if deferred is not None:
                # Mamba/spec corrections are gated off for windows; apply
                # eagerly so per-step state starts consistent.
                deferred()

            # Batch identity must be read after _update_states: under async
            # scheduling the plan may admit requests the input_batch has not
            # seen yet (the scheduler's plan is authoritative for the step).
            num_reqs = runner.input_batch.num_reqs
            req_ids = list(runner.input_batch.req_ids)
            ones = np.ones(num_reqs, dtype=np.int32)

            # Uniform decode layout: one query token per request, frozen for
            # the whole window (the batch composition cannot change mid-window).
            runner.query_start_loc.np[0] = 0
            runner.query_start_loc.np[1 : num_reqs + 1] = runner.arange_np[1 : num_reqs + 1]
            runner.query_start_loc.copy_to_gpu()
            runner.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)
            runner.query_pos.np[:num_reqs] = 0
            runner.query_pos.copy_to_gpu(num_reqs)
            runner.req_indices.np[:num_reqs] = runner.arange_np[:num_reqs]
            runner.req_indices.copy_to_gpu(num_reqs)
            runner.num_scheduled_tokens.np[:num_reqs] = ones
            runner.num_scheduled_tokens.copy_to_gpu(num_reqs)
            runner.decode_token_per_req = 1
            runner._build_attn_state(num_reqs, ones, ones)
            runner.query_lens = torch.from_numpy(ones)
            runner.logits_indices = runner.query_start_loc.gpu[1 : num_reqs + 1] - 1
            runner.with_prefill = False
            runner.num_discarded_requests = 0
            runner.discard_request_mask.np[:num_reqs] = False
            runner.discard_request_mask.copy_to_gpu(num_reqs)
            runner._omni_num_scheduled_tokens_np = ones

            runner.input_batch.block_table.commit_block_table(num_reqs)

            (
                cudagraph_mode,
                batch_desc,
                _should_ubatch,
                _num_tokens_across_dp,
                cudagraph_stats,
            ) = runner._determine_batch_execution_and_padding(
                num_tokens=num_reqs,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=ones,
                max_num_scheduled_tokens=1,
                use_cascade_attn=False,
                force_eager=runner.model_config.enforce_eager,
            )
            num_tokens_padded = batch_desc.num_tokens

            # Mirror model_runner_v1._pad_query_start_loc_for_fia (FULL branch):
            # under FULL graph replay, FIA's TND layout requires
            # sum(query_start_loc) == hidden_states rows.  When cudagraph
            # padding added rows (num_tokens_padded > num_reqs), insert one
            # dummy request whose query covers the padding instead of leaving
            # the -1 sentinels in place -- actual_seq_lengths_q containing -1
            # makes aclnnFusedInferAttentionScoreV3 fail with error 561002
            # (batch=1 windows have no padding, which is why they pass).
            # The dummy's KV row points at block 0 (filled by
            # _build_attention_metadata) and never writes the cache because
            # reshape_and_cache slices to the unpadded token count.
            num_reqs_padded = num_reqs
            if cudagraph_mode == CUDAGraphMode.FULL and num_tokens_padded > num_reqs:
                runner.query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
                num_reqs_padded += 1
                runner.query_start_loc.copy_to_gpu()

            input_ids = runner.input_ids.gpu[:num_tokens_padded]
            inputs_embeds = runner.inputs_embeds.gpu[:num_tokens_padded]
            positions = runner.positions[:num_tokens_padded]
            logits_indices = runner.logits_indices
            sampling_metadata = runner.input_batch.sampling_metadata

            # Step 0 consumes the token sampled by the previous engine step
            # via the async-scheduling feedback path.
            prev_sampled = runner.input_batch.prev_sampled_token_ids
            prev_index = runner.input_batch.prev_req_id_to_index or {}
            if prev_sampled is not None:
                for i, req_id in enumerate(req_ids):
                    row = prev_index.get(req_id)
                    if row is None:
                        continue
                    input_ids[i] = prev_sampled[row, 0]
            else:
                runner.input_ids.copy_to_gpu(num_reqs)

            kv_connector_output = None
            per_req_hidden: list[list[torch.Tensor]] = [[] for _ in range(num_reqs)]
            per_req_deltas: list[list[torch.Tensor]] = [[] for _ in range(num_reqs)]
            per_req_finished = [False] * num_reqs
            sampled_steps: list[torch.Tensor] = []
            comp_cpu = runner.input_batch.num_computed_tokens_cpu
            steps_done = 0

            for step in range(window_k):
                with record_function_or_nullcontext("multi_step_window:step"):
                    if step > 0:
                        comp_cpu[:num_reqs] += 1
                        for req_id in req_ids:
                            req_state = runner.requests.get(req_id)
                            if req_state is not None:
                                req_state.num_computed_tokens += 1
                    # Fresh CPU staging tensor per copy: the pinned batch
                    # buffer is rewritten next step while this H2D copy may
                    # still be in flight.
                    runner.num_computed_tokens[:num_reqs].copy_(
                        torch.from_numpy(comp_cpu[:num_reqs].copy()), non_blocking=True
                    )
                    runner.positions[:num_reqs] = runner.num_computed_tokens[:num_reqs].to(torch.int64)
                    runner.positions[num_reqs:num_tokens_padded].zero_()
                    runner.seq_lens[:num_reqs] = (
                        runner.num_computed_tokens[:num_reqs]
                        + runner.num_scheduled_tokens.gpu[:num_reqs]
                    )
                    runner.seq_lens[num_reqs:].fill_(0)
                    runner.optimistic_seq_lens_cpu[:num_reqs].copy_(
                        torch.from_numpy(comp_cpu[:num_reqs] + 1)
                    )
                    runner.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                    runner.input_batch.block_table.compute_slot_mapping(
                        num_reqs,
                        runner.query_start_loc.gpu[: num_reqs + 1],
                        runner.positions[:num_reqs],
                    )
                    update_cos_sin(positions)

                    (attn_metadata, _spec_common) = runner._build_attention_metadata(
                        num_tokens=num_reqs,
                        num_reqs=num_reqs,
                        max_query_len=1,
                        num_tokens_padded=num_tokens_padded,
                        num_reqs_padded=num_reqs_padded,
                        ubatch_slices=None,
                        logits_indices=logits_indices,
                        use_spec_decode=False,
                        num_scheduled_tokens={req_id: 1 for req_id in req_ids},
                        num_scheduled_tokens_np=ones,
                        cascade_attn_prefix_lens=None,
                    )

                    # Per-request decode embeddings: the talker derives its
                    # input from the previous step's sampled codec token
                    # (request-local state advanced in place by
                    # make_omni_output).
                    for i, req_id in enumerate(req_ids):
                        req_state = runner.requests.get(req_id)
                        info = runner.model_intermediate_buffer.get(req_id)
                        if info is None:
                            info = runner.model_intermediate_buffer.setdefault(req_id, {})
                        info["request_id"] = req_id
                        info["duplex_token_offset"] = int(comp_cpu[i])
                        info["duplex_prompt_len"] = (
                            len(req_state.prompt_token_ids) if req_state is not None else None
                        )
                        info["_omni_prompt_len"] = (
                            len(req_state.prompt_token_ids) if req_state is not None else 0
                        )
                        info["_omni_num_computed_tokens"] = int(comp_cpu[i])
                        info["_omni_is_prefill"] = False
                        req_input_ids, req_embeds, update_dict = runner.model.preprocess(
                            input_ids=runner.input_ids.gpu[i : i + 1],
                            input_embeds=inputs_embeds[i : i + 1],
                            **info,
                        )
                        seg_len = min(1, int(req_embeds.shape[0]))
                        if seg_len:
                            inputs_embeds[i : i + seg_len] = req_embeds[:seg_len]
                        if isinstance(req_input_ids, torch.Tensor) and req_input_ids.numel() == 1:
                            runner.input_ids.gpu[i] = req_input_ids.reshape(-1)[0]
                        if update_dict:
                            runner._update_intermediate_buffer(req_id, update_dict)

                    with (
                        set_ascend_forward_context(
                            attn_metadata,
                            runner.vllm_config,
                            num_tokens=num_tokens_padded,
                            num_tokens_across_dp=None,
                            aclgraph_runtime_mode=cudagraph_mode,
                            batch_descriptor=batch_desc,
                            num_actual_tokens=num_reqs,
                            model_instance=runner.model,
                            max_tokens_across_pcp=0,
                            skip_compiled=False,
                        ),
                        runner.maybe_get_kv_connector_output(scheduler_output) as kv_output_step,
                    ):
                        hidden_states = runner._model_forward(
                            num_tokens_padded, input_ids, positions, None, inputs_embeds
                        )
                    kv_connector_output = kv_output_step

                hidden_states, mm_outputs = runner.extract_multimodal_outputs(hidden_states)

                # Intermediate steps skip the full OmniOutput wire wrapping:
                # hidden rows and codec deltas are accumulated and shipped
                # once at window end.
                audio_deltas = None
                finished_flags = None
                if isinstance(mm_outputs, dict):
                    codes = mm_outputs.get("codes")
                    if isinstance(codes, dict):
                        audio_deltas = codes.get("audio")
                    meta = mm_outputs.get("meta")
                    if isinstance(meta, dict):
                        finished_flags = meta.get("finished")
                for i in range(num_reqs):
                    per_req_hidden[i].append(hidden_states[i : i + 1].detach())
                    if audio_deltas is not None and i < len(audio_deltas):
                        delta = audio_deltas[i]
                        if isinstance(delta, torch.Tensor) and delta.numel():
                            per_req_deltas[i].append(delta)
                    if finished_flags is not None and i < len(finished_flags):
                        flag = finished_flags[i]
                        if isinstance(flag, torch.Tensor):
                            per_req_finished[i] = per_req_finished[i] or bool(flag.item())
                        else:
                            per_req_finished[i] = per_req_finished[i] or bool(flag)

                sample_hidden_states = hidden_states[logits_indices]
                try:
                    logits = runner.model.compute_logits(
                        sample_hidden_states, sampling_metadata=sampling_metadata
                    )
                except TypeError:
                    logits = runner.model.compute_logits(sample_hidden_states)
                if step == 0:
                    # Fill the previous engine step's pending placeholder
                    # exactly like the normal _sample path would.
                    runner.input_batch.update_async_output_token_ids()
                # Mirror _sample's custom-sampler branch (npu_ar_model_runner
                # _sample): logit bias -> duplex hook -> model sampler with
                # prepared metadata, falling back to the engine sampler when
                # the model sampler declines.  Keeping this identical to the
                # single-step path is what allows prefer_model_sampler models
                # (e.g. the MiniCPM-o talker) to host windows.
                model_sample = getattr(runner.model, "sample", None)
                if callable(model_sample) and getattr(runner.model, "prefer_model_sampler", False):
                    if hasattr(runner.sampler, "logit_bias_state"):
                        runner.sampler.logit_bias_state.apply_logit_bias(
                            logits,
                            runner.input_batch.expanded_idx_mapping,
                            runner.input_batch.idx_mapping_np,
                            runner.input_batch.positions[runner.input_batch.logits_indices],
                        )
                    prepared_sampling_metadata = runner._sampling_metadata_for_model_sampler(
                        sampling_metadata
                    )
                    runner._apply_duplex_sampling(logits, prepared_sampling_metadata)
                    sampler_output = model_sample(logits, prepared_sampling_metadata)
                    if sampler_output is None:
                        sampler_output = runner.sampler(
                            logits=logits, sampling_metadata=sampling_metadata
                        )
                else:
                    sampler_output = runner.sampler(
                        logits=logits, sampling_metadata=sampling_metadata
                    )
                sampled_steps.append(sampler_output.sampled_token_ids)
                steps_done = step + 1

                # Mirror _bookkeeping_sync's async bookkeeping: one -1
                # placeholder per request per step.  Values are filled at
                # window end; only length-based processors (min_tokens)
                # read them inside the window (penalties are gated off).
                for i in range(num_reqs):
                    pos = runner.input_batch.num_tokens_no_spec[i]
                    runner.input_batch.token_ids_cpu[i, pos : pos + 1] = -1
                    runner.input_batch.is_token_ids[i, pos : pos + 1] = True
                    runner.input_batch.num_tokens_no_spec[i] = pos + 1
                    req_state = runner.requests.get(req_ids[i])
                    if req_state is not None:
                        req_state.output_token_ids.append(-1)

                # Early exit: every request already signalled completion;
                # remaining steps would only burn device time on dead rows.
                if steps_done < window_k and all(per_req_finished):
                    logger.debug(
                        "Multi-step window early exit at step %d/%d (all requests finished)",
                        steps_done,
                        window_k,
                    )
                    break

            # ---- window end: materialize results ----
            if steps_done < window_k:
                sampled_steps = sampled_steps[:steps_done]
            window_k = steps_done
            sampled_all = torch.stack(sampled_steps, dim=0).reshape(window_k, num_reqs)
            sampled_lists = sampled_all.cpu().tolist()

            for i, req_id in enumerate(req_ids):
                req_state = runner.requests.get(req_id)
                if req_state is None:
                    continue
                out_ids = req_state.output_token_ids
                for s in range(window_k):
                    idx = len(out_ids) - window_k + s
                    if 0 <= idx < len(out_ids) and out_ids[idx] == -1:
                        out_ids[idx] = int(sampled_lists[s][i])

            # Async feedback for the next engine step: the last window
            # step's sampled token becomes the next input token.
            runner.input_batch.prev_sampled_token_ids = sampled_steps[-1]
            runner.input_batch.prev_req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}
            copy_stream = getattr(runner, "async_output_copy_stream", None)
            if copy_stream is not None:
                with torch.npu.stream(copy_stream):
                    copy_stream.wait_stream(torch.npu.current_stream())
                    last_sampled_cpu = sampled_steps[-1].to("cpu", non_blocking=True)
                ready_event = torch.npu.Event(blocking=True)
                ready_event.record(copy_stream)
            else:
                last_sampled_cpu = sampled_steps[-1].cpu()
                ready_event = None
            runner.input_batch.set_async_sampled_token_ids(last_sampled_cpu, ready_event)
            runner.kv_connector_output = None

            runner._multi_step_pending_output = _build_window_output(
                runner,
                req_ids,
                sampled_lists,
                per_req_hidden,
                per_req_deltas,
                per_req_finished,
                kv_connector_output,
                cudagraph_stats,
            )


def _build_window_output(
    runner: Any,
    req_ids: list[str],
    sampled_per_step: list[list[int]],
    per_req_hidden: list[list[torch.Tensor]],
    per_req_deltas: list[list[torch.Tensor]],
    per_req_finished: list[bool],
    kv_connector_output: Any,
    cudagraph_stats: CUDAGraphStat | None,
) -> OmniModelRunnerOutput:
    """Assemble the OmniModelRunnerOutput for a completed window.

    ``sampled_per_step`` is [K][req_index]; the engine's reconciliation
    consumes K sampled tokens per request, matching the K tokens the
    scheduler reserved for the window step.  When the window exited early
    only the produced steps are reported, and the scheduler-side
    ``reconcile_window_shortfall`` rolls the reservation back.
    """
    num_reqs = len(req_ids)
    window_k = len(sampled_per_step)
    sampled_token_ids = [
        [int(sampled_per_step[s][i]) for s in range(window_k)] for i in range(num_reqs)
    ]

    _engine_output_type, downstream_req_ids = runner._resolve_pooler_payload_req_ids(list(req_ids))
    downstream_req_id_set = set(downstream_req_ids)
    pooler_output: list[dict[str, object]] = []
    for i, req_id in enumerate(req_ids):
        if req_id not in downstream_req_id_set:
            pooler_output.append({})
            continue
        payload: dict[str, object] = {}
        # Hidden rows of the K window steps (tail-aligned per request); the
        # talker batch is not the sparse-audio path (gated upstream).
        if per_req_hidden[i]:
            payload["hidden"] = torch.cat(per_req_hidden[i], dim=0).to("cpu").contiguous()
        if per_req_deltas[i]:
            payload["codes.audio"] = torch.cat(per_req_deltas[i], dim=0).to("cpu")
        payload["meta.finished"] = torch.tensor(bool(per_req_finished[i]), dtype=torch.bool)
        pooler_output.append(flatten_payload(payload))

    pooler_output = pooler_output or []
    if runner._async_chunk and stage_sends_async_output(runner.model_config):
        pooler_inter, pooler_client = partition_payload_list(pooler_output)
    else:
        # Non-async-chunk ships the full payload to the next stage via
        # inter_stage_outputs (the NPU runner has no separate full-payload
        # accumulate).
        pooler_inter, pooler_client = pooler_output, pooler_output

    if pooler_inter and runner._should_accumulate_full_payload_output():
        for i, req_id in enumerate(req_ids):
            req_state = runner.requests.get(req_id)
            if req_state is not None and pooler_inter[i]:
                runner.accumulate_full_payload_output(req_id, pooler_inter[i], req_state)

    inter_stage_outputs = runner._build_multimodal_outputs(pooler_inter)
    multimodal_outputs = (
        inter_stage_outputs
        if pooler_client is pooler_inter
        else runner._build_multimodal_outputs(pooler_client)
    )
    model_runner_output = OmniModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=multimodal_outputs,
        inter_stage_outputs=inter_stage_outputs,
        kv_connector_output=kv_connector_output,
        ec_connector_output=None,
        cudagraph_stats=cudagraph_stats,
    )
    model_runner_output.kv_extracted_req_ids = getattr(runner, "kv_extracted_req_ids", None)
    runner.kv_extracted_req_ids = None
    model_runner_output.omni_connector_output = runner.get_omni_connector_output()
    return model_runner_output

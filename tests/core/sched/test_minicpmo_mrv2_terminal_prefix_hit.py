# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A preempted MiniCPM-o length stop readmitted through a prefix hit.

Real OmniARAsyncScheduler, KV cache manager, schedule(), _preempt_request and
update_from_output. A stale output commits the final token after preemption;
the readmission then hits the prefix cache up to that token, where upstream
admission pads a one-token decode row with draft slots.
"""

import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.model_executor.models.minicpmo_4_5 import handoff
from vllm_omni.model_executor.models.minicpmo_4_5.handoff import caches_turn_outputs, ends_with_own_row
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK = 4
NUM_BLOCKS = 16
NUM_SPEC = 3
DRAFT_ID = 7777
PROMPT = list(range(1001, 1013))  # three full blocks
FINAL = 112

init_none_hash(sha256)
_HASHER = get_request_block_hasher(BLOCK, sha256)


def _speculative_config():
    spec = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=NUM_SPEC,
        num_speculative_tokens_per_batch_size=None,
        disable_eagle_block_drop=False,
        parallel_drafting=True,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(sample_from_anchor=True)),
    )
    for name in (
        "use_eagle", "needs_eagle_block_drop", "use_eagle_block_drop", "use_dflash", "use_dspark",
        "uses_draft_model", "uses_dynamic_speculative_decoding", "uses_draft_kv_cache",
        "uses_extract_hidden_states", "use_multi_module_mtp",
    ):  # fmt: skip
        setattr(spec, name, types.MethodType(getattr(SpeculativeConfig, name), spec))
    spec.max_num_new_slots_for_drafting = SpeculativeConfig.max_num_new_slots_for_drafting.fget(spec)
    return spec


def _scheduler(prefix_caching, num_spec=NUM_SPEC):
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        multimodal_config=None,
        uses_mrope=False,
        is_encoder_decoder=False,
        max_model_len=256,
        runner_type="generate",
        is_diffusion=False,
        supports_multimodal_inputs=False,
        return_sampling_mask=False,
        recompute_preemption="allow",
        # Stage 0 of minicpmo_4_5_turn_mrv2.yaml, with a stage-0 draft unless num_spec is 0.
        model_arch="MiniCPMO45OmniForConditionalGeneration",
        model_stage="llm",
        session_mode="turn",
        use_v2_model_runner=True,
        async_chunk=False,
        supports_native_mrv2_data_plane=True,
        stage_id=0,
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=256, max_num_seqs=4, max_num_batched_tokens=32, is_encoder_decoder=False, async_scheduling=True
    )
    if num_spec:
        config.speculative_config = _speculative_config()
    config.cache_config = CacheConfig(block_size=BLOCK, enable_prefix_caching=prefix_caching)
    config.cache_config.num_gpu_blocks = NUM_BLOCKS
    register_all_kvcache_specs(config)
    spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float16)
    kv_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)]
    )
    scheduler = OmniARAsyncScheduler(config, kv_config, Mock(spec=StructuredOutputManager), BLOCK)
    assert scheduler._minicpmo45_terminal_rows_enabled() and scheduler.num_spec_tokens == num_spec
    return scheduler


def _request(rid, prompt, max_tokens):
    return OmniRequest(rid, prompt, SamplingParams(max_tokens=max_tokens), None, client_index=0, block_hasher=_HASHER)


def _booked(scheduler, scheduler_output):
    """Each request's booked span, read after schedule() advanced it."""
    spans = {}
    for rid, n in scheduler_output.num_scheduled_tokens.items():
        request = scheduler.requests[rid]
        start = request.num_computed_tokens - n
        ids = [request.all_token_ids[p] if p < request.num_tokens else DRAFT_ID for p in range(start, start + n)]
        spans[rid] = (start, ids)
    return spans


def _deliver(scheduler, scheduler_output, spans, sampled):
    """One runner output: rows for every booked position, the given sample per request."""
    rids = list(spans)
    payloads = []
    for rid in rids:
        start, ids = spans[rid]
        payloads.append(
            {
                "latent": torch.zeros(len(ids), 2),
                "latent_input_ids": torch.tensor(ids).reshape(-1, 1),
                "latent_positions": torch.arange(start, start + len(ids)).reshape(-1, 1),
            }
        )
    output = ModelRunnerOutput(
        req_ids=rids,
        req_id_to_index={rid: i for i, rid in enumerate(rids)},
        sampled_token_ids=[sampled[rid] for rid in rids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    output.multimodal_outputs = output.inter_stage_outputs = payloads
    finishes = []
    for engine_outputs in scheduler.update_from_output(scheduler_output, output).values():
        finishes += [(o.request_id, o.finish_reason) for o in engine_outputs.outputs if o.finish_reason is not None]
    return dict(zip(rids, payloads)), finishes


def _stale_final_token(scheduler):
    """A's prefill samples its final token; A is preempted before that output returns."""
    a = _request("A", PROMPT, 1)
    scheduler.add_request(a)
    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"A": len(PROMPT)}
    spans = _booked(scheduler, prefill)
    scheduler.running.remove(a)
    scheduler._preempt_request(a, 0.0)
    assert a.status == RequestStatus.PREEMPTED and a.num_stale_output_tokens > 0
    _, finishes = _deliver(scheduler, prefill, spans, {"A": [FINAL]})
    # The stale output commits the final token; the stop waits for the recompute.
    assert finishes == [] and "A" in scheduler._minicpmo45_terminal_rows
    assert a.status == RequestStatus.PREEMPTED and a.num_computed_tokens == 0
    assert list(a.output_token_ids) == [FINAL] and a.num_tokens == len(PROMPT) + 1
    return a


@pytest.mark.parametrize("prefix_caching", [True, False], ids=["prefix_hit", "cold"])
def test_readmitted_terminal_forward_is_draft_free(prefix_caching):
    scheduler = _scheduler(prefix_caching)
    a = _stale_final_token(scheduler)

    readmit = scheduler.schedule()
    assert readmit.num_scheduled_tokens.keys() == {"A"}
    start, ids = _booked(scheduler, readmit)["A"]
    if prefix_caching:
        # The hit reaches the final token: one input left, the row upstream pads.
        assert start == a.num_tokens - 1
    else:
        assert start == 0
    booked = (
        readmit.num_scheduled_tokens["A"],
        readmit.total_num_scheduled_tokens,
        readmit.scheduled_spec_decode_tokens.get("A"),
        a.num_computed_tokens,
        a.num_in_flight_tokens,
        a.num_output_placeholders,
    )

    payloads, finishes = _deliver(scheduler, readmit, {"A": (start, ids)}, {"A": [DRAFT_ID]})
    assert finishes == [("A", FinishReason.LENGTH)]
    assert ends_with_own_row(payloads["A"], a.num_tokens - 1, FINAL)
    # The terminal forward was the final token alone, booked without drafts.
    span = a.num_tokens - start
    assert booked == (span, span, None, a.num_tokens, span, 1)
    # Usage is the committed reply: the forward's sample is never appended.
    assert list(a.output_token_ids) == [FINAL] and a.num_prompt_tokens == len(PROMPT)
    assert not scheduler.requests and not scheduler._minicpmo45_terminal_rows
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


def test_ordinary_readmission_keeps_its_draft_slots():
    scheduler = _scheduler(True)
    _stale_final_token(scheduler)
    readmit = scheduler.schedule()
    _deliver(scheduler, readmit, _booked(scheduler, readmit), {"A": [DRAFT_ID]})
    assert not scheduler.requests

    # A request that is not a pending terminal row, hitting up to its last token.
    c = _request("C", [*PROMPT, FINAL], 3)
    scheduler.add_request(c)
    step = scheduler.schedule()
    start, _ = _booked(scheduler, step)["C"]
    assert start == c.num_tokens - 1
    assert step.num_scheduled_tokens["C"] == 1 + NUM_SPEC
    assert step.scheduled_spec_decode_tokens["C"] == [-1] * NUM_SPEC


class _TrimmingRunner:
    """Rows as the MRv2 stage-0 prefix cache hands them on: the real delivery adapter
    sees every step, and its trim drops already-delivered leading rows of a step's
    slice the way PrefixCacheManager.materialize does (current[start + trim:end])."""

    def __init__(self, prefix_caching):
        self.adapter = PrefixCacheSchedulerAdapter(track_requests=True) if prefix_caching else None

    def rows(self, scheduler, scheduler_output):
        spans = _booked(scheduler, scheduler_output)
        if self.adapter is None:
            return spans
        trims = {d.req_id: d.trim for d in self.adapter.translate_step(scheduler_output).deliveries}
        return {rid: (start + trims.get(rid, 0), ids[trims.get(rid, 0) :]) for rid, (start, ids) in spans.items()}


@pytest.mark.parametrize("prefix_caching", [True, False], ids=["prefix_hit", "cold"])
@pytest.mark.parametrize("stale_row", [True, False], ids=["stale_row_emitted", "stale_row_dropped"])
def test_recompute_after_the_emitted_own_row(prefix_caching, stale_row):
    """Drafts off: the terminal forward is preempted in flight and its stale output returns.

    With the stage-0 prefix cache the recompute carries no rows (all were delivered);
    the stop holds only if the stale output ended at the own row.
    """
    scheduler = _scheduler(prefix_caching, num_spec=0)
    runner = _TrimmingRunner(prefix_caching)
    a = _request("A", PROMPT, 2)
    scheduler.add_request(a)
    for sampled in (111, FINAL):
        step = scheduler.schedule()
        _deliver(scheduler, step, runner.rows(scheduler, step), {"A": [sampled]})
    assert "A" in scheduler._minicpmo45_terminal_rows and a.num_tokens == len(PROMPT) + 2

    terminal = scheduler.schedule()
    terminal_rows = runner.rows(scheduler, terminal)
    assert terminal_rows["A"][0] == a.num_tokens - 1
    scheduler.running.remove(a)
    scheduler._preempt_request(a, 0.0)
    start, ids = terminal_rows["A"]
    _, finishes = _deliver(scheduler, terminal, {"A": (start, ids if stale_row else [])}, {"A": [DRAFT_ID]})
    assert finishes == [] and a.status == RequestStatus.PREEMPTED

    readmit = scheduler.schedule()
    start, ids = _booked(scheduler, readmit)["A"]
    assert start == (len(PROMPT) if prefix_caching else 0)
    rows = runner.rows(scheduler, readmit)["A"]
    # The real adapter trims every recomputed row only under the prefix cache.
    assert (rows[1] == []) is prefix_caching
    payloads, finishes = _deliver(scheduler, readmit, {"A": rows}, {"A": [DRAFT_ID]})
    expected = FinishReason.LENGTH if (stale_row or not prefix_caching) else FinishReason.ERROR
    assert finishes == [("A", expected)]
    assert list(a.output_token_ids) == [111, FINAL]
    assert not scheduler.requests and not scheduler._minicpmo45_terminal_rows
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


@pytest.mark.parametrize(
    "damage",
    ["stale_row_other_position", "empty_without_prefix_cache", "empty_without_output_cache", "reused_request_id"],
)
def test_empty_recompute_is_accepted_only_after_this_rows_emission(damage, monkeypatch):
    """The record is tied to the stop's token and position, to the prefix-cache trim, and to one request."""
    prefix_caching = damage != "empty_without_prefix_cache"
    scheduler = _scheduler(prefix_caching, num_spec=0)
    if damage == "empty_without_output_cache":
        # Native KV caching on, output cache off (as with a stage-0 draft): no worker trim.
        monkeypatch.setattr(handoff, "caches_turn_outputs", lambda vllm_config: False)
        assert scheduler.kv_cache_manager.enable_caching
    trims = prefix_caching and damage != "empty_without_output_cache"
    runner = _TrimmingRunner(trims)

    def cycle(stale_row_ok):
        a = _request("A", PROMPT, 2)
        scheduler.add_request(a)
        for sampled in (111, FINAL):
            step = scheduler.schedule()
            _deliver(scheduler, step, runner.rows(scheduler, step), {"A": [sampled]})
        terminal = scheduler.schedule()
        start, ids = runner.rows(scheduler, terminal)["A"]
        scheduler.running.remove(a)
        scheduler._preempt_request(a, 0.0)
        if not stale_row_ok:
            start, ids = start - 1, ids[:1]  # the final token's id one position early: not its own row
        _deliver(scheduler, terminal, {"A": (start, ids)}, {"A": [DRAFT_ID]})
        readmit = scheduler.schedule()
        booked = runner.rows(scheduler, readmit)["A"]
        # Without the worker trim, force the empty ledger the rule must not accept there.
        rows = booked if trims else (booked[0] + len(booked[1]), [])
        _, finishes = _deliver(scheduler, readmit, {"A": rows}, {"A": [DRAFT_ID]})
        return finishes

    if damage == "stale_row_other_position":
        assert cycle(stale_row_ok=False) == [("A", FinishReason.ERROR)]
    elif damage in ("empty_without_prefix_cache", "empty_without_output_cache"):
        assert cycle(stale_row_ok=True) == [("A", FinishReason.ERROR)]
    else:
        # The first request's record must not carry over to a new request with the same id.
        assert cycle(stale_row_ok=True) == [("A", FinishReason.LENGTH)]
        assert not scheduler._minicpmo45_terminal_rows
        assert cycle(stale_row_ok=False) == [("A", FinishReason.ERROR)]
    assert not scheduler.requests and not scheduler._minicpmo45_terminal_rows


@pytest.mark.parametrize(
    "stage,session,draft,expected",
    [("llm", "turn", False, True), ("llm", "turn", True, False), ("llm", "duplex", False, False),
     ("tts", "turn", False, False)],
)  # fmt: skip
def test_output_cache_opt_in(stage, session, draft, expected):
    """The trimmed-recompute rule and the model's output-cache opt-in read the same predicate."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage=stage, session_mode=session),
        speculative_config=_speculative_config() if draft else None,
    )
    assert caches_turn_outputs(config) is expected

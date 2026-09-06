# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the model-neutral CFG request-pairing scheduler patches.

The patches keep a cond/uncond request pair step-locked inside one engine:
admitted together, kept at equal ``num_computed_tokens``, finished together.
They are installed onto a lightweight fake scheduler here; vLLM's real
``Scheduler`` needs a full engine (and a GPU) to construct.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.model_executor.models.common import cfg_pairing
from vllm_omni.model_executor.models.common.cfg_pairing import (
    _equalize_cfg_pair_progress,
    _patch_scheduler,
    cfg_uncond_suffixes,
    register_cfg_uncond_suffix,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ───────── fakes ─────────


class _FakeRequest:
    """Minimal stand-in for ``vllm.v1.request.Request`` (identity equality)."""

    def __init__(self, request_id: str, extra_args: dict[str, Any] | None = None, step_tokens: int = 1) -> None:
        self.request_id = request_id
        self.sampling_params = SimpleNamespace(extra_args=extra_args)
        self.num_computed_tokens = 0
        self.step_tokens = step_tokens
        self.skip_reading_prefix_cache = False


class _FakeWaitingQueue(deque):
    """FCFS waiting queue: vLLM's is also a ``deque`` subclass."""

    def remove_requests(self, requests: list[_FakeRequest]) -> None:
        keep = [req for req in self if not any(req is dropped for dropped in requests)]
        self.clear()
        self.extend(keep)

    def prepend_request(self, request: _FakeRequest) -> None:
        self.appendleft(request)


@dataclass
class _FakeSchedulerOutput:
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0


def _make_scheduler_cls() -> type:
    """Fresh class per test so the in-place patches never stack."""

    class FakeScheduler:
        def __init__(self) -> None:
            self.waiting = _FakeWaitingQueue()
            self.running: list[_FakeRequest] = []
            self.requests: dict[str, _FakeRequest] = {}
            self.finished: list[set[str]] = []
            self.observed_waiting: list[list[str]] = []
            self.max_num_scheduled_tokens = 512
            self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=4096)

        def add_request(self, request: _FakeRequest) -> None:
            self.requests[request.request_id] = request
            self.waiting.append(request)

        def finish_requests(self, request_ids: Any, finished_status: Any) -> None:
            if request_ids is None:
                ids = set(self.requests)
            elif isinstance(request_ids, str):
                ids = {request_ids}
            else:
                ids = set(request_ids)
            self.finished.append(ids)
            self.running = [req for req in self.running if req.request_id not in ids]
            for req_id in ids:
                self.requests.pop(req_id, None)

        def schedule(self) -> _FakeSchedulerOutput:
            self.observed_waiting.append([req.request_id for req in self.waiting])
            while self.waiting:
                self.running.append(self.waiting.popleft())
            output = _FakeSchedulerOutput()
            for req in self.running:
                req.num_computed_tokens += req.step_tokens
                output.num_scheduled_tokens[req.request_id] = req.step_tokens
                output.total_num_scheduled_tokens += req.step_tokens
            return output

    return FakeScheduler


@pytest.fixture
def scheduler() -> Any:
    cls = _make_scheduler_cls()
    _patch_scheduler(cls)
    return cls()


@pytest.fixture(autouse=True)
def _restore_suffix_registry():
    original = set(cfg_pairing._UNCOND_REQUEST_ID_SUFFIXES)
    yield
    cfg_pairing._UNCOND_REQUEST_ID_SUFFIXES.clear()
    cfg_pairing._UNCOND_REQUEST_ID_SUFFIXES.update(original)


def _add_pair(
    scheduler: Any,
    pair_id: str | None = None,
    base_id: str = "req-1",
    suffix: str = "__cfg_uncond",
) -> tuple[_FakeRequest, _FakeRequest]:
    cond_extra: dict[str, Any] = {"cfg_role": "cond"}
    uncond_extra: dict[str, Any] = {"cfg_role": "uncond"}
    if pair_id is not None:
        cond_extra["cfg_pair_id"] = pair_id
        uncond_extra["cfg_pair_id"] = pair_id
    cond = _FakeRequest(base_id, cond_extra)
    uncond = _FakeRequest(f"{base_id}{suffix}", uncond_extra)
    scheduler.add_request(cond)
    scheduler.add_request(uncond)
    return cond, uncond


# ───────── pair-id derivation ─────────


def test_pair_id_is_derived_from_request_id_when_absent(scheduler: Any) -> None:
    cond, uncond = _add_pair(scheduler)

    assert scheduler._cfg_pairs == {"req-1": {"cond": cond.request_id, "uncond": uncond.request_id}}
    assert scheduler._cfg_req_to_pair == {cond.request_id: "req-1", uncond.request_id: "req-1"}


def test_registered_companion_suffix_is_stripped(scheduler: Any) -> None:
    register_cfg_uncond_suffix("::negative")
    assert "::negative" in cfg_uncond_suffixes()

    cond, uncond = _add_pair(scheduler, base_id="music-7", suffix="::negative")

    assert scheduler._cfg_pairs == {"music-7": {"cond": cond.request_id, "uncond": uncond.request_id}}


def test_register_rejects_empty_suffix() -> None:
    with pytest.raises(ValueError):
        register_cfg_uncond_suffix("")


def test_uncond_without_known_suffix_keeps_its_own_request_id(scheduler: Any) -> None:
    # Degenerate input: the companion never joins the cond pair, so the pair
    # stays incomplete and the hold budget eventually releases it.
    scheduler.add_request(_FakeRequest("solo-uncond", {"cfg_role": "uncond"}))

    assert scheduler._cfg_pairs == {"solo-uncond": {"uncond": "solo-uncond"}}


def test_explicit_pair_id_takes_precedence(scheduler: Any) -> None:
    cond, uncond = _add_pair(scheduler, pair_id="pair-A")

    assert scheduler._cfg_pairs == {"pair-A": {"cond": cond.request_id, "uncond": uncond.request_id}}
    assert cond.request_id not in scheduler._cfg_pairs
    assert scheduler._cfg_req_to_pair[uncond.request_id] == "pair-A"


def test_cfg_requests_skip_prefix_cache_reads(scheduler: Any) -> None:
    cond, uncond = _add_pair(scheduler)

    assert cond.skip_reading_prefix_cache is True
    assert uncond.skip_reading_prefix_cache is True


# ───────── admission holding ─────────


def test_lone_member_is_held_then_released(scheduler: Any) -> None:
    scheduler.add_request(_FakeRequest("req-1", {"cfg_role": "cond"}))

    scheduler.schedule()

    assert scheduler.observed_waiting == [[]]  # the base scheduler saw nothing
    assert [req.request_id for req in scheduler.waiting] == ["req-1"]
    assert scheduler.running == []


def test_complete_pair_is_admitted_together(scheduler: Any) -> None:
    scheduler.add_request(_FakeRequest("req-1", {"cfg_role": "cond"}))
    scheduler.schedule()
    scheduler.add_request(_FakeRequest("req-1__cfg_uncond", {"cfg_role": "uncond"}))

    scheduler.schedule()

    assert scheduler.observed_waiting[-1] == ["req-1", "req-1__cfg_uncond"]
    assert [req.request_id for req in scheduler.running] == ["req-1", "req-1__cfg_uncond"]
    assert list(scheduler.waiting) == []


def test_hold_budget_expiry_drops_the_stale_pair(scheduler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg_pairing, "_MAX_PAIR_HOLD_STEPS", 2)
    scheduler.add_request(_FakeRequest("req-1", {"cfg_role": "cond"}))

    scheduler.schedule()
    scheduler.schedule()
    assert scheduler.running == []
    assert scheduler._cfg_pairs  # still hoping for the partner

    scheduler.schedule()

    assert scheduler._cfg_pairs == {}
    assert scheduler._cfg_req_to_pair == {}
    assert [req.request_id for req in scheduler.running] == ["req-1"]


def test_waiting_queue_reorder_puts_partners_adjacent(scheduler: Any) -> None:
    scheduler.add_request(_FakeRequest("req-1", {"cfg_role": "cond"}))
    scheduler.add_request(_FakeRequest("other", None))
    scheduler.add_request(_FakeRequest("req-1__cfg_uncond", {"cfg_role": "uncond"}))

    scheduler.schedule()

    assert scheduler.observed_waiting[-1] == ["req-1", "req-1__cfg_uncond", "other"]


# ───────── progress equalization ─────────


def _equalize_fixture(
    cond_computed: int,
    uncond_computed: int,
    cond_sched: int,
    uncond_sched: int,
) -> tuple[Any, _FakeSchedulerOutput, _FakeRequest, _FakeRequest]:
    cond = _FakeRequest("req-1", {"cfg_role": "cond"})
    uncond = _FakeRequest("req-1__cfg_uncond", {"cfg_role": "uncond"})
    cond.num_computed_tokens = cond_computed
    uncond.num_computed_tokens = uncond_computed
    fake = SimpleNamespace(
        _cfg_pairs={"req-1": {"cond": cond.request_id, "uncond": uncond.request_id}},
        requests={cond.request_id: cond, uncond.request_id: uncond},
    )
    output = _FakeSchedulerOutput(
        num_scheduled_tokens={cond.request_id: cond_sched, uncond.request_id: uncond_sched},
        total_num_scheduled_tokens=cond_sched + uncond_sched,
    )
    return fake, output, cond, uncond


def test_equalize_pulls_the_ahead_member_back() -> None:
    fake, output, cond, uncond = _equalize_fixture(10, 6, 5, 4)

    _equalize_cfg_pair_progress(fake, output)

    assert cond.num_computed_tokens == 6
    assert uncond.num_computed_tokens == 6
    assert output.num_scheduled_tokens == {cond.request_id: 1, uncond.request_id: 4}
    assert output.total_num_scheduled_tokens == 5


def test_equalize_is_a_noop_when_the_pair_is_already_aligned() -> None:
    fake, output, cond, uncond = _equalize_fixture(8, 8, 4, 4)

    _equalize_cfg_pair_progress(fake, output)

    assert (cond.num_computed_tokens, uncond.num_computed_tokens) == (8, 8)
    assert output.total_num_scheduled_tokens == 8


def test_equalize_skips_an_infeasible_rollback() -> None:
    # Clawing back 4 tokens would leave the cond row with 0 scheduled tokens,
    # which the scheduler output cannot express; leave the pair to _drop_split_pairs.
    fake, output, cond, uncond = _equalize_fixture(10, 6, 4, 4)

    _equalize_cfg_pair_progress(fake, output)

    assert cond.num_computed_tokens == 10
    assert output.total_num_scheduled_tokens == 8


# ───────── finish ─────────


def test_finish_requests_finishes_the_partner(scheduler: Any) -> None:
    cond, uncond = _add_pair(scheduler)
    scheduler.schedule()

    scheduler.finish_requests(cond.request_id, "FINISHED_STOPPED")

    assert scheduler.finished == [{cond.request_id, uncond.request_id}]
    assert scheduler._cfg_pairs == {}
    assert scheduler._cfg_req_to_pair == {}
    assert scheduler.running == []


def test_finish_all_requests_clears_the_registry(scheduler: Any) -> None:
    _add_pair(scheduler)
    scheduler.schedule()

    scheduler.finish_requests(None, "FINISHED_ABORTED")

    assert scheduler._cfg_pairs == {}
    assert scheduler._cfg_req_to_pair == {}


# ───────── non-CFG traffic ─────────


def test_plain_requests_are_untouched(scheduler: Any) -> None:
    scheduler.add_request(_FakeRequest("plain-1", None))
    scheduler.add_request(_FakeRequest("plain-2", {"seed": 3}))

    output = scheduler.schedule()

    assert scheduler._cfg_pairs == {}
    assert scheduler.observed_waiting == [["plain-1", "plain-2"]]
    assert output.num_scheduled_tokens == {"plain-1": 1, "plain-2": 1}
    assert output.total_num_scheduled_tokens == 2
    # No pair ever existed, so the hold bookkeeping is never allocated.
    assert not hasattr(scheduler, "_cfg_hold_counts")


def test_default_suffix_matches_the_audex_stage_processor() -> None:
    from vllm_omni.model_executor.stage_input_processors.audex import CFG_UNCOND_SUFFIX

    assert CFG_UNCOND_SUFFIX in cfg_uncond_suffixes()


def test_audex_cfg_module_still_exports_the_patch_entry_points() -> None:
    from vllm_omni.model_executor.models.audex.cfg import apply_cfg_patches, cfg_patches_applied

    assert apply_cfg_patches is cfg_pairing.apply_cfg_patches
    assert cfg_patches_applied is cfg_pairing.cfg_patches_applied


# ───────── install gate ─────────


def test_gate_triggers_on_a_cfg_logits_processor() -> None:
    from vllm_omni.engine.stage_init_utils import _stage_declares_cfg_pairs

    class AudexCFGLogitsProcessor:
        pass

    assert _stage_declares_cfg_pairs(SimpleNamespace(logits_processors=[AudexCFGLogitsProcessor]))
    # vLLM keeps unresolved entries as dotted strings.
    assert _stage_declares_cfg_pairs(
        SimpleNamespace(logits_processors=["vllm_omni.model_executor.models.audex.cfg:AudexCFGLogitsProcessor"])
    )


def test_gate_triggers_on_a_declared_cfg_role_default() -> None:
    from vllm_omni.engine.stage_init_utils import _stage_declares_cfg_pairs

    assert _stage_declares_cfg_pairs(
        SimpleNamespace(logits_processors=None, sampling_extra_args_keys=("cfg_role", "cfg_scale"))
    )


def test_gate_stays_off_for_non_cfg_stages() -> None:
    from vllm_omni.engine.stage_init_utils import _stage_declares_cfg_pairs

    assert not _stage_declares_cfg_pairs(SimpleNamespace(logits_processors=[], sampling_extra_args_keys=()))
    assert not _stage_declares_cfg_pairs(
        SimpleNamespace(logits_processors=["some.module:MinPLogitsProcessor"], sampling_extra_args_keys=("tta_rvq",))
    )
    assert not _stage_declares_cfg_pairs(SimpleNamespace())


def test_sampling_extra_args_keys_are_derived_from_stage_defaults() -> None:
    from vllm_omni.engine.stage_init_utils import _sampling_extra_args_keys

    assert _sampling_extra_args_keys({"extra_args": {"cfg_scale": 1.5, "cfg_role": "cond"}}) == (
        "cfg_role",
        "cfg_scale",
    )
    assert _sampling_extra_args_keys({"temperature": 0.1}) == ()
    assert _sampling_extra_args_keys(None) == ()

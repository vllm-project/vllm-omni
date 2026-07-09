# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Audex CFG logits processor and scheduler pair helpers."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality

from vllm_omni.model_executor.models.audex import cfg as audex_cfg
from vllm_omni.model_executor.models.audex.cfg import (
    AudexCFGLogitsProcessor,
    _equalize_cfg_pair_progress,
    _hold_incomplete_pairs,
    _release_held,
    _reorder_waiting_for_cfg,
    apply_cfg_patches,
    cfg_patches_applied,
)

VOCAB = 16


def _params(role: str | None = None, scale: float = 1.5, pair_id: str = "pair-0") -> SamplingParams:
    if role is None:
        return SamplingParams()
    return SamplingParams(extra_args={"cfg_scale": scale, "cfg_role": role, "cfg_pair_id": pair_id})


def _processor() -> AudexCFGLogitsProcessor:
    return AudexCFGLogitsProcessor(vllm_config=None, device=torch.device("cpu"), is_pin_memory=False)


def _add(proc: AudexCFGLogitsProcessor, added, removed=(), moved=(), batch_size: int = 8) -> None:
    proc.update_state(BatchUpdate(batch_size=batch_size, removed=list(removed), added=list(added), moved=list(moved)))


class TestBlendMath:
    def test_blend_written_to_both_rows(self):
        proc = _processor()
        _add(proc, [(0, _params("cond"), None, []), (1, _params("uncond"), None, [])])
        logits = torch.arange(4 * VOCAB, dtype=torch.float32).reshape(4, VOCAB)
        cond, uncond = logits[0].clone(), logits[1].clone()
        untouched = logits[2:].clone()

        out = proc.apply(logits)

        expected = uncond + 1.5 * (cond - uncond)
        assert torch.equal(out[0], expected)
        assert torch.equal(out[1], expected)
        assert torch.equal(out[2:], untouched)

    def test_non_cfg_rows_untouched(self):
        proc = _processor()
        _add(proc, [(0, _params(), None, []), (1, _params(), None, [])])
        logits = torch.randn(2, VOCAB)
        reference = logits.clone()
        assert torch.equal(proc.apply(logits), reference)

    def test_scale_one_is_identity_blend(self):
        proc = _processor()
        _add(proc, [(0, _params("cond", scale=1.0), None, []), (1, _params("uncond", scale=1.0), None, [])])
        logits = torch.randn(2, VOCAB)
        cond = logits[0].clone()
        out = proc.apply(logits)
        assert torch.allclose(out[0], cond)
        assert torch.allclose(out[1], cond)

    def test_one_sided_pair_does_not_blend(self):
        proc = _processor()
        _add(proc, [(0, _params("cond"), None, [])])
        logits = torch.randn(2, VOCAB)
        reference = logits.clone()
        assert torch.equal(proc.apply(logits), reference)


class TestBatchUpdateLifecycle:
    def test_removed_row_stops_blending(self):
        proc = _processor()
        _add(proc, [(0, _params("cond"), None, []), (1, _params("uncond"), None, [])])
        _add(proc, [], removed=[0])
        logits = torch.randn(2, VOCAB)
        reference = logits.clone()
        assert torch.equal(proc.apply(logits), reference)

    def test_unidirectional_move_follows_row(self):
        proc = _processor()
        _add(proc, [(0, _params("cond"), None, []), (1, _params("uncond"), None, [])])
        _add(proc, [], moved=[(1, 3, MoveDirectionality.UNIDIRECTIONAL)])
        logits = torch.randn(4, VOCAB)
        cond, uncond = logits[0].clone(), logits[3].clone()
        out = proc.apply(logits)
        expected = uncond + 1.5 * (cond - uncond)
        assert torch.equal(out[0], expected)
        assert torch.equal(out[3], expected)

    def test_swap_move_keeps_pairing(self):
        proc = _processor()
        _add(
            proc,
            [
                (0, _params("cond"), None, []),
                (1, _params("uncond"), None, []),
                (2, _params(), None, []),
            ],
        )
        _add(proc, [], moved=[(0, 2, MoveDirectionality.SWAP)])
        logits = torch.randn(3, VOCAB)
        cond, uncond = logits[2].clone(), logits[1].clone()
        out = proc.apply(logits)
        expected = uncond + 1.5 * (cond - uncond)
        assert torch.equal(out[2], expected)
        assert torch.equal(out[1], expected)
        assert torch.equal(out[0], logits[0])

    def test_added_replacing_cfg_row_clears_it(self):
        proc = _processor()
        _add(proc, [(0, _params("cond"), None, []), (1, _params("uncond"), None, [])])
        _add(proc, [(0, _params(), None, [])])
        logits = torch.randn(2, VOCAB)
        reference = logits.clone()
        assert torch.equal(proc.apply(logits), reference)


class TestValidateParams:
    def test_valid_pair_args_accepted(self):
        AudexCFGLogitsProcessor.validate_params(_params("cond"))
        AudexCFGLogitsProcessor.validate_params(_params("uncond"))
        AudexCFGLogitsProcessor.validate_params(_params())

    @pytest.mark.parametrize("role", ["conditional", "negative", ""])
    def test_bad_role_rejected(self, role):
        params = SamplingParams(extra_args={"cfg_role": role})
        with pytest.raises(ValueError, match="cfg_role"):
            AudexCFGLogitsProcessor.validate_params(params)

    @pytest.mark.parametrize("scale", [0.5, 0.0, -1.0, "big"])
    def test_bad_scale_rejected(self, scale):
        params = SamplingParams(extra_args={"cfg_scale": scale})
        with pytest.raises(ValueError, match="cfg_scale"):
            AudexCFGLogitsProcessor.validate_params(params)


class TestSamplePatch:
    def test_ar_runner_sample_patched_on_init(self):
        _processor()
        from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

        assert getattr(GPUARModelRunner._sample, "_audex_cfg_sync", False)


class _FakeRequest:
    """Hashable stand-in (FCFSRequestQueue.remove_requests builds a set)."""

    def __init__(self, request_id: str, num_computed_tokens: int = 0):
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens


def _fake_request(request_id: str, num_computed_tokens: int = 0):
    return _FakeRequest(request_id, num_computed_tokens)


def _fake_scheduler(pairs: dict[str, dict[str, str]], admitted: dict[str, object]):
    scheduler = SimpleNamespace(
        waiting=FCFSRequestQueue(),
        skipped_waiting=FCFSRequestQueue(),
        running=[],
        requests=admitted,
        _cfg_pairs=pairs,
        _cfg_req_to_pair={rid: pid for pid, roles in pairs.items() for rid in roles.values()},
    )
    return scheduler


class TestPairHold:
    def test_lone_member_held_until_partner_admitted(self):
        cond = _fake_request("req-cond")
        scheduler = _fake_scheduler({"p0": {"cond": "req-cond"}}, {"req-cond": cond})
        scheduler.waiting.append(cond)

        held = _hold_incomplete_pairs(scheduler)

        assert [req.request_id for _, req in held] == ["req-cond"]
        assert len(scheduler.waiting) == 0
        _release_held(held)
        assert [req.request_id for req in scheduler.waiting] == ["req-cond"]

    def test_complete_pair_not_held(self):
        cond, uncond = _fake_request("c"), _fake_request("u")
        scheduler = _fake_scheduler({"p0": {"cond": "c", "uncond": "u"}}, {"c": cond, "u": uncond})
        scheduler.waiting.append(cond)
        scheduler.waiting.append(uncond)

        assert _hold_incomplete_pairs(scheduler) == []
        assert len(scheduler.waiting) == 2

    def test_lone_member_in_skipped_waiting_held(self):
        cond = _fake_request("req-cond")
        scheduler = _fake_scheduler({"p0": {"cond": "req-cond"}}, {"req-cond": cond})
        scheduler.skipped_waiting.append(cond)

        held = _hold_incomplete_pairs(scheduler)

        assert [req.request_id for _, req in held] == ["req-cond"]
        assert len(scheduler.skipped_waiting) == 0
        _release_held(held)
        assert [req.request_id for req in scheduler.skipped_waiting] == ["req-cond"]

    def test_non_cfg_requests_never_held(self):
        plain = _fake_request("plain")
        scheduler = _fake_scheduler({}, {"plain": plain})
        scheduler.waiting.append(plain)
        assert _hold_incomplete_pairs(scheduler) == []

    def test_stale_pair_released_unguided_after_hold_budget(self):
        cond = _fake_request("req-cond")
        scheduler = _fake_scheduler({"p0": {"cond": "req-cond"}}, {"req-cond": cond})
        scheduler.waiting.append(cond)

        for _ in range(audex_cfg._MAX_PAIR_HOLD_STEPS):
            held = _hold_incomplete_pairs(scheduler)
            assert [req.request_id for _, req in held] == ["req-cond"]
            _release_held(held)

        # Budget exhausted: the request is released unguided and the pair
        # registration is dropped so it never blocks again.
        assert _hold_incomplete_pairs(scheduler) == []
        assert scheduler._cfg_pairs == {}
        assert scheduler._cfg_req_to_pair == {}
        assert [req.request_id for req in scheduler.waiting] == ["req-cond"]

    def test_partner_arrival_resets_hold_budget(self):
        cond = _fake_request("c")
        scheduler = _fake_scheduler({"p0": {"cond": "c"}}, {"c": cond})
        scheduler.waiting.append(cond)
        _release_held(_hold_incomplete_pairs(scheduler))
        assert scheduler._cfg_hold_counts["c"] == 1

        uncond = _fake_request("u")
        scheduler._cfg_pairs["p0"]["uncond"] = "u"
        scheduler._cfg_req_to_pair["u"] = "p0"
        scheduler.requests["u"] = uncond
        scheduler.waiting.append(uncond)

        assert _hold_incomplete_pairs(scheduler) == []
        assert "c" not in scheduler._cfg_hold_counts


class TestReorder:
    def test_partners_made_adjacent(self):
        cond, uncond = _fake_request("c"), _fake_request("u")
        other = _fake_request("x")
        scheduler = _fake_scheduler({"p0": {"cond": "c", "uncond": "u"}}, {"c": cond, "u": uncond})
        for req in (cond, other, uncond):
            scheduler.waiting.append(req)

        _reorder_waiting_for_cfg(scheduler)

        assert [req.request_id for req in scheduler.waiting] == ["c", "u", "x"]


class TestEqualize:
    def test_faster_member_pulled_back(self):
        cond = _fake_request("c", num_computed_tokens=12)
        uncond = _fake_request("u", num_computed_tokens=10)
        scheduler = _fake_scheduler({"p0": {"cond": "c", "uncond": "u"}}, {"c": cond, "u": uncond})
        output = SimpleNamespace(
            num_scheduled_tokens={"c": 4, "u": 4},
            total_num_scheduled_tokens=8,
        )

        _equalize_cfg_pair_progress(scheduler, output)

        assert cond.num_computed_tokens == 10
        assert uncond.num_computed_tokens == 10
        assert output.num_scheduled_tokens == {"c": 2, "u": 4}
        assert output.total_num_scheduled_tokens == 6

    def test_infeasible_gap_left_alone(self):
        cond = _fake_request("c", num_computed_tokens=20)
        uncond = _fake_request("u", num_computed_tokens=10)
        scheduler = _fake_scheduler({"p0": {"cond": "c", "uncond": "u"}}, {"c": cond, "u": uncond})
        output = SimpleNamespace(
            num_scheduled_tokens={"c": 4, "u": 4},
            total_num_scheduled_tokens=8,
        )

        _equalize_cfg_pair_progress(scheduler, output)

        assert cond.num_computed_tokens == 20
        assert output.total_num_scheduled_tokens == 8


class TestStageInitHook:
    def test_non_cfg_engine_config_is_untouched(self):
        from vllm_omni.engine.stage_init_utils import maybe_apply_audex_cfg_patches

        maybe_apply_audex_cfg_patches(None)
        maybe_apply_audex_cfg_patches(SimpleNamespace(model_config=SimpleNamespace(logits_processors=None)))
        maybe_apply_audex_cfg_patches(
            SimpleNamespace(model_config=SimpleNamespace(logits_processors=["some.other.Processor"]))
        )

    def test_cfg_engine_config_applies_and_asserts(self):
        from vllm_omni.engine.stage_init_utils import maybe_apply_audex_cfg_patches

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                logits_processors=["vllm_omni.model_executor.models.audex.cfg.AudexCFGLogitsProcessor"]
            )
        )
        maybe_apply_audex_cfg_patches(config)
        assert cfg_patches_applied()

        from vllm.v1.core.sched.scheduler import Scheduler

        assert getattr(Scheduler.schedule, "_audex_cfg_patched", False)

    def test_cfg_engine_config_accepts_class_entries(self):
        from vllm_omni.engine.stage_init_utils import maybe_apply_audex_cfg_patches

        config = SimpleNamespace(model_config=SimpleNamespace(logits_processors=[AudexCFGLogitsProcessor]))
        maybe_apply_audex_cfg_patches(config)
        assert cfg_patches_applied()


class TestApplyPatches:
    def test_patches_idempotent_and_marked(self):
        apply_cfg_patches()
        assert cfg_patches_applied()

        from vllm.v1.core.sched.scheduler import Scheduler

        assert getattr(Scheduler.schedule, "_audex_cfg_patched", False)
        patched = Scheduler.schedule
        apply_cfg_patches()
        assert Scheduler.schedule is patched

    def test_module_flag_tracks_state(self):
        assert audex_cfg._patches_applied == cfg_patches_applied()


class TestReviewHardening:
    """Fixes from the adversarial 0.24 scheduler review."""

    def test_partner_in_skipped_waiting_holds_waiting_member(self):
        cond, uncond = _fake_request("c"), _fake_request("u")
        scheduler = _fake_scheduler({"p0": {"cond": "c", "uncond": "u"}}, {"c": cond, "u": uncond})
        scheduler.waiting.append(cond)
        scheduler.skipped_waiting.append(uncond)

        held = _hold_incomplete_pairs(scheduler)

        held_ids = {req.request_id for _, req in held}
        assert "c" in held_ids, "waiting member must hold while its partner is parked in skipped_waiting"

    def test_split_pair_dropped_and_released_unguided(self):
        from vllm_omni.model_executor.models.audex.cfg import _drop_split_pairs

        cond = _fake_request("c", num_computed_tokens=40)
        uncond = _fake_request("u", num_computed_tokens=0)
        scheduler = _fake_scheduler({"p0": {"cond": "c", "uncond": "u"}}, {"c": cond, "u": uncond})
        scheduler.running = [cond]  # uncond preempted back to waiting

        _drop_split_pairs(scheduler)

        assert scheduler._cfg_pairs == {}
        assert scheduler._cfg_req_to_pair == {}

    def test_prefill_only_split_not_dropped(self):
        from vllm_omni.model_executor.models.audex.cfg import _drop_split_pairs

        cond = _fake_request("c", num_computed_tokens=0)
        uncond = _fake_request("u", num_computed_tokens=0)
        scheduler = _fake_scheduler({"p0": {"cond": "c", "uncond": "u"}}, {"c": cond, "u": uncond})
        scheduler.running = [cond]  # partner merely not admitted yet

        _drop_split_pairs(scheduler)

        assert "p0" in scheduler._cfg_pairs


class TestPartialStepRobustness:
    """Pairs indexed beyond a step's rows must be skipped, not crash."""

    def test_blend_skips_rows_beyond_step(self):
        proc = _processor()
        _add(proc, [(2, _params("cond"), None, []), (3, _params("uncond"), None, [])])
        logits = torch.randn(2, VOCAB)  # partially scheduled step: 2 rows only
        reference = logits.clone()
        assert torch.equal(proc.apply(logits), reference)

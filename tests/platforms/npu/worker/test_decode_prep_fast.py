# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the steady-state decode input reuse gates.

Every one of these runs on CPU with duck-typed stand-ins: the gates read plain
attributes off the runner, the scheduler output and the input batch, so none of
them needs an accelerator, a model or vLLM's engine.

The property under test throughout is the same one: **a gate that cannot prove
the step is a plain unchanged decode must return a reason, and the caller then
takes the generic path.** A false "yes" is a wrong step; a false "no" is only a
slow one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

from vllm_omni.platforms.npu.worker import decode_prep_fast  # noqa: E402


def _input_batch(*, num_reqs=1, req_ids=("r0",), computed=None, **overrides):
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=list(req_ids),
        prev_sampled_token_ids=object(),
        prev_req_id_to_index={"r0": 0},
        req_prompt_embeds=None,
        num_computed_tokens_cpu=np.array(computed if computed is not None else [7], dtype=np.int32),
        block_table=SimpleNamespace(block_tables=[]),
    )
    for key, value in overrides.items():
        setattr(batch, key, value)
    return batch


def _runner(**overrides):
    runner = SimpleNamespace(
        input_batch=_input_batch(),
        calculate_kv_scales=False,
        num_accepted_tokens_event=None,
        routed_experts_initialized=False,
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def _scheduler_output(**overrides):
    out = SimpleNamespace(scheduled_spec_decode_tokens={}, total_num_scheduled_tokens=1)
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


class TestStepGate:
    """`_step_gate_block` decides whether *this* step is a plain decode."""

    def test_a_plain_one_token_decode_is_allowed(self):
        assert decode_prep_fast._step_gate_block(_runner(), _scheduler_output(), np.array([1], dtype=np.int32)) is None

    @pytest.mark.parametrize(
        ("overrides", "reason"),
        [
            ({"calculate_kv_scales": True}, "kv_scales"),
            ({"num_accepted_tokens_event": object()}, "accepted_tokens_event"),
            ({"routed_experts_initialized": True}, "routed_experts"),
        ],
    )
    def test_runner_state_that_the_fast_path_does_not_reproduce_blocks(self, overrides, reason):
        assert (
            decode_prep_fast._step_gate_block(_runner(**overrides), _scheduler_output(), np.array([1], dtype=np.int32))
            == reason
        )

    def test_a_speculative_step_blocks(self):
        blocked = decode_prep_fast._step_gate_block(
            _runner(), _scheduler_output(scheduled_spec_decode_tokens={"r0": [5]}), np.array([1], dtype=np.int32)
        )
        assert blocked == "spec_decode_tokens"

    def test_more_than_one_token_for_a_request_blocks(self):
        blocked = decode_prep_fast._step_gate_block(
            _runner(), _scheduler_output(total_num_scheduled_tokens=3), np.array([3], dtype=np.int32)
        )
        assert blocked == "not_one_token_per_request"

    def test_a_request_still_on_its_prompt_blocks(self):
        """`num_computed_tokens == 0` means this request is prefilling, not decoding."""
        runner = _runner(input_batch=_input_batch(computed=[0]))
        blocked = decode_prep_fast._step_gate_block(runner, _scheduler_output(), np.array([1], dtype=np.int32))
        assert blocked == "prefill_in_batch"

    def test_a_batch_with_no_previous_sample_blocks(self):
        """There is nothing to scatter the new token from on the first step."""
        runner = _runner(input_batch=_input_batch(prev_sampled_token_ids=None))
        blocked = decode_prep_fast._step_gate_block(runner, _scheduler_output(), np.array([1], dtype=np.int32))
        assert blocked == "no_prev_sampled_token_ids"

    def test_an_empty_batch_blocks(self):
        runner = _runner(input_batch=_input_batch(num_reqs=0))
        blocked = decode_prep_fast._step_gate_block(
            runner, _scheduler_output(total_num_scheduled_tokens=0), np.array([], dtype=np.int32)
        )
        assert blocked == "no_requests"


class TestBlockTableSnapshot:
    """A decode allocates a block once per `block_size` tokens; the other steps
    re-upload a table the device already has."""

    @staticmethod
    def _runner_with_rows(rows):
        table = SimpleNamespace(block_table=SimpleNamespace(np=rows))
        return SimpleNamespace(input_batch=SimpleNamespace(block_table=SimpleNamespace(block_tables=[table])))

    def test_the_first_call_records_rather_than_claims_unchanged(self):
        cache = decode_prep_fast.DecodeInputCache()
        runner = self._runner_with_rows(np.arange(8, dtype=np.int32).reshape(1, 8))
        assert decode_prep_fast._block_tables_unchanged(runner, cache, 1) is False
        assert cache.block_table_snapshot is not None

    def test_an_identical_table_is_recognised(self):
        cache = decode_prep_fast.DecodeInputCache()
        runner = self._runner_with_rows(np.arange(8, dtype=np.int32).reshape(1, 8))
        decode_prep_fast._block_tables_unchanged(runner, cache, 1)
        assert decode_prep_fast._block_tables_unchanged(runner, cache, 1) is True

    def test_a_newly_allocated_block_is_noticed(self):
        cache = decode_prep_fast.DecodeInputCache()
        rows = np.arange(8, dtype=np.int32).reshape(1, 8)
        runner = self._runner_with_rows(rows)
        decode_prep_fast._block_tables_unchanged(runner, cache, 1)
        rows[0, 4] = 99  # the step that crosses a block boundary
        assert decode_prep_fast._block_tables_unchanged(runner, cache, 1) is False

    def test_the_snapshot_is_a_copy_not_a_view(self):
        """Holding a view would compare the live buffer against itself and
        report 'unchanged' for a table that had in fact moved."""
        cache = decode_prep_fast.DecodeInputCache()
        rows = np.arange(8, dtype=np.int32).reshape(1, 8)
        runner = self._runner_with_rows(rows)
        decode_prep_fast.snapshot_block_tables(runner, cache, 1)
        rows[0, 0] = 12345
        assert decode_prep_fast._block_tables_unchanged(runner, cache, 1) is False

    def test_an_unreadable_layout_keeps_committing_every_step(self):
        cache = decode_prep_fast.DecodeInputCache()
        table = SimpleNamespace(block_table=SimpleNamespace(np=None))
        runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=SimpleNamespace(block_tables=[table])))
        assert decode_prep_fast._block_tables_unchanged(runner, cache, 1) is False
        assert cache.block_table_snapshot is None


class TestSignatureAndCache:
    def test_the_signature_tracks_the_request_set(self):
        one = decode_prep_fast.signature_of(_runner(input_batch=_input_batch(num_reqs=1, req_ids=("r0",))))
        same = decode_prep_fast.signature_of(_runner(input_batch=_input_batch(num_reqs=1, req_ids=("r0",))))
        other = decode_prep_fast.signature_of(_runner(input_batch=_input_batch(num_reqs=1, req_ids=("r1",))))
        two = decode_prep_fast.signature_of(_runner(input_batch=_input_batch(num_reqs=2, req_ids=("r0", "r1"))))
        assert one == same
        assert one != other
        assert one != two

    def test_invalidate_drops_the_signature(self):
        """Anything that rewrites the shared buffers behind `_prepare_inputs`
        has to make the next step ineligible."""
        cache = decode_prep_fast.DecodeInputCache()
        cache.signature = (1, ("r0",))
        cache.block_table_snapshot = [np.zeros(4, dtype=np.int32)]
        cache.invalidate()
        assert cache.signature is None

    def test_the_cache_holds_no_torch_tensors(self):
        """A tensor created under `inference_mode` cannot be carried into a
        later step, so nothing cached here may be one."""
        import torch

        cache = decode_prep_fast.DecodeInputCache()
        cache.signature = (1, ("r0",))
        cache.cu_num_tokens = np.zeros(2, dtype=np.int32)
        cache.num_tokens_np = np.ones(1, dtype=np.int32)
        cache.block_table_snapshot = [np.zeros((1, 4), dtype=np.int32)]
        for slot in decode_prep_fast.DecodeInputCache.__slots__:
            value = getattr(cache, slot)
            values = value if isinstance(value, list) else [value]
            assert not any(isinstance(v, torch.Tensor) for v in values), slot

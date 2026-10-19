# SPDX-License-Identifier: Apache-2.0
"""Runner-side validation tests for the multi-step decode window.

``validate_multi_step_plan`` re-checks every window invariant against
runner-local state before the K-step replay runs (fail-closed: any
refusal falls back to the normal single-step path).  These tests drive
the validator with duck-typed fakes, no NPU device required: the module
under test is loaded from source with stubbed ``vllm_ascend`` imports so
the file also collects on the CUDA CPU lane (see ``_load_window_module``).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_WINDOW_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_omni"
    / "platforms"
    / "npu"
    / "worker"
    / "multi_step_decode.py"
)


def _install_vllm_ascend_stubs() -> None:
    """Stub the ``vllm_ascend`` attributes the window module needs.

    The module top-level-imports the Ascend forward context and the rotary
    cos/sin cache, and its package ``vllm_omni/platforms/npu/__init__.py``
    imports ``vllm_ascend.platform``.  None of them exist on the CUDA CPU
    images that collect ``tests/worker`` for the PR/merge lane, where a
    collection ImportError fails the whole job before markers can deselect.
    A real install wins (``setdefault``), so NPU hosts are unaffected.
    """
    try:
        import vllm_ascend  # noqa: F401
    except ImportError:
        pass
    else:
        return

    class _NPUPlatform:
        """Stand-in base class for ``vllm_omni.platforms.npu.platform``."""

    ascend = types.ModuleType("vllm_ascend")
    forward_ctx = types.ModuleType("vllm_ascend.ascend_forward_context")
    forward_ctx.set_ascend_forward_context = lambda *args, **kwargs: None
    platform = types.ModuleType("vllm_ascend.platform")
    platform.NPUPlatform = _NPUPlatform
    ops = types.ModuleType("vllm_ascend.ops")
    rotary = types.ModuleType("vllm_ascend.ops.rotary_embedding")
    rotary.update_cos_sin = lambda *args, **kwargs: None
    rotary._cos = None
    rotary._sin = None
    rotary._cos_sin_cache = None
    ascend.ascend_forward_context = forward_ctx
    ascend.platform = platform
    ascend.ops = ops
    ops.rotary_embedding = rotary
    for name, module in (
        ("vllm_ascend", ascend),
        ("vllm_ascend.ascend_forward_context", forward_ctx),
        ("vllm_ascend.platform", platform),
        ("vllm_ascend.ops", ops),
        ("vllm_ascend.ops.rotary_embedding", rotary),
    ):
        sys.modules.setdefault(name, module)


def _load_window_module():
    """Load the NPU window module from source.

    Importing it as ``vllm_omni.platforms.npu.worker.multi_step_decode`` would
    execute ``vllm_omni/platforms/npu/__init__.py`` and therefore need the whole
    NPU platform stack.  Source loading is the established pattern for NPU unit
    tests on CPU lanes (see ``tests/platforms/npu/test_310p_patches.py``).
    """
    _install_vllm_ascend_stubs()
    spec = importlib.util.spec_from_file_location(
        "vllm_omni_npu_multi_step_decode_under_test", _WINDOW_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {_WINDOW_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


msd_module = _load_window_module()
validate_multi_step_plan = msd_module.validate_multi_step_plan



@pytest.fixture(autouse=True)
def _stub_parallel_groups(monkeypatch):
    """Single-process stand-ins for the distributed state accessors."""
    group = SimpleNamespace(world_size=1, is_last_rank=True)
    monkeypatch.setattr(msd_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(msd_module, "get_tp_group", lambda: group)


def make_scheduler_output(req_ids, *, window_k=8, scheduled=None):
    scheduled = scheduled or {rid: window_k for rid in req_ids}
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_scheduled_tokens=scheduled,
        multi_step_plan={rid: window_k for rid in req_ids},
    )


def make_runner(req_ids, *, model=None, input_batch=None, **overrides):
    num_reqs = len(req_ids)
    runner = SimpleNamespace(
        use_async_scheduling=True,
        num_spec_tokens=0,
        speculative_config=None,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
        ),
        pcp_size=1,
        dcp_size=1,
        use_cp=False,
        lora_config=None,
        is_pooling_model=False,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            enable_return_routed_experts=False,
            enforce_eager=False,
        ),
        supports_mm_inputs=False,
        omni_prefix_cache=None,
        cache_config=SimpleNamespace(mamba_cache_mode="off"),
        calculate_kv_scales=False,
        dynamic_eplb=False,
        debugger=None,
        ascend_config=SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(enabled=False),
        ),
        model=model or make_model(),
        input_batch=input_batch or make_input_batch(req_ids),
        _resolve_duplex_sampling_hook=lambda: None,
        num_prompt_logprobs=0,
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def make_model(**overrides):
    model = SimpleNamespace(
        has_preprocess=True,
        make_omni_output=lambda *a, **k: None,
        prefer_model_sampler=False,
    )
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def make_input_batch(req_ids):
    num_reqs = len(req_ids)
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_reqs=num_reqs,
        num_computed_tokens_cpu=np.full(num_reqs, 200, dtype=np.int32),
        num_prompt_tokens=np.full(num_reqs, 100, dtype=np.int32),
        sampling_metadata=SimpleNamespace(
            no_penalties=True,
            max_num_logprobs=None,
        ),
        bad_words_token_ids=None,
        no_allowed_token_ids=True,
        logprob_token_ids=None,
    )


def test_validate_accepts_eligible_window():
    runner = make_runner(["r0", "r1"])
    output = make_scheduler_output(["r0", "r1"])
    assert validate_multi_step_plan(runner, output) == {"r0": 8, "r1": 8}


def test_validate_refuses_missing_or_empty_plan():
    runner = make_runner(["r0"])
    output = make_scheduler_output(["r0"])
    output.multi_step_plan = {}
    assert validate_multi_step_plan(runner, output) is None
    output.multi_step_plan = None
    assert validate_multi_step_plan(runner, output) is None


def test_validate_refuses_non_uniform_window():
    runner = make_runner(["r0", "r1"])
    output = make_scheduler_output(["r0", "r1"])
    output.multi_step_plan = {"r0": 8, "r1": 4}
    output.num_scheduled_tokens = {"r0": 8, "r1": 4}
    assert validate_multi_step_plan(runner, output) is None


def test_validate_refuses_mismatched_schedule():
    # num_scheduled_tokens disagrees with the plan (scheduler bug or a
    # rewritten output that was not meant for this runner).
    runner = make_runner(["r0"])
    output = make_scheduler_output(["r0"])
    output.num_scheduled_tokens = {"r0": 1}
    assert validate_multi_step_plan(runner, output) is None


def test_validate_accepts_sync_scheduling():
    # The window executor supports the sync scheduler (one window in flight,
    # step-0 input from the request-local output chain); no sync refusal.
    runner = make_runner(["r0"], use_async_scheduling=False)
    output = make_scheduler_output(["r0"])
    assert validate_multi_step_plan(runner, output) == {"r0": 8}


def test_validate_refuses_spec_decode():
    runner = make_runner(["r0"], num_spec_tokens=2)
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is None


def test_validate_refuses_model_without_contract():
    runner = make_runner(["r0"], model=make_model(has_preprocess=False))
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is None


def test_validate_refuses_penalties():
    input_batch = make_input_batch(["r0"])
    input_batch.sampling_metadata.no_penalties = False
    runner = make_runner(["r0"], input_batch=input_batch)
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is None


def test_accepts_stale_input_batch_views():
    # The runner-side validation must NOT re-check per-step mutable state:
    # under async scheduling the input_batch view lags the scheduler (the
    # plan is validated before _update_states syncs it), so a stale view
    # would produce false refusals.  A refusal after the scheduler planned
    # follow-up windows on top of this one is not recoverable -- the
    # follow-up plans' KV positions would skip this window's reserved
    # slots.  Phase and batch composition are gated scheduler-side with
    # authoritative state instead.
    input_batch = make_input_batch(["r0"])
    input_batch.num_computed_tokens_cpu[0] = 50  # stale prefill view
    runner = make_runner(["r0"], input_batch=input_batch)
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is not None


def test_accepts_plan_covering_requests_missing_from_input_batch():
    runner = make_runner(["r0"])
    output = make_scheduler_output(["r0", "r1"])
    output.multi_step_plan = {"r0": 8, "r1": 8}
    output.num_scheduled_tokens = {"r0": 8, "r1": 8}
    # input_batch still shows only r0 (async lag); the plan is authoritative.
    assert validate_multi_step_plan(runner, output) is not None


def test_validate_returns_none_on_internal_error():
    class ExplodingRunner(SimpleNamespace):
        # any attribute access on use_async_scheduling raises
        def __getattribute__(self, name):
            if name == "use_async_scheduling":
                raise RuntimeError("boom")
            return super().__getattribute__(name)

    runner = ExplodingRunner()
    assert validate_multi_step_plan(runner, make_scheduler_output(["r0"])) is None

def test_window_cos_sin_positions_are_step_major():
    """Slot ``j*num_reqs + i`` must be request i's position at step j.

    The replay loop copies the prebuilt block for step j into the rows the
    captured graph reads, so a request-major flatten would give every request
    but the first another request's rotary phase from step 0 onwards.
    """
    positions = msd_module.window_cos_sin_positions([100, 200], 2, 3)
    assert positions == [100, 200, 101, 201, 102, 202]
    # The request-major order this used before the fix -- request 0's whole
    # step run first -- would hand request 1 request 0's positions.
    request_major = [100, 101, 102, 200, 201, 202]
    assert positions != request_major
    num_reqs = 2
    for step in range(3):  # each per-step slice is one step across requests
        block = positions[step * num_reqs : (step + 1) * num_reqs]
        assert block == [100 + step, 200 + step]


def test_window_cos_sin_positions_single_request_matches_per_step_updates():
    """``num_reqs == 1`` collapses both orders, which is why c1 was unaffected."""
    assert msd_module.window_cos_sin_positions([7], 1, 4) == [7, 8, 9, 10]

def test_notify_finished_requests_forwards_to_the_model():
    """The window path must not skip the model's finished-request hook."""
    calls = []
    model = SimpleNamespace(on_requests_finished=lambda ids: calls.append(set(ids)))
    runner = SimpleNamespace(model=model)
    output = SimpleNamespace(finished_req_ids={"r0", "r1"})
    msd_module.notify_finished_requests(runner, output)
    assert calls == [{"r0", "r1"}]


def test_notify_finished_requests_is_a_noop_without_finished_or_hook():
    calls = []
    runner = SimpleNamespace(
        model=SimpleNamespace(on_requests_finished=lambda ids: calls.append(ids))
    )
    # No finished ids this step.
    msd_module.notify_finished_requests(runner, SimpleNamespace(finished_req_ids=set()))
    # Scheduler output without the field at all.
    msd_module.notify_finished_requests(runner, SimpleNamespace())
    # Model without the hook.
    msd_module.notify_finished_requests(
        SimpleNamespace(model=object()), SimpleNamespace(finished_req_ids={"r0"})
    )
    assert calls == []

def test_shrink_refused_window_restores_a_single_token_step():
    """A runner refusal must shrink the whole K-token schedule to one token.

    Otherwise the fallback would run the K-token schedule as a plain multi-token
    decode: rows 1..K-1 would carry stale embeddings and KV would be written on
    slots the window never produced.
    """
    output = SimpleNamespace(
        num_scheduled_tokens={"r0": 8, "r1": 8},
        total_num_scheduled_tokens=16,
        multi_step_plan={"r0": 8, "r1": 8},
    )
    msd_module.shrink_refused_multi_step_window(output)
    assert output.num_scheduled_tokens == {"r0": 1, "r1": 1}
    assert output.total_num_scheduled_tokens == 2


def test_shrink_refused_window_leaves_unplanned_requests_alone():
    output = SimpleNamespace(
        num_scheduled_tokens={"r0": 8, "r1": 3},
        total_num_scheduled_tokens=11,
        multi_step_plan={"r0": 8},
    )
    msd_module.shrink_refused_multi_step_window(output)
    assert output.num_scheduled_tokens == {"r0": 1, "r1": 3}
    assert output.total_num_scheduled_tokens == 4


def test_shrink_refused_window_is_a_noop_without_a_plan():
    for plan in ({}, None):
        output = SimpleNamespace(
            num_scheduled_tokens={"r0": 1},
            total_num_scheduled_tokens=1,
            multi_step_plan=plan,
        )
        msd_module.shrink_refused_multi_step_window(output)
        assert output.num_scheduled_tokens == {"r0": 1}
        assert output.total_num_scheduled_tokens == 1

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Idle-step reference prefetch for the MiniCPM-o 4.5 Code2Wav stage.

``on_requests_added`` queues a placeholder's reference, and a
``run_idle_prefetch`` call runs one phase of it only while it is the lone
queued record and no stream is live: phase A materializes and pins the WAV and
runs ``prepare_prompt``; phase B (``token2wav_ref_prefetch_setup`` only) runs
``setup_batch``. Chunk 0's own path is unchanged: it hits the warm caches and
must stay bit-exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav as code2wav_module
from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import (
    _config,
    _enable_fake_ragged_kernel,
    _FakeToken2Wav,
    _forward,
    _info,
    _runtime_ref_info,
)
from vllm_omni.core.sched.output import OmniRequestPrewarm
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SR = 16000
_REF_A = torch.tensor([0.0, 0.25, -0.25, 0.0])
_REF_B = torch.tensor([0.0, 0.2])
_REF_C = torch.tensor([0.0, 0.3])
_REF_D = torch.tensor([0.0, 0.4])


def _prefetch_model(
    *,
    capacity: int = 4,
    setup_cache_size: int = 1,
    ref_prefetch: bool | None = None,
    prefetch_setup: bool | None = None,
) -> tuple[MiniCPMO45Code2Wav, _FakeToken2Wav]:
    """Code2Wav over the fake Token2wav; ``token2wav.cold_setups`` records the
    prompt key of every setup built on a setup-cache miss."""
    config = _config(runtime_prompt_cache_size=capacity, setup_cache_size=setup_cache_size)
    extra = config.model_config.stage_connector_config["extra"]
    if ref_prefetch is not None:
        extra["token2wav_ref_prefetch"] = ref_prefetch
    if prefetch_setup is not None:
        extra["token2wav_ref_prefetch_setup"] = prefetch_setup
    token2wav = _FakeToken2Wav()
    backend = BatchedToken2Wav(token2wav, setup_cache_size=setup_cache_size)
    _enable_fake_ragged_kernel(backend)
    token2wav.cold_setups = []
    create_initial_states = backend._create_initial_states

    def counting_create_initial_states(features, batch_size):
        token2wav.cold_setups.append(features.cache_key)
        return create_initial_states(features, batch_size)

    backend._create_initial_states = counting_create_initial_states
    model = MiniCPMO45Code2Wav(vllm_config=config)
    model.backend = backend
    return model, token2wav


def _cold(token2wav: _FakeToken2Wav) -> tuple[int, int]:
    """Cold work so far: (prompt extractions, setup builds)."""
    return token2wav.prompt_calls, len(token2wav.cold_setups)


def _prewarm(request_id: str, reference: torch.Tensor = _REF_A) -> OmniRequestPrewarm:
    return OmniRequestPrewarm(request_id=request_id, payload={"ref_audio": reference.clone(), "ref_audio_sr": _SR})


def _chunk0(reference: torch.Tensor, *, initial_codes: list[int] | None = None) -> dict[str, Any]:
    info = _runtime_ref_info("external-id", reference)
    if initial_codes is not None:
        info["codes"]["audio"] = torch.tensor(initial_codes, dtype=torch.long)
        if not initial_codes:
            # Empty initial segment marker: chunk 0 only sets the stream up.
            info["meta"].update(code_flat_numel=0, tts_is_last_chunk=True, turn_end=False)
    return info


def _placeholder(request_id: str) -> dict[str, Any]:
    # A prewarm placeholder step: runner bookkeeping only, no producer payload.
    return {"request_id": request_id, "meta": {"request_id": request_id}}


def _fail_first_call(original):
    calls = 0

    def wrapper(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected prefetch failure")
        return original(*args, **kwargs)

    return wrapper


def _assert_no_prefetch_state(model: MiniCPMO45Code2Wav) -> None:
    assert not model._prefetch_queue
    assert not model._prefetch_pins
    assert model._prefetch_setup_holder is None
    assert model._prefetch_setup_features is None
    assert not any(entry.pins for entry in model._runtime_prompts.values())


def _assert_bit_exact(model, baseline, make_info, request_id: str = "req-a") -> None:
    """Forward the same chunk through ``model`` and a no-prefetch ``baseline``;
    audio and request state must match bit for bit."""
    got_audio, want_audio = (
        _forward(m, [make_info()], request_ids=[request_id]).multimodal_outputs["model_outputs"]
        for m in (model, baseline)
    )
    torch.testing.assert_close(got_audio, want_audio, rtol=0, atol=0)
    got, want = model._states[request_id], baseline._states[request_id]
    fields = ("cache_epoch", "chunk_seq", "prompt_cache_id")
    assert [getattr(got, field) for field in fields] == [getattr(want, field) for field in fields]
    for cache in ("flow_cache", "hift_cache"):
        torch.testing.assert_close(getattr(got.token2wav, cache), getattr(want.token2wav, cache), rtol=0, atol=0)


@pytest.mark.parametrize("capacity", [0, 4])
@pytest.mark.parametrize("initial_codes", [[10, 11], []], ids=["codes", "empty-marker"])
@pytest.mark.parametrize("prefetch_setup", [False, True], ids=["default", "with-setup"])
def test_prefetch_warms_the_caches_and_chunk0_is_bit_exact(prefetch_setup, initial_codes, capacity):
    # Leave the setup knob unset in the default case: prefetch is on, prompt only.
    model, token2wav = _prefetch_model(capacity=capacity, prefetch_setup=prefetch_setup or None)
    baseline, _ = _prefetch_model(capacity=capacity, ref_prefetch=False)
    assert (model._ref_prefetch_enabled, model._ref_prefetch_setup) == (True, prefetch_setup)

    model.on_requests_added([_prewarm("req-a")])
    assert model.run_idle_prefetch() is True  # phase A
    key = model._prefetch_pins["req-a"]
    entry = model._runtime_prompts[key]
    prompt = (entry.cache_id, entry.path)
    assert (entry.owners, entry.pins) == (set(), {"req-a"})
    assert Path(entry.path).is_file()
    assert _cold(token2wav) == (1, 0)
    # The prefetch never claims request ownership or state; only chunk 0 does.
    assert (model._request_prompt_keys, model._states) == ({}, {})
    if prefetch_setup:
        assert model.run_idle_prefetch() is True  # phase B
        assert model._prefetch_setup_holder == "req-a"
    assert model.run_idle_prefetch() is False
    assert _cold(token2wav) == (1, int(prefetch_setup))

    _assert_bit_exact(model, baseline, lambda: _chunk0(_REF_A, initial_codes=initial_codes))

    # Chunk 0 hit everything that was prefetched and built only the rest.
    assert _cold(token2wav) == (1, 1)
    assert token2wav.cold_setups == [prompt]
    assert model._request_prompt_keys == {"req-a": key}
    assert entry.owners == {"req-a"}
    _assert_no_prefetch_state(model)
    _assert_bit_exact(model, baseline, lambda: _info("external-id", 1, [12, 13, 14]))

    model.on_requests_finished(["req-a"])
    _assert_no_prefetch_state(model)
    # The entry is kept unowned, or evicted with its WAV at capacity 0.
    assert list(model._runtime_prompts) == ([key] if capacity else [])
    assert Path(entry.path).exists() == bool(capacity)


# (did work, queue after the call) for three idle calls after each of a, b, c
# arrives alone; the earlier ones keep their pins, their chunk 0 still pending.
_LONE_STEPS = {
    "default": [[(True, ""), (False, ""), (False, "")]] * 3,
    # Phase A leaves the record queued and phase B pops it. b and c find the
    # setup slot held by a, so their phase B does no work.
    "with-setup": [
        [(True, "a"), (True, ""), (False, "")],
        [(True, "b"), (False, ""), (False, "")],
        [(True, "c"), (False, ""), (False, "")],
    ],
    # No setup cache means no slot to warm: every phase B is a no-op.
    "no-setup-cache": [
        [(True, "a"), (False, ""), (False, "")],
        [(True, "b"), (False, ""), (False, "")],
        [(True, "c"), (False, ""), (False, "")],
    ],
}


@pytest.mark.parametrize("case", list(_LONE_STEPS))
def test_idle_prefetch_runs_one_phase_per_call_for_requests_arriving_one_at_a_time(case):
    setup_cache_size = 0 if case == "no-setup-cache" else 1
    model, token2wav = _prefetch_model(setup_cache_size=setup_cache_size, prefetch_setup=case != "default")
    assert model.run_idle_prefetch() is False  # nothing queued

    steps = []
    for request_id, reference in zip("abc", (_REF_A, _REF_B, _REF_C), strict=True):
        model.on_requests_added([_prewarm(request_id, reference)])
        steps.append([(model.run_idle_prefetch(), "".join(model._prefetch_queue)) for _ in range(3)])

    assert steps == _LONE_STEPS[case]
    warmed_setup = case == "with-setup"
    assert _cold(token2wav) == (3, int(warmed_setup))
    assert model._prefetch_setup_holder == ("a" if warmed_setup else None)
    assert list(model._prefetch_pins) == ["a", "b", "c"]
    assert len(set(model._prefetch_pins.values())) == 3

    # b's chunk 0 hits its warm prompt and builds its setup, never prefetched.
    _forward(model, [_chunk0(_REF_B)], request_ids=["b"])
    assert _cold(token2wav) == (3, int(warmed_setup) + 1)
    model.on_requests_finished(["a", "b", "c"])
    _assert_no_prefetch_state(model)


@pytest.mark.parametrize("stream_end", ["last-chunk", "finished"])
@pytest.mark.parametrize(
    "prefetch_setup,deferred_phase",
    [(False, "A"), (True, "A"), (True, "B")],
    ids=["default", "setup-phase-a", "setup-phase-b"],
)
def test_busy_gate_defers_each_phase_while_another_stream_is_live(prefetch_setup, deferred_phase, stream_end):
    """No phase starts while a request streams here: it would delay that
    stream's next chunk."""
    model, token2wav = _prefetch_model(prefetch_setup=prefetch_setup)
    model.on_requests_added([_prewarm("req-a", _REF_A)])
    if deferred_phase == "B":
        assert model.run_idle_prefetch() is True  # phase A while idle
    _forward(model, [_chunk0(_REF_B)], request_ids=["req-live"])
    cold = _cold(token2wav)
    pins = dict(model._prefetch_pins)

    assert [model.run_idle_prefetch() for _ in range(3)] == [False] * 3
    assert _cold(token2wav) == cold
    assert list(model._prefetch_queue) == ["req-a"]
    assert model._prefetch_pins == pins
    assert model._prefetch_setup_holder is None

    if stream_end == "last-chunk":
        _forward(model, [_info("external-id", 1, [12, 13], last_chunk=True)], request_ids=["req-live"])
    else:
        model.on_requests_finished(["req-live"])
    assert not model._states

    # The gate lifts and the deferred phase runs.
    assert model.run_idle_prefetch() is True
    if deferred_phase == "A":
        assert _cold(token2wav) == (cold[0] + 1, cold[1])
        assert list(model._prefetch_pins) == ["req-a"]
    else:
        assert _cold(token2wav) == (cold[0], cold[1] + 1)
        assert model._prefetch_setup_holder == "req-a"

    model.on_requests_finished(["req-live", "req-a"])
    _assert_no_prefetch_state(model)


@pytest.mark.parametrize("drop", ["chunk0", "abort"])
@pytest.mark.parametrize(
    "prefetch_setup,deferred_phase", [(False, "A"), (True, "B")], ids=["default-phase-a", "setup-phase-b"]
)
def test_lone_waiter_gate_defers_each_phase_while_several_references_are_queued(prefetch_setup, deferred_phase, drop):
    """No phase starts while two or more references wait: a burst of
    placeholders means their upstream work is starting on the shared GPU."""
    model, token2wav = _prefetch_model(prefetch_setup=prefetch_setup)
    model.on_requests_added([_prewarm("req-a", _REF_A)])
    if deferred_phase == "B":
        assert model.run_idle_prefetch() is True  # phase A while alone
    model.on_requests_added([_prewarm("req-b", _REF_B), _prewarm("req-c", _REF_C)])
    cold = _cold(token2wav)
    pins = dict(model._prefetch_pins)

    assert [model.run_idle_prefetch() for _ in range(3)] == [False] * 3
    assert _cold(token2wav) == cold
    assert list(model._prefetch_queue) == ["req-a", "req-b", "req-c"]
    assert model._prefetch_pins == pins
    assert model._prefetch_setup_holder is None

    # The others leave. req-b's own chunk 0 runs the cold path inline,
    # bit-exact, and keeps no prefetch state; or req-b is aborted.
    if drop == "chunk0":
        baseline, _ = _prefetch_model(ref_prefetch=False)
        _assert_bit_exact(model, baseline, lambda: _chunk0(_REF_B), "req-b")
        assert _cold(token2wav) == (cold[0] + 1, cold[1] + 1)
        assert model._runtime_prompts[model._request_prompt_keys["req-b"]].owners == {"req-b"}
    else:
        model.on_requests_finished(["req-b"])
        # Exactly two still queued and no stream live: the gate alone holds them.
        assert list(model._prefetch_queue) == ["req-a", "req-c"]
        assert model.run_idle_prefetch() is False
        assert _cold(token2wav) == cold
    model.on_requests_finished(["req-c"])
    assert list(model._prefetch_queue) == ["req-a"]
    assert model._prefetch_pins == pins
    if drop == "chunk0":
        # The lone record still waits while req-b streams (busy gate).
        assert model.run_idle_prefetch() is False
        model.on_requests_finished(["req-b"])

    # Now alone with no live stream, req-a's deferred phase runs.
    cold = _cold(token2wav)
    assert model.run_idle_prefetch() is True
    if deferred_phase == "A":
        assert _cold(token2wav) == (cold[0] + 1, cold[1])
        assert list(model._prefetch_pins) == ["req-a"]
    else:
        assert _cold(token2wav) == (cold[0], cold[1] + 1)
        assert model._prefetch_setup_holder == "req-a"

    model.on_requests_finished(["req-a"])
    _assert_no_prefetch_state(model)


@pytest.mark.parametrize(
    "phases_run,busy",
    [(0, False), (1, False), (0, True)],
    ids=["before-phase-a", "between-phases", "deferred-by-busy-gate"],
)
def test_chunk0_that_overtakes_the_prefetch_retires_it_without_duplicate_work(phases_run, busy):
    model, token2wav = _prefetch_model(capacity=0, prefetch_setup=True)
    baseline, _ = _prefetch_model(capacity=0, ref_prefetch=False)
    if busy:
        _forward(model, [_chunk0(_REF_B)], request_ids=["req-live"])
    model.on_requests_added([_prewarm("req-a")])
    for _ in range(phases_run):
        assert model.run_idle_prefetch() is True
    if busy:
        assert model.run_idle_prefetch() is False

    _assert_bit_exact(model, baseline, lambda: _chunk0(_REF_A))

    _assert_no_prefetch_state(model)
    expected = 2 if busy else 1
    assert _cold(token2wav) == (expected, expected)
    assert model._runtime_prompts[model._request_prompt_keys["req-a"]].owners == {"req-a"}


@pytest.mark.parametrize("prefetch_setup", [False, True], ids=["default", "with-setup"])
def test_mismatched_chunk0_reference_runs_inline_and_releases_the_prefetch_pin(prefetch_setup):
    # Capacity 0: the released entry has no owner or pin left and is evicted.
    model, token2wav = _prefetch_model(capacity=0, prefetch_setup=prefetch_setup)
    baseline, _ = _prefetch_model(capacity=0, ref_prefetch=False)
    model.on_requests_added([_prewarm("req-a", _REF_A)])
    phases = 2 if prefetch_setup else 1
    assert [model.run_idle_prefetch() for _ in range(phases + 1)] == [True] * phases + [False]
    prefetched = model._runtime_prompts[model._prefetch_pins["req-a"]]
    prefetched_prompt = (prefetched.cache_id, prefetched.path)

    _assert_bit_exact(model, baseline, lambda: _chunk0(_REF_B))

    # Chunk 0's own reference is prepared inline, and the prefetched one is
    # evicted with its WAV, prompt features and setup.
    assert _cold(token2wav) == (2, phases)
    key = model._request_prompt_keys["req-a"]
    assert list(model._runtime_prompts) == [key]
    assert model._runtime_prompts[key].owners == {"req-a"}
    _assert_no_prefetch_state(model)
    assert not Path(prefetched.path).exists()
    assert prefetched_prompt not in model.backend._prompt_features
    assert all(setup_key[0] != prefetched_prompt for setup_key in model.backend._setup_cache)


@pytest.mark.parametrize(
    "prefetch_setup,phases_run",
    [(False, 1), (True, 0), (True, 1), (True, 2)],
    ids=["default", "queued", "after-phase-a", "after-phase-b"],
)
def test_abort_before_chunk0_releases_the_pin_and_setup_slot(prefetch_setup, phases_run):
    # Capacity 0: a released entry is evicted, and one pin is the cap.
    model, token2wav = _prefetch_model(capacity=0, prefetch_setup=prefetch_setup)
    model.on_requests_added([_prewarm("req-a")])
    for _ in range(phases_run):
        assert model.run_idle_prefetch() is True
    entry = model._runtime_prompts[model._prefetch_pins["req-a"]] if phases_run else None
    assert (model._prefetch_setup_holder == "req-a") == (phases_run == 2)

    model.on_requests_finished(["req-a"])

    _assert_no_prefetch_state(model)
    assert model.run_idle_prefetch() is False
    assert token2wav.prompt_calls == min(phases_run, 1)
    assert not model._runtime_prompts
    assert not model.backend._prompt_features
    assert not model.backend._setup_cache
    if entry is not None:
        assert not Path(entry.path).exists()

    # The freed pin (and setup slot) serve the next placeholder.
    model.on_requests_added([_prewarm("req-b", _REF_B)])
    assert model.run_idle_prefetch() is True
    assert list(model._prefetch_pins) == ["req-b"]
    assert model.run_idle_prefetch() is prefetch_setup
    assert model._prefetch_setup_holder == ("req-b" if prefetch_setup else None)


@pytest.mark.parametrize("prefetch_setup", [False, True], ids=["default", "with-setup"])
def test_prefetch_pin_survives_placeholder_steps_and_trims_by_other_requests(prefetch_setup):
    # Capacity 0: an entry that lost its pin is evicted at the next trim.
    model, token2wav = _prefetch_model(capacity=0, prefetch_setup=prefetch_setup)
    baseline, _ = _prefetch_model(capacity=0, ref_prefetch=False)
    model.on_requests_added([_prewarm("req-a", _REF_A)])
    assert model.run_idle_prefetch() is True  # phase A only
    key = model._prefetch_pins["req-a"]
    entry = model._runtime_prompts[key]

    # A placeholder silence step, then req-b streaming and finishing, all trim.
    silence = _forward(model, [_placeholder("req-a")], request_ids=["req-a"])
    assert silence.multimodal_outputs["model_outputs"][0].numel() == 0
    assert not model._states  # a placeholder is not a live stream
    _forward(model, [_chunk0(_REF_B)], request_ids=["req-b"])
    path_b = Path(model._runtime_prompts[model._request_prompt_keys["req-b"]].path)
    model.on_requests_finished(["req-b"])

    assert list(model._runtime_prompts) == [key]
    assert model._prefetch_pins == {"req-a": key}
    assert (entry.owners, entry.pins) == (set(), {"req-a"})
    assert Path(entry.path).is_file()
    assert not path_b.exists()

    # Phase B still runs on the pinned entry and survives another placeholder
    # step. The prompt is never re-extracted, and exactly one setup is built:
    # by phase B when it runs, else inline by chunk 0.
    cold = _cold(token2wav)
    assert model.run_idle_prefetch() is prefetch_setup
    _forward(model, [_placeholder("req-a")], request_ids=["req-a"])
    assert model._prefetch_setup_holder == ("req-a" if prefetch_setup else None)
    _assert_bit_exact(model, baseline, lambda: _chunk0(_REF_A))
    assert _cold(token2wav) == (cold[0], cold[1] + 1)
    assert entry.owners == {"req-a"}
    _assert_no_prefetch_state(model)


@pytest.mark.parametrize("prefetch_setup", [False, True], ids=["default", "with-setup"])
def test_requests_sharing_a_reference_hold_separate_pins(prefetch_setup):
    # Two outstanding pins need a runtime-prompt capacity (the pin cap) of 2.
    model, token2wav = _prefetch_model(capacity=2, prefetch_setup=prefetch_setup)
    phases = []
    for request_id in ("req-a", "req-b"):  # each arrives alone
        model.on_requests_added([_prewarm(request_id, _REF_A)])
        phases += [model.run_idle_prefetch() for _ in range(2)]
    # req-b's phase B (with-setup) finds the setup slot held by req-a.
    assert phases == [True, prefetch_setup, True, False]
    assert not model._prefetch_queue
    key = model._prefetch_pins["req-a"]
    entry = model._runtime_prompts[key]
    assert model._prefetch_pins == {"req-a": key, "req-b": key}
    assert (entry.owners, entry.pins) == (set(), {"req-a", "req-b"})
    assert _cold(token2wav) == (1, int(prefetch_setup))

    # Aborting one keeps the entry for the other, even with the cache squeezed
    # to zero so that only owners and pins keep entries.
    model._runtime_prompt_cache_size = 0
    model.on_requests_finished(["req-a"])
    assert model._runtime_prompts[key] is entry
    assert (entry.owners, entry.pins) == (set(), {"req-b"})
    assert Path(entry.path).is_file()

    # req-b hits the shared prompt, and the setup too when it was prefetched.
    _forward(model, [_chunk0(_REF_A)], request_ids=["req-b"])
    assert _cold(token2wav) == (1, 1)
    assert entry.owners == {"req-b"}
    _assert_no_prefetch_state(model)


@pytest.mark.parametrize("request_id", ["prefetch:foo", "bar"], ids=["pin-like-id", "plain-id"])
def test_releasing_a_pin_keeps_the_reference_of_a_request_whose_id_looks_like_the_pin(request_id):
    """Pins are kept apart from request ownership, so no request id, however
    it is spelled, can alias another request's pin: releasing foo's pin must
    not strip ``request_id``'s ownership and let a trim evict its reference."""
    # Capacity 1: the next new reference trims every entry nothing holds.
    model, _ = _prefetch_model(capacity=1)
    model.on_requests_added([_prewarm("foo", _REF_A)])
    assert [model.run_idle_prefetch() for _ in range(2)] == [True, False]  # phase A
    key = model._prefetch_pins["foo"]
    entry = model._runtime_prompts[key]

    # request_id's chunk 0 commits the same reference while foo's pin holds it.
    _forward(model, [_chunk0(_REF_A)], request_ids=[request_id])
    assert model._request_prompt_keys[request_id] == key
    assert request_id in entry.owners
    assert model._prefetch_pins == {"foo": key}

    # foo is aborted before its chunk 0, releasing its pin; then a third
    # request's new reference pushes the cache over capacity and trims.
    model.on_requests_finished(["foo"])
    _forward(model, [_chunk0(_REF_B)], request_ids=["baz"])

    # The trim spares the entry: still cached, its WAV on disk, and owned by
    # the request that is still streaming from it.
    assert model._runtime_prompts.get(key) is entry
    assert Path(entry.path).is_file()
    assert set(model._runtime_prompts) == {key, model._request_prompt_keys["baz"]}
    assert request_id in model._states
    assert model._request_prompt_keys[request_id] == key
    assert entry.owners == {request_id}
    # foo's pin is gone, from the index and from the entry.
    assert model._prefetch_pins == {}
    assert entry.pins == set()


def test_finishing_a_request_whose_id_looks_like_a_pin_keeps_that_pin():
    """The reverse aliasing: a request named like foo's pin finishing must not
    release foo's pin, or a capacity-0 trim evicts foo's prefetched entry."""
    model, _ = _prefetch_model(capacity=0)
    model.on_requests_added([_prewarm("foo", _REF_A)])
    assert [model.run_idle_prefetch() for _ in range(2)] == [True, False]  # phase A
    key = model._prefetch_pins["foo"]
    entry = model._runtime_prompts[key]

    _forward(model, [_chunk0(_REF_A)], request_ids=["prefetch:foo"])
    model.on_requests_finished(["prefetch:foo"])

    assert model._runtime_prompts.get(key) is entry
    assert Path(entry.path).is_file()
    assert (entry.owners, entry.pins) == (set(), {"foo"})
    assert model._prefetch_pins == {"foo": key}


@pytest.mark.parametrize("capacity", [0, 4])
@pytest.mark.parametrize("failure", ["wav_write", "prepare_prompt", "setup_batch"])
def test_prefetch_failure_is_dropped_and_unpinned_then_chunk0_runs_inline(failure, capacity, monkeypatch, mocker):
    log = mocker.patch.object(code2wav_module, "logger")
    # Phase B (setup_batch) runs only with the setup prefetch.
    model, token2wav = _prefetch_model(capacity=capacity, prefetch_setup=failure == "setup_batch")
    baseline, _ = _prefetch_model(capacity=capacity, ref_prefetch=False)
    if failure == "wav_write":
        monkeypatch.setattr(code2wav_module.sf, "write", _fail_first_call(code2wav_module.sf.write))
    else:
        monkeypatch.setattr(model.backend, failure, _fail_first_call(getattr(model.backend, failure)))
    model.on_requests_added([_prewarm("req-a")])
    if failure == "setup_batch":
        assert model.run_idle_prefetch() is True  # phase A succeeds

    # The failing phase is logged, dropped and unpinned; it never raises.
    assert model.run_idle_prefetch() is True
    log.warning.assert_called_once()
    assert "req-a" in log.warning.call_args.args
    _assert_no_prefetch_state(model)
    assert model.run_idle_prefetch() is False
    if not capacity:
        assert not model._runtime_prompts

    _assert_bit_exact(model, baseline, lambda: _chunk0(_REF_A))
    entry = model._runtime_prompts[model._request_prompt_keys["req-a"]]
    assert entry.owners == {"req-a"}
    assert Path(entry.path).is_file()
    # Features extracted before a phase-B failure survive only as an unowned
    # LRU entry, so chunk 0 re-extracts them when there is no spare capacity.
    assert token2wav.prompt_calls == (2 if failure == "setup_batch" and not capacity else 1)
    _assert_no_prefetch_state(model)


def test_setup_slot_holds_one_waiting_setup_until_chunk0_or_eviction():
    """Phase B warms at most one setup still waiting for its chunk 0. The slot
    frees at the holder's chunk 0, or once another request's inline setup has
    evicted the holder's from the one-entry setup cache."""
    model, token2wav = _prefetch_model(prefetch_setup=True)
    baseline, _ = _prefetch_model(ref_prefetch=False)
    model.on_requests_added([_prewarm("req-a", _REF_A)])
    assert [model.run_idle_prefetch() for _ in range(2)] == [True, True]
    model.on_requests_added([_prewarm("req-b", _REF_B)])  # arrives alone
    assert [model.run_idle_prefetch() for _ in range(3)] == [True, False, False]
    assert model._prefetch_setup_holder == "req-a"
    entry_b = model._runtime_prompts[model._prefetch_pins["req-b"]]
    assert _cold(token2wav) == (2, 1)  # req-b got its prompt but no setup

    # req-a's chunk 0 hits its prefetched setup and frees the slot for req-c.
    _forward(model, [_chunk0(_REF_A)], request_ids=["req-a"])
    assert _cold(token2wav) == (2, 1)
    assert model._prefetch_setup_holder is None
    model.on_requests_finished(["req-a"])  # lifts the busy gate
    model.on_requests_added([_prewarm("req-c", _REF_C)])
    assert [model.run_idle_prefetch() for _ in range(2)] == [True, True]
    assert model._prefetch_setup_holder == "req-c"
    features_c = model._prefetch_setup_features
    assert _cold(token2wav) == (3, 2)

    # req-b's chunk 0 hits its warm prompt and builds its setup inline, which
    # evicts req-c's from the only slot. The holder is released lazily.
    _forward(model, [_chunk0(_REF_B)], request_ids=["req-b"])
    model.on_requests_finished(["req-b"])
    assert _cold(token2wav) == (3, 3)
    assert token2wav.cold_setups[-1] == (entry_b.cache_id, entry_b.path)
    assert not model.backend.has_cached_setup(features_c)
    assert model._prefetch_setup_holder == "req-c"

    # So the next placeholder's phase B takes over the stale slot...
    model.on_requests_added([_prewarm("req-d", _REF_D)])
    assert [model.run_idle_prefetch() for _ in range(2)] == [True, True]
    assert model._prefetch_setup_holder == "req-d"
    assert _cold(token2wav) == (4, 4)

    # ...and req-c's chunk 0 rebuilds its evicted setup inline, bit-exact,
    # without taking req-d's slot.
    _assert_bit_exact(model, baseline, lambda: _chunk0(_REF_C), "req-c")
    assert token2wav.cold_setups[-1] == features_c.cache_key
    assert _cold(token2wav) == (4, 5)
    assert model._prefetch_setup_holder == "req-d"
    assert set(model._prefetch_pins) == {"req-d"}

    model.on_requests_finished(["req-c", "req-d"])
    _assert_no_prefetch_state(model)


@pytest.mark.parametrize("capacity", [0, 2])
def test_outstanding_pins_are_capped_by_the_runtime_prompt_capacity(capacity):
    """Pinned entries are never trimmed, so phase A waits while
    ``max(1, capacity)`` requests hold pins; the record stays queued. Pins pile
    up when requests arrive one at a time ahead of their chunk 0."""
    model, token2wav = _prefetch_model(capacity=capacity)
    cap = max(1, capacity)
    request_ids = [f"req-{index}" for index in range(cap + 1)]
    started = []
    for index, request_id in enumerate(request_ids):
        model.on_requests_added([_prewarm(request_id, torch.tensor([0.0, 0.1 * (index + 1)]))])
        started.append(model.run_idle_prefetch())

    assert started + [model.run_idle_prefetch()] == [True] * cap + [False, False]
    assert set(model._prefetch_pins) == set(request_ids[:cap])
    assert list(model._prefetch_queue) == [request_ids[cap]]
    assert token2wav.prompt_calls == cap

    # Releasing a pin (here an abort) lets the waiting record start phase A.
    model.on_requests_finished([request_ids[0]])
    assert model.run_idle_prefetch() is True
    assert set(model._prefetch_pins) == set(request_ids[1:])
    assert token2wav.prompt_calls == cap + 1

    model.on_requests_finished(request_ids)
    _assert_no_prefetch_state(model)


def test_prefetch_queue_is_capped_and_overflow_runs_inline_at_chunk0():
    model, token2wav = _prefetch_model()
    limit = code2wav_module._MAX_QUEUED_PREFETCHES
    request_ids = [f"req-{index}" for index in range(limit + 1)]
    overflow = request_ids[limit]

    model.on_requests_added([_prewarm(request_id) for request_id in request_ids])
    assert list(model._prefetch_queue) == request_ids[:limit]
    assert model.run_idle_prefetch() is False  # no lone waiter: records wait

    # The skipped request prepares its reference inline at chunk 0.
    _forward(model, [_chunk0(_REF_A)], request_ids=[overflow])
    assert token2wav.prompt_calls == 1
    assert overflow in model._request_prompt_keys

    model.on_requests_finished(request_ids)
    _assert_no_prefetch_state(model)


@pytest.mark.parametrize("case", ["knob-off", "no-backend"])
def test_hooks_are_noops_when_disabled_or_before_the_backend_exists(case, monkeypatch):
    if case == "knob-off":
        model, _ = _prefetch_model(ref_prefetch=False, prefetch_setup=True)
    else:
        model = MiniCPMO45Code2Wav(vllm_config=_config())
        assert model.backend is None
        monkeypatch.setattr(model, "_build_backend", lambda: pytest.fail("the hooks must not build the backend"))

    model.on_requests_added([_prewarm("req-a")])

    assert not model._prefetch_queue
    assert model.run_idle_prefetch() is False
    assert not model._runtime_prompts


def test_on_requests_added_skips_known_requests_and_is_idempotent():
    model, _ = _prefetch_model(prefetch_setup=True)
    model.on_requests_added([_prewarm("req-a", _REF_A), _prewarm("req-a", _REF_B)])
    assert list(model._prefetch_queue) == ["req-a"]
    torch.testing.assert_close(model._prefetch_queue["req-a"].ref_audio, _REF_A, rtol=0, atol=0)

    # Neither a queued-and-pinned record (after phase A) nor a pinned-only one
    # (after phase B) is replaced or re-queued.
    assert model.run_idle_prefetch() is True
    record = model._prefetch_queue["req-a"]
    model.on_requests_added([_prewarm("req-a", _REF_B)])
    assert model._prefetch_queue["req-a"] is record
    assert model.run_idle_prefetch() is True
    model.on_requests_added([_prewarm("req-a", _REF_B)])
    assert not model._prefetch_queue

    # Nor is a request that is streaming, or whose final chunk committed a
    # reference before it finished.
    _forward(model, [_chunk0(_REF_A)], request_ids=["req-a"])
    final = _chunk0(_REF_B)
    final["meta"]["last_chunk"] = True
    _forward(model, [final], request_ids=["req-c"])
    assert "req-c" not in model._states
    model.on_requests_added([_prewarm("req-a", _REF_B), _prewarm("req-c", _REF_C)])
    assert not model._prefetch_queue

    # Once finished, a reused id is a new request again.
    model.on_requests_finished(["req-c"])
    model.on_requests_added([_prewarm("req-c", _REF_C)])
    assert list(model._prefetch_queue) == ["req-c"]


@pytest.mark.parametrize(
    "payload,warns",
    [
        pytest.param({"ref_audio": _REF_A.reshape(1, -1), "ref_audio_sr": _SR}, True, id="two-dim-ref"),
        pytest.param({"ref_audio": torch.tensor([0, 1], dtype=torch.int16), "ref_audio_sr": _SR}, True, id="int-ref"),
        pytest.param({"ref_audio": torch.empty(0), "ref_audio_sr": _SR}, True, id="empty-ref"),
        pytest.param({"ref_audio": [0.0, 0.25], "ref_audio_sr": _SR}, True, id="list-ref"),
        pytest.param({"ref_audio": _REF_A}, True, id="missing-sr"),
        pytest.param({"ref_audio": _REF_A, "ref_audio_sr": 0}, True, id="zero-sr"),
        pytest.param({"ref_audio": _REF_A, "ref_audio_sr": True}, True, id="bool-sr"),
        pytest.param({"ref_audio": _REF_A, "ref_audio_sr": float(_SR)}, True, id="float-sr"),
        # No reference at all: nothing to prefetch, and nothing to warn about.
        pytest.param({"ref_audio_sr": _SR}, False, id="no-ref"),
        pytest.param(None, False, id="not-a-mapping"),
    ],
)
def test_bad_prefetch_payload_is_skipped_without_raising(payload, warns, mocker):
    log = mocker.patch.object(code2wav_module, "logger")
    model, token2wav = _prefetch_model()

    model.on_requests_added([OmniRequestPrewarm(request_id="req-bad", payload=payload), _prewarm("req-ok")])

    assert list(model._prefetch_queue) == ["req-ok"]
    assert log.warning.call_count == int(warns)
    if warns:
        assert "req-bad" in log.warning.call_args.args
    assert [model.run_idle_prefetch() for _ in range(2)] == [True, False]
    assert set(model._prefetch_pins) == {"req-ok"}
    assert token2wav.prompt_calls == 1


@pytest.mark.parametrize("allow_tf32", [False, True])
def test_prefetch_phases_run_under_the_stage_matmul_policy(allow_tf32, monkeypatch):
    # The prefetch runs chunk 0's preparation calls outside forward, so it must
    # apply the same TF32 policy as forward and restore the caller's afterwards.
    model, _ = _prefetch_model(prefetch_setup=True)
    model.vllm_config.model_config.stage_connector_config["extra"]["token2wav_allow_tf32"] = allow_tf32
    seen: list[tuple[str, bool]] = []
    for name in ("prepare_prompt", "setup_batch"):
        original = getattr(model.backend, name)

        def recording(*args, _name=name, _original=original, **kwargs):
            seen.append((_name, torch.backends.cuda.matmul.allow_tf32))
            return _original(*args, **kwargs)

        monkeypatch.setattr(model.backend, name, recording)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        model.on_requests_added([_prewarm("req-a")])
        assert [model.run_idle_prefetch() for _ in range(3)] == [True, True, False]
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    # Phase A prepares the prompt; phase B prepares it again (cache hit) and builds the setup.
    assert seen == [("prepare_prompt", allow_tf32), ("prepare_prompt", allow_tf32), ("setup_batch", allow_tf32)]

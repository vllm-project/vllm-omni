# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU fallback/state and mocked lifetime contracts; CUDA qualification separate."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.model_executor.models.test_lychee_f0_math import FakeAPI, fake_library
from vllm_omni.model_executor.models.lychee_fd.token2wav import LycheeToken2WavCore, LycheeToken2WavSessionStore
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.flashcosyvoice.modules import hifigan
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.flashcosyvoice.modules.hifigan_components import (
    f0_math,
    preconv_math,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_full_generator_state_future_rng_and_independent_owner_class(monkeypatch):
    cls = hifigan.PreconvConv1d
    monkeypatch.setattr(hifigan, "PreconvConv1d", hifigan.Conv1d)
    torch.manual_seed(73)
    original = hifigan.HiFTGenerator()
    expected_future = torch.rand(8)
    monkeypatch.setattr(hifigan, "PreconvConv1d", cls)
    torch.manual_seed(73)
    actual = hifigan.HiFTGenerator()
    observed_future = torch.rand(8)
    torch.testing.assert_close(observed_future, expected_future, rtol=0, atol=0)
    assert len(actual.state_dict()) == 328 and actual.state_dict().keys() == original.state_dict().keys()
    assert sum(value.numel() for value in actual.state_dict().values()) == 20821295
    for name, value in original.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
    assert len(actual.conv_pre.state_dict()) == 3 and len(actual.f0_predictor.state_dict()) == 17
    assert isinstance(actual.conv_pre, preconv_math.PreconvConv1d) and not isinstance(actual.conv_pre, f0_math.F0Conv1d)
    assert actual.conv_pre._conv_forward.__module__.endswith(".preconv_math")
    assert sum(isinstance(m, f0_math.F0Conv1d) for m in actual.modules()) == 5
    assert sum(isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)) for m in actual.modules()) == 85
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("length", [2, 14, 58, 60, 62])
def test_cpu_fallback_numeric_and_gradient_semantics(length):
    torch.manual_seed(81)
    a = preconv_math.PreconvConv1d(80, 512, 7, padding=3).eval()
    b = hifigan.Conv1d(80, 512, 7, padding=3).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 80, length, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    torch.testing.assert_close(a(x), b(y), rtol=0, atol=0)
    a(x).sum().backward()
    b(y).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    torch.testing.assert_close(a.weight.grad, b.weight.grad, rtol=0, atol=0)


@pytest.mark.parametrize("length", sorted(preconv_math.QUALIFIED_LENGTHS))
def test_independent_k7_p3_descriptor_and_f0_unchanged(monkeypatch, length):
    lib = FakeAPI()
    fake_library(monkeypatch, lib)
    plan = preconv_math._PreconvPlan((1, 80, length), (512, 80, 7), "cuda:0")
    assert [v for a, v in lib.attributes if a == 902] == [[1, 80, 1, length], [512, 80, 1, 7], [1, 512, 1, length]]
    assert [v for a, v in lib.attributes if a in (104, 105)] == [[0, 3], [0, 3]]
    with pytest.raises(ValueError):
        f0_math._F0Plan.validate_shapes((1, 80, length), (512, 80, 7))
    plan.close()
    assert sorted(lib.created) == sorted(lib.destroyed)


def test_eligibility_and_runtime_no_init_guards(monkeypatch):
    monkeypatch.setattr(preconv_math, "_qualified_runtime", lambda x: True)
    model = preconv_math.PreconvConv1d(80, 512, 7, padding=3).eval()
    x = torch.ones(1, 80, 58)
    assert not model._eligible(x, model.weight, model.bias)
    with torch.no_grad():
        assert model._eligible(x, model.weight, model.bias)
        for length in (1, 59, 60, 62):
            assert not model._eligible(torch.ones(1, 80, length), model.weight, model.bias)
        model.train()
        assert not model._eligible(x, model.weight, model.bias)

    def forbidden(*a, **kw):
        raise AssertionError("CPU predicate initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    assert not f0_math._qualified_runtime(x)


class Plan:
    instances: list["Plan"] = []

    def __init__(self, shape, weight_shape, device):
        self.shape, self.weight_shape, self.device = shape, weight_shape, device
        self.workspace_bytes = 1024**2
        self.closed = 0
        self.close_failures = 0
        self.execute_failure = False
        self.instances.append(self)

    def allocate_workspace(self):
        pass

    def execute(self, x, w, b):
        if self.execute_failure:
            raise ValueError("primary execute")
        return torch.nn.functional.conv1d(x, w, b, padding=3)

    def close(self):
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("completion fence")
        self.closed += 1


@pytest.fixture
def cache(monkeypatch):
    Plan.instances = []
    monkeypatch.setattr(preconv_math, "_PreconvPlan", Plan)
    monkeypatch.setattr(preconv_math.PreconvConv1d, "_eligible", lambda *a: True)
    monkeypatch.setattr(torch.accelerator, "device_index", lambda d: nullcontext())
    return preconv_math.PreconvConv1d(80, 512, 7, padding=3).eval()


def call(model, length):
    with torch.no_grad():
        return model(torch.ones(1, 80, length))


def test_lru_and_workspace_cleanup_copy_apply(cache):
    for length in (50, 58, 32, 14, 20):
        call(cache, length)
    assert len(cache._f0_plans) == 4 and Plan.instances[0].closed == 1
    replica = deepcopy(cache)
    assert not replica._f0_plans and not replica._f0_failed_plans
    assert set(cache.__getstate__()).isdisjoint({"_f0_plans", "_f0_failed_plans", "_f0_lock"})
    cache.double()
    assert not cache._f0_plans and all(p.closed == 1 for p in Plan.instances)


def test_failed_eviction_retains_owner_and_retry(cache):
    for length in (50, 58, 32, 14):
        call(cache, length)
    victim = Plan.instances[0]
    victim.close_failures = 2
    with pytest.raises(RuntimeError, match="completion fence"):
        call(cache, 20)
    assert victim in cache._f0_plans.values() and victim in cache._f0_failed_plans
    with pytest.raises(RuntimeError, match="completion fence"):
        cache.clear_released_plans()
    assert victim in cache._f0_plans.values()
    cache.clear_released_plans()
    assert not cache._f0_plans and not cache._f0_failed_plans


def test_execute_primary_survives_failed_fence_and_apply_waits(cache):
    call(cache, 50)
    plan = Plan.instances[0]
    plan.execute_failure = True
    plan.close_failures = 2
    with pytest.raises(ValueError, match="primary execute") as captured:
        call(cache, 50)
    assert captured.value._lychee_cleanup_errors and plan in cache._f0_plans.values()
    before = cache.weight.dtype
    with pytest.raises(RuntimeError, match="completion fence"):
        cache.double()
    assert cache.weight.dtype == before and plan in cache._f0_failed_plans
    cache.clear_released_plans()
    cache.double()
    assert not cache._f0_plans


@pytest.mark.parametrize("failure_count", [1, 2])
def test_plan_failed_fence_preserves_all_resources_until_retry(monkeypatch, failure_count):
    lib = FakeAPI()
    fake_library(monkeypatch, lib)
    plan = preconv_math._PreconvPlan((1, 80, 14), (512, 80, 7), "cuda:0")
    owned = object()
    plan.workspace = owned

    class Event:
        remaining = failure_count

        def synchronize(self):
            if self.remaining:
                self.remaining -= 1
                raise RuntimeError("not complete")

    event = Event()
    plan.last_event = event
    for attempt in range(failure_count):
        with pytest.raises(RuntimeError, match="not complete"):
            plan.close()
        assert plan.last_event is event and plan.workspace is owned and plan.handle and not lib.destroyed
    plan.close()
    assert plan.workspace is None and not plan.handle and not plan.descriptors


def test_hift_shutdown_attempts_both_owners_and_preserves_first():
    model = hifigan.HiFTGenerator.__new__(hifigan.HiFTGenerator)
    nn.Module.__init__(model)
    calls = []

    def failed():
        calls.append("f0")
        raise RuntimeError("first owner")

    model.f0_predictor = SimpleNamespace(shutdown=failed)
    model.conv_pre = SimpleNamespace(clear_released_plans=lambda: calls.append("preconv"))
    with pytest.raises(RuntimeError, match="first owner"):
        model.shutdown()
    assert calls == ["f0", "preconv"]


@pytest.mark.parametrize("total,final_tokens,mel_length", [(51, 26, 60), (52, 27, 62)])
def test_real_stream_store_final_pending_reaches_unqualified_60_62(total, final_tokens, mel_length):
    core = LycheeToken2WavCore.__new__(LycheeToken2WavCore)
    nn.Module.__init__(core)
    core.device = torch.device("cpu")
    core.float16 = False
    core.estimator_cache_keep = 4
    core.sample_rate = 24000
    core.speech_window = torch.ones(7680)

    def setup(state):
        state.stream_cache = {}
        state.hift_cache = dict(mel=torch.zeros(1, 80, 0), source=torch.zeros(1, 1, 0), speech=torch.zeros(1, 0))

    core.setup = setup
    core.prepare_prompt = lambda path: (
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(1, 192),
        torch.zeros(1, 1, 80),
    )
    calls = []

    class ClockFlow(nn.Module):
        def inference_chunk(self, *, token, spk, cache, last_chunk, n_timesteps):
            # Released actual27-final capture has54 Mel frames; nonfinal28 has50.
            frames = 2 * (token.shape[1] - (0 if last_chunk else 3))
            calls.append((token.shape[1], last_chunk))
            return torch.zeros(1, 80, frames), {"estimator_att_cache": torch.zeros(1, 1, 1, 1, 1, 1)}

    seen = []

    class ShapeHiFT(nn.Module):
        def forward(self, mel, source):
            seen.append(mel.shape[-1])
            return torch.zeros(1, mel.shape[-1] * 480), torch.zeros(1, 1, mel.shape[-1] * 480)

    core.flow = ClockFlow()
    core.hift = ShapeHiFT()
    store = LycheeToken2WavSessionStore(core, "voice.wav")
    metadata = dict(
        request_id="request",
        session_id="session",
        response_id="response",
        response_number=1,
        execution_epoch=0,
        session_epoch=0,
    )
    store.process([1] * 28, dict(metadata, chunk_seq=0, final=False))
    store.process([1] * (total - 28), dict(metadata, chunk_seq=1, final=False))
    assert next(iter(store.states.values())).pending_tokens == [1] * final_tokens
    store.process([], dict(metadata, chunk_seq=2, final=True))
    assert calls == [(28, False), (final_tokens, True)] and seen == [50, mel_length]
    assert mel_length not in preconv_math.QUALIFIED_LENGTHS and mel_length not in f0_math.QUALIFIED_LENGTHS
    assert not store.states and not torch.cuda.is_initialized()


@pytest.mark.parametrize("event_failure", ["create", "record"])
@pytest.mark.parametrize("pack_failure", [False, True])
def test_backend_primary_retained_through_event_and_pack_failures(monkeypatch, event_failure, pack_failure):
    lib = FakeAPI()
    fake_library(monkeypatch, lib)
    plan = preconv_math._PreconvPlan((1, 80, 14), (512, 80, 7), "cuda:0")
    events = []

    class Stream:
        cuda_stream = 7

        def synchronize(self):
            events.append("owned fence")

    stream = Stream()

    class Event:
        def __init__(self):
            if event_failure == "create":
                raise RuntimeError("event create secondary")

        def record(self, owner):
            assert owner is stream
            raise RuntimeError("event record secondary")

    class Tensor:
        dtype = torch.float32
        is_cuda = True
        device = torch.device("cuda:0")

        def __init__(self, shape):
            self.shape = shape

        def contiguous(self):
            return self

        def record_stream(self, owner):
            assert owner is stream

        def data_ptr(self):
            return 1024

        def reshape(self, *shape):
            return Tensor(shape)

        def __add__(self, other):
            return self

    def failed_execute(*a):
        events.append("queued execute")
        raise ValueError("backend primary")

    lib.functions["cudnnBackendExecute"] = failed_execute
    original_destroy = lib.cudnnBackendDestroyDescriptor
    failed_pack: list[int] = []

    def destroy(descriptor):
        if pack_failure and not failed_pack:
            failed_pack.append(descriptor.value)
            raise RuntimeError("variant pack secondary")
        return original_destroy(descriptor)

    lib.functions["cudnnBackendDestroyDescriptor"] = destroy
    plan.workspace = Tensor((0,))
    monkeypatch.setattr(torch.accelerator, "device_index", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch, "empty", lambda shape, **kw: Tensor(shape))
    with pytest.raises(ValueError, match="backend primary") as captured:
        plan.execute(Tensor((1, 80, 14)), Tensor((512, 80, 7)), Tensor((512,)))
    assert len(captured.value._lychee_cleanup_errors) == (2 if pack_failure else 1)
    assert plan.last_stream is stream
    if pack_failure:
        assert failed_pack[0] in [d.value for d in plan.descriptors]
    plan.close()
    assert events == ["queued execute", "owned fence"]
    assert not plan.handle and not plan.descriptors and plan.workspace is None
    assert sorted(lib.created) == sorted(lib.destroyed)


def test_descriptor_cleanup_failure_attempts_all_and_retries_only_retained_owner(monkeypatch):
    lib = FakeAPI()
    fake_library(monkeypatch, lib)
    plan = preconv_math._PreconvPlan((1, 80, 14), (512, 80, 7), "cuda:0")
    original = lib.cudnnBackendDestroyDescriptor
    attempted, remaining = [], [1]
    failed_id = plan.descriptors[-1].value

    def destroy(descriptor):
        attempted.append(descriptor.value)
        if descriptor.value == failed_id and remaining:
            remaining.pop()
            raise RuntimeError("descriptor failure")
        return original(descriptor)

    lib.functions["cudnnBackendDestroyDescriptor"] = destroy
    total = len(plan.descriptors)
    with pytest.raises(RuntimeError, match="descriptor failure"):
        plan.close()
    assert len(attempted) == total and [d.value for d in plan.descriptors] == [failed_id]
    assert plan.handle and not lib.events.count("destroy_handle")
    plan.close()
    assert attempted[-1] == failed_id and len(attempted) == total + 1
    assert not plan.descriptors and not plan.handle
    assert sorted(lib.created) == sorted(lib.destroyed)


def test_partial_plan_build_cleanup_failure_retains_retryable_owner(monkeypatch):
    lib = FakeAPI(fail_attribute=1301)
    fake_library(monkeypatch, lib)
    original = lib.cudnnBackendDestroyDescriptor
    remaining = [1]

    def destroy(descriptor):
        if remaining:
            remaining.pop()
            raise RuntimeError("partial cleanup failure")
        return original(descriptor)

    lib.functions["cudnnBackendDestroyDescriptor"] = destroy
    with pytest.raises(RuntimeError, match="set 1301") as captured:
        preconv_math._PreconvPlan((1, 80, 14), (512, 80, 7), "cuda:0")
    assert str(captured.value._lychee_cleanup_errors[0]) == "partial cleanup failure"
    owner = captured.value._lychee_failed_plan
    assert owner.descriptors and owner.handle
    owner.close()
    assert not owner.handle and not owner.descriptors
    assert sorted(lib.created) == sorted(lib.destroyed)

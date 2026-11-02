# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.minicpmo_4_5.talker_codec_sample import TalkerCudaKStepState
from vllm_omni.worker.talker_cuda_graph import CUDANarrowStep, FlashAttentionKVTransaction

pytestmark = pytest.mark.core_model


class _State:
    def __init__(self, device="cpu"):
        self.workspace = TalkerCudaKStepState.allocate(1, 2, torch.device(device))
        self.embeds = torch.zeros((1, 2), device=device)
        self.cache = torch.zeros((2, 1, 4, 2), device=device)
        self.slots = torch.tensor([0, 1, -1, 1], device=device)
        self.transaction = FlashAttentionKVTransaction({"layer": self.cache}, {"layer": self.slots})
        self.generator = torch.Generator(device=device).manual_seed(1234)
        self.calls = 0

    def snapshot(self):
        return (
            self.workspace.snapshot(),
            self.transaction.snapshot(),
            self.embeds.clone(),
            self.generator.get_state().clone(),
        )

    def restore(self, saved):
        self.workspace.restore(saved[0])
        self.transaction.restore(saved[1])
        self.embeds.copy_(saved[2])
        self.generator.set_state(saved[3])

    def advance(self):
        self.calls += 1
        self.workspace.steps.add_(2)
        self.workspace.history.add_(1)
        self.workspace.last_tokens.add_(1)
        self.workspace.tokens.add_(1)
        self.workspace.finished.logical_not_()
        self.workspace.emits.logical_not_()
        self.workspace.stops.logical_not_()
        self.workspace.uniforms.uniform_(generator=self.generator)
        self.embeds.add_(1)
        self.cache[0, :, :2, :].add_(1)
        return self.embeds.clone()

    def assert_matches(self, saved):
        for name, tensor in self.workspace._tensor_fields().items():
            assert torch.equal(tensor, saved[0][name]), name
        for name, tensor in self.transaction.snapshot().items():
            assert torch.equal(tensor, saved[1][name]), name
        assert torch.equal(self.embeds, saved[2])
        assert torch.equal(self.generator.get_state(), saved[3])
        assert torch.count_nonzero(self.cache[1]).item() == 0
        assert torch.count_nonzero(self.cache[0, :, 2:]).item() == 0


def _narrow(state, *, fake_cuda=False):
    return CUDANarrowStep(
        rows=1,
        frames=2,
        inputs_embeds=SimpleNamespace(device=torch.device("cuda")) if fake_cuda else state.embeds,
        positions=torch.zeros(1, dtype=torch.long),
        step_slots=torch.zeros((1, 2), dtype=torch.long),
        step_positions=torch.zeros((1, 2), dtype=torch.long),
        step_seq_lens=torch.ones((1, 2), dtype=torch.long),
        metadata={},
    )


class _FakeCUDA:
    def __init__(self, monkeypatch, forward, failure):
        self.failure = failure
        self.forward = forward
        self.warmups = 0
        self.captures = 0
        self.registrations = 0
        self.replays = 0
        self.in_capture = False
        monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: self)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: self)
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
        monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: self)
        monkeypatch.setattr(torch.cuda, "graph", self.capture)

    def wait_stream(self, stream):
        pass

    def register_generator_state(self, generator):
        self.registrations += 1
        if self.failure == "registration":
            raise RuntimeError("registration failed")

    @contextmanager
    def capture(self, graph):
        self.captures += 1
        self.in_capture = True
        try:
            yield
        finally:
            self.in_capture = False

    def replay(self):
        self.replays += 1
        self.forward()


@pytest.mark.cpu
@pytest.mark.parametrize("failure", ["warmup", "capture", "registration", None])
def test_fused_trial_advances_state_once_and_failed_key_does_not_retry(monkeypatch, failure):
    monkeypatch.setenv("KSTEP_CUDA_GRAPH", "1")
    state = _State()
    narrow = _narrow(state, fake_cuda=True)
    initial = state.snapshot()
    state.advance()
    expected_once = state.snapshot()
    state.advance()
    expected_twice = state.snapshot()
    state.restore(initial)
    attempted = False

    def forward():
        nonlocal attempted
        output = state.advance()
        if failure in ("warmup", "capture") and not attempted:
            if (failure == "warmup" and not cuda.in_capture) or (failure == "capture" and cuda.in_capture):
                attempted = True
                raise RuntimeError(f"{failure} failed after state mutation")
        return output

    cuda = _FakeCUDA(monkeypatch, forward, failure)
    kwargs = dict(state_key="state", snapshot=state.snapshot, restore=state.restore, generators=(state.generator,))
    output = narrow.run_fused(forward, **kwargs)
    state.assert_matches(expected_once)
    assert torch.equal(output, expected_once[2])
    assert narrow.eager_calls == int(failure is not None)
    assert narrow.captures == int(failure is None)
    assert narrow.replays == int(failure is None)
    trial_calls = state.calls
    trial_captures = cuda.captures
    narrow.run_fused(forward, **kwargs)
    state.assert_matches(expected_twice)
    assert state.calls == trial_calls + 1
    assert cuda.captures == trial_captures
    if failure is not None:
        assert ("fused", "state") in narrow.capture_failures
        assert narrow.capture_failure_eager == 1
        assert cuda.replays == 0
    else:
        assert cuda.replays == 2


@pytest.mark.cpu
@pytest.mark.parametrize("restore_number", [1, 2])
def test_fused_restore_failure_propagates_without_eager_or_cache(monkeypatch, restore_number):
    monkeypatch.setenv("KSTEP_CUDA_GRAPH", "1")
    state = _State()
    narrow = _narrow(state, fake_cuda=True)
    cuda = _FakeCUDA(monkeypatch, state.advance, "registration" if restore_number == 2 else None)
    restores = 0

    def restore(saved):
        nonlocal restores
        restores += 1
        if restores == restore_number:
            raise RuntimeError("rollback failed")
        state.restore(saved)

    with pytest.raises(RuntimeError, match="rollback failed"):
        narrow.run_fused(
            state.advance, state_key="state", snapshot=state.snapshot, restore=restore, generators=(state.generator,)
        )
    assert state.calls == 1
    assert narrow.eager_calls == 0
    assert narrow.captures == narrow.replays == 0
    assert not narrow.graphs
    assert not narrow.capture_failures
    assert cuda.replays == 0


def test_snapshot_restores_every_workspace_field_rng_kv_and_embedding_in_place():
    state = _State()
    initial = state.snapshot()
    pointers = {name: tensor.data_ptr() for name, tensor in state.workspace._tensor_fields().items()}
    for tensor in state.workspace._tensor_fields().values():
        if tensor.dtype == torch.bool:
            tensor.logical_not_()
        else:
            tensor.add_(1)
    state.cache[0, :, :2, :].fill_(7)
    state.embeds.fill_(9)
    torch.rand(4, generator=state.generator)
    state.restore(initial)
    state.assert_matches(initial)
    assert pointers == {name: tensor.data_ptr() for name, tensor in state.workspace._tensor_fields().items()}


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_real_cuda_fused_first_capture_and_replay_match_one_eager_step(monkeypatch):
    monkeypatch.setenv("KSTEP_CUDA_GRAPH", "1")
    state = _State("cuda")
    narrow = _narrow(state)
    initial = state.snapshot()
    state.advance()
    expected_once = state.snapshot()
    state.advance()
    expected_twice = state.snapshot()
    state.restore(initial)
    kwargs = dict(state_key="state", snapshot=state.snapshot, restore=state.restore, generators=(state.generator,))
    output = narrow.run_fused(state.advance, **kwargs)
    torch.accelerator.synchronize()
    state.assert_matches(expected_once)
    assert torch.equal(output, expected_once[2])
    assert narrow.captures == narrow.replays == 1
    narrow.run_fused(state.advance, **kwargs)
    torch.accelerator.synchronize()
    state.assert_matches(expected_twice)
    assert narrow.captures == 1
    assert narrow.replays == 2
    assert narrow.eager_calls == 0

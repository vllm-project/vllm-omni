# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 MRv2 request state: resuming a preempted request
and per-request seeds. The model is a stand-in that builds fixed prompt rows."""

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.fish_omni import model_state as ms
from vllm_omni.model_executor.models.fish_omni.fish_omni_model import FishOmniBuffers
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CODEBOOKS = 3
PROMPT = torch.tensor([[7, 0, 0, 0], [8, 0, 0, 0], [100, 0, 1, 2], [101, 1, 2, 3]])


class _Model:
    frame_width = 1 + CODEBOOKS
    max_top_k = 64

    def __init__(self) -> None:
        self.buffers = FishOmniBuffers(
            max_num_tokens=16,
            max_num_seqs=4,
            max_model_len=32,
            num_codebooks=CODEBOOKS,
            hidden_size=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.buffers.noise = torch.zeros((4, 5, 8))

    def build_frames(self, info, device):
        assert info["text"] == "hello"
        return PROMPT.clone()


@pytest.fixture
def make_state(monkeypatch):
    monkeypatch.setattr(OmniModelState, "add_request", lambda self, index, data: None)
    monkeypatch.setattr(OmniModelState, "on_request_preempted", lambda self, req_id, index: None)
    monkeypatch.setattr(OmniModelState, "on_requests_finished", lambda self, req_ids: None)

    def make() -> ms.FishOmniModelState:
        state = ms.FishOmniModelState.__new__(ms.FishOmniModelState)
        state.model = _Model()
        state.buffers = state.model.buffers
        state.device = torch.device("cpu")
        state.max_num_seqs = 4
        state.frame_width = state.model.frame_width
        state.prompts = {}
        state.prompt_len = np.zeros(4, dtype=np.int64)
        state.base_prompt_len = np.zeros(4, dtype=np.int64)
        state._pending_slots = {}
        state._pending_history = {}
        state._warned_top_k = False
        state._seeds = {}
        state._seed_generator = None
        state._suspended = {}
        state.intermediate_buffer = SimpleNamespace(buffers={i: {"text": "hello"} for i in range(4)})
        return state

    return make


def _request(req_id: str, generated: Sequence[int] = (), seed: int | None = None):
    prompt = [1] * PROMPT.shape[0]
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt,
        prefill_token_ids=prompt + list(generated),
        sampling_params=SimpleNamespace(temperature=1.0, top_k=30, top_p=0.9, seed=seed),
    )


def test_new_request_stages_its_prompt_rows(make_state):
    state = make_state()
    state.add_request(1, _request("a"))
    assert np.array_equal(state.prompts[1], PROMPT.numpy())
    assert state.prompt_len[1] == PROMPT.shape[0]
    _, (top_k, generated), ras = state._pending_slots[1]
    assert (top_k, generated) == (30, 0)
    assert ras[-PROMPT.shape[0] :] == PROMPT[:, 0].tolist()


def test_preempted_request_resumes_with_its_generated_frames(make_state):
    state = make_state()
    state.add_request(1, _request("a"))
    codes = torch.tensor([[5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=torch.int16)
    state.buffers.slot_history[1, :3] = codes
    state.buffers.slot_generated[1] = 3
    state.on_request_preempted("a", 1)
    assert state._suspended["a"].history.device.type == "cpu"
    assert state._suspended["a"].count.device.type == "cpu"
    # Another request takes the slot before "a" comes back.
    state.buffers.slot_history[1].zero_()
    state.buffers.slot_generated[1] = 0

    main = [105, 108, 111]
    state.add_request(2, _request("a", generated=main))
    rows = state.prompts[2]
    assert rows.shape == (PROMPT.shape[0] + 3, 1 + CODEBOOKS)
    assert np.array_equal(rows[: PROMPT.shape[0]], PROMPT.numpy())
    assert rows[PROMPT.shape[0] :, 0].tolist() == main
    assert np.array_equal(rows[PROMPT.shape[0] :, 1:], codes.numpy())
    assert state.prompt_len[2] == PROMPT.shape[0] + 3
    # Its next sampled frame is frame 3, which keys its seeded noise.
    assert state.base_prompt_len[2] == PROMPT.shape[0]
    _, (_, generated), ras = state._pending_slots[2]
    assert generated == 3
    assert ras[-3:] == main
    # The resumed slot gets its code history back, for a later preemption.
    assert list(state._pending_history) == [2] and torch.equal(state._pending_history[2], codes)
    state.on_requests_finished({"a"})
    assert "a" not in state._suspended


def test_slot_reused_before_a_step_keeps_only_the_newest_request(make_state):
    """A request added and finished before any step (e.g. aborted at once) leaves
    nothing for the next request in its slot: one slot write, that request's own
    values, no code history, and the finished request's seed is gone."""
    state = make_state()
    state.add_request(1, _request("a"))
    state.buffers.slot_history[1, :2] = 1
    state.buffers.slot_generated[1] = 2
    state.on_request_preempted("a", 1)
    state.add_request(0, _request("a", generated=[105, 106], seed=7))
    assert 0 in state._pending_history and "a" in state._seeds
    state.on_requests_finished({"a"})

    later = _request("b")
    later.sampling_params = SimpleNamespace(temperature=0.5, top_k=8, top_p=0.7, seed=None)
    state.add_request(0, later)
    (temperature, top_p), (top_k, generated), _ = state._pending_slots[0]
    assert (temperature, top_p, top_k, generated) == (0.5, 0.7, 8, 0)
    assert 0 not in state._pending_history
    assert "a" not in state._seeds and "a" not in state._suspended


def test_resume_without_kept_codes_fails_loudly(make_state):
    state = make_state()
    with pytest.raises(RuntimeError, match="without the codes"):
        state.add_request(0, _request("b", generated=[105]))


def test_resume_needs_every_generated_frame(make_state):
    state = make_state()
    state.add_request(0, _request("c"))
    state.buffers.slot_generated[0] = 1
    state.on_request_preempted("c", 0)
    with pytest.raises(RuntimeError, match="needs 2 generated frames"):
        state.add_request(0, _request("c", generated=[105, 106]))


def test_finished_request_drops_kept_codes_and_generator(make_state):
    state = make_state()
    state.add_request(0, _request("d", seed=3))
    state.on_request_preempted("d", 0)
    state.on_requests_finished({"d"})
    assert "d" not in state._suspended and "d" not in state._seeds


def test_seeded_noise_depends_only_on_seed_and_frame(make_state):
    """The noise of a seeded request's frame k is the same whatever its batch row,
    the other requests, the global RNG, or how many steps it took to get there."""
    first, second = make_state(), make_state()
    first.add_request(0, _request("s", seed=1234))
    second.add_request(3, _request("s", seed=1234))
    second.add_request(1, _request("u"))
    yes = np.array([True, True])

    torch.manual_seed(0)
    first._draw_noise(["s"], np.array([5]), yes)
    torch.manual_seed(1)
    second._draw_noise(["u", "s"], np.array([9, 5]), yes)
    assert torch.equal(first.buffers.noise[0], second.buffers.noise[1])
    assert not torch.equal(second.buffers.noise[0], second.buffers.noise[1])

    # Another frame of the same request, or the same frame of another seed, differs.
    frame_five = first.buffers.noise[0].clone()
    first._draw_noise(["s"], np.array([6]), yes)
    assert not torch.equal(first.buffers.noise[0], frame_five)
    other = make_state()
    other.add_request(0, _request("s", seed=4321))
    torch.manual_seed(0)
    other._draw_noise(["s"], np.array([5]), yes)
    assert not torch.equal(other.buffers.noise[0], frame_five)


def test_seed_is_ignored_on_rows_that_sample_nothing(make_state):
    seeded, plain = make_state(), make_state()
    seeded.add_request(0, _request("s", seed=7))
    plain.add_request(0, _request("p"))
    for state, req_id in ((seeded, "s"), (plain, "p")):
        torch.manual_seed(3)
        state._draw_noise([req_id], np.array([0]), np.array([False]))
    assert torch.equal(seeded.buffers.noise, plain.buffers.noise)


@pytest.mark.parametrize(
    "value, ok",
    [(None, True), (32, True), (64, True), (4096, True), (100, False), (16, False), (1, False), (8192, False)],
)
def test_max_top_k_is_a_power_of_two_from_the_ras_top_k_to_the_codebook(value, ok):
    """The RAS re-draw keeps top-k 30, so fewer than 32 candidates would cut it."""
    from vllm_omni.model_executor.models.fish_omni.fish_omni_model import max_top_k

    config = SimpleNamespace() if value is None else SimpleNamespace(fish_omni_max_top_k=value)
    if ok:
        assert max_top_k(config, 4096) == (64 if value is None else value)
    else:
        with pytest.raises(ValueError, match="power of two"):
            max_top_k(config, 4096)


def test_npu_device_selects_ascend_model_state(monkeypatch):
    sentinel = object()
    from vllm_omni.platforms.npu.models import fish_omni_model_state as npu_ms

    monkeypatch.setattr(npu_ms, "NPUFishOmniModelState", lambda *args, **kwargs: sentinel)
    assert ms.create_fish_omni_model_state(None, None, None, SimpleNamespace(type="npu")) is sentinel


def test_cuda_device_keeps_the_upstream_model_state(monkeypatch):
    monkeypatch.setattr(ms.FishOmniModelState, "__init__", lambda self, *args, **kwargs: None)
    state = ms.create_fish_omni_model_state(None, None, None, SimpleNamespace(type="cuda"))
    assert type(state) is ms.FishOmniModelState
    assert not getattr(state, "uses_ascend_attn_metadata", False)


def test_sampling_noise_is_broadcast_across_tensor_parallel_ranks(make_state):
    state = make_state()
    state.add_request(0, _request("s", seed=1))
    seen = {}

    def broadcast(tensor, src=0):
        seen["src"] = src
        seen["shape"] = tuple(tensor.shape)
        return tensor

    with (
        patch.object(ms, "_broadcast_tp_tensor", side_effect=lambda tensor: seen.update(called=True) or None),
    ):
        state._draw_noise(["s"], np.array([1]), np.array([True]))
    assert seen["called"] is True

    with (
        patch("vllm.distributed.parallel_state.model_parallel_is_initialized", return_value=True),
        patch("vllm.distributed.parallel_state.get_tensor_model_parallel_world_size", return_value=2),
        patch("vllm.distributed.parallel_state.get_tp_group") as group,
    ):
        group.return_value.broadcast = broadcast
        ms._broadcast_tp_tensor(state.buffers.noise)
    assert seen["src"] == 0
    assert seen["shape"] == tuple(state.buffers.noise.shape)


def test_pipeline_capacity_preemption_restores_frames_after_slot_reuse(make_state):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request, RequestStatus

    from tests.core.sched.test_omni_recompute_preemption import _output, _scheduler
    from vllm_omni.model_executor.models.fish_omni.pipeline import FISH_OMNI_PIPELINE

    stage = FISH_OMNI_PIPELINE.get_stage(0)
    scheduler = _scheduler(stage.recompute_preemption, block_size=4, num_blocks=3)
    scheduler.use_v2_model_runner = True
    state = make_state()
    # Hold one of two usable blocks: prefill fits, first decode cannot extend.
    blocker = Request("blocker", [1] * 4, SamplingParams(max_tokens=8), None)
    assert scheduler.kv_cache_manager.allocate_slots(blocker, 4) is not None
    request = Request("a", [1] * 4, SamplingParams(max_tokens=8), None)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    state.add_request(1, prefill.scheduled_new_reqs[0])
    codes = torch.tensor([[5, 6, 7]], dtype=torch.int16)
    state.buffers.slot_history[1, :1] = codes
    state.buffers.slot_generated[1] = 1
    scheduler.update_from_output(prefill, _output(request))

    preempt = scheduler.schedule()
    assert request.status == RequestStatus.PREEMPTED
    assert preempt.preempted_req_ids == {"a"}
    assert scheduler.update_from_output(preempt, _output()) == {}
    state.on_request_preempted("a", 1)
    # Reuse the former slot for another request before resuming in a new slot.
    state.add_request(1, _request("other"))
    state.buffers.slot_history[1].fill_(99)
    state.buffers.slot_generated[1] = 0
    scheduler.kv_cache_manager.free(blocker)

    resumed = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert resumed.num_scheduled_tokens == {"a": 5}
    state.add_request(2, resumed.scheduled_new_reqs[0])
    assert np.array_equal(state.prompts[2][:4], PROMPT.numpy())
    assert state.prompts[2][4:, 0].tolist() == list(request.output_token_ids)
    assert np.array_equal(state.prompts[2][4:, 1:], codes.numpy())
    assert torch.equal(state._pending_history[2], codes)
    assert state._pending_slots[2][1][1] == 1
    assert "a" not in state._suspended
    outputs = scheduler.update_from_output(resumed, _output(request))
    assert all(output.finish_reason is None for batch in outputs.values() for output in batch.outputs)

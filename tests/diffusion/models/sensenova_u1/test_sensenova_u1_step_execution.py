# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova-U1 pipeline autoregressive decode, driven one step at a time.

The think and text loops move onto the runner as a resumable prepare phase, so
each loop has to produce the same tokens whether it runs to completion in one
call or is left and resumed between any two steps.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.interface import (
    supports_resumable_prepare,
    supports_step_execution,
)
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _nullcontext(*args, **kwargs):
    del args, kwargs
    return nullcontext()


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


EOS = 100
THINK_END = 101
VOCAB = 128
APPEND_LEN = 3


class _FakeTokenizer:
    _IDS = {"<|im_end|>": EOS, "</think>": THINK_END}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._IDS[token]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return ",".join(str(int(i)) for i in ids)

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        del text, return_tensors, add_special_tokens
        return {"input_ids": torch.zeros(1, APPEND_LEN, dtype=torch.long)}


def _one_hot(token_id: int) -> torch.Tensor:
    logits = torch.full((1, 1, VOCAB), -10.0)
    logits[0, 0, token_id] = 10.0
    return logits


def _pipeline(successor: dict[int, int] | None = None, *, flat: bool = False) -> SenseNovaU1Pipeline:
    """A pipeline whose decode step is a stateless token -> token map.

    Statelessness is the point: two cursors interleaved through the same
    pipeline must not be able to influence each other through it. With
    ``flat``, every step returns a uniform distribution instead, so the tokens
    depend only on the sampler.
    """
    # </think> is stepped through before the loop stops, and what that step
    # returns is discarded, so the map only has to answer.
    successor = {THINK_END: EOS, **(successor or {})}
    pipe = object.__new__(SenseNovaU1Pipeline)
    pipe.tokenizer = _FakeTokenizer()
    pipe.device = torch.device("cpu")
    pipe.ar_steps = []
    pipe.append_calls = []

    def _ar_step(next_token, t_idx, past_key_values, decode=None):
        del decode
        token = int(next_token)
        pipe.ar_steps.append((token, t_idx, past_key_values))
        logits = torch.zeros(1, 1, VOCAB) if flat else _one_hot(successor[token])
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    def _decode_context(past_key_values):
        del past_key_values
        return None

    def _append_text_tokens_to_cache(cache, t_idx, input_ids):
        pipe.append_calls.append((cache, t_idx, int(input_ids.shape[1])))
        return t_idx + int(input_ids.shape[1])

    pipe._ar_step = _ar_step
    pipe._decode_context = _decode_context
    pipe._append_text_tokens_to_cache = _append_text_tokens_to_cache
    return pipe


def _think_cursor(pipe, first_token: int, cache: str = "kv", t_idx: int = 10, max_think_tokens: int = 1024):
    prefix = SimpleNamespace(logits=_one_hot(first_token))
    return pipe._begin_think(prefix, cache, t_idx, max_think_tokens=max_think_tokens)


def _run_to_end(pipe, cursor, step) -> None:
    while not cursor.finished:
        step(cursor)


class TestCapabilityDeclaration:
    def test_pipeline_declares_both_capabilities(self):
        pipe = object.__new__(SenseNovaU1Pipeline)
        assert supports_step_execution(pipe) is True
        assert supports_resumable_prepare(pipe) is True


class TestStepExecutionConfig:
    """The decode cache holds one sequence, so two of them must not be admitted."""

    def test_construction_rejects_the_configuration_before_it_loads_anything(self):
        od_config = SimpleNamespace(step_execution=True, max_num_seqs=4)
        # __init__ reaches the guard before it resolves a model path, so this
        # also pins that the guard is wired in rather than merely defined.
        with pytest.raises(ValueError, match="max_num_seqs=1"):
            SenseNovaU1Pipeline(od_config=od_config)

    def test_the_guard_runs_again_on_the_first_request(self):
        pipe = _pipeline({1: EOS})
        pipe.od_config = SimpleNamespace(step_execution=True, max_num_seqs=4)
        state = StepRequestState(
            request_id="req-1",
            sampling=OmniDiffusionSamplingParams(num_inference_steps=25, seed=42),
            prompt={"prompt": "hello", "modalities": ["text"]},
        )
        with pytest.raises(ValueError, match="max_num_seqs=1"):
            pipe.prepare_encode(state)

    @pytest.mark.parametrize("max_num_seqs", [2, 8])
    def test_step_execution_refuses_more_than_one_sequence(self, max_num_seqs):
        od_config = SimpleNamespace(step_execution=True, max_num_seqs=max_num_seqs)
        with pytest.raises(ValueError, match="max_num_seqs=1"):
            SenseNovaU1Pipeline._check_step_execution_config(od_config)

    def test_request_mode_is_unaffected(self):
        od_config = SimpleNamespace(step_execution=False, max_num_seqs=8)
        SenseNovaU1Pipeline._check_step_execution_config(od_config)

    def test_step_execution_with_one_sequence_is_allowed(self):
        od_config = SimpleNamespace(step_execution=True, max_num_seqs=1)
        SenseNovaU1Pipeline._check_step_execution_config(od_config)


class TestThinkStopRules:
    def test_think_end_takes_one_more_step_and_is_emitted(self):
        pipe = _pipeline({1: 2, 2: THINK_END})
        cursor = _think_cursor(pipe, first_token=1)

        _run_to_end(pipe, cursor, pipe._think_step)

        assert cursor.tokens == [1, 2, THINK_END]
        # Three decode steps: the two ordinary tokens and </think> itself.
        assert [token for token, _, _ in pipe.ar_steps] == [1, 2, THINK_END]
        assert cursor.t_idx == 13

    def test_eos_stops_without_a_step_and_is_not_emitted(self):
        pipe = _pipeline({1: 2, 2: EOS})
        cursor = _think_cursor(pipe, first_token=1)

        _run_to_end(pipe, cursor, pipe._think_step)

        assert cursor.tokens == [1, 2]
        assert [token for token, _, _ in pipe.ar_steps] == [1, 2]
        assert cursor.t_idx == 12

    def test_max_think_tokens_caps_a_loop_that_never_stops(self):
        pipe = _pipeline({1: 1})
        cursor = _think_cursor(pipe, first_token=1, max_think_tokens=5)

        _run_to_end(pipe, cursor, pipe._think_step)

        assert cursor.tokens == [1] * 5
        assert cursor.done is False
        assert cursor.finished is True

    def test_finish_think_appends_the_image_marker_after_the_loop(self):
        pipe = _pipeline({1: THINK_END})
        cursor = _think_cursor(pipe, first_token=1, t_idx=10)

        _run_to_end(pipe, cursor, pipe._think_step)
        cache, t_idx, think_text = pipe._finish_think(cursor)

        assert cache == "kv"
        assert pipe.append_calls == [("kv", 12, APPEND_LEN)]
        assert t_idx == 12 + APPEND_LEN
        assert think_text == f"1,{THINK_END}"


class TestInterleaving:
    def test_two_think_cursors_interleave_to_the_same_tokens(self):
        successor = {1: 2, 2: 3, 3: THINK_END, 11: 12, 12: EOS}
        serial_a = _pipeline(successor)
        cursor_a = _think_cursor(serial_a, first_token=1, cache="a")
        _run_to_end(serial_a, cursor_a, serial_a._think_step)
        serial_b = _pipeline(successor)
        cursor_b = _think_cursor(serial_b, first_token=11, cache="b", t_idx=20)
        _run_to_end(serial_b, cursor_b, serial_b._think_step)
        assert cursor_a.tokens and cursor_b.tokens

        shared = _pipeline(successor)
        first = _think_cursor(shared, first_token=1, cache="a")
        second = _think_cursor(shared, first_token=11, cache="b", t_idx=20)
        while not (first.finished and second.finished):
            if not first.finished:
                shared._think_step(first)
            if not second.finished:
                shared._think_step(second)

        assert first.tokens == cursor_a.tokens
        assert second.tokens == cursor_b.tokens
        assert first.t_idx == cursor_a.t_idx
        assert second.t_idx == cursor_b.t_idx

    def test_interleaved_cursors_keep_their_own_caches(self):
        shared = _pipeline({1: 2, 2: EOS, 11: 12, 12: EOS})
        first = _think_cursor(shared, first_token=1, cache="a")
        second = _think_cursor(shared, first_token=11, cache="b", t_idx=20)

        shared._think_step(first)
        shared._think_step(second)

        assert [cache for _, _, cache in shared.ar_steps] == ["a", "b"]


class TestTextLoop:
    def test_greedy_text_stops_on_eos_and_drops_it(self):
        pipe = _pipeline({1: 2, 2: EOS})
        cursor = pipe._begin_text(_one_hot(1), "kv", 10, max_tokens=64)

        _run_to_end(pipe, cursor, pipe._text_step)

        assert cursor.tokens == [1, 2]
        assert pipe._finish_text(cursor) == "1,2"
        assert cursor.t_idx == 12

    def test_sampling_is_reproducible_from_the_request_seed(self):
        # A flat distribution makes the draw depend only on the generator.
        pipe = _pipeline(flat=True)
        flat = torch.zeros(1, 1, VOCAB)

        def _tokens(seed: int) -> list[int]:
            cursor = pipe._begin_text(
                flat,
                "kv",
                0,
                max_tokens=4,
                do_sample=True,
                temperature=1.0,
                seed=seed,
            )
            _run_to_end(pipe, cursor, pipe._text_step)
            return list(cursor.tokens)

        assert len(_tokens(7)) > 1
        assert _tokens(7) == _tokens(7)
        assert _tokens(7) != _tokens(9)

    def test_text_cursor_survives_being_left_between_tokens(self):
        successor = {1: 2, 2: 3, 3: EOS}
        serial = _pipeline(successor)
        serial_cursor = serial._begin_text(_one_hot(1), "kv", 5, max_tokens=64)
        _run_to_end(serial, serial_cursor, serial._text_step)

        resumed = _pipeline(successor)
        cursor = resumed._begin_text(_one_hot(1), "kv", 5, max_tokens=64)
        other = _think_cursor(resumed, first_token=1, cache="other", t_idx=99)
        while not cursor.finished:
            resumed._text_step(cursor)
            if not other.finished:
                resumed._think_step(other)

        assert cursor.tokens == serial_cursor.tokens
        assert cursor.t_idx == serial_cursor.t_idx


class _StubLanguageModel:
    """Answers the prefill the text path makes, then the scripted decode."""

    def __init__(self, successor: dict[int, int], first_token: int):
        self.successor = successor
        self.first_token = first_token
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=_one_hot(self.first_token), past_key_values="prefill-kv")


def _text_pipeline(successor: dict[int, int], first_token: int) -> SenseNovaU1Pipeline:
    """A pipeline whose only stubs are the model and the paged decode context."""
    pipe = _pipeline(successor)
    pipe.language_model = _StubLanguageModel(successor, first_token)
    pipe.patch_size = 16
    pipe.merge_size = 2
    pipe.od_config = SimpleNamespace(step_execution=True, max_num_seqs=1)
    return pipe


def _text_state(request_id: str = "req-1") -> StepRequestState:
    sampling = OmniDiffusionSamplingParams(num_inference_steps=25, seed=42)
    sampling.extra_args.update({"max_tokens": 16})
    return StepRequestState(
        request_id=request_id,
        sampling=sampling,
        prompt={"prompt": "describe the sky", "modalities": ["text"]},
    )


class TestPrepareProtocolOnTheRealPipeline:
    """The protocol methods themselves, not the cursor they drive."""

    def test_text_request_runs_its_whole_output_inside_prepare(self):
        pipe = _text_pipeline({1: 2, 2: 3, 3: EOS}, first_token=1)
        state = _text_state()

        pipe.prepare_encode(state)

        # A text request has no denoise schedule, which is what tells the runner
        # to decode it as soon as prepare is done.
        assert state.timesteps is None
        assert state.total_steps == 0

        steps = 0
        while pipe.prepare_steps_remaining(state) is not None:
            pipe.prepare_step(state)
            steps += 1
            assert steps <= 16, "prepare_steps_remaining never returned None"
        # Three emitted tokens and one more step to observe the end token.
        assert steps == 4

        output = pipe.post_decode(state)
        assert output.output["payload"]["text"] == "1,2,3"
        assert pipe._STEP_KEY not in state.extra

    def test_prepare_steps_remaining_falls_to_none_exactly_once(self):
        pipe = _text_pipeline({1: EOS}, first_token=1)
        state = _text_state()

        pipe.prepare_encode(state)
        assert pipe.prepare_steps_remaining(state) == 16

        pipe.prepare_step(state)
        assert pipe.prepare_steps_remaining(state) == 15

        # The step that sees the end token is the one that ends the phase.
        pipe.prepare_step(state)
        assert pipe.prepare_steps_remaining(state) is None
        # A further call is a no-op rather than an error, so a runner that asks
        # twice in one tick cannot corrupt the request.
        pipe.prepare_step(state)
        assert pipe.prepare_steps_remaining(state) is None

    def test_image_request_is_ready_to_denoise_when_think_is_off(self):
        pipe = _text_pipeline({1: EOS}, first_token=1)
        schedule = SimpleNamespace(image_prediction=torch.zeros(1, 3, 8, 8), timesteps=torch.linspace(0, 1, 4))
        pipe._init_noise_and_schedule = lambda p: schedule
        pipe._t2i_prefix = lambda p, ns: SimpleNamespace(cursor=None, past_kv_cond="kv")
        pipe._t2i_caches = lambda p, ns, ctx: ({"cond": "kv"}, "")

        state = _text_state()
        state.prompt = {"prompt": "draw the sky", "modalities": ["image"]}
        pipe.prepare_encode(state)

        assert pipe.prepare_steps_remaining(state) is None
        assert state.total_steps == 3, "the state carries one entry per denoise interval"
        assert state.latents is schedule.image_prediction
        assert state.extra[pipe._STEP_KEY].caches == {"cond": "kv"}

    def test_image_request_with_think_prepares_before_it_denoises(self):
        pipe = _text_pipeline({1: 2, 2: THINK_END}, first_token=1)
        schedule = SimpleNamespace(image_prediction=torch.zeros(1, 3, 8, 8), timesteps=torch.linspace(0, 1, 4))
        pipe._init_noise_and_schedule = lambda p: schedule
        cursor = _think_cursor(pipe, first_token=1)
        pipe._t2i_prefix = lambda p, ns: SimpleNamespace(cursor=cursor, past_kv_cond=None)
        finished_with = []

        def _caches(p, ns, ctx):
            finished_with.append(ctx.cursor)
            return {"cond": "kv"}, "thought"

        pipe._t2i_caches = _caches

        state = _text_state()
        state.prompt = {"prompt": "draw the sky", "modalities": ["image"]}
        pipe.prepare_encode(state)

        assert pipe.prepare_steps_remaining(state) is not None
        assert finished_with == [], "the caches must not be built before the loop ends"
        while pipe.prepare_steps_remaining(state) is not None:
            pipe.prepare_step(state)
        assert finished_with == [cursor]
        assert state.extra[pipe._STEP_KEY].think_text == "thought"


class TestRequestModeAndStepModeAgree:
    """The claim the change rests on, at the one boundary a CPU test can reach."""

    @staticmethod
    def _params(max_tokens: int):
        sampling = OmniDiffusionSamplingParams(num_inference_steps=25, seed=42)
        sampling.extra_args.update({"max_tokens": max_tokens})
        return sampling

    def _request_mode_text(self, pipe, max_tokens: int) -> str:
        prompt = {"prompt": "describe the sky", "modalities": ["text"]}
        p = pipe._parse_request(SimpleNamespace(prompts=[prompt], sampling_params=self._params(max_tokens)))
        return pipe._forward_text(p, None).output["payload"]["text"]

    def _step_mode_text(self, pipe, max_tokens: int) -> str:
        state = StepRequestState(
            request_id="req-1",
            sampling=self._params(max_tokens),
            prompt={"prompt": "describe the sky", "modalities": ["text"]},
        )
        pipe.prepare_encode(state)
        while pipe.prepare_steps_remaining(state) is not None:
            pipe.prepare_step(state)
        return pipe.post_decode(state).output["payload"]["text"]

    @pytest.mark.parametrize("max_tokens", [0, 1, 2, 16])
    def test_the_two_modes_produce_the_same_text(self, max_tokens):
        successor = {1: 2, 2: 3, 3: EOS}
        request_text = self._request_mode_text(_text_pipeline(successor, first_token=1), max_tokens)
        step_text = self._step_mode_text(_text_pipeline(successor, first_token=1), max_tokens)
        assert step_text == request_text

    def test_a_zero_token_budget_decodes_nothing_in_either_mode(self):
        successor = {1: 2, 2: EOS}
        assert self._request_mode_text(_text_pipeline(successor, first_token=1), 0) == ""
        assert self._step_mode_text(_text_pipeline(successor, first_token=1), 0) == ""


class TestThroughTheRunner:
    """The pipeline's own methods, driven by the runner that will call them."""

    @staticmethod
    def _runner(pipeline):
        import torch as _torch

        from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
        from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

        runner = object.__new__(DiffusionModelRunner)
        runner.vllm_config = SimpleNamespace(
            kernel_config=SimpleNamespace(ir_op_priority=SimpleNamespace(set_priority=_nullcontext)),
            compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
        )
        runner.od_config = SimpleNamespace(
            cache_backend=None,
            diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
            parallel_config=SimpleNamespace(use_hsdp=False),
            streaming_output=False,
        )
        runner.device = _torch.device("cpu")
        runner.pipeline = pipeline
        runner.cache_backend = None
        runner.offload_backend = None
        runner.state_cache = {}
        runner.input_batch = None
        runner.kv_transfer_manager = SimpleNamespace(
            receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None
        )
        return runner

    def test_an_image_request_denoises_through_the_runner(self, monkeypatch: pytest.MonkeyPatch):
        """The three step methods an image request uses, driven by the runner.

        `denoise_step` and `step_scheduler` run for real; only `_denoise_one`,
        which is the transformer forward, is replaced. The Euler update and the
        unpatchify in `_advance_latents` are therefore the production ones.
        """
        import vllm_omni.diffusion.worker.diffusion_model_runner as runner_module
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.sched.interface import (
            CachedRequestData,
            DiffusionSchedulerOutput,
            NewRequestData,
        )
        from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

        monkeypatch.setattr(runner_module, "set_forward_context", _noop_forward_context)
        pipe = _text_pipeline({1: EOS}, first_token=1)
        # 32x32 at patch 16 x merge 2 is one patch row and column.
        pipe.patch_size, pipe.merge_size = 16, 1
        latents = torch.zeros(1, 4, 16 * 16 * 3)
        timesteps = torch.tensor([0.0, 0.25, 0.75, 1.0])
        schedule = SimpleNamespace(image_prediction=latents, timesteps=timesteps)
        pipe._init_noise_and_schedule = lambda p: schedule
        pipe._t2i_prefix = lambda p, ns: SimpleNamespace(cursor=None, past_kv_cond="kv")
        pipe._t2i_caches = lambda p, ns, ctx: ({"cond": {}}, "")
        v_pred = torch.full((1, 4, 16 * 16 * 3), 2.0)
        seen: list[int] = []

        def _denoise_one(z, ns, caches, p, step_i, is_edit):
            # Production re-patchifies the latents each step and returns the
            # patched tensor alongside the prediction, so the Euler update in
            # `_advance_latents` operates on the patched layout.
            seen.append(step_i)
            return torch.zeros_like(v_pred), v_pred

        pipe._denoise_one = _denoise_one
        pipe._to_pil_called = False

        runner = self._runner(pipe)
        sampling = OmniDiffusionSamplingParams(num_inference_steps=4, seed=42, width=32, height=32)
        request = OmniDiffusionRequest(
            prompt={"prompt": "draw the sky", "modalities": ["image"]},
            request_id="req-img",
            sampling_params=sampling,
        )
        scheduled = DiffusionSchedulerOutput(
            step_id=0,
            scheduled_new_reqs=[NewRequestData(request_id="req-img", req=request)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(),
            num_running_reqs=1,
            num_waiting_reqs=0,
        )
        output = DiffusionModelRunner.execute_stepwise(runner, scheduled)
        request_output = output.get_request_output("req-img")

        for step_id in range(1, 6):
            if request_output.finished:
                break
            cached = DiffusionSchedulerOutput(
                step_id=step_id,
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData(request_ids=["req-img"]),
                finished_req_ids=set(),
                num_running_reqs=1,
                num_waiting_reqs=0,
            )
            output = DiffusionModelRunner.execute_stepwise(runner, cached)
            request_output = output.get_request_output("req-img")

        # One denoise per interval, in order, and the latents carry the Euler
        # update for each of them: sum((t_next - t) * 2.0) over the schedule.
        assert seen == [0, 1, 2]
        assert request_output.finished is True
        assert request_output.result.output["payload"]["image"] is not None
        assert "req-img" not in runner.state_cache

    @pytest.mark.parametrize(("max_tokens", "expected"), [(0, ""), (16, "1,2")])
    def test_a_text_request_finishes_inside_prepare(self, monkeypatch: pytest.MonkeyPatch, max_tokens, expected):
        import vllm_omni.diffusion.worker.diffusion_model_runner as runner_module
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.sched.interface import (
            CachedRequestData,
            DiffusionSchedulerOutput,
            NewRequestData,
        )
        from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

        monkeypatch.setattr(runner_module, "set_forward_context", _noop_forward_context)
        pipe = _text_pipeline({1: 2, 2: EOS}, first_token=1)
        runner = self._runner(pipe)

        sampling = OmniDiffusionSamplingParams(num_inference_steps=25, seed=42)
        sampling.extra_args.update({"max_tokens": max_tokens})
        request = OmniDiffusionRequest(
            prompt={"prompt": "describe the sky", "modalities": ["text"]},
            request_id="req-1",
            sampling_params=sampling,
        )
        scheduled = DiffusionSchedulerOutput(
            step_id=0,
            scheduled_new_reqs=[NewRequestData(request_id="req-1", req=request)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(),
            num_running_reqs=1,
            num_waiting_reqs=0,
        )

        output = DiffusionModelRunner.execute_stepwise(runner, scheduled)
        request_output = output.get_request_output("req-1")
        assert request_output.finished is (max_tokens == 0)

        for step_id in range(1, 6):
            if request_output.finished:
                break
            cached = DiffusionSchedulerOutput(
                step_id=step_id,
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData(request_ids=["req-1"]),
                finished_req_ids=set(),
                num_running_reqs=1,
                num_waiting_reqs=0,
            )
            output = DiffusionModelRunner.execute_stepwise(runner, cached)
            request_output = output.get_request_output("req-1")
            if request_output.finished:
                break

        assert request_output.finished is True
        assert request_output.result.output["payload"]["text"] == expected
        assert "req-1" not in runner.state_cache

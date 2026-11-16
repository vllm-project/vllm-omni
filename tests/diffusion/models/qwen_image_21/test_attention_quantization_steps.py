# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise step selectors through both Qwen2.1 denoising entry points."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import get_forward_context, override_forward_context, set_forward_context
from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import QwenImage21CFGParallelMixin
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
    QwenImage21Pipeline,
    get_qwen_image_21_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput, StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.quantization.mxfp4_config import _is_w4a8_fallback_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _attention(config: OmniDiffusionConfig) -> Attention:
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer._kv_cache_dtype = config.diffusion_kv_cache_dtype
    layer._kv_cache_skip_steps = config.diffusion_kv_cache_skip_step_indices
    layer._kv_cache_skip_layers = None
    layer._disable_kv_quant = False
    return layer


class _Transformer:
    causal_condition = False
    transformer_blocks = ()

    def __init__(self) -> None:
        self.releases = 0

    def release_request_geometry(self) -> None:
        self.releases += 1


class _Scheduler:
    def set_begin_index(self, index: int) -> None:
        assert index == 0


class _Loop(QwenImage21CFGParallelMixin):
    def __init__(self, attention: Attention, fail_at: int | None = None) -> None:
        self.attention = attention
        self.transformer = _Transformer()
        self.scheduler = _Scheduler()
        self.interrupt = False
        self._progress_bar_config = {"disable": True}
        self.seen: list[tuple[int | None, str]] = []
        self.fail_at = fail_at

    def predict_noise(self, **kwargs: object) -> torch.Tensor:
        step = get_forward_context().denoise_step_idx
        metadata = self.attention._with_kv_cache_dtype(None)
        self.seen.append((step, metadata.extra["kv_cache_dtype"]))
        if step == self.fail_at and self.fail_at is not None:
            raise RuntimeError("denoise failed")
        hidden_states = kwargs["hidden_states"]
        assert isinstance(hidden_states, torch.Tensor)
        return hidden_states

    def scheduler_step_maybe_with_cfg(self, noise_pred, timestep, latents, do_true_cfg):
        return latents


def _diffuse(pipe: _Loop, do_true_cfg: bool = False) -> torch.Tensor:
    return pipe.diffuse(
        prompt_embeds=torch.zeros(1, 1, 4),
        prompt_embeds_mask=torch.ones(1, 1, dtype=torch.bool),
        negative_prompt_embeds=torch.zeros(1, 1, 4) if do_true_cfg else None,
        negative_prompt_embeds_mask=torch.ones(1, 1, dtype=torch.bool) if do_true_cfg else None,
        img_mask=torch.ones(1, 3, dtype=torch.bool),
        negative_img_mask=torch.ones(1, 3, dtype=torch.bool) if do_true_cfg else None,
        latents=torch.zeros(1, 2, 4),
        image_latents=None,
        img_shapes=[[(1, 1, 2)]],
        timesteps=torch.tensor([999.0, 700.0, 200.0, 1.0]),
        do_true_cfg=do_true_cfg,
        true_cfg_scale=4.0 if do_true_cfg else 1.0,
    )


@pytest.mark.parametrize("do_true_cfg", [False, True])
@pytest.mark.parametrize("previous_step", [None, 0, 9])
def test_loop_publishes_step_indices_and_restores_between_requests(do_true_cfg, previous_step):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2")
    pipe = _Loop(_attention(config))
    expected = [(0, "float"), (1, "mxfp8"), (2, "float"), (3, "mxfp8")]
    if do_true_cfg:
        expected = [entry for entry in expected for _ in range(2)]
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=previous_step):
        for _ in range(2):
            pipe.seen.clear()
            _diffuse(pipe, do_true_cfg)
            assert pipe.seen == expected
            assert get_forward_context().denoise_step_idx == previous_step
    assert pipe.transformer.releases == 2


@pytest.mark.parametrize("previous_step", [None, 0, 9])
def test_loop_restores_step_context_after_failure(previous_step):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _Loop(_attention(config), fail_at=1)
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=previous_step):
        with pytest.raises(RuntimeError, match="denoise failed"):
            _diffuse(pipe)
        assert get_forward_context().denoise_step_idx == previous_step
    assert pipe.transformer.releases == 1


def _state(request_id: str, step: int, fill: float, rows: int = 1) -> StepRequestState:
    return StepRequestState(
        request_id=request_id,
        sampling=OmniDiffusionSamplingParams(true_cfg_scale=1.0),
        prompt_embeds=torch.ones(rows, 1, 4),
        prompt_embeds_mask=torch.ones(rows, 1, dtype=torch.bool),
        latents=torch.full((rows, 2, 4), fill),
        timesteps=torch.tensor([999.0, 700.0, 200.0, 1.0]),
        step_index=step,
        img_shapes=[[(1, 1, 2)]] * rows,
        extra={"image_pad_mask": torch.ones(rows, 1, dtype=torch.bool), "image_latents": None, "kv_cache": None},
    )


def _step_pipeline(config: OmniDiffusionConfig) -> QwenImage21Pipeline:
    pipe = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(pipe)
    pipe.od_config = config
    pipe.transformer = nn.Module()
    pipe._interrupt = False
    pipe._attention_kwargs = {}
    return pipe


@pytest.mark.parametrize("step,expected", [(0, "float"), (1, "mxfp8"), (2, "float"), (3, "mxfp8")])
def test_step_mode_publishes_request_step(monkeypatch, step, expected):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2")
    pipe = _step_pipeline(config)
    attention = _attention(config)
    seen = []

    def predict(*args, **kwargs):
        seen.append(
            (get_forward_context().denoise_step_idx, attention._with_kv_cache_dtype(None).extra["kv_cache_dtype"])
        )
        return args[2]["hidden_states"]

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        result = pipe.denoise_step(InputBatch.make_batch([_state("a", step, 1.0)]))
    assert result.shape == (1, 2, 4)
    assert seen == [(step, expected)]


def test_scheduled_requests_publish_the_common_step(monkeypatch, tmp_path):
    # Use the Qwen admission hook and the real scheduler. Only encoding and
    # device computation are replaced with small CPU tensors.
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae/config.json").write_text('{"scale_factor_spatial": 16}')
    config = OmniDiffusionConfig(
        model=str(tmp_path), max_num_seqs=3, diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2"
    )
    preprocess = get_qwen_image_21_pre_process_func(config)
    scheduler = StepScheduler()
    scheduler.initialize(config)
    pipe = _step_pipeline(config)
    attention = _attention(config)
    observed = []

    def add(request_id, total):
        params = OmniDiffusionSamplingParams(num_inference_steps=total, height=1024, width=1024, seed=42)
        scheduler.add_request(
            preprocess(OmniDiffusionRequest(prompt="teapot", request_id=request_id, sampling_params=params))
        )

    def predict(_cfg, _scale, positive, *args, **kwargs):
        hidden = positive["hidden_states"]
        observed.append(
            (
                get_forward_context().denoise_step_idx,
                attention._with_kv_cache_dtype(None).extra["kv_cache_dtype"],
                _is_w4a8_fallback_step([2]),
                hidden.shape[0],
            )
        )
        return hidden

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    monkeypatch.setattr(pipe, "scheduler_step_maybe_with_cfg", lambda noise, t, latents, cfg, **kw: latents)
    states = {}
    waves = []
    add("long", 4)
    add("short", 2)
    for tick in range(6):
        if tick == 1:
            add("late", 2)
        scheduled = scheduler.schedule()
        for new in scheduled.scheduled_new_reqs:
            state = _state(new.request_id, 0, 1.0)
            state.timesteps = state.timesteps[: new.req.sampling_params.num_inference_steps]
            states[new.request_id] = state
        batch = [states[rid] for rid in scheduled.scheduled_request_ids]
        waves.append([(state.request_id, state.step_index) for state in batch])
        with set_forward_context(omni_diffusion_config=config):
            noise = pipe.denoise_step(InputBatch.make_batch(batch))
        outputs = []
        for row, state in enumerate(batch):
            pipe.step_scheduler(state, noise[row : row + 1])
            outputs.append(
                RunnerOutput(request_id=state.request_id, step_index=state.step_index, finished=state.denoise_completed)
            )
        scheduler.update_from_output(scheduled, BatchRunnerOutput.from_list(outputs))

    assert waves == [
        [("long", 0), ("short", 0)],
        [("long", 1), ("short", 1)],
        [("long", 2)],
        [("long", 3)],
        [("late", 0)],
        [("late", 1)],
    ]
    assert observed == [
        (0, "float", False, 2),
        (1, "mxfp8", False, 2),
        (2, "float", True, 1),
        (3, "mxfp8", False, 1),
        (0, "float", False, 1),
        (1, "mxfp8", False, 1),
    ]


@pytest.mark.parametrize("different_prefix_lengths", [False, True])
def test_decode_checks_the_common_index_after_prefix_grouping(monkeypatch, different_prefix_lengths):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)
    second_step = 2 if different_prefix_lengths else 3
    states = [_state("a", 3, 1.0), _state("b", second_step, 2.0, rows=2), _state("c", 3, 3.0)]
    for state in states:
        prefix = 2 if different_prefix_lengths and state.request_id == "b" else 1
        state.extra["prefill_prompt_seq_len"] = prefix
        state.extra["kv_cache"] = [
            {"cond": {part: torch.zeros(state.latents.shape[0], prefix, 4) for part in ("key", "value")}}
        ]
    calls = []

    def predict(_cfg, _scale, positive, *args, **kwargs):
        hidden = positive["hidden_states"]
        calls.append((get_forward_context().denoise_step_idx, hidden[:, 0, 0].tolist()))
        return hidden

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=9):
        result = pipe.denoise_step(InputBatch.make_batch(states))
        assert get_forward_context().denoise_step_idx == 9
    assert calls == ([(3, [1.0, 3.0]), (2, [2.0, 2.0])] if different_prefix_lengths else [(3, [1.0, 2.0, 2.0, 3.0])])
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0, 2.0, 3.0]))


@pytest.mark.parametrize("dtype", ["float", "mxfp8"])
def test_mixed_steps_raise_before_publication_or_forward(mocker, dtype):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype=dtype, diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)
    predict = mocker.patch.object(pipe, "predict_noise_maybe_with_cfg")
    runtime = mocker.Mock()
    batch = InputBatch.make_batch([_state("a", 3, 1.0), _state("b", 1, 2.0)])
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=9, paged_kv_runtime=runtime):
        with pytest.raises(ValueError, match=r"common denoise step per forward, got \[1, 3\]"):
            pipe.denoise_step(batch)
        assert get_forward_context().denoise_step_idx == 9
    predict.assert_not_called()
    runtime.ensure_active.assert_not_called()


@pytest.mark.parametrize("previous_step", [None, 0, 9])
def test_step_mode_restores_context_after_failure(monkeypatch, previous_step):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)

    def predict(*args, **kwargs):
        assert get_forward_context().denoise_step_idx == 1
        raise RuntimeError("denoise failed")

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config, denoise_step_idx=previous_step):
        with pytest.raises(RuntimeError, match="denoise failed"):
            pipe.denoise_step(InputBatch.make_batch([_state("a", 1, 1.0)]))
        assert get_forward_context().denoise_step_idx == previous_step


def _entry_point(monkeypatch, config, mode, predict):
    pipe = _Loop(_attention(config)) if mode == "request" else _step_pipeline(config)
    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    if mode == "request":
        return lambda: _diffuse(pipe)
    return lambda: pipe.denoise_step(InputBatch.make_batch([_state("inner", 1, 1.0)]))


@pytest.mark.parametrize("mode", ["request", "step"])
@pytest.mark.parametrize("failure", [None, "activation", "forward"])
@pytest.mark.parametrize("previous_step", [None, 0, 9])
def test_restoration_does_not_reactivate_runtime(monkeypatch, mocker, mode, failure, previous_step):
    config = OmniDiffusionConfig()
    runtime = mocker.Mock()
    if failure == "activation":
        runtime.ensure_active.side_effect = RuntimeError("activation failed")

    def predict(*args, **kwargs):
        if failure == "forward":
            raise RuntimeError("forward failed")
        return args[2]["hidden_states"]

    run = _entry_point(monkeypatch, config, mode, predict)
    with set_forward_context(denoise_step_idx=previous_step, paged_kv_runtime=runtime):
        if failure:
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                run()
        else:
            run()
        assert get_forward_context().denoise_step_idx == previous_step
    expected_steps = [0, 1, 2, 3] if mode == "request" else [1]
    if failure:
        expected_steps = expected_steps[:1]
    assert [call.args[0] for call in runtime.ensure_active.call_args_list] == expected_steps


@pytest.mark.parametrize("mode", ["request", "step"])
@pytest.mark.parametrize("inner_fails", [False, True])
def test_nested_calls_restore_outer_step(monkeypatch, mode, inner_fails):
    config = OmniDiffusionConfig()

    def inner_predict(*args, **kwargs):
        assert get_forward_context().denoise_step_idx in (0, 1, 2, 3)
        if inner_fails:
            raise RuntimeError("inner failed")
        return args[2]["hidden_states"]

    inner = _entry_point(monkeypatch, config, mode, inner_predict)

    def outer_predict(*args, **kwargs):
        outer_step = get_forward_context().denoise_step_idx
        if inner_fails:
            with pytest.raises(RuntimeError, match="inner failed"):
                inner()
        else:
            inner()
        assert get_forward_context().denoise_step_idx == outer_step
        return args[2]["hidden_states"]

    outer = _entry_point(monkeypatch, config, mode, outer_predict)
    with set_forward_context(denoise_step_idx=9):
        outer()
        assert get_forward_context().denoise_step_idx == 9


@pytest.mark.parametrize("mode", ["request", "step"])
def test_entry_without_forward_context(monkeypatch, mode):
    def predict(*args, **kwargs):
        return args[2]["hidden_states"]

    run = _entry_point(monkeypatch, OmniDiffusionConfig(), mode, predict)
    with override_forward_context(None):
        assert run().shape == (1, 2, 4)

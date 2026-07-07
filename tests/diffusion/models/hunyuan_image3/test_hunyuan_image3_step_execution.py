# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as hy3_module
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _STEP_AR_KV,
    _STEP_CFG_FACTOR,
    _STEP_GENERATOR,
    _STEP_GUIDANCE_SCALE,
    _STEP_INPUT_IDS,
    _STEP_MODEL_KWARGS,
    _STEP_PROMPT_KV,
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _single_rank_pp_group(monkeypatch):
    """Stub a single-rank pipeline-parallel group.

    The grouped denoise / scheduler paths call ``get_pp_group()`` (and
    ``get_pipeline_parallel_world_size()``) since the DiT pipeline-parallel
    wiring landed. These are CPU logic tests for grouping/merge/scheduler
    behavior at pipeline-parallel size 1, not distributed-comm tests, so stub a
    single-rank group (first == last rank) rather than standing up a real
    process group.
    """

    class _SingleRankPPGroup:
        is_first_rank = True
        is_last_rank = True
        world_size = 1

    monkeypatch.setattr(hy3_module, "get_pp_group", lambda: _SingleRankPPGroup())
    monkeypatch.setattr(hy3_module, "get_pipeline_parallel_world_size", lambda: 1)


def _pipeline():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline._tkwrapper = SimpleNamespace(pad_token_id=0)
    pipeline.od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        parallel_config=SimpleNamespace(sequence_parallel_size=1, cfg_parallel_size=1),
        cache_backend=None,
        diffusion_kv_cache_skip_step_indices=None,
    )
    pipeline._pipeline = SimpleNamespace()
    return pipeline


def _state(request_id: str, step_index: int) -> DiffusionRequestState:
    state = DiffusionRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(),
        prompt="prompt",
    )
    state.step_index = step_index
    state.timesteps = torch.tensor([1.0, 0.5, 0.25, 0.0])
    state.latents = torch.zeros(1, 4, 8, 8)
    state.extra = {
        _STEP_CFG_FACTOR: 1,
        _STEP_AR_KV: None,
        _STEP_INPUT_IDS: None,
        _STEP_GUIDANCE_SCALE: 1.0,
        _STEP_MODEL_KWARGS: {
            "num_image_tokens": 17,
            "ar_kv_reuse_len": 0,
        },
    }
    return state


def _sampling_params(**extra_args):
    return SimpleNamespace(
        timesteps=None,
        sigmas=None,
        num_outputs_per_prompt=None,
        extra_args=extra_args,
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=1.0,
        guidance_scale_provided=True,
        guidance_rescale=0.0,
        generator=None,
    )


def test_hunyuan_step_group_key_ignores_step_index_for_later_steps():
    pipeline = _pipeline()
    states = [_state("req-0", 1), _state("req-1", 3)]

    groups = pipeline._split_step_groups(states)

    assert len(groups) == 1
    assert [state.request_id for state in groups[0]] == ["req-0", "req-1"]


@pytest.mark.parametrize(
    ("sampling", "prompt_item", "expected_model_bot_task", "expected_system_bot_task"),
    [
        pytest.param(
            _sampling_params(bot_task="think_recaption", use_system_prompt="dynamic"),
            {"prompt": "prompt", "bot_task": "vanilla"},
            "think",
            "think",
            id="extra-args-precedence",
        ),
        pytest.param(
            _sampling_params(use_system_prompt="dynamic"),
            {"prompt": "prompt", "bot_task": "vanilla"},
            "image",
            "image",
            id="prompt-dict-fallback",
        ),
        pytest.param(
            _sampling_params(use_system_prompt="dynamic"),
            {"prompt": "prompt"},
            "auto",
            "image",
            id="default-auto-system-prompt",
        ),
    ],
)
def test_prepare_encode_preserves_normal_hunyuan_bot_task_semantics(
    monkeypatch,
    sampling,
    prompt_item,
    expected_model_bot_task,
    expected_system_bot_task,
):
    pipeline = _pipeline()
    captured: dict[str, object] = {}

    def fake_get_system_prompt(sys_type, bot_task, system_prompt=None):
        del sys_type, system_prompt
        captured["system_prompt_bot_task"] = bot_task
        return "system prompt"

    def fake_prepare_model_inputs(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after prepare_model_inputs")

    monkeypatch.setattr(hy3_module, "get_system_prompt", fake_get_system_prompt)
    pipeline.prepare_model_inputs = fake_prepare_model_inputs
    state = DiffusionRequestState(
        request_id="req-bot-task",
        sampling=sampling,
        prompt=prompt_item,
    )

    with pytest.raises(RuntimeError, match="stop after prepare_model_inputs"):
        pipeline.prepare_encode(state)

    assert captured["bot_task"] == expected_model_bot_task
    assert captured["system_prompt_bot_task"] == expected_system_bot_task


def test_forward_uses_same_hunyuan_bot_task_semantics(monkeypatch):
    pipeline = _pipeline()
    captured: dict[str, object] = {}

    def fake_get_system_prompt(sys_type, bot_task, system_prompt=None):
        del sys_type, system_prompt
        captured["system_prompt_bot_task"] = bot_task
        return "system prompt"

    def fake_prepare_model_inputs(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after prepare_model_inputs")

    monkeypatch.setattr(hy3_module, "get_system_prompt", fake_get_system_prompt)
    pipeline.prepare_model_inputs = fake_prepare_model_inputs
    req = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="req-forward-bot-task",
                sampling_params=_sampling_params(bot_task="think_recaption", use_system_prompt="dynamic"),
                prompt={"prompt": "prompt", "bot_task": "vanilla"},
            )
        ]
    )

    with pytest.raises(RuntimeError, match="stop after prepare_model_inputs"):
        pipeline.forward(req)

    assert captured["bot_task"] == "think"
    assert captured["system_prompt_bot_task"] == "think"


def test_grouped_denoise_rejects_non_sdpa_attention_backend():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_attention_config = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))

    with pytest.raises(ValueError, match="only supports TORCH_SDPA"):
        pipeline._ensure_grouped_attention_backend_supported(2)


def test_single_denoise_allows_non_sdpa_attention_backend():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_attention_config = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))

    pipeline._ensure_grouped_attention_backend_supported(1)


def test_grouped_denoise_allows_sdpa_attention_backend():
    pipeline = _pipeline()

    pipeline._ensure_grouped_attention_backend_supported(2)


def test_step_scheduler_preserves_latent_dtype_for_mixed_progress_batches():
    pipeline = _pipeline()
    pipeline._pipeline = SimpleNamespace(prepare_extra_func_kwargs=lambda step, kwargs: {})

    class FakeScheduler:
        def step(self, noise_pred, timestep, latents, **kwargs):
            del timestep, kwargs
            return (latents.float() + noise_pred.float(),)

    state = _state("req", 0)
    state.timesteps = torch.tensor([1.0])
    state.scheduler = FakeScheduler()
    state.latents = torch.zeros(1, 4, 8, 8, dtype=torch.bfloat16)
    state.extra[_STEP_GENERATOR] = None

    pipeline.step_scheduler(state, torch.ones_like(state.latents, dtype=torch.float32))

    assert state.latents.dtype == torch.bfloat16
    assert state.step_index == 1


def test_later_step_merge_shifts_spans_without_polluting_request_state():
    pipeline = _pipeline()
    states = [_state("short", 2), _state("long", 4)]
    states[0].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 5, dtype=torch.bool),
            "full_attn_spans": [[(2, 5)]],
        }
    )
    states[1].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 7, dtype=torch.bool),
            "full_attn_spans": [[(4, 7)]],
        }
    )
    states[0].extra[_STEP_PROMPT_KV] = {0: {"lens": torch.tensor([2])}}
    states[1].extra[_STEP_PROMPT_KV] = {0: {"lens": torch.tensor([4])}}

    row_state_indexes = [0, 1]
    row_branches = [0, 0]
    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes,
        row_branches,
        first_step=False,
    )

    assert merged["attention_mask"].shape == (2, 1, 3, 7)
    assert merged["full_attn_spans"] == [[(4, 7)], [(4, 7)]]

    pipeline._split_merged_kwargs_to_states(states, merged, row_state_indexes, row_branches)

    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (1, 1, 3, 5)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (1, 1, 3, 7)
    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 5)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 7)]]


def test_later_step_merge_allows_request_local_step_counts_and_guidance_values():
    pipeline = _pipeline()
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(1, 1, 2, 4, dtype=torch.bool),
                "full_attn_spans": [[(2, 4)]],
                "guidance_scale": 3.0 + idx,
                "num_inference_steps": 20 + idx,
            }
        )
        state.extra[_STEP_PROMPT_KV] = {0: {"lens": torch.tensor([2])}}

    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes=[0, 1],
        row_branches=[0, 0],
        first_step=False,
    )

    assert "guidance_scale" not in merged
    assert "num_inference_steps" not in merged


@pytest.mark.parametrize(
    ("request_id", "mutate_state", "error_match"),
    [
        pytest.param(
            "broken-req",
            lambda state: state.extra.pop(_STEP_MODEL_KWARGS),
            "broken-req",
            id="missing-model-kwargs",
        ),
        pytest.param(
            "bad-cfg",
            lambda state: state.extra.__setitem__(_STEP_CFG_FACTOR, 3),
            "bad-cfg",
            id="unsupported-cfg-factor",
        ),
    ],
)
def test_denoise_step_reports_invalid_group_state_with_request_id(request_id, mutate_state, error_match):
    pipeline = _pipeline()
    state = _state(request_id, 0)
    mutate_state(state)

    with pytest.raises(ValueError, match=error_match):
        pipeline.denoise_step(InputBatch.make_batch([state]))


def test_denoise_step_uses_input_batch_group_order_and_splits_back(monkeypatch):
    pipeline = _pipeline()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        prefix_len = 2 + idx * 2
        state.latents = torch.full((1, 1), float(idx))
        state.extra[_STEP_CFG_FACTOR] = 2
        state.extra[_STEP_GUIDANCE_SCALE] = 1.0
        state.extra[_STEP_INPUT_IDS] = None
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(2, 1, 2, prefix_len + 2, dtype=torch.bool),
                "full_attn_spans": [[(prefix_len, prefix_len + 2)], [(prefix_len, prefix_len + 2)]],
            }
        )
        state.extra[_STEP_PROMPT_KV] = {
            0: {
                "key": torch.zeros(2, prefix_len, 1, 1),
                "value": torch.zeros(2, prefix_len, 1, 1),
                "lens": torch.tensor([prefix_len, prefix_len]),
            }
        }

    captured = {}

    def fake_restore_prompt_kv_cache(states_arg, row_state_indexes, row_branches):
        del states_arg
        captured["row_state_indexes"] = list(row_state_indexes)
        captured["row_branches"] = list(row_branches)

    def fake_prepare_inputs_for_generation(input_ids, images, timestep, **model_kwargs):
        captured["input_ids"] = input_ids
        captured["images"] = images.clone()
        captured["timestep"] = timestep.clone()
        captured["merged_attention_mask_shape"] = tuple(model_kwargs["attention_mask"].shape)
        captured["merged_full_attn_spans"] = model_kwargs["full_attn_spans"]
        return {"model_kwargs": model_kwargs}

    pipeline._restore_prompt_kv_cache = fake_restore_prompt_kv_cache
    pipeline.prepare_inputs_for_generation = fake_prepare_inputs_for_generation
    pipeline.forward_call = lambda **kwargs: {"diffusion_prediction": torch.tensor([[10.0], [20.0], [1.0], [2.0]])}
    pipeline._update_model_kwargs_for_generation = lambda model_output, model_kwargs: model_kwargs
    pipeline._pipeline = SimpleNamespace(cfg_operator=lambda cond, uncond, scale, step: cond + uncond)

    batch = InputBatch.make_batch(states)
    out = pipeline.denoise_step(batch)

    assert captured["row_state_indexes"] == [0, 1, 0, 1]
    assert captured["row_branches"] == [0, 0, 1, 1]
    assert captured["input_ids"] is None
    assert tuple(captured["images"].shape) == (4, 1)
    assert captured["timestep"].tolist() == [0.5, 0.0, 0.5, 0.0]
    assert captured["merged_attention_mask_shape"] == (4, 1, 2, 6)
    assert captured["merged_full_attn_spans"] == [[(4, 6)], [(4, 6)], [(4, 6)], [(4, 6)]]
    torch.testing.assert_close(out, torch.tensor([[11.0], [22.0]]))
    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 4)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 6)
    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 4)], [(2, 4)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 6)], [(4, 6)]]


def test_step_microbatch_plan_cfg_policy_splits_branches():
    pipeline = _pipeline()
    states = [_state("a", 1), _state("b", 1), _state("c", 1)]

    plan = pipeline._step_microbatch_plan(states, cfg_factor=2, axes=frozenset({"cfg"}))
    assert len(plan) == 2
    cond, uncond = plan
    assert cond.row_state_indexes == [0, 1, 2]
    assert cond.row_branches == [0, 0, 0]
    assert cond.global_rows == [0, 1, 2]
    assert uncond.row_state_indexes == [0, 1, 2]
    assert uncond.row_branches == [1, 1, 1]
    assert uncond.global_rows == [3, 4, 5]

    # Without a second branch there is nothing to split.
    plan1 = pipeline._step_microbatch_plan(states, cfg_factor=1, axes=frozenset({"cfg"}))
    assert len(plan1) == 1
    assert plan1[0].row_branches == [0, 0, 0]
    assert plan1[0].global_rows == [0, 1, 2]


def test_denoise_step_cfg_policy_matches_cat_output(monkeypatch):
    # The cfg policy runs two single-branch microbatches; the combined output and
    # the next-step request state must match the single-forward cat path.
    pipeline = _pipeline()
    pipeline.od_config.parallel_config.microbatch_axes = ["cfg"]
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        prefix_len = 2 + idx * 2
        state.latents = torch.full((1, 1), float(idx))
        state.extra[_STEP_CFG_FACTOR] = 2
        state.extra[_STEP_GUIDANCE_SCALE] = 1.0
        state.extra[_STEP_INPUT_IDS] = None
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(2, 1, 2, prefix_len + 2, dtype=torch.bool),
                "full_attn_spans": [[(prefix_len, prefix_len + 2)], [(prefix_len, prefix_len + 2)]],
            }
        )
        state.extra[_STEP_PROMPT_KV] = {
            0: {
                "key": torch.zeros(2, prefix_len, 1, 1),
                "value": torch.zeros(2, prefix_len, 1, 1),
                "lens": torch.tensor([prefix_len, prefix_len]),
            }
        }

    restore_calls = []

    def fake_restore_prompt_kv_cache(states_arg, row_state_indexes, row_branches):
        del states_arg
        restore_calls.append((list(row_state_indexes), list(row_branches)))

    pipeline._restore_prompt_kv_cache = fake_restore_prompt_kv_cache
    pipeline.prepare_inputs_for_generation = lambda input_ids, images, timestep, **mk: {"model_kwargs": mk}
    # Per-microbatch forward: branch 0 (cond) -> rows [10, 20]; branch 1 (uncond) -> rows [1, 2].
    forward_returns = [torch.tensor([[10.0], [20.0]]), torch.tensor([[1.0], [2.0]])]
    forward_call_index = {"i": 0}

    def fake_forward(**kwargs):
        out = forward_returns[forward_call_index["i"]]
        forward_call_index["i"] += 1
        return {"diffusion_prediction": out}

    pipeline.forward_call = fake_forward
    pipeline._update_model_kwargs_for_generation = lambda model_output, model_kwargs: model_kwargs
    pipeline._pipeline = SimpleNamespace(cfg_operator=lambda cond, uncond, scale, step: cond + uncond)

    out = pipeline.denoise_step(InputBatch.make_batch(states))

    # Two single-branch microbatches, cond then uncond.
    assert restore_calls == [([0, 1], [0, 0]), ([0, 1], [1, 1])]
    assert forward_call_index["i"] == 2
    # Reassembled + combined output equals the cat path's [[11], [22]].
    torch.testing.assert_close(out, torch.tensor([[11.0], [22.0]]))
    # Next-step split runs over the full batch -> same per-state shapes as the cat path.
    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 4)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 6)


def test_prompt_kv_accumulation_matches_single_capture():
    # Capturing both branches in one microbatch must equal capturing each branch
    # in its own microbatch (the cfg policy), including joint re-padding when the
    # branches have different prompt lengths.
    pipeline = _pipeline()

    def _fake_layers(kv_map, lens):
        image_attn = SimpleNamespace(image_kv_cache_map=kv_map, image_kv_cache_lens=lens)
        return [SimpleNamespace(self_attn=SimpleNamespace(image_attn=image_attn))]

    key = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3, 1, 1)
    key[0, 2:] = 0.0  # cond prompt length 2 (model zero-pads beyond each row's len)
    value = key + 100.0
    lens = torch.tensor([2, 3])  # cond len 2, uncond len 3

    # Single full microbatch (both branches at once).
    state_full = _state("s", 1)
    pipeline.model = SimpleNamespace(layers=_fake_layers((key.clone(), value.clone()), lens.clone()))
    pipeline._capture_prompt_kv_cache([state_full], row_state_indexes=[0, 0], row_branches=[0, 1])
    full_cache = state_full.extra[_STEP_PROMPT_KV]

    # Two microbatches: cond, then uncond (each overwrites the per-layer singleton).
    state_cfg = _state("s", 1)
    accumulator: dict = {}
    pipeline.model = SimpleNamespace(layers=_fake_layers((key[0:1].clone(), value[0:1].clone()), lens[0:1].clone()))
    pipeline._accumulate_prompt_kv_cache([state_cfg], [0], [0], accumulator)
    pipeline.model = SimpleNamespace(layers=_fake_layers((key[1:2].clone(), value[1:2].clone()), lens[1:2].clone()))
    pipeline._accumulate_prompt_kv_cache([state_cfg], [0], [1], accumulator)
    pipeline._finalize_prompt_kv_cache([state_cfg], accumulator)
    cfg_cache = state_cfg.extra[_STEP_PROMPT_KV]

    assert full_cache.keys() == cfg_cache.keys()
    for layer_idx in full_cache:
        torch.testing.assert_close(full_cache[layer_idx]["key"], cfg_cache[layer_idx]["key"])
        torch.testing.assert_close(full_cache[layer_idx]["value"], cfg_cache[layer_idx]["value"])
        torch.testing.assert_close(full_cache[layer_idx]["lens"], cfg_cache[layer_idx]["lens"])


def test_step_microbatch_plan_requests_axis_chunks_states():
    pipeline = _pipeline()
    states = [_state(str(i), 1) for i in range(4)]

    # Default chunk size 1: one microbatch per request, carrying both branches.
    plan = pipeline._step_microbatch_plan(states, cfg_factor=2, axes=frozenset({"requests"}))
    assert len(plan) == 4
    assert plan[0].row_state_indexes == [0, 0]
    assert plan[0].row_branches == [0, 1]
    assert plan[0].global_rows == [0, 4]  # N=4 -> cond pos 0, uncond pos 4
    assert plan[1].global_rows == [1, 5]

    # Chunk size 2: two microbatches, each two requests x both branches.
    pipeline.od_config.parallel_config.microbatch_requests = 2
    plan2 = pipeline._step_microbatch_plan(states, cfg_factor=2, axes=frozenset({"requests"}))
    assert len(plan2) == 2
    assert plan2[0].row_state_indexes == [0, 1, 0, 1]
    assert plan2[0].row_branches == [0, 0, 1, 1]
    assert plan2[0].global_rows == [0, 1, 4, 5]
    assert plan2[1].row_state_indexes == [2, 3, 2, 3]
    assert plan2[1].global_rows == [2, 3, 6, 7]

    # No CFG: the requests split still chunks requests (single branch) -> fills the bubble.
    plan3 = pipeline._step_microbatch_plan(states, cfg_factor=1, axes=frozenset({"requests"}))
    assert len(plan3) == 2
    assert plan3[0].row_state_indexes == [0, 1]
    assert plan3[0].row_branches == [0, 0]
    assert plan3[0].global_rows == [0, 1]


def test_denoise_step_requests_axis_matches_cat_output(monkeypatch):
    # The requests axis runs one microbatch per request (both branches); the
    # combined output and next-step state must match the single-forward cat path.
    pipeline = _pipeline()
    pipeline.od_config.parallel_config.microbatch_axes = ["requests"]  # default chunk size 1
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        prefix_len = 2 + idx * 2
        state.latents = torch.full((1, 1), float(idx))
        state.extra[_STEP_CFG_FACTOR] = 2
        state.extra[_STEP_GUIDANCE_SCALE] = 1.0
        state.extra[_STEP_INPUT_IDS] = None
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(2, 1, 2, prefix_len + 2, dtype=torch.bool),
                "full_attn_spans": [[(prefix_len, prefix_len + 2)], [(prefix_len, prefix_len + 2)]],
            }
        )
        state.extra[_STEP_PROMPT_KV] = {
            0: {
                "key": torch.zeros(2, prefix_len, 1, 1),
                "value": torch.zeros(2, prefix_len, 1, 1),
                "lens": torch.tensor([prefix_len, prefix_len]),
            }
        }

    restore_calls = []

    def fake_restore_prompt_kv_cache(states_arg, row_state_indexes, row_branches):
        del states_arg
        restore_calls.append((list(row_state_indexes), list(row_branches)))

    pipeline._restore_prompt_kv_cache = fake_restore_prompt_kv_cache
    pipeline.prepare_inputs_for_generation = lambda input_ids, images, timestep, **mk: {"model_kwargs": mk}
    # Per-microbatch forward (one request, both branches): req-0 -> [cond 10, uncond 1];
    # req-1 -> [cond 20, uncond 2].
    forward_returns = [torch.tensor([[10.0], [1.0]]), torch.tensor([[20.0], [2.0]])]
    forward_call_index = {"i": 0}

    def fake_forward(**kwargs):
        out = forward_returns[forward_call_index["i"]]
        forward_call_index["i"] += 1
        return {"diffusion_prediction": out}

    pipeline.forward_call = fake_forward
    pipeline._update_model_kwargs_for_generation = lambda model_output, model_kwargs: model_kwargs
    pipeline._pipeline = SimpleNamespace(cfg_operator=lambda cond, uncond, scale, step: cond + uncond)

    out = pipeline.denoise_step(InputBatch.make_batch(states))

    # One microbatch per request, each carrying both branches.
    assert restore_calls == [([0, 0], [0, 1]), ([1, 1], [0, 1])]
    assert forward_call_index["i"] == 2
    # Reassembled + combined output equals the cat path's [[11], [22]].
    torch.testing.assert_close(out, torch.tensor([[11.0], [22.0]]))
    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 4)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 6)


def test_step_microbatch_plan_cfg_requests_2d_split():
    pipeline = _pipeline()
    states = [_state(str(i), 1) for i in range(3)]

    # c=1, cfg=2: one microbatch per (request, branch) -> 3*2 = 6, deepest split.
    plan = pipeline._step_microbatch_plan(states, cfg_factor=2, axes=frozenset({"cfg", "requests"}))
    assert len(plan) == 6
    # branch 0 chunks first, then branch 1 (N=3 -> uncond positions offset by 3)
    assert [mb.row_state_indexes for mb in plan] == [[0], [1], [2], [0], [1], [2]]
    assert [mb.row_branches for mb in plan] == [[0], [0], [0], [1], [1], [1]]
    assert [mb.global_rows for mb in plan] == [[0], [1], [2], [3], [4], [5]]

    # c=2, cfg=2: ceil(3/2)*2 = 4 microbatches.
    pipeline.od_config.parallel_config.microbatch_requests = 2
    plan2 = pipeline._step_microbatch_plan(states, cfg_factor=2, axes=frozenset({"cfg", "requests"}))
    assert len(plan2) == 4
    assert plan2[0].row_state_indexes == [0, 1]
    assert plan2[0].row_branches == [0, 0]
    assert plan2[0].global_rows == [0, 1]
    assert plan2[2].row_state_indexes == [0, 1]  # branch 1 chunk 0
    assert plan2[2].row_branches == [1, 1]
    assert plan2[2].global_rows == [3, 4]

    # CFG off -> reduces to the requests-only split.
    plan3 = pipeline._step_microbatch_plan(states, cfg_factor=1, axes=frozenset({"cfg", "requests"}))
    requests3 = pipeline._step_microbatch_plan(states, cfg_factor=1, axes=frozenset({"requests"}))
    assert [mb.row_state_indexes for mb in plan3] == [mb.row_state_indexes for mb in requests3]
    assert [mb.global_rows for mb in plan3] == [mb.global_rows for mb in requests3]


def test_denoise_step_cfg_requests_matches_cat_output(monkeypatch):
    # 2-D split (one microbatch per request per branch) must match the cat path.
    pipeline = _pipeline()
    pipeline.od_config.parallel_config.microbatch_axes = ["cfg", "requests"]  # default chunk size 1
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        prefix_len = 2 + idx * 2
        state.latents = torch.full((1, 1), float(idx))
        state.extra[_STEP_CFG_FACTOR] = 2
        state.extra[_STEP_GUIDANCE_SCALE] = 1.0
        state.extra[_STEP_INPUT_IDS] = None
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(2, 1, 2, prefix_len + 2, dtype=torch.bool),
                "full_attn_spans": [[(prefix_len, prefix_len + 2)], [(prefix_len, prefix_len + 2)]],
            }
        )
        state.extra[_STEP_PROMPT_KV] = {
            0: {
                "key": torch.zeros(2, prefix_len, 1, 1),
                "value": torch.zeros(2, prefix_len, 1, 1),
                "lens": torch.tensor([prefix_len, prefix_len]),
            }
        }

    restore_calls = []

    def fake_restore_prompt_kv_cache(states_arg, row_state_indexes, row_branches):
        del states_arg
        restore_calls.append((list(row_state_indexes), list(row_branches)))

    pipeline._restore_prompt_kv_cache = fake_restore_prompt_kv_cache
    pipeline.prepare_inputs_for_generation = lambda input_ids, images, timestep, **mk: {"model_kwargs": mk}
    # 4 microbatches: (req0,cond)=10, (req1,cond)=20, (req0,uncond)=1, (req1,uncond)=2.
    forward_returns = [torch.tensor([[10.0]]), torch.tensor([[20.0]]), torch.tensor([[1.0]]), torch.tensor([[2.0]])]
    forward_call_index = {"i": 0}

    def fake_forward(**kwargs):
        out = forward_returns[forward_call_index["i"]]
        forward_call_index["i"] += 1
        return {"diffusion_prediction": out}

    pipeline.forward_call = fake_forward
    pipeline._update_model_kwargs_for_generation = lambda model_output, model_kwargs: model_kwargs
    pipeline._pipeline = SimpleNamespace(cfg_operator=lambda cond, uncond, scale, step: cond + uncond)

    out = pipeline.denoise_step(InputBatch.make_batch(states))

    assert restore_calls == [([0], [0]), ([1], [0]), ([0], [1]), ([1], [1])]
    assert forward_call_index["i"] == 4
    torch.testing.assert_close(out, torch.tensor([[11.0], [22.0]]))
    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 4)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 6)


def test_microbatch_axes_config_driven():
    pipeline = _pipeline()
    pc = pipeline.od_config.parallel_config

    # No axes set -> empty split, default chunk size.
    assert pipeline._selected_microbatch_axes(2) == frozenset()
    assert pipeline._requests_microbatch_size() == 1

    # The config list drives the active axes (and the request chunk size).
    pc.microbatch_axes = ["requests"]
    pc.microbatch_requests = 4
    assert pipeline._selected_microbatch_axes(2) == frozenset({"requests"})
    assert pipeline._requests_microbatch_size() == 4

    # cfg drops out without a second branch.
    pc.microbatch_axes = ["cfg"]
    assert pipeline._selected_microbatch_axes(1) == frozenset()
    assert pipeline._selected_microbatch_axes(2) == frozenset({"cfg"})

    # Two-axis list, order-insensitive.
    pc.microbatch_axes = ["cfg", "requests"]
    assert pipeline._selected_microbatch_axes(2) == frozenset({"cfg", "requests"})
    pc.microbatch_axes = ["requests", "cfg"]
    assert pipeline._selected_microbatch_axes(2) == frozenset({"cfg", "requests"})
    # With no second branch it collapses to the requests axis only.
    assert pipeline._selected_microbatch_axes(1) == frozenset({"requests"})

    # Empty list is the no-op.
    pc.microbatch_axes = []
    assert pipeline._selected_microbatch_axes(2) == frozenset()


def test_microbatch_axes_config_validation():
    """DiffusionParallelConfig accepts valid axis lists and rejects malformed ones."""
    from pydantic import ValidationError

    from vllm_omni.diffusion.data import DiffusionParallelConfig

    # Valid: empty, single, and 2-axis (order-insensitive).
    for policies in ([], ["cfg"], ["requests"], ["cfg", "requests"], ["requests", "cfg"]):
        DiffusionParallelConfig(microbatch_axes=policies)

    # Unknown axis rejected (only "cfg" / "requests" are valid).
    for bad in (["foo"], ["bar"], ["cfg", "baz"]):
        with pytest.raises(ValidationError):
            DiffusionParallelConfig(microbatch_axes=bad)

    # Duplicate axis rejected.
    with pytest.raises(ValidationError):
        DiffusionParallelConfig(microbatch_axes=["cfg", "cfg"])

    # Bare scalar (not a list) rejected.
    with pytest.raises(ValidationError):
        DiffusionParallelConfig(microbatch_axes="cfg")

    # microbatch_requests < 1 rejected.
    with pytest.raises(ValidationError):
        DiffusionParallelConfig(microbatch_requests=0)


def test_validate_microbatch_axes_supported_axes():
    """The runner load-time check rejects axes the pipeline does not implement."""
    from vllm_omni.diffusion.worker.diffusion_model_runner import validate_microbatch_axes

    def _cfg(policies, supported=frozenset({"cfg", "requests"})):
        pipe = SimpleNamespace(supported_microbatch_axes=supported)
        od = SimpleNamespace(
            model_class_name="TestPipeline",
            parallel_config=SimpleNamespace(microbatch_axes=list(policies)),
        )
        return pipe, od

    # Supported subsets pass.
    for policies in ([], ["cfg"], ["requests"], ["cfg", "requests"]):
        validate_microbatch_axes(*_cfg(policies))

    # A pipeline supporting only "cfg" rejects "requests".
    with pytest.raises(ValueError):
        validate_microbatch_axes(*_cfg(["requests"], supported=frozenset({"cfg"})))

    # A pipeline declaring no axes rejects any non-empty policy.
    pipe = SimpleNamespace()  # no supported_microbatch_axes attr
    od = SimpleNamespace(model_class_name="NoAxes", parallel_config=SimpleNamespace(microbatch_axes=["cfg"]))
    with pytest.raises(ValueError):
        validate_microbatch_axes(pipe, od)

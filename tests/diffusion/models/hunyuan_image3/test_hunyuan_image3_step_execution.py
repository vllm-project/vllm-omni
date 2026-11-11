# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

import vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer as hy3_transformer_module
import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as hy3_module
import vllm_omni.diffusion.models.hunyuan_image3.request_layout as hy3_layout_module
from tests.diffusion.attention.test_attention_schedule_candidates import _make_metadata_candidate
from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
    InvalidAttentionScheduleError,
)
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.diffusion.forward_context import (
    bind_attention_schedule,
    get_attention_schedule_profile,
    get_forward_context,
    set_forward_context,
)
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerEncodeOutput
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Text2ImagePipeline,
    ImageInfo,
)
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
from vllm_omni.diffusion.models.hunyuan_image3.request_layout import HunyuanPreparedLayout
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pipeline():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline._tkwrapper = SimpleNamespace(pad_token_id=0)
    pipeline.od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        parallel_config=SimpleNamespace(sequence_parallel_size=1, cfg_parallel_size=1),
        cache_backend=None,
        diffusion_kv_cache_skip_step_indices=None,
        diffusion_attention_schedule=None,
    )
    pipeline.hf_config = SimpleNamespace(cfg_distilled=False, use_meanflow=False)
    pipeline._pipeline = SimpleNamespace()
    return pipeline


def _state(request_id: str, step_index: int) -> StepRequestState:
    state = StepRequestState(
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


def _prepared_layout() -> HunyuanPreparedLayout:
    return HunyuanPreparedLayout(
        tokenizer_output=TokenizerEncodeOutput(
            tokens=torch.arange(21).reshape(1, 21),
            gen_image_mask=torch.ones(1, 21, dtype=torch.bool),
            gen_timestep_scatter_index=torch.tensor([[4]]),
            all_image_slices=[[slice(5, 21)]],
            joint_image_slices=[[]],
            gen_image_slices=[[slice(5, 21)]],
            real_pos=torch.tensor([[21]]),
        ),
        rope_image_info=[[(slice(5, 21), (4, 4))]],
        generated_image_info=ImageInfo(
            image_type="gen_image",
            image_width=512,
            image_height=512,
            token_width=4,
            token_height=4,
            image_token_length=16,
        ),
    )


def test_prepare_model_inputs_reuses_prepared_layout(monkeypatch):
    pipeline = _pipeline()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    rope_kwargs = {}

    def fake_build_batch_2d_rope(**kwargs):
        rope_kwargs.update(kwargs)
        return torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(hy3_module, "build_batch_2d_rope", fake_build_batch_2d_rope)

    def fail_apply_chat_template(**_kwargs):
        pytest.fail("prepared Hunyuan execution must not apply the chat template again")

    pipeline._tkwrapper = SimpleNamespace(
        apply_chat_template=fail_apply_chat_template,
        eos_token_id=2,
        boi_token_id=3,
        end_recaption_token_id=4,
        end_answer_token_id=5,
    )
    pipeline.config = SimpleNamespace(
        attention_head_dim=2,
        rope_theta=10000.0,
    )
    prepared = _prepared_layout()

    model_inputs = pipeline.prepare_model_inputs(
        prompt="prompt",
        mode="gen_image",
        guidance_scale=1.0,
        image_size=(512, 512),
        device=torch.device("cpu"),
        generator=[torch.Generator().manual_seed(0)],
        prepared_layout=prepared,
    )

    assert model_inputs["tokenizer_output"] is prepared.tokenizer_output
    assert model_inputs["batch_gen_image_info"] == [prepared.generated_image_info]
    assert rope_kwargs["image_infos"] is prepared.rope_image_info
    torch.testing.assert_close(model_inputs["input_ids"], prepared.tokenizer_output.tokens)


def test_slice_cached_prefix_inputs_keeps_full_kv_axis() -> None:
    inputs_embeds = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    attention_mask = torch.arange(36).reshape(1, 1, 6, 6)
    position_ids = torch.arange(6).reshape(1, 6)
    custom_pos_emb = (torch.arange(6).reshape(1, 6), torch.arange(10, 16).reshape(1, 6))
    image_mask = torch.tensor([[False, False, True, True, True, True]])
    gen_timestep_scatter_index = torch.tensor([[2]])

    sliced = HunyuanImage3Pipeline._slice_cached_prefix_inputs(
        inputs_embeds,
        attention_mask,
        position_ids,
        custom_pos_emb,
        image_mask,
        gen_timestep_scatter_index,
        [6],
        [6],
        2,
    )

    (
        sliced_embeds,
        sliced_mask,
        sliced_positions,
        sliced_rope,
        sliced_image_mask,
        sliced_scatter,
        sliced_query_lens,
        sliced_seq_len,
    ) = sliced
    torch.testing.assert_close(sliced_embeds, inputs_embeds[:, 2:])
    torch.testing.assert_close(sliced_mask, attention_mask[:, :, 2:, :])
    torch.testing.assert_close(sliced_positions, position_ids[:, 2:])
    torch.testing.assert_close(sliced_rope[0], custom_pos_emb[0][:, 2:])
    torch.testing.assert_close(sliced_rope[1], custom_pos_emb[1][:, 2:])
    torch.testing.assert_close(sliced_image_mask, image_mask[:, 2:])
    torch.testing.assert_close(sliced_scatter, torch.tensor([[0]]))
    assert sliced_query_lens == [4]
    assert sliced_seq_len == 4


def test_slice_cached_prefix_inputs_allows_right_padded_rows() -> None:
    inputs_embeds = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    attention_mask = torch.arange(64).reshape(1, 1, 8, 8)
    position_ids = torch.arange(8).reshape(1, 8)
    custom_pos_emb = (torch.arange(8).reshape(1, 8), torch.arange(10, 18).reshape(1, 8))
    image_mask = torch.tensor([[False, False, True, True, True, True, False, False]])
    gen_timestep_scatter_index = torch.tensor([[2]])

    sliced = HunyuanImage3Pipeline._slice_cached_prefix_inputs(
        inputs_embeds,
        attention_mask,
        position_ids,
        custom_pos_emb,
        image_mask,
        gen_timestep_scatter_index,
        [6],
        [6],
        2,
    )

    (
        sliced_embeds,
        sliced_mask,
        sliced_positions,
        sliced_rope,
        sliced_image_mask,
        sliced_scatter,
        sliced_query_lens,
        sliced_seq_len,
    ) = sliced
    torch.testing.assert_close(sliced_embeds, inputs_embeds[:, 2:])
    torch.testing.assert_close(sliced_mask, attention_mask[:, :, 2:, :])
    torch.testing.assert_close(sliced_positions, position_ids[:, 2:])
    torch.testing.assert_close(sliced_rope[0], custom_pos_emb[0][:, 2:])
    torch.testing.assert_close(sliced_image_mask, image_mask[:, 2:])
    torch.testing.assert_close(sliced_scatter, torch.tensor([[0]]))
    assert sliced_query_lens == [4]
    assert sliced_seq_len == 6


def test_slice_cached_prefix_inputs_cuts_mixed_prefix_rows() -> None:
    inputs_embeds = torch.arange(32, dtype=torch.float32).reshape(2, 8, 2)
    attention_mask = torch.arange(128).reshape(2, 1, 8, 8)
    position_ids = torch.arange(16).reshape(2, 8)
    custom_pos_emb = (position_ids.clone(), position_ids.clone() + 10)
    image_mask = torch.zeros(2, 8, dtype=torch.bool)
    gen_timestep_scatter_index = torch.tensor([[4], [2]])

    sliced = HunyuanImage3Pipeline._slice_cached_prefix_inputs(
        inputs_embeds,
        attention_mask,
        position_ids,
        custom_pos_emb,
        image_mask,
        gen_timestep_scatter_index,
        [6, 8],
        [6, 8],
        [2, 0],
    )

    sliced_embeds, sliced_mask, sliced_positions, _, _, sliced_scatter, sliced_query_lens, sliced_seq_len = sliced
    assert sliced_query_lens == [4, 8]
    assert sliced_seq_len == 8
    torch.testing.assert_close(sliced_embeds[0, :4], inputs_embeds[0, 2:6])
    torch.testing.assert_close(sliced_embeds[1], inputs_embeds[1])
    torch.testing.assert_close(sliced_positions[0, :4], position_ids[0, 2:6])
    torch.testing.assert_close(sliced_mask[0, :, :4, :], attention_mask[0, :, 2:6, :])
    torch.testing.assert_close(sliced_scatter, torch.tensor([[2], [2]]))
    # Mixed-prefix pack leaves heterogeneous logical lengths inside padded T;
    # RoPE / _forward_paged must reshape on T, not query_lens[0].
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        _reshape_width_for_query_batch,
    )

    batch, pad_width = _reshape_width_for_query_batch(
        sliced_embeds.shape[0] * sliced_embeds.shape[1],
        sliced_query_lens,
    )
    assert (batch, pad_width) == (2, 8)


@pytest.mark.parametrize("local_prefix_hit", [True, False])
def test_ar_reuse_does_not_consume_local_prefix_hits(local_prefix_hit):
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        HunyuanImage3Text2ImagePipeline,
    )

    pipe = object.__new__(HunyuanImage3Text2ImagePipeline)
    input_ids = torch.arange(12).reshape(1, 12)
    cond_image = torch.ones(1, 1)
    kwargs = dict(
        query_lens=[12],
        attention_mask=torch.ones(1, 1, 12, 12),
        position_ids=input_ids.clone(),
        image_mask=torch.zeros_like(input_ids, dtype=torch.bool),
        gen_timestep_scatter_index=torch.tensor([[8]]),
        cond_vae_images=cond_image,
    )
    runtime = SimpleNamespace(metadata=SimpleNamespace(prefill_rows=[SimpleNamespace(kv_start_pos=4)]))
    with set_forward_context(paged_kv_runtime=runtime, paged_kv_cached_prefix_len=4 if local_prefix_hit else 0):
        output, ar_reuse_len = pipe._maybe_handle_ar_kv_reuse(input_ids, kwargs, 1, False, None, torch.device("cpu"))
    if local_prefix_hit:
        assert ar_reuse_len == 0
        assert output is input_ids
        assert kwargs["query_lens"] == [12]
        assert kwargs["cond_vae_images"] is cond_image
    else:
        assert ar_reuse_len == 4
        torch.testing.assert_close(output, input_ids[:, 4:])
        assert kwargs["query_lens"] == [8]
        assert "cond_vae_images" not in kwargs


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

    monkeypatch.setattr(hy3_layout_module, "get_system_prompt", fake_get_system_prompt)
    pipeline.prepare_model_inputs = fake_prepare_model_inputs
    state = StepRequestState(
        request_id="req-bot-task",
        sampling=sampling,
        prompt=prompt_item,
        prepared_layout=_prepared_layout(),
    )

    with pytest.raises(RuntimeError, match="stop after prepare_model_inputs"):
        pipeline.prepare_encode(state)

    assert captured["bot_task"] == expected_model_bot_task
    assert captured["system_prompt_bot_task"] == expected_system_bot_task
    assert captured["prepared_layout"] is state.prepared_layout


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

    monkeypatch.setattr(hy3_layout_module, "get_system_prompt", fake_get_system_prompt)
    pipeline.prepare_model_inputs = fake_prepare_model_inputs
    request = SimpleNamespace(
        request_id="req-forward-bot-task",
        sampling_params=_sampling_params(bot_task="think_recaption", use_system_prompt="dynamic"),
        prompt={"prompt": "prompt", "bot_task": "vanilla"},
        prepared_layout=_prepared_layout(),
    )
    req = DiffusionRequestBatch(requests=[request])

    with pytest.raises(RuntimeError, match="stop after prepare_model_inputs"):
        pipeline.forward(req)

    assert captured["bot_task"] == "think"
    assert captured["system_prompt_bot_task"] == "think"
    assert captured["prepared_layout"] is request.prepared_layout


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


def test_grouped_denoise_allows_flash_attn_when_paged():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_kv_mode = hy3_module.DiffusionKVCacheMode.PAGED_SCHEDULER
    pipeline.od_config.diffusion_attention_config = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))

    pipeline._ensure_grouped_attention_backend_supported(2)


def test_scheduler_paged_allows_step_request_validation():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_kv_mode = hy3_module.DiffusionKVCacheMode.PAGED_SCHEDULER
    state = _state("paged-step", 0)
    state.sampling = _sampling_params()

    pipeline._validate_step_request(state)


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
    states[0].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([2])}]
    states[1].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([4])}]

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


def test_later_step_paged_merge_keeps_per_row_spans_and_seq_lens():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_kv_mode = hy3_module.DiffusionKVCacheMode.PAGED_SCHEDULER
    states = [_state("short", 2), _state("long", 4)]
    states[0].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 5, dtype=torch.bool),
            "full_attn_spans": [[(2, 5)]],
            "num_image_tokens": 3,
        }
    )
    states[1].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 7, dtype=torch.bool),
            "full_attn_spans": [[(4, 7)]],
            "num_image_tokens": 3,
        }
    )
    states[0].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([2])}]
    states[1].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([4])}]

    _, merged = pipeline._merge_step_model_inputs(
        states,
        [0, 1],
        [0, 0],
        first_step=False,
    )

    assert merged["attention_mask"] is None
    assert merged["full_attn_spans"] == [[(2, 5)], [(4, 7)]]
    assert merged["query_lens"] == [3, 3]
    assert merged["seq_lens"] == [5, 7]

    pipeline._split_merged_kwargs_to_states(states, merged, [0, 1], [0, 0])

    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 5)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 7)]]
    assert states[0].extra[_STEP_MODEL_KWARGS]["query_lens"] == [3]
    assert states[1].extra[_STEP_MODEL_KWARGS]["query_lens"] == [3]
    assert states[0].extra[_STEP_MODEL_KWARGS]["seq_lens"] == [5]
    assert states[1].extra[_STEP_MODEL_KWARGS]["seq_lens"] == [7]


def test_later_step_paged_cfg_merge_keeps_cfg_major_per_row_spans():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_kv_mode = hy3_module.DiffusionKVCacheMode.PAGED_SCHEDULER
    states = [_state("short", 2), _state("long", 4)]
    for state, prefix, seq_len, span in (
        (states[0], 2, 5, (2, 5)),
        (states[1], 4, 7, (4, 7)),
    ):
        state.extra[_STEP_CFG_FACTOR] = 2
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(2, 1, 3, seq_len, dtype=torch.bool),
                "full_attn_spans": [[span], [span]],
                "num_image_tokens": 3,
            }
        )
        state.extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([prefix, prefix])}]

    row_state_indexes, row_branches = HunyuanImage3Pipeline._step_row_order(states, cfg_factor=2)
    assert row_state_indexes == [0, 1, 0, 1]
    assert row_branches == [0, 0, 1, 1]

    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes,
        row_branches,
        first_step=False,
    )

    assert merged["full_attn_spans"] == [[(2, 5)], [(4, 7)], [(2, 5)], [(4, 7)]]
    assert merged["query_lens"] == [3, 3, 3, 3]
    assert merged["seq_lens"] == [5, 7, 5, 7]


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
        state.extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([2])}]

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
        state.extra[_STEP_PROMPT_KV] = [
            {
                "key": torch.zeros(2, prefix_len, 1, 1),
                "value": torch.zeros(2, prefix_len, 1, 1),
                "lens": torch.tensor([prefix_len, prefix_len]),
            }
        ]

    captured: dict[str, object] = {}

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
    assert isinstance(captured["images"], torch.Tensor)
    assert tuple(captured["images"].shape) == (4, 1)
    assert isinstance(captured["timestep"], torch.Tensor)
    assert captured["timestep"].tolist() == [0.5, 0.0, 0.5, 0.0]
    assert captured["merged_attention_mask_shape"] == (4, 1, 2, 6)
    assert captured["merged_full_attn_spans"] == [[(4, 6)], [(4, 6)], [(4, 6)], [(4, 6)]]
    torch.testing.assert_close(out, torch.tensor([[11.0], [22.0]]))
    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 4)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 6)
    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 4)], [(2, 4)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 6)], [(4, 6)]]


def test_distilled_step_supplies_guidance_and_meanflow_timestep(monkeypatch):
    pipeline = _pipeline()
    pipeline.hf_config = SimpleNamespace(cfg_distilled=True, use_meanflow=True)
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    state = _state("distilled", 1)
    state.latents = torch.zeros(1, 1)
    state.extra[_STEP_GUIDANCE_SCALE] = 2.5
    state.extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 2, 4, dtype=torch.bool),
            "full_attn_spans": [[(2, 4)]],
        }
    )
    state.extra[_STEP_PROMPT_KV] = [
        {
            "key": torch.zeros(1, 2, 1, 1),
            "value": torch.zeros(1, 2, 1, 1),
            "lens": torch.tensor([2]),
        }
    ]
    state.scheduler = SimpleNamespace(get_timestep_r=lambda _timestep: torch.tensor(0.25))
    captured = {}

    pipeline._restore_prompt_kv_cache = lambda *_args: None

    def fake_prepare_inputs(input_ids, images, timestep, **model_kwargs):
        del input_ids, images, timestep
        captured.update(model_kwargs)
        return {"model_kwargs": model_kwargs}

    pipeline.prepare_inputs_for_generation = fake_prepare_inputs
    pipeline.forward_call = lambda **_kwargs: {"diffusion_prediction": torch.tensor([[1.0]])}
    pipeline._update_model_kwargs_for_generation = lambda _output, model_kwargs: model_kwargs

    output = pipeline.denoise_step(InputBatch.make_batch([state]))

    torch.testing.assert_close(output, torch.tensor([[1.0]]))
    torch.testing.assert_close(captured["guidance"], torch.tensor([2500.0], dtype=torch.bfloat16))
    torch.testing.assert_close(captured["timesteps_r"], torch.tensor([0.25]))


# Attention schedule integration. Step-mode tests evaluate A at step 2 of 8 and B at step 5 of 10.
_STEP_SCHEDULE = (AttentionScheduleRange(start=3, end=6, profile="sparse"),)
_REQUEST_SCHEDULE = (AttentionScheduleRange(start=1, end=3, profile="sparse"),)


def _progress(ctx) -> tuple:
    return ctx.denoise_step_idx, ctx.total_denoise_steps, ctx.denoise_timestep, get_attention_schedule_profile()


@contextmanager
def _denoise_context(schedule, *, stale_timestep=None, stale_total=None, from_request=False):
    with set_forward_context(), bind_attention_schedule(schedule, from_request=from_request):
        ctx = get_forward_context()
        ctx.denoise_timestep = stale_timestep
        ctx.total_denoise_steps = stale_total
        yield ctx


def _scheduled_state(request_id: str, step_index: int, total_steps: int, latent: float) -> StepRequestState:
    state = _state(request_id, step_index)
    # Step k has timestep (total_steps - k) * 100, normalized by num_train_timesteps=1000.
    state.timesteps = torch.arange(total_steps, 0, -1, dtype=torch.float32) * 100.0
    state.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    state.latents = torch.full((1, 1), latent)
    state.extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 2, 4, dtype=torch.bool),
            "full_attn_spans": [[(2, 4)]],
        }
    )
    state.extra[_STEP_PROMPT_KV] = [
        {
            "key": torch.zeros(1, 2, 1, 1),
            "value": torch.zeros(1, 2, 1, 1),
            "lens": torch.tensor([2]),
        }
    ]
    return state


def _recording_step_pipeline(monkeypatch, records: list, *, fail_on_call: int | None = None):
    pipeline = _pipeline()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))

    def fake_forward_call(images, first_step):
        del first_step
        records.append((*_progress(get_forward_context()), images.flatten().tolist()))
        if len(records) == fail_on_call:
            raise RuntimeError("forward failed")
        return {"diffusion_prediction": images * 10.0}

    pipeline._restore_prompt_kv_cache = lambda *_args: None
    pipeline.prepare_inputs_for_generation = lambda input_ids, images, timestep, **kwargs: {"images": images}
    pipeline.forward_call = fake_forward_call
    pipeline._update_model_kwargs_for_generation = lambda _output, model_kwargs: model_kwargs
    return pipeline


def test_scheduled_step_requests_run_separately_with_their_own_progress(monkeypatch):
    records: list = []
    pipeline = _recording_step_pipeline(monkeypatch, records)
    states = [_scheduled_state("req-a", 2, 8, 1.0), _scheduled_state("req-b", 5, 10, 2.0)]

    with _denoise_context(_STEP_SCHEDULE, stale_timestep=0.9, stale_total=99) as ctx:
        out = pipeline.denoise_step(InputBatch.make_batch(states))
        after = _progress(ctx)

    assert records == [
        (2, 8, 0.6, None, [1.0]),
        (5, 10, 0.5, "sparse", [2.0]),
    ]
    torch.testing.assert_close(out, torch.tensor([[10.0], [20.0]]))
    assert after == (None, 99, 0.9, None)


def test_scheduled_step_rejects_a_scheduler_it_cannot_normalize(monkeypatch):
    pipeline = _recording_step_pipeline(monkeypatch, [])
    state = _scheduled_state("req-a", 2, 8, 1.0)
    state.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    with _denoise_context(_STEP_SCHEDULE):
        with pytest.raises(RuntimeError, match="needs a current timestep and a flow-matching scheduler"):
            pipeline.denoise_step(InputBatch.make_batch([state]))


def test_scheduled_step_restores_context_when_second_request_fails(monkeypatch):
    records: list = []
    pipeline = _recording_step_pipeline(monkeypatch, records, fail_on_call=2)
    states = [_scheduled_state("req-a", 2, 8, 1.0), _scheduled_state("req-b", 5, 10, 2.0)]

    with _denoise_context(_STEP_SCHEDULE, stale_timestep=0.9, stale_total=99) as ctx:
        with pytest.raises(RuntimeError, match="forward failed"):
            pipeline.denoise_step(InputBatch.make_batch(states))
        after = _progress(ctx)

    assert [record[:4] for record in records] == [(2, 8, 0.6, None), (5, 10, 0.5, "sparse")]
    assert after == (None, 99, 0.9, None)


class _FakeFlowScheduler:
    order = 1
    config = SimpleNamespace(num_train_timesteps=1000)

    def step(self, model_output, timestep, sample, return_dict=False):
        del model_output, timestep, return_dict
        return (sample,)


def _request_mode_pipeline(monkeypatch, *, num_steps: int, events: list, fail_at_step: int | None = None):
    def fake_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas):
        del scheduler, num_inference_steps, device, timesteps, sigmas
        return torch.arange(num_steps, 0, -1, dtype=torch.float32) * 100.0, num_steps

    def fake_forward_call(images, first_step):
        ctx = get_forward_context()
        events.append(("forward", *_progress(ctx), first_step))
        if ctx.denoise_step_idx == fail_at_step:
            raise RuntimeError("forward failed")
        return {"diffusion_prediction": torch.zeros_like(images)}

    def fake_prepare_latents(**_kwargs):
        events.append("prepare_latents")
        return torch.zeros(1, 1)

    def fake_ar_kv_reuse(input_ids, model_kwargs, batch_size, cfg_parallel_ready, cfg_rank, device):
        del model_kwargs, batch_size, cfg_parallel_ready, cfg_rank, device
        events.append("ar_kv_reuse")
        return input_ids, 0

    cpu = property(lambda self: torch.device("cpu"))
    monkeypatch.setattr(hy3_transformer_module, "retrieve_timesteps", fake_retrieve_timesteps)
    monkeypatch.setattr(HunyuanImage3Text2ImagePipeline, "_execution_device", cpu)
    monkeypatch.setattr(HunyuanImage3Text2ImagePipeline, "device", cpu)

    # The request-mode pipeline publishes through its model, which is the HunyuanImage3Pipeline itself.
    model = _pipeline()
    model.config = hy3_transformer_module.HunyuanImage3Config(
        cfg_distilled=False, use_meanflow=False, vae={"latent_channels": 1}
    )
    model.generation_config = None
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    model._prepare_attention_mask_for_generation = lambda input_ids, generation_config, model_kwargs: mask
    model.prepare_inputs_for_generation = lambda input_ids, images, timestep, **kwargs: {"images": images}
    model.forward_call = fake_forward_call
    model._update_model_kwargs_for_generation = lambda _output, model_kwargs: model_kwargs

    pipe = object.__new__(HunyuanImage3Text2ImagePipeline)
    pipe.model = model
    pipe.scheduler = _FakeFlowScheduler()
    pipe.vae = SimpleNamespace(config=SimpleNamespace(), decode=lambda latents, return_dict, generator: (latents,))
    pipe.progress_bar = lambda total: nullcontext(SimpleNamespace(update=lambda: None))
    pipe.prepare_latents = fake_prepare_latents
    pipe._maybe_handle_ar_kv_reuse = fake_ar_kv_reuse
    return pipe


def _run_request_loop(pipe, *, num_inference_steps: int = 4):
    return pipe(
        batch_size=1,
        image_size=[16, 16],
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
        return_dict=False,
        model_kwargs={"input_ids": None},
    )


def test_request_loop_publishes_actual_total_and_timestep_when_scheduled(monkeypatch):
    events: list = []
    pipe = _request_mode_pipeline(monkeypatch, num_steps=4, events=events)

    with _denoise_context(_REQUEST_SCHEDULE) as ctx:
        _run_request_loop(pipe)
        after = _progress(ctx)

    assert events == [
        "prepare_latents",
        "ar_kv_reuse",
        ("forward", 0, 4, 0.4, None, True),
        ("forward", 1, 4, 0.3, "sparse", False),
        ("forward", 2, 4, 0.2, "sparse", False),
        ("forward", 3, 4, 0.1, None, False),
    ]
    assert after == (None, None, None, None)


def test_request_loop_rejects_schedule_past_actual_steps_before_any_forward(monkeypatch):
    events: list = []
    pipe = _request_mode_pipeline(monkeypatch, num_steps=4, events=events)

    # 8 steps are requested but the sequence built has 4, so [3, 6) does not fit.
    with (
        _denoise_context(_STEP_SCHEDULE, from_request=True),
        pytest.raises(InvalidAttentionScheduleError, match="exceeds total_steps=4"),
    ):
        _run_request_loop(pipe, num_inference_steps=8)

    assert events == []


class _SpecDependentMaskBackend:
    supports_piecewise_spans = False

    @classmethod
    def supports_attention_mask(cls, attention_spec=None) -> bool:
        return attention_spec.backend == "TORCH_SDPA"

    @classmethod
    def supports_dense_4d_mask(cls, attention_spec=None) -> bool:
        return attention_spec.backend == "TORCH_SDPA"


def test_image_attention_mask_check_rejects_candidates_without_mask_support():
    check = hy3_transformer_module._require_attention_mask_support

    masked = _make_metadata_candidate(_SpecDependentMaskBackend, AttentionSpec(backend="TORCH_SDPA"))
    plain = _make_metadata_candidate(_SpecDependentMaskBackend, AttentionSpec(backend="TRTLLM_ATTN"))
    assert check(masked) is None
    reason = check(plain)
    assert reason is not None and "4D attention mask" in reason


def _capture_hy3_log(caplog, monkeypatch):
    # Whether vllm loggers reach the root logger that caplog listens on depends on the vllm
    # version, so attach caplog's handler directly and stop propagation: each record is seen once.
    monkeypatch.setattr(hy3_module.logger, "handlers", [*hy3_module.logger.handlers, caplog.handler])
    monkeypatch.setattr(hy3_module.logger, "propagate", False)


@pytest.mark.parametrize(
    ("default", "logged"),
    [
        ((AttentionScheduleRange(start=0, end=2, profile="approx"),), True),
        ((AttentionScheduleRange(start=2, end=None, profile="approx"),), False),
        ((), False),
    ],
    ids=["covers-step-0", "starts-later", "empty"],
)
def test_startup_notes_a_default_schedule_that_covers_step_0(default, logged, caplog, monkeypatch):
    _capture_hy3_log(caplog, monkeypatch)
    schedule = SimpleNamespace(default=default)
    with caplog.at_level("INFO", logger=hy3_module.logger.name):
        hy3_module._log_step0_profile_reuse(SimpleNamespace(diffusion_attention_schedule=schedule))
        hy3_module._log_step0_profile_reuse(SimpleNamespace(diffusion_attention_schedule=None))
    messages = [record.getMessage() for record in caplog.records if "step-0 prompt K/V" in record.getMessage()]
    assert len(messages) == (1 if logged else 0)
    assert all("profile 'approx'" in message for message in messages)


def test_startup_step_0_note_logs_only_on_rank_0(caplog, monkeypatch):
    _capture_hy3_log(caplog, monkeypatch)
    monkeypatch.setattr(hy3_module, "_is_rank_zero", lambda: False)
    schedule = SimpleNamespace(default=(AttentionScheduleRange(start=0, end=2, profile="approx"),))
    with caplog.at_level("INFO", logger=hy3_module.logger.name):
        hy3_module._log_step0_profile_reuse(SimpleNamespace(diffusion_attention_schedule=schedule))
    assert "step-0 prompt K/V" not in caplog.text


def test_supports_request_batch_enabled():
    assert HunyuanImage3Pipeline.supports_request_batch is True


def _layout_rows(token_rows: list[list[int]], image_start: int) -> HunyuanPreparedLayout:
    tokens = torch.tensor(token_rows, dtype=torch.long)
    branches, seq_len = tokens.shape
    image_span = slice(image_start, seq_len)
    return HunyuanPreparedLayout(
        tokenizer_output=TokenizerEncodeOutput(
            tokens=tokens,
            gen_image_mask=torch.ones(branches, seq_len, dtype=torch.bool),
            gen_timestep_scatter_index=torch.tensor([[image_start] for _ in range(branches)]),
            all_image_slices=[[image_span] for _ in range(branches)],
            joint_image_slices=[[] for _ in range(branches)],
            gen_image_slices=[[image_span] for _ in range(branches)],
            real_pos=torch.tensor([[seq_len] for _ in range(branches)]),
        ),
        rope_image_info=[[(image_span, (4, 4))] for _ in range(branches)],
        generated_image_info=ImageInfo(
            image_type="gen_image",
            image_width=512,
            image_height=512,
            token_width=4,
            token_height=4,
            image_token_length=16,
        ),
    )


def _omni_request(request_id: str, layout: HunyuanPreparedLayout, **sampling_updates):
    sampling = _sampling_params()
    for key, value in sampling_updates.items():
        setattr(sampling, key, value)
    return SimpleNamespace(
        request_id=request_id,
        prompt=f"prompt-{request_id}",
        sampling_params=sampling,
        prepared_layout=layout,
        kv_computed_tokens=(),
    )


def test_pack_hunyuan_prepared_layouts_is_cfg_major():
    cond_uncond_a = _layout_rows([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]], image_start=1)
    cond_uncond_b = _layout_rows([[21, 22, 23, 24, 25, 26, 27], [31, 32, 33, 34, 35, 36, 37]], image_start=3)
    packed = hy3_layout_module.pack_hunyuan_prepared_layouts(
        [cond_uncond_a, cond_uncond_b],
        pad_token_id=0,
    )

    assert packed.tokenizer_output.tokens.tolist() == [
        [1, 2, 3, 4, 5, 0, 0],
        [21, 22, 23, 24, 25, 26, 27],
        [10, 11, 12, 13, 14, 0, 0],
        [31, 32, 33, 34, 35, 36, 37],
    ]
    assert packed.num_branches == 4
    assert packed.generated_image_info.image_token_length == 16


def test_pack_then_slice_cached_prefix_keeps_logical_seq_lens():
    packed = hy3_layout_module.pack_hunyuan_prepared_layouts(
        [
            _layout_rows([[1] * 21], image_start=5),
            _layout_rows([[2] * 25], image_start=9),
        ],
        pad_token_id=0,
    )
    tokens = packed.tokenizer_output.tokens
    seq_lens = [int(row[-1].item()) for row in packed.tokenizer_output.real_pos]
    assert seq_lens == [21, 25]
    batch, width = tokens.shape
    sliced = HunyuanImage3Pipeline._slice_cached_prefix_inputs(
        tokens.unsqueeze(-1).to(torch.float32),
        torch.ones(batch, 1, width, width),
        tokens.clone(),
        (tokens.clone(),),
        torch.zeros(batch, width, dtype=torch.bool),
        packed.tokenizer_output.gen_timestep_scatter_index.clone(),
        list(seq_lens),
        list(seq_lens),
        4,
    )
    assert sliced[-2] == [17, 21]
    assert sliced[-1] == width - 4
    torch.testing.assert_close(sliced[0][0, :17, 0], tokens[0, 4:21].to(torch.float32))
    torch.testing.assert_close(sliced[0][1, :, 0], tokens[1, 4:].to(torch.float32))


def _i2i_layout_rows(token_row: list[int], ref_span: tuple[int, int], image_start: int) -> HunyuanPreparedLayout:
    tokens = torch.tensor([token_row], dtype=torch.long)
    seq_len = tokens.shape[1]
    image_span = slice(image_start, seq_len)
    ref = slice(ref_span[0], ref_span[1])
    return HunyuanPreparedLayout(
        tokenizer_output=TokenizerEncodeOutput(
            tokens=tokens,
            gen_image_mask=torch.ones(1, seq_len, dtype=torch.bool),
            gen_timestep_scatter_index=torch.tensor([[image_start]]),
            all_image_slices=[[ref, image_span]],
            joint_image_slices=[[ref]],
            gen_image_slices=[[image_span]],
            real_pos=torch.tensor([[seq_len]]),
        ),
        rope_image_info=[[(ref, (2, 2)), (image_span, (4, 4))]],
        generated_image_info=ImageInfo(
            image_type="gen_image",
            image_width=512,
            image_height=512,
            token_width=4,
            token_height=4,
            image_token_length=16,
        ),
    )


def test_request_groups_pack_same_structure_different_seq_len():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_kv_mode = hy3_module.DiffusionKVCacheMode.PAGED_SCHEDULER
    matching = [
        _omni_request("match-0", _layout_rows([[1] * 21], image_start=5)),
        _omni_request("match-1", _layout_rows([[2] * 21], image_start=5)),
    ]
    different_seq = _omni_request("long", _layout_rows([[3] * 25], image_start=9))
    different_prefix = _omni_request("hit", _layout_rows([[4] * 21], image_start=5))
    different_prefix.kv_computed_tokens = (4,)
    ar_kv = _omni_request("ar", _layout_rows([[5] * 21], image_start=5))
    ar_kv.sampling_params.past_key_values = SimpleNamespace(
        key_cache=[torch.zeros(2, 4, 1, 1)],
        value_cache=[torch.zeros(2, 4, 1, 1)],
    )

    groups = pipeline._split_request_groups([*matching, different_seq, different_prefix, ar_kv])
    group_ids = {frozenset(request.request_id for request in group) for group in groups}

    assert frozenset({"match-0", "match-1", "long", "hit"}) in group_ids
    assert frozenset({"ar"}) in group_ids


def test_request_groups_pack_i2i_reference_spans_at_different_prefix_offsets():
    pipeline = _pipeline()
    pipeline.od_config.diffusion_kv_mode = hy3_module.DiffusionKVCacheMode.PAGED_SCHEDULER
    first = _omni_request("i2i-a", _i2i_layout_rows([1] * 25, (1, 4), 9))
    first.kv_computed_tokens = (9,)
    second = _omni_request("i2i-b", _i2i_layout_rows([2] * 21, (2, 4), 5))
    second.kv_computed_tokens = (5,)

    groups = pipeline._split_request_groups([first, second])

    assert [[request.request_id for request in group] for group in groups] == [["i2i-a", "i2i-b"]]


def test_prepare_model_inputs_accepts_packed_request_batch(monkeypatch):
    pipeline = _pipeline()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    monkeypatch.setattr(
        hy3_module,
        "build_batch_2d_rope",
        lambda **_kwargs: (torch.zeros(1), torch.zeros(1)),
    )
    pipeline._tkwrapper = SimpleNamespace(
        apply_chat_template=lambda **_kwargs: pytest.fail("packed execution must not tokenize again"),
        eos_token_id=2,
        boi_token_id=3,
        end_recaption_token_id=4,
        end_answer_token_id=5,
        pad_token_id=0,
    )
    pipeline.config = SimpleNamespace(attention_head_dim=2, rope_theta=10000.0)
    packed = hy3_layout_module.pack_hunyuan_prepared_layouts(
        [
            _layout_rows([[1] * 21, [11] * 21], image_start=5),
            _layout_rows([[2] * 21, [12] * 21], image_start=5),
        ],
        pad_token_id=0,
    )

    model_inputs = pipeline.prepare_model_inputs(
        prompt=["prompt-a", "prompt-b"],
        mode="gen_image",
        guidance_scale=2.5,
        image_size=(512, 512),
        device=torch.device("cpu"),
        generator=[torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)],
        prepared_layout=packed,
    )

    assert model_inputs["tokenizer_output"].tokens.shape[0] == 4
    assert model_inputs["batch_gen_image_info"] == [packed.generated_image_info] * 2
    assert model_inputs["input_ids"].shape == (4, 21)


def test_forward_request_batch_returns_list_and_keeps_single_output(monkeypatch):
    pipeline = _pipeline()
    captured: dict[str, object] = {}

    def fake_prepare_model_inputs(**kwargs):
        captured["prepare"] = kwargs
        layout = kwargs["prepared_layout"]
        return {
            "mode": "gen_image",
            "tokenizer_output": layout.tokenizer_output,
            "batch_gen_image_info": [layout.generated_image_info],
            "prepared_layout": layout,
            "paged_request_ids": kwargs.get("paged_request_ids"),
        }

    def fake_generate(**kwargs):
        captured["paged_request_ids"] = kwargs.get("paged_request_ids")
        captured["prepared_rows"] = int(kwargs["prepared_layout"].num_branches)
        batch_size = len(kwargs["paged_request_ids"])
        return torch.arange(batch_size * 2).reshape(batch_size, 2)

    monkeypatch.setattr(pipeline, "prepare_model_inputs", fake_prepare_model_inputs)
    pipeline._generate = fake_generate

    matching = [
        _omni_request("req-0", _layout_rows([[1] * 21], image_start=5), guidance_scale=1.0),
        _omni_request("req-1", _layout_rows([[2] * 21], image_start=5), guidance_scale=1.0),
    ]
    outputs = pipeline.forward(DiffusionRequestBatch(requests=matching))
    assert isinstance(outputs, list)
    assert [output.output["payload"]["image"].tolist() for output in outputs] == [[0, 1], [2, 3]]
    assert captured["paged_request_ids"] == ["req-0", "req-1"]
    assert captured["prepared_rows"] == 2
    assert captured["prepare"]["paged_request_ids"] == ["req-0", "req-1"]

    single = pipeline.forward(DiffusionRequestBatch(requests=matching[:1]))
    assert not isinstance(single, list)
    assert single.output["payload"]["image"].tolist() == [0, 1]


def test_forward_request_batch_packs_different_seq_len_under_paged(monkeypatch):
    pipeline = _pipeline()
    pipeline.od_config.diffusion_kv_mode = hy3_module.DiffusionKVCacheMode.PAGED_SCHEDULER
    generate_ids: list[list[str]] = []

    def fake_prepare_model_inputs(**kwargs):
        layout = kwargs["prepared_layout"]
        return {
            "mode": "gen_image",
            "tokenizer_output": layout.tokenizer_output,
            "batch_gen_image_info": [layout.generated_image_info],
            "prepared_layout": layout,
            "paged_request_ids": kwargs.get("paged_request_ids"),
        }

    def fake_generate(**kwargs):
        request_ids = list(kwargs["paged_request_ids"])
        generate_ids.append(request_ids)
        return torch.tensor([[idx] for idx in range(len(request_ids))])

    monkeypatch.setattr(pipeline, "prepare_model_inputs", fake_prepare_model_inputs)
    pipeline._generate = fake_generate

    requests = [
        _omni_request("short", _layout_rows([[1] * 21], image_start=5), guidance_scale=1.0),
        _omni_request("long", _layout_rows([[2] * 25], image_start=9), guidance_scale=1.0),
    ]
    outputs = pipeline.forward(DiffusionRequestBatch(requests=requests))

    assert [output.output["payload"]["image"].tolist() for output in outputs] == [[0], [1]]
    assert generate_ids == [["short", "long"]]
    assert requests[1].prepared_layout.tokenizer_output.tokens.shape[1] == 25


def test_forward_request_group_publishes_per_row_mixed_prefix(monkeypatch):
    from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context

    pipeline = _pipeline()
    pipeline.od_config.diffusion_kv_mode = hy3_module.DiffusionKVCacheMode.PAGED_SCHEDULER
    captured: dict[str, object] = {}

    def fake_prepare_model_inputs(**kwargs):
        captured["kv_computed_tokens"] = kwargs.get("kv_computed_tokens")
        layout = kwargs["prepared_layout"]
        return {
            "mode": "gen_image",
            "tokenizer_output": layout.tokenizer_output,
            "batch_gen_image_info": [layout.generated_image_info],
            "prepared_layout": layout,
            "paged_request_ids": kwargs.get("paged_request_ids"),
        }

    def fake_generate(**kwargs):
        context = get_forward_context()
        captured["prefix_len"] = context.paged_kv_cached_prefix_len
        captured["prefix_cut"] = kwargs.get("hunyuan_paged_prefix_cut")
        captured["row_keys"] = kwargs.get("paged_row_keys")
        return torch.tensor([[0], [1]])

    monkeypatch.setattr(pipeline, "prepare_model_inputs", fake_prepare_model_inputs)
    pipeline._generate = fake_generate

    cold = _omni_request("cold", _layout_rows([[1] * 21], image_start=5), guidance_scale=1.0)
    hit = _omni_request("hit", _layout_rows([[2] * 21], image_start=5), guidance_scale=1.0)
    hit.kv_computed_tokens = (4,)
    with set_forward_context():
        outputs = pipeline._forward_request_group(
            [cold, hit],
            prompt="",
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=1.0,
            generator=None,
            kwargs={},
        )

    assert [output.output["payload"]["image"].tolist() for output in outputs] == [[0], [1]]
    assert captured["prefix_len"] == 0
    assert captured["prefix_cut"] == (0, 4)
    assert captured["row_keys"] == [("cold", 0), ("hit", 0)]
    assert captured["kv_computed_tokens"] == ()


def test_forward_request_batch_preserves_order_when_groups_split(monkeypatch):
    pipeline = _pipeline()
    generate_ids: list[list[str]] = []

    def fake_prepare_model_inputs(**kwargs):
        layout = kwargs["prepared_layout"]
        return {
            "mode": "gen_image",
            "tokenizer_output": layout.tokenizer_output,
            "batch_gen_image_info": [layout.generated_image_info],
            "prepared_layout": layout,
            "paged_request_ids": kwargs.get("paged_request_ids"),
        }

    def fake_generate(**kwargs):
        request_ids = list(kwargs["paged_request_ids"])
        generate_ids.append(request_ids)
        return torch.tensor([[ord(request_ids[0][0])]])

    monkeypatch.setattr(pipeline, "prepare_model_inputs", fake_prepare_model_inputs)
    pipeline._generate = fake_generate

    ar_kv = _omni_request("ar", _layout_rows([[1] * 21], image_start=5), guidance_scale=1.0)
    ar_kv.sampling_params.past_key_values = SimpleNamespace(
        key_cache=[torch.zeros(2, 4, 1, 1)],
        value_cache=[torch.zeros(2, 4, 1, 1)],
    )
    requests = [
        _omni_request("plain", _layout_rows([[2] * 21], image_start=5), guidance_scale=1.0),
        ar_kv,
    ]
    outputs = pipeline.forward(DiffusionRequestBatch(requests=requests))

    assert [output.output["payload"]["image"].tolist() for output in outputs] == [[ord("p")], [ord("a")]]
    assert generate_ids == [["plain"], ["ar"]]


def test_record_denoise_step_forwards_paged_row_selection(monkeypatch):
    """Scheduled publish must keep the active group's row keys (review P1)."""
    from vllm_omni.diffusion.forward_context import DenoiseProgressMixin, set_forward_context

    captured: list[dict[str, object]] = []

    class _Runtime:
        def ensure_active(self, step_idx, request_ids=None, row_keys=None):
            captured.append(
                {
                    "step_idx": step_idx,
                    "request_ids": None if request_ids is None else list(request_ids),
                    "row_keys": None if row_keys is None else list(row_keys),
                }
            )
            return object()

    class _Pipe(DenoiseProgressMixin):
        pass

    with set_forward_context(paged_kv_runtime=_Runtime()):
        _Pipe().record_denoise_step(
            2,
            total_steps=8,
            request_ids=["req-a", "req-b"],
            paged_row_keys=[("req-a", 0), ("req-b", 0), ("req-a", 1), ("req-b", 1)],
            paged_step_idx=1,
        )

    assert captured == [
        {
            "step_idx": 1,
            "request_ids": ["req-a", "req-b"],
            "row_keys": [("req-a", 0), ("req-b", 0), ("req-a", 1), ("req-b", 1)],
        }
    ]


def test_scheduled_generate_path_keeps_paged_row_keys_for_split_groups(monkeypatch):
    """Nonempty schedule must not activate the full Scheduler batch without row keys."""
    from vllm_omni.diffusion.attention.schedule import AttentionScheduleRange
    from vllm_omni.diffusion.forward_context import (
        DenoiseProgressMixin,
        bind_attention_schedule,
        set_forward_context,
    )
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        HunyuanImage3Text2ImagePipeline,
    )

    captured: list[list[tuple[str, int]] | None] = []

    class _Runtime:
        def ensure_active(self, step_idx, request_ids=None, row_keys=None):
            captured.append(None if row_keys is None else list(row_keys))
            return object()

    class _Model(DenoiseProgressMixin):
        pass

    pipe = object.__new__(HunyuanImage3Text2ImagePipeline)
    pipe.model = _Model()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    pipe.progress_bar = lambda total: nullcontext()

    # Drive only the denoise-step publish branch used under a bound schedule.
    model_kwargs = {
        "paged_request_ids": ["short", "long"],
        "paged_row_keys": [("short", 0), ("long", 0)],
    }
    with (
        set_forward_context(paged_kv_runtime=_Runtime()),
        bind_attention_schedule((AttentionScheduleRange(start=0, end=None, profile="sparse"),)),
    ):
        from vllm_omni.diffusion.forward_context import begin_scheduled_denoise

        scheduled_total = begin_scheduled_denoise(2)
        assert scheduled_total == 2
        for i in range(2):
            pipe.model.record_denoise_step(
                i,
                torch.tensor(100.0 * (2 - i)),
                scheduler=pipe.scheduler,
                total_steps=scheduled_total,
                request_ids=model_kwargs.get("paged_request_ids"),
                paged_row_keys=model_kwargs.get("paged_row_keys"),
                paged_step_idx=0 if i == 0 else 1,
            )

    assert captured == [
        [("short", 0), ("long", 0)],
        [("short", 0), ("long", 0)],
    ]

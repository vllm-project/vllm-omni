# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline():
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoPipeline

    pipeline = object.__new__(LingBotVideoPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline._execution_device = pipeline.device
    pipeline.default_negative_prompt = "default negative"
    pipeline.default_image_negative_prompt = "default image negative"
    pipeline.od_config = SimpleNamespace(flow_shift=None)
    return pipeline


def test_execution_options_resolve_refiner_startup_default_and_request_override():
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotGenerationMode,
        LingBotRefinerConfig,
        normalize_lingbot_execution_options,
    )

    startup = LingBotRefinerConfig(enabled=True, default_run=True)
    defaulted = normalize_lingbot_execution_options(
        {},
        refiner_config=startup,
        mode=LingBotGenerationMode.T2V,
    )
    assert defaulted.refiner.run is True
    assert defaulted.refiner.explicitly_requested is False

    bypassed = normalize_lingbot_execution_options(
        {"run_refiner": False, "refiner_steps": 4},
        refiner_config=startup,
        mode=LingBotGenerationMode.T2V,
    )
    assert bypassed.refiner.run is False
    assert bypassed.refiner.explicitly_requested is True
    assert bypassed.refiner.num_inference_steps == 4


def test_execution_options_reject_unloaded_or_t2i_refiner_before_model_work():
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotGenerationMode,
        LingBotRefinerConfig,
        normalize_lingbot_execution_options,
    )

    with pytest.raises(ValueError, match="startup configuration"):
        normalize_lingbot_execution_options(
            {"run_refiner": True},
            mode=LingBotGenerationMode.T2V,
        )
    with pytest.raises(ValueError, match="only supported for video"):
        normalize_lingbot_execution_options(
            {"run_refiner": True},
            refiner_config=LingBotRefinerConfig(enabled=True),
            mode=LingBotGenerationMode.T2I,
        )

    t2i_default = normalize_lingbot_execution_options(
        {},
        refiner_config=LingBotRefinerConfig(enabled=True, default_run=True),
        mode=LingBotGenerationMode.T2I,
    )
    assert t2i_default.refiner.run is False


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        ({"refiner_height": 1080}, "multiples of 16"),
        ({"refiner_steps": 0}, "positive integer"),
        ({"refiner_t_thresh": 1.1}, "must lie in"),
        ({"refiner_max_video_frames": 8}, "must be 1 or"),
        ({"refiner_null_cond_clone_zero": "true"}, "must be a boolean"),
    ],
)
def test_execution_options_validate_refiner_fields_even_when_bypassed(extra_args, message):
    from vllm_omni.diffusion.models.lingbot_video import (
        normalize_lingbot_execution_options,
    )

    with pytest.raises(ValueError, match=message):
        normalize_lingbot_execution_options(extra_args)


def test_execution_options_use_configured_vae_temporal_factor():
    from vllm_omni.diffusion.models.lingbot_video import (
        normalize_lingbot_execution_options,
    )

    options = normalize_lingbot_execution_options(
        {"refiner_max_video_frames": 7},
        vae_temporal_factor=3,
    )

    assert options.refiner.max_video_frames == 7
    with pytest.raises(ValueError, match=r"3n\+1"):
        normalize_lingbot_execution_options(
            {"refiner_max_video_frames": 5},
            vae_temporal_factor=3,
        )


def test_postprocess_preserves_refined_video_metadata_envelope():
    from vllm_omni.diffusion.models.lingbot_video import (
        get_lingbot_video_post_process_func,
    )

    frames = torch.ones(2, 4, 4, 3)
    result = get_lingbot_video_post_process_func(SimpleNamespace())(
        {
            "payload": {"video": frames},
            "metadata": {
                "video": {
                    "fps": 24,
                    "refined": True,
                    "source_fps": 12.0,
                    "sample_fps": 24,
                    "sample_frames": 2,
                }
            },
        },
        SimpleNamespace(output_type="pt"),
    )

    assert result["payload"]["video"] is frames
    assert result["metadata"]["video"]["fps"] == 24
    assert result["metadata"]["video"]["refined"] is True


def test_refiner_reuses_t2v_positive_but_rebuilds_ti2v_as_text_only():
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotGenerationMode,
        LingBotRefinerOptions,
    )
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        LingBotStageCondition,
    )

    pipeline = _make_pipeline()
    positive = torch.ones(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.long)
    base_condition = LingBotStageCondition(
        prompt_embeds=positive,
        prompt_mask=mask,
        negative_prompt_embeds=None,
        negative_prompt_mask=None,
        image_condition=None,
        cfg_parallel_group=None,
        cfg_parallel_rank=0,
    )
    encode_calls = []

    def fake_encode(prompt, *, images, device):
        encode_calls.append((prompt, images, device))
        return torch.full((1, 2, 4), 2.0), torch.ones(1, 2, dtype=torch.long)

    pipeline.encode_prompt = fake_encode
    options = LingBotRefinerOptions(null_cond_clone_zero=True)
    t2v = pipeline._prepare_refiner_condition(
        prompt="positive",
        negative_prompt="negative",
        mode=LingBotGenerationMode.T2V,
        base_condition=base_condition,
        guidance_scale=3.0,
        cfg_parallel_group=None,
        options=options,
        clean_prefix=None,
    )
    assert t2v.prompt_embeds is positive
    assert encode_calls == []
    assert torch.equal(t2v.negative_prompt_embeds, torch.zeros_like(positive))

    ti2v = pipeline._prepare_refiner_condition(
        prompt="positive",
        negative_prompt="negative",
        mode=LingBotGenerationMode.TI2V,
        base_condition=base_condition,
        guidance_scale=3.0,
        cfg_parallel_group=None,
        options=options,
        clean_prefix=torch.ones(1, 1, 1, 1, 1),
    )
    assert [(prompt, images) for prompt, images, _ in encode_calls] == [("positive", None)]
    assert ti2v.prompt_embeds.shape[1] == 2
    assert ti2v.clean_prefix is not None
    assert torch.equal(ti2v.negative_prompt_embeds, torch.zeros_like(ti2v.prompt_embeds))


def test_generate_runs_in_memory_refiner_handoff_and_returns_only_final_video():
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotExecutionOptions,
        LingBotGenerationMode,
        LingBotRefinerConfig,
        LingBotRefinerInputs,
        LingBotRefinerOptions,
    )
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        LingBotStageCondition,
    )

    pipeline = _make_pipeline()
    pipeline.vae = nn.Linear(1, 1, bias=False)
    pipeline.refiner_transformer = nn.Linear(1, 1, bias=False)
    pipeline.refiner_scheduler = SimpleNamespace()
    pipeline.refiner_config = LingBotRefinerConfig(
        enabled=True,
        offload_vae_during_denoise=False,
    )
    condition = LingBotStageCondition(
        prompt_embeds=torch.ones(1, 2, 4),
        prompt_mask=torch.ones(1, 2, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_prompt_mask=None,
        image_condition=None,
        cfg_parallel_group=None,
        cfg_parallel_rank=0,
    )
    pipeline._prepare_base_condition = lambda **kwargs: condition
    pipeline.diffuse = lambda **kwargs: torch.zeros(1, 1, 3, 2, 2)
    decode_calls = []

    def fake_decode(latents):
        decode_calls.append(latents)
        return torch.ones(1, 3, 9, 16, 16)

    pipeline._decode_latents_internal = fake_decode
    refiner_inputs = LingBotRefinerInputs(
        latents=torch.ones(1, 1, 3, 2, 2),
        clean_prefix=None,
        num_frames=9,
        source_fps=24.0,
        sample_fps=24,
    )
    refiner_generators = []

    def fake_prepare_refiner_inputs(**kwargs):
        refiner_generators.append(kwargs["generator"])
        return refiner_inputs

    pipeline._prepare_refiner_inputs = fake_prepare_refiner_inputs
    pipeline._prepare_refiner_condition = lambda **kwargs: condition

    def fake_diffuse_refiner(**kwargs):
        refiner_generators.append(kwargs["generator"])
        return torch.full((1, 1, 3, 2, 2), 2.0)

    pipeline._diffuse_refiner = fake_diffuse_refiner
    base_generator = torch.Generator(device="cpu").manual_seed(123)

    result = pipeline._generate(
        prompt="a robot",
        mode=LingBotGenerationMode.T2V,
        height=16,
        width=16,
        num_frames=9,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=base_generator,
        output_type="pt",
        execution_options=LingBotExecutionOptions(
            refiner=LingBotRefinerOptions(
                run=True,
                height=16,
                width=16,
                num_inference_steps=1,
            )
        ),
    )

    assert len(decode_calls) == 2
    assert torch.equal(
        decode_calls[1],
        torch.full((1, 1, 3, 2, 2), 2.0),
    )
    assert result.shape == (9, 16, 16, 3)
    assert len(refiner_generators) == 2
    assert refiner_generators[0] is refiner_generators[1]
    assert refiner_generators[0] is not base_generator
    assert refiner_generators[0].initial_seed() == base_generator.initial_seed()


def test_refiner_offloads_vae_before_ti2v_text_condition():
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotExecutionOptions,
        LingBotGenerationMode,
        LingBotRefinerConfig,
        LingBotRefinerInputs,
        LingBotRefinerOptions,
    )
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        LingBotStageCondition,
    )

    pipeline = _make_pipeline()
    pipeline.vae = nn.Linear(1, 1, bias=False)
    pipeline.refiner_transformer = nn.Linear(1, 1, bias=False)
    pipeline.refiner_scheduler = SimpleNamespace()
    pipeline.refiner_config = LingBotRefinerConfig(
        enabled=True,
        offload_vae_during_denoise=True,
    )
    condition = LingBotStageCondition(
        prompt_embeds=torch.ones(1, 2, 4),
        prompt_mask=torch.ones(1, 2, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_prompt_mask=None,
        image_condition=None,
        cfg_parallel_group=None,
        cfg_parallel_rank=0,
    )
    refiner_inputs = LingBotRefinerInputs(
        latents=torch.ones(1, 1, 1, 2, 2),
        clean_prefix=None,
        num_frames=1,
        source_fps=24.0,
        sample_fps=24,
    )
    events = []
    pipeline._prepare_base_condition = lambda **kwargs: condition
    pipeline.diffuse = lambda **kwargs: torch.zeros(1, 1, 1, 2, 2)
    pipeline._decode_latents_internal = lambda latents: events.append("decode") or torch.ones(1, 3, 1, 16, 16)
    pipeline._prepare_refiner_inputs = lambda **kwargs: events.append("handoff") or refiner_inputs
    pipeline._offload_vae_for_denoise = lambda **kwargs: events.append("offload:" + str(kwargs["enabled"])) or None
    pipeline._restore_vae_for_decode = lambda device: events.append("restore")
    pipeline._prepare_refiner_condition = lambda **kwargs: events.append("refiner_condition") or condition
    pipeline._diffuse_refiner = lambda **kwargs: events.append("refiner_diffuse") or torch.ones(1, 1, 1, 2, 2)

    pipeline._generate(
        prompt="a robot",
        mode=LingBotGenerationMode.TI2V,
        input_image=object(),
        height=16,
        width=16,
        num_frames=1,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="pt",
        execution_options=LingBotExecutionOptions(
            refiner=LingBotRefinerOptions(
                run=True,
                height=16,
                width=16,
                num_inference_steps=1,
            )
        ),
    )

    assert events.index("handoff") < events.index("offload:True")
    assert events.index("offload:True") < events.index("refiner_condition")
    assert events.index("refiner_condition") < events.index("refiner_diffuse")


def test_pipeline_profiler_distinguishes_base_handoff_and_refiner_stages():
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotExecutionOptions,
        LingBotGenerationMode,
        LingBotRefinerConfig,
        LingBotRefinerInputs,
        LingBotRefinerOptions,
    )
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        LingBotStageCondition,
    )

    pipeline = _make_pipeline()
    pipeline.vae = nn.Linear(1, 1, bias=False)
    pipeline.refiner_transformer = nn.Linear(1, 1, bias=False)
    pipeline.refiner_scheduler = SimpleNamespace()
    pipeline.refiner_config = LingBotRefinerConfig(
        enabled=True,
        offload_vae_during_denoise=False,
    )
    condition = LingBotStageCondition(
        prompt_embeds=torch.ones(1, 2, 4),
        prompt_mask=torch.ones(1, 2, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_prompt_mask=None,
        image_condition=None,
        cfg_parallel_group=None,
        cfg_parallel_rank=0,
    )
    refiner_inputs = LingBotRefinerInputs(
        latents=torch.ones(1, 1, 1, 2, 2),
        clean_prefix=None,
        num_frames=1,
        source_fps=24.0,
        sample_fps=24,
    )
    pipeline._prepare_base_condition = lambda **kwargs: condition
    pipeline.diffuse = lambda **kwargs: torch.zeros(1, 1, 1, 2, 2)
    pipeline._decode_latents_internal = lambda latents: torch.ones(1, 3, 1, 16, 16)
    pipeline._prepare_refiner_inputs = lambda **kwargs: refiner_inputs
    pipeline._prepare_refiner_condition = lambda **kwargs: condition
    pipeline._diffuse_refiner = lambda **kwargs: torch.ones(1, 1, 1, 2, 2)
    pipeline.setup_diffusion_pipeline_profiler(
        profiler_targets=list(pipeline._PROFILER_TARGETS),
        enable_diffusion_pipeline_profiler=True,
    )

    pipeline._generate(
        prompt="a robot",
        mode=LingBotGenerationMode.T2V,
        height=16,
        width=16,
        num_frames=1,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="pt",
        execution_options=LingBotExecutionOptions(
            refiner=LingBotRefinerOptions(
                run=True,
                height=16,
                width=16,
                num_inference_steps=1,
            )
        ),
    )

    assert set(pipeline.stage_durations) == {
        "LingBotVideoPipeline._prepare_base_condition",
        "LingBotVideoPipeline.diffuse",
        "LingBotVideoPipeline._decode_latents_internal",
        "LingBotVideoPipeline._prepare_refiner_inputs",
        "LingBotVideoPipeline._prepare_refiner_condition",
        "LingBotVideoPipeline._diffuse_refiner",
    }

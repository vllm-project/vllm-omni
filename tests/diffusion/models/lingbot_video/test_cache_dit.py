# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.diffusion,
    pytest.mark.cache,
]


def _make_pipeline():
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoPipeline

    pipeline = object.__new__(LingBotVideoPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline._cache_dit_stage_refreshers = {}
    return pipeline


def _make_config(*, parallel_kwargs=None, **config_kwargs):
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    defaults = {
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": False,
    }
    defaults.update(config_kwargs)
    return SimpleNamespace(
        cache_backend="cache_dit",
        parallel_config=DiffusionParallelConfig(**(parallel_kwargs or {})),
        **defaults,
    )


@pytest.mark.parametrize(
    ("parallel_kwargs", "config_kwargs", "message"),
    [
        ({"pipeline_parallel_size": 2}, {}, "pipeline parallelism"),
        ({"tensor_parallel_size": 2}, {}, "tensor parallelism"),
        ({"enable_expert_parallel": True}, {}, "expert parallelism"),
        ({"ulysses_degree": 2}, {}, "sequence parallelism"),
        ({"cfg_parallel_size": 2}, {}, "CFG parallelism"),
        ({"vae_patch_parallel_size": 2}, {}, "VAE patch parallelism"),
        (
            {"use_hsdp": True, "hsdp_shard_size": 2},
            {},
            "HSDP",
        ),
        (
            {},
            {"enable_distributed_layerwise_offload": True},
            "distributed layerwise offload",
        ),
    ],
)
def test_cache_dit_rejects_unvalidated_parallel_and_offload_combinations(
    parallel_kwargs,
    config_kwargs,
    message,
):
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoPipeline

    config = _make_config(
        parallel_kwargs=parallel_kwargs,
        **config_kwargs,
    )

    with pytest.raises(ValueError, match=message):
        LingBotVideoPipeline._validate_cache_dit_configuration(config)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"enable_cpu_offload": True},
        {"enable_layerwise_offload": True},
    ],
)
def test_cache_dit_allows_supported_offload_modes(config_kwargs):
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoPipeline

    LingBotVideoPipeline._validate_cache_dit_configuration(_make_config(**config_kwargs))


def test_invalid_cache_dit_config_fails_before_model_prefetch(monkeypatch):
    from vllm_omni.diffusion.models.lingbot_video import (
        pipeline_lingbot_video as module,
    )

    prefetch_calls = []
    monkeypatch.setattr(
        module,
        "prefetch_subfolders",
        lambda *args, **kwargs: prefetch_calls.append((args, kwargs)),
    )
    config = _make_config(parallel_kwargs={"tensor_parallel_size": 2})

    with pytest.raises(ValueError, match="tensor parallelism"):
        module.LingBotVideoPipeline(od_config=config)

    assert prefetch_calls == []


@pytest.mark.parametrize(
    (
        "base_guidance",
        "base_batch_cfg",
        "refiner_guidance",
        "refiner_batch_cfg",
        "message",
    ),
    [
        (1.0, False, None, False, "Base guidance_scale"),
        (3.0, True, None, False, "Base batch_cfg"),
        (3.0, False, 1.0, False, "Refiner guidance_scale"),
        (3.0, False, 3.0, True, "Refiner batch_cfg"),
    ],
)
def test_cache_dit_requires_sequential_two_pass_cfg(
    base_guidance,
    base_batch_cfg,
    refiner_guidance,
    refiner_batch_cfg,
    message,
):
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotExecutionOptions,
        LingBotRefinerOptions,
    )

    pipeline = _make_pipeline()
    pipeline._cache_dit_stage_refreshers = {"transformer": lambda *args: None}
    options = LingBotExecutionOptions(
        batch_cfg=base_batch_cfg,
        refiner=LingBotRefinerOptions(
            run=refiner_guidance is not None,
            guidance_scale=refiner_guidance or 3.0,
            batch_cfg=refiner_batch_cfg,
        ),
    )

    with pytest.raises(ValueError, match=message):
        pipeline._validate_cache_dit_request(
            SimpleNamespace(guidance_scale=base_guidance),
            options,
        )


def test_cache_dit_request_checks_are_inactive_without_backend():
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotExecutionOptions,
    )

    pipeline = _make_pipeline()
    pipeline._validate_cache_dit_request(
        SimpleNamespace(guidance_scale=1.0),
        LingBotExecutionOptions(batch_cfg=True),
    )


def test_cache_dit_allows_cfg_off_for_engine_dummy_run():
    from vllm_omni.diffusion.models.lingbot_video import LingBotExecutionOptions

    pipeline = _make_pipeline()
    pipeline._cache_dit_stage_refreshers = {"transformer": lambda *args: None}
    pipeline._validate_cache_dit_request(
        SimpleNamespace(guidance_scale=0.0),
        LingBotExecutionOptions(batch_cfg=False),
        is_dummy_run=True,
    )


class _ShortScheduler:
    sigma_max = 1.0
    sigma_min = 0.0

    def set_timesteps(self, num_inference_steps, **kwargs):
        del kwargs
        self.timesteps = list(range(max(int(num_inference_steps) - 1, 0)))


def test_base_cache_refresh_uses_actual_scheduler_length():
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        LingBotStageSettings,
    )

    pipeline = _make_pipeline()
    pipeline.scheduler = _ShortScheduler()
    pipeline.transformer = nn.Identity()
    pipeline.prepare_latents = lambda *args: torch.zeros(1, 1, 1, 1, 1)
    pipeline._run_denoise_stage = lambda **kwargs: kwargs["latents"]
    refresh_calls = []
    pipeline._cache_dit_stage_refreshers = {
        "transformer": lambda active, steps, verbose: refresh_calls.append((active, steps, verbose))
    }

    pipeline.diffuse(
        num_frames=1,
        height=16,
        width=16,
        generator=None,
        latents=None,
        condition=SimpleNamespace(clean_prefix=None),
        settings=LingBotStageSettings(
            num_inference_steps=10,
            guidance_scale=3.0,
            shift=3.0,
            batch_cfg=False,
            base_low_noise_threshold=None,
            base_sigma_tail_steps=0,
        ),
    )

    assert refresh_calls == [(pipeline, 9, False)]


def test_refiner_cache_refresh_uses_actual_scheduler_length(monkeypatch):
    from vllm_omni.diffusion.models.lingbot_video import LingBotRefinerOptions
    from vllm_omni.diffusion.models.lingbot_video import (
        pipeline_lingbot_video as module,
    )

    pipeline = _make_pipeline()
    pipeline.refiner_transformer = nn.Identity()
    pipeline.refiner_scheduler = _ShortScheduler()
    pipeline._run_denoise_stage = lambda **kwargs: kwargs["latents"]
    refresh_calls = []
    pipeline._cache_dit_stage_refreshers = {
        "refiner_transformer": lambda active, steps, verbose: refresh_calls.append((active, steps, verbose))
    }
    monkeypatch.setattr(
        module,
        "compute_refiner_sigmas",
        lambda **kwargs: np.linspace(1.0, 0.0, 6),
    )

    pipeline._diffuse_refiner(
        inputs=SimpleNamespace(latents=torch.zeros(1, 1, 1, 1, 1)),
        condition=SimpleNamespace(),
        generator=None,
        options=LingBotRefinerOptions(
            run=True,
            num_inference_steps=8,
        ),
    )

    assert refresh_calls == [(pipeline, 5, False)]

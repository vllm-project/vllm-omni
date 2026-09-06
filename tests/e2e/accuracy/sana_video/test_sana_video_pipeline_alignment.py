# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.benchmark,
]

PROMPT = "A small red panda walks through a bamboo forest."
NEGATIVE_PROMPT = "blur, distortion, abrupt cuts"
SEED = 42
GUIDANCE_SCALE = 6.0
NUM_INFERENCE_STEPS = 2
HEIGHT = 192
WIDTH = 320
NUM_FRAMES = 9

VARIANTS = {
    "480p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        "revision": "fed3bce411c58a0f688a31afe8f52e61acc2b15f",
        "model_env": "SANA_VIDEO_480P_MODEL",
        "max_abs": 0.2,
        "relative_l2": 0.02,
        "cosine": 0.999,
        "cfg_max_abs": 0.5,
        "cfg_relative_l2": 0.05,
        "cfg_cosine": 0.999,
        "step_max_abs": 0.075,
        "step_relative_l2": 0.012,
        "step_cosine": 0.9999,
    },
    "720p": {
        "model": "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        "revision": "8bda5e623d0f48cd6da3b387b10ca35d15cf1c4e",
        "model_env": "SANA_VIDEO_720P_MODEL",
        "max_abs": 0.1,
        "relative_l2": 0.015,
        "cosine": 0.999,
        "cfg_max_abs": 0.6,
        "cfg_relative_l2": 0.065,
        "cfg_cosine": 0.998,
        "step_max_abs": 0.08,
        "step_relative_l2": 0.014,
        "step_cosine": 0.9999,
    },
}


@dataclass(frozen=True)
class ErrorMetrics:
    shape: tuple[int, ...]
    actual_dtype: torch.dtype
    expected_dtype: torch.dtype
    max_abs: float
    mean_abs: float
    relative_l2: float
    cosine: float

    def __str__(self) -> str:
        return (
            f"shape={self.shape}, actual_dtype={self.actual_dtype}, "
            f"expected_dtype={self.expected_dtype}, max_abs={self.max_abs:.8g}, "
            f"mean_abs={self.mean_abs:.8g}, relative_l2={self.relative_l2:.8g}, "
            f"cosine={self.cosine:.8g}"
        )


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> ErrorMetrics:
    assert actual.shape == expected.shape
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    error = actual_float - expected_float
    expected_norm = torch.linalg.vector_norm(expected_float)
    relative_l2 = torch.linalg.vector_norm(error) / expected_norm.clamp_min(torch.finfo(torch.float32).tiny)
    if actual_float.numel() == 1:
        cosine = torch.tensor(
            1.0 if bool(torch.equal(actual_float, expected_float)) else 0.0,
            device=actual.device,
        )
    else:
        cosine = F.cosine_similarity(actual_float.flatten(), expected_float.flatten(), dim=0)
    return ErrorMetrics(
        shape=tuple(actual.shape),
        actual_dtype=actual.dtype,
        expected_dtype=expected.dtype,
        max_abs=error.abs().max().item(),
        mean_abs=error.abs().mean().item(),
        relative_l2=relative_l2.item(),
        cosine=cosine.item(),
    )


def _assert_stage(
    stage: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_abs: float,
    relative_l2: float,
    cosine: float,
) -> ErrorMetrics:
    result = _metrics(actual, expected)
    print(f"\nSANA-Video alignment [{stage}]: {result}")
    assert result.actual_dtype == result.expected_dtype, f"{stage}: {result}"
    assert result.max_abs <= max_abs, f"{stage}: {result}"
    assert result.relative_l2 <= relative_l2, f"{stage}: {result}"
    assert result.cosine >= cosine, f"{stage}: {result}"
    return result


def _model_source(variant: str) -> tuple[str, str | None]:
    config = VARIANTS[variant]
    model = os.getenv(config["model_env"], config["model"])
    revision = None if Path(model).exists() else config["revision"]
    return model, revision


def _native_method_proxy(reference_pipeline: Any):
    """Build the minimum native-pipeline surface needed by stage helpers.

    Sharing the tokenizer and text encoder makes this a comparison of pipeline
    orchestration rather than a second copy of the same multi-billion-parameter
    encoder. Transformer execution still uses separate Diffusers and native
    modules below.
    """
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    proxy = object.__new__(SanaVideoPipeline)
    torch.nn.Module.__init__(proxy)
    proxy.tokenizer = reference_pipeline.tokenizer
    proxy.text_encoder = reference_pipeline.text_encoder
    proxy.device = torch.device("cuda")
    proxy.vae_scale_factor_temporal = reference_pipeline.vae_scale_factor_temporal
    proxy.vae_scale_factor_spatial = reference_pipeline.vae_scale_factor_spatial
    return proxy


def _cfg_prediction(
    prediction: torch.Tensor,
    *,
    guidance_scale: float,
    latent_channels: int,
    out_channels: int,
) -> torch.Tensor:
    unconditional, conditional = prediction.float().chunk(2)
    prediction = unconditional + guidance_scale * (conditional - unconditional)
    if out_channels // 2 == latent_channels:
        prediction = prediction.chunk(2, dim=1)[0]
    return prediction


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "_hardware",
    [pytest.param(None, marks=hardware_marks(res={"cuda": "H100"}))],
)
def test_sana_video_quick_pipeline_stages_match_diffusers(
    variant: str,
    _hardware,
) -> None:
    """Compare the first denoising step without replacing native attention.

    This deliberately stops after the first scheduler step. It keeps the test
    useful as a stage-localizing PR/nightly check while the frozen 81-frame,
    50-step golden test covers the final latent and decoded-video boundary.
    """
    del _hardware
    from diffusers import SanaVideoPipeline as DiffusersSanaVideoPipeline
    from diffusers.pipelines.sana_video.pipeline_sana_video import (
        retrieve_timesteps as diffusers_retrieve_timesteps,
    )
    from diffusers.schedulers import DPMSolverMultistepScheduler

    from vllm_omni.diffusion.models.sana_video import (
        SanaVideoPipeline,
        SanaVideoTransformer3DModel,
    )
    from vllm_omni.diffusion.models.sana_video.pipeline_sana_video import (
        retrieve_timesteps as native_retrieve_timesteps,
    )

    model, revision = _model_source(variant)
    config = VARIANTS[variant]
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        # No stage in this test decodes latents. Avoid loading a multi-GB VAE
        # that would add memory and runtime without increasing coverage.
        "vae": None,
    }
    if revision is not None:
        load_kwargs["revision"] = revision

    reference = DiffusersSanaVideoPipeline.from_pretrained(model, **load_kwargs).to("cuda")
    assert type(reference) is DiffusersSanaVideoPipeline
    assert type(reference.transformer).__module__.startswith("diffusers.")
    native_proxy = _native_method_proxy(reference)
    assert type(native_proxy) is SanaVideoPipeline
    assert type(native_proxy).__module__.startswith("vllm_omni.")
    try:
        with torch.inference_mode():
            reference_prompt = reference.encode_prompt(
                PROMPT,
                True,
                negative_prompt=NEGATIVE_PROMPT,
                device=torch.device("cuda"),
                clean_caption=False,
                max_sequence_length=300,
                complex_human_instruction=None,
            )
            native_prompt = SanaVideoPipeline.encode_prompt(
                native_proxy,
                PROMPT,
                True,
                negative_prompt=NEGATIVE_PROMPT,
                device=torch.device("cuda"),
                clean_caption=False,
                max_sequence_length=300,
                complex_human_instruction=None,
            )
        prompt_stage_names = (
            "prompt_embeddings",
            "prompt_attention_mask",
            "negative_prompt_embeddings",
            "negative_prompt_attention_mask",
        )
        for stage, actual, expected in zip(prompt_stage_names, native_prompt, reference_prompt, strict=True):
            _assert_stage(
                stage,
                actual,
                expected,
                max_abs=0.0,
                relative_l2=0.0,
                cosine=0.999999,
            )

        reference_prompt_embeds, reference_prompt_mask, reference_negative_embeds, reference_negative_mask = (
            reference_prompt
        )
        encoder_hidden_states = torch.cat([reference_negative_embeds, reference_prompt_embeds])
        encoder_attention_mask = torch.cat([reference_negative_mask, reference_prompt_mask])

        latent_channels = reference.transformer.config.in_channels
        reference_latents = reference.prepare_latents(
            1,
            latent_channels,
            HEIGHT,
            WIDTH,
            NUM_FRAMES,
            torch.float32,
            torch.device("cuda"),
            torch.Generator(device="cuda").manual_seed(SEED),
        )
        native_latents = SanaVideoPipeline.prepare_latents(
            native_proxy,
            1,
            latent_channels,
            HEIGHT,
            WIDTH,
            NUM_FRAMES,
            torch.float32,
            torch.device("cuda"),
            torch.Generator(device="cuda").manual_seed(SEED),
        )
        _assert_stage(
            "initial_noise_latents",
            native_latents,
            reference_latents,
            max_abs=0.0,
            relative_l2=0.0,
            cosine=0.999999,
        )

        reference_scheduler = DPMSolverMultistepScheduler.from_config(reference.scheduler.config)
        native_scheduler = DPMSolverMultistepScheduler.from_config(reference.scheduler.config)
        isolated_scheduler = DPMSolverMultistepScheduler.from_config(reference.scheduler.config)
        reference_timesteps, reference_num_steps = diffusers_retrieve_timesteps(
            reference_scheduler,
            NUM_INFERENCE_STEPS,
            torch.device("cuda"),
        )
        native_timesteps, native_num_steps = native_retrieve_timesteps(
            native_scheduler,
            NUM_INFERENCE_STEPS,
            torch.device("cuda"),
        )
        isolated_retrieve_timesteps, isolated_num_steps = native_retrieve_timesteps(
            isolated_scheduler,
            NUM_INFERENCE_STEPS,
            torch.device("cuda"),
        )
        assert reference_num_steps == native_num_steps == isolated_num_steps
        _assert_stage(
            "timesteps",
            native_timesteps,
            reference_timesteps,
            max_abs=0.0,
            relative_l2=0.0,
            cosine=0.999999,
        )
        _assert_stage(
            "sigmas",
            native_scheduler.sigmas,
            reference_scheduler.sigmas,
            max_abs=0.0,
            relative_l2=0.0,
            cosine=0.999999,
        )
        torch.testing.assert_close(isolated_retrieve_timesteps, reference_timesteps, rtol=0, atol=0)

        native_transformer = SanaVideoTransformer3DModel.from_config(dict(reference.transformer.config)).to(
            device="cuda",
            dtype=torch.bfloat16,
        )
        incompatible = native_transformer.load_state_dict(reference.transformer.state_dict(), strict=True)
        assert not incompatible.missing_keys
        assert not incompatible.unexpected_keys
        assert type(native_transformer) is SanaVideoTransformer3DModel
        assert type(native_transformer).__module__.startswith("vllm_omni.")
        assert type(native_transformer) is not type(reference.transformer)
        reference_parameter = next(reference.transformer.parameters())
        native_parameter = next(native_transformer.parameters())
        assert native_parameter.data_ptr() != reference_parameter.data_ptr()
        torch.testing.assert_close(native_parameter, reference_parameter, rtol=0, atol=0)
        native_transformer.eval()
        reference.transformer.eval()

        latent_model_input = torch.cat([reference_latents, reference_latents])
        timestep = reference_timesteps[0].expand(latent_model_input.shape[0])
        with torch.inference_mode():
            reference_raw_prediction = reference.transformer(
                latent_model_input.to(dtype=torch.bfloat16),
                encoder_hidden_states=encoder_hidden_states.to(dtype=torch.bfloat16),
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                return_dict=False,
            )[0]
            native_raw_prediction = native_transformer(
                latent_model_input.to(dtype=torch.bfloat16),
                encoder_hidden_states=encoder_hidden_states.to(dtype=torch.bfloat16),
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                return_dict=False,
            )[0]

        _assert_stage(
            "first_transformer_raw_prediction",
            native_raw_prediction,
            reference_raw_prediction,
            max_abs=config["max_abs"],
            relative_l2=config["relative_l2"],
            cosine=config["cosine"],
        )
        reference_cfg_prediction = _cfg_prediction(
            reference_raw_prediction,
            guidance_scale=GUIDANCE_SCALE,
            latent_channels=latent_channels,
            out_channels=reference.transformer.config.out_channels,
        )
        native_cfg_prediction = _cfg_prediction(
            native_raw_prediction,
            guidance_scale=GUIDANCE_SCALE,
            latent_channels=latent_channels,
            out_channels=native_transformer.config.out_channels,
        )
        _assert_stage(
            "first_cfg_prediction",
            native_cfg_prediction,
            reference_cfg_prediction,
            max_abs=config["cfg_max_abs"],
            relative_l2=config["cfg_relative_l2"],
            cosine=config["cfg_cosine"],
        )

        reference_first_step = reference_scheduler.step(
            reference_cfg_prediction,
            reference_timesteps[0],
            reference_latents,
            return_dict=False,
        )[0]
        # Isolate scheduler orchestration from native-attention error by
        # feeding the Diffusers prediction through an independently configured
        # scheduler initialized by the native helper.
        isolated_first_step = isolated_scheduler.step(
            reference_cfg_prediction,
            isolated_retrieve_timesteps[0],
            reference_latents,
            return_dict=False,
        )[0]
        _assert_stage(
            "first_scheduler_step_isolated",
            isolated_first_step,
            reference_first_step,
            max_abs=0.0,
            relative_l2=0.0,
            cosine=0.999999,
        )

        native_first_step = native_scheduler.step(
            native_cfg_prediction,
            native_timesteps[0],
            native_latents,
            return_dict=False,
        )[0]
        _assert_stage(
            "first_scheduler_step_end_to_end",
            native_first_step,
            reference_first_step,
            max_abs=config["step_max_abs"],
            relative_l2=config["step_relative_l2"],
            cosine=config["step_cosine"],
        )
    finally:
        del native_proxy
        del reference
        gc.collect()
        torch.accelerator.empty_cache()

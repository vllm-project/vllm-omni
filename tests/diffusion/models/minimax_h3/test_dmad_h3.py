# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DMAD artifact identity, CPU noise order, and full/step execution parity."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3 import lora as lora_module
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
from vllm_omni.diffusion.models.minimax_h3.dmad import DmadNoise, DmadSpec, dmad_sigmas
from vllm_omni.diffusion.models.minimax_h3.lora import load_dmad_h3_lora
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _STEP_AUDIO_ROWS, MiniMaxH3Pipeline
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

from .test_minimax_h3_lora import _TINY_TARGET_DIMS, _write_tiny_turbo
from .test_minimax_h3_step_execution import _make_branch, _make_state, _SegmentMeanModel, _step_pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _metadata(steps):
    return {
        "model": f"DMAD {steps}-step student of MiniMax-H3 (text-to-audio-video), LoRA on the T2VA transformer",
        "sampling": f"{steps} steps, video shift 12, audio shift 2, re-noise step rule, 1344x768, 124 frames at 24 fps, no CFG",
        "lora_rank": "128",
        "lora_alpha": "128",
        "lora_scale": "1.0",
    }


def _artifact(tmp_path, steps=4, metadata=None, **kwargs):
    path = tmp_path / f"dmad_minimax_h3_{steps}step_full_critic.safetensors"
    _write_tiny_turbo(path, **kwargs)
    state = {
        (
            key.replace(".default.weight", ".weight")
            if ".ff." in key
            else key.replace(".lora_A.default.weight", ".lora.down.weight").replace(
                ".lora_B.default.weight", ".lora.up.weight"
            )
        ): value
        for key, value in load_file(path).items()
    }
    save_file(state, path, metadata=_metadata(steps) if metadata is None else metadata)
    return path


def _load(path, **kwargs):
    return load_dmad_h3_lora(
        lora_request=LoRARequest("dmad", 8, str(path)),
        lora_path=path,
        **{"partition": "fl2va", "dtype": torch.bfloat16, **kwargs},
    )


@pytest.mark.parametrize("steps", [4, 8])
def test_dmad_metadata_selects_steps_and_complete_packed_factors(tmp_path, monkeypatch, steps):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)
    path = _artifact(tmp_path, steps)
    renamed = path.rename(tmp_path / "renamed.safetensors")
    model, peft, spec = _load(renamed)
    assert spec.denoise_steps == steps
    assert len(model.loras) == 312
    assert peft.r == peft.lora_alpha == 128
    fc1 = model.get_lora("blocks.0.mlp.fc1")
    assert fc1.scaling == [1.0, 1.0]
    torch.testing.assert_close(fc1.lora_b[0], torch.full((2, 128), 2, dtype=torch.bfloat16))
    torch.testing.assert_close(fc1.lora_b[1], torch.ones(2, 128, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "field,value", [("lora_alpha", "8"), ("lora_rank", "64"), ("sampling", "Euler"), ("steps", "8")]
)
def test_dmad_rejects_changed_release_metadata(tmp_path, field, value):
    metadata = {**_metadata(4), field: value}
    with pytest.raises(ValueError, match="DMAD"):
        _load(_artifact(tmp_path, metadata=metadata))


def test_dmad_rejects_partial_binding_wrong_partition_and_ambiguous_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)
    path = _artifact(tmp_path, omit_target="transformer_blocks.49.attn.to_q")
    with pytest.raises(ValueError, match="target set"):
        _load(path)
    with pytest.raises(ValueError, match="FL2VA/T2VA"):
        _load(path, partition="ref2va")
    with pytest.raises(ValueError, match="BF16"):
        _load(path, dtype=torch.float32)
    _artifact(tmp_path, steps=8)
    with pytest.raises(ValueError, match="one DMAD adapter"):
        _load(tmp_path)


def test_dmad_rejects_duplicate_aliases_for_one_factor(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)
    path = _artifact(tmp_path)
    state = load_file(path)
    target = "transformer_blocks.0.attn.to_q"
    state[target + ".lora_A.weight"] = state[target + ".lora.down.weight"].clone()
    save_file(state, path, metadata=_metadata(4))
    with pytest.raises(ValueError, match="Duplicate DMAD"):
        _load(path)


@pytest.mark.parametrize("steps", [4, 8])
@pytest.mark.parametrize(
    "override",
    [
        {"num_inference_steps": 6},
        {"lora_scale": 0.5},
        {"extra_args": {"audio_flow_shift": 3}},
        {"extra_args": {"flow_shift": 6}},
        {"extra_args": {"sampler": "res_multistep"}},
        {"extra_args": {"continue_from": "clip"}},
        {"extra_args": {"audio_mode": "lock_source"}},
    ],
)
def test_dmad_rejects_incompatible_sampling(steps, override):
    with pytest.raises(ValueError, match="DMAD"):
        DmadSpec(steps).check_request(OmniDiffusionSamplingParams(**override), "t2va")


def _noise(seed):
    return DmadNoise(seed=seed, device=torch.device("cpu"), latent_t=2, latent_h=4, latent_w=4, audio_t=3)


@pytest.mark.parametrize("steps", [4, 8])
def test_dmad_cpu_stream_matches_release_draw_order_and_is_request_owned(steps):
    noise = _noise(42)
    before = torch.random.get_rng_state().clone()
    generator = torch.Generator().manual_seed(42)
    expected_video = torch.randn((1, 24, 2, 4, 4), generator=generator)
    expected_audio = torch.randn((6, 32), generator=generator)
    video, audio = noise.initial_rows()
    torch.testing.assert_close(
        video, minimax_h3_patchify_video_latent(expected_video, patch_size=(1, 2, 2)), rtol=0, atol=0
    )
    torch.testing.assert_close(audio, expected_audio, rtol=0, atol=0)
    for sv, sa in zip(dmad_sigmas(steps, 12)[1:], dmad_sigmas(steps, 2)[1:]):
        expected_video = torch.randn(video.shape, generator=generator)
        expected_audio = torch.randn(audio.shape, generator=generator)
        for rows, sigma, expected, modality in (
            (video, sv, expected_video, "video"),
            (audio, sa, expected_audio, "audio"),
        ):
            s = torch.tensor(sigma)
            torch.testing.assert_close(
                noise.step(rows, sigma, modality=modality), (1 - s) * rows + s * expected, rtol=0, atol=0
            )
    assert torch.equal(noise.generator.get_state(), generator.get_state())
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("steps", [4, 8])
def test_dmad_full_and_interleaved_step_execution_match(steps):
    model = _SegmentMeanModel()
    pipe = _step_pipeline(model)
    states, expected = [], []
    for seed in (42, 43):
        branch, _, _ = _make_branch(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3, seed=seed)
        noise = _noise(seed)
        video, audio = noise.initial_rows()
        sv, sa = dmad_sigmas(steps, 12), dmad_sigmas(steps, 2)
        expected.append(
            minimax_h3_denoise_loop(
                model=model,
                positive=branch,
                initial_video_rows=video,
                initial_audio_rows=audio,
                keyframe_cond_rows=None,
                sigmas_video=sv,
                sigmas_audio=sa,
                device=torch.device("cpu"),
                renoise_noise=noise,
            )
        )
        noise = _noise(seed)
        video, audio = noise.initial_rows()
        state = _make_state(str(seed), model, branch, video, audio, sv, sa)
        state.extra["renoise_noise"] = noise
        states.append(state)
    for _ in range(steps):
        for state in states:
            velocity = pipe.denoise_step(None, states=[state])
            pipe.step_scheduler(state, velocity)
    for state, pair in zip(states, expected):
        assert state.denoise_completed and state.step_index == steps
        for actual, target in zip((state.latents, state.extra[_STEP_AUDIO_ROWS]), pair):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [4, 8])
def test_dmad_schedule_selection_and_scale_zero_restore_h3(steps):
    pipe = object.__new__(MiniMaxH3Pipeline)
    pipe.od_config = OmniDiffusionConfig()
    pipe._dmad_lora_specs = {8: DmadSpec(steps)}
    pipe._turbo_lora_specs = {}
    pipe._lora_sigma_schedules = {}
    pipe._native_lora_adapter_ids = set()
    sampling = OmniDiffusionSamplingParams(lora_request=LoRARequest("dmad", 8, "/unused"), num_inference_steps=steps)
    assert pipe._resolve_sigma_positions("t2va", sampling) == (DmadSpec(steps).base_schedule, steps)
    with pytest.raises(ValueError, match="t2va only"):
        DmadSpec(steps).check_request(sampling, "ref2va")
    sampling.lora_scale = 0
    assert pipe._active_renoise_spec(sampling) is None


@pytest.mark.parametrize("steps", [4, 8])
def test_dmad_preparation_uses_audio_shift_two_and_release_cpu_noise(steps):
    pipe = _step_pipeline(_SegmentMeanModel())
    spec = DmadSpec(steps)
    inputs = pipe._build_denoise_inputs(
        task="t2va",
        text_embeddings=torch.ones(3, 8),
        text_tags=torch.ones(3, dtype=torch.long),
        seed=42,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        num_frames=22,
        num_steps=steps,
        video_shift=12,
        audio_shift=3,
        base_schedule=spec.base_schedule,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
        renoise_spec=spec,
        precomputed_initial_noise=(torch.zeros(8, 96), torch.zeros(6, 32)),
    )
    video, audio = _noise(42).initial_rows()
    torch.testing.assert_close(inputs["video_rows"], video, rtol=0, atol=0)
    torch.testing.assert_close(inputs["audio_rows"], audio, rtol=0, atol=0)
    assert inputs["sigmas_video"] == dmad_sigmas(steps, 12)
    assert inputs["sigmas_audio"] == dmad_sigmas(steps, 2)

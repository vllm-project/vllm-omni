# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pika artifact validation, reference noise semantics and both execution modes."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3 import lora as lora_module
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop
from vllm_omni.diffusion.models.minimax_h3.lora import load_pika_h3_lora
from vllm_omni.diffusion.models.minimax_h3.pika import (
    PIKA_BASE_REVISION,
    PIKA_KNOTS,
    PIKA_TARGET_MODULES,
    PikaNoise,
    PikaSpec,
    pika_sigmas,
    read_pika_spec,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

from .test_minimax_h3_lora import _TINY_TARGET_DIMS, _write_tiny_turbo
from .test_minimax_h3_step_execution import _make_branch, _make_state, _SegmentMeanModel, _step_pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _manifest(task="fl2va", digest="0" * 64):
    return {
        "model": "pika-flex",
        "format_version": 1,
        "release_id": f"pika-flex-{task}-v1",
        "task": task,
        "rank": 128,
        "alpha": 128,
        "trunk": "transformer_ref" if task == "ref2va" else "transformer",
        "numerics_policy": "fp32_conditioning_v1" if task == "ref2va" else "bf16_conditioning_v1",
        "base": {"repo_id": "MiniMaxAI/MiniMax-H3", "revision": PIKA_BASE_REVISION},
        "sampling": {"knots": list(PIKA_KNOTS), "video_shift": 12, "audio_shift": 3, "method": "renoise"},
        "target_modules": list(PIKA_TARGET_MODULES),
        "sha256": digest,
    }


def _artifact(tmp_path, task="fl2va", **kwargs):
    weights = tmp_path / "adapter.safetensors"
    _write_tiny_turbo(weights, **kwargs)
    manifest = _manifest(task, hashlib.sha256(weights.read_bytes()).hexdigest())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return weights


def _load(path, partition="fl2va"):
    return load_pika_h3_lora(
        partition=partition,
        lora_request=LoRARequest("pika", 7, str(path)),
        lora_path=path,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_pika_loads_all_factors_and_preserves_alpha_and_gate_order(tmp_path, monkeypatch, task):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)
    path = _artifact(tmp_path, task)
    model, peft, spec = _load(path, task)
    assert len(model.loras) == 312
    assert peft.r == peft.lora_alpha == 128
    assert spec.task == task
    fc1 = model.get_lora("blocks.0.mlp.fc1")
    assert all(x.dtype == torch.bfloat16 for x in fc1.lora_a + fc1.lora_b)
    assert all(x == 1.0 for x in fc1.scaling)
    torch.testing.assert_close(fc1.lora_b[0], torch.full((2, 128), 2, dtype=torch.bfloat16))
    torch.testing.assert_close(fc1.lora_b[1], torch.ones(2, 128, dtype=torch.bfloat16))
    assert read_pika_spec(tmp_path) == read_pika_spec(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", 64),
        ("alpha", 8),
        ("trunk", "transformer_ref"),
        ("numerics_policy", "fp32_conditioning_v1"),
        ("base", {}),
        ("sampling", {"method": "euler"}),
        ("target_modules", []),
        ("sha256", "invalid"),
    ],
)
def test_pika_rejects_modified_contract(tmp_path, field, value):
    manifest = _manifest()
    manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_pika_spec(tmp_path)


def test_pika_rejects_bad_hash_wrong_partition_and_missing_factors(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_module, "_TURBO_TARGET_DIMS", _TINY_TARGET_DIMS)
    path = _artifact(tmp_path, omit_target="transformer_blocks.49.attn.to_q")
    with pytest.raises(ValueError, match="target set"):
        _load(path)
    with pytest.raises(ValueError, match="dedicated"):
        _load(path, "ref2va")
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        _load(path)


def test_pika_does_not_claim_unrelated_adapter_or_allow_ambiguous_root(tmp_path):
    save_file({"x": torch.ones(1)}, str(tmp_path / "adapter.safetensors"))
    assert _load(tmp_path) is None
    (tmp_path / "fl2va").mkdir()
    (tmp_path / "fl2va" / "manifest.json").write_text(json.dumps(_manifest()))
    with pytest.raises(ValueError, match="directory"):
        read_pika_spec(tmp_path)


def _sampling(**overrides):
    return OmniDiffusionSamplingParams(
        lora_request=LoRARequest("pika", 7, "/unused/pika"),
        **{
            "lora_scale": 1.0,
            "num_inference_steps": 6,
            "extra_args": {},
            **overrides,
        },
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_inference_steps": 7},
        {"num_inference_steps": 4},
        {"lora_scale": 0.5},
        {"extra_args": {"flow_shift": 6}},
        {"extra_args": {"audio_flow_shift": float("nan")}},
        {"extra_args": {"long_video_mode": "continuation"}},
        {"extra_args": {"audio_mode": "lock_source"}},
        {"extra_args": {"sampler": "res_multistep"}},
        {"extra_args": {"continue_from": "clip-123"}},
    ],
)
def test_pika_request_cannot_silently_change_sampling(overrides):
    with pytest.raises(ValueError):
        PikaSpec("fl2va", "0" * 64).check_request(_sampling(**overrides), "t2va")


def test_pika_active_schedule_and_scale_zero_restore_base():
    pipe = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    pipe.od_config = OmniDiffusionConfig(step_execution=True)
    pipe._pika_lora_specs = {7: PikaSpec("fl2va", "0" * 64)}
    pipe._fasth3 = None
    pipe._sigma_schedule_for_request = lambda *args: None
    positions, steps = pipe._resolve_sigma_positions("t2va", _sampling())
    assert steps == 6 and positions == (1, 0.75, 0.5, 0.25, 0.1, 0.04, 0)
    assert pipe._resolve_sigma_positions("t2va", _sampling(lora_scale=0, num_inference_steps=50)) == (None, 50)
    with pytest.raises(ValueError, match="explicit"):
        PikaSpec("fl2va", "0" * 64).check_request(_sampling(num_inference_steps=None), "t2va", step_execution=True)


def _noise(seed=42):
    return PikaNoise(seed=seed, device=torch.device("cpu"), latent_t=2, latent_h=4, latent_w=6, audio_t=3)


def _video_rows(external):
    # Independent explicit packing of [1,T,24,H,W] to [T*H/2*W/2,96].
    _, t, c, h, w = external.shape
    return external.reshape(t, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(-1, 96)


def _audio_rows(external):
    return torch.cat((external[0, :, :32], external[0, :, 32:]), dim=0)


def test_pika_noise_matches_reference_external_layout_and_isolates_rng():
    global_state = torch.random.get_rng_state().clone()
    noise = _noise()
    reference = torch.Generator().manual_seed(42)
    video = torch.randn(1, 2, 24, 4, 6, generator=reference)
    audio = torch.randn(1, 3, 64, generator=reference)
    torch.testing.assert_close(noise.video(noise.initial), _video_rows(video), rtol=0, atol=0)
    torch.testing.assert_close(noise.audio(noise.initial), _audio_rows(audio), rtol=0, atol=0)
    reference.manual_seed(43)
    video_noise = _video_rows(torch.randn(1, 2, 24, 4, 6, generator=reference))
    audio_noise = _audio_rows(torch.randn(1, 3, 64, generator=reference))
    sv, sa = torch.tensor(pika_sigmas(12)[1]), torch.tensor(pika_sigmas(3)[1])
    torch.testing.assert_close(
        noise.step(torch.ones_like(video_noise), sv.item(), modality="video"),
        (1 - sv) + sv * video_noise,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        noise.step(torch.ones_like(audio_noise), sa.item(), modality="audio"),
        (1 - sa) + sa * audio_noise,
        rtol=0,
        atol=0,
    )
    before = noise.renoise.get_state().clone()
    clean = torch.ones_like(video_noise)
    assert noise.step(clean, 0, modality="video") is clean
    assert torch.equal(before, noise.renoise.get_state())
    assert torch.equal(global_state, torch.random.get_rng_state())


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_pika_condition_precision_and_fixed_augmentation(task):
    spec = PikaSpec(task, "0" * 64)
    rows = torch.linspace(-1.001, 1.001, 6 * 96).reshape(6, 96)
    noise = _noise()
    actual = noise.visual_anchor(rows, [(1, 4, 6)], spec.conditioning_dtype)
    generator = torch.Generator().manual_seed(42 + 7919)
    eps = _video_rows(torch.randn(1, 1, 24, 4, 6, generator=generator))
    expected = 0.999 * rows.to(spec.conditioning_dtype).float() + (1 - 0.999) * eps
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pika_full_loop_and_interleaved_step_execution_match():
    model = _SegmentMeanModel()
    pipe = _step_pipeline(model)
    states, expected = [], []
    for seed in (7, 99):
        branch, video, audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=seed)
        expected.append(
            minimax_h3_denoise_loop(
                model=model,
                positive=branch,
                initial_video_rows=video,
                initial_audio_rows=audio,
                keyframe_cond_rows=None,
                sigmas_video=pika_sigmas(12),
                sigmas_audio=pika_sigmas(3),
                device=torch.device("cpu"),
                renoise_noise=_noise(seed),
            )
        )
        state = _make_state(str(seed), model, branch, video, audio, pika_sigmas(12), pika_sigmas(3))
        state.extra["renoise_noise"] = _noise(seed)
        states.append(state)
    assert len(model.calls) == 12
    for _ in range(6):
        for state in reversed(states):
            velocity = pipe.denoise_step(None, states=[state])
            pipe.step_scheduler(state, velocity)
    for state, (video, audio) in zip(states, expected, strict=True):
        assert state.denoise_completed and state.step_index == 6
        torch.testing.assert_close(state.latents, video, rtol=0, atol=0)
        torch.testing.assert_close(state.extra[pipeline_module._STEP_AUDIO_ROWS], audio, rtol=0, atol=0)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_pika_request_preparation_uses_release_noise_and_pins_anchors(task):
    model = _SegmentMeanModel()
    pipe = _step_pipeline(model)
    spec = PikaSpec(task, "0" * 64)
    clean = torch.linspace(-1.001, 1.001, 6 * 96).reshape(6, 96)
    inputs = pipe._build_denoise_inputs(
        task=task,
        text_embeddings=torch.ones(3, 8),
        text_tags=torch.ones(3, dtype=torch.long),
        seed=42,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        num_frames=22,
        num_steps=6,
        video_shift=12,
        audio_shift=3,
        base_schedule=spec.base_schedule,
        visual_condition=clean,
        visual_condition_shape=(1, 4, 6),
        audio_condition=None,
        ref_audio_t=None,
        keyframe_frame_indices=[0],
        ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 6}] if task == "ref2va" else None,
        renoise_spec=spec,
        # The base runtime can prepare CPU noise ahead of encoding. Pika must
        # retain its own device/layout/RNG convention even if these are present.
        precomputed_initial_noise=(torch.zeros(12, 96), torch.zeros(6, 32)),
        precomputed_visual_condition_noise=torch.zeros_like(clean),
    )
    branch = inputs["branch"]
    expected = _noise()
    torch.testing.assert_close(
        inputs["video_rows"][branch.update_mask], expected.video(expected.initial), rtol=0, atol=0
    )
    torch.testing.assert_close(inputs["audio_rows"], expected.audio(expected.initial), rtol=0, atol=0)
    anchor = expected.visual_anchor(clean, [(1, 4, 6)], spec.conditioning_dtype)
    torch.testing.assert_close(inputs["cond_anchor"], anchor, rtol=0, atol=0)
    assert inputs["sigmas_video"] == pika_sigmas(12)
    assert inputs["sigmas_audio"] == pika_sigmas(3)
    video, _ = minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=inputs["video_rows"],
        initial_audio_rows=inputs["audio_rows"],
        keyframe_cond_rows=inputs["cond_anchor"],
        sigmas_video=inputs["sigmas_video"],
        sigmas_audio=inputs["sigmas_audio"],
        device=torch.device("cpu"),
        renoise_noise=inputs["renoise_noise"],
    )
    assert len(model.calls) == 6
    torch.testing.assert_close(video[~branch.update_mask], anchor, rtol=0, atol=0)
    for call in model.calls:
        torch.testing.assert_close(call["video_rows"][~branch.update_mask], anchor, rtol=0, atol=0)


def test_pika_cobatching_preserves_independent_random_streams():
    model = _SegmentMeanModel()
    pipe = _step_pipeline(model)
    states, expected = [], []
    for seed in (7, 99):
        branch, video, audio = _make_branch(text_len=9, latent_t=2, latent_h=4, latent_w=6, audio_t=3, seed=seed)
        expected.append(
            minimax_h3_denoise_loop(
                model=model,
                positive=branch,
                initial_video_rows=video,
                initial_audio_rows=audio,
                keyframe_cond_rows=None,
                sigmas_video=pika_sigmas(12),
                sigmas_audio=pika_sigmas(3),
                device=torch.device("cpu"),
                renoise_noise=_noise(seed),
            )
        )
        state = _make_state(str(seed), model, branch, video, audio, pika_sigmas(12), pika_sigmas(3))
        state.extra["renoise_noise"] = _noise(seed)
        states.append(state)
    for _ in range(6):
        velocity = pipe.denoise_step(None, states=states)
        offset = 0
        for state in states:
            count = state.latents.shape[0]
            pipe.step_scheduler(state, velocity[offset : offset + count])
            offset += count
    for state, (video, audio) in zip(states, expected, strict=True):
        torch.testing.assert_close(state.latents, video, rtol=0, atol=0)
        torch.testing.assert_close(state.extra[pipeline_module._STEP_AUDIO_ROWS], audio, rtol=0, atol=0)

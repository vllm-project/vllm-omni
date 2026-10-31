# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Open-H3-Regenerate: bundle loading, LoRA fusion and the refine contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3 import regenerate
from vllm_omni.diffusion.models.minimax_h3.regenerate import (
    REGENERATE_DEFAULT_UPSCALE,
    MiniMaxH3RegenerateAdapter,
    MiniMaxH3RegenerateError,
    regenerate_schedule,
)
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HEAD_DIM, HEADS, HIDDEN, FFN, RANK = 4, 2, 8, 16, 3
NUM_BLOCKS, NUM_REFINER_BLOCKS = 2, 1
ADALN_OUT = 18 * HIDDEN
# Diffusers-namespace target -> (out, in) of the base projection.
_TARGETS = {
    "attn.to_q": (HEADS * HEAD_DIM, HIDDEN),
    "attn.to_k": (HEADS * HEAD_DIM, HIDDEN),
    "attn.to_v": (HEADS * HEAD_DIM, HIDDEN),
    "attn.to_out.0": (HIDDEN, HEADS * HEAD_DIM),
    "ff.net.0.proj": (2 * FFN, HIDDEN),
    "ff.net.2": (HIDDEN, FFN),
}


def _write_bundle(root, *, blocks: int = NUM_BLOCKS, active_curve: int | None = None, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    lora = {}
    prefixes = [f"transformer.transformer_blocks.{i}." for i in range(blocks)]
    prefixes += [f"transformer.token_refiner.refiner_blocks.{i}." for i in range(NUM_REFINER_BLOCKS)]
    for prefix in prefixes:
        for target, (out_features, in_features) in _TARGETS.items():
            lora[f"{prefix}{target}.lora_A.weight"] = torch.randn(RANK, in_features, generator=generator)
            lora[f"{prefix}{target}.lora_B.weight"] = torch.randn(out_features, RANK, generator=generator)
    curves = {}
    for i in range(blocks):
        stem = f"transformer.transformer_blocks.{i}.adaln_proj.linear"
        curves[f"{stem}.curve_A.weight"] = torch.randn(2, 8, generator=generator)
        # The release's up factors are all zero, which makes every branch inert.
        up = torch.randn(ADALN_OUT, 2, generator=generator) if i == active_curve else torch.zeros(ADALN_OUT, 2)
        curves[f"{stem}.curve_B.weight"] = up
    (root / "lora").mkdir(parents=True)
    (root / "latent_upsampler").mkdir()
    save_file(lora, str(root / regenerate.REGENERATE_LORA_FILE), metadata={"format": "lora"})
    save_file(curves, str(root / regenerate.REGENERATE_CURVE_FILE), metadata={"format": "lms-sharp-curve-adaln"})
    (root / regenerate.REGENERATE_LORA_CONFIG_FILE).write_text(json.dumps({"base_model": "MiniMaxAI/MiniMax-H3@x"}))
    (root / regenerate.REGENERATE_UPSCALER_FILE).write_bytes(b"")
    return lora


def _load(root):
    return MiniMaxH3RegenerateAdapter.from_path(
        root, head_dim=HEAD_DIM, num_blocks=NUM_BLOCKS, num_refiner_blocks=NUM_REFINER_BLOCKS
    )


def test_schedule_reproduces_the_release_grid():
    for steps, expected in ((3, [0.909375, 6 / 7, 2 / 3, 0.0]), (2, [0.7, 0.4941176470588235, 0.0])):
        sigmas = regenerate_schedule(steps).shifted_sigmas(regenerate.REGENERATE_VIDEO_SHIFT)
        assert sigmas == pytest.approx(expected, abs=1e-12)
    with pytest.raises(OmniClientError, match="num_inference_steps=4"):
        regenerate_schedule(4)


def test_adapter_fuses_every_lora_into_the_native_layout(tmp_path):
    lora = _write_bundle(tmp_path)
    adapter = _load(tmp_path)
    assert adapter is not None

    def delta(prefix, target):
        return lora[f"{prefix}{target}.lora_B.weight"] @ lora[f"{prefix}{target}.lora_A.weight"]

    native = {}
    for block, prefix in [(f"blocks.{i}.", f"transformer.transformer_blocks.{i}.") for i in range(NUM_BLOCKS)] + [
        (f"token_refiner.blocks.{i}.", f"transformer.token_refiner.refiner_blocks.{i}.")
        for i in range(NUM_REFINER_BLOCKS)
    ]:
        # H3 stores QKV one head group at a time, and fc1 gate-first where the
        # Diffusers export is value-first.
        q, k, v = (delta(prefix, f"attn.to_{x}").reshape(HEADS, HEAD_DIM, HIDDEN) for x in "qkv")
        native[f"{block}attn.qkv_proj.weight"] = torch.stack((q, k, v), dim=1).reshape(-1, HIDDEN)
        native[f"{block}attn.out_proj.weight"] = delta(prefix, "attn.to_out.0")
        value, gate = delta(prefix, "ff.net.0.proj").chunk(2)
        native[f"{block}mlp.fc1.weight"] = torch.cat((gate, value))
        native[f"{block}mlp.fc2.weight"] = delta(prefix, "ff.net.2")

    base = {name: torch.full_like(expected, 0.5, dtype=torch.bfloat16) for name, expected in native.items()}
    fused = dict(adapter.apply(iter([*base.items(), ("blocks.0.norm1.weight", torch.ones(HIDDEN))])))
    for name, expected in native.items():
        torch.testing.assert_close(fused[name].cpu(), (expected + 0.5).to(torch.bfloat16))
    torch.testing.assert_close(fused["blocks.0.norm1.weight"], torch.ones(HIDDEN))
    adapter.validate_fully_applied(set(fused))


def test_adapter_claims_only_release_bundles(tmp_path):
    assert _load(tmp_path) is None
    _write_bundle(tmp_path / "bundle")
    (tmp_path / "bundle" / regenerate.REGENERATE_CURVE_FILE).unlink()
    assert _load(tmp_path / "bundle") is None


def test_adapter_refuses_sharp_curve_branches_that_contribute(tmp_path):
    _write_bundle(tmp_path, active_curve=1)
    with pytest.raises(MiniMaxH3RegenerateError, match=r"nonzero Sharp curve branches in blocks \[1\]"):
        _load(tmp_path)


def test_adapter_rejects_a_bundle_for_another_depth(tmp_path):
    _write_bundle(tmp_path, blocks=NUM_BLOCKS - 1)
    with pytest.raises(MiniMaxH3RegenerateError, match="1 of the model's 2 blocks"):
        _load(tmp_path)


def test_request_is_held_to_the_release_sampler(tmp_path):
    _write_bundle(tmp_path)
    adapter = _load(tmp_path)

    def sampling(**extra):
        steps = extra.pop("steps", None)
        lora = extra.pop("lora", None)
        return SimpleNamespace(lora_request=lora, num_inference_steps=steps, extra_args=extra)

    assert adapter.check_request(sampling()) == 3
    assert adapter.check_request(sampling(steps=2, flow_shift=6, sampler="euler")) == 2
    for request, message in (
        (sampling(lora=object()), "drop the lora field"),
        (sampling(steps=4), "num_inference_steps=4"),
        (sampling(flow_shift=12), "flow_shift=6"),
        (sampling(latent_refine=0.5), "latent_refine"),
        (sampling(sampler="res_multistep"), "sampler"),
    ):
        with pytest.raises(OmniClientError, match=message):
            adapter.check_request(request)


def test_adapter_refuses_dit_offload_and_other_partitions(tmp_path, monkeypatch):
    from vllm_omni.diffusion.offloader.config import OffloadStrategy

    _write_bundle(tmp_path)
    adapter = _load(tmp_path)

    def offload(strategy, *components):
        return SimpleNamespace(strategy=strategy, offloads=lambda component: component in components)

    monkeypatch.setattr(regenerate, "resolve_offload", lambda config: config)
    adapter.check_serving_contract(partition="fl2va", od_config=offload(OffloadStrategy.NONE))
    adapter.check_serving_contract(partition="fl2va", od_config=offload(OffloadStrategy.LAYER_WISE, "text_encoder"))
    with pytest.raises(ValueError, match="cannot serve 'ref2va'"):
        adapter.check_serving_contract(partition="ref2va", od_config=offload(OffloadStrategy.NONE))
    with pytest.raises(ValueError, match="DiT cannot use"):
        adapter.check_serving_contract(partition="fl2va", od_config=offload(OffloadStrategy.LAYER_WISE, "dit"))


def test_prepare_edit_video_cover_crops_instead_of_stretching(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.minimax_h3 import reference_video

    commands: list[list[str]] = []
    monkeypatch.setattr(reference_video, "_probe_video", lambda _path: {"frame_count": 121, "audio_codecs": ()})
    monkeypatch.setattr(reference_video.subprocess, "run", lambda command, **_kwargs: commands.append(command))
    for cover in (False, True):
        reference_video.prepare_edit_video(
            tmp_path / "draft.mp4",
            target_width=1344,
            target_height=768,
            target_frame_count=124,
            workdir=str(tmp_path / "prepared"),
            cover=cover,
        )
    stretched, covered = (command[command.index("-vf") + 1] for command in commands)
    assert "scale=1344:768:flags=lanczos" in stretched and "crop" not in stretched
    assert "scale=1344:768:force_original_aspect_ratio=increase:flags=bilinear,crop=1344:768" in covered
    assert covered.endswith("tpad=stop_mode=clone:stop=-1")


@pytest.mark.parametrize("draft_has_audio", [False, True])
def test_encoder_routes_the_draft_as_an_all_keep_edit_source(monkeypatch, tmp_path, draft_has_audio):
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing

    latent_t, audio_t = 7, 37
    calls: dict = {}

    def prepare_video(value, **kwargs):
        calls["prepare"] = kwargs
        return {"prepared_path": str(tmp_path / "prepared.mp4"), "input_has_audio": draft_has_audio}

    def load_video_audio(value, **kwargs):
        calls["audio"] = value
        return torch.zeros(3_200), 32_000

    monkeypatch.setattr(processing, "resolve_minimax_h3_shape", lambda *_args: (64, 96, 22, latent_t, audio_t))
    monkeypatch.setattr(processing, "prepare_edit_video", prepare_video)
    monkeypatch.setattr(
        processing, "load_video_frames", lambda _path: torch.zeros(22, 64, 96, 3, dtype=torch.uint8).numpy()
    )
    monkeypatch.setattr(processing, "load_video_audio", load_video_audio)

    prepared = processing.prepare_encoder_inputs(
        {"prompt": "regenerate", "multi_modal_data": {"source_video": "draft.mp4"}},
        SimpleNamespace(extra_args={"regenerate": True}),
    )
    assert calls["prepare"]["cover"] is True
    assert prepared.media.video_edit is not None
    torch.testing.assert_close(prepared.media.video_edit_mask, torch.zeros(latent_t, 4, 6))
    if draft_has_audio:
        assert calls["audio"] == "draft.mp4"
        torch.testing.assert_close(prepared.media.audio_edit_mask, torch.zeros(2, audio_t))
    else:
        assert "audio" not in calls and prepared.media.audio_edit is None

    for mm, extra, message in (
        ({}, {"regenerate": True}, "one source_video"),
        ({"source_video": "d.mp4", "video_noise_mask": 1.0}, {"regenerate": True}, "noise masks"),
        ({"source_video": "d.mp4"}, {"regenerate": "yes"}, "boolean"),
    ):
        with pytest.raises(OmniClientError, match=message):
            processing.prepare_encoder_inputs(
                {"prompt": "regenerate", "multi_modal_data": mm}, SimpleNamespace(extra_args=extra)
            )


def _regenerate_pipeline(adapter):
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentUpscaleTarget
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.load_vae_encoder = True
    pipeline.od_config = SimpleNamespace(step_execution=False)
    pipeline._regenerate = adapter
    pipeline._quality_policy = SimpleNamespace(resolve=lambda **_kwargs: SimpleNamespace(cache_dit=None))
    pipeline._cache_dit_runtime = SimpleNamespace(prepare=lambda _plan: None)
    pipeline._validate_refine_token_budget = lambda **_kwargs: None
    pipeline._regenerate_silence = ((22, 3), torch.full((2 * 3, 32), -0.25))
    requested = {}

    def resolve_upscale(extra, *, latent_h, latent_w):
        requested["upscale"] = (extra["latent_upscale"], latent_h, latent_w)
        return MiniMaxH3LatentUpscaleTarget(latent_height=8, latent_width=12, scale=2.0)

    pipeline._resolve_latent_upscale = resolve_upscale
    return pipeline, requested


def _draft_conditioning(*, video_mask=None, audio=True):
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderConditioning

    latent_t, height, width, audio_t = 2, 64, 96, 3
    rows = latent_t * (height // 32) * (width // 32)
    return MiniMaxH3EncoderConditioning(
        hidden_states=torch.zeros(5, 4, dtype=torch.bfloat16),
        token_tags=torch.ones(5, dtype=torch.long),
        task="t2va",
        height=height,
        width=width,
        num_frames=22,
        latent_t=latent_t,
        audio_t=audio_t,
        video_edit_clean_rows=torch.linspace(-1, 1, rows * 96).reshape(rows, 96),
        video_edit_mask=torch.zeros(latent_t, 4, 6) if video_mask is None else video_mask,
        audio_edit_clean_rows=torch.full((2 * audio_t, 32), 0.75) if audio else None,
        audio_edit_mask=torch.zeros(2, audio_t) if audio else None,
        audio_edit_source_t=audio_t if audio else 0,
    )


def _sampling(**extra):
    return SimpleNamespace(
        extra_args={"regenerate": True, **extra},
        quality=None,
        seed=17,
        num_outputs_per_prompt=1,
        lora_request=None,
        num_inference_steps=None,
    )


def test_pipeline_turns_the_draft_into_one_refine_pass(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent

    _write_bundle(tmp_path)
    pipeline, requested = _regenerate_pipeline(_load(tmp_path))
    conditioning = _draft_conditioning()
    context = pipeline._prepare_encoder_conditioning_inputs(conditioning, _sampling())

    assert requested["upscale"] == (REGENERATE_DEFAULT_UPSCALE, 4, 6)
    assert context["latent_refine"].strength == 1.0
    assert context["video_shift"] == regenerate.REGENERATE_VIDEO_SHIFT
    assert context["num_steps"] == 3
    assert context["base_schedule"] == regenerate_schedule(3).base_schedule
    video_latent, audio_latent = context["regenerate_source"]
    torch.testing.assert_close(
        minimax_h3_patchify_video_latent(video_latent, patch_size=(1, 2, 2)), conditioning.video_edit_clean_rows
    )
    torch.testing.assert_close(audio_latent, torch.full((2, 32, 3), -0.25))
    torch.testing.assert_close(context["locked_audio_rows"], torch.full((6, 32), -0.25))
    torch.testing.assert_close(context["regenerate_output_audio"], torch.full((2, 32, 3), 0.75))
    assert context["video_edit_clean_rows"] is None and context["audio_edit_clean_rows"] is None

    silent = pipeline._prepare_encoder_conditioning_inputs(_draft_conditioning(audio=False), _sampling())
    assert silent["regenerate_output_audio"] is None


def test_pipeline_refuses_requests_outside_the_regenerate_contract(tmp_path):
    _write_bundle(tmp_path)
    pipeline, _ = _regenerate_pipeline(_load(tmp_path))
    plain = _sampling()
    plain.extra_args = {}
    for conditioning, sampling, message in (
        (_draft_conditioning(), plain, "regenerates drafts only"),
        (_draft_conditioning(video_mask=torch.ones(2, 4, 6)), _sampling(), "no video_noise_mask"),
        (_draft_conditioning(), _sampling(flow_shift=12), "flow_shift=6"),
        # A 2K regenerate leaves the base canvas a cross-request continuation shares.
        (_draft_conditioning(), _sampling(continue_from="video_gen_source"), "continue_from"),
    ):
        with pytest.raises(OmniClientError, match=message):
            pipeline._prepare_encoder_conditioning_inputs(conditioning, sampling)
    pipeline.od_config.controlnet_model_path = "/configured/control.safetensors"
    with pytest.raises(OmniClientError, match="control inputs"):
        pipeline._prepare_encoder_conditioning_inputs(
            _draft_conditioning(), _sampling(canny={}), control_media={"control_path": "/tmp/hint.mp4"}
        )
    pipeline._regenerate = None
    with pytest.raises(OmniClientError, match="--lora-path"):
        pipeline._prepare_encoder_conditioning_inputs(_draft_conditioning(), _sampling())


def test_refine_pass_starts_from_the_upscaled_draft_at_the_release_sigma():
    from vllm_omni.diffusion.models.minimax_h3.latent_upscaler import MiniMaxH3LatentRefineSpec
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    latent_t, latent_h, latent_w, audio_t = 2, 4, 6, 3
    upscaled = torch.randn(1, 24, latent_t, latent_h, latent_w, generator=torch.Generator().manual_seed(1))
    silence = torch.full((2 * audio_t, 32), -0.25)
    inputs = pipeline._build_denoise_inputs(
        task="t2va",
        text_embeddings=torch.zeros(2, 4),
        text_tags=torch.ones(2, dtype=torch.long),
        seed=17,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        num_frames=22,
        num_steps=3,
        video_shift=regenerate.REGENERATE_VIDEO_SHIFT,
        audio_shift=3.0,
        base_schedule=regenerate_schedule(3).base_schedule,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
        locked_audio_rows=silence,
        init_latents=(upscaled, silence.reshape(2, audio_t, 32).permute(0, 2, 1)),
        refine=MiniMaxH3LatentRefineSpec(strength=1.0),
    )
    assert inputs["sigmas_video"] == pytest.approx([0.909375, 6 / 7, 2 / 3, 0.0], abs=1e-12)
    noise, _ = pipeline._initial_noise(seed=17, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, audio_t=3)
    rows = minimax_h3_patchify_video_latent(upscaled, patch_size=(1, 2, 2))
    sigma = inputs["sigmas_video"][0]
    torch.testing.assert_close(inputs["video_rows"], (1 - sigma) * rows + sigma * noise)
    torch.testing.assert_close(inputs["audio_rows"], silence)
    assert inputs["branch"].locked_audio_rows is not None


def test_draft_is_upscaled_once_for_every_output_seed():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.load_vae_decoder = False
    draft, silence = torch.zeros(1, 24, 2, 4, 6), torch.zeros(2, 32, 3)
    target = SimpleNamespace(height=128, width=192)
    pipeline._prepare_request_inputs = lambda _prompt, _sampling: {
        "num_outputs": 2,
        "seed": 5,
        "latent_upscale": target,
        "latent_refine": object(),
        "regenerate_source": (draft, silence),
        "regenerate_output_audio": None,
        "preencode_mp4": False,
    }
    pipeline._denoise_kwargs = lambda _context: {}
    upscaled, refined = [], []

    def upscale(video_latent, requested):
        assert video_latent is draft and requested is target
        upscaled.append(torch.ones(1, 24, 2, 8, 12))
        return upscaled[-1]

    def refine(video_latent, audio_latent, *, context, seed):
        assert video_latent is upscaled[0] and audio_latent is silence
        refined.append(seed)
        return video_latent + seed, audio_latent

    pipeline._upscaled_latent = upscale
    pipeline._refined_latents = refine
    output = pipeline.forward(SimpleNamespace(prompts=["draft"], sampling_params=SimpleNamespace()))

    assert len(upscaled) == 1 and refined == [5, 6]
    videos = output.output["payload"]["trajectory"]["latents"]["video"]
    assert [float(video[0, 0, 0, 0, 0]) for video in videos] == [6.0, 7.0]


def test_silence_is_reencoded_when_the_clip_length_changes():
    from contextlib import nullcontext

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._component_on_device = lambda _module: nullcontext()
    calls = []

    def encode_waveform(waveform, sample_rate):
        calls.append(waveform.shape[-1])
        return torch.zeros(6, 32, dtype=torch.bfloat16), 3

    pipeline.audio_vae = SimpleNamespace(encode_waveform=encode_waveform)
    first = pipeline._regenerate_silence_rows(22, 3)
    assert pipeline._regenerate_silence_rows(22, 3) is first and len(calls) == 1
    # Another frame count that happens to share the audio length still re-encodes.
    pipeline._regenerate_silence_rows(23, 3)
    assert len(calls) == 2 and calls[0] != calls[1]
    assert first.device.type == "cpu" and first.dtype == torch.float32

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU runtime tests: real pipeline dispatch, batching and stage boundaries."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan_module
from tests.diffusion.models.wan2_2.test_wan22_pipeline_diffuse import _make_pipeline, _make_sampling
from vllm_omni.config.stage_config import DiffusionStageRole
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _ImageVAE(torch.nn.Module):
    dtype = torch.float32
    config = SimpleNamespace(z_dim=4, latents_mean=[0.1] * 4, latents_std=[2.0] * 4)

    def __init__(self):
        super().__init__()
        self.calls = []

    def encode(self, image):
        self.calls.append(image.clone())
        latent = image[:, :1, :, ::8, ::8].repeat(1, 4, 1, 1, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))


def _image_pipeline(role):
    pipeline = _make_pipeline()
    pipeline.stage_role = role
    pipeline.expand_timesteps = True
    pipeline._conditioning_patch_size = (1, 2, 2)
    pipeline.vae = _ImageVAE() if role in (DiffusionStageRole.FULL, DiffusionStageRole.ENCODE) else None
    if role != DiffusionStageRole.FULL and role != DiffusionStageRole.ENCODE:
        pipeline.text_encoder = None
        pipeline.encode_prompt = lambda **kwargs: pytest.fail("G must never encode text")
    else:

        def encode_prompt(**kwargs):
            prompts = kwargs["prompt"]
            n = (1 if isinstance(prompts, str) else len(prompts)) * kwargs["num_videos_per_prompt"]
            positive = torch.ones(n, 3, 8)
            return positive, -positive if kwargs["do_classifier_free_guidance"] else None

        pipeline.encode_prompt = encode_prompt
    pipeline.prepare_latents = Wan22Pipeline.prepare_latents.__get__(pipeline)
    pipeline.captured = {}

    def diffuse(**kwargs):
        pipeline.captured.update(kwargs)
        return kwargs["latents"] + 0.25

    pipeline.diffuse = diffuse
    return pipeline


def _requests(count=2, outputs=2, image=True, cfg=4.0):
    return DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id=f"req-{i}",
                prompt={"prompt": f"prompt-{i}", "multi_modal_data": {"image": torch.full((3, 16, 32), i + 0.5)}}
                if image
                else {"prompt": f"prompt-{i}"},
                sampling_params=_make_sampling(
                    height=16,
                    width=32,
                    num_frames=5,
                    num_outputs_per_prompt=outputs,
                    generator=torch.Generator().manual_seed(123 + i),
                    guidance_scale=cfg,
                ),
            )
            for i in range(count)
        ]
    )


@pytest.mark.parametrize("count,outputs", [(1, 1), (1, 3), (2, 2)])
@pytest.mark.parametrize("image", [True, False])
@pytest.mark.parametrize("role", [DiffusionStageRole.DENOISE, DiffusionStageRole.DENOISE_DECODE])
def test_full_and_e_g_match_conditioning_noise_masks_and_rng(count, outputs, image, role):
    full = _image_pipeline(DiffusionStageRole.FULL)
    encode = _image_pipeline(DiffusionStageRole.ENCODE)
    generate = _image_pipeline(role)
    reference = _requests(count, outputs, image)
    staged = copy.deepcopy(reference)
    full_outputs = full.forward(reference)
    before = [req.sampling_params.generator.get_state().clone() for req in staged.requests]
    payloads = encode.encode_batch(staged)
    for req, payload, state in zip(staged.requests, payloads, before):
        torch.testing.assert_close(req.sampling_params.generator.get_state(), state)
        # Exercise the actual connector-delivered nested prompt convention.
        req.prompt["additional_information"] = payload.custom_output
    staged_outputs = generate.forward(staged)
    for key in ("latents", "prompt_embeds", "negative_prompt_embeds", "latent_condition", "first_frame_mask"):
        expected, actual = full.captured[key], generate.captured[key]
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)
    for actual, expected in zip(staged_outputs, full_outputs):
        value = actual.custom_output["latents"] if role == DiffusionStageRole.DENOISE else actual.output
        torch.testing.assert_close(value, expected.output)
    for req, expected in zip(staged.requests, reference.requests):
        torch.testing.assert_close(
            req.sampling_params.generator.get_state(), expected.sampling_params.generator.get_state()
        )
    if image:
        assert len(encode.vae.calls) == count
        assert all(call.shape[0] == 1 for call in encode.vae.calls)
        assert full.vae.calls[0].shape[0] == count * outputs


@pytest.mark.parametrize(
    "damage,match",
    [
        ("metadata", "wan_conditioning_metadata"),
        ("version", "version"),
        ("height", "height"),
        ("frames", "num_frames"),
        ("condition", "wan_image_condition"),
        ("shape", "shape"),
        ("normalization", "normalization"),
        ("negative", "negative_prompt_embeds"),
        ("positive", "prompt_embeds"),
        ("false", "has_image=False"),
    ],
)
def test_g_rejects_bad_handoff_before_rng_or_encoder_fallback(damage, match):
    encode = _image_pipeline(DiffusionStageRole.ENCODE)
    generate = _image_pipeline(DiffusionStageRole.DENOISE)
    batch = _requests(1)
    payload = encode.encode_batch(batch)[0].custom_output
    if damage == "metadata":
        payload.pop("wan_conditioning_metadata")
    elif damage in ("version", "height", "normalization"):
        payload["wan_conditioning_metadata"][damage] = "invalid"
    elif damage == "frames":
        payload["wan_conditioning_metadata"]["num_frames"] = 9
    elif damage in ("condition", "negative", "positive"):
        payload.pop(
            {"condition": "wan_image_condition", "negative": "negative_prompt_embeds", "positive": "prompt_embeds"}[
                damage
            ]
        )
    elif damage == "shape":
        payload["wan_image_condition"] = payload["wan_image_condition"].repeat(2, 1, 1, 1, 1)
    else:
        payload["wan_conditioning_metadata"]["has_image"] = False
    batch.requests[0].prompt["additional_information"] = payload
    state = batch.requests[0].sampling_params.generator.get_state().clone()
    with pytest.raises(ValueError, match=match):
        generate.forward(batch)
    torch.testing.assert_close(batch.requests[0].sampling_params.generator.get_state(), state)


def test_encode_cfg_uses_same_defaults_and_high_stage_guidance_as_full():
    pipeline = _image_pipeline(DiffusionStageRole.ENCODE)
    batch = _requests(1, image=False, cfg=1.0)
    sampling = batch.requests[0].sampling_params
    sampling.guidance_scale_provided = False
    assert "negative_prompt_embeds" in pipeline.encode_batch(batch)[0].custom_output
    sampling.guidance_scale_provided = True
    sampling.guidance_scale_2_provided = True
    sampling.guidance_scale_2 = 3.0
    assert "negative_prompt_embeds" in pipeline.encode_batch(batch)[0].custom_output


def test_cfg_disabled_does_not_require_negative_embeddings():
    encode = _image_pipeline(DiffusionStageRole.ENCODE)
    generate = _image_pipeline(DiffusionStageRole.DENOISE)
    batch = _requests(1, cfg=1.0)
    payload = encode.encode_batch(batch)[0].custom_output
    assert "negative_prompt_embeds" not in payload
    batch.requests[0].prompt["additional_information"] = payload
    assert len(generate.forward(batch)) == 1


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"guidance_scale": 1.0}, "CFG"),
        ({"max_sequence_length": 256}, "max_sequence_length"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_encode_rejects_incompatible_text_batch_before_compute(overrides, match, reverse):
    pipeline = _image_pipeline(DiffusionStageRole.ENCODE)
    batch = _requests(2)
    vars(batch.requests[1].sampling_params).update(overrides)
    if reverse:
        batch.requests.reverse()
    pipeline.encode_prompt = lambda **kwargs: pytest.fail("validation must precede text encoding")
    states = [req.sampling_params.generator.get_state().clone() for req in batch.requests]
    with pytest.raises(ValueError, match=match):
        pipeline.encode_batch(batch)
    assert not pipeline.vae.calls
    for req, state in zip(batch.requests, states):
        torch.testing.assert_close(req.sampling_params.generator.get_state(), state)


def test_encode_accepts_different_guidance_magnitudes_and_generation_settings():
    pipeline = _image_pipeline(DiffusionStageRole.ENCODE)
    batch = _requests(2, image=False)
    vars(batch.requests[1].sampling_params).update(
        guidance_scale=6.0,
        height=32,
        num_frames=9,
        num_outputs_per_prompt=3,
        num_inference_steps=7,
        output_type="np",
        extra_args={"sample_solver": "euler", "flow_shift": 8.0},
    )
    outputs = pipeline.encode_batch(batch)
    assert len(outputs) == 2
    assert all(output.custom_output["prompt_embeds"].shape == (3, 8) for output in outputs)
    assert all("negative_prompt_embeds" in output.custom_output for output in outputs)
    assert outputs[1].custom_output["wan_conditioning_metadata"]["height"] == 32


@pytest.mark.parametrize(
    "role", [DiffusionStageRole.FULL, DiffusionStageRole.DENOISE, DiffusionStageRole.DENOISE_DECODE]
)
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"height": 32}, "height, width and num_frames"),
        ({"width": 48}, "height, width and num_frames"),
        ({"num_frames": 9}, "height, width and num_frames"),
        ({"guidance_scale": 6.0}, "guidance scales"),
        ({"guidance_scale_2_provided": True, "guidance_scale_2": 6.0}, "guidance scales"),
        ({"num_outputs_per_prompt": 3}, "num_outputs_per_prompt"),
        ({"num_inference_steps": 7}, "num_inference_steps"),
        ({"boundary_ratio": 0.5}, "boundary_ratio"),
        ({"extra_args": {"sample_solver": "euler"}}, "sample_solver"),
        ({"extra_args": {"flow_shift": 8.0}}, "flow_shift"),
    ],
)
def test_generation_rejects_common_setting_mismatch_before_compute(role, expand, overrides, match):
    pipeline = _image_pipeline(role)
    pipeline.expand_timesteps = expand
    pipeline.boundary_ratio = None
    batch = _requests(2, image=False)
    if role != DiffusionStageRole.FULL:
        encode = _image_pipeline(DiffusionStageRole.ENCODE)
        for req, output in zip(batch.requests, encode.encode_batch(batch)):
            req.prompt["additional_information"] = output.custom_output
    vars(batch.requests[1].sampling_params).update(overrides)
    pipeline.encode_prompt = lambda **kwargs: pytest.fail("validation must precede text encoding")
    pipeline.prepare_latents = lambda **kwargs: pytest.fail("validation must precede latent preparation")
    states = [req.sampling_params.generator.get_state().clone() for req in batch.requests]
    with pytest.raises(ValueError, match=match):
        pipeline.forward(batch)
    assert not pipeline.scheduler.set_timesteps_calls
    assert not pipeline.captured
    for req, state in zip(batch.requests, states):
        torch.testing.assert_close(req.sampling_params.generator.get_state(), state)


@pytest.mark.parametrize("role", [DiffusionStageRole.FULL, DiffusionStageRole.DENOISE_DECODE])
def test_generation_rejects_different_output_types(role):
    pipeline = _image_pipeline(role)
    batch = _requests(2, image=False)
    if role != DiffusionStageRole.FULL:
        encode = _image_pipeline(DiffusionStageRole.ENCODE)
        for req, output in zip(batch.requests, encode.encode_batch(batch)):
            req.prompt["additional_information"] = output.custom_output
    batch.requests[1].sampling_params.output_type = "np"
    with pytest.raises(ValueError, match="output_type"):
        pipeline.forward(batch)
    assert not pipeline.scheduler.set_timesteps_calls


@pytest.mark.parametrize(
    "precomputed,negative,cfg", [(False, False, 1.0), (True, False, 4.0), (True, True, 4.0), (True, False, 1.0)]
)
def test_full_checks_text_length_only_when_encoding(precomputed, negative, cfg):
    pipeline = _image_pipeline(DiffusionStageRole.FULL)
    batch = _requests(2, image=False, cfg=cfg)
    batch.requests[1].sampling_params.max_sequence_length = 256
    if precomputed:
        for req in batch.requests:
            req.prompt["prompt_embeds"] = torch.ones(3, 8)
            if negative:
                req.prompt["negative_prompt_embeds"] = -torch.ones(3, 8)
    if not precomputed or (cfg > 1.0 and not negative):
        with pytest.raises(ValueError, match="max_sequence_length"):
            pipeline.forward(batch)
    else:
        pipeline.encode_prompt = lambda **kwargs: pytest.fail("precomputed embeddings must not be re-encoded")
        assert len(pipeline.forward(batch)) == 2


@pytest.mark.parametrize(
    "role", [DiffusionStageRole.FULL, DiffusionStageRole.DENOISE, DiffusionStageRole.DENOISE_DECODE]
)
def test_generation_accepts_effective_defaults_rounding_and_request_local_values(role):
    pipeline = _image_pipeline(role)
    batch = _requests(2, image=False)
    # Request-local generators already differ. Irrelevant controls must not
    # accidentally turn this into a scheduler-key equality check.
    vars(batch.requests[0].sampling_params).update(num_inference_steps=None, max_sequence_length=None)
    vars(batch.requests[1].sampling_params).update(
        height=31,
        width=47,
        num_frames=4,
        num_inference_steps=40,
        max_sequence_length=512,
        guidance_scale=[4.0, 4.0],
        boundary_ratio=0.2,  # Engine boundary overrides this.
        seed=999,
        extra_args={"sample_solver": " UNIPC ", "flow_shift": "5.0", "unused": True},
    )
    if role != DiffusionStageRole.FULL:
        encode = _image_pipeline(DiffusionStageRole.ENCODE)
        for req, output in zip(batch.requests, encode.encode_batch(batch)):
            req.prompt["additional_information"] = output.custom_output
        batch.requests[1].sampling_params.max_sequence_length = 256  # G consumes embeddings, not this setting.
    if role == DiffusionStageRole.DENOISE:
        batch.requests[1].sampling_params.output_type = "np"  # D, not G, consumes this.
    assert len(pipeline.forward(batch)) == 2


def test_dmd_batch_ignores_request_step_count_solver_and_shift():
    pipeline = _image_pipeline(DiffusionStageRole.FULL)
    pipeline.is_dmd = True
    batch = _requests(2, image=False, cfg=1.0)
    vars(batch.requests[1].sampling_params).update(
        num_inference_steps=99, extra_args={"sample_solver": "unused", "flow_shift": "unused"}
    )
    assert len(pipeline.forward(batch)) == 2
    assert not pipeline.scheduler.set_timesteps_calls


@pytest.mark.parametrize("mismatch", ["image", "frames"])
def test_g_rejects_incompatible_request_batch(mismatch):
    encode = _image_pipeline(DiffusionStageRole.ENCODE)
    generate = _image_pipeline(DiffusionStageRole.DENOISE)
    batch = _requests(2)
    if mismatch == "image":
        batch.requests[1].prompt.pop("multi_modal_data")
    else:
        batch.requests[1].sampling_params.num_frames = 9
    for req, output in zip(batch.requests, encode.encode_batch(batch)):
        req.prompt["additional_information"] = output.custom_output
    with pytest.raises(ValueError, match="mix of provided|matching effective"):
        generate.forward(batch)


@pytest.mark.parametrize("image", [True, False])
def test_ti2v_dummy_has_valid_conditioning_without_vae(image):
    pipeline = _image_pipeline(DiffusionStageRole.DENOISE)
    pipeline.transformer_config.text_dim = 8
    batch = _requests(2, image=image)
    pipeline._prepare_dummy_stage_payload(batch)
    outputs = pipeline.forward(batch)
    assert len(outputs) == 2


@pytest.mark.parametrize("output_type", ["latent", "np"])
@pytest.mark.parametrize("non_owner", [False, True])
def test_decode_batch_owner_and_non_owner(output_type, non_owner):
    pipeline = _make_pipeline()
    pipeline.stage_role = DiffusionStageRole.DECODE
    if non_owner:
        pipeline.vae.decode = lambda *args, **kwargs: (torch.empty(0),)
    batch = _requests(2)
    for i, req in enumerate(batch.requests):
        req.prompt = {"additional_information": {"latents": torch.ones(i + 1, 4, 2, 2, 4)}}
        req.sampling_params.output_type = output_type
    outputs = pipeline.decode_batch(batch)
    for i, output in enumerate(outputs):
        if output_type == "latent":
            assert output.output.shape[0] == i + 1
            assert output.media is None
        elif non_owner:
            assert output.output.numel() == 0
            assert output.media is None
        else:
            assert output.media.video.tensor.shape[0] == i + 1
            assert output.output is None


@pytest.mark.parametrize("role", list(DiffusionStageRole))
@pytest.mark.parametrize("expand", [True, False])
def test_constructor_role_modules_and_weight_sources(monkeypatch, tmp_path, role, expand):
    (tmp_path / "model_index.json").write_text(json.dumps({"expand_timesteps": expand}))
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps({"patch_size": [1, 2, 2]}))

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)
            self.decoder = torch.nn.Linear(1, 1)
            self.quant_conv = torch.nn.Linear(1, 1)
            self.post_quant_conv = torch.nn.Linear(1, 1)
            self._cached_conv_counts = {"encoder": 1, "decoder": 1}

        retain_stage_components = DistributedAutoencoderKLWan.retain_stage_components

        def clear_cache(self):
            pass

    def load(_loader, _model, **kwargs):
        return VAE() if kwargs["subfolder"] == "vae" else torch.nn.Linear(1, 1)

    def transformer(self, config):
        module = torch.nn.Linear(1, 1)
        module.config = SimpleNamespace(patch_size=(1, 2, 2))
        return module

    monkeypatch.setattr(wan_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(wan_module, "prefetch_subfolders", lambda *args, **kwargs: None)
    monkeypatch.setattr(wan_module, "from_pretrained_with_prefetch", load)
    monkeypatch.setattr(wan_module, "load_wan_vae_scale_factors", lambda *args: (4, 8))
    monkeypatch.setattr(Wan22Pipeline, "_create_transformer", transformer)
    monkeypatch.setattr(Wan22Pipeline, "setup_diffusion_pipeline_profiler", lambda *args, **kwargs: None)
    pipeline = Wan22Pipeline(
        od_config=SimpleNamespace(
            model=str(tmp_path),
            stage_role=role,
            dtype=torch.float32,
            boundary_ratio=None,
            flow_shift=None,
            enable_diffusion_pipeline_profiler=False,
        )
    )
    assert (pipeline.text_encoder is not None) == (role in (DiffusionStageRole.FULL, DiffusionStageRole.ENCODE))
    dit = role in (DiffusionStageRole.FULL, DiffusionStageRole.DENOISE, DiffusionStageRole.DENOISE_DECODE)
    assert (pipeline.transformer is not None) == dit
    assert bool(pipeline.weights_sources) == dit
    encoder = role == DiffusionStageRole.FULL or (role == DiffusionStageRole.ENCODE and expand)
    decoder = role in (DiffusionStageRole.FULL, DiffusionStageRole.DECODE, DiffusionStageRole.DENOISE_DECODE)
    assert (pipeline.vae is not None) == (encoder or decoder)
    if pipeline.vae is not None:
        assert (pipeline.vae.encoder is not None) == encoder
        assert (pipeline.vae.quant_conv is not None) == encoder
        assert (pipeline.vae.decoder is not None) == decoder
        assert (pipeline.vae.post_quant_conv is not None) == decoder


@pytest.mark.parametrize("encode", [True, False])
def test_pruned_real_tiny_distributed_wan_vae_retains_numerics(encode):
    # No pretrained weights, device placement, or process groups are needed.
    vae = DistributedAutoencoderKLWan(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    ).eval()
    image = torch.zeros(1, 3, 1, 16, 16)
    with torch.no_grad():
        latent = vae.encode(image).latent_dist.mode()
        expected = latent if encode else vae.decode(latent).sample
        vae.retain_stage_components(encode=encode)
        actual = vae.encode(image).latent_dist.mode() if encode else vae.decode(latent).sample
    torch.testing.assert_close(actual, expected)
    assert vae._cached_conv_counts["decoder" if encode else "encoder"] == 0

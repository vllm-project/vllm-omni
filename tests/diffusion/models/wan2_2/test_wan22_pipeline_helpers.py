# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 as wan22_module
from vllm_omni.config.stage_config import DiffusionStageRole
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    Wan22Pipeline,
    create_transformer_from_config,
    get_wan22_post_process_func,
    load_transformer_config,
    load_wan_vae_scale_factors,
    retrieve_latents,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _float32_default_dtype():
    # Other suites leave the global default dtype at bfloat16, which skews the
    # exact-value comparisons below.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


class _LatentDist:
    def sample(self, generator):
        assert isinstance(generator, torch.Generator)
        return torch.tensor([1.0])

    def mode(self):
        return torch.tensor([2.0])


def test_wan22_postprocess_honors_request_output_type() -> None:
    video = torch.zeros(1, 4, 1, 2, 2)

    output = get_wan22_post_process_func(SimpleNamespace())(
        video,
        sampling_params=SimpleNamespace(output_type="latent"),
    )

    assert output is video


def test_retrieve_latents_supports_sample_mode_argmax_and_direct_latents() -> None:
    generator = torch.Generator(device="cpu")

    assert retrieve_latents(SimpleNamespace(latent_dist=_LatentDist()), generator).item() == 1.0
    assert retrieve_latents(SimpleNamespace(latent_dist=_LatentDist()), sample_mode="argmax").item() == 2.0
    torch.testing.assert_close(retrieve_latents(SimpleNamespace(latents=torch.tensor([3.0]))), torch.tensor([3.0]))


def test_retrieve_latents_rejects_unknown_encoder_output() -> None:
    with pytest.raises(AttributeError, match="Could not access latents"):
        retrieve_latents(SimpleNamespace())


def test_load_transformer_config_reads_local_subfolder_config(tmp_path) -> None:
    config_dir = tmp_path / "transformer_2"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({"patch_size": [1, 2, 2], "num_layers": 2}))

    assert load_transformer_config(str(tmp_path), "transformer_2") == {"patch_size": [1, 2, 2], "num_layers": 2}
    assert load_transformer_config(str(tmp_path), "missing") == {}


def test_wan_ti2v_denoise_uses_vae_scale_for_latent_resolution(monkeypatch) -> None:
    def fake_load_config(model, *, subfolder, local_files_only):
        assert model == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        assert subfolder == "vae"
        assert local_files_only is False
        return {"scale_factor_temporal": 4, "scale_factor_spatial": 16}

    monkeypatch.setattr(wan22_module.DistributedAutoencoderKLWan, "load_config", fake_load_config)
    temporal_scale, spatial_scale = load_wan_vae_scale_factors(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", local_files_only=False
    )
    pipeline = object.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor_temporal = temporal_scale
    pipeline.vae_scale_factor_spatial = spatial_scale

    latents = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=48,
        height=384,
        width=384,
        num_frames=17,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu"),
    )

    assert latents.shape == (1, 48, 5, 24, 24)


def test_wan_legacy_vae_config_uses_default_scale_factors(monkeypatch) -> None:
    monkeypatch.setattr(
        wan22_module.DistributedAutoencoderKLWan,
        "load_config",
        lambda *args, **kwargs: {"temperal_downsample": [False, True, True]},
    )

    assert load_wan_vae_scale_factors("legacy-wan", local_files_only=False) == (4, 8)


def test_create_transformer_from_config_maps_supported_keys(monkeypatch) -> None:
    captured = {}

    class FakeTransformer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(wan22_module, "WanTransformer3DModel", FakeTransformer)

    transformer = create_transformer_from_config(
        {
            "patch_size": [1, 2, 2],
            "num_attention_heads": 8,
            "attention_head_dim": 128,
            "in_channels": 16,
            "out_channels": 16,
            "text_dim": 4096,
            "vace_layers": [0],
            "ignored": "value",
        }
    )

    assert isinstance(transformer, FakeTransformer)
    assert captured == {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 8,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 4096,
    }


def test_wan_denoise_outputs_split_latents_per_request() -> None:
    requests = [SimpleNamespace(request_id=f"req-{idx}") for idx in range(2)]
    batch = DiffusionRequestBatch(requests=requests)
    latents = torch.arange(4 * 2 * 1 * 1 * 1, dtype=torch.float32).view(4, 2, 1, 1, 1)

    outputs = Wan22Pipeline._denoise_outputs(batch, latents, num_outputs_per_prompt=2)

    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0].custom_output["latents"], latents[:2])
    torch.testing.assert_close(outputs[1].custom_output["latents"], latents[2:])
    assert all(output.output is None for output in outputs)


def test_wan_denoise_outputs_reject_mismatched_batch() -> None:
    batch = DiffusionRequestBatch(requests=[SimpleNamespace(request_id="req")])

    with pytest.raises(ValueError, match="expected 2"):
        Wan22Pipeline._denoise_outputs(batch, torch.zeros(1, 2, 1, 1, 1), num_outputs_per_prompt=2)


def test_wan_diffuse_batches_transformer_once_per_denoise_step() -> None:
    class FakeTransformer(torch.nn.Module):
        dtype = torch.float32

        def __init__(self) -> None:
            super().__init__()
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            return (kwargs["hidden_states"],)

    @contextmanager
    def progress_bar(total):
        yield SimpleNamespace(update=lambda: None)

    pipeline = object.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = FakeTransformer()
    pipeline.transformer_2 = None
    pipeline.expand_timesteps = False
    pipeline.progress_bar = progress_bar
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, timestep, latents, do_true_cfg: noise

    def predict_noise_maybe_with_cfg(**kwargs):
        positive_kwargs = dict(kwargs["positive_kwargs"])
        current_model = positive_kwargs.pop("current_model")
        return current_model(**positive_kwargs)[0]

    pipeline.predict_noise_maybe_with_cfg = predict_noise_maybe_with_cfg

    latents = torch.zeros(2, 4, 1, 2, 2)
    prompt_embeds = torch.zeros(2, 8, 16)
    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([2.0, 1.0]),
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=None,
        guidance_low=1.0,
        guidance_high=1.0,
        boundary_timestep=None,
        dtype=torch.float32,
        attention_kwargs={},
    )

    assert len(pipeline.transformer.calls) == 2
    for call in pipeline.transformer.calls:
        assert call["hidden_states"].shape[0] == 2
        assert call["timestep"].shape == (2,)
        assert call["encoder_hidden_states"].shape[0] == 2
    torch.testing.assert_close(result, latents)


def test_wan_decode_batch_consumes_latents_with_vae_only() -> None:
    class FakeVAE:
        dtype = torch.float32
        config = SimpleNamespace(latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0], z_dim=2)

        def __init__(self) -> None:
            self.inputs = []

        def decode(self, latents, return_dict=False):
            assert return_dict is False
            self.inputs.append(latents)
            return (latents + 1,)

    pipeline = object.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae = FakeVAE()
    latents = torch.zeros(1, 2, 1, 1, 1)
    request = SimpleNamespace(
        prompt={"latents": latents},
        sampling_params=SimpleNamespace(output_type="np"),
    )

    outputs = pipeline.decode_batch(DiffusionRequestBatch(requests=[request]))

    assert len(pipeline.vae.inputs) == 1
    torch.testing.assert_close(outputs[0].output, latents + 1)


def test_wan_decode_batch_fuses_requests_and_splits_outputs() -> None:
    class FakeVAE:
        dtype = torch.float32
        config = SimpleNamespace(latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0], z_dim=2)

        def __init__(self) -> None:
            self.inputs = []

        def decode(self, latents, return_dict=False):
            assert return_dict is False
            self.inputs.append(latents)
            return (latents + 1,)

    pipeline = object.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae = FakeVAE()
    first = torch.zeros(1, 2, 1, 1, 1)
    second = torch.full((2, 2, 1, 1, 1), 2.0)
    requests = [
        SimpleNamespace(prompt={"latents": latents}, sampling_params=SimpleNamespace(output_type="np"))
        for latents in (first, second)
    ]

    outputs = pipeline.decode_batch(DiffusionRequestBatch(requests=requests))

    assert len(pipeline.vae.inputs) == 1
    assert pipeline.vae.inputs[0].shape == (3, 2, 1, 1, 1)
    torch.testing.assert_close(outputs[0].output, first + 1)
    torch.testing.assert_close(outputs[1].output, second + 1)


def test_wan_decode_batch_requires_latent_payload() -> None:
    pipeline = object.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(pipeline)
    request = SimpleNamespace(prompt={}, sampling_params=SimpleNamespace(output_type="np"))

    with pytest.raises(ValueError, match="requires a tensor 'latents'"):
        pipeline.decode_batch(DiffusionRequestBatch(requests=[request]))


@pytest.mark.parametrize(
    "stage_role",
    [DiffusionStageRole.DENOISE, DiffusionStageRole.DENOISE_DECODE],
)
def test_wan_denoise_dummy_run_synthesizes_prompt_embeds(stage_role: DiffusionStageRole) -> None:
    pipeline = object.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.stage_role = stage_role
    pipeline.device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(dtype=torch.float32)
    pipeline.transformer_config = SimpleNamespace(text_dim=8)
    pipeline.forward = lambda batch: (
        batch.requests[0].prompt["prompt_embeds"],
        batch.requests[0].prompt["negative_prompt_embeds"],
    )
    request = SimpleNamespace(
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
        is_dummy_run_request_id=lambda request_id: request_id == DUMMY_DIFFUSION_REQUEST_ID,
        prompt={"prompt": "dummy run"},
        sampling_params=SimpleNamespace(max_sequence_length=4),
    )

    prompt_embeds, negative_prompt_embeds = pipeline.run_stage(DiffusionRequestBatch(requests=[request]))

    assert prompt_embeds.shape == (1, 4, 8)
    torch.testing.assert_close(negative_prompt_embeds, torch.zeros_like(prompt_embeds))


def test_wan_decode_dummy_run_synthesizes_latents() -> None:
    pipeline = object.__new__(Wan22Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.stage_role = DiffusionStageRole.DECODE
    pipeline.device = torch.device("cpu")
    pipeline.vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(z_dim=16))
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.decode_batch = lambda batch: batch.requests[0].prompt["latents"]
    request = SimpleNamespace(
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
        is_dummy_run_request_id=lambda request_id: request_id == DUMMY_DIFFUSION_REQUEST_ID,
        prompt={"prompt": "dummy run"},
        sampling_params=SimpleNamespace(height=64, width=96, num_frames=9),
    )

    result = pipeline.run_stage(DiffusionRequestBatch(requests=[request]))

    assert result.shape == (1, 16, 3, 8, 12)

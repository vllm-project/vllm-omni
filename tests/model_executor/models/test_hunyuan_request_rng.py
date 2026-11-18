# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real scheduler/runner batching and posterior RNG, with weight-free networks."""

from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import BatchFeature, PretrainedConfig
from vllm import SamplingParams
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem, PlaceholderRange
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import ProcessorInputs
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.request import Request

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS
from vllm_omni.model_executor.mm_request_rng import RandomDrawSpec, RequestRNGProvider
from vllm_omni.model_executor.models.hunyuan_image3 import autoencoder_kl_3d
from vllm_omni.model_executor.models.hunyuan_image3 import hunyuan_image3 as hunyuan
from vllm_omni.model_executor.models.hunyuan_image3.autoencoder_kl_3d import DiagonalGaussianDistribution

pytestmark = [pytest.mark.core_model]


@dataclass
class ModelCase:
    model: hunyuan.HunyuanImage3ForConditionalGeneration
    device: torch.device
    _mm_rng_provider: RequestRNGProvider


class TinyVAE:
    config = SimpleNamespace(shift_factor=0.25, scaling_factor=0.5)
    ffactor_spatial = 1

    def __init__(self, device, dtype):
        self.device, self.dtype = device, dtype

    def encode(self, images):
        mean = images.unsqueeze(2)
        return AutoencoderKLOutput(
            latent_dist=DiagonalGaussianDistribution(torch.cat((mean, torch.zeros_like(mean)), dim=1))
        )


def runner_case(device, dtype, *values):
    return pytest.param(
        (device, dtype),
        *values,
        id="-".join(map(str, (device, dtype, *values))),
        marks=pytest.mark.cpu if device == "cpu" else hardware_marks(res={"cuda": "L4"}),
    )


@pytest.fixture
def runner(request, monkeypatch, mocker):
    device, dtype = getattr(request, "param", ("cpu", torch.float32))
    if device == "cuda" and not torch.get_device_module(device).is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device if device == "cpu" else "cuda:0")
    model = hunyuan.HunyuanImage3ForConditionalGeneration.__new__(hunyuan.HunyuanImage3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = PretrainedConfig(vae_downsample_factor=(1, 1), patch_size=1, vae={"latent_channels": 3})
    model.vae = TinyVAE(device, dtype)
    model.use_data_parallel = False
    model.vision_model = lambda pixels, **kwargs: pixels.to(dtype=dtype)
    model.vision_aligner = torch.nn.Identity()
    model.time_embed = torch.nn.Identity()
    model.patch_embed = lambda latent, time: (latent.flatten(2).transpose(1, 2), None, None)
    model._mrope_joint_img_sep_token_id = 0
    model.model = mocker.Mock()
    model.model.embed_input_ids = lambda ids: torch.zeros((len(ids), 3), device=ids.device, dtype=dtype)

    provider = RequestRNGProvider(model.advance_mm_rng)
    model.bind_mm_rng_provider(provider)
    return ModelCase(model, device, provider)


def images():
    return [torch.full((3, 2, 2), 0.25), torch.full((3, 2, 3), 0.75), torch.full((3, 3, 3), -0.5)]


def _make_request(mocker, req_id, images, seed=42):
    features = BatchFeature(
        {
            "vit_pixel_values": torch.stack([im[:, :1, :1].reshape(1, 3) for im in images]),
            "vit_pixel_attention_mask": torch.ones((len(images), 1), dtype=torch.bool),
            "vit_spatial_shapes": torch.ones((len(images), 2), dtype=torch.long),
            "vae_pixel_values": torch.cat([im.flatten() for im in images]),
            "vae_pixel_size": torch.tensor([im.numel() for im in images]),
            "vae_token_grid_hw": torch.tensor([im.shape[1:] for im in images]),
        }
    )
    processor = hunyuan.HunyuanImage3MultiModalProcessor.__new__(hunyuan.HunyuanImage3MultiModalProcessor)
    processor.info = mocker.Mock(spec=hunyuan.HunyuanImage3ProcessingInfo)
    processor.info.model_id = "hunyuan-test"
    processor.info.ctx = mocker.Mock()
    processor.info.ctx.get_mm_config.return_value = MultiModalConfig()
    processor.info.get_image_processor.return_value = lambda *args, **kwargs: features
    kwargs = {} if seed is None else {"vae_generator_seed": seed}
    identities = processor._get_mm_hashes(
        ProcessorInputs(
            prompt=[],
            mm_data_items=MultiModalDataItems({"image": ImageProcessorItems(images)}),
            hf_processor_mm_kwargs=kwargs,
        )
    )["image"]
    output = processor._call_hf_processor("", {"images": images}, kwargs, {})
    fields = processor._get_mm_fields_config(output, kwargs)
    elements = {key: field.build_elems(key, output[key]) for key, field in fields.items()}

    mm_features, offset = [], 1
    for i, (im, identity) in enumerate(zip(images, identities, strict=True)):
        length = im.shape[1] * im.shape[2] + 2
        mm_features.append(
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({key: elems[i] for key, elems in elements.items()}),
                modality="image",
                identifier=identity,
                mm_hash=identity,
                mm_position=PlaceholderRange(offset=offset, length=length),
            )
        )
        offset += length
    return Request(req_id, [1] * (offset + 1), SamplingParams(seed=999), None, mm_features=mm_features)


@pytest.fixture
def make_request(mocker):
    return partial(_make_request, mocker)


def schedule(inputs=None, *, new=(), finished=()):
    return SchedulerOutput(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs=inputs or {},
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
    )


def reference(runner, images, seed):
    vae = runner.model.vae
    generator = torch.Generator(device=vae.device).manual_seed(seed)
    outputs = []
    for im in images:
        mean = im.to(device=vae.device, dtype=vae.dtype)[None, :, None]
        latent = mean + randn_tensor(mean.shape, generator=generator, device=vae.device, dtype=vae.dtype)
        latent.sub_(vae.config.shift_factor).mul_(vae.config.scaling_factor)
        outputs.append(latent.squeeze(2).flatten(2).transpose(1, 2)[0])
    return outputs, generator.get_state()


def assert_latents(outputs, expected):
    assert len(outputs) == len(expected)
    for output, latent in zip(outputs, expected):
        assert torch.equal(output[: len(latent)], latent)


@pytest.mark.cpu
def test_random_draw_spec_resolves_device_aliases_once(mocker):
    module = mocker.Mock()
    module.current_device.return_value = 1
    get_module = mocker.patch("torch.get_device_module", return_value=module)
    cpu = RandomDrawSpec((1,), torch.float32, torch.device("cpu"))
    assert replace(cpu, device=torch.device("cpu:0")) == cpu
    get_module.assert_not_called()
    implicit = replace(cpu, device=torch.device("cuda"))
    assert implicit.device == torch.device("cuda:1")
    assert implicit == replace(cpu, device=torch.device("cuda:1"))
    module.current_device.assert_called_once()
    module.current_device.return_value = 0
    assert implicit.device == torch.device("cuda:1")
    assert replace(cpu, device=torch.device("cuda")) == replace(cpu, device=torch.device("cuda:0"))
    assert implicit != replace(cpu, device=torch.device("cuda:0"))


@pytest.mark.cpu
def test_sparse_image_indices_restore_after_reverse_order_and_failure():
    spec = RandomDrawSpec((1, 3, 1, 2, 2), torch.float32, torch.device("cpu"))
    provider = RequestRNGProvider(hunyuan.HunyuanImage3ForConditionalGeneration.advance_mm_rng)
    handle = provider.register("A", 0, {8: spec, 2: spec, 5: spec})
    reference_generator = torch.Generator(device=spec.device).manual_seed(0)

    def draw(generator):
        return randn_tensor(spec.shape, generator=generator, device=spec.device, dtype=spec.dtype)

    expected = {index: draw(reference_generator) for index in (2, 5, 8)}
    advance = provider.advance

    def fail_replay(generator, draw_spec):
        advance(generator, draw_spec)
        raise RuntimeError("replay")

    provider.advance = fail_replay
    with pytest.raises(RuntimeError, match="replay"):
        with provider.sample(handle, 5, spec):
            pytest.fail("failed predecessor replay reached target sampling")
    assert len(provider.streams[handle].checkpoints) == 1
    provider.advance = advance
    # A failed first sample may retain replayed predecessors, but not its own boundary.
    with pytest.raises(RuntimeError, match="first sample"):
        with provider.sample(handle, 5, spec) as generator:
            draw(generator)
            raise RuntimeError("first sample")
    assert len(provider.streams[handle].checkpoints) == 2
    with pytest.raises(ValueError, match="specification"):
        with provider.sample(handle, 5, replace(spec, shape=(1,))):
            pytest.fail("invalid posterior accepted")
    for index in (8, 2, 5):
        with provider.sample(handle, index, spec) as generator:
            assert torch.equal(draw(generator), expected[index])
    with pytest.raises(RuntimeError, match="retry"):
        with provider.sample(handle, 2, spec) as generator:
            draw(generator)
            raise RuntimeError("retry")
    for index in (2, 8):
        with provider.sample(handle, index, spec) as generator:
            assert torch.equal(draw(generator), expected[index])
    assert torch.equal(provider.streams[handle].checkpoints[3], reference_generator.get_state())


@pytest.mark.parametrize(
    "runner",
    [runner_case(device, dtype) for device in ("cpu", "cuda") for dtype in (torch.float32, torch.bfloat16)],
    indirect=True,
)
def test_real_vae_posterior_matches_draw_plan_and_replay(runner, make_request, monkeypatch):
    # Exact official encoder config on CUDA; CPU shrinks only widths/image sizes.
    # https://huggingface.co/tencent/HunyuanImage-3.0-Instruct/blob/
    # 2ec2c78bee7d4b94157341fba86c4c2c7b1858b2/config.json
    cuda = runner.device.type == "cuda"
    widths = [128, 256, 512, 1024, 1024] if cuda else [32, 32, 32, 64, 64]
    sizes = ((1024, 1024), (768, 1024), (1024, 768)) if cuda else ((64, 64), (64, 96), (96, 64))
    config = dict(
        block_out_channels=widths,
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        layers_per_block=2,
        ffactor_spatial=16,
        ffactor_temporal=4,
        sample_size=384,
        sample_tsize=96,
        downsample_match_channel=True,
        upsample_match_channel=True,
        scaling_factor=0.562679178327931,
    )
    monkeypatch.setattr(autoencoder_kl_3d, "get_local_device", lambda: runner.device)
    model, dtype = runner.model, runner.model.vae.dtype
    model.vae = (
        autoencoder_kl_3d.AutoencoderKLConv3D.from_config(config, only_encoder=True)
        .to(device=runner.device, dtype=dtype)
        .eval()
    )
    if cuda:
        model.vae.enable_spatial_tiling()
        buckets = hunyuan.HunyuanImage3Processor.ResolutionGroup(
            base_size=1024,
            extra_resolutions=[hunyuan.HunyuanImage3Processor.Resolution(s) for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS],
        )
        assert all(buckets.get_target_size(w, h) == (w, h) for h, w in sizes)
    model.config = PretrainedConfig(vae_downsample_factor=(16, 16), patch_size=1, vae=config)
    refs = [torch.zeros((3, h, w)) for h, w in sizes]
    req = make_request("real-vae", refs)
    for feature, (h, w) in zip(req.mm_features, sizes, strict=True):
        grid = feature.data["vae_token_grid_hw"]
        feature.data["vae_token_grid_hw"] = replace(grid, data=torch.tensor([h // 16, w // 16]))
    specs = model.get_mm_rng_specs(req.mm_features, 42)
    provider = runner._mm_rng_provider
    handle = provider.register(req.request_id, 42, specs)
    reference_generator = torch.Generator(device=runner.device).manual_seed(42)
    with torch.inference_mode():
        posteriors, expected = [], []
        for index, pixels in enumerate(refs):
            posterior = model.vae.encode(pixels.unsqueeze(0).to(device=runner.device, dtype=dtype)).latent_dist
            assert specs[index] == RandomDrawSpec(
                tuple(posterior.mean.shape), posterior.parameters.dtype, posterior.parameters.device
            )
            posteriors.append(posterior)
            expected.append(posterior.sample(reference_generator))
            assert torch.isfinite(expected[-1]).all()
        # Start after two skipped draws, then reorder and recompute.
        for index in (2, 0, 1, 2):
            with provider.sample(handle, index, specs[index]) as generator:
                assert torch.equal(posteriors[index].sample(generator), expected[index])
    assert torch.equal(provider.streams[handle].checkpoints[3], reference_generator.get_state())

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


import pytest
import torch
from diffusers import AutoencoderDC, SCMScheduler
from diffusers import SanaSprintPipeline as ReferencePipeline
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel

from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.sana_sprint.pipeline_sana_sprint import SanaSprintPipeline
from vllm_omni.diffusion.models.sana_sprint.transformer_sana_sprint import SanaSprintTransformer2DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TINY_CONFIG = dict(
    in_channels=4,
    out_channels=4,
    num_attention_heads=2,
    attention_head_dim=8,
    num_cross_attention_heads=2,
    cross_attention_head_dim=8,
    cross_attention_dim=16,
    caption_channels=12,
    num_layers=2,
    sample_size=4,
    guidance_embeds=True,
    qk_norm="rms_norm_across_heads",
)


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.layer.get_attn_backend_for_role", lambda **kwargs: (SDPABackend, None)
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        reference = SanaTransformer2DModel(**TINY_CONFIG).eval()
        native = SanaSprintTransformer2DModel(**TINY_CONFIG).eval()
    loaded = native.load_weights(iter(reference.state_dict().items()))
    assert loaded == set(dict(native.named_parameters()))
    return reference, native


@pytest.mark.parametrize("shape", [(1, 4, 4, 4), (2, 4, 4, 6)])
@torch.no_grad()
def test_transformer_matches_reference(models, shape):
    reference, native = models
    generator = torch.Generator().manual_seed(7)
    inputs = dict(
        hidden_states=torch.randn(shape, generator=generator),
        encoder_hidden_states=torch.randn(shape[0], 7, 12, generator=generator),
        timestep=torch.full((shape[0],), 0.8),
        guidance=torch.full((shape[0],), 0.45),
        encoder_attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0, 0]]).expand(shape[0], -1),
    )
    torch.testing.assert_close(native(**inputs), reference(**inputs).sample, atol=2e-6, rtol=1e-5)
    # Masked text positions must not change the image prediction.
    expected = native(**inputs)
    inputs["encoder_hidden_states"][:, 3:] += 100
    torch.testing.assert_close(native(**inputs), expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("guidance", [None, 0.0, 6.0])
@torch.no_grad()
def test_scm_denoising_matches_reference(models, steps, guidance, monkeypatch):
    reference_transformer, native_transformer = models
    vae = AutoencoderDC(
        in_channels=3,
        latent_channels=4,
        encoder_block_types=("ResBlock", "ResBlock"),
        decoder_block_types=("ResBlock", "ResBlock"),
        encoder_block_out_channels=(16, 16),
        decoder_block_out_channels=(16, 16),
        encoder_layers_per_block=(1, 1),
        decoder_layers_per_block=(1, 1),
        encoder_qkv_multiscales=((), ()),
        decoder_qkv_multiscales=((), ()),
        attention_head_dim=8,
    )
    reference = ReferencePipeline(
        tokenizer=None,
        text_encoder=None,
        transformer=reference_transformer,
        vae=vae,
        scheduler=SCMScheduler(),
    )
    native = SanaSprintPipeline.__new__(SanaSprintPipeline)
    torch.nn.Module.__init__(native)
    native.transformer = native_transformer
    native.scheduler = SCMScheduler()
    native.scheduler.set_timesteps(steps, max_timesteps=1.57080, intermediate_timesteps=1.3 if steps == 2 else None)
    generator = torch.Generator().manual_seed(8)
    embeds = torch.randn(1, 7, 12, generator=generator)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    # Match the reference's minimum 32-pixel geometry with a 2x tiny VAE.
    latents = torch.randn(1, 4, 16, 16, generator=generator)
    expected = reference(
        guidance_scale=4.5 if guidance is None else guidance,
        prompt_embeds=embeds,
        prompt_attention_mask=mask,
        latents=latents.clone(),
        height=32,
        width=32,
        num_inference_steps=steps,
        intermediate_timesteps=1.3 if steps == 2 else None,
        use_resolution_binning=False,
        output_type="latent",
        generator=torch.Generator().manual_seed(42),
    ).images
    native.device = torch.device("cpu")
    native.vae_scale_factor = 2
    native.od_config = OmniDiffusionConfig(output_type="latent")
    native.enable_diffusion_pipeline_profiler = False
    monkeypatch.setattr(native, "encode_prompt", lambda *_: (embeds, mask))
    request = OmniDiffusionRequest(
        prompt="a cat",
        request_id="parity",
        sampling_params=OmniDiffusionSamplingParams(
            guidance_scale=guidance,
            height=32,
            width=32,
            num_inference_steps=steps,
            latents=latents.clone(),
            generator=torch.Generator().manual_seed(42),
            extra_args={"use_resolution_binning": False},
        ),
    )
    actual = native(DiffusionRequestBatch([request])).output
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("prompt", "params", "message"),
    [
        ({"prompt": "cat", "negative_prompt": "blur"}, {}, "negative prompts"),
        ({"prompt": "cat", "multi_modal_data": {"image": "image"}}, {}, "image inputs"),
        ("cat", {"height": 513}, "multiples of 32"),
        ("cat", {"width": 0}, "multiples of 32"),
        ("cat", {"num_inference_steps": 0}, "must be positive"),
        ("cat", {"max_sequence_length": 1}, "at least 2"),
    ],
)
def test_invalid_request_fails_before_encoding(prompt, params, message, mocker):
    pipe = SanaSprintPipeline.__new__(SanaSprintPipeline)
    torch.nn.Module.__init__(pipe)
    pipe.encode_prompt = mocker.Mock(side_effect=AssertionError("encoding should not run"))
    request = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=OmniDiffusionSamplingParams(**params),
                request_id="invalid",
            )
        ]
    )
    with pytest.raises(ValueError, match=message):
        pipe(request)
    pipe.encode_prompt.assert_not_called()


def test_prompt_repetition_keeps_masks_aligned(mocker):
    pipe = SanaSprintPipeline.__new__(SanaSprintPipeline)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    from transformers import BatchEncoding

    tokenized = BatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        }
    )
    pipe.tokenizer = mocker.Mock(return_value=tokenized)
    pipe.tokenizer.encode.return_value = [1, 2]
    embeddings = torch.arange(8).reshape(2, 4, 1).float()
    pipe.text_encoder = mocker.Mock(return_value=(embeddings,))
    actual, masks = pipe.encode_prompt(["CAT", "DOG"], 2, 4)
    torch.testing.assert_close(actual, embeddings.repeat_interleave(2, dim=0))
    torch.testing.assert_close(masks, tokenized.attention_mask.repeat_interleave(2, dim=0))


@pytest.mark.parametrize(
    "option",
    [
        "tensor_parallel_size",
        "sequence_parallel_size",
        "cfg_parallel_size",
        "vae_patch_parallel_size",
        "pipeline_parallel_size",
        "text_encoder_tp_size",
    ],
)
def test_parallel_modes_rejected_before_loading(option):
    config = OmniDiffusionConfig()
    setattr(config.parallel_config, option, 2)
    with pytest.raises(ValueError, match="single-device"):
        SanaSprintPipeline(od_config=config)

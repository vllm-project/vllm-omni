import gc
import os

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Test Configurations
# ──────────────────────────────────────────────────────────────────────────────

RM_MODEL = "liuhuohuo/DiNa-LRM-SD35M-12layers"
SD3_MODEL = "stabilityai/stable-diffusion-3.5-medium"
PROMPT = "A girl walking in the street"
NOISE_SIGMA = 0.4
RTOL = 0.01

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype() -> torch.dtype:
    return torch.bfloat16


@pytest.fixture(scope="module")
def synthetic_image() -> Image.Image:
    """Creates a deterministic synthetic RGB image for baseline testing."""
    rng = np.random.RandomState(seed=0)
    arr = rng.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _image_to_latent(pipe, image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode a PIL image to SD3 latent space using the provided pipeline VAE."""
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    img_t = transform(image).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(img_t).latent_dist.sample()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latents


# ──────────────────────────────────────────────────────────────────────────────
# Path A – official DRMInferencer
# ──────────────────────────────────────────────────────────────────────────────


def run_official(prompts, image, device, dtype) -> torch.Tensor:
    """Run inference using the official DRMInferencer from diffusion_rm.

    Returns raw reward scores as a float32 CPU tensor of shape (B,).
    """
    from diffusers import StableDiffusion3Pipeline
    from diffusion_rm.infer.inference import DRMInferencer
    from diffusion_rm.models.sd3_rm import encode_prompt

    local = os.path.exists(SD3_MODEL)
    pipe = StableDiffusion3Pipeline.from_pretrained(SD3_MODEL, torch_dtype=dtype, local_files_only=local)
    for comp in [
        pipe.vae,
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
        pipe.transformer,
    ]:
        comp.to(device, dtype=dtype)

    scorer = DRMInferencer(
        pipeline=pipe,
        config_path=None,
        model_path=RM_MODEL,
        device=device,
        model_dtype=dtype,
        load_from_disk=os.path.exists(RM_MODEL),
    )

    text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    with torch.no_grad():
        prompt_embeds, pooled_embeds = encode_prompt(text_encoders, tokenizers, prompts, max_sequence_length=256)
    prompt_embeds = prompt_embeds.to(device)
    pooled_embeds = pooled_embeds.to(device)
    latents = _image_to_latent(pipe, image, device, dtype)

    torch.manual_seed(42)
    with torch.no_grad():
        scores = scorer.reward(
            text_conds={
                "encoder_hidden_states": prompt_embeds,
                "pooled_projections": pooled_embeds,
            },
            latents=latents,
            u=NOISE_SIGMA,
        )

    result = scores[0].float().cpu()
    result = (result + 10.0) / 10.0  # normalization
    del scorer, pipe
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Path B – OmniDiffusion.generate (vLLM-Omni integration)
# ──────────────────────────────────────────────────────────────────────────────


def run_vllm_omni(prompts, image, device, dtype) -> torch.Tensor:
    """Run inference using OmniDiffusion.generate (the vLLM-Omni integration path).

    Returns raw reward scores as a float32 CPU tensor of shape (B,).
    """
    from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    dtype_str = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dtype, "bfloat16")

    client = OmniDiffusion(model=RM_MODEL, dtype=dtype_str)
    request_prompts = {"prompt": prompts[0], "multi_modal_data": {"image": image}}
    sampling_params = OmniDiffusionSamplingParams(
        extra_args={"noise_level": NOISE_SIGMA},
    )

    torch.manual_seed(42)
    outputs = client.generate(request_prompts, sampling_params)

    scores = outputs[0].latents[0].float().cpu()
    scores = (scores + 10.0) / 10.0  # normalization
    del client
    gc.collect()
    torch.cuda.empty_cache()

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.omni
def test_dina_lrm_numerical_equivalence(synthetic_image, device, dtype):
    """
    Verifies that DRMInferencer (official) and DiNaLRMPipeline (vllm-omni)
    produce numerically identical reward scores under identical initialization.
    """
    score_official = run_official([PROMPT], synthetic_image, device, dtype)
    score_vllm = run_vllm_omni([PROMPT], synthetic_image, device, dtype)

    max_diff = (score_official - score_vllm).abs().max().item() / max(
        score_official.abs().max().item(), score_vllm.abs().max().item(), 1.0
    )

    assert torch.allclose(score_official, score_vllm, atol=0.0, rtol=RTOL), (
        f"Equivalence check FAILED: max_abs_diff = {max_diff:.3e} (rtol = {RTOL:.3e})\n"
        f"  official  = {score_official.tolist()}\n"
        f"  vllm-omni = {score_vllm.tolist()}"
    )

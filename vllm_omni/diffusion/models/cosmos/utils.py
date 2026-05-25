import importlib.util

import torch

if importlib.util.find_spec("cosmos_guardrail") is not None:
    from cosmos_guardrail import CosmosSafetyChecker
else:

    class CosmosSafetyChecker:
        def __init__(self, *args, **kwargs):
            message = (
                "`cosmos_guardrail` is not installed. Please install it to use "
                "the safety checker for Cosmos: `pip install cosmos_guardrail`."
            )
            raise ImportError(message)


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, "
    "shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed "
    "scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, "
    "color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, "
    "poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
)

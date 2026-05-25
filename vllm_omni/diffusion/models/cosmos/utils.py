import functools
import importlib.util
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _patch_cosmos_guardrail_snapshot_download() -> None:
    """Avoid materializing nested ``.locks/`` files as cache symlinks."""

    import cosmos_guardrail.cosmos_guardrail as _cg

    original = _cg.snapshot_download
    if getattr(original, "_vllm_omni_lock_patched", False):
        return

    @functools.wraps(original)
    def patched(*args, **kwargs):
        extra = ["**/.locks/**", "**/.locks/**/*"]
        existing = kwargs.get("ignore_patterns") or []
        if isinstance(existing, str):
            existing = [existing]
        kwargs["ignore_patterns"] = list(existing) + extra
        checkpoint_dir = original(*args, **kwargs)
        _strip_bogus_aegis_locks(checkpoint_dir)
        return checkpoint_dir

    patched._vllm_omni_lock_patched = True  # type: ignore[attr-defined]
    _cg.snapshot_download = patched


def _strip_bogus_aegis_locks(checkpoint_dir: str) -> None:
    """Remove pre-baked ``.lock`` symlinks under ``aegis/`` in a cached snapshot."""
    aegis_dir = os.path.join(checkpoint_dir, "aegis")
    if not os.path.isdir(aegis_dir):
        return
    for root, _, files in os.walk(aegis_dir):
        if os.path.basename(root) != ".locks" and ".locks" not in root.split(os.sep):
            continue
        for name in files:
            if not name.endswith(".lock"):
                continue
            path = os.path.join(root, name)
            if os.path.islink(path):
                try:
                    os.unlink(path)
                except OSError as exc:
                    logger.debug("Failed to remove bogus lock symlink %s: %s", path, exc)


if importlib.util.find_spec("cosmos_guardrail") is not None:
    _patch_cosmos_guardrail_snapshot_download()
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

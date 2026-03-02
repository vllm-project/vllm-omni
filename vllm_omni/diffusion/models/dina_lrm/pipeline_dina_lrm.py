"""DiNa-LRM pipeline adapter for vLLM-Omni.

DiNa-LRM is a *discriminative* reward model that scores (text, image) pairs
in the SD3.5-M latent space.  This module wraps it behind vLLM-Omni's standard
pipeline interface so it can be registered and served alongside generative models.

This file (together with dina_lrm_model.py) is **fully self-contained** –
it does NOT import from the ``diffusion_rm`` package, so it can be dropped
directly into vLLM-Omni without installing that package as a dependency.

─────────────────────────────────────────────────────────────────────────────
DiNa-LRM checkpoint layout  (od_config.model)
─────────────────────────────────────────────────────────────────────────────
    <checkpoint_root>/
        config.json                 ← OmegaConf model config
        checkpoint/
            backbone_lora/          ← PEFT LoRA adapter  (when use_lora=True)
            rm_head.pt              ← RewardHead state dict

─────────────────────────────────────────────────────────────────────────────
Request / Response contract
─────────────────────────────────────────────────────────────────────────────
Input  (OmniDiffusionRequest):
    req.prompts                 list[str | dict]
                                Text prompt(s) to score against.
                                dict form: {"prompt": "..."}

    req.multi_modal_data        {"image": value}
        value = PIL.Image       → encoded to latents via the SD3 VAE
        value = torch.Tensor    (B, C, H, W) already in SD3 latent space
                                (e.g. pipeline output_type='latent')

    req.sampling_params.extra_args["noise_level"]
                                float  (default 0.1)
                                Noise sigma for the reward query.
                                0.1  → recommended for clean/disk images
                                0.4  → recommended for pipeline-generated latents

Output (DiffusionOutput):
    output                      torch.Tensor  (B,)
                                Raw reward scores.
                                Normalised: (score + 10.0) / 10.0
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import snapshot_download
from PIL import Image as PILImage
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.utils import get_local_device

from .dina_lrm_model import SD3RewardModel, encode_prompt, load_rm_config

logger = init_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pre / Post processing functions
# ──────────────────────────────────────────────────────────────────────────────


def get_dina_lrm_pre_process_func(od_config):
    """Pre-processing hook: pass-through.

    Image encoding (PIL → latent) is handled inside ``forward`` because the
    VAE lives on the pipeline object itself.
    """

    def pre_process_func(request):
        return request

    return pre_process_func


def get_dina_lrm_post_process_func(od_config):
    """Post-processing hook: normalise raw reward scores.

    Applies  ``normalised = (raw_score + 10.0) / 10.0``.
    The result is typically in [0, 2], centred near 1.
    """

    def post_process_func(scores: torch.Tensor):
        if isinstance(scores, torch.Tensor):
            return (scores + 10.0) / 10.0
        return scores

    return post_process_func


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────


class DiNaLRMPipeline(nn.Module):
    """vLLM-Omni pipeline for DiNa-LRM reward scoring.

    DiNa-LRM is a *discriminative* model — it does NOT generate images.
    It evaluates text-image alignment by scoring the image's SD3 latent
    representation with a truncated SD3.5-M backbone (first 12 transformer
    layers, LoRA fine-tuned) and a cross-attention reward head.

    Parameters
    ----------
    od_config:
        vLLM-Omni ``OmniDiffusionConfig``.
        ``od_config.model`` must point to a DiNa-LRM HF repo ID
        (e.g. ``"liuhuohuo/DiNa-LRM-SD35M-12layers"``) or a local directory.
    prefix:
        Unused weight prefix (kept for API compatibility with vLLM-Omni).
    """

    _repeated_blocks: list[str] = []
    support_image_input = True

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.weights_sources: list = []

        # ── resolve device ────────────────────────────────────────────────────
        self.device = get_local_device()

        # ── 1. Resolve checkpoint path ────────────────────────────────────────
        rm_model_path = od_config.model
        if not os.path.exists(rm_model_path):
            logger.info("Downloading DiNa-LRM checkpoint: %s", rm_model_path)
            rm_model_path = snapshot_download(repo_id=rm_model_path)
        else:
            logger.info("Loading DiNa-LRM from local path: %s", rm_model_path)

        # ── 2. Load DiNa-LRM config ───────────────────────────────────────────
        config_path = os.path.join(rm_model_path, "config.json")
        self.rm_config = load_rm_config(config_path)

        # ── 3. Resolve model dtype ────────────────────────────────────────────
        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        raw_dtype = getattr(od_config, "dtype", None) or "bfloat16"
        self.model_dtype = (
            raw_dtype if isinstance(raw_dtype, torch.dtype) else _dtype_map.get(str(raw_dtype), torch.bfloat16)
        )

        # ── 4. Load SD3.5-M base components (text encoders, transformer, VAE) ─
        sd3_base = self.rm_config.model.backbone_model_id
        sd3_local = os.path.exists(sd3_base)
        logger.info("Loading SD3.5-M base components from: %s", sd3_base)

        sd3_pipe = StableDiffusion3Pipeline.from_pretrained(
            sd3_base,
            torch_dtype=self.model_dtype,
            local_files_only=sd3_local,
        )
        for comp in [
            sd3_pipe.vae,
            sd3_pipe.text_encoder,
            sd3_pipe.text_encoder_2,
            sd3_pipe.text_encoder_3,
            sd3_pipe.transformer,
        ]:
            comp.to(self.device, dtype=self.model_dtype)

        # ── 5. Build SD3RewardModel ───────────────────────────────────────────
        self.reward_model = SD3RewardModel(
            pipeline=sd3_pipe,
            config_model=self.rm_config.model,
            device=self.device,
            dtype=self.model_dtype,
        )
        self.text_encoders = [
            sd3_pipe.text_encoder,
            sd3_pipe.text_encoder_2,
            sd3_pipe.text_encoder_3,
        ]
        self.tokenizers = [
            sd3_pipe.tokenizer,
            sd3_pipe.tokenizer_2,
            sd3_pipe.tokenizer_3,
        ]
        self.vae = sd3_pipe.vae

        scheduler_cls = type(sd3_pipe.scheduler)
        self.noise_scheduler = scheduler_cls.from_config(sd3_pipe.scheduler.config)

        # ── 6. Load LoRA + reward head weights ───────────────────────────────
        checkpoint_path = os.path.join(rm_model_path, "checkpoint")
        self._load_reward_checkpoint(checkpoint_path)

        self.add_noise: bool = (
            self.rm_config.training.add_noise
            if (hasattr(self.rm_config, "training") and hasattr(self.rm_config.training, "add_noise"))
            else True
        )

        self.reward_model.eval()
        logger.info("DiNaLRMPipeline ready on %s (dtype=%s).", self.device, self.model_dtype)

    # ── checkpoint loading ────────────────────────────────────────────────────

    def _load_reward_checkpoint(self, checkpoint_path: str) -> None:
        """Load LoRA adapter and reward head from *checkpoint_path*."""
        rm_cfg = self.rm_config.model
        device = self.device

        if rm_cfg.use_lora:
            lora_dir = os.path.join(checkpoint_path, "backbone_lora")
            if os.path.exists(lora_dir):
                logger.info("Loading LoRA weights from: %s", lora_dir)
                self.reward_model.backbone.load_adapter(lora_dir, adapter_name="rm_lora")
            else:
                logger.warning("LoRA directory not found: %s", lora_dir)

            rm_head_path = os.path.join(checkpoint_path, "rm_head.pt")
            if os.path.exists(rm_head_path):
                logger.info("Loading reward head from: %s", rm_head_path)
                head_state = torch.load(rm_head_path, map_location=device, weights_only=True)
                self.reward_model.reward_head.load_state_dict(head_state)
            else:
                logger.warning("Reward head file not found: %s", rm_head_path)

        elif not rm_cfg.freeze_backbone:
            full_path = os.path.join(checkpoint_path, "full_model.pt")
            if os.path.exists(full_path):
                logger.info("Loading full model from: %s", full_path)
                state = torch.load(full_path, map_location=device, weights_only=True)
                self.reward_model.load_state_dict(state)
            else:
                logger.warning("Full model file not found: %s", full_path)

        else:
            # Frozen backbone variant: only the reward head is saved.
            rm_head_path = os.path.join(checkpoint_path, "rm_head.pt")
            if os.path.exists(rm_head_path):
                logger.info("Loading reward head (frozen backbone) from: %s", rm_head_path)
                head_state = torch.load(rm_head_path, map_location=device, weights_only=True)
                self.reward_model.reward_head.load_state_dict(head_state)
            else:
                logger.warning("Reward head file not found: %s", rm_head_path)

    def load_weights(self, weights: Iterable[tuple]) -> set:
        """vLLM-Omni weight-loading hook (pass-through).

        All weights are loaded in ``__init__`` via ``_load_reward_checkpoint``.
        Returning an empty set tells vLLM-Omni's weight manager to skip its
        default loading path.
        """
        return set()

    # ── inference helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _get_timesteps_from_sigma(
        noise_scheduler,
        sigma_target: torch.Tensor,
        n_dim: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the nearest scheduler (sigma, timestep) for each value in *sigma_target*.

        Parameters
        ----------
        sigma_target : (B,) tensor in [0, 1]
        n_dim        : number of dimensions the sigma tensor should have
                       (for broadcasting with the latent tensor).
        """
        sigmas = noise_scheduler.sigmas.to(sigma_target.device)
        idx = torch.argmin((sigmas[None, :] - sigma_target[:, None]).abs(), dim=1)
        timesteps = noise_scheduler.timesteps.to(sigma_target.device)[idx]
        sigma = sigmas[idx]
        while sigma.dim() < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma, timesteps

    def _encode_image_to_latent(self, image) -> torch.Tensor:
        """Encode a PIL Image or normalised RGB tensor into SD3 latent space.

        Parameters
        ----------
        image : PIL.Image or torch.Tensor (C,H,W) | (1,C,H,W)  values in [-1,1]
        """
        if isinstance(image, PILImage.Image):
            transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
            image = transform(image).unsqueeze(0).to(self.device, dtype=self.model_dtype)
        elif isinstance(image, torch.Tensor):
            image = image.to(self.device, dtype=self.model_dtype)
            if image.dim() == 3:
                image = image.unsqueeze(0)

        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents

    # ── main forward ──────────────────────────────────────────────────────────

    def forward(self, req):
        """Score (prompt, image) pairs and return raw reward tensors.

        Parameters
        ----------
        req : OmniDiffusionRequest
            * ``req.prompts``                    – list[str | dict]
            * ``req.prompts[0].multi_modal_data["image"]``  – PIL.Image **or**
                                                   torch.Tensor (B,C,H,W)
                                                   (already in SD3 latent space)
            * ``req.sampling_params.extra_args["noise_level"]``
                                                   – float, default 0.1

        Returns
        -------
        DiffusionOutput
            ``output``: torch.Tensor (B,) of raw reward scores.
            Apply ``(score + 10.0) / 10.0`` for human-readable values.
        """
        if len(req.prompts) > 1:
            logger.warning(
                "This model only supports a single prompt, not a batched request."
                "Taking only the first image for now."
            )
        # ── text prompts ──────────────────────────────────────────────────────
        first_prompt = req.prompts[0]
        prompt_texts: list[str] = first_prompt.get("prompt", "")
        prompt_texts: list[str] = [first_prompt.get("prompt", "")]
        # ── noise level u (passed as extra_args["noise_level"]) ──────────────
        extra_args: dict = getattr(req.sampling_params, "extra_args", {}) or {}
        u: float = float(extra_args.get("noise_level", 0.1))

        # ── image / latent ────────────────────────────────────────────────────
        multi_modal = first_prompt.get("multi_modal_data", {})
        image_input = multi_modal.get("image", None)
        if image_input is None:
            raise ValueError(
                "DiNaLRMPipeline requires an image in "
                "req.multi_modal_data['image'].  "
                "Pass a PIL.Image or a (B, C, H, W) latent tensor."
            )

        with torch.no_grad():
            # ── encode text ───────────────────────────────────────────────────
            prompt_embeds, pooled_embeds = encode_prompt(
                self.text_encoders,
                self.tokenizers,
                prompt_texts,
                max_sequence_length=256,
            )
            prompt_embeds = prompt_embeds.to(self.device)
            pooled_embeds = pooled_embeds.to(self.device)

            # ── resolve latents ───────────────────────────────────────────────
            if isinstance(image_input, torch.Tensor) and image_input.dim() == 4:
                # Already in SD3 latent space.
                latents = image_input.to(self.device, dtype=self.model_dtype)
            else:
                latents = self._encode_image_to_latent(image_input)

            # ── add noise at sigma = u ────────────────────────────────────────
            bsz = latents.shape[0]
            u_tensor = torch.full((bsz,), u, device=self.device, dtype=torch.float32)
            sigmas, timesteps = self._get_timesteps_from_sigma(self.noise_scheduler, u_tensor, n_dim=latents.dim())
            if self.add_noise:
                noisy_latents = (1.0 - sigmas) * latents + sigmas * torch.randn_like(latents)
            else:
                noisy_latents = latents
            noisy_latents = noisy_latents.to(self.model_dtype)

            # ── LoRA adapter ──────────────────────────────────────────────────
            if self.rm_config.model.use_lora:
                self.reward_model.backbone.set_adapter("rm_lora")

            # ── reward forward ────────────────────────────────────────────────
            scores = self.reward_model(
                latents=noisy_latents,
                timesteps=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
            )

        return DiffusionOutput(output=scores, trajectory_latents=scores)
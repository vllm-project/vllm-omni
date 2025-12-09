# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Any, Optional, Union

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import (
    AutoencoderKLFlux2,
)
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


def format_text_input(prompts: list[str], system_message: str = None):
    """Format prompts for Mistral3 tokenizer."""
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_txt
    ]


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for flow matching scheduler."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_flux2_post_process_func(od_config: OmniDiffusionConfig):
    """Create post-processing function to convert tensors to PIL images."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    # Read VAE config to determine scale factor
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config.get("block_out_channels", [3])) - 1)

    # Create image processor
    image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        """
        Convert [-1, 1] tensor to PIL images.

        Args:
            images: [B, C, H, W] tensor in range [-1, 1]

        Returns:
            List of PIL.Image objects
        """
        return image_processor.postprocess(images, output_type="pil")

    return post_process_func


class Flux2Pipeline(nn.Module):
    """
    Pipeline for Flux 2.0 text-to-image generation.

    Components:
    - Text Encoder: Mistral3-Small-24B (generates 15360-dim embeddings)
    - Transformer: 8 dual-stream + 48 single-stream blocks
    - VAE: AutoencoderKLFlux2 with BatchNorm (128-channel latents)
    - Scheduler: FlowMatchEulerDiscreteScheduler
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        model = od_config.model

        # Check if model is a local path
        local_files_only = os.path.exists(model)

        # Load scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        logger.info("Loaded Flux2 scheduler successfully")

        with set_default_torch_dtype(torch.bfloat16):
            # Load text encoder
            self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                model, subfolder="text_encoder", local_files_only=local_files_only, torch_dtype=torch.bfloat16
            )
            logger.info("Loaded Flux2 text encoder successfully")

            # Load VAE
            self.vae = AutoencoderKLFlux2.from_pretrained(
                model, subfolder="vae", local_files_only=local_files_only, torch_dtype=torch.bfloat16
            ).to(self.device)
            logger.info("Loaded Flux2 VAE successfully")

            # Initialize transformer (with config support for CI testing)
            self.transformer = Flux2Transformer2DModel(od_config=od_config)
            logger.info("Initialized Flux2 transformer successfully.")

            # Load tokenizer
            self.tokenizer = AutoProcessor.from_pretrained(
                model, subfolder="tokenizer", local_files_only=local_files_only
            )
            logger.info("Loaded Flux2 tokenizer successfully.")

        self.stage = None

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        # Flux2 latents are turned into 2x2 patches and packed.
        # This means the latent width and height has to be divisible by the patch size.
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

        # System message for Mistral3
        self.system_message = (
            "You are an AI that reasons about image descriptions. "
            "You give structured responses focusing on object relationships, "
            "object attribution and actions without speculation."
        )

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        """Check input validity."""
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. "
                "Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    @staticmethod
    def _get_mistral_3_small_prompt_embeds(
        text_encoder: Mistral3ForConditionalGeneration,
        tokenizer: AutoProcessor,
        prompt: Union[str, list[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        system_message: str = "You are an AI that reasons about image descriptions. "
        "You give structured responses focusing on object relationships, "
        "object attribution and actions without speculation.",
        hidden_states_layers: list[int] = (10, 20, 30),
    ):
        """Extract prompt embeddings from Mistral3 text encoder."""
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format input messages
        messages_batch = format_text_input(prompts=prompt, system_message=system_message)

        # Process all messages at once
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @staticmethod
    def _prepare_text_ids(
        x: torch.Tensor,  # (B, L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        """Generate 4D position IDs for text tokens."""
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            layer_dim = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, layer_dim)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor):  # (B, C, H, W)
        """
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents: Latent tensor of shape (B, C, H, W)

        Returns:
            Position IDs tensor of shape (B, H*W, 4)
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        layer_dim = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, layer_dim)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _patchify_latents(latents):
        """Apply 2x2 patch packing to latents."""
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        """Reverse 2x2 patch packing."""
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents):
        """Pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        """
        Using position ids to scatter tokens into place.

        Args:
            x: Latents tensor [B, seq_len, channels]
            x_ids: Position IDs [B, seq_len, 4]

        Returns:
            Unpacked latents [B, channels, height, width]
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # Reshape from (H * W, C) to (C, H, W)
            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def encode_prompt(
        self,
        prompt: Union[str, list[str]],
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (10, 20, 30),
    ):
        """
        Encode text prompt(s) to embeddings.

        Args:
            prompt: Text prompt(s)
            num_images_per_prompt: Images to generate per prompt
            prompt_embeds: Pre-computed embeddings (optional)
            max_sequence_length: Maximum token length
            text_encoder_out_layers: Layer indices to extract from Mistral3

        Returns:
            (prompt_embeds, text_ids): Embeddings and position IDs tensors
        """
        device = self.device

        if prompt_embeds is None:
            prompt_embeds = self._get_mistral_3_small_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                system_message=self.system_message,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)

        return prompt_embeds, text_ids

    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        """
        Prepare initial noisy latents.

        Args:
            batch_size: Number of samples
            num_latents_channels: Latent channels (128 for Flux2)
            height: Image height (pixels)
            width: Image width (pixels)
            dtype: Tensor dtype
            device: Tensor device
            generator: Random number generator for reproducibility
            latents: Pre-computed latents (optional)

        Returns:
            (latents, latent_ids): Initial latents and position IDs
        """
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        latents = self._pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
        return latents, latent_ids

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: Union[str, list[str]] = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        attention_kwargs: Optional[dict[str, Any]] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (10, 20, 30),
    ) -> DiffusionOutput:
        """
        Main generation method.

        Args:
            req: Request object (for metadata)
            prompt: Text prompt(s)
            height: Output height
            width: Output width
            num_inference_steps: Diffusion steps
            guidance_scale: Classifier-free guidance strength
            num_images_per_prompt: Images per prompt
            generator: RNG for reproducibility
            latents: Pre-computed latents (optional)
            prompt_embeds: Pre-computed embeddings (optional)
            output_type: Output format ("pil" or "latent")
            attention_kwargs: Additional attention kwargs
            max_sequence_length: Maximum text sequence length
            text_encoder_out_layers: Mistral3 layer indices to extract

        Returns:
            DiffusionOutput with generated images
        """
        device = self.device
        dtype = prompt_embeds.dtype if prompt_embeds is not None else torch.bfloat16

        # Extract parameters from request if available
        prompt = req.prompt if req.prompt is not None else prompt
        height = req.height or height or self.default_sample_size * self.vae_scale_factor
        width = req.width or width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = req.num_inference_steps or num_inference_steps
        generator = req.generator or generator
        guidance_scale = req.guidance_scale or guidance_scale
        req_num_outputs = getattr(req, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        # 1. Check inputs
        self.check_inputs(prompt, height, width, prompt_embeds)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        # 2. Determine batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0] if prompt_embeds is not None else 1

        # 3. Encode prompt
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        # 4. Prepare latents
        # Flux2 uses 128-channel latents, packed into 4 patches = 32 channels per patch
        num_channels_latents = self.transformer.in_channels // 4 if hasattr(self.transformer, "in_channels") else 32
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 5. Prepare timesteps
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 6. Handle guidance
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # 7. Diffusion loop
        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            if self._interrupt:
                continue

            self._current_timestep = t
            # Broadcast to batch dimension
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            latent_model_input = latents.to(self.transformer.x_embedder.weight.dtype)
            latent_image_ids = latent_ids

            # Predict noise/velocity
            noise_pred = self.transformer(
                hidden_states=latent_model_input,  # (B, image_seq_len, C)
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,  # B, text_seq_len, 4
                img_ids=latent_image_ids,  # B, image_seq_len, 4
                joint_attention_kwargs=self._attention_kwargs,
                return_dict=False,
            )[0]

            # Scheduler step
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # Some platforms (eg. apple mps) misbehave due to a pytorch bug
                    latents = latents.to(latents_dtype)

        self._current_timestep = None

        # 8. Decode latents
        if output_type == "latent":
            image = latents
        else:
            # Unpack latents using position IDs
            latents = self._unpack_latents_with_ids(latents, latent_ids)

            # Apply BatchNorm handling
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
            ).to(latents.device, latents.dtype)
            latents = latents * latents_bn_std + latents_bn_mean

            # Unpatchify
            latents = self._unpatchify_latents(latents)

            # VAE decode
            with torch.no_grad():
                image = self.vae.decode(latents, return_dict=False)[0]

            # Post-process
            image = self.image_processor.postprocess(image, output_type=output_type)

        return DiffusionOutput(output=image)

    def load_weights(self):
        """Load transformer weights."""
        self.load_transformer()

    def load_transformer(self):
        """Load transformer weights from checkpoint."""
        import glob

        # Define the weight iterator
        def weight_iterator(transformer_path):
            if not os.path.exists(transformer_path):
                logger.warning(f"Path {transformer_path} does not exist.")
                return

            # Look for safetensors first
            safetensors_files = glob.glob(os.path.join(transformer_path, "*.safetensors"))
            if safetensors_files:
                try:
                    from safetensors.torch import load_file
                except ImportError:
                    logger.warning("safetensors not installed, cannot load .safetensors files.")
                    return

                for file_path in safetensors_files:
                    state_dict = load_file(file_path)
                    for name, tensor in state_dict.items():
                        yield name, tensor
            else:
                # Fallback to bin
                bin_files = glob.glob(os.path.join(transformer_path, "*.bin"))
                for file_path in bin_files:
                    state_dict = torch.load(file_path)
                    for name, tensor in state_dict.items():
                        yield name, tensor

        try:
            # Get model path from config or download from HF
            model_name = self.od_config.model if hasattr(self, "od_config") else "black-forest-labs/FLUX.2-dev"
            if os.path.exists(model_name):
                model_path = model_name
            else:
                model_path = download_weights_from_hf_specific(model_name, None, ["*"])

            transformer_path = os.path.join(model_path, "transformer")
            self.transformer.load_weights(weight_iterator(transformer_path))
            logger.info("Loaded Flux2 transformer weights successfully")

        except Exception as e:
            logger.error(f"An error occurred loading transformer weights: {e}")
            raise e

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt


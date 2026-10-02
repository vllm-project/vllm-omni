# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native SANA-Video 2.0 T2V/TI2V pipeline for the official 50-step checkpoint."""

import math
from pathlib import Path

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from huggingface_hub import snapshot_download
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoTokenizer

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import DistributedAutoencoderKLLTX2Video
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import resolve_video_num_frames
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .conditioning import (
    DEFAULT_NEGATIVE_PROMPT,
    PROMPT_INSTRUCTION,
    denormalize_latents,
    encode_text,
    normalize_latents,
    prepare_image,
)
from .sampling import sample_flow_dpm, sample_ltx_euler
from .transformer_sana_video2 import SanaVideo2TransformerConfig, SanaVideo2TransformerModel

MODEL_ID = "Efficient-Large-Model/SANA-Video_2.0_5B_720p"
MODEL_REVISION = "f2d95fa06400f186fd1b077d12c69c8ac58aba48"
VAE_ID = "Efficient-Large-Model/LTX-2.3-Diffusers"
VAE_REVISION = "362acdf779d42e785fb26910c32254e9458e78c2"
TEXT_ENCODER_ID = "Efficient-Large-Model/gemma-2-2b-it"
TEXT_ENCODER_REVISION = "569d9809d0c8b6722d4d31b5a77a2ec7a400650a"


def validate_parallel_config(config):
    for name in (
        "tensor_parallel_size",
        "cfg_parallel_size",
        "pipeline_parallel_size",
        "text_encoder_tp_size",
        "vae_patch_parallel_size",
        "ring_degree",
        "allgather_degree",
    ):
        if (getattr(config.parallel_config, name, 1) or 1) != 1:
            raise ValueError(f"SANA-Video 2.0 currently requires {name}=1")
    parallel = config.parallel_config
    sp_size = parallel.sequence_parallel_size or 1
    if sp_size not in (1, 2, 4, 8):
        raise ValueError("SANA-Video 2.0 requires sequence_parallel_size in (1, 2, 4, 8)")
    if parallel.ulysses_degree != sp_size:
        raise ValueError("SANA-Video 2.0 requires ulysses_degree=sequence_parallel_size")
    if sp_size > 1 and parallel.ulysses_mode == "strict" and 10 % sp_size:
        raise ValueError("SANA-Video 2.0 has 10 softmax heads; use ulysses_mode='advanced_uaa' for SP4/SP8")
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("SANA-Video 2.0 quantization is not implemented")
    if config.parallel_config.use_hsdp:
        raise ValueError("SANA-Video 2.0 HSDP is not implemented")
    if config.cache_backend not in (None, "none"):
        raise ValueError("SANA-Video 2.0 cache support is not implemented")
    if any(
        getattr(config, key, False)
        for key in (
            "enable_cpu_offload",
            "enable_layerwise_offload",
            "enable_distributed_layerwise_offload",
        )
    ):
        raise ValueError("SANA-Video 2.0 offload is not yet supported")


def _image(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("SANA-Video 2.0 accepts exactly one conditioning image")
        value = value[0]
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raise TypeError("The conditioning image must be a PIL image or local file path")


def get_sana_video2_pre_process_func(od_config):
    def preprocess(request):
        if isinstance(request.prompt, dict):
            prompt = dict(request.prompt)
            media = dict(prompt.get("multi_modal_data") or {})
            if media.get("image") is not None:
                media["image"] = _image(media["image"])
            prompt["multi_modal_data"] = media
            request.prompt = prompt
        return request

    return preprocess


def get_sana_video2_post_process_func(od_config):
    processor = VideoProcessor()

    def postprocess(video, output_type="np", sampling_params=None):
        if output_type == "latent" or getattr(sampling_params, "output_type", None) == "latent":
            return video
        return {"payload": {"video": processor.postprocess_video(video, output_type=output_type)}, "metadata": {}}

    return postprocess


class SanaVideo2Pipeline(nn.Module, SupportImageInput, SupportsComponentDiscovery, ProgressBarMixin):
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules = ["vae"]
    supports_step_execution = False
    dummy_run_num_frames = 9
    default_num_inference_steps = 50

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        prefix="",
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=None,
        instruction=PROMPT_INSTRUCTION,
    ):
        super().__init__()
        self.weights_sources = []
        self.instruction = tuple(instruction)
        if od_config is not None:
            validate_parallel_config(od_config)
            tokenizer, text_encoder, vae, transformer = self._load_components(od_config)
        if any(component is None for component in (tokenizer, text_encoder, vae, transformer)):
            raise ValueError("SANA-Video 2.0 requires tokenizer, text encoder, LTX 2.3 VAE, and transformer")
        self.tokenizer, self.text_encoder, self.vae, self.transformer = tokenizer, text_encoder, vae, transformer
        self.sp_group = None
        if od_config is not None and (od_config.parallel_config.sequence_parallel_size or 1) > 1:
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            self.sp_group = get_sp_group()
            self.transformer.set_sequence_parallel(self.sp_group)
        if (
            vae.config.latent_channels,
            vae.config.temporal_compression_ratio,
            vae.config.spatial_compression_ratio,
        ) != (128, 8, 32):
            raise ValueError("SANA-Video 2.0 requires a 128-channel LTX VAE with stride (8,32,32)")
        self.eval().requires_grad_(False)

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    def _load_components(self, config):
        options = dict(config.model_config or {})
        allowed = {
            "vae_model",
            "vae_revision",
            "text_encoder_model",
            "text_encoder_revision",
            "tokenizer_model",
            "tokenizer_revision",
        }
        if set(options) - allowed:
            raise ValueError(f"Unknown SANA-Video 2.0 component options: {sorted(set(options) - allowed)}")
        root = Path(config.model)
        if not root.is_dir():
            root = Path(
                snapshot_download(
                    config.model,
                    revision=config.revision or (MODEL_REVISION if config.model == MODEL_ID else None),
                    allow_patterns=["config.yaml", "checkpoints/SANA_Video_2.0_5B_720p.pth"],
                )
            )
        model_config = SanaVideo2TransformerConfig.from_file(root / "config.yaml")
        device = torch.get_default_device()
        dtype = config.dtype
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("SANA-Video 2.0 requires float32 or bfloat16 inference")
        with torch.device("meta"):
            transformer = SanaVideo2TransformerModel(model_config)
        transformer.to_empty(device=device).to(dtype=dtype)
        self.checkpoint_report = transformer.load_checkpoint(root / "checkpoints/SANA_Video_2.0_5B_720p.pth")
        text_path = options.get("text_encoder_model", TEXT_ENCODER_ID)
        text_revision = options.get(
            "text_encoder_revision", TEXT_ENCODER_REVISION if text_path == TEXT_ENCODER_ID else None
        )
        tokenizer_path = options.get("tokenizer_model", text_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            revision=options.get("tokenizer_revision", text_revision if tokenizer_path == text_path else None),
        )
        text_encoder = AutoModel.from_pretrained(text_path, revision=text_revision, torch_dtype=torch.bfloat16).to(
            device
        )
        vae_path = options.get("vae_model", VAE_ID)
        vae_revision = options.get("vae_revision", VAE_REVISION if vae_path == VAE_ID else None)
        vae_config = DistributedAutoencoderKLLTX2Video.load_config(vae_path, subfolder="vae", revision=vae_revision)
        overrides = {}
        if "upsample_type" not in vae_config and vae_config.get("decoder_upsample_type") is not None:
            overrides["upsample_type"] = tuple(reversed(vae_config["decoder_upsample_type"]))
        vae = DistributedAutoencoderKLLTX2Video.from_pretrained(
            vae_path,
            subfolder="vae",
            revision=vae_revision,
            torch_dtype=torch.bfloat16,
            **overrides,
        ).to(device)
        return tokenizer, text_encoder, vae, transformer

    def load_weights(self, weights):
        # These components are fully loaded in the constructor: the model uses
        # upstream .pth rather than Diffusers' transformer shard layout.
        if next(iter(weights), None) is not None:
            raise ValueError("SANA-Video 2.0 components were already loaded from their explicit sources")
        return set(dict(self.named_parameters()))

    @torch.no_grad()
    def encode_prompt(
        self, prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT, guidance_scale=8.0, motion_score=10, high_motion=False
    ):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        suffix = (
            f" motion score: {int(motion_score)}."
            if motion_score > 0
            else ("" if motion_score < 0 else (" high motion" if high_motion else " low motion"))
        )
        positive, mask = encode_text(
            self.tokenizer,
            self.text_encoder,
            [p.strip() + suffix for p in prompts],
            instruction=self.instruction,
            device=self.device,
        )
        negative, negative_mask = None, None
        if guidance_scale > 1:
            negatives = [negative_prompt] * len(prompts) if isinstance(negative_prompt, str) else list(negative_prompt)
            if len(negatives) != len(prompts):
                raise ValueError("Negative and positive prompt batch sizes must match")
            negative, negative_mask = encode_text(
                self.tokenizer, self.text_encoder, negatives, instruction=(), device=self.device
            )
        return positive, mask, negative, negative_mask

    @staticmethod
    def check_inputs(height, width, num_frames, steps, guidance_scale, flow_shift):
        if any(type(v) is not int or v <= 0 or v % 32 for v in (height, width)):
            raise ValueError("Height and width must be positive multiples of 32")
        if type(num_frames) is not int or num_frames < 1 or (num_frames - 1) % 8:
            raise ValueError("num_frames must satisfy (num_frames - 1) % 8 == 0")
        if num_frames > 193 or height * width > 736 * 1280 or max(height, width) > 1280:
            raise ValueError("Requests must fit the release envelope: 193 frames, 736x1280 area, axes <=1280")
        if type(steps) is not int or steps < 2:
            raise ValueError("num_inference_steps must be at least 2; the 4-step preview is a separate model")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if not math.isfinite(flow_shift) or flow_shift <= 0:
            raise ValueError("flow_shift must be finite and positive")

    @torch.no_grad()
    def generate(
        self,
        prompt=None,
        *,
        image=None,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        height=736,
        width=1280,
        num_frames=193,
        num_inference_steps=50,
        guidance_scale=8.0,
        flow_shift=12.0,
        motion_score=10,
        high_motion=False,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        output_type="raw",
        callback=None,
    ):
        self.check_inputs(height, width, num_frames, num_inference_steps, guidance_scale, flow_shift)
        if output_type not in ("raw", "latent", "np", "pt", "pil"):
            raise ValueError(f"Unsupported output type: {output_type}")
        if prompt_embeds is None:
            if prompt is None:
                raise ValueError("A prompt or fixed prompt embeddings are required")
            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
                self.encode_prompt(
                    prompt,
                    negative_prompt,
                    guidance_scale,
                    motion_score,
                    high_motion,
                )
            )
        batch = prompt_embeds.shape[0]
        do_cfg = guidance_scale > 1
        if do_cfg and (negative_prompt_embeds is None or negative_prompt_attention_mask is None):
            raise ValueError("CFG requires negative prompt embeddings and masks")
        if prompt_attention_mask is None:
            raise ValueError("Prompt embeddings require a two-dimensional bool mask")
        embeddings = torch.cat([negative_prompt_embeds, prompt_embeds]) if do_cfg else prompt_embeds
        mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask]) if do_cfg else prompt_attention_mask
        embeddings, mask = embeddings.to(self.device), mask.to(self.device)
        shape = (batch, 128, (num_frames - 1) // 8 + 1, height // 32, width // 32)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=self.device, dtype=torch.float32)
        else:
            if tuple(latents.shape) != shape:
                raise ValueError(f"Expected latents with shape {shape}, got {tuple(latents.shape)}")
            latents = latents.to(device=self.device, dtype=torch.float32).clone()
        if image is not None:
            if batch != 1:
                raise ValueError("TI2V currently accepts a single image/prompt per request")
            pixels = prepare_image(_image(image), height, width)[None, :, None].to(self.device, self.vae.dtype)
            image_latents = normalize_latents(self.vae, self.vae.encode(pixels).latent_dist.mode())
            latents[:, :, :1] = image_latents

        if self.sp_group is not None:
            latents = self.sp_group.broadcast(latents.contiguous())
            embeddings = self.sp_group.broadcast(embeddings.contiguous())
            mask = self.sp_group.broadcast(mask.contiguous())

        def predict(x, time, noise_space=False):
            inputs = torch.cat([x, x]) if do_cfg else x
            if time.ndim == 0:
                timestep = (time * 1000).expand(inputs.shape[0])
            else:
                timestep = torch.cat([time, time]) if do_cfg else time
            prediction = self.transformer(inputs, timestep, embeddings, mask)
            if noise_space:
                # Upstream applies CFG after flow->noise conversion, not before.
                sigma = time.reshape((1,) * x.ndim).to(inputs)
                prediction = (1 - sigma) * prediction + inputs
            if do_cfg:
                uncond, cond = prediction.chunk(2)
                prediction = uncond + guidance_scale * (cond - uncond)
            return prediction

        if image is None:
            latents = sample_flow_dpm(
                lambda x, t: predict(x, t, True), latents, num_inference_steps, flow_shift, callback
            )
        else:
            latents = sample_ltx_euler(predict, latents, num_inference_steps, flow_shift, callback)
        if output_type == "latent":
            return latents
        decoded = denormalize_latents(self.vae, latents.to(self.vae.dtype))
        video = self.vae.decode(decoded, temb=None, return_dict=False)[0]
        return video if output_type == "raw" else VideoProcessor().postprocess_video(video, output_type=output_type)

    def forward(self, req: DiffusionRequestBatch):
        if len(req.prompts) != 1:
            raise ValueError("SANA-Video 2.0 accepts one prompt per request")
        prompt = req.prompts[0]
        data = {"prompt": prompt} if isinstance(prompt, str) else prompt
        sampling = req.sampling_params
        if sampling.timesteps is not None or sampling.sigmas is not None:
            raise ValueError("SANA-Video 2.0 requires the release timestep schedule")
        if (sampling.num_outputs_per_prompt or 1) != 1:
            raise ValueError("SANA-Video 2.0 currently generates one video per request")
        extra = dict(sampling.extra_args or {})
        if req.is_dummy_run():
            # Generic warmup supplies BAGEL guidance knobs to every pipeline.
            extra.pop("cfg_text_scale", None)
            extra.pop("cfg_img_scale", None)
        allowed = {"motion_score", "high_motion", "flow_shift"}
        if set(extra) - allowed:
            raise ValueError(f"Unsupported SANA-Video 2.0 request options: {sorted(set(extra) - allowed)}")
        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)
        steps = 50 if sampling.num_inference_steps is None else sampling.num_inference_steps
        if req.is_dummy_run():
            steps = max(2, steps)
        video = self.generate(
            data.get("prompt", ""),
            negative_prompt=(
                DEFAULT_NEGATIVE_PROMPT if data.get("negative_prompt") is None else data["negative_prompt"]
            ),
            image=(data.get("multi_modal_data") or {}).get("image"),
            height=736 if sampling.height is None else sampling.height,
            width=1280 if sampling.width is None else sampling.width,
            num_frames=resolve_video_num_frames(
                sampling.num_frames, default_num_frames=193, is_dummy_run=req.is_dummy_run()
            ),
            num_inference_steps=steps,
            guidance_scale=sampling.guidance_scale if sampling.guidance_scale_provided else 8.0,
            generator=generator,
            latents=sampling.latents,
            output_type="latent" if sampling.output_type == "latent" else "raw",
            **extra,
        )
        return DiffusionOutput(output=video)

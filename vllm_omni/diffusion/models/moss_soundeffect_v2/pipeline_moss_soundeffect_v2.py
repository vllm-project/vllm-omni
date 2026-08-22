import html
import json
import math
import os
import string
from collections.abc import Iterable
from typing import ClassVar

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .dac import DAC
from .moss_soundeffect_transformer import WanAudioModel, sinusoidal_embedding_1d

_DIT_PREFIX = "dit."
_HF_DIT_BLOCK_RENAME = {
    "attn1.norm_k.weight": "self_attn.norm_k.weight",
    "attn1.norm_q.weight": "self_attn.norm_q.weight",
    "attn1.to_k.bias": "self_attn.k.bias",
    "attn1.to_k.weight": "self_attn.k.weight",
    "attn1.to_out.0.bias": "self_attn.o.bias",
    "attn1.to_out.0.weight": "self_attn.o.weight",
    "attn1.to_q.bias": "self_attn.q.bias",
    "attn1.to_q.weight": "self_attn.q.weight",
    "attn1.to_v.bias": "self_attn.v.bias",
    "attn1.to_v.weight": "self_attn.v.weight",
    "attn2.norm_k.weight": "cross_attn.norm_k.weight",
    "attn2.norm_q.weight": "cross_attn.norm_q.weight",
    "attn2.to_k.bias": "cross_attn.k.bias",
    "attn2.to_k.weight": "cross_attn.k.weight",
    "attn2.to_out.0.bias": "cross_attn.o.bias",
    "attn2.to_out.0.weight": "cross_attn.o.weight",
    "attn2.to_q.bias": "cross_attn.q.bias",
    "attn2.to_q.weight": "cross_attn.q.weight",
    "attn2.to_v.bias": "cross_attn.v.bias",
    "attn2.to_v.weight": "cross_attn.v.weight",
    "ffn.net.0.proj.bias": "ffn.0.bias",
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.2.bias": "ffn.2.bias",
    "ffn.net.2.weight": "ffn.2.weight",
    "norm2.bias": "norm3.bias",
    "norm2.weight": "norm3.weight",
    "scale_shift_table": "modulation",
}


_HF_DIT_GLOBAL_RENAME = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "patch_embedding.weight": "patch_embedding.weight",
}


def _rename_dit_weight(name: str) -> str:
    relative_name = name.removeprefix(_DIT_PREFIX)
    if relative_name in _HF_DIT_GLOBAL_RENAME:
        return f"{_DIT_PREFIX}{_HF_DIT_GLOBAL_RENAME[relative_name]}"

    for source_suffix, target_suffix in _HF_DIT_BLOCK_RENAME.items():
        if relative_name.endswith(source_suffix):
            prefix = relative_name[: -len(source_suffix)]
            return f"{_DIT_PREFIX}{prefix}{target_suffix}"
    return name


def get_moss_soundeffect_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(
        audio: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return audio
        if output_type == "pt":
            return audio
        # Convert to numpy
        audio_np = audio.cpu().float().numpy()
        return audio_np

    return post_process_func


class Qwen3TextEncoder(nn.Module):
    """Wraps Qwen3 (decoder-only) as a text encoder for Wan audio pipeline.

    Loads the full Qwen3 model and extracts last-layer hidden states
    as text embeddings. Interface matches WanTextEncoder.forward(ids, mask).
    """

    def __init__(self, model_path, dtype=torch.bfloat16):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            output_hidden_states=True,
        )
        self.model.eval()
        self.dim = self.model.config.hidden_size  # 2048 for Qwen3-1.7B

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            ids:  [batch, seq_len] token ids
            mask: [batch, seq_len] attention mask (1=valid, 0=pad)
        Returns:
            hidden_states: [batch, seq_len, dim] last-layer hidden states
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
            )
        return outputs.hidden_states[-1]


class WanPrompter:
    def __init__(self, tokenizer_path, text_encoder, text_len=512):
        self.text_len = text_len
        self.text_encoder = text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @staticmethod
    def _basic_clean(text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    @classmethod
    def _whitespace_clean(cls, text: str) -> str:
        return re.sub(r"\s+", " ", cls._basic_clean(text)).strip()

    @classmethod
    def _canonicalize(cls, text: str, keep_punctuation_exact_string: str | None = None) -> str:
        text = cls._basic_clean(text).replace("_", " ")
        if keep_punctuation_exact_string:
            text = keep_punctuation_exact_string.join(
                part.translate(str.maketrans("", "", string.punctuation))
                for part in text.split(keep_punctuation_exact_string)
            )
        else:
            text = text.translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _tokenize(self, prompt: str | list[str]):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        prompts = [self._whitespace_clean(value) for value in prompts]
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_len,
            add_special_tokens=True,
        )

    def encode_prompt(self, prompt: str | list[str], positive: bool = True, device="cuda"):
        del positive
        tokens = self._tokenize(prompt)
        ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, seq_len in enumerate(seq_lens.tolist()):
            prompt_emb[i, seq_len:] = 0
        return prompt_emb


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, shift=None, dynamic_shift_len=None):
        if shift is not None:
            self.shift = shift
        # sigma is the noise strength at each step.
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            # Linear schedule from high noise to low noise.
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            # Classic flow-match shift formula.
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            # Last step: jump straight to the boundary.
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stabilized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stabilized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        # [B, len_timesteps] distance matrix.
        dists = (self.timesteps[None, :] - timestep[:, None]).abs()
        # [B] nearest timestep id per sample.
        timestep_ids = dists.argmin(dim=1)
        # [B] noise strength per sample.
        sigmas = self.sigmas[timestep_ids].to(original_samples.device)
        # Reshape for broadcasting to [B, C, T].
        sigmas = sigmas.view(-1, 1, 1)

        # x_t = (1 - sigma) * x_0 + sigma * eps
        sample = (1 - sigmas) * original_samples + sigmas * noise
        return sample

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu


class MossSoundEffectPipeline(torch.nn.Module, ProgressBarMixin):
    r"""Text-to-audio diffusion pipeline.

    Combines a Wan-based DiT, DAC VAE, Qwen3 text encoder, and flow-match
    scheduler in the vLLM-Omni diffusion pipeline interface.
    """

    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 48000
    supports_request_batch = False

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        sample_rate: int = 48000,
        max_inference_seconds: int = 30,
    ):
        super().__init__()
        self.od_config = od_config
        model = od_config.model
        if os.path.isdir(model):
            model_root = model
        else:
            from huggingface_hub import snapshot_download

            model_root = snapshot_download(
                repo_id=model,
                revision=od_config.revision,
            )
        self.device = get_local_device()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_root,
                subfolder="transformer",
                revision=None,
                prefix=_DIT_PREFIX,
                fall_back_to_pt=True,
            ),
        ]
        scheduler_path = os.path.join(model_root, "scheduler/scheduler_config.json")
        with open(scheduler_path) as f:
            sched_cfg = json.load(f)
        self.scheduler = FlowMatchScheduler(
            shift=sched_cfg.get("shift", 5.0),
            sigma_min=0.0,
            extra_one_step=True,
        )

        self.torch_dtype = torch.bfloat16
        te_path = os.path.join(model_root, "text_encoder")
        self.text_encoder = Qwen3TextEncoder(te_path, dtype=self.torch_dtype)
        self.text_encoder.to(self.device)

        tok_path = os.path.join(model_root, "tokenizer")
        self.prompter = WanPrompter(tokenizer_path=tok_path, text_encoder=self.text_encoder)

        with open(os.path.join(model_root, "transformer", "config.json")) as f:
            dit_cfg = json.load(f)
        self.dit = WanAudioModel(
            in_dim=dit_cfg["in_dim"],
            out_dim=dit_cfg["out_dim"],
            text_dim=dit_cfg["text_dim"],
            freq_dim=dit_cfg["freq_dim"],
            eps=dit_cfg["eps"],
            patch_size=tuple(dit_cfg["patch_size"]),
            has_image_input=dit_cfg["has_image_input"],
            dim=dit_cfg["dim"],
            ffn_dim=dit_cfg["ffn_dim"],
            num_heads=dit_cfg["num_heads"],
            num_layers=dit_cfg["num_layers"],
            vae_type=dit_cfg.get("vae_type", "dac"),
            quant_config=od_config.quantization_config,
            prefix=_DIT_PREFIX.rstrip("."),
        )
        index_path = os.path.join(model_root, "model_index.json")
        with open(index_path) as f:
            index = json.load(f)
        self.sample_rate = int(index.get("sample_rate", sample_rate))
        self.max_inference_seconds = int(index.get("max_inference_seconds", max_inference_seconds))
        self.vae = DAC.load(os.path.join(model_root, "vae/vae_128d_48k.pth")).to(self.device)
        self.audio_latent_dim = dit_cfg["in_dim"]
        self.num_samples_division_factor = self.vae.hop_length

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Run denoising for one request and return ``(B, C, T)`` audio.

        Args:
            req: Request batch containing the prompt and sampling parameters.
        """
        if len(req.requests) != 1:
            raise ValueError(f"MossSoundEffectPipeline expects one request, got {len(req.requests)}")

        request = req.requests[0]
        sampling_params = request.sampling_params
        extra_args = dict(sampling_params.extra_args or {})

        start_s = float(extra_args.get("audio_start_in_s", 0.0))
        end_s = round(float(extra_args.get("audio_end_in_s", 10.0)), 1)
        if start_s < 0:
            raise ValueError(f"start_s must be >= 0, got {start_s}")
        if end_s <= 0:
            raise ValueError(f"end_s must be > 0, got {end_s}")
        if end_s <= start_s:
            raise ValueError(f"end_s={end_s} must be greater than start_s={start_s}")
        if end_s > self.max_inference_seconds:
            raise ValueError(f"end_s={end_s} exceeds max_inference_seconds={self.max_inference_seconds}")

        prompt = request.prompt["prompt"]
        prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        prompts = [f"{value.strip()} duration: {end_s:.1f}s" for value in prompts]

        negative_prompt = request.prompt.get("negative_prompt", "")
        num_inference_steps = sampling_params.num_inference_steps or 10
        cfg_scale = sampling_params.guidance_scale if sampling_params.guidance_scale_provided else 5.0
        sigma_shift = float(extra_args.get("sigma_shift", 5.0))
        seed = sampling_params.seed

        num_samples_full = self.sample_rate * self.max_inference_seconds
        with torch.autocast(device_type=self.device.type, dtype=self.torch_dtype):
            audio = self._generate_audio(
                prompt=prompts if len(prompts) > 1 else prompts[0],
                negative_prompt=negative_prompt,
                seed=seed,
                cfg_scale=float(cfg_scale),
                sigma_shift=sigma_shift,
                num_inference_steps=int(num_inference_steps),
                num_samples=num_samples_full,
                num_channels=1,
            )
        start_samples = int(self.sample_rate * start_s)
        end_samples = int(self.sample_rate * end_s)
        audio = audio[:, :, start_samples:end_samples]
        return DiffusionOutput(output=audio)

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        return super().to(*args, **kwargs)

    def _check_audio_shape(self, num_channels: int, num_samples: int) -> tuple[int, int]:
        self.num_samples_division_factor = int(np.prod(self.vae.encoder_rates))
        if num_samples % self.num_samples_division_factor != 0:
            num_samples = num_samples // self.num_samples_division_factor * self.num_samples_division_factor
        return num_channels, num_samples

    def _initialize_noise(
        self,
        num_samples: int,
        batch_size: int,
        seed: int | None,
        rand_device: str = "cpu",
    ) -> torch.Tensor:
        shape = (batch_size, self.audio_latent_dim, num_samples // self.num_samples_division_factor)
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=torch.float32)
        return noise.to(dtype=self.torch_dtype, device=self.device)

    def _encode_prompt(self, prompt: str | list[str] | None, positive: bool) -> torch.Tensor:
        return self.prompter.encode_prompt(prompt, positive=positive, device=self.device)

    def _decode_audio(self, latents: torch.Tensor, max_batch_size: int = 8) -> torch.Tensor:
        audio_chunks = []
        for start in range(0, latents.size(0), max_batch_size):
            end = min(start + max_batch_size, latents.size(0))
            with torch.autocast("cuda", dtype=torch.float32):
                audio_chunks.append(self.vae.decode(latents[start:end]))
        return torch.cat(audio_chunks, dim=0)

    @torch.no_grad()
    def _generate_audio(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = "",
        denoising_strength: float = 1.0,
        seed: int | None = None,
        rand_device: str = "cpu",
        num_samples=44100 * 10,
        num_channels=2,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 10,
        sigma_shift: float = 5.0,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        _num_channels, num_samples = self._check_audio_shape(
            num_channels=num_channels,
            num_samples=num_samples,
        )
        batch_size = len(prompt) if isinstance(prompt, (list, tuple)) else 1
        noise = self._initialize_noise(
            num_samples=num_samples, batch_size=batch_size, seed=seed, rand_device=rand_device
        )
        latents = noise

        context = self._encode_prompt(prompt=prompt, positive=True)
        negative_context = None
        if cfg_scale != 1.0:
            if batch_size > 1 and not isinstance(negative_prompt, (list, tuple)):
                negative_prompt = [negative_prompt] * batch_size
            negative_context = self._encode_prompt(prompt=negative_prompt, positive=False)

        timesteps = self.scheduler.timesteps.to(device=self.device)
        with self.progress_bar(total=len(self.scheduler.timesteps)) as pbar:
            # Denoise
            for i, timestep in enumerate(timesteps):
                timestep = timestep.unsqueeze(0)

                # Inference
                noise_pred_posi = self._denoise_step(latents=latents, context=context, timestep=timestep)
                if cfg_scale != 1.0:
                    noise_pred_nega = self._denoise_step(
                        latents=latents,
                        context=negative_context,
                        timestep=timestep,
                    )
                    noise_pred_posi = noise_pred_posi.float()
                    noise_pred_nega = noise_pred_nega.float()
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[i], latents)
                pbar.update()

        return self._decode_audio(latents)

    def _denoise_step(
        self,
        latents: torch.Tensor = None,
        timestep: torch.Tensor = None,
        context: torch.Tensor = None,
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        reference_latents=None,
        control_camera_latents_input=None,
        **kwargs,
    ):
        dit = self.dit
        with torch.autocast("cuda", dtype=torch.float32):
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

        context = dit.text_embedding(context)

        x = latents
        # Merged cfg
        if x.shape[0] != context.shape[0]:
            x = torch.concat([x] * context.shape[0], dim=0)
        if timestep.shape[0] != context.shape[0]:
            timestep = torch.concat([timestep] * context.shape[0], dim=0)

        # Image Embedding
        if y is not None and dit.require_vae_embedding:
            x = torch.cat([x, y], dim=1)
        if clip_feature is not None and dit.require_clip_embedding:
            clip_embedding = dit.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # Add camera control
        x, (f,) = dit.patchify(x, control_camera_latents_input)

        # Reference image
        if reference_latents is not None:
            if len(reference_latents.shape) == 5:
                reference_latents = reference_latents[:, :, 0]
            reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
            x = torch.concat([reference_latents, x], dim=1)
            f += 1

        # Keep RoPE caches real-valued for the fused rotary kernel.
        cos = dit.rope_cos_cache[:f].reshape(f, 1, -1)
        sin = dit.rope_sin_cache[:f].reshape(f, 1, -1)

        for block in dit.blocks:
            x = block(x, context, t_mod, (cos, sin))

        x = dit.head(x, t)

        # Remove reference latents
        if reference_latents is not None:
            x = x[:, reference_latents.shape[1] :]
            f -= 1
        x = dit.unpatchify(x, (f,))
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.dit.named_parameters())
        loaded_params: set[str] = set()
        stacked_params_mapping = (
            (".self_attn.to_qkv", ".self_attn.q", "q"),
            (".self_attn.to_qkv", ".self_attn.k", "k"),
            (".self_attn.to_qkv", ".self_attn.v", "v"),
        )
        merged_params_mapping = (
            (".cross_attn.kv", ".cross_attn.k", 0),
            (".cross_attn.kv", ".cross_attn.v", 1),
        )
        for name, loaded_weight in weights:
            full_param_name = _rename_dit_weight(name)
            param_name = full_param_name.removeprefix(_DIT_PREFIX)
            for fused_name, source_name, shard_id in stacked_params_mapping:
                if source_name not in param_name:
                    continue
                fused_param_name = param_name.replace(source_name, fused_name)
                if fused_param_name not in params_dict:
                    continue
                param = params_dict[fused_param_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(f"{_DIT_PREFIX}{fused_param_name}")
                break
            else:
                for merged_name, source_name, shard_id in merged_params_mapping:
                    if source_name not in param_name:
                        continue
                    merged_param_name = param_name.replace(source_name, merged_name)
                    if merged_param_name not in params_dict:
                        continue
                    param = params_dict[merged_param_name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(f"{_DIT_PREFIX}{merged_param_name}")
                    break
                else:
                    if param_name not in params_dict:
                        continue
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(full_param_name)
        return loaded_params

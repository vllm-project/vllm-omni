# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from LongCat-AudioDiT (https://github.com/meituan-longcat/LongCat-AudioDiT)

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import UMT5Config, UMT5EncoderModel

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.models.longcat_audio_dit.longcat_audio_dit_transformer import LongCatAudioDiTTransformer
from vllm_omni.diffusion.models.longcat_audio_dit.longcat_audio_dit_vae import LongCatAudioDiTVae
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs


def _lens_to_mask(lengths: torch.Tensor, length: int | None = None) -> torch.BoolTensor:
    if length is None:
        length = lengths.amax()
    seq = torch.arange(length, device=lengths.device)
    return seq[None, :] < lengths[:, None]


def _odeint_euler(fn, y0, t):
    ys = [y0]
    y = y0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        y = y + fn(t[i], y) * dt
        ys.append(y)
    return torch.stack(ys)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'["\u201c\u201d\u2018\u2019]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _approx_duration_from_text(text: str, max_duration: float = 30.0) -> float:
    EN_DUR_PER_CHAR = 0.082
    ZH_DUR_PER_CHAR = 0.21
    text = re.sub(r"\s+", "", text)
    num_zh = num_en = num_other = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            num_zh += 1
        elif c.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, num_zh * ZH_DUR_PER_CHAR + num_en * EN_DUR_PER_CHAR)


def _approx_batch_duration_from_prompts(prompts: list[str]) -> float:
    if not prompts:
        return 0.0
    return max(_approx_duration_from_text(prompt) for prompt in prompts)


class _MomentumBuffer:
    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def _project(v0: torch.Tensor, v1: torch.Tensor, dims=(-1, -2)):
    dtype = v0.dtype
    device = v0.device
    if device.type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()
    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device), v0_orthogonal.to(dtype).to(device)


def _apg_forward(
    pred_cond, pred_uncond, guidance_scale, momentum_buffer=None, eta=0.0, norm_threshold=2.5, dims=(-1, -2)
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = _project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return pred_cond + guidance_scale * normalized_update


def get_longcat_audio_dit_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return audio
        if output_type == "pt":
            return audio
        audio_np = audio.cpu().float().numpy()
        return audio_np

    return post_process_func


class LongCatAudioDiTPipeline(nn.Module, SupportAudioOutput, DiffusionPipelineProfilerMixin):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        # LongCat-AudioDiT has all weights in root (model.safetensors), not in transformer/ subfolder
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=None,
                prefix="",
                fall_back_to_pt=True,
            ),
        ]

        # Build UMT5 text encoder from config.json text_encoder_config.
        text_encoder = None
        if hasattr(od_config, "tf_model_config") and od_config.tf_model_config is not None:
            te_config_dict = getattr(od_config.tf_model_config, "text_encoder_config", None)
            if te_config_dict:
                te_config = UMT5Config.from_dict(te_config_dict)
                text_encoder = UMT5EncoderModel(te_config)

        if text_encoder is None:
            try:
                te_config = UMT5Config.from_pretrained("google/umt5-base")
                text_encoder = UMT5EncoderModel(te_config)
            except Exception:
                text_encoder = None

        self.text_encoder = text_encoder
        if self.text_encoder is not None:
            self.text_encoder.eval()

        self._tokenizer = None
        try:
            from transformers import AutoTokenizer

            text_encoder_model = getattr(od_config.tf_model_config, "text_encoder_model", "google/umt5-base")
            self._tokenizer = AutoTokenizer.from_pretrained(text_encoder_model)
        except Exception:
            pass

        tf_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, LongCatAudioDiTTransformer)

        self.transformer = LongCatAudioDiTTransformer(
            dit_dim=tf_kwargs.get("dit_dim", 1536),
            dit_depth=tf_kwargs.get("dit_depth", 24),
            dit_heads=tf_kwargs.get("dit_heads", 24),
            dit_text_dim=tf_kwargs.get("dit_text_dim", 768),
            latent_dim=tf_kwargs.get("latent_dim", 64),
            dropout=tf_kwargs.get("dit_dropout", 0.0),
            bias=tf_kwargs.get("dit_bias", True),
            cross_attn=tf_kwargs.get("dit_cross_attn", True),
            adaln_type=tf_kwargs.get("dit_adaln_type", "global"),
            adaln_use_text_cond=tf_kwargs.get("dit_adaln_use_text_cond", True),
            long_skip=tf_kwargs.get("dit_long_skip", True),
            text_conv=tf_kwargs.get("dit_text_conv", True),
            qk_norm=tf_kwargs.get("dit_qk_norm", True),
            cross_attn_norm=tf_kwargs.get("dit_cross_attn_norm", False),
            eps=tf_kwargs.get("dit_eps", 1e-6),
            use_latent_condition=tf_kwargs.get("dit_use_latent_condition", True),
        )

        vae_config = dict(getattr(od_config.tf_model_config, "vae_config", {}) or {})
        vae_config.pop("model_type", None)
        self.vae = LongCatAudioDiTVae(
            in_channels=vae_config.get("in_channels", 1),
            channels=vae_config.get("channels", 128),
            c_mults=vae_config.get("c_mults", [1, 2, 4, 8, 16]),
            strides=vae_config.get("strides", [2, 4, 4, 8, 8]),
            latent_dim=vae_config.get("latent_dim", tf_kwargs.get("latent_dim", 64)),
            encoder_latent_dim=vae_config.get("encoder_latent_dim", 128),
            use_snake=vae_config.get("use_snake", True),
            downsample_shortcut=vae_config.get("downsample_shortcut", "averaging"),
            upsample_shortcut=vae_config.get("upsample_shortcut", "duplicating"),
            out_shortcut=vae_config.get("out_shortcut", "averaging"),
            in_shortcut=vae_config.get("in_shortcut", "duplicating"),
            final_tanh=vae_config.get("final_tanh", False),
            downsampling_ratio=vae_config.get("downsampling_ratio", 2048),
            sample_rate=vae_config.get("sample_rate", 24000),
            scale=vae_config.get("scale", 0.71),
        )
        self.vae.to_half()

        self.sample_rate = getattr(self.vae, "sample_rate", vae_config.get("sample_rate", 24000))
        self.latent_hop = getattr(self.vae, "downsampling_ratio", vae_config.get("downsampling_ratio", 2048))
        self.latent_dim = tf_kwargs.get("latent_dim", vae_config.get("latent_dim", 64))
        self.max_wav_duration = 30.0
        self.repa_layer = tf_kwargs.get("repa_dit_layer", 8)

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def encode_text(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        if self.text_encoder is None:
            return torch.zeros(input_ids.shape[0], input_ids.shape[1], 768, device=input_ids.device)

        # Read config for text encoding behavior
        text_norm_feat = getattr(self.od_config.tf_model_config, "text_norm_feat", True)
        text_add_embed = getattr(self.od_config.tf_model_config, "text_add_embed", True)

        with torch.no_grad():
            output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        emb = output.last_hidden_state

        d_model = self.text_encoder.config.d_model

        if text_norm_feat:
            emb = F.layer_norm(emb, (d_model,), eps=1e-6)

        if text_add_embed:
            first_hidden = output.hidden_states[0]
            if text_norm_feat:
                first_hidden = F.layer_norm(first_hidden, (d_model,), eps=1e-6)
            emb = emb + first_hidden

        return emb.float()

    def encode_prompt_audio(self, prompt_audio: torch.FloatTensor) -> tuple[torch.FloatTensor, int]:
        full_hop = self.latent_hop
        off = 3
        wav = prompt_audio.to(self.device)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        if wav.shape[-1] % full_hop != 0:
            wav = F.pad(wav, (0, full_hop - wav.shape[-1] % full_hop))
        wav = F.pad(wav, (0, full_hop * off))
        latent = self.vae.encode(wav)
        if off != 0:
            latent = latent[..., :-off]
        prompt_duration_frames = latent.shape[-1]
        return latent.permute(0, 2, 1), prompt_duration_frames

    def prepare_latents(self, batch_size: int, duration: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y0 = []
        for _ in range(batch_size):
            noise = torch.randn(duration, self.latent_dim, device=device)
            y0.append(noise)
        return pad_sequence(y0, padding_value=0, batch_first=True)

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        duration: int | None = None,
        num_inference_steps: int = 16,
        guidance_scale: float = 4.0,
        guidance_method: str = "cfg",
        prompt_audio: torch.Tensor | None = None,
        prompt_text: str | None = None,
        output_type: str = "np",
    ) -> DiffusionOutput:
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if prompt is None:
            prompt = []
        elif isinstance(prompt, str):
            prompt = [prompt]
        else:
            prompt = list(prompt)

        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        device = self.device
        batch_size = len(prompt)

        # Normalize text (matching official implementation)
        normalized_prompts = [_normalize_text(p) for p in prompt]

        # Determine duration: prefer explicit audio_end_in_s, else estimate from text
        if duration is None:
            audio_end_in_s = req.sampling_params.extra_args.get("audio_end_in_s")
            if audio_end_in_s is not None:
                duration = int(audio_end_in_s * self.sample_rate // self.latent_hop)
            else:
                est_duration_s = _approx_batch_duration_from_prompts(normalized_prompts)
                duration = int(est_duration_s * self.sample_rate // self.latent_hop)

        duration = min(duration, int(self.max_wav_duration * self.sample_rate // self.latent_hop))

        duration_tensor = torch.full((batch_size,), duration, device=device, dtype=torch.long)

        if self.text_encoder is not None and prompt_text is None:
            if self._tokenizer is not None:
                text_inputs = self._tokenizer(normalized_prompts, padding="longest", return_tensors="pt")
                input_ids = text_inputs.input_ids.to(device)
                attention_mask = text_inputs.attention_mask.to(device)
                text_condition = self.encode_text(input_ids, attention_mask)
                text_condition_len = attention_mask.sum(dim=1).to(device)
            else:
                text_condition = torch.zeros(batch_size, 1, 768, device=device)
                text_condition_len = torch.ones(batch_size, device=device, dtype=torch.long)
        elif prompt_text is not None:
            if self._tokenizer is not None:
                text_inputs = self._tokenizer([prompt_text], padding="longest", return_tensors="pt")
                input_ids = text_inputs.input_ids.to(device)
                attention_mask = text_inputs.attention_mask.to(device)
                text_condition = self.encode_text(input_ids, attention_mask)
                text_condition_len = attention_mask.sum(dim=1).to(device)
            else:
                text_condition = torch.zeros(batch_size, 1, 768, device=device)
                text_condition_len = torch.ones(batch_size, device=device, dtype=torch.long)
        else:
            text_condition = torch.zeros(batch_size, 1, 768, device=device)
            text_condition_len = torch.ones(batch_size, device=device, dtype=torch.long)

        if prompt_audio is not None:
            prompt_latent, prompt_dur = self.encode_prompt_audio(prompt_audio)
        else:
            prompt_latent = torch.empty(batch_size, 0, self.latent_dim, device=device)
            prompt_dur = 0

        mask = _lens_to_mask(duration_tensor)
        text_mask = _lens_to_mask(text_condition_len, length=text_condition.shape[1])

        neg_text = torch.zeros_like(text_condition)
        neg_text_len = text_condition_len

        latent_len = prompt_dur
        if prompt_audio is not None:
            gen_len = duration - latent_len
            latent_cond = F.pad(prompt_latent, (0, 0, 0, gen_len))
            empty_latent_cond = torch.zeros_like(latent_cond)
        else:
            latent_cond = torch.zeros(batch_size, duration, self.latent_dim, device=device)
            empty_latent_cond = latent_cond

        if guidance_method == "apg":
            apg_buffer = _MomentumBuffer(momentum=-0.3)
        else:
            apg_buffer = None

        y0 = self.prepare_latents(batch_size, duration, device, text_condition.dtype)
        t = torch.linspace(0, 1, num_inference_steps, device=device)
        prompt_noise = y0[:, :latent_len].clone()

        def fn(t, x):
            x[:, :latent_len] = prompt_noise * (1 - t) + latent_cond[:, :latent_len] * t

            output = self.transformer(
                x=x,
                text=text_condition,
                text_len=text_condition_len,
                time=t,
                mask=mask,
                cond_mask=text_mask,
                latent_cond=latent_cond,
                return_ith_layer=self.repa_layer,
            )
            pred = output["last_hidden_state"]

            if guidance_scale < 1e-5:
                return pred

            x[:, :latent_len] = 0
            null_output = self.transformer(
                x=x,
                text=neg_text,
                text_len=neg_text_len,
                time=t,
                mask=mask,
                cond_mask=text_mask,
                latent_cond=empty_latent_cond,
                return_ith_layer=self.repa_layer,
            )
            null_pred = null_output["last_hidden_state"]

            if guidance_method == "cfg":
                return pred + (pred - null_pred) * guidance_scale

            x_s = x[:, latent_len:]
            pred_s = pred[:, latent_len:]
            null_s = null_pred[:, latent_len:]
            pred_sample = x_s + (1 - t) * pred_s
            null_sample = x_s + (1 - t) * null_s
            out = _apg_forward(
                pred_sample, null_sample, guidance_scale, apg_buffer, eta=0.5, norm_threshold=0.0, dims=[-1, -2]
            )
            out = (out - x_s) / (1 - t)
            return F.pad(out, (0, 0, latent_len, 0), value=0.0)

        trajectory = _odeint_euler(fn, y0, t)
        sampled = trajectory[-1]

        pred_latent = sampled
        if prompt_audio is not None:
            pred_latent = pred_latent[:, prompt_dur:]

        pred_latent = pred_latent.permute(0, 2, 1).float()
        waveform = self.vae.decode(pred_latent).squeeze(1).detach()

        return DiffusionOutput(
            output=waveform,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state = self.state_dict()
        shapes = {k: tuple(v.shape) for k, v in state.items()}

        def _try_match(name: str):
            if name in shapes:
                return name
            for prefix in ["transformer.", "vae.", "text_encoder."]:
                if name.startswith(prefix):
                    stripped = name[len(prefix) :]
                    if stripped in shapes:
                        return stripped
            return None

        loaded = set()

        for name, tensor in weights:
            matched = _try_match(name)
            if matched and shapes[matched] == tuple(tensor.shape):
                param = state[matched]
                if hasattr(param, "weight_loader"):
                    param.weight_loader(param, tensor)
                else:
                    param.copy_(tensor)
                loaded.add(matched)

        # Handle shared.weight -> embed_tokens.weight (T5 tie)
        if "shared.weight" not in loaded and "text_encoder.shared.weight" not in loaded:
            embed_key = "text_encoder.encoder.embed_tokens.weight"
            if embed_key in shapes:
                shared_key = "text_encoder.shared.weight"
                if shared_key in shapes:
                    param = state[shared_key]
                    embed_param = state[embed_key]
                    if hasattr(param, "weight_loader"):
                        param.weight_loader(param, embed_param)
                    else:
                        param.copy_(embed_param)
                    loaded.add(shared_key)

        return loaded

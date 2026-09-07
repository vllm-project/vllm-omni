# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/dit.py
#
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/modules.py
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/cfm.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Partial of the following source code
# is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import os

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase, Qwen2Config
from vllm.logger import init_logger
from x_transformers.x_transformers import RotaryEmbedding

from vllm_omni.model_executor.layers.timestep_embedding import DiTTimestepEmbedding
from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE, AudioVAEConfig
from vllm_omni.model_executor.models.common.ming.cfm_solver import (
    apply_sway_sampling,
    get_epss_timesteps,
    integrate_cfm_steps,
)
from vllm_omni.model_executor.models.common.ming.dit_blocks import CondEmbedder, DiTBlock, FinalLayer

logger = init_logger(__name__)


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone for audio latent generation."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 1024,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_cond_dim: int = 896,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.t_embedder = DiTTimestepEmbedding(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size)
        if "spk_dim" in kwargs:
            self.spk_embedder = nn.Linear(kwargs["spk_dim"], hidden_size)
        else:
            self.spk_embedder = None
        self.hidden_size = hidden_size

        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([latent_history, x], dim=1)
        x = self.x_embedder(x)
        t = self.t_embedder(t).unsqueeze(1)
        c = self.c_embedder(c)
        y = t + c
        if spk_emb is None:
            assert self.spk_embedder is None
            x = torch.cat([y, x], dim=1)
        else:
            x = torch.cat([self.spk_embedder(spk_emb), y, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])

        for block in self.blocks:
            x = block(x, None, rope)
        x = self.final_layer(x)
        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with classifier-free guidance (doubles batch for CFG)."""
        x = torch.cat([x, x], dim=0)
        latent_history = torch.cat([latent_history, latent_history], dim=0)
        fake_latent = torch.zeros_like(c)
        c = torch.cat([c, fake_latent], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        if spk_emb is not None:
            spk_emb = torch.cat([spk_emb, spk_emb], dim=0)
        model_out = self.forward(x, t, c, latent_history, spk_emb)
        return model_out[:, -x.shape[1] :, :]


class CFM(nn.Module):
    """Conditional Flow Matching module for audio latent generation."""

    def __init__(self, model: nn.Module, steps: int = 10, sway_sampling_coef: float | None = -1.0):
        """
        Args:
            model: DiT used for the velocity prediction.
            steps: number of integration steps per sample call.
            sway_sampling_coef: coefficient used to skew the integration
                grid towards low-noise timesteps. Defaults to -1.0 which
                packs more steps near t=0, where prediction error is highest.
                Set to `None` to use the linear grid as-is.
        """
        super().__init__()
        self.model = model
        self.steps = steps
        self.sway_sampling_coef = sway_sampling_coef

    @torch.no_grad()
    def sample(
        self,
        llm_cond: torch.Tensor,
        lat_cond: torch.Tensor,
        y0: torch.Tensor,
        t: torch.Tensor,
        sde_args: torch.Tensor,
        sde_rnd: torch.Tensor,
    ):
        """Sample audio latent via ODE/SDE integration with CFG.

        Args:
            llm_cond: LLM hidden state (B, 1, hidden_size)
            lat_cond: latent history (B, his_patch_size, latent_dim)
            y0: initial noise (B, patch_size, latent_dim)
            t: timesteps from get_epss_timesteps
            sde_args: [cfg_strength, sigma, temperature]
            sde_rnd: random noise for SDE steps (steps, B, patch_size, latent_dim)
        """

        def fn(fn_t, x):
            pred_cfg = self.model.forward_with_cfg(x, fn_t, llm_cond, lat_cond, None)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * sde_args[0]

        t = apply_sway_sampling(t, self.sway_sampling_coef)
        return integrate_cfm_steps(fn, y0, t, sde_args, sde_rnd, self.steps)


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a waveform via linear interpolation (no torchaudio dep).

    Args:
        waveform: Tensor shaped ``(..., num_samples)``.
        orig_sr: Source sample rate (Hz); must be > 0.
        target_sr: Target sample rate (Hz); must be > 0.

    Raises:
        ValueError: If sample rates are non-positive, the waveform is empty,
            or the resampled length would round to zero.
    """
    if orig_sr <= 0:
        raise ValueError(f"orig_sr must be positive, got {orig_sr}")
    if target_sr <= 0:
        raise ValueError(f"target_sr must be positive, got {target_sr}")
    if waveform.numel() == 0 or waveform.shape[-1] == 0:
        raise ValueError("waveform must contain at least one sample")
    if orig_sr == target_sr:
        return waveform

    ratio = target_sr / orig_sr
    new_len = int(waveform.shape[-1] * ratio)
    if new_len <= 0:
        raise ValueError(
            f"resampled waveform would be empty for input length {waveform.shape[-1]}, "
            f"orig_sr={orig_sr}, target_sr={target_sr}"
        )
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def trim_trailing_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    sil_th: float = 1e-3,
    tail_silence_s: float = 0.3,
) -> torch.Tensor:
    """Trim low-energy tail while keeping a short trailing silence.

    Works on 2-D ``(channels, samples)`` or 3-D ``(batch, channels, samples)``
    tensors. Any other shape is returned unchanged.
    """
    if waveform.numel() == 0:
        return waveform

    original_dim = waveform.dim()
    if original_dim == 3:
        speech = waveform[:, 0, :]
    elif original_dim == 2:
        speech = waveform
    else:
        return waveform

    frame_step = int(sample_rate * 0.1)
    frame_size = int(sample_rate * 0.1)
    if speech.shape[-1] < frame_size:
        keep = min(speech.shape[-1], int(tail_silence_s * sample_rate))
        trimmed = speech[..., :keep]
    else:
        num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
        cur_len = (num_frame - 1) * frame_step + frame_size
        speech = speech[..., :cur_len]
        spe_frames = speech.unfold(-1, frame_size, frame_step)
        scores = spe_frames.abs().mean(dim=-1)
        scores = scores.mean(dim=list(range(scores.dim() - 1)))
        idx = scores.shape[0] - 1
        while idx >= 0 and scores[idx] <= sil_th:
            idx -= 1
        if idx < 0:
            keep = min(speech.shape[-1], int(tail_silence_s * sample_rate))
            trimmed = speech[..., :keep]
        else:
            non_sil_len = idx * frame_step + frame_size + int(tail_silence_s * sample_rate)
            non_sil_len = min(non_sil_len, speech.shape[-1])
            trimmed = speech[..., :non_sil_len]

    if original_dim == 3:
        return trimmed.unsqueeze(1)
    return trimmed


def silence_holder(
    speech: torch.Tensor,
    sample_rate: int,
    sil_cache: dict | None = None,
    last_chunk: bool = True,
    sil_th: float = 1e-3,
    last_sil: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """Ming-style streaming silence holder.

    Used during streaming VAE decode to defer emission of silent regions
    until a non-silent frame arrives (or the stream ends). ``sil_cache``
    is carried across chunks and updated in place.
    """
    if speech.numel() == 0:
        return speech, sil_cache or {"holder": [], "buffer": []}

    frame_step = int(sample_rate * 0.1)
    frame_size = int(sample_rate * 0.1)
    if sil_cache is None:
        sil_cache = {"holder": [], "buffer": []}

    if sil_cache["buffer"]:
        speech = torch.cat([*sil_cache["buffer"], speech], dim=-1)
        sil_cache["buffer"] = []

    if speech.shape[-1] < frame_size:
        sil_cache["buffer"].append(speech)
        if last_chunk:
            speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
            return speech[..., : int(last_sil * sample_rate)], sil_cache
        return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

    num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
    cur_len = (num_frame - 1) * frame_step + frame_size
    if speech.shape[-1] > cur_len:
        sil_cache["buffer"].append(speech[..., cur_len:])
        speech = speech[..., :cur_len]

    spe_frames = speech.unfold(-1, frame_size, frame_step)
    scores = spe_frames.abs().mean(dim=-1)
    scores = scores.mean(dim=list(range(scores.dim() - 1)))
    idx = scores.shape[0] - 1
    while idx >= 0 and scores[idx] <= sil_th:
        idx -= 1

    if idx < 0:
        sil_cache["holder"].append(speech)
        if last_chunk:
            speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
            return speech[..., : int(last_sil * sample_rate)], sil_cache
        return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

    non_sil_len = idx * frame_step + frame_size
    if last_chunk:
        non_sil_len += int(last_sil * sample_rate)
    non_sil_len = min(non_sil_len, speech.shape[-1])
    speech_out = torch.cat([*sil_cache["holder"], speech[..., :non_sil_len]], dim=-1)
    sil_cache["holder"] = []
    if non_sil_len < speech.shape[-1]:
        sil_cache["holder"].append(speech[..., non_sil_len:])
    return speech_out, sil_cache


########################################################################
# Prompt Builder
# Adapted from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py
########################################################################

_MUSIC_TAGS = ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")


def _looks_like_music_prompt(text: str) -> bool:
    return all(tag in text for tag in _MUSIC_TAGS)


def build_tts_input(
    *,
    tokenizer: PreTrainedTokenizerBase,
    embed_tokens: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    text: str,
    prompt: str,
    spk_emb: list[torch.Tensor] | None = None,
    instruction: str | None = None,
    prompt_text: str | None = None,
    prompt_wav_emb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (inputs_embeds, input_ids) for one TTS segment.

    Args:
        tokenizer: HF tokenizer
        embed_tokens: The LLM's input-embedding module
        device: Device to place the returned tensors on.
        dtype: dtype for the returned `inputs_embeds`.
        text: Text to synthesize.
        prompt: System-level instruction prompt prepended to the user turn.
        spk_emb: Optional list of speaker embeddings already projected into
            LLM hidden dim; each is injected at a `<|vision_start|>` slot.
        instruction: Optional free-form instruction
        prompt_text: Reference text for zero-shot voice cloning.
        prompt_wav_emb: Reference-wav embeddings to inject.
    """
    input_part = build_tts_prompt_token_ids(
        tokenizer=tokenizer,
        text=text,
        prompt=prompt,
        spk_emb_count=(len(spk_emb) if spk_emb is not None else 0),
        instruction=instruction,
        prompt_text=prompt_text,
        prompt_wav_len=(int(prompt_wav_emb.size(1)) if prompt_wav_emb is not None else 0),
    )

    input_ids = torch.tensor(input_part, dtype=torch.long, device=device).unsqueeze(0)
    inputs_embeds = embed_tokens(input_ids).to(device=device, dtype=dtype)

    # inject speaker embeddings
    if spk_emb is not None:
        spk_token_id = tokenizer.encode("<|vision_start|>")
        assert len(spk_token_id) == 1, "<|vision_start|> must tokenize to a single id"
        spk_indices = torch.where(input_ids[0] == spk_token_id[0])[0]
        assert len(spk_indices) > 0, "expected at least one <|vision_start|> slot"
        for i, se in enumerate(spk_emb):
            inputs_embeds[0, spk_indices[i] + 1] = se

    # inject prompt-wav embeddings after <audio>
    if prompt_wav_emb is not None and prompt_text is not None:
        audio_token_id = tokenizer.encode("<audio>")
        assert len(audio_token_id) == 1, "<audio> must tokenize to a single id"
        audio_indices = torch.where(input_ids[0] == audio_token_id[0])[0]
        assert len(audio_indices) > 0, "expected at least one <audio> slot"
        start = audio_indices[0] + 1
        inputs_embeds[0, start : start + prompt_wav_emb.size(1), :] = prompt_wav_emb[0]

    return inputs_embeds, input_ids


def build_tts_prompt_token_ids(
    *,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    prompt: str,
    spk_emb_count: int = 0,
    instruction: str | None = None,
    prompt_text: str | None = None,
    prompt_wav_len: int = 0,
) -> list[int]:
    """Build token IDs for the Ming talker prefill prompt.

    This mirrors ``build_tts_input`` exactly, but does not require model
    weights. Native vLLM scheduling needs the prompt length before model
    ``preprocess()`` builds embeddings, so stage input processors use this
    helper to allocate the correct number of paged-KV slots.
    """
    spk_emb_prompt: list[int] = []
    for i in range(max(0, int(spk_emb_count))):
        spk_emb_prompt.extend(
            tokenizer.encode(f"  speaker_{i + 1}:")
            + tokenizer.encode("<|vision_start|>")
            + tokenizer.encode("<|vision_pad|>")
            + tokenizer.encode("<|vision_end|>\n")
        )

    instruction_prompt: list[int] = []
    if instruction is not None:
        instruction_prompt = tokenizer.encode(instruction) + tokenizer.encode("<|im_end|>")

    prompt_text_token: list[int] = []
    prompt_latent_token: list[int] = []
    if prompt_wav_len > 0 and prompt_text is not None:
        prompt_text_token = tokenizer.encode(prompt_text)
        prompt_latent_token = tokenizer.encode("<audioPatch>") * int(prompt_wav_len)

    prompt2 = [] if _looks_like_music_prompt(text) else tokenizer.encode(" Text input:\n")

    return (
        tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        + tokenizer.encode("<|im_start|>user\n")
        + tokenizer.encode(prompt)
        + spk_emb_prompt
        + prompt2
        + prompt_text_token
        + tokenizer.encode(text)
        + tokenizer.encode("<|im_end|>\n")
        + tokenizer.encode("<|im_start|>assistant\n")
        + instruction_prompt
        + tokenizer.encode("<audio>")
        + prompt_latent_token
    )


def ming_prompt_wav_len(
    num_samples: int,
    *,
    hop_size: int,
    vae_patch_size: int,
    patch_size: int,
) -> int:
    """Prompt-wav frames for ``num_samples`` samples at the AudioVAE sample rate.

    One LLM token per ``hop_size * vae_patch_size * patch_size`` samples, with a
    partial group zero-padded to a whole one (see ``_build_wav_embeddings``), so
    the stage input processor can size prompt-KV slots without loading the VAE.
    """
    samples_per_frame = int(hop_size) * max(1, int(vae_patch_size)) * int(patch_size)
    if samples_per_frame <= 0 or num_samples <= 0:
        return 0
    return -(-int(num_samples) // samples_per_frame)  # ceil division


def resolve_audio_vae_config(
    audio_vae_path: str | None, talker_dir: str, model_path: str
) -> tuple[AudioVAEConfig, str | tuple[str, str]] | None:
    """Resolve the AudioVAE config and where its weights live.

    Returns ``(config, source)`` where source is a local directory or an
    ``(repo_id, subfolder)`` HF hub tuple. Shared by the talker's VAE init and
    the prompt-wav geometry derivation so the two can never disagree about
    which config applies.
    """
    vae_path = audio_vae_path or os.path.join(talker_dir, "vae")
    if os.path.isdir(vae_path):
        try:
            return AudioVAEConfig.from_pretrained(vae_path), vae_path
        except Exception as e:
            logger.warning("Failed to load AudioVAE config from %s: %s", vae_path, e)
    for subfolder in ("talker/vae", "vae"):
        try:
            config = AudioVAEConfig.from_pretrained(model_path, subfolder=subfolder, trust_remote_code=True)
            return config, (model_path, subfolder)
        except Exception:
            continue
    return None


########################################################################
# Audio Generator
########################################################################


class MingAudioGenerator:
    """Audio-side helpers for the scheduler-driven talker: per-step CFM
    sampling, the duration cap, and VAE waveform decode. The AR loop itself
    is owned by the vLLM scheduler; this class is stateless across requests.
    """

    def __init__(
        self,
        config,
        llm_config: Qwen2Config,
        model: nn.Module,
        cfm: CFM,
        aggregator: Aggregator,
        stop_head: torch.nn.Module,
        audio_vae: AudioVAE | None,
        patch_size: int,
        his_patch_size: int,
        latent_dim: int,
        cfg_strength: float,
    ) -> None:
        self._config = config
        self._llm_config = llm_config
        self._model = model
        self._cfm = cfm
        self._aggregator = aggregator
        self._stop_head = stop_head
        self._audio_vae = audio_vae

        self.patch_size = patch_size
        self.his_patch_size = his_patch_size
        self.latent_dim = latent_dim
        self.cfg_strength = cfg_strength

        # For FA2, let it see a full-length seq Q
        # trailing latent frames prepended on each decode call
        self._vae_decode_pad_frames = 32

    def duration_capped_steps(self, text_len: int, requested_max_steps: int) -> int:
        """Apply the original Ming duration heuristic as a cap on decode steps."""
        if self._audio_vae is None:
            return requested_max_steps

        # Transformers >=5.x may expose these config values as 0-d tensors.
        sample_rate = float(self._audio_vae.config.sample_rate)
        vae_patch_size = float(getattr(self._audio_vae.config, "patch_size", 4))
        hop_size = float(getattr(self._audio_vae.decoder, "hop_length", 320))
        seconds_per_step = (self.patch_size * vae_patch_size * hop_size) / sample_rate
        if seconds_per_step <= 0:
            return requested_max_steps

        max_duration_s = max(2.0, float(text_len) * (5818.0 / 16000.0))
        max_steps_by_duration = max(1, int(max_duration_s / seconds_per_step))
        return min(requested_max_steps, max_steps_by_duration)

    def cfm_sample_step(
        self,
        last_hidden_state: torch.Tensor,
        his_lat: torch.Tensor,
        *,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
        randn_tensor: torch.Tensor | None = None,
        sde_rnd: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one CFM sampling step (eager).

        CUDA-graph acceleration of this step is intentionally out of scope for
        the native-paged port; it is tracked separately under the CFMGraphExecutor
        perf work in RFC #4129.
        """
        if cfg is None:
            cfg = self.cfg_strength

        bat_size, _, z_dim = his_lat.shape
        expected_randn_shape = (bat_size, self.patch_size, z_dim)
        if randn_tensor is None:
            randn_tensor = torch.randn(
                expected_randn_shape,
                device=last_hidden_state.device,
                dtype=last_hidden_state.dtype,
            )
        else:
            if tuple(randn_tensor.shape) != expected_randn_shape:
                raise ValueError(f"randn_tensor shape must be {expected_randn_shape}, got {tuple(randn_tensor.shape)}")
            randn_tensor = randn_tensor.to(device=last_hidden_state.device, dtype=last_hidden_state.dtype)

        t = get_epss_timesteps(self._config.steps, device=last_hidden_state.device, dtype=last_hidden_state.dtype)
        expected_sde_shape = (self._config.steps, *expected_randn_shape)
        if sde_rnd is None:
            sde_rnd = torch.randn(
                expected_sde_shape,
                device=last_hidden_state.device,
                dtype=last_hidden_state.dtype,
            )
        else:
            if tuple(sde_rnd.shape) != expected_sde_shape:
                raise ValueError(f"sde_rnd shape must be {expected_sde_shape}, got {tuple(sde_rnd.shape)}")
            sde_rnd = sde_rnd.to(device=last_hidden_state.device, dtype=last_hidden_state.dtype)
        sde_args = torch.tensor(
            [cfg, sigma, temperature],
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )

        gen_lat = self._cfm.sample(last_hidden_state, his_lat, randn_tensor, t, sde_args, sde_rnd)
        inputs_embeds = self._aggregator(gen_lat)
        stop_out = self._stop_head(last_hidden_state[:, -1, :]).softmax(dim=-1)

        return gen_lat, inputs_embeds, stop_out

    def decode_to_waveform(self, latents: list[torch.Tensor], stream_decode: bool = True) -> torch.Tensor:
        """Decode accumulated latents to waveform via AudioVAE."""
        if self._audio_vae is None:
            raise RuntimeError("AudioVAE not loaded. Cannot decode audio latents to waveform.")

        if stream_decode:
            return self._stream_decode(latents)

        all_lat = torch.cat(latents, dim=1)
        waveform, _, _ = self._audio_vae.decode(
            all_lat, use_cache=False, stream_state=(None, None, None), last_chunk=True
        )
        return waveform

    def _init_his_lat(
        self, prompt_wav_lat: torch.Tensor | None, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        his_lat = torch.zeros(1, self.his_patch_size, self.latent_dim, device=device, dtype=dtype)
        if prompt_wav_lat is not None:
            start_index = self.his_patch_size - prompt_wav_lat.size(1)
            if start_index < 0:
                his_lat[:] = prompt_wav_lat[:, -start_index:, :]
            else:
                his_lat[:, start_index:, :] = prompt_wav_lat
        return his_lat

    def _update_his_lat(self, his_lat: torch.Tensor, gen_lat: torch.Tensor) -> torch.Tensor:
        if self.his_patch_size == self.patch_size:
            return gen_lat
        if self.his_patch_size > self.patch_size:
            return torch.cat([his_lat[:, self.patch_size - self.his_patch_size :], gen_lat], dim=1)
        raise NotImplementedError(f"his_patch_size ({self.his_patch_size}) < patch_size ({self.patch_size})")

    # VAE streaming decode
    def _stream_decode(self, latents: list[torch.Tensor]) -> torch.Tensor:
        sr = int(self._audio_vae.config.sample_rate)
        decode_pad: torch.Tensor | None = None
        sil_cache: dict | None = None
        wav_chunks: list[torch.Tensor] = []

        for i, lat in enumerate(latents):
            last_chunk = i == (len(latents) - 1)

            if decode_pad is not None:
                vae_input = torch.cat([decode_pad, lat], dim=1)
                pad_frames = decode_pad.shape[1]
            else:
                vae_input = lat
                pad_frames = 0

            # Stateless, no KV cache accum intentionally.
            speech, _, _ = self._audio_vae.decode(
                vae_input,
                use_cache=False,
                stream_state=(None, None, None),
                last_chunk=True,
            )

            total_frames = vae_input.shape[1]
            dcs = speech.shape[-1] // total_frames

            # keep only the new audio.
            speech_chunk = speech[:, :, pad_frames * dcs :][0].detach().float()
            speech_chunk, sil_cache = silence_holder(
                speech_chunk,
                sr,
                sil_cache=sil_cache,
                last_chunk=last_chunk,
            )
            if speech_chunk.numel() > 0:
                wav_chunks.append(speech_chunk)

            # Advance the sliding buffer
            decode_pad = vae_input[:, -self._vae_decode_pad_frames :, :].detach()

        if not wav_chunks:
            device = next(self._model.parameters()).device
            dtype = next(self._model.parameters()).dtype
            return torch.zeros((1, 1, 0), device=device, dtype=dtype)
        return torch.cat(wav_chunks, dim=-1).unsqueeze(0)

    # Post-decode helper
    def trim_trailing_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._audio_vae is None:
            return waveform
        return trim_trailing_silence(waveform, int(self._audio_vae.config.sample_rate))

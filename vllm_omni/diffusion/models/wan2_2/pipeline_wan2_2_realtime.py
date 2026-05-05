# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Realtime video streaming pipeline for Causal WanVideo models.
# Supports both T2V (text-to-video) and V2V (video-to-video) streaming.
# Ported from sglang's Krea realtime video implementation.

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from vllm_omni.diffusion.models.wan2_2.causal_wan2_2_transformer import (
    CausalWanTransformer3DModel,
)
from vllm_omni.diffusion.models.wan2_2.kv_cache import KVCacheManager
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import retrieve_latents

logger = logging.getLogger(__name__)


class RealtimeVideoMode(str, Enum):
    T2V = "t2v"
    V2V = "v2v"


@dataclass
class RealtimeSession:
    """Persistent state across video blocks within a streaming session."""

    last_prompts: str | list[str] | None = None
    last_embeds: list[torch.Tensor] = field(default_factory=list)
    interpolated_embeds: list[list[torch.Tensor]] = field(default_factory=list)
    kv_cache_manager: KVCacheManager | None = None
    current_denoised_latents: torch.Tensor | None = None
    frame_cache_context: deque | None = None
    decoder_cache: Any = None
    v2v_strength: float | None = None
    first_frame_latent: torch.Tensor | None = None
    block_idx: int = 0

    def is_prompt_changed(self, prompts: str | list[str]) -> bool:
        return prompts != self.last_prompts

    def save_prompt_changed(
        self,
        prompts: str | list[str],
        prompt_embeds: list[torch.Tensor],
        interpolated: list[list[torch.Tensor]] | None = None,
    ) -> None:
        self.last_prompts = prompts
        self.last_embeds = prompt_embeds
        if interpolated is not None:
            self.interpolated_embeds.extend(interpolated)

    def get_current_embeds(self) -> list[torch.Tensor]:
        if self.interpolated_embeds:
            step_embeds = self.interpolated_embeds.pop(0)
            return step_embeds
        return self.last_embeds

    @property
    def update_prompt_embeds(self) -> bool:
        return len(self.interpolated_embeds) > 0

    def dispose(self) -> None:
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.release()
            self.kv_cache_manager = None
        self.current_denoised_latents = None
        self.frame_cache_context = None
        self.decoder_cache = None
        self.first_frame_latent = None
        self.last_embeds.clear()
        self.interpolated_embeds.clear()
        torch.cuda.empty_cache()


class Wan22RealtimePipeline:
    """Realtime video streaming pipeline using Causal WanVideo with KV cache.

    Generates video one block at a time, maintaining state across blocks
    via RealtimeSession. Supports both T2V and V2V modes.
    """

    INTERPOLATION_STEPS = 4
    DEFAULT_NUM_INFERENCE_STEPS = 4
    DEFAULT_FLOW_SHIFT = 5.0

    def __init__(
        self,
        transformer: CausalWanTransformer3DModel,
        vae,
        tokenizer,
        text_encoder,
        *,
        num_frames_per_block: int = 3,
        kv_cache_num_frames: int = 3,
        vae_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
    ):
        self.transformer = transformer
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.num_frames_per_block = num_frames_per_block
        self.kv_cache_num_frames = kv_cache_num_frames
        self.vae_dtype = vae_dtype
        self.transformer_dtype = transformer_dtype
        self.video_processor = VideoProcessor(vae_scale_factor=8)

        self._vae_latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=self.vae.device,
            dtype=vae_dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        self._vae_latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=self.vae.device,
            dtype=vae_dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)

        self._patch_size = transformer.config.patch_size
        self.frame_seq_length = self._compute_frame_seq_length(480, 832)

    def _compute_frame_seq_length(self, height: int, width: int) -> int:
        _, p_h, p_w = self._patch_size
        return (height // p_h // 8) * (width // p_w // 8)

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    _MIN_PROMPT_TOKENS = 40

    def _pad_short_prompt(self, prompt: str) -> str:
        """Repeat short prompts to fill more cross-attention positions.

        Short prompts produce few non-zero positions in the 512-length embedding,
        so cross-attention signal gets diluted by zero-padded bias-only keys.
        Repeating the prompt text gives the model more real-token positions to
        attend to, making prompt changes more visible during streaming.
        """
        tokens = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if len(tokens) >= self._MIN_PROMPT_TOKENS:
            return prompt
        reps = (self._MIN_PROMPT_TOKENS // max(len(tokens), 1)) + 1
        return ". ".join([prompt] * reps)

    def encode_prompt(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        device = self.device
        dtype = self.text_encoder.dtype

        if isinstance(prompt, str):
            prompt = [self._pad_short_prompt(prompt)]
        else:
            prompt = [self._pad_short_prompt(p) for p in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )
        return prompt_embeds

    def interpolate_embeds(
        self,
        prev_embeds: list[torch.Tensor],
        curr_embeds: list[torch.Tensor],
    ) -> list[list[torch.Tensor]]:
        """Interpolate between previous and current prompt embeddings.

        Weights start at 1/steps (not 0) so the first block after a prompt
        change already shows new-prompt influence.  All steps are computed
        in a single vectorised ``torch.lerp`` call per embedding layer.
        """
        steps = self.INTERPOLATION_STEPS
        assert len(prev_embeds) == len(curr_embeds)

        weights = torch.linspace(1.0 / steps, 1.0, steps=steps).unsqueeze(1).unsqueeze(2)

        per_layer_chunks: list[list[torch.Tensor]] = []
        for prev, curr in zip(prev_embeds, curr_embeds):
            assert prev.shape == curr.shape
            x = torch.lerp(prev, curr, weights.to(prev))
            per_layer_chunks.append(list(x.chunk(steps, dim=0)))

        result: list[list[torch.Tensor]] = []
        for step_idx in range(steps):
            result.append([chunks[step_idx] for chunks in per_layer_chunks])
        return result

    def prepare_timesteps(
        self,
        shift: float,
        strength: float,
        num_inference_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare timestep schedule for flow matching."""
        device = self.device
        sigmas = torch.linspace(1.0, 0.0, 1001, device=device)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        all_timesteps = sigmas * 1000.0
        zero_padded = torch.cat([all_timesteps, torch.tensor([0], device=device, dtype=all_timesteps.dtype)])
        denoising_steps = torch.linspace(strength * 1000, 0, num_inference_steps, dtype=torch.float32).to(torch.long)
        timesteps = zero_padded[1000 - denoising_steps]
        return timesteps, all_timesteps, sigmas

    def _reset_vae_encode_cache(self) -> None:
        if hasattr(self.vae, "_original_clear_cache"):
            self.vae._original_clear_cache()
        else:
            self.vae.clear_cache()

        if hasattr(self.vae, "_enc_conv_idx"):
            self.vae._enc_conv_idx = 0
        if hasattr(self.vae, "_enc_conv_num"):
            self.vae._enc_feat_map = [None] * self.vae._enc_conv_num
        elif hasattr(self.vae, "_enc_feat_map"):
            self.vae._enc_feat_map = [None] * len(self.vae._enc_feat_map)
        else:
            self.vae._enc_feat_map = [None] * 55

    def _encode_video_frames(
        self,
        video: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode video frames to latent space via VAE."""
        self._reset_vae_encode_cache()
        init_latents = [
            retrieve_latents(
                self.vae.encode(vid.unsqueeze(0).transpose(2, 1).to(device=self.vae.device, dtype=dtype).contiguous()),
                sample_mode="argmax",
            )
            for vid in video
        ]
        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        init_latents = (init_latents - self._vae_latents_mean) * self._vae_latents_std
        return init_latents

    def _prepare_frame_latents(
        self,
        frames: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode single frame(s) to latents for context recomputation."""
        self._reset_vae_encode_cache()
        frames = frames.to(device=self.vae.device, dtype=dtype).contiguous()
        latents = retrieve_latents(self.vae.encode(frames), sample_mode="argmax")
        latents = (latents - self._vae_latents_mean) * self._vae_latents_std
        return latents

    def _get_context_frames(
        self,
        session: RealtimeSession,
    ) -> torch.Tensor:
        """Get context frames for KV cache recomputation."""
        total_frames_generated = (session.block_idx - 1) * self.num_frames_per_block

        if total_frames_generated < self.kv_cache_num_frames:
            return session.current_denoised_latents[:, :, : self.kv_cache_num_frames]

        context = session.current_denoised_latents
        context = context[:, :, 1:][:, :, -self.kv_cache_num_frames + 1 :]
        if session.first_frame_latent is None:
            session.first_frame_latent = self._prepare_frame_latents(
                frames=session.frame_cache_context[0],
                dtype=self.vae_dtype,
            )
        first_frame_latent = session.first_frame_latent.to(context)
        return torch.cat((first_frame_latent, context), dim=2)

    def _setup_kv_cache(
        self,
        session: RealtimeSession,
    ) -> KVCacheManager:
        """Create or reset KV cache manager."""
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
        num_heads = self.transformer.num_attention_heads // tp_size
        head_dim = self.transformer.attention_head_dim
        num_blocks = len(self.transformer.blocks)
        sa_max_size = (self.kv_cache_num_frames + self.num_frames_per_block) * self.frame_seq_length

        if session.kv_cache_manager is None:
            sink_size = self.transformer.blocks[0].attn1.sink_size
            session.kv_cache_manager = KVCacheManager(
                num_blocks=num_blocks,
                sa_batch_size=1,
                sa_max_size=sa_max_size,
                sa_num_heads=num_heads,
                sa_head_dim=head_dim,
                dtype=self.transformer_dtype,
                device=self.device,
                sink_size=sink_size,
                frame_seq_length=self.frame_seq_length,
            )
        else:
            session.kv_cache_manager.reset_self_attn()
            if session.update_prompt_embeds:
                session.kv_cache_manager.reset_cross_attn()

        return session.kv_cache_manager

    def _recompute_context(
        self,
        session: RealtimeSession,
        prompt_embeds: torch.Tensor,
        manager: KVCacheManager,
    ) -> None:
        """Recompute KV cache from context frames for temporal coherence."""
        context_frames = self._get_context_frames(session)
        block_mask = CausalWanTransformer3DModel._prepare_blockwise_causal_attn_mask(
            self.device,
            num_frames=context_frames.shape[2],
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frames_per_block,
            local_attn_size=-1,
        )
        self.transformer.block_mask = block_mask
        context_timestep = torch.zeros(
            (context_frames.shape[0],),
            device=self.device,
            dtype=torch.int64,
        )
        self.transformer(
            hidden_states=context_frames.to(self.transformer_dtype),
            timestep=context_timestep,
            encoder_hidden_states=prompt_embeds.to(self.transformer_dtype),
            kv_cache=manager.self_attn_caches,
            crossattn_cache=manager.cross_attn_caches,
            current_start=0,
            cache_start=None,
        )
        self.transformer.block_mask = None

    @staticmethod
    def _add_noise(
        sample: torch.Tensor,
        noise: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        sigma = sigma_t.to(device=sample.device).reshape(1, 1, 1, 1)
        return ((1 - sigma.double()) * sample.double() + sigma.double() * noise.double()).type_as(noise)

    @staticmethod
    def _update_latents(
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        latents_dtype = latents.dtype
        sigma_t = sigma_t.to(device=latents.device)
        return (latents.double() - sigma_t.double() * noise_pred.double()).to(latents_dtype)

    def _predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        kv_cache: list,
        crossattn_cache: list,
        current_start_frame: int,
    ) -> torch.Tensor:
        start_frame = min(current_start_frame, self.kv_cache_num_frames)
        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep.expand(latents.shape[0]),
            encoder_hidden_states=prompt_embeds.to(self.transformer_dtype),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=start_frame * self.frame_seq_length,
            start_frame=start_frame,
            cache_start=None,
            return_dict=False,
        )
        if isinstance(noise_pred, tuple):
            noise_pred = noise_pred[0]
        return noise_pred

    @torch.no_grad()
    def denoise_block(
        self,
        session: RealtimeSession,
        prompt: str,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int | None = None,
        input_video_frames: list | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run text encoding, latent prep, context recomputation, and denoising.

        Returns denoised latents. Updates session.current_denoised_latents
        and increments session.block_idx.
        """
        if num_inference_steps is None:
            num_inference_steps = self.DEFAULT_NUM_INFERENCE_STEPS

        _, p_h, p_w = self._patch_size
        align_h = 8 * p_h
        align_w = 8 * p_w
        height = (height // align_h) * align_h
        width = (width // align_w) * align_w

        new_seq_len = self._compute_frame_seq_length(height, width)
        if new_seq_len != self.frame_seq_length:
            self.frame_seq_length = new_seq_len
            if session.kv_cache_manager is not None:
                session.kv_cache_manager = None

        block_idx = session.block_idx
        has_video = input_video_frames is not None
        strength = (session.v2v_strength or 0.7) if has_video else 1.0

        prompt_embeds = self._encode_with_interpolation(session, prompt)

        timesteps, all_timesteps, sigmas = self.prepare_timesteps(
            self.DEFAULT_FLOW_SHIFT, strength, num_inference_steps
        )

        if has_video:
            latents = self._prepare_v2v_latents(input_video_frames, height, width, timesteps, generator)
        else:
            latents = self._prepare_t2v_latents(block_idx, height, width, generator)

        current_start_frame = block_idx * self.num_frames_per_block
        manager = self._setup_kv_cache(session)

        if block_idx > 0:
            self._recompute_context(session, prompt_embeds, manager)

        step_timestep_ids = torch.argmin((all_timesteps.unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1)
        step_sigmas = sigmas[step_timestep_ids]

        for i, t in enumerate(timesteps):
            noise_pred = self._predict_noise(
                latents=latents,
                timestep=t,
                prompt_embeds=prompt_embeds,
                kv_cache=manager.self_attn_caches,
                crossattn_cache=manager.cross_attn_caches,
                current_start_frame=current_start_frame,
            )
            latents = self._update_latents(latents, noise_pred, step_sigmas[i])
            if i < num_inference_steps - 1:
                sample = latents.transpose(1, 2).squeeze(0)
                noise = randn_tensor(
                    sample.shape,
                    device=latents.device,
                    dtype=latents.dtype,
                    generator=generator,
                )
                latents = self._add_noise(sample, noise, step_sigmas[i + 1]).unsqueeze(0).transpose(1, 2)

        session.current_denoised_latents = latents
        session.block_idx += 1
        return latents

    def decode_block(
        self,
        session: RealtimeSession,
        block_idx: int,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """VAE decode latents to video tensor. Updates session caches.

        Returns clamped video tensor on GPU. Call postprocess_decoded()
        to convert to numpy.
        """
        if session.frame_cache_context is None:
            frame_cache_len = 1 + (self.kv_cache_num_frames - 1) * 4
            session.frame_cache_context = deque(maxlen=frame_cache_len)

        if block_idx == 0:
            if not hasattr(self.vae, "_original_clear_cache"):
                self.vae._original_clear_cache = self.vae.clear_cache
            self.vae._original_clear_cache()
            self.vae.clear_cache = lambda: None
            decoder_cache_len = getattr(self.vae, "_conv_num", 55)
            self.vae._feat_map = [None] * decoder_cache_len
        else:
            self.vae._feat_map = session.decoder_cache

        decode_latents = latents.to(device=self.vae.device, dtype=self.vae_dtype)
        decode_latents = decode_latents / self._vae_latents_std + self._vae_latents_mean

        from contextlib import nullcontext

        ctx = self.vae._execution_context() if hasattr(self.vae, "_execution_context") else nullcontext()

        with ctx:
            x = self.vae.post_quant_conv(decode_latents)
            num_frames = x.shape[2]

            for i in range(num_frames):
                self.vae._conv_idx = [0]
                is_first_chunk = block_idx == 0 and i == 0
                frame = x[:, :, i : i + 1, :, :]
                decoded = self.vae.decoder(
                    frame,
                    feat_cache=self.vae._feat_map,
                    feat_idx=self.vae._conv_idx,
                    first_chunk=is_first_chunk,
                )
                if i == 0:
                    videos = decoded
                else:
                    videos = torch.cat([videos, decoded], dim=2)

            patch_size = getattr(self.vae.config, "patch_size", None)
            if patch_size is not None:
                from diffusers.models.autoencoders.autoencoder_kl_wan import (
                    unpatchify,
                )

                videos = unpatchify(videos, patch_size=patch_size)

            videos = torch.clamp(videos, min=-1.0, max=1.0)

        session.decoder_cache = self.vae._feat_map
        session.frame_cache_context.extend(videos.split(1, dim=2))
        return videos

    def postprocess_decoded(self, video_tensor: torch.Tensor) -> np.ndarray:
        """Convert decoded video tensor to numpy array."""
        return self.video_processor.postprocess_video(video_tensor, output_type="np")

    @torch.no_grad()
    def generate_block(
        self,
        session: RealtimeSession,
        prompt: str,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int | None = None,
        input_video_frames: list | None = None,
        generator: torch.Generator | None = None,
    ) -> np.ndarray:
        """Generate one video block (synchronous).

        For async pipeline use denoise_block() + decode_block() +
        postprocess_decoded() separately.
        """
        block_idx = session.block_idx

        latents = self.denoise_block(
            session,
            prompt,
            height,
            width,
            num_inference_steps,
            input_video_frames,
            generator,
        )

        videos = self.decode_block(session, block_idx, latents)

        return self.postprocess_decoded(videos)

    def _encode_with_interpolation(
        self,
        session: RealtimeSession,
        prompt: str,
    ) -> torch.Tensor:
        """Encode prompt with interpolation on change."""
        if session.is_prompt_changed(prompt):
            new_embeds = self.encode_prompt(prompt)
            interpolated = None
            if session.last_embeds:
                interpolated = self.interpolate_embeds(session.last_embeds, [new_embeds])
            session.save_prompt_changed(prompt, [new_embeds], interpolated)

        embeds = session.get_current_embeds()
        return embeds[0] if isinstance(embeds, list) else embeds

    def _prepare_v2v_latents(
        self,
        frames: list,
        height: int,
        width: int,
        timesteps: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Prepare latents for V2V mode by encoding input frames."""
        if len(frames) < self.num_frames_per_block:
            frames = frames + [frames[-1]] * (self.num_frames_per_block - len(frames))

        video = self.video_processor.preprocess(frames, height, width).unsqueeze(0).to(self.vae_dtype)
        init_latents = self._encode_video_frames(video, self.vae_dtype)
        init_latents = init_latents[:, :, -self.num_frames_per_block :]

        strength = timesteps[0].item() / 1000.0
        noise = randn_tensor(
            init_latents.shape,
            device=self.device,
            dtype=self.transformer_dtype,
            generator=generator,
        )
        init_latents = init_latents * (1.0 - strength) + noise * strength
        return init_latents.to(self.transformer_dtype).contiguous()

    def _prepare_t2v_latents(
        self,
        block_idx: int,
        height: int,
        width: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Prepare random latents for T2V mode."""
        num_channels = self.transformer.config.in_channels
        vae_sf = 8
        effective_blocks = max(1, block_idx + 1)
        num_latent_frames = effective_blocks * self.num_frames_per_block
        shape = (
            1,
            num_channels,
            num_latent_frames,
            height // vae_sf,
            width // vae_sf,
        )
        all_latents = randn_tensor(
            shape,
            generator=generator,
            device=self.device,
            dtype=self.transformer_dtype,
        )
        start = block_idx * self.num_frames_per_block
        end = start + self.num_frames_per_block
        return all_latents[:, :, start:end].contiguous()
